import sys
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2

import rclpy
import DR_init
from sensor_msgs.msg import CompressedImage
from tf2_ros import Buffer, TransformListener
import tf2_ros
from visualization_msgs.msg import Marker
import rclpy.duration

from EEpose_from_mesh.mesh_utils import (
    query_mesh_normals,
    visualize_normals,
    compute_se3_pose,
)
import trimesh

rclpy.init()
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
node = rclpy.create_node("coverage_path", namespace=ROBOT_ID)
DR_init.__dsr__node = node
from DSR_ROBOT2 import (
    movej,
    posj,
    movel,
    posx,
    set_robot_mode,
    get_current_posx,
    ROBOT_MODE_AUTONOMOUS,
)

# Gazebo cam frame: X=forward, Y=left, Z=up
# ROS optical frame: X=right, Y=down, Z=forward
GAZ_TO_OPT_R = R.from_matrix(
    np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]
    )
)

GAZ_TO_OPT_T = np.array([0, 0, 0])

WLD_TO_BASE_R = R.from_euler("xyz", [0, 0, 3.141519])
WLD_TO_BASE_T = np.array([-0.61, 0.365, 0.91])
L6_TO_CAM_R = R.from_euler("xyz", [0, -1.5708, 3.141519])
L6_TO_CAM_T = np.array([0.05, 0, 0.01])
WLD_TO_CNC = np.array([0.0, 0.0, 0.0])

# vice mesh origin in world frame
# X: world_x at mesh_x=0  (from calibration corner 1)
# Y: world_y at mesh_y=0  (from calibration corner 3, the bottom-right)
# Z: vice top surface height in world
VICE_ORIGIN_WORLD = np.array([-0.295, -0.239, 1.095])
# VICE_ORIGIN_WORLD = np.array([-0.295, -0.239, 1.095])

# mesh axes vs world axes (from calibration):
# mesh_x increases in world -X direction → negate X
# mesh_y increases in world -Y direction → negate Y
MESH_X_SIGN = -1
MESH_Y_SIGN = -1
MESH_DIMENSION = 1000  # 1 mesh unit = 0.001 m

fx = 762.72
fy = fx
cx_k = 640
cy_k = 360
camera_K = np.array([[fx, 0, cx_k], [0, fy, cy_k], [0, 0, 1]])

vice_mesh_path = (
    "/home/robot_llam/BKyoon/working/chipblowing/EEpose_from_mesh/mesh/Vice.stl"
)


# ── helpers ───────────────────────────────────────────────────────────────────


def make_SE3(R_mat: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


# ── ROS helpers ───────────────────────────────────────────────────────────────


def grab_image() -> np.ndarray:
    """Subscribe, grab one frame, unsubscribe."""
    latest = {"img": None}

    def cb(msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        latest["img"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sub = node.create_subscription(
        CompressedImage, "/camera/image_raw/compressed", cb, 10
    )
    print("[img] Waiting for frame...")
    t0 = time.time()
    while latest["img"] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 5.0:
            node.destroy_subscription(sub)
            raise RuntimeError("[img] Timed out waiting for camera frame.")
    node.destroy_subscription(sub)
    print(f"[img] Got frame: {latest['img'].shape[1]}x{latest['img'].shape[0]}")
    return latest["img"]


def get_base_to_link6() -> tuple[np.ndarray, np.ndarray]:
    """Returns (R_b_6 [3x3], t_b_6 [3])"""
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    print("[tf] Warming up tf buffer...")
    t0 = time.time()
    while time.time() - t0 < 1.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    print("[tf] Waiting for base_link -> link_6 transform...")
    t0 = time.time()
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            t = tf_buffer.lookup_transform("base_link", "link_6", rclpy.time.Time())
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException):
            if time.time() - t0 > 5.0:
                raise RuntimeError("[tf] Timed out waiting for base_link -> link_6.")
        except tf2_ros.ExtrapolationException as e:
            raise RuntimeError(f"[tf] Extrapolation error: {e}")

    tf_listener.unregister()
    del tf_buffer

    trans = t.transform.translation
    rot = t.transform.rotation
    translation = np.array([trans.x, trans.y, trans.z])
    rotation = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
    print(f"[tf] base_link -> link_6\n  t={translation}\n  R=\n{rotation}")
    return rotation, translation

# ── UI ────────────────────────────────────────────────────────────────────────
def pick_xy_from_camera(snapshot: np.ndarray) -> tuple[float, float]:
    h, w = snapshot.shape[:2]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.08, 0.10, 0.88, 0.85])
    fig.canvas.manager.set_window_title(
        "Camera view — click to select (x, y), then press Enter"
    )
    ax.imshow(snapshot, origin="upper")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(
        "Click to set query point  |  Press Enter or click Confirm", fontsize=11
    )

    state = {"x": None, "y": None}
    markers = []

    def _clear():
        for a in markers:
            try:
                a.remove()
            except Exception:
                pass
        markers.clear()

    def _on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        _clear()
        px, py = float(event.xdata), float(event.ydata)
        state["x"], state["y"] = px, py
        markers.extend(
            [
                ax.axhline(py, color="red", lw=0.8, ls="--", alpha=0.7),
                ax.axvline(px, color="red", lw=0.8, ls="--", alpha=0.7),
                ax.plot(px, py, "r+", ms=14, mew=2)[0],
                ax.text(
                    px,
                    py,
                    f"  ({px:.1f}, {py:.1f})",
                    color="red",
                    fontsize=9,
                    va="bottom",
                ),
            ]
        )
        fig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "enter" and state["x"] is not None:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    ax_btn = fig.add_axes([0.35, 0.01, 0.30, 0.05])
    btn = Button(
        ax_btn, "Confirm  (or press Enter)", color="#d0e8ff", hovercolor="#90c8ff"
    )
    btn.on_clicked(lambda _: plt.close(fig) if state["x"] is not None else None)

    plt.show()

    if state["x"] is None:
        raise RuntimeError("No point selected.")

    print(f"[picker] Selected pixel: x={state['x']:.1f},  y={state['y']:.1f}")
    return state["x"], state["y"]


# ── robot motion ──────────────────────────────────────────────────────────────


def move_to_init():
    print("moving to initial point")
    movej(posj(90, 0, -90, 0, 0, 0), vel=60, acc=60)
    time.sleep(1)
    movej(posj(-70, 0, -90, 0, 0, 0), vel=60, acc=60)
    time.sleep(1)
    movej(posj(-70, -30, -60, 0, 0, 0), vel=60, acc=60)
    time.sleep(1)
    movej(posj(-75, -30, -60, 0, -90, 0), vel=60, acc=60)
    time.sleep(1)
    print("moved to initial point")


# ── coordinate helpers ────────────────────────────────────────────────────────


def pixel_to_world(
    pu: float, pv: float, z_cam: float, T_w_cm: np.ndarray
) -> np.ndarray:
    """
    Unproject pixel (u, v) at known depth z_cam (in camera frame) to world frame.
    """
    x_cam = (pu - cx_k) / fx * z_cam
    y_cam = (pv - cy_k) / fy * z_cam
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    p_world = T_w_cm @ p_cam
    return p_world[:3]


def world_to_mesh(query_world: np.ndarray) -> tuple[float, float]:
    """
    Convert world XY to mesh units.
    Mesh origin = VICE_ORIGIN_WORLD.
    Mesh X increases in world -X, mesh Y increases in world -Y.
    """
    print("vice_origin_world: ", VICE_ORIGIN_WORLD)
    delta = query_world - VICE_ORIGIN_WORLD
    mesh_x = MESH_X_SIGN * delta[0] * MESH_DIMENSION
    mesh_y = MESH_Y_SIGN * delta[1] * MESH_DIMENSION
    return mesh_x, mesh_y


def get_depth_from_mesh(
    pu: float,
    pv: float,
    T_w_cm: np.ndarray,
    mesh: trimesh.Trimesh,
) -> float:
    """
    Given a pixel (u, v), cast the corresponding ray from the camera
    into the mesh and return the depth (z in camera frame) of the hit.

    The mesh is in mesh coordinates; camera position and ray are
    converted from world to mesh coordinates before raycasting.

    Parameters
    ----------
    pu, pv : pixel coordinates
    T_w_cm : world-from-camera SE3 (4x4)
    mesh   : trimesh object in mesh coordinates (origin at bounds min)

    Returns
    -------
    z_cam : depth along camera z-axis at the hit point
    """
    cam_pos_world = T_w_cm[:3, 3]

    # world -> mesh coordinates
    cam_pos_mesh = np.array(
        [
            (cam_pos_world[0]) * MESH_DIMENSION,
            (cam_pos_world[1]) * MESH_DIMENSION,
            (cam_pos_world[2]) * MESH_DIMENSION,
        ]
    )

    # pixel -> ray direction in camera frame -> world frame
    p_cam_unit = np.linalg.inv(camera_K) @ np.array([pu, pv, 1.0])
    ray_world = T_w_cm[:3, :3] @ p_cam_unit
    ray_world = ray_world / np.linalg.norm(ray_world)

    # world ray direction -> mesh ray direction (scale signs, no translation)
    ray_mesh = np.array(
        [
            ray_world[0],
            ray_world[1],
            ray_world[2],
        ]
    )
    ray_mesh = ray_mesh / np.linalg.norm(ray_mesh)

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=cam_pos_mesh.reshape(1, 3),
        ray_directions=ray_mesh.reshape(1, 3),
    )

    if len(locations) == 0:
        raise RuntimeError(f"No mesh surface hit for pixel ({pu:.1f}, {pv:.1f})")

    # closest hit to camera
    dists = np.linalg.norm(locations - cam_pos_mesh, axis=1)
    hit_mesh = locations[np.argmin(dists)]

    # mesh -> world
    hit_world = np.array(
        [
            hit_mesh[0] / MESH_DIMENSION,
            hit_mesh[1] / MESH_DIMENSION,
            hit_mesh[2] / MESH_DIMENSION,
        ]
    )

    # world -> camera frame to get z_cam
    T_cm_w = np.linalg.inv(T_w_cm)
    hit_cam = T_cm_w @ np.append(hit_world, 1.0)

    return hit_cam[2]


# ── main ──────────────────────────────────────────────────────────────────────


def main(args=None):
    cnc_mesh = trimesh.load_mesh(
        "/home/robot_llam/BKyoon/working/chipblowing/arm_ws/src/scripts/meshes/VMC-300-l.stl"
    )
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    # move_to_init()
    time.sleep(3)

    snapshot = grab_image()
    R_b_6, t_b_6 = get_base_to_link6()

    # build transform chain
    T_w_b = make_SE3(WLD_TO_BASE_R.as_matrix(), WLD_TO_BASE_T)
    T_b_6 = make_SE3(R_b_6, t_b_6)
    T_6_cm = make_SE3(
        (L6_TO_CAM_R * GAZ_TO_OPT_R).as_matrix(), L6_TO_CAM_T
    )  # link 6 to camera model frame (z-forward / y-downward)
    print("T_6_cm, ", T_6_cm)
    T_w_cm = T_w_b @ T_b_6 @ T_6_cm

    print(f"[cam] position in world:     {T_w_cm[:3, 3]}")
    print(f"[cam] optical axis in world: {T_w_cm[:3, 2]}")

    # pick pixel
    pu, pv = pick_xy_from_camera(snapshot)
    print(f"[main] Picked pixel: ({pu:.1f}, {pv:.1f})")

    # pixel -> world
    # query_world = pixel_to_world(pu, pv, T_w_cm)
    z_cam = get_depth_from_mesh(pu, pv, T_w_cm, cnc_mesh)
    print("depth: ", z_cam, " world_z: ", T_w_cm[2, 3] - z_cam)
    query_world = pixel_to_world(pu, pv, z_cam, T_w_cm)
    print(f"[main] query_world: {query_world}")

    # world -> mesh
    query_mesh_x, query_mesh_y = world_to_mesh(query_world)
    print(f"[main] query_mesh_x={query_mesh_x:.2f}  query_mesh_y={query_mesh_y:.2f}")
    print(
        f"[main] in range: X={0 <= query_mesh_x <= 320},  Y={0 <= query_mesh_y <= 84}"
    )

    print("T_w_b:\n", T_w_b)
    print("T_b_6:\n", T_b_6)
    print("T_6_cm:\n", T_6_cm)
    print("T_w_cm:\n", T_w_cm)
    print("cam optical axis (world):", T_w_cm[:3, 2])
    print("cam position (world):", T_w_cm[:3, 3])

    # query normals
    query_mesh_z = (query_world[2] - VICE_ORIGIN_WORLD[2]) * MESH_DIMENSION
    normals = query_mesh_normals(
        vice_mesh_path, query_mesh_x, query_mesh_y, 10, z=query_mesh_z
    )
    vice_mesh = trimesh.load(vice_mesh_path, force="mesh")
    vice_mesh.vertices -= vice_mesh.bounds[0]

    # find desired SE(3)
    d = 0.05
    pose = compute_se3_pose(normals, d * MESH_DIMENSION)
    normals["pose"] = pose

    # visualize normals and pose
    visualize_normals(vice_mesh, normals)

    ## TODO: convert desired EE pose to base frame

    rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
TODO: merge GMM estimation
"""
