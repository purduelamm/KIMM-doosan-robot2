"""
INPUT:
* CNC zero is world coordinate
* CNC zero to robot arm BASE
* robot arm BASE to camera Optical
* camera intrinsic (pinhole model)
* CNC mesh (OBJ)

OPERATION:
1. grab image and camera pose from ROS
2. estimate normal from mesh
3. generate endeffector pose

OUTPUT:

"""

import sys
import time
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Slerp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
from ros_gz_interfaces.srv import SpawnEntity

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
import pandas as pd

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
    get_current_tool_flange_posx,
    ROBOT_MODE_AUTONOMOUS,
)

# Gazebo cam frame: X=forward, Y=left, Z=up
# camera optical frame: X=right, Y=down, Z=forward
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

MESH_DIMENSION = 1000  # 1 mesh unit = 0.001 m

fx = 762.72
fy = fx
cx_k = 640
cy_k = 360
camera_K = np.array([[fx, 0, cx_k], [0, fy, cy_k], [0, 0, 1]])

CNC_mesh_path = "/home/robot_llam/BKyoon/working/chipblowing/arm_ws/src/doosan-robot2/scripts/meshes/VMC-300-l.obj"

IMG_TOPIC = "/camera/image_raw/compressed"


# ── helpers ───────────────────────────────────────────────────────────────────


def make_SE3(R_mat: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


# ── ROS helpers ───────────────────────────────────────────────────────────────


def grab_image(img_topic) -> np.ndarray:
    """Subscribe, grab one frame, unsubscribe."""
    latest = {"img": None}

    def cb(msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        latest["img"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sub = node.create_subscription(CompressedImage, img_topic, cb, 10)
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
    delta = query_world
    mesh_x = delta[0] * MESH_DIMENSION
    mesh_y = delta[1] * MESH_DIMENSION
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


def match_mesh_with_world(mesh):
    bounds = mesh.bounds  # shape (2, 3): [min_xyz, max_xyz]
    x_min, y_min = bounds[0, 0], bounds[0, 1]
    x_max, y_max = bounds[1, 0], bounds[1, 1]

    # Flip X and Y, then shift zero point to the opposite side
    # After flipping: new_x = -x, new_y = -y
    # Zero point moves to (-x_max, -y_max), so shift to bring it to (0, 0)
    mesh.vertices[:, 0] -= x_max  # flip X
    mesh.vertices[:, 1] -= y_max  # flip Y
    # Z is untouched
    return mesh

# ── trajectory helpers ────────────────────────────────────────────────────────


def consistent_rotations(poses: list[np.ndarray]) -> list[np.ndarray]:
    """
    Ensure consecutive rotation matrices don't flip sign during SLERP.
    If the dot product of consecutive quaternions is negative, negate the
    second quaternion to force the short-arc interpolation path.
    """
    poses = [p.copy() for p in poses]
    quats = [R.from_matrix(T[:3, :3]).as_quat() for T in poses]

    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] = -quats[i]
        poses[i][:3, :3] = R.from_quat(quats[i]).as_matrix()

    return poses


def se3_to_doosan_posx(T_b_ee: np.ndarray) -> list:
    """
    SE(3) in BASE frame (meters) -> Doosan posx [x(mm), y(mm), z(mm), rz1, ry, rz2].
    Doosan uses extrinsic ZYZ Euler (degrees).
    """
    t_mm = T_b_ee[:3, 3] * 1000.0
    rz1, ry, rz2 = R.from_matrix(T_b_ee[:3, :3]).as_euler("ZYZ", degrees=True)
    return [
        float(t_mm[0]),
        float(t_mm[1]),
        float(t_mm[2] + 35),
        float(rz1),
        float(ry),
        float(rz2),
    ]


def mesh_pose_to_base(T_world_ee: np.ndarray, T_w_b: np.ndarray) -> np.ndarray:
    """
    Convert EE pose from world frame (meters) to robot base frame (meters).
    T_world_ee must already be in meters before calling this.

    T_b_ee = inv(T_w_b) @ T_world_ee
    """
    return np.linalg.inv(T_w_b) @ T_world_ee


def generate_smooth_trajectory(
    poses: list[np.ndarray],
    n_interp: int = 50,
) -> list[np.ndarray]:
    """
    Given a list of SE(3) poses (4×4, world frame, meters),
    generate a smooth trajectory via:
      - Cubic spline for translation (arc-length parameterised)
      - SLERP for rotation (short-arc guaranteed)

    Parameters
    ----------
    poses    : list of N 4×4 SE(3) matrices (meters)
    n_interp : number of interpolated poses between each consecutive pair

    Returns
    -------
    trajectory : list of 4×4 SE(3) matrices (dense, smooth, meters)
    """
    N = len(poses)
    if N < 2:
        raise ValueError("Need at least 2 poses to generate a trajectory.")

    # arc-length parameterisation
    translations = np.array([T[:3, 3] for T in poses])
    dists = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    dists = np.maximum(dists, 1e-8)
    t_knots = np.concatenate([[0.0], np.cumsum(dists)])
    t_knots /= t_knots[-1]

    # cubic spline on translation
    cs = CubicSpline(t_knots, translations)

    # SLERP on rotation — enforce short-arc via quaternion sign consistency
    quats = [R.from_matrix(T[:3, :3]).as_quat() for T in poses]
    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] = -quats[i]
    rotations = R.from_quat(quats)
    slerp = Slerp(t_knots, rotations)

    # dense parameter values
    t_dense = np.linspace(0.0, 1.0, (N - 1) * n_interp + 1)

    trajectory = []
    for t in t_dense:
        T = np.eye(4)
        T[:3, 3] = cs(t)
        T[:3, :3] = slerp(t).as_matrix()
        trajectory.append(T)

    return trajectory


def visualize_trajectory(
    mesh: trimesh.Trimesh,
    keyframes: list[np.ndarray],
    trajectory: list[np.ndarray],
    axis_len: float = 0.02,
) -> None:
    import open3d as o3d

    geometries = []

    # ── mesh ──────────────────────────────────────────────────────────────────
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices  = o3d.utility.Vector3dVector(mesh.vertices / MESH_DIMENSION)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.75, 0.75, 0.75])
    geometries.append(o3d_mesh)

    # ── trajectory line set ───────────────────────────────────────────────────
    traj_pts = np.array([T[:3, 3] for T in trajectory])
    lines    = [[i, i + 1] for i in range(len(traj_pts) - 1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(traj_pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color([0.26, 0.52, 0.96])   # royalblue
    geometries.append(ls)

    # ── keyframe coordinate frames ────────────────────────────────────────────
    for T in keyframes:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_len, origin=T[:3, 3]
        )
        # rotate the canonical frame into the EE orientation
        frame.rotate(T[:3, :3], center=T[:3, 3])
        geometries.append(frame)

    # ── world origin frame (small, for reference) ────────────────────────────
    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len * 0.5)
    )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="EE Trajectory over CNC Mesh",
        width=1280,
        height=800,
        mesh_show_back_face=True,
    )


def check_reachable(T_base_ee: np.ndarray, robot_reach: float = 0.900) -> bool:
    """
    Rough reachability check for M0609 (max reach ~900mm).
    Returns False if too far from base or below base plane.
    """
    t = T_base_ee[:3, 3]
    dist = np.linalg.norm(t)
    if dist > robot_reach:
        print(f"[check] UNREACHABLE: distance {dist*1000:.1f}mm > {robot_reach*1000:.0f}mm")
        return False
    if t[2] < -0.05:
        print(f"[check] UNREACHABLE: z={t[2]*1000:.1f}mm is below base plane")
        return False
    return True


def execute_trajectory(
    trajectory: list[np.ndarray],
    T_w_b: np.ndarray,
    vel: float = 50.0,
    acc: float = 50.0,
    skip: int = 5,
) -> None:
    """
    Execute the trajectory on the Doosan arm waypoint by waypoint.
    movel() with default mod=0 is synchronous — the arm stops at each
    waypoint before the next command is sent, faithfully following the path.

    Parameters
    ----------
    trajectory : dense list of 4×4 SE(3) in world frame (meters)
    T_w_b      : world-from-base SE(3) (4×4)
    vel, acc   : movel velocity / acceleration limits
    skip       : send every Nth dense pose to reduce command count
    """
    waypoints = trajectory[::skip]
    if trajectory[-1] is not waypoints[-1]:
        waypoints = list(waypoints) + [trajectory[-1]]

    doosan_pose_list = np.zeros((len(waypoints), 6))

    # pre-flight reachability check
    print(f"[exec] Pre-flight check on {len(waypoints)} waypoints...")
    for i, T_world_ee in enumerate(waypoints):
        T_base_ee = mesh_pose_to_base(T_world_ee, T_w_b)
        if not check_reachable(T_base_ee):
            raise RuntimeError(
                f"[exec] Waypoint {i} not reachable — aborting.\n"
                f"  world={T_world_ee[:3,3]}  base={T_base_ee[:3,3]}"
            )

    print(f"[exec] Executing {len(waypoints)} waypoints (skip={skip})...")
    for i, T_world_ee in enumerate(waypoints):
        T_base_ee   = mesh_pose_to_base(T_world_ee, T_w_b)
        doosan_pose = se3_to_doosan_posx(T_base_ee)
        doosan_pose_list[i,:] = doosan_pose
        print(f"[exec, world] wp{i:03d}  base(m)={np.round(T_world_ee[:3,3],7)}")
        print(f"[exec] wp{i:03d}  base(m)={np.round(T_base_ee[:3,3],3)}  posx={[f'{v:.1f}' for v in doosan_pose]}")
        movel(posx(*doosan_pose), vel=vel, acc=acc)
        time.sleep(0.5)
        actual, _ = get_current_posx()
        print(f"[verify] commanded: {[f'{v:.1f}' for v in doosan_pose]}")
        print(f"[verify] actual   : {[f'{v:.1f}' for v in actual]}")
        print(f"[verify] delta    : {[f'{doosan_pose[i]-actual[i]:.1f}' for i in range(6)]}")


    print("[exec] Trajectory complete.")

    df = pd.DataFrame(doosan_pose_list)
    df.to_csv("doosan_pose_list.csv", index = False, header = ["tx", "ty", "tz", "rz1", "ry", "rz2"])

# ── main ──────────────────────────────────────────────────────────────────────


def main(args=None):
    # load CNC mesh
    CNC_mesh = trimesh.load_mesh(CNC_mesh_path)
    # wake up robot arm and move to initial pose
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    move_to_init()
    time.sleep(3)

    # get current camera view
    snapshot = grab_image(IMG_TOPIC)
    # get current link 6 pose (w.r.t. BASE)
    R_b_6, t_b_6 = get_base_to_link6()

    # build transform chain
    T_w_b = make_SE3(WLD_TO_BASE_R.as_matrix(), WLD_TO_BASE_T)
    T_b_6 = make_SE3(R_b_6, t_b_6)
    T_6_cm = make_SE3(
        (L6_TO_CAM_R * GAZ_TO_OPT_R).as_matrix(), L6_TO_CAM_T
    )  # link 6 to camera model frame (z-forward / y-downward)
    print("T_6_cm, ", T_6_cm)
    T_w_cm = T_w_b @ T_b_6 @ T_6_cm


    # stack keyframes
    keyframes = []
    point_idx = 0
    while True:
        print(f"\n[loop] === Point {point_idx} ===")

        # pick pixel
        pu, pv = pick_xy_from_camera(snapshot)
        print(f"[main] Picked pixel: ({pu:.1f}, {pv:.1f})")

        # depth via mesh raycasting
        z_cam = get_depth_from_mesh(pu, pv, T_w_cm, CNC_mesh)
        print("depth: ", z_cam, " world_z: ", T_w_cm[2, 3] - z_cam)

        # pixel -> world
        query_world = pixel_to_world(pu, pv, z_cam, T_w_cm)
        print(f"[main] query_world: {query_world}")

        # world -> mesh
        query_mesh_x, query_mesh_y = world_to_mesh(query_world)
        query_mesh_z = (query_world[2]) * MESH_DIMENSION
        print(
            f"[main] query_mesh_x={query_mesh_x:.2f}  query_mesh_y={query_mesh_y:.2f}"
        )

        # query normals
        normals = query_mesh_normals(
            CNC_mesh,
            query_mesh_x,
            query_mesh_y,
            10,
            total_points=5_000_000,
            z=query_mesh_z,
        )

        # find desired SE(3)
        d = 0.2  ## offset from query surface: 0.1m
        pose = compute_se3_pose(normals, d * MESH_DIMENSION)
        normals["pose"] = pose

        # visualize normals and pose
        visualize_normals(CNC_mesh, normals, normal_length=100)

        # stack ee pose
        # pose["T"] is 4×4 in mesh coords (mm), z-axis = approach direction
        T_world_ee = pose["T"].copy()
        T_world_ee[:3, 3] /= MESH_DIMENSION  # mm -> m
        keyframes.append(T_world_ee)

        ans = input("[loop] Add another point? [y/N]: ").strip().lower()
        if ans != "y":
            break

    print(f"\n[traj] Collected {len(keyframes)} keyframe(s).")

    # ── generate smooth trajectory ────────────────────────────────────────────
    keyframes = consistent_rotations(keyframes)

    if len(keyframes) >= 2:
        trajectory = generate_smooth_trajectory(keyframes, n_interp=50)
        print(f"[traj] Generated {len(trajectory)} interpolated poses.")
    else:
        trajectory = keyframes
        print("[traj] Single keyframe — skipping interpolation.")

    # ── visualise ─────────────────────────────────────────────────────────────
    visualize_trajectory(CNC_mesh, keyframes, trajectory, axis_len=0.2)

    # ── execute ───────────────────────────────────────────────────────────────
    confirm = input("[exec] Execute trajectory on robot? [y/N]: ").strip().lower()
    if confirm == "y":
        execute_trajectory(trajectory, T_w_b, vel=50, acc=50, skip=5)

    rclpy.shutdown()

if __name__ == "__main__":
    main()

"""
TODO: merge GMM estimation
"""
