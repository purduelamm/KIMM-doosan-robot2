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
from dataclasses import dataclass, field
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
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    BoundingVolume,
    CollisionObject,
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
from sensor_msgs.msg import CompressedImage
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
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
INIT_POSX = [
    -142.39569091796875,
    555.5761108398438,
    369.9364318847656,
    81.62047576904297,
    180.0,
    156.6204833984375,
]
MOVEIT_GROUP = "manipulator"
MOVEIT_EE_LINK = "link_6"
MOVEIT_BASE_FRAME = "base_link"
MOVEIT_PLANNING_SERVICE = "plan_kinematic_path"
MOVEIT_MOVE_ACTION = "move_action"
MOVEIT_SCENE_SERVICE = "apply_planning_scene"
MOVEIT_SCENE_TOPIC = "planning_scene"
MOVEIT_EXECUTE_ACTION = "execute_trajectory"
MOVEIT_PLANNING_TIME = 8.0
MOVEIT_PLANNING_ATTEMPTS = 10
MOVEIT_POS_TOLERANCE = 0.003
MOVEIT_ORI_TOLERANCE = 0.05
CNC_COLLISION_OBJECT_ID = "cnc_mesh_obstacle"
ALLOW_CNC_COLLISION_LINKS = ["blower_link"]
DISABLED_SELF_COLLISION_PAIRS = [
    ("base_link", "link_1"),
    ("base_link", "link_3"),
    ("link_1", "link_2"),
    ("link_1", "link_3"),
    ("link_1", "link_5"),
    ("link_1", "link_6"),
    ("link_2", "link_3"),
    ("link_2", "link_4"),
    ("link_2", "link_5"),
    ("link_2", "link_6"),
    ("link_3", "link_4"),
    ("link_3", "link_5"),
    ("link_3", "link_6"),
    ("link_4", "link_5"),
    ("link_4", "link_6"),
    ("link_5", "link_6"),
    ("blower_link", "link_6"),
]


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


def doosan_posx_to_base_se3(doosan_pose: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array(doosan_pose[:3], dtype=float) / 1000.0
    T[:3, :3] = R.from_euler("ZYZ", doosan_pose[3:6], degrees=True).as_matrix()
    return T


def move_to_init(cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray):
    print("[init] Planning collision-free move to initial posx with MoveIt2...")
    apply_cnc_obstacle(cnc_mesh, T_w_b)

    planner_kind, planner_client = create_moveit_planner()
    target_base_ee = doosan_posx_to_base_se3(INIT_POSX)
    if not check_reachable(target_base_ee):
        raise RuntimeError("[init] Initial posx failed rough reachability check.")

    traj = plan_moveit_segment(planner_kind, planner_client, target_base_ee, None, 0)
    init_plan = TrajectoryPlan(
        poses=[],
        robot_trajectories=[traj],
        planned_with_moveit=True,
    )
    execute_moveit_trajectory(init_plan)
    time.sleep(1)
    print("[init] Moved to initial point.")
    print("current: ", get_current_posx())


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


@dataclass
class TrajectoryPlan:
    """
    Cartesian guide poses are kept for visualization; robot_trajectories are
    the MoveIt collision-checked plans used for execution.
    """

    poses: list[np.ndarray]
    robot_trajectories: list = field(default_factory=list)
    planned_with_moveit: bool = False

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, item):
        return self.poses[item]

    def __iter__(self):
        return iter(self.poses)


def moveit_name_candidates(name: str) -> list[str]:
    base = name.lstrip("/")
    namespace = node.get_namespace().strip("/")
    candidates = []
    if namespace:
        candidates.append(f"/{namespace}/{base}")
    candidates.append(f"/{base}")
    candidates.append(base)
    return list(dict.fromkeys(candidates))


def create_available_service_client(srv_type, name: str, timeout_sec: float = 10.0):
    deadline = time.time() + timeout_sec
    last_client = None
    for candidate in moveit_name_candidates(name):
        last_client = node.create_client(srv_type, candidate)
        remaining = max(0.1, deadline - time.time())
        print(f"[moveit] Waiting for {candidate}...")
        if last_client.wait_for_service(timeout_sec=min(2.0, remaining)):
            print(f"[moveit] Using service {candidate}.")
            return last_client
    raise RuntimeError(
        f"[moveit] {name} is not available. Tried: "
        f"{', '.join(moveit_name_candidates(name))}. Launch MoveIt2 before running this script."
    )


def create_available_action_client(action_type, name: str, timeout_sec: float = 10.0):
    deadline = time.time() + timeout_sec
    last_client = None
    for candidate in moveit_name_candidates(name):
        last_client = ActionClient(node, action_type, candidate)
        remaining = max(0.1, deadline - time.time())
        print(f"[exec] Waiting for MoveIt action {candidate}...")
        if last_client.wait_for_server(timeout_sec=min(2.0, remaining)):
            print(f"[exec] Using MoveIt action {candidate}.")
            return last_client
    raise RuntimeError(
        f"[exec] {name} is not available. Tried: "
        f"{', '.join(moveit_name_candidates(name))}. Is move_group running?"
    )


def create_moveit_planner(timeout_sec: float = 10.0):
    try:
        return (
            "service",
            create_available_service_client(
                GetMotionPlan, MOVEIT_PLANNING_SERVICE, timeout_sec=timeout_sec
            ),
        )
    except RuntimeError as service_error:
        print(f"[moveit] {service_error}")

    try:
        return (
            "action",
            create_available_action_client(
                MoveGroup, MOVEIT_MOVE_ACTION, timeout_sec=timeout_sec
            ),
        )
    except RuntimeError as action_error:
        raise RuntimeError(
            "[moveit] No MoveIt planner endpoint is available. Tried planning "
            f"service candidates ({', '.join(moveit_name_candidates(MOVEIT_PLANNING_SERVICE))}) "
            f"and MoveGroup action candidates ({', '.join(moveit_name_candidates(MOVEIT_MOVE_ACTION))}). "
            "Start MoveIt2 move_group for this robot before planning."
        ) from action_error


def publish_planning_scene_diff(scene: PlanningScene, timeout_sec: float = 3.0) -> None:
    publishers = []
    for candidate in moveit_name_candidates(MOVEIT_SCENE_TOPIC):
        print(f"[moveit] Advertising planning-scene fallback topic {candidate}...")
        publishers.append(node.create_publisher(PlanningScene, candidate, 10))

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        for publisher in publishers:
            publisher.publish(scene)
        rclpy.spin_once(node, timeout_sec=0.1)

    print("[moveit] Published CNC mesh planning-scene diff by topic fallback.")


def wait_for_service(client, name: str, timeout_sec: float = 10.0) -> None:
    print(f"[moveit] Waiting for {name}...")
    if not client.wait_for_service(timeout_sec=timeout_sec):
        raise RuntimeError(
            f"[moveit] {name} is not available. Launch MoveIt2 before running this script."
        )


def call_service_sync(client, request, name: str, timeout_sec: float = 30.0):
    future = client.call_async(request)
    t0 = time.time()
    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > timeout_sec:
            raise RuntimeError(f"[moveit] Timed out waiting for {name}.")
    result = future.result()
    if result is None:
        raise RuntimeError(f"[moveit] {name} failed: {future.exception()}")
    return result


def make_pose_msg(T: np.ndarray) -> Pose:
    quat = R.from_matrix(T[:3, :3]).as_quat()
    pose = Pose()
    pose.position = Point(x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3]))
    pose.orientation = Quaternion(
        x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
    )
    return pose


def make_pose_constraint(T_base_ee: np.ndarray) -> Constraints:
    pose = make_pose_msg(T_base_ee)

    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [MOVEIT_POS_TOLERANCE]

    region = BoundingVolume()
    region.primitives.append(sphere)
    region.primitive_poses.append(pose)

    pos_constraint = PositionConstraint()
    pos_constraint.header.frame_id = MOVEIT_BASE_FRAME
    pos_constraint.link_name = MOVEIT_EE_LINK
    pos_constraint.constraint_region = region
    pos_constraint.weight = 1.0

    ori_constraint = OrientationConstraint()
    ori_constraint.header.frame_id = MOVEIT_BASE_FRAME
    ori_constraint.link_name = MOVEIT_EE_LINK
    ori_constraint.orientation = pose.orientation
    ori_constraint.absolute_x_axis_tolerance = MOVEIT_ORI_TOLERANCE
    ori_constraint.absolute_y_axis_tolerance = MOVEIT_ORI_TOLERANCE
    ori_constraint.absolute_z_axis_tolerance = MOVEIT_ORI_TOLERANCE
    ori_constraint.weight = 1.0

    constraints = Constraints()
    constraints.position_constraints.append(pos_constraint)
    constraints.orientation_constraints.append(ori_constraint)
    return constraints


def mesh_to_collision_object(
    cnc_mesh: trimesh.Trimesh,
    T_w_b: np.ndarray,
    object_id: str = CNC_COLLISION_OBJECT_ID,
) -> CollisionObject:
    mesh_for_collision = cnc_mesh.copy()
    print(
        "[moveit] Using full CNC mesh collision object "
        f"({len(mesh_for_collision.vertices)} vertices, {len(mesh_for_collision.faces)} faces)."
    )

    T_b_w = np.linalg.inv(T_w_b)
    vertices_world = mesh_for_collision.vertices / MESH_DIMENSION
    vertices_base = (T_b_w @ np.c_[vertices_world, np.ones(len(vertices_world))].T).T[
        :, :3
    ]

    mesh_msg = Mesh()
    mesh_msg.vertices = [
        Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in vertices_base
    ]
    mesh_msg.triangles = [
        MeshTriangle(vertex_indices=[int(f[0]), int(f[1]), int(f[2])])
        for f in mesh_for_collision.faces
    ]

    obj = CollisionObject()
    obj.header.frame_id = MOVEIT_BASE_FRAME
    obj.id = object_id
    obj.meshes.append(mesh_msg)
    obj.mesh_poses.append(Pose())
    obj.mesh_poses[0].orientation.w = 1.0
    obj.operation = CollisionObject.ADD
    return obj


def allow_collision_pairs(scene: PlanningScene, object_id: str, link_names: list[str]) -> None:
    allowed_pairs = {frozenset(pair) for pair in DISABLED_SELF_COLLISION_PAIRS}
    for link_name in link_names:
        allowed_pairs.add(frozenset((object_id, link_name)))

    names = sorted({name for pair in allowed_pairs for name in pair})
    scene.allowed_collision_matrix.entry_names = names
    scene.allowed_collision_matrix.entry_values = []

    for row_name in names:
        entry = AllowedCollisionEntry()
        entry.enabled = [
            frozenset((row_name, col_name)) in allowed_pairs
            for col_name in names
        ]
        scene.allowed_collision_matrix.entry_values.append(entry)


def apply_cnc_obstacle(cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray) -> None:
    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects.append(
        mesh_to_collision_object(cnc_mesh, T_w_b, CNC_COLLISION_OBJECT_ID)
    )
    allow_collision_pairs(scene, CNC_COLLISION_OBJECT_ID, ALLOW_CNC_COLLISION_LINKS)

    try:
        client = create_available_service_client(
            ApplyPlanningScene, MOVEIT_SCENE_SERVICE, timeout_sec=6.0
        )
    except RuntimeError as exc:
        print(f"[moveit] {exc}")
        publish_planning_scene_diff(scene)
        return

    req = ApplyPlanningScene.Request()
    req.scene = scene
    resp = call_service_sync(client, req, MOVEIT_SCENE_SERVICE, timeout_sec=20.0)
    if not resp.success:
        raise RuntimeError("[moveit] Failed to apply CNC mesh collision object.")
    print("[moveit] CNC mesh added to the planning scene as an obstacle.")


def final_state_from_trajectory(robot_trajectory) -> RobotState:
    joint_traj = robot_trajectory.joint_trajectory
    if not joint_traj.points:
        raise RuntimeError("[moveit] Planned trajectory has no joint points.")

    state = RobotState()
    state.joint_state.name = list(joint_traj.joint_names)
    state.joint_state.position = list(joint_traj.points[-1].positions)
    state.is_diff = False
    return state


def plan_moveit_segment(
    planner_kind: str,
    planner_client,
    target_base_ee: np.ndarray,
    start_state: RobotState | None,
    segment_idx: int,
):
    motion_req = MotionPlanRequest()
    motion_req.group_name = MOVEIT_GROUP
    motion_req.num_planning_attempts = MOVEIT_PLANNING_ATTEMPTS
    motion_req.allowed_planning_time = MOVEIT_PLANNING_TIME
    motion_req.max_velocity_scaling_factor = 0.2
    motion_req.max_acceleration_scaling_factor = 0.2
    motion_req.goal_constraints.append(make_pose_constraint(target_base_ee))

    if start_state is None:
        motion_req.start_state.is_diff = True
    else:
        motion_req.start_state = start_state

    if planner_kind == "service":
        req = GetMotionPlan.Request()
        req.motion_plan_request = motion_req
        resp = call_service_sync(
            planner_client,
            req,
            MOVEIT_PLANNING_SERVICE,
            timeout_sec=MOVEIT_PLANNING_TIME + 15.0,
        )

        error_code = resp.motion_plan_response.error_code.val
        if error_code != 1:
            raise RuntimeError(
                f"[moveit] Segment {segment_idx} planning failed with error code {error_code}."
            )

        traj = resp.motion_plan_response.trajectory
    else:
        goal = MoveGroup.Goal()
        goal.request = motion_req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True

        send_future = planner_client.send_goal_async(goal)
        while rclpy.ok() and not send_future.done():
            rclpy.spin_once(node, timeout_sec=0.1)

        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            raise RuntimeError(f"[moveit] MoveGroup rejected planning segment {segment_idx}.")

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(node, timeout_sec=0.1)

        action_result = result_future.result()
        if action_result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                f"[moveit] MoveGroup action failed for segment {segment_idx}; "
                f"action status={action_result.status}."
            )

        result = action_result.result
        error_code = result.error_code.val
        if error_code != 1:
            raise RuntimeError(
                f"[moveit] Segment {segment_idx} planning failed with error code {error_code}."
            )
        traj = result.planned_trajectory

    print(
        f"[moveit] Segment {segment_idx} planned: "
        f"{len(traj.joint_trajectory.points)} joint points."
    )
    return traj


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


def generate_cartesian_guide_trajectory(
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


def generate_smooth_trajectory(
    poses: list[np.ndarray],
    n_interp: int = 50,
    T_w_b: np.ndarray | None = None,
    cnc_mesh: trimesh.Trimesh | None = None,
) -> TrajectoryPlan:
    """
    Generate a trajectory through the keyframes. When T_w_b and cnc_mesh are
    provided, the CNC mesh is added to MoveIt2 as a collision object and each
    segment is planned by MoveIt2. The returned Cartesian poses are only a
    visualization guide; execution uses MoveIt's robot trajectories.
    """
    if len(poses) < 2:
        raise ValueError("Need at least 2 poses to generate a trajectory.")

    cartesian_guide = generate_cartesian_guide_trajectory(poses, n_interp=n_interp)
    if T_w_b is None or cnc_mesh is None:
        print("[moveit] Missing T_w_b or CNC mesh; using Cartesian guide only.")
        return TrajectoryPlan(poses=cartesian_guide, planned_with_moveit=False)

    apply_cnc_obstacle(cnc_mesh, T_w_b)
    planner_kind, planner_client = create_moveit_planner()

    robot_trajectories = []
    start_state = None
    for i, target_world_ee in enumerate(poses):
        target_base_ee = mesh_pose_to_base(target_world_ee, T_w_b)
        target_base_ee = target_base_ee.copy()
        target_base_ee[2, 3] += 0.035

        if not check_reachable(target_base_ee):
            raise RuntimeError(f"[moveit] Keyframe {i} is outside the rough reach check.")

        traj = plan_moveit_segment(
            planner_kind, planner_client, target_base_ee, start_state, i
        )
        robot_trajectories.append(traj)
        start_state = final_state_from_trajectory(traj)

    return TrajectoryPlan(
        poses=cartesian_guide,
        robot_trajectories=robot_trajectories,
        planned_with_moveit=True,
    )


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


def execute_moveit_trajectory(plan: TrajectoryPlan) -> None:
    if not plan.robot_trajectories:
        raise RuntimeError("[exec] MoveIt plan has no robot trajectories to execute.")

    action_client = create_available_action_client(
        ExecuteTrajectory, MOVEIT_EXECUTE_ACTION
    )

    for i, robot_trajectory in enumerate(plan.robot_trajectories):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_trajectory
        send_future = action_client.send_goal_async(goal)
        while rclpy.ok() and not send_future.done():
            rclpy.spin_once(node, timeout_sec=0.1)

        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            raise RuntimeError(f"[exec] MoveIt rejected trajectory segment {i}.")

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(node, timeout_sec=0.1)

        result = result_future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                f"[exec] MoveIt failed executing segment {i}; action status={result.status}."
            )
        print(f"[exec] MoveIt executed segment {i}.")

    print("[exec] MoveIt trajectory complete.")


def execute_trajectory(
    trajectory: list[np.ndarray] | TrajectoryPlan,
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
    if isinstance(trajectory, TrajectoryPlan) and trajectory.planned_with_moveit:
        execute_moveit_trajectory(trajectory)
        return

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
    T_w_b = make_SE3(WLD_TO_BASE_R.as_matrix(), WLD_TO_BASE_T)

    # wake up robot arm and move to initial pose
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    move_to_init(CNC_mesh, T_w_b)
    time.sleep(3)

    # get current camera view
    snapshot = grab_image(IMG_TOPIC)
    # get current link 6 pose (w.r.t. BASE)
    R_b_6, t_b_6 = get_base_to_link6()

    # build transform chain
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
        trajectory = generate_smooth_trajectory(
            keyframes, n_interp=50, T_w_b=T_w_b, cnc_mesh=CNC_mesh
        )
        if trajectory.planned_with_moveit:
            print(
                f"[traj] MoveIt planned {len(trajectory.robot_trajectories)} "
                f"segment(s); {len(trajectory)} poses kept for visualization."
            )
        else:
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
