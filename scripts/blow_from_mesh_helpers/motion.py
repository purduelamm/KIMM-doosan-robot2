import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import cv2
import matplotlib
import numpy as np
import rclpy
import tf2_ros
import trimesh
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
from PIL import Image as PILImage
from PIL import ImageTk
from rclpy.action import ActionClient
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sensor_msgs.msg import CompressedImage, Image
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from tf2_ros import Buffer, TransformListener

try:
    matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
except Exception as exc:
    print(f"[detect] Could not select preferred Matplotlib backend: {exc}")
import matplotlib.pyplot as plt

from .config import *
from .math_utils import make_SE3
from .ros_context import (
    air_node,
    get_current_posx,
    get_current_tool_flange_posx,
    is_doosan_robot,
    movej,
    movel,
    node,
    posj,
    posx,
    set_robot_mode,
)

def doosan_posx_to_base_se3(doosan_pose: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array(doosan_pose[:3], dtype=float) / 1000.0
    T[:3, :3] = R.from_euler("ZYZ", doosan_pose[3:6], degrees=True).as_matrix()
    return T


def typed_target_to_base_se3(target: dict, T_w_b: np.ndarray | None = None) -> np.ndarray:
    target_type = target.get("type")
    if target_type == "doosan_posx":
        if not is_doosan_robot():
            raise ValueError("Target type 'doosan_posx' is only supported for robot.type 'doosan'.")
        return doosan_posx_to_base_se3(target["values"])
    if target_type == "se3_xyz_quat":
        frame = target.get("frame", "base")
        T = np.eye(4)
        T[:3, 3] = np.array(target["translation"], dtype=float)
        T[:3, :3] = R.from_quat(np.array(target["quaternion_xyzw"], dtype=float)).as_matrix()
        if frame == "base":
            return T
        if frame == "world":
            if T_w_b is None:
                raise ValueError("world-frame se3_xyz_quat target requires T_w_b.")
            return np.linalg.inv(T_w_b) @ T
        raise ValueError(f"Unsupported se3_xyz_quat frame '{frame}'. Use base or world.")
    raise ValueError(f"Target type '{target_type}' cannot be converted to a base pose.")


def typed_target_to_joints(target: dict) -> list[float]:
    if target.get("type") != "joint_values":
        raise ValueError(f"Target type '{target.get('type')}' is not a joint target.")
    return [float(value) for value in target["values"]]


def normalize_motion_target(target) -> dict:
    if isinstance(target, dict):
        return target
    if isinstance(target, (list, tuple)):
        return {"type": "doosan_posx", "values": list(target)}
    raise ValueError(f"Unsupported motion target: {target!r}")


class MotionBackend:
    """Replace this interface to connect a different robot motion stack."""

    def apply_obstacles(self, cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray) -> None:
        raise NotImplementedError

    def plan_to_pose(self, T_base_ee: np.ndarray, start_state=None, segment_idx: int = 0):
        raise NotImplementedError

    def plan_to_joints(self, joints: list[float], start_state=None, segment_idx: int = 0):
        raise NotImplementedError

    def execute_plan(self, plan) -> None:
        raise NotImplementedError

    def plan_to_target(self, target, T_w_b: np.ndarray | None = None, segment_idx: int = 0):
        target = normalize_motion_target(target)
        if target.get("type") == "joint_values":
            return self.plan_to_joints(
                typed_target_to_joints(target),
                start_state=None,
                segment_idx=segment_idx,
            )
        return self.plan_to_pose(
            typed_target_to_base_se3(target, T_w_b),
            start_state=None,
            segment_idx=segment_idx,
        )

    def move_to_init(self, init_target, cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray):
        plan = self.plan_to_target(init_target, T_w_b=T_w_b, segment_idx=0)
        self.execute_plan(plan)

    def move_to_joints(self, joints_or_target) -> None:
        target = normalize_motion_target(joints_or_target)
        if target.get("type") == "joint_values":
            joints = typed_target_to_joints(target)
        else:
            joints = typed_target_to_joints({"type": "joint_values", "values": joints_or_target})
        plan = self.plan_to_joints(joints, start_state=None, segment_idx=0)
        self.execute_plan(plan)


class MoveItMotionBackend(MotionBackend):
    """Default obstacle-aware motion backend."""

    def __init__(self):
        self.planner_kind = None
        self.planner_client = None

    def apply_obstacles(self, cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray) -> None:
        if ROBOT_TYPE == "ur":
            print("[moveit] Skipping CNC mesh collision object for robot.type=ur.")
            return
        apply_cnc_obstacle(cnc_mesh, T_w_b)

    def _planner(self):
        if self.planner_kind is None or self.planner_client is None:
            self.planner_kind, self.planner_client = create_moveit_planner()
        return self.planner_kind, self.planner_client

    def plan_to_pose(self, T_base_ee: np.ndarray, start_state=None, segment_idx: int = 0):
        if not check_reachable(T_base_ee):
            raise RuntimeError(f"[moveit] Target pose {segment_idx} failed rough reachability check.")
        planner_kind, planner_client = self._planner()
        return plan_moveit_segment(
            planner_kind, planner_client, T_base_ee, start_state, segment_idx
        )

    def plan_to_joints(self, joints: list[float], start_state=None, segment_idx: int = 0):
        planner_kind, planner_client = self._planner()
        return plan_moveit_joints(
            planner_kind, planner_client, joints, start_state, segment_idx
        )

    def execute_plan(self, plan) -> None:
        if isinstance(plan, TrajectoryPlan):
            if plan.target_base_pose_groups:
                self.execute_target_groups_online(plan)
                return
            execute_moveit_trajectory(plan)
            return
        execute_moveit_trajectory(
            TrajectoryPlan(poses=[], robot_trajectories=[plan], planned_with_moveit=True)
        )

    def wait_for_current_state_update(self) -> None:
        settle_sec = float(CONFIG.get("execution", {}).get("online_segment_settle_sec", 0.2))
        if settle_sec > 0:
            time.sleep(settle_sec)
        rclpy.spin_once(node, timeout_sec=0.1)

    def execute_target_groups_online(self, plan: "TrajectoryPlan") -> None:
        executed = 0
        skipped_keypoints = 0
        skipped_cones = 0
        for group in plan.target_base_pose_groups:
            keypoint_idx = int(group["keypoint_idx"])
            try:
                print(f"[online] Planning {planned_pose_label(keypoint_idx)} from current state...")
                keypoint_traj = self.plan_to_pose(
                    group["keypoint_base_pose"],
                    start_state=None,
                    segment_idx=keypoint_idx,
                )
                execute_moveit_trajectory(
                    TrajectoryPlan(
                        poses=[],
                        robot_trajectories=[keypoint_traj],
                        planned_with_moveit=True,
                    )
                )
                self.wait_for_current_state_update()
                executed += 1
            except RuntimeError as exc:
                skipped_keypoints += 1
                skipped_cones += len(group["cone_base_poses"])
                print(
                    f"[online] Skipping keypoint {keypoint_idx} and its cone sweep: {exc}"
                )
                continue

            for cone_idx, cone_base_pose in enumerate(group["cone_base_poses"]):
                try:
                    print(
                        f"[online] Planning {planned_pose_label(keypoint_idx, cone_idx)} "
                        "from current state..."
                    )
                    cone_traj = self.plan_to_pose(
                        cone_base_pose,
                        start_state=None,
                        segment_idx=keypoint_idx,
                    )
                    execute_moveit_trajectory(
                        TrajectoryPlan(
                            poses=[],
                            robot_trajectories=[cone_traj],
                            planned_with_moveit=True,
                        )
                    )
                    self.wait_for_current_state_update()
                    executed += 1
                except RuntimeError as exc:
                    skipped_cones += 1
                    print(
                        f"[online] Skipping {planned_pose_label(keypoint_idx, cone_idx)}: {exc}"
                    )

        print(
            f"[online] Executed {executed} segment(s); "
            f"skipped {skipped_keypoints} keypoint(s), {skipped_cones} cone pose(s)."
        )
        if executed == 0:
            raise RuntimeError("[online] No target segment could be planned and executed.")

    def move_to_init(self, init_target, cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray):
        while True:
            try:
                print("[init] Planning collision-free move to initial target with MoveIt2...")
                self.apply_obstacles(cnc_mesh, T_w_b)
                super().move_to_init(init_target, cnc_mesh, T_w_b)
                time.sleep(1)
                print("[init] Moved to initial point.")
                if is_doosan_robot():
                    print("current: ", get_current_posx())
                break
            except RuntimeError:
                print("[init] Failed to find trajectory. Retrying...")

    def move_to_joints(self, joints: list[float]) -> None:
        print(f"[init] Planning collision-free joint move: {joints}")
        super().move_to_joints(joints)
        time.sleep(1)


class DoosanDirectMotionBackend(MotionBackend):
    """Editable direct-motion backend for users who want movel/movej commands."""

    def apply_obstacles(self, cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray) -> None:
        pass

    def plan_to_pose(self, T_base_ee: np.ndarray, start_state=None, segment_idx: int = 0):
        return T_base_ee

    def plan_to_joints(self, joints: list[float], start_state=None, segment_idx: int = 0):
        return list(joints)

    def execute_plan(self, plan) -> None:
        if isinstance(plan, TrajectoryPlan):
            for target in plan.robot_trajectories:
                self.execute_plan(target)
            return
        if isinstance(plan, list) and len(plan) == len(MOVEIT_JOINT_NAMES):
            self.move_to_joints(plan)
            return
        doosan_pose = se3_to_doosan_posx(plan)
        movel(posx(*doosan_pose), vel=50, acc=50)

    def move_to_joints(self, joints_or_target) -> None:
        target = normalize_motion_target(joints_or_target)
        if target.get("type") == "joint_values":
            joints = typed_target_to_joints(target)
        else:
            joints = typed_target_to_joints({"type": "joint_values", "values": joints_or_target})
        movej(posj(*joints), vel=60, acc=60)


def create_motion_backend() -> MotionBackend:
    backend_name = CONFIG.get("motion_backend", "moveit")
    if backend_name == "moveit":
        return MoveItMotionBackend()
    if backend_name == "doosan_direct":
        if not is_doosan_robot():
            raise ValueError("motion_backend 'doosan_direct' is only supported for robot.type 'doosan'.")
        return DoosanDirectMotionBackend()
    raise ValueError(f"Unsupported motion_backend '{backend_name}'.")


# ── coordinate helpers ────────────────────────────────────────────────────────

@dataclass
class TrajectoryPlan:
    """
    Cartesian guide poses are kept for visualization; robot_trajectories are
    the MoveIt collision-checked plans used for execution.
    """

    poses: list[np.ndarray]
    robot_trajectories: list = field(default_factory=list)
    planned_with_moveit: bool = False
    target_poses: list[np.ndarray] = field(default_factory=list)
    target_base_pose_groups: list[dict] = field(default_factory=list)

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


def make_joint_constraint(joints: list[float]) -> Constraints:
    if len(joints) != len(MOVEIT_JOINT_NAMES):
        raise ValueError(
            f"Joint target has {len(joints)} values, expected "
            f"{len(MOVEIT_JOINT_NAMES)} for {MOVEIT_JOINT_NAMES}."
        )

    constraints = Constraints()
    for name, position in zip(MOVEIT_JOINT_NAMES, joints):
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = name
        joint_constraint.position = float(position)
        joint_constraint.tolerance_above = MOVEIT_JOINT_TOLERANCE
        joint_constraint.tolerance_below = MOVEIT_JOINT_TOLERANCE
        joint_constraint.weight = 1.0
        constraints.joint_constraints.append(joint_constraint)
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
    print(f"[moveit] Allowed {len(allowed_pairs)} collision pair(s) in planning scene.")


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
    motion_req.max_velocity_scaling_factor = MOVEIT_VELOCITY_SCALING
    motion_req.max_acceleration_scaling_factor = MOVEIT_ACCELERATION_SCALING
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
            print(
                f"[moveit] NO PATH FOUND for segment {segment_idx}; "
                f"planner returned error code {error_code}."
            )
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
            print(
                f"[moveit] NO PATH FOUND for segment {segment_idx}; "
                f"planner returned error code {error_code}."
            )
            raise RuntimeError(
                f"[moveit] Segment {segment_idx} planning failed with error code {error_code}."
            )
        traj = result.planned_trajectory

    print(
        f"[moveit] Segment {segment_idx} planned: "
        f"{len(traj.joint_trajectory.points)} joint points."
    )
    return traj


def plan_moveit_joints(
    planner_kind: str,
    planner_client,
    joints: list[float],
    start_state: RobotState | None,
    segment_idx: int,
):
    motion_req = MotionPlanRequest()
    motion_req.group_name = MOVEIT_GROUP
    motion_req.num_planning_attempts = MOVEIT_PLANNING_ATTEMPTS
    motion_req.allowed_planning_time = MOVEIT_PLANNING_TIME
    motion_req.max_velocity_scaling_factor = MOVEIT_VELOCITY_SCALING
    motion_req.max_acceleration_scaling_factor = MOVEIT_ACCELERATION_SCALING
    motion_req.goal_constraints.append(make_joint_constraint(joints))

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
            print(
                f"[moveit] NO PATH FOUND for joint target {segment_idx}; "
                f"planner returned error code {error_code}."
            )
            raise RuntimeError(
                f"[moveit] Joint target {segment_idx} planning failed with error code {error_code}."
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
            raise RuntimeError(f"[moveit] MoveGroup rejected joint target {segment_idx}.")

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(node, timeout_sec=0.1)

        action_result = result_future.result()
        if action_result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                f"[moveit] MoveGroup action failed for joint target {segment_idx}; "
                f"action status={action_result.status}."
            )

        result = action_result.result
        error_code = result.error_code.val
        if error_code != 1:
            print(
                f"[moveit] NO PATH FOUND for joint target {segment_idx}; "
                f"planner returned error code {error_code}."
            )
            raise RuntimeError(
                f"[moveit] Joint target {segment_idx} planning failed with error code {error_code}."
            )
        traj = result.planned_trajectory

    print(
        f"[moveit] Joint target {segment_idx} planned: "
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
        # float(t_mm[2] + 35),
        float(t_mm[2]),
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


def cone_sweep_config() -> dict:
    return CONFIG.get("path_actions", {}).get("cone_sweep", {})


def cone_sweep_enabled() -> bool:
    return bool(cone_sweep_config().get("enabled", False))


def cone_sweep_poses(T_world_ee: np.ndarray) -> list[np.ndarray]:
    cfg = cone_sweep_config()
    if not bool(cfg.get("enabled", False)):
        return []

    samples = max(0, int(cfg.get("samples", 0)))
    if samples <= 0:
        return []

    cone_angle = np.deg2rad(float(cfg.get("angle_deg", 0.0)))
    x0 = T_world_ee[:3, 0]
    y0 = T_world_ee[:3, 1]
    z0 = T_world_ee[:3, 2]
    position = T_world_ee[:3, 3]

    poses = []
    for i in range(samples):
        phi = 2.0 * np.pi * float(i) / float(samples)
        z_axis = (
            np.cos(cone_angle) * z0
            + np.sin(cone_angle) * (np.cos(phi) * x0 + np.sin(phi) * y0)
        )
        z_axis /= np.linalg.norm(z_axis)

        x_axis = x0 - np.dot(x0, z_axis) * z_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            x_axis = y0 - np.dot(y0, z_axis) * z_axis
            x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            raise RuntimeError("[cone] Cannot build cone pose frame from degenerate axes.")
        x_axis /= x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        T = T_world_ee.copy()
        T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        T[:3, 3] = position
        poses.append(T)

    return poses


def planned_pose_label(keypoint_idx: int, cone_idx: int | None = None) -> str:
    if cone_idx is None:
        return f"keypoint {keypoint_idx}"
    return f"keypoint {keypoint_idx} cone {cone_idx}"


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
    motion_backend: MotionBackend | None = None,
) -> TrajectoryPlan:
    """
    Generate a trajectory through the keyframes. When T_w_b and cnc_mesh are
    provided, the CNC mesh is added to MoveIt2 as a collision object and each
    segment is planned by MoveIt2. The returned Cartesian poses are only a
    visualization guide; execution uses MoveIt's robot trajectories.
    """
    if len(poses) < 1:
        raise ValueError("Need at least 1 pose to generate a trajectory.")

    if T_w_b is None or cnc_mesh is None:
        expanded_poses = []
        for pose in poses:
            expanded_poses.append(pose)
            expanded_poses.extend(cone_sweep_poses(pose))
        if len(expanded_poses) >= 2:
            cartesian_guide = generate_cartesian_guide_trajectory(
                expanded_poses,
                n_interp=n_interp,
            )
        else:
            cartesian_guide = expanded_poses
        print("[moveit] Missing T_w_b or CNC mesh; using Cartesian guide only.")
        return TrajectoryPlan(
            poses=cartesian_guide,
            planned_with_moveit=False,
            target_poses=expanded_poses,
        )

    backend = motion_backend or MoveItMotionBackend()
    backend.apply_obstacles(cnc_mesh, T_w_b)

    target_world_poses = []
    target_base_pose_groups = []
    for i, target_world_ee in enumerate(poses):
        target_base_ee = mesh_pose_to_base(target_world_ee, T_w_b)
        target_base_ee = target_base_ee.copy()
        # target_base_ee[2, 3] += 0.035

        target_world_poses.append(target_world_ee)
        group = {
            "keypoint_idx": i,
            "keypoint_base_pose": target_base_ee,
            "cone_base_poses": [],
        }

        try:
            cone_world_poses = cone_sweep_poses(target_world_ee)
        except RuntimeError as exc:
            print(f"[moveit] Skipping cone sweep target generation for keypoint {i}: {exc}")
            cone_world_poses = []

        for cone_world_ee in cone_world_poses:
            cone_base_ee = mesh_pose_to_base(cone_world_ee, T_w_b)
            cone_base_ee = cone_base_ee.copy()
            group["cone_base_poses"].append(cone_base_ee)
            target_world_poses.append(cone_world_ee)

        target_base_pose_groups.append(group)

    print(
        f"[moveit] Prepared {len(target_base_pose_groups)} keypoint group(s) "
        f"and {len(target_world_poses)} target pose(s) for online planning."
    )
    if len(target_world_poses) >= 2:
        cartesian_guide = generate_cartesian_guide_trajectory(
            target_world_poses,
            n_interp=n_interp,
        )
    else:
        cartesian_guide = target_world_poses

    return TrajectoryPlan(
        poses=cartesian_guide,
        robot_trajectories=[],
        planned_with_moveit=True,
        target_poses=target_world_poses,
        target_base_pose_groups=target_base_pose_groups,
    )
def check_reachable(
    T_base_ee: np.ndarray,
    robot_reach: float = float(CONFIG["robot"].get("reach_m", 0.900)),
) -> bool:
    """
    Rough reachability check for M0609 (max reach ~900mm).
    Returns False if too far from base or below base plane.
    """
    t = T_base_ee[:3, 3]
    dist = np.linalg.norm(t)
    if dist > robot_reach:
        print(f"[check] UNREACHABLE: distance {dist*1000:.1f}mm > {robot_reach*1000:.0f}mm")
        return False
    if t[2] < -2:
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

    with open("doosan_pose_list.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tx", "ty", "tz", "rz1", "ry", "rz2"])
        writer.writerows(doosan_pose_list)
