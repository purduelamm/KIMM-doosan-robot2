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

import os
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
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
from sensor_msgs.msg import CompressedImage, Image
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from tf2_ros import Buffer, TransformListener
import tf2_ros
from visualization_msgs.msg import Marker
import rclpy.duration
from cv_bridge import CvBridge

from EEpose_from_mesh.mesh_utils import (
    query_mesh_normals,
    visualize_normals,
    compute_se3_pose,
)
import trimesh
import pandas as pd
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "blow_from_mesh.yaml")
DETECTOR_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "KIMM_chipblowing_detection")
if DETECTOR_DIR not in sys.path:
    sys.path.append(DETECTOR_DIR)

from FitGMM import FitDepth, FitRGB_SIFT

# Edit this list for non-interactive runs. The optional pick_xy_from_camera()
# helper remains available for collecting points manually.
QUERY_PIXELS = [
    # (640.0, 360.0),
]


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rotation_from_config(cfg: dict) -> R:
    if "matrix" in cfg:
        return R.from_matrix(np.array(cfg["matrix"], dtype=float))
    if "euler_xyz" in cfg:
        return R.from_euler("xyz", cfg["euler_xyz"], degrees=cfg.get("degrees", False))
    raise ValueError("Rotation config must contain 'matrix' or 'euler_xyz'.")


def translation_from_config(cfg: dict) -> np.ndarray:
    return np.array(cfg.get("translation", [0.0, 0.0, 0.0]), dtype=float)


CONFIG = load_config()

rclpy.init()
ROBOT_ID = CONFIG["robot"]["id"]
ROBOT_MODEL = CONFIG["robot"]["model"]
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
node = rclpy.create_node(CONFIG["robot"].get("node_name", "coverage_path"), namespace=ROBOT_ID)
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

TRANSFORMS = CONFIG["transforms"]
GAZ_TO_OPT_R = rotation_from_config(TRANSFORMS["gazebo_to_optical"])
GAZ_TO_OPT_T = translation_from_config(TRANSFORMS["gazebo_to_optical"])
WLD_TO_BASE_R = rotation_from_config(TRANSFORMS["world_to_base"])
WLD_TO_BASE_T = translation_from_config(TRANSFORMS["world_to_base"])
L6_TO_CAM_R = rotation_from_config(TRANSFORMS["link_to_camera"])
L6_TO_CAM_T = translation_from_config(TRANSFORMS["link_to_camera"])
WLD_TO_CNC = np.array(CONFIG["mesh"].get("world_to_cnc", [0.0, 0.0, 0.0]), dtype=float)

MESH_DIMENSION = float(CONFIG["mesh"]["dimension"])
CAMERA_CFG = CONFIG["camera"]
fx = float(CAMERA_CFG["intrinsics"]["fx"])
fy = float(CAMERA_CFG["intrinsics"].get("fy", fx))
cx_k = float(CAMERA_CFG["intrinsics"]["cx"])
cy_k = float(CAMERA_CFG["intrinsics"]["cy"])
camera_K = np.array([[fx, 0, cx_k], [0, fy, cy_k], [0, 0, 1]])

CNC_mesh_path = CONFIG["mesh"]["path"]
if not os.path.isabs(CNC_mesh_path):
    CNC_mesh_path = os.path.join(SCRIPT_DIR, CNC_mesh_path)

IMG_TOPIC = CAMERA_CFG["image_topic"]
INIT_POSX = CONFIG["robot"]["init_posx"]
TF_SOURCE_FRAME = CONFIG["robot"]["frames"]["tf_source"]
TF_TARGET_FRAME = CONFIG["robot"]["frames"]["tf_target"]
MOVEIT_CFG = CONFIG["moveit"]
MOVEIT_GROUP = MOVEIT_CFG["group"]
MOVEIT_EE_LINK = MOVEIT_CFG["ee_link"]
MOVEIT_BASE_FRAME = MOVEIT_CFG["base_frame"]
MOVEIT_PLANNING_SERVICE = MOVEIT_CFG["planning_service"]
MOVEIT_MOVE_ACTION = MOVEIT_CFG["move_action"]
MOVEIT_SCENE_SERVICE = MOVEIT_CFG["scene_service"]
MOVEIT_SCENE_TOPIC = MOVEIT_CFG["scene_topic"]
MOVEIT_EXECUTE_ACTION = MOVEIT_CFG["execute_action"]
MOVEIT_PLANNING_TIME = float(MOVEIT_CFG["planning_time"])
MOVEIT_PLANNING_ATTEMPTS = int(MOVEIT_CFG["planning_attempts"])
MOVEIT_POS_TOLERANCE = float(MOVEIT_CFG["position_tolerance"])
MOVEIT_ORI_TOLERANCE = float(MOVEIT_CFG["orientation_tolerance"])
MOVEIT_VELOCITY_SCALING = float(MOVEIT_CFG.get("velocity_scaling", 0.2))
MOVEIT_ACCELERATION_SCALING = float(MOVEIT_CFG.get("acceleration_scaling", 0.2))
MOVEIT_JOINT_NAMES = MOVEIT_CFG.get(
    "joint_names",
    ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
)
MOVEIT_JOINT_TOLERANCE = float(MOVEIT_CFG.get("joint_tolerance", 0.001))
CNC_COLLISION_OBJECT_ID = MOVEIT_CFG["collision_object_id"]
ALLOW_CNC_COLLISION_LINKS = MOVEIT_CFG["allow_cnc_collision_links"]
DISABLED_SELF_COLLISION_PAIRS = [tuple(pair) for pair in MOVEIT_CFG["disabled_self_collision_pairs"]]
ALL_ZERO_JOINTS = CONFIG["robot"].get("all_zero_joints", [0.0] * len(MOVEIT_JOINT_NAMES))


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


def depth_image_to_mm(depth_image: np.ndarray, unit: str = "auto") -> np.ndarray:
    depth = depth_image.astype(np.float32)
    if unit == "mm":
        return depth
    if unit == "m":
        return depth * 1000.0
    if unit != "auto":
        raise ValueError(f"Unsupported depth_unit '{unit}'. Use auto, mm, or m.")

    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return depth
    # RealSense float depth is usually meters; uint16 depth is usually millimeters.
    return depth * 1000.0 if float(np.nanpercentile(finite, 95)) < 20.0 else depth


class RGBDFrameGrabber:
    """Stores latest RGB-D frames from RealSense topics configured in YAML."""

    def __init__(self, rgb_topic: str, depth_topic: str, depth_unit: str = "auto"):
        self.bridge = CvBridge()
        self.depth_unit = depth_unit
        self.latest_rgb_bgr = None
        self.latest_depth_mm = None
        self._depth_samples = []
        self._collect_depth = False
        self.rgb_sub = node.create_subscription(Image, rgb_topic, self._rgb_cb, 10)
        self.depth_sub = node.create_subscription(Image, depth_topic, self._depth_cb, 10)
        print(f"[rgbd] Subscribed RGB topic: {rgb_topic}")
        print(f"[rgbd] Subscribed depth topic: {depth_topic}")

    def _rgb_cb(self, msg):
        self.latest_rgb_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _depth_cb(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_mm = depth_image_to_mm(depth, self.depth_unit)
        self.latest_depth_mm = depth_mm
        if self._collect_depth:
            self._depth_samples.append(depth_mm.copy())

    def wait_for_frames(self, timeout_sec: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        print("[rgbd] Waiting for RGB-D frames...")
        t0 = time.time()
        while self.latest_rgb_bgr is None or self.latest_depth_mm is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - t0 > timeout_sec:
                raise RuntimeError("[rgbd] Timed out waiting for RGB-D frames.")
        return self.latest_rgb_bgr.copy(), self.latest_depth_mm.copy()

    def average_depth(self, frames_to_average: int, timeout_sec: float = 10.0) -> np.ndarray:
        frames_to_average = max(1, int(frames_to_average))
        self._depth_samples = []
        self._collect_depth = True
        print(f"[rgbd] Collecting {frames_to_average} depth frame(s) for average...")
        t0 = time.time()
        while len(self._depth_samples) < frames_to_average:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - t0 > timeout_sec:
                self._collect_depth = False
                raise RuntimeError(
                    f"[rgbd] Timed out after collecting "
                    f"{len(self._depth_samples)}/{frames_to_average} depth frames."
                )
        self._collect_depth = False
        stacked = np.stack(self._depth_samples[:frames_to_average], axis=0)
        averaged = np.nanmean(stacked, axis=0).astype(np.float32)
        print(
            "[rgbd] Averaged depth stats: "
            f"min={np.nanmin(averaged):.2f}mm max={np.nanmax(averaged):.2f}mm"
        )
        return averaged


def normalize_pdf(pdf: np.ndarray) -> np.ndarray:
    pdf = np.asarray(pdf, dtype=np.float32)
    pdf = np.where(np.isfinite(pdf) & (pdf > 0.0), pdf, 0.0)
    if float(pdf.max(initial=0.0)) <= 0.0:
        return np.zeros_like(pdf, dtype=np.float32)
    return (pdf / float(pdf.max())).astype(np.float32)


def roi_mask(shape: tuple[int, int], cfg: dict) -> np.ndarray:
    height, width = shape
    roi = cfg.get("mask_roi", {})
    x_min = int(roi.get("x_min", 0))
    y_min = int(roi.get("y_min", 0))
    x_max_cfg = roi.get("x_max", width)
    y_max_cfg = roi.get("y_max", height)
    x_max = width if x_max_cfg is None or int(x_max_cfg) < 0 else int(x_max_cfg)
    y_max = height if y_max_cfg is None or int(y_max_cfg) < 0 else int(y_max_cfg)
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y_min):min(height, y_max), max(0, x_min):min(width, x_max)] = True
    return mask


class RGBDChipDetector:
    """Depth-difference + RGB SIFT chip probability detector."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def detect(
        self,
        reference_depth_mm: np.ndarray,
        current_depth_mm: np.ndarray,
        current_rgb_bgr: np.ndarray,
    ) -> np.ndarray:
        if reference_depth_mm.shape != current_depth_mm.shape:
            raise RuntimeError(
                f"[detect] Depth shape mismatch: reference={reference_depth_mm.shape}, "
                f"current={current_depth_mm.shape}"
            )

        min_valid_mm = float(self.cfg.get("min_valid_mm", 10.0))
        mask = roi_mask(current_depth_mm.shape, self.cfg)
        valid = (
            mask
            & (current_depth_mm > min_valid_mm)
            & (reference_depth_mm > min_valid_mm)
            & np.isfinite(current_depth_mm)
            & np.isfinite(reference_depth_mm)
        )

        diff_mm = np.zeros_like(current_depth_mm, dtype=np.float32)
        diff_mm[valid] = reference_depth_mm[valid] - current_depth_mm[valid]
        print(
            "[detect] Depth delta stats in ROI: "
            f"valid={int(valid.sum())} max={float(np.nanmax(diff_mm)):.2f}mm"
        )

        depth_pdf = self._depth_pdf(diff_mm)
        rgb_pdf = self._rgb_pdf(current_rgb_bgr, current_depth_mm.shape)
        merged = (
            float(self.cfg.get("depth_weight", 0.5)) * normalize_pdf(depth_pdf)
            + float(self.cfg.get("rgb_weight", 0.5)) * normalize_pdf(rgb_pdf)
        )
        merged *= mask.astype(np.float32)
        merged = normalize_pdf(merged)
        if float(merged.sum()) <= 0.0:
            raise RuntimeError("[detect] Chip detector produced an empty PDF.")
        return merged

    def _depth_pdf(self, diff_mm: np.ndarray) -> np.ndarray:
        scale = float(self.cfg.get("depth_fit_scale", 0.5))
        if scale != 1.0:
            fit_map = cv2.resize(diff_mm, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        else:
            fit_map = diff_mm

        thresh_min = float(self.cfg.get("thresh_min_mm", 1.0))
        thresh_max = float(self.cfg.get("thresh_max_mm", 50.0))
        candidates = (np.abs(fit_map) >= thresh_min) & (np.abs(fit_map) < thresh_max)
        if int(candidates.sum()) == 0:
            print("[detect] Depth PDF has no thresholded candidates; using RGB PDF only.")
            return np.zeros(diff_mm.shape, dtype=np.float32)

        depth_gmm = FitDepth(fit_map)
        depth_gmm.fit_depth(
            thresh_min_mm=thresh_min,
            thresh_max_mm=thresh_max,
            kmax=int(self.cfg.get("kmax", 3)),
            random_state=int(self.cfg.get("random_state", 0)),
            resample_cap=int(self.cfg.get("resample_cap", 1000)),
        )
        pdf = depth_gmm.gmm_to_pdf(
            downsample_factor=int(self.cfg.get("depth_pdf_downsample_factor", 2))
        )
        return cv2.resize(pdf, (diff_mm.shape[1], diff_mm.shape[0]), interpolation=cv2.INTER_LINEAR)

    def _rgb_pdf(self, rgb_bgr: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        if rgb_bgr is None:
            return np.zeros(output_shape, dtype=np.float32)

        scale = float(self.cfg.get("rgb_fit_scale", 0.5))
        if scale != 1.0:
            fit_img = cv2.resize(rgb_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            fit_img = rgb_bgr

        rgb_gmm = FitRGB_SIFT(fit_img)
        gmm = rgb_gmm.fit_rgb(
            num_features=int(self.cfg.get("rgb_num_features", 500)),
            sift_contrast=float(self.cfg.get("rgb_sift_contrast", 0.06)),
            sift_edge=float(self.cfg.get("rgb_sift_edge", 6)),
            n_components=int(self.cfg.get("rgb_n_components", 5)),
        )
        if gmm is None:
            print("[detect] RGB SIFT found no keypoints; using depth PDF only.")
            return np.zeros(output_shape, dtype=np.float32)

        pdf = rgb_gmm.rgb_gmm_to_pdf(
            include_outliers=bool(self.cfg.get("rgb_include_outliers", True)),
            outlier_radius=int(self.cfg.get("rgb_outlier_radius", 15)),
            outlier_probability=float(self.cfg.get("rgb_outlier_probability", 0.8)),
            downsample_factor=int(self.cfg.get("rgb_pdf_downsample_factor", 2)),
        )
        return cv2.resize(pdf, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)


def sample_pixels_from_pdf(pdf: np.ndarray, sample_count: int, random_seed: int | None = None) -> list[tuple[float, float]]:
    weights = np.asarray(pdf, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("[detect] Cannot sample pixels from an empty PDF.")

    flat = weights.ravel() / total
    nonzero = int(np.count_nonzero(flat))
    count = max(1, int(sample_count))
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(flat.size, size=count, replace=count > nonzero, p=flat)
    ys, xs = np.unravel_index(idx, weights.shape)
    points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    print(f"[detect] Sampled {len(points)} chip target pixel(s): {points}")
    return points


def get_base_to_link6() -> tuple[np.ndarray, np.ndarray]:
    """Returns (R_source_target [3x3], t_source_target [3])"""
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    print("[tf] Warming up tf buffer...")
    t0 = time.time()
    while time.time() - t0 < 1.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f"[tf] Waiting for {TF_SOURCE_FRAME} -> {TF_TARGET_FRAME} transform...")
    t0 = time.time()
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            t = tf_buffer.lookup_transform(TF_SOURCE_FRAME, TF_TARGET_FRAME, rclpy.time.Time())
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException):
            if time.time() - t0 > 5.0:
                raise RuntimeError(
                    f"[tf] Timed out waiting for {TF_SOURCE_FRAME} -> {TF_TARGET_FRAME}."
                )
        except tf2_ros.ExtrapolationException as e:
            raise RuntimeError(f"[tf] Extrapolation error: {e}")

    tf_listener.unregister()
    del tf_buffer

    trans = t.transform.translation
    rot = t.transform.rotation
    translation = np.array([trans.x, trans.y, trans.z])
    rotation = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
    print(f"[tf] {TF_SOURCE_FRAME} -> {TF_TARGET_FRAME}\n  t={translation}\n  R=\n{rotation}")
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

    def move_to_init(self, init_pose: list[float], cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray):
        target_base_ee = doosan_posx_to_base_se3(init_pose)
        plan = self.plan_to_pose(target_base_ee, start_state=None, segment_idx=0)
        self.execute_plan(plan)

    def move_to_joints(self, joints: list[float]) -> None:
        plan = self.plan_to_joints(joints, start_state=None, segment_idx=0)
        self.execute_plan(plan)


class MoveItMotionBackend(MotionBackend):
    """Default obstacle-aware motion backend."""

    def __init__(self):
        self.planner_kind = None
        self.planner_client = None

    def apply_obstacles(self, cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray) -> None:
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
            execute_moveit_trajectory(plan)
            return
        execute_moveit_trajectory(
            TrajectoryPlan(poses=[], robot_trajectories=[plan], planned_with_moveit=True)
        )

    def move_to_init(self, init_pose: list[float], cnc_mesh: trimesh.Trimesh, T_w_b: np.ndarray):
        print("[init] Planning collision-free move to initial posx with MoveIt2...")
        self.apply_obstacles(cnc_mesh, T_w_b)
        super().move_to_init(init_pose, cnc_mesh, T_w_b)
        time.sleep(1)
        print("[init] Moved to initial point.")
        print("current: ", get_current_posx())

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

    def move_to_joints(self, joints: list[float]) -> None:
        movej(posj(*joints), vel=60, acc=60)


def create_motion_backend() -> MotionBackend:
    backend_name = CONFIG.get("motion_backend", "moveit")
    if backend_name == "moveit":
        return MoveItMotionBackend()
    if backend_name == "doosan_direct":
        return DoosanDirectMotionBackend()
    raise ValueError(f"Unsupported motion_backend '{backend_name}'.")


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


def compute_keyframe_from_pixel(
    pu: float,
    pv: float,
    T_w_cm: np.ndarray,
    cnc_mesh: trimesh.Trimesh,
) -> np.ndarray:
    z_cam = get_depth_from_mesh(pu, pv, T_w_cm, cnc_mesh)
    print("depth: ", z_cam, " world_z: ", T_w_cm[2, 3] - z_cam)

    query_world = pixel_to_world(pu, pv, z_cam, T_w_cm)
    print(f"[main] query_world: {query_world}")

    query_mesh_x, query_mesh_y = world_to_mesh(query_world)
    query_mesh_z = query_world[2] * MESH_DIMENSION
    print(f"[main] query_mesh_x={query_mesh_x:.2f}  query_mesh_y={query_mesh_y:.2f}")

    mesh_cfg = CONFIG["mesh"]
    normals = query_mesh_normals(
        cnc_mesh,
        query_mesh_x,
        query_mesh_y,
        mesh_cfg["normal_query_radius"],
        total_points=mesh_cfg["normal_sample_count"],
        z=query_mesh_z,
    )

    offset_m = float(mesh_cfg["surface_offset_m"])
    pose = compute_se3_pose(normals, offset_m * MESH_DIMENSION)
    normals["pose"] = pose

    if CONFIG.get("visualization", {}).get("show_normals", True):
        visualize_normals(cnc_mesh, normals, normal_length=100)

    T_world_ee = pose["T"].copy()
    T_world_ee[:3, 3] /= MESH_DIMENSION
    return T_world_ee


def get_current_camera_transform(T_w_b: np.ndarray) -> np.ndarray:
    R_b_6, t_b_6 = get_base_to_link6()
    T_b_6 = make_SE3(R_b_6, t_b_6)
    T_6_cm = make_SE3(
        (L6_TO_CAM_R * GAZ_TO_OPT_R).as_matrix(), L6_TO_CAM_T
    )  # link 6 to camera model frame (z-forward / y-downward)
    print("T_6_cm, ", T_6_cm)
    return T_w_b @ T_b_6 @ T_6_cm


def get_query_pixels(snapshot: np.ndarray | None = None) -> list[tuple[float, float]]:
    if QUERY_PIXELS:
        return [(float(pu), float(pv)) for pu, pv in QUERY_PIXELS]

    point_cfg = CONFIG.get("point_input", {})
    if not point_cfg.get("use_interactive_picker_when_query_pixels_empty", False):
        raise RuntimeError(
            "QUERY_PIXELS is empty. Add camera pixel points near the top of "
            "blow_from_mesh.py, for example QUERY_PIXELS = [(640.0, 360.0)]."
        )

    if snapshot is None:
        snapshot = grab_image(IMG_TOPIC)

    points = []
    while True:
        points.append(pick_xy_from_camera(snapshot))
        ans = input("[loop] Add another point? [y/N]: ").strip().lower()
        if ans != "y":
            break
    return points


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
    motion_backend: MotionBackend | None = None,
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

    backend = motion_backend or MoveItMotionBackend()
    backend.apply_obstacles(cnc_mesh, T_w_b)

    robot_trajectories = []
    start_state = None
    for i, target_world_ee in enumerate(poses):
        target_base_ee = mesh_pose_to_base(target_world_ee, T_w_b)
        target_base_ee = target_base_ee.copy()
        target_base_ee[2, 3] += 0.035

        traj = backend.plan_to_pose(target_base_ee, start_state=start_state, segment_idx=i)
        robot_trajectories.append(traj)
        if hasattr(traj, "joint_trajectory"):
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
    motion_backend = create_motion_backend()
    detector_cfg = CONFIG.get("chip_detector", {})
    execution_cfg = CONFIG.get("execution", {})

    # wake up robot arm and move to initial pose
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    motion_backend.move_to_init(INIT_POSX, CNC_mesh, T_w_b)
    time.sleep(3)

    rgbd = None
    if detector_cfg.get("enabled", False):
        rgbd = RGBDFrameGrabber(
            CAMERA_CFG["rgb_topic"],
            CAMERA_CFG["depth_topic"],
            depth_unit=CAMERA_CFG.get("depth_unit", "auto"),
        )
        rgbd.wait_for_frames(timeout_sec=float(detector_cfg.get("frame_timeout_sec", 10.0)))

    if detector_cfg.get("enabled", False):
        if rgbd is None:
            raise RuntimeError("[detect] RGB-D frame grabber was not initialized.")

        reference_depth = rgbd.average_depth(
            frames_to_average=int(detector_cfg.get("frames_to_average", 30)),
            timeout_sec=float(detector_cfg.get("reference_timeout_sec", 15.0)),
        )
        print("[detect] Saved reference depth image at initial pose.")

        motion_backend.move_to_joints(ALL_ZERO_JOINTS)
        input("[detect] Robot is at all-zero pose. Prepare chips, then press Enter to detect...")

        current_rgb_bgr, _ = rgbd.wait_for_frames(
            timeout_sec=float(detector_cfg.get("frame_timeout_sec", 10.0))
        )
        current_depth = rgbd.average_depth(
            frames_to_average=int(detector_cfg.get("frames_to_average", 30)),
            timeout_sec=float(detector_cfg.get("current_timeout_sec", 15.0)),
        )

        detector = RGBDChipDetector(detector_cfg)
        chip_pdf = detector.detect(reference_depth, current_depth, current_rgb_bgr)
        query_pixels = sample_pixels_from_pdf(
            chip_pdf,
            sample_count=int(detector_cfg.get("sample_count", 5)),
            random_seed=detector_cfg.get("sample_random_seed", None),
        )
        T_w_cm = get_current_camera_transform(T_w_b)
    else:
        T_w_cm = get_current_camera_transform(T_w_b)
        snapshot = None
        if not QUERY_PIXELS and CONFIG.get("point_input", {}).get(
            "use_interactive_picker_when_query_pixels_empty", False
        ):
            snapshot = grab_image(IMG_TOPIC)
        query_pixels = get_query_pixels(snapshot)

    keyframes = []
    for point_idx, (pu, pv) in enumerate(query_pixels):
        print(f"\n[loop] === Point {point_idx} ===")
        print(f"[main] Picked pixel: ({pu:.1f}, {pv:.1f})")
        keyframes.append(compute_keyframe_from_pixel(pu, pv, T_w_cm, CNC_mesh))

    print(f"\n[traj] Collected {len(keyframes)} keyframe(s).")

    # ── generate smooth trajectory ────────────────────────────────────────────
    keyframes = consistent_rotations(keyframes)

    if len(keyframes) >= 2:
        trajectory = generate_smooth_trajectory(
            keyframes,
            n_interp=50,
            T_w_b=T_w_b,
            cnc_mesh=CNC_mesh,
            motion_backend=motion_backend,
        )
        if trajectory.planned_with_moveit:
            print(
                f"[traj] MoveIt planned {len(trajectory.robot_trajectories)} "
                f"segment(s); {len(trajectory)} poses kept for visualization."
            )
        else:
            print(f"[traj] Generated {len(trajectory)} interpolated poses.")
    else:
        target_base_ee = mesh_pose_to_base(keyframes[0], T_w_b)
        target_base_ee = target_base_ee.copy()
        target_base_ee[2, 3] += 0.035
        trajectory = TrajectoryPlan(
            poses=keyframes,
            robot_trajectories=[motion_backend.plan_to_pose(target_base_ee)],
            planned_with_moveit=True,
        )
        print("[traj] Single keyframe — planned direct segment.")

    # ── visualise ─────────────────────────────────────────────────────────────
    if CONFIG.get("visualization", {}).get("show_trajectory", True):
        visualize_trajectory(
            CNC_mesh,
            keyframes,
            trajectory,
            axis_len=float(CONFIG.get("visualization", {}).get("trajectory_axis_len", 0.2)),
        )

    # ── execute ───────────────────────────────────────────────────────────────
    if execution_cfg.get("execute_gazebo_first", True):
        print("[exec] Executing planned trajectory on the current Gazebo backend...")
        motion_backend.execute_plan(trajectory)

    if execution_cfg.get("confirm_real_execution", True):
        confirm = input(
            "[exec] Press y to execute the same planned trajectory on the real robot. [y/N]: "
        ).strip().lower()
        if confirm == "y":
            motion_backend.execute_plan(trajectory)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
