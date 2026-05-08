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
    movej,
    movel,
    node,
    posj,
    posx,
    set_robot_mode,
)

_fit_depth_cls = None
_fit_rgb_sift_cls = None


def get_detector_classes():
    global _fit_depth_cls, _fit_rgb_sift_cls

    if _fit_depth_cls is not None and _fit_rgb_sift_cls is not None:
        return _fit_depth_cls, _fit_rgb_sift_cls

    try:
        from FitGMM import FitDepth, FitRGB_SIFT
    except ModuleNotFoundError as exc:
        searched = "\n  - ".join(DETECTOR_DIR_CANDIDATES)
        raise ModuleNotFoundError(
            "Could not import FitGMM. Install/copy FitGMM.py into one of these "
            "detector directories, or set KIMM_CHIPBLOWING_DETECTION_DIR:\n"
            f"  - {searched}"
        ) from exc

    _fit_depth_cls = FitDepth
    _fit_rgb_sift_cls = FitRGB_SIFT
    return _fit_depth_cls, _fit_rgb_sift_cls


def grab_image(img_topic) -> np.ndarray:
    """Subscribe, grab one frame, unsubscribe."""
    latest = {"img": None}
    bridge = CvBridge()

    def compressed_cb(msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        latest["img"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def raw_cb(msg):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        latest["img"] = img

    if img_topic.endswith("/compressed"):
        msg_type = CompressedImage
        callback = compressed_cb
    else:
        msg_type = Image
        callback = raw_cb

    sub = node.create_subscription(msg_type, img_topic, callback, 10)
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
        self.last_depth_diff_mm = None
        self.last_depth_pdf = None
        self.last_rgb_pdf = None

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
        valid_depth_overlap = (
            (current_depth_mm > min_valid_mm)
            & (reference_depth_mm > min_valid_mm)
            & np.isfinite(current_depth_mm)
            & np.isfinite(reference_depth_mm)
        )
        overlap_ratio = float(valid_depth_overlap.sum()) / float(valid_depth_overlap.size)
        print(
            "[detect] Valid depth/color overlap: "
            f"{100.0 * overlap_ratio:.1f}% of image "
            f"({int(valid_depth_overlap.sum())}/{valid_depth_overlap.size} px)"
        )

        diff_mm = np.zeros_like(current_depth_mm, dtype=np.float32)
        diff_mm[valid] = reference_depth_mm[valid] - current_depth_mm[valid]
        self.last_depth_diff_mm = diff_mm.copy()
        print(
            "[detect] Depth delta stats in ROI: "
            f"valid={int(valid.sum())} max={float(np.nanmax(diff_mm)):.2f}mm"
        )

        depth_pdf = self._depth_pdf(diff_mm)
        rgb_pdf = self._rgb_pdf(current_rgb_bgr, current_depth_mm.shape)
        rgb_fallback_pdf = self._rgb_saliency_pdf(current_rgb_bgr, current_depth_mm.shape)
        depth_weight = float(self.cfg.get("depth_weight", 0.5))
        rgb_weight = float(self.cfg.get("rgb_weight", 0.5))
        depth_norm = normalize_pdf(depth_pdf)
        rgb_norm = normalize_pdf(rgb_pdf)
        rgb_fallback_norm = normalize_pdf(rgb_fallback_pdf)
        if bool(self.cfg.get("rgb_only_outside_depth", True)):
            depth_region = valid_depth_overlap & mask
            rgb_only_region = mask & ~valid_depth_overlap
            if bool(self.cfg.get("rgb_saliency_fallback_outside_depth", True)):
                rgb_norm = rgb_norm.copy()
                rgb_norm[rgb_only_region] = np.maximum(
                    rgb_norm[rgb_only_region],
                    rgb_fallback_norm[rgb_only_region],
                )
            merged = np.zeros_like(depth_norm, dtype=np.float32)
            merged = depth_weight * depth_norm + rgb_weight * rgb_norm
            print(
                "[detect] PDF regions: "
                f"depth+rgb={int(depth_region.sum())} px, "
                f"rgb_only={int(rgb_only_region.sum())} px, "
                f"rgb_only_max={float(rgb_norm[rgb_only_region].max(initial=0.0)):.3f}"
            )
        self.last_depth_pdf = depth_norm.copy()
        self.last_rgb_pdf = rgb_norm.copy()
        merged = normalize_pdf(merged)
        if float(merged.sum()) <= 0.0:
            raise RuntimeError("[detect] Chip detector produced an empty PDF.")
        return merged

    def _depth_pdf(self, diff_mm: np.ndarray) -> np.ndarray:
        FitDepth, _ = get_detector_classes()
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
        _, FitRGB_SIFT = get_detector_classes()
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

    def _rgb_saliency_pdf(self, rgb_bgr: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        if rgb_bgr is None:
            return np.zeros(output_shape, dtype=np.float32)
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        saliency = cv2.magnitude(grad_x, grad_y)
        saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=8.0, sigmaY=8.0)
        saliency = cv2.resize(
            saliency,
            (output_shape[1], output_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return normalize_pdf(saliency)


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


def visualize_pdf_debug(
    rgb_bgr: np.ndarray,
    depth_diff_mm: np.ndarray,
    pdf: np.ndarray,
    depth_pdf: np.ndarray | None,
    rgb_pdf: np.ndarray | None,
    sampled_points: list[tuple[float, float]],
    alpha: float = 0.55,
) -> None:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    pdf_norm = normalize_pdf(pdf)
    depth_pdf_norm = normalize_pdf(depth_pdf) if depth_pdf is not None else np.zeros_like(pdf_norm)
    rgb_pdf_norm = normalize_pdf(rgb_pdf) if rgb_pdf is not None else np.zeros_like(pdf_norm)
    heat_bgr = cv2.applyColorMap((pdf_norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    alpha_map = np.clip(pdf_norm[..., None] * float(alpha), 0.0, float(alpha))
    blended = (rgb.astype(np.float32) * (1.0 - alpha_map) + heat_rgb.astype(np.float32) * alpha_map)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    if depth_diff_mm is None:
        depth_diff_mm = np.zeros_like(pdf_norm, dtype=np.float32)
    depth_diff_mm = np.asarray(depth_diff_mm, dtype=np.float32)
    finite_depth = depth_diff_mm[np.isfinite(depth_diff_mm)]
    if finite_depth.size:
        depth_abs_max = float(np.percentile(np.abs(finite_depth), 99.0))
        depth_abs_max = max(depth_abs_max, 1.0)
    else:
        depth_abs_max = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        ("RGB image", rgb, None, None, None),
        ("Depth difference image", depth_diff_mm, "coolwarm", -depth_abs_max, depth_abs_max),
        ("Depth PDF", depth_pdf_norm, "turbo", 0.0, 1.0),
        ("RGB PDF", rgb_pdf_norm, "turbo", 0.0, 1.0),
        ("Merged PDF", pdf_norm, "turbo", 0.0, 1.0),
        ("Merged RGB + PDF", blended, None, None, None),
    ]

    for ax, (title, image, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        if title in ("Merged PDF", "Merged RGB + PDF"):
            for i, (x, y) in enumerate(sampled_points):
                ax.scatter([x], [y], s=70, facecolors="white", edgecolors="black", linewidths=1.5)
                ax.text(
                    float(x) + 8.0,
                    float(y) - 8.0,
                    str(i),
                    color="white",
                    fontsize=10,
                    weight="bold",
                    path_effects=[],
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
                )
        if title in ("Depth difference image", "Depth PDF", "RGB PDF", "Merged PDF"):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if matplotlib.get_backend().lower() == "agg":
        out_path = os.path.join("/tmp", "chip_pdf_debug.png")
        fig.savefig(out_path, dpi=150)
        print(f"[detect] Saved PDF debug visualization to {out_path}")
        plt.close(fig)
    else:
        plt.show()
