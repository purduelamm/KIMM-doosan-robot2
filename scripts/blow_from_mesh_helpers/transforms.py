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


def lookup_link_transforms(
    source_frame: str,
    target_frames: list[str],
    timeout_sec: float = 5.0,
) -> dict[str, np.ndarray]:
    """Return source-from-target transforms for all TF frames that are available."""
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    print("[tf] Warming up tf buffer for debug visualization...")
    t0 = time.time()
    while time.time() - t0 < 1.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    transforms = {}
    if source_frame in target_frames:
        transforms[source_frame] = np.eye(4)
    deadline = time.time() + timeout_sec
    pending = set(target_frames) - {source_frame}
    while pending and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        for frame in list(pending):
            try:
                t = tf_buffer.lookup_transform(source_frame, frame, rclpy.time.Time())
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue

            trans = t.transform.translation
            rot = t.transform.rotation
            T = np.eye(4)
            T[:3, :3] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
            T[:3, 3] = [trans.x, trans.y, trans.z]
            transforms[frame] = T
            pending.remove(frame)

    tf_listener.unregister()
    del tf_buffer

    if pending:
        print(f"[debug-vis] Missing TF frame(s): {sorted(pending)}")
    return transforms


def get_current_camera_transform(T_w_b: np.ndarray) -> np.ndarray:
    R_b_6, t_b_6 = get_base_to_link6()
    T_b_6 = make_SE3(R_b_6, t_b_6)
    T_6_cm = make_SE3(
        (L6_TO_CAM_R * GAZ_TO_OPT_R).as_matrix(), L6_TO_CAM_T
    )  # link 6 to camera model frame (z-forward / y-downward)
    print("T_6_cm, ", T_6_cm)
    return T_w_b @ T_b_6 @ T_6_cm
