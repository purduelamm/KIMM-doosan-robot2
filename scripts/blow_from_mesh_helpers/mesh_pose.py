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
from EEpose_from_mesh.mesh_utils import (
    compute_se3_pose,
    query_mesh_normals,
    visualize_normals,
)
from .detection import grab_image
from .ui import pick_xy_from_camera


class MeshRayMissError(RuntimeError):
    """Raised when a camera ray does not intersect the CNC mesh."""


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
        raise MeshRayMissError(f"No mesh surface hit for pixel ({pu:.1f}, {pv:.1f})")

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


def get_depth_from_world_xy_plane(
    pu: float,
    pv: float,
    T_w_cm: np.ndarray,
    world_z_m: float,
) -> float:
    """Return camera depth where a pixel ray meets ``world z = world_z_m``."""
    ray_camera = np.linalg.inv(camera_K) @ np.array([pu, pv, 1.0])
    ray_world = T_w_cm[:3, :3] @ ray_camera
    ray_origin_world = T_w_cm[:3, 3]

    if abs(ray_world[2]) < 1e-12:
        raise RuntimeError(
            f"Pixel ({pu:.1f}, {pv:.1f}) ray is parallel to the fallback XY plane."
        )

    distance_along_ray = (world_z_m - ray_origin_world[2]) / ray_world[2]
    if distance_along_ray <= 0.0:
        raise RuntimeError(
            f"Fallback XY plane z={world_z_m:.6f} m is behind the camera for "
            f"pixel ({pu:.1f}, {pv:.1f})."
        )

    hit_world = ray_origin_world + distance_along_ray * ray_world
    hit_world[2] = world_z_m
    hit_camera = np.linalg.inv(T_w_cm) @ np.append(hit_world, 1.0)
    return float(hit_camera[2])


def compute_keyframe_from_pixel(
    pu: float,
    pv: float,
    T_w_cm: np.ndarray,
    cnc_mesh: trimesh.Trimesh,
) -> np.ndarray:
    mesh_cfg = CONFIG["mesh"]
    used_fallback_plane = False
    try:
        z_cam = get_depth_from_mesh(pu, pv, T_w_cm, cnc_mesh)
    except MeshRayMissError:
        fallback_cfg = mesh_cfg.get("ray_miss_fallback_plane", {})
        if "world_z_m" not in fallback_cfg:
            raise ValueError(
                "mesh.ray_miss_fallback_plane.world_z_m must be set in the YAML config."
            )
        fallback_world_z = float(fallback_cfg["world_z_m"])
        z_cam = get_depth_from_world_xy_plane(
            pu, pv, T_w_cm, fallback_world_z
        )
        used_fallback_plane = True
        print(
            f"[raycast] Pixel ({pu:.1f}, {pv:.1f}) missed the CNC mesh; "
            f"using world XY plane z={fallback_world_z:.6f} m."
        )

    query_world = pixel_to_world(pu, pv, z_cam, T_w_cm)
    print(f"depth: {z_cam}  world_z: {query_world[2]}")
    print(f"[main] query_world: {query_world}")

    pose_mode = mesh_cfg.get("pose_mode", "mesh_normal")
    if pose_mode == "fixed_blower_z_axis":
        return compute_fixed_blower_z_axis_keyframe(query_world, mesh_cfg)
    if pose_mode != "mesh_normal":
        raise ValueError(
            f"Unknown mesh.pose_mode '{pose_mode}'. "
            "Expected 'mesh_normal' or 'fixed_blower_z_axis'."
        )

    query_mesh_x, query_mesh_y = world_to_mesh(query_world)
    query_mesh_z = query_world[2] * MESH_DIMENSION
    print(f"[main] query_mesh_x={query_mesh_x:.2f}  query_mesh_y={query_mesh_y:.2f}")

    if used_fallback_plane:
        plane_normal = np.array([0.0, 0.0, 1.0])
        if T_w_cm[2, 3] < query_world[2]:
            plane_normal *= -1.0
        surface_point_mesh = query_world * MESH_DIMENSION
        normals = {
            "surface_point": surface_point_mesh,
            "roi_points": surface_point_mesh.reshape(1, 3),
            "unique_normals": plane_normal.reshape(1, 3),
        }
    else:
        normals = query_mesh_normals(
            cnc_mesh,
            query_mesh_x,
            query_mesh_y,
            mesh_cfg["normal_query_radius"],
            total_points=mesh_cfg["normal_sample_count"],
            z=query_mesh_z,
        )
    cam_pos_mesh = T_w_cm[:3, 3] * MESH_DIMENSION
    surface_to_camera = cam_pos_mesh - normals["surface_point"]
    surface_to_camera_norm = np.linalg.norm(surface_to_camera)
    if surface_to_camera_norm > 1e-12:
        normals["view_direction"] = surface_to_camera / surface_to_camera_norm

    offset_m = float(mesh_cfg["surface_offset_m"])
    pose = compute_se3_pose(normals, offset_m * MESH_DIMENSION)
    normals["pose"] = pose

    if CONFIG.get("visualization", {}).get("show_normals", True):
        visualize_normals(cnc_mesh, normals, normal_length=100)

    T_world_ee = pose["T"].copy()
    T_world_ee[:3, 3] /= MESH_DIMENSION
    return T_world_ee


def compute_fixed_blower_z_axis_keyframe(
    surface_point_world: np.ndarray,
    mesh_cfg: dict,
) -> np.ndarray:
    fixed_cfg = mesh_cfg.get("fixed_blower_z_axis", {})
    angle_deg = float(fixed_cfg.get("angle_from_world_y_deg", 45.0))
    distance_m = float(fixed_cfg.get("distance_m", mesh_cfg.get("surface_offset_m", 0.2)))

    theta = np.deg2rad(angle_deg)
    offset_dir = np.array([0.0, np.cos(theta), np.sin(theta)], dtype=np.float64)
    offset_dir /= np.linalg.norm(offset_dir)
    z_axis = -offset_dir

    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    T_world_ee = np.eye(4)
    T_world_ee[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T_world_ee[:3, 3] = surface_point_world + offset_dir * distance_m

    print(
        "[pose] fixed_blower_z_axis: "
        f"angle_from_world_y={angle_deg:.2f}deg  distance={distance_m:.3f}m  "
        f"z_axis={np.round(z_axis, 6)}"
    )
    return T_world_ee


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
