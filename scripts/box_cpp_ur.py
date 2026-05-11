"""Coverage path generation and execution for a UR arm from YAML box settings."""

import os
import sys
import time

import numpy as np
import rclpy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UR_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "blow_from_mesh_ur.yaml")

# Force the shared helper stack to load the UR config before it imports config.py.
os.environ["BLOW_FROM_MESH_CONFIG"] = UR_CONFIG_PATH
os.environ["BLOW_FROM_MESH_ROBOT"] = "ur"

from blow_from_mesh_helpers import ros_context
from blow_from_mesh_helpers.config import CONFIG, TF_SOURCE_FRAME, TF_TARGET_FRAME
from blow_from_mesh_helpers.motion import create_motion_backend
from blow_from_mesh_helpers.transforms import get_base_to_link6


def get_box_config() -> dict:
    box_cfg = CONFIG.get("robot", {}).get("box_cpp", {})
    if not box_cfg:
        raise ValueError("robot.box_cpp is required in blow_from_mesh_ur.yaml.")
    if "initial_garget" not in box_cfg:
        raise ValueError("robot.box_cpp.initial_garget is required in blow_from_mesh_ur.yaml.")
    if "up_configuration" not in box_cfg:
        raise ValueError("robot.box_cpp.up_configuration is required in blow_from_mesh_ur.yaml.")
    for key in ("box_width_m", "box_height_m", "line_spacing_m"):
        if key not in box_cfg:
            raise ValueError(f"robot.box_cpp.{key} is required in blow_from_mesh_ur.yaml.")
    return box_cfg


def make_pose(rotation: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, y, z]
    return pose


def generate_box_path(center_pose: np.ndarray, box_cfg: dict) -> list[np.ndarray]:
    box_width = float(box_cfg["box_width_m"])
    box_height = float(box_cfg["box_height_m"])
    line_spacing = float(box_cfg["line_spacing_m"])
    if box_width <= 0.0 or box_height <= 0.0 or line_spacing <= 0.0:
        raise ValueError("box_width_m, box_height_m, and line_spacing_m must be positive.")

    rotation = center_pose[:3, :3]
    center = center_pose[:3, 3]
    start_x = float(center[0] - box_width / 2.0)
    start_y = float(center[1] - box_height / 2.0)
    z = float(center[2])
    num_lines = int(box_height / line_spacing) + 1

    path = []
    for i in range(num_lines):
        y = start_y + i * line_spacing
        if i % 2 == 0:
            path.append(make_pose(rotation, start_x, y, z))
            path.append(make_pose(rotation, start_x + box_width, y, z))
        else:
            path.append(make_pose(rotation, start_x + box_width, y, z))
            path.append(make_pose(rotation, start_x, y, z))

    print(f"Coverage path: {box_width:.3f}x{box_height:.3f}m area centered on home")
    print(f"Line spacing: {line_spacing:.3f}m, Total lines: {num_lines}")
    print(
        "Box bounds: "
        f"X[{start_x:.3f}, {start_x + box_width:.3f}], "
        f"Y[{start_y:.3f}, {start_y + box_height:.3f}]"
    )
    return path


def current_tool_pose() -> np.ndarray:
    rotation, translation = get_base_to_link6()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    print(
        f"Center position from TF {TF_SOURCE_FRAME}->{TF_TARGET_FRAME}: "
        f"x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}"
    )
    return pose


def execute_waypoints(motion_backend, waypoints: list[np.ndarray]) -> None:
    ros_context.air_node.tool_airgun(True)
    time.sleep(1.0)
    try:
        for i, waypoint in enumerate(waypoints):
            t = waypoint[:3, 3]
            print(
                f"Waypoint {i + 1}/{len(waypoints)}: "
                f"x={t[0]:.3f}, y={t[1]:.3f}, z={t[2]:.3f}"
            )
            plan = motion_backend.plan_to_pose(waypoint, start_state=None, segment_idx=i + 1)
            motion_backend.execute_plan(plan)
    finally:
        ros_context.air_node.tool_airgun(False)
        time.sleep(1.0)


def main(args=None):
    ros_context.init_ros()
    box_cfg = get_box_config()
    motion_backend = create_motion_backend()

    print("[init] Moving to robot.box_cpp.initial_garget...")
    motion_backend.move_to_joints(box_cfg["initial_garget"])
    time.sleep(float(box_cfg.get("settle_sec", 1.0)))

    center_pose = current_tool_pose()
    path = generate_box_path(center_pose, box_cfg)

    execute_waypoints(motion_backend, path)

    print("[finish] Moving to robot.box_cpp.up_configuration...")
    motion_backend.move_to_joints(box_cfg["up_configuration"])

    print("Coverage path complete!")
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
