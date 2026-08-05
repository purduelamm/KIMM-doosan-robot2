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
from .transforms import lookup_link_transforms

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


def visualize_sampled_poses(
    mesh: trimesh.Trimesh,
    sampled_poses: list[np.ndarray],
    axis_len: float = 0.02,
) -> None:
    import open3d as o3d

    geometries = []

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices / MESH_DIMENSION)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.75, 0.75, 0.75])
    geometries.append(o3d_mesh)

    for T in sampled_poses:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_len, origin=T[:3, 3]
        )
        frame.rotate(T[:3, :3], center=T[:3, 3])
        geometries.append(frame)

    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len * 0.5)
    )

    print(f"[vis] Showing {len(sampled_poses)} sampled pose(s).")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Sampled EE Poses over CNC Mesh",
        width=1280,
        height=800,
        mesh_show_back_face=True,
    )


def o3d_mesh_from_trimesh(mesh: trimesh.Trimesh, color: list[float]):
    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color(color)
    return o3d_mesh


def load_collada_geometry_simple(path: str) -> trimesh.Trimesh | None:
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        print(f"[debug-vis] Could not parse Collada file {path}: {exc}")
        return None

    geometry_meshes = {}
    for geometry in root.findall(".//c:library_geometries/c:geometry", ns):
        mesh_elem = geometry.find("c:mesh", ns)
        if mesh_elem is None:
            continue

        sources = {}
        for source in mesh_elem.findall("c:source", ns):
            float_array = source.find("c:float_array", ns)
            if float_array is None or not float_array.text:
                continue
            values = np.fromstring(float_array.text, sep=" ", dtype=np.float64)
            if len(values) % 3 != 0:
                continue
            sources[source.attrib["id"]] = values.reshape((-1, 3))

        vertex_sources = {}
        for vertices in mesh_elem.findall("c:vertices", ns):
            pos_input = vertices.find("c:input[@semantic='POSITION']", ns)
            if pos_input is not None:
                vertex_sources[vertices.attrib["id"]] = pos_input.attrib["source"].lstrip("#")

        meshes = []
        for triangles in mesh_elem.findall("c:triangles", ns):
            p_elem = triangles.find("c:p", ns)
            if p_elem is None or not p_elem.text:
                continue

            inputs = triangles.findall("c:input", ns)
            if not inputs:
                continue

            stride = max(int(inp.attrib.get("offset", 0)) for inp in inputs) + 1
            vertex_input = next(
                (
                    inp
                    for inp in inputs
                    if inp.attrib.get("semantic") in ("VERTEX", "POSITION")
                ),
                None,
            )
            if vertex_input is None:
                continue

            source_id = vertex_input.attrib["source"].lstrip("#")
            if vertex_input.attrib.get("semantic") == "VERTEX":
                source_id = vertex_sources.get(source_id)
            if source_id not in sources:
                continue

            indices = np.fromstring(p_elem.text, sep=" ", dtype=np.int64)
            if len(indices) % stride != 0:
                continue
            faces = indices.reshape((-1, stride))[:, int(vertex_input.attrib.get("offset", 0))]
            if len(faces) % 3 != 0:
                continue
            faces = faces.reshape((-1, 3))
            meshes.append(trimesh.Trimesh(vertices=sources[source_id], faces=faces, process=False))

        if meshes:
            geometry_meshes[geometry.attrib["id"]] = trimesh.util.concatenate(meshes)

    if not geometry_meshes:
        return None

    scene_meshes = []

    def collect_node_meshes(node: ET.Element, parent_T: np.ndarray) -> None:
        T = parent_T.copy()
        matrix_elem = node.find("c:matrix", ns)
        if matrix_elem is not None and matrix_elem.text:
            values = np.fromstring(matrix_elem.text, sep=" ", dtype=np.float64)
            if len(values) == 16:
                T = T @ values.reshape((4, 4))

        for instance in node.findall("c:instance_geometry", ns):
            geometry_id = instance.attrib.get("url", "").lstrip("#")
            mesh = geometry_meshes.get(geometry_id)
            if mesh is None:
                continue
            mesh = mesh.copy()
            mesh.apply_transform(T)
            scene_meshes.append(mesh)

        for child in node.findall("c:node", ns):
            collect_node_meshes(child, T)

    visual_scene = root.find(".//c:library_visual_scenes/c:visual_scene", ns)
    if visual_scene is not None:
        for node_elem in visual_scene.findall("c:node", ns):
            collect_node_meshes(node_elem, np.eye(4))

    if scene_meshes:
        return trimesh.util.concatenate(scene_meshes)
    return trimesh.util.concatenate(list(geometry_meshes.values()))


def load_trimesh_geometry(path: str, scale: float = 1.0) -> trimesh.Trimesh | None:
    try:
        loaded = trimesh.load(path, force="scene")
    except Exception as exc:
        if path.lower().endswith(".dae"):
            loaded = load_collada_geometry_simple(path)
            if loaded is None:
                print(f"[debug-vis] Could not load {path}: {exc}")
                return None
        else:
            print(f"[debug-vis] Could not load {path}: {exc}")
            return None

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom.copy()
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            print(f"[debug-vis] No triangle geometry in {path}")
            return None
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        print(f"[debug-vis] Unsupported mesh type in {path}: {type(loaded)}")
        return None

    mesh = mesh.copy()
    mesh.apply_scale(scale)
    return mesh


def robot_visual_specs() -> dict[str, list[tuple[str, float, np.ndarray]]]:
    if ROBOT_MODEL != "m0609":
        print(f"[debug-vis] Robot visual mesh list is not configured for {ROBOT_MODEL}.")
        return {}

    mesh_root = os.path.join(
        os.path.dirname(SCRIPT_DIR),
        "dsr_description2",
        "meshes",
        "m0609_white",
    )

    def spec(filename: str, T_link_visual: np.ndarray | None = None):
        if T_link_visual is None:
            T_link_visual = np.eye(4)
        return (os.path.join(mesh_root, filename), 0.001, T_link_visual)

    T_blower_visual = make_SE3(
        R.from_euler("xyz", [0.0, 0.0, 1.5707963267948966]).as_matrix(),
        np.array([-0.006, 0.0, -1.035]),
    )

    return {
        "base_link": [spec("MF0609_0_0.dae")],
        "link_1": [spec("MF0609_1_0.dae")],
        "link_2": [
            spec("MF0609_2_0.dae"),
            spec("MF0609_2_1.dae"),
            spec("MF0609_2_2.dae"),
        ],
        "link_3": [spec("MF0609_3_0.dae")],
        "link_4": [
            spec("MF0609_4_0.dae"),
            spec("MF0609_4_1.dae"),
        ],
        "link_5": [spec("MF0609_5_0.dae")],
        "link_6": [spec("MF0609_6_0.dae")],
        "blower_link": [spec("blower.stl", T_blower_visual)],
    }


def visualize_debug_scene(
    cnc_mesh: trimesh.Trimesh,
    T_w_b: np.ndarray,
    T_w_cm: np.ndarray,
    T_w_cnc: np.ndarray,
    axis_len: float = 0.2,
) -> None:
    import open3d as o3d

    geometries = []

    cnc_o3d = o3d_mesh_from_trimesh(cnc_mesh.copy(), [0.72, 0.72, 0.72])
    cnc_o3d.scale(1.0 / MESH_DIMENSION, center=np.zeros(3))
    geometries.append(cnc_o3d)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_len, origin=[0.0, 0.0, 0.0]
    )
    geometries.append(world_frame)

    cnc_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_len * 0.6, origin=T_w_cnc[:3, 3]
    )
    cnc_frame.rotate(T_w_cnc[:3, :3], center=T_w_cnc[:3, 3])
    geometries.append(cnc_frame)

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_len * 0.6, origin=T_w_cm[:3, 3]
    )
    camera_frame.rotate(T_w_cm[:3, :3], center=T_w_cm[:3, 3])
    geometries.append(camera_frame)

    specs = robot_visual_specs()
    link_transforms = lookup_link_transforms(TF_SOURCE_FRAME, list(specs.keys()))
    mesh_count = 0
    link_points = []
    for link_name, visual_specs in specs.items():
        T_b_link = link_transforms.get(link_name)
        if link_name == TF_SOURCE_FRAME:
            T_b_link = np.eye(4)
        if T_b_link is None:
            continue

        T_w_link = T_w_b @ T_b_link
        link_points.append(T_w_link[:3, 3])
        link_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_len * 0.25, origin=T_w_link[:3, 3]
        )
        link_frame.rotate(T_w_link[:3, :3], center=T_w_link[:3, 3])
        geometries.append(link_frame)

        for path, scale, T_link_visual in visual_specs:
            mesh = load_trimesh_geometry(path, scale=scale)
            if mesh is None:
                continue
            robot_o3d = o3d_mesh_from_trimesh(mesh, [0.95, 0.95, 0.95])
            robot_o3d.transform(T_w_link @ T_link_visual)
            geometries.append(robot_o3d)
            mesh_count += 1

    if len(link_points) >= 2:
        link_points = np.array(link_points)
        link_lines = [[i, i + 1] for i in range(len(link_points) - 1)]
        robot_skeleton = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(link_points),
            lines=o3d.utility.Vector2iVector(link_lines),
        )
        robot_skeleton.paint_uniform_color([0.05, 0.05, 0.05])
        geometries.append(robot_skeleton)

    print(
        "[debug-vis] Showing world frame, CNC frame, camera frame, "
        f"{mesh_count} robot visual mesh(es), current robot link frames, and CNC mesh."
    )
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Debug: World, Camera, Robot, CNC",
        width=1280,
        height=800,
        mesh_show_back_face=True,
    )
