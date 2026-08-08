"""Launch script for generating and executing chip blowing poses from a CNC mesh."""

import time

import rclpy
import trimesh

from blow_from_mesh_helpers import ros_context
from blow_from_mesh_helpers.config import (
    CAMERA_CFG,
    CNC_MESH_LOCAL_OFFSET,
    CNC_mesh_path,
    CONFIG,
    DETECTION_STAGING_TARGET,
    IMG_TOPIC,
    INIT_TARGET,
    MESH_DIMENSION,
    QUERY_PIXELS,
    WLD_TO_BASE_R,
    WLD_TO_BASE_T,
    WLD_TO_CNC_R,
    WLD_TO_CNC_T,
)
from blow_from_mesh_helpers.detection import (
    RGBDChipDetector,
    RGBDFrameGrabber,
    grab_image,
    sample_pixels_from_pdf,
    visualize_pdf_debug,
)
from blow_from_mesh_helpers.math_utils import make_mesh_world_transform, make_SE3
from blow_from_mesh_helpers.mesh_pose import (
    compute_keyframe_from_pixel,
    get_query_pixels,
)
from blow_from_mesh_helpers.motion import (
    check_reachable,
    consistent_rotations,
    create_motion_backend,
    generate_smooth_trajectory,
    mesh_pose_to_base,
)
from blow_from_mesh_helpers.transforms import get_current_camera_transform
from blow_from_mesh_helpers.visualization import (
    visualize_debug_scene,
    visualize_sampled_poses,
    visualize_trajectory,
)


def execute_blowing_trajectory(motion_backend, trajectory) -> None:
    """Execute one trajectory while keeping the airgun active."""
    print("[exec] Activating airgun...")
    try:
        ros_context.air_node.tool_airgun(True)
        time.sleep(1.0)
        motion_backend.execute_plan(trajectory)
    finally:
        print("[exec] Deactivating airgun...")
        ros_context.air_node.tool_airgun(False)
        time.sleep(1.0)


def main(args=None):
    ros_context.init_ros()

    CNC_mesh = trimesh.load_mesh(CNC_mesh_path)
    # Mesh vertices use mesh units (millimetres for the Doosan configuration).
    # Place the geometry in the global/world coordinate system once here so
    # visualization, ray casting, normal queries, and MoveIt collision geometry
    # all consume the same transformed mesh.
    R_w_cnc = WLD_TO_CNC_R.as_matrix()
    mesh_world_transform = make_mesh_world_transform(
        R_w_cnc,
        WLD_TO_CNC_T,
        CNC_MESH_LOCAL_OFFSET,
        MESH_DIMENSION,
    )
    CNC_mesh.apply_transform(mesh_world_transform)

    T_w_cnc = make_SE3(R_w_cnc, WLD_TO_CNC_T)
    print("[mesh] T_world_cnc from YAML:\n", T_w_cnc)
    if any(abs(value) > 0.0 for value in CNC_MESH_LOCAL_OFFSET):
        print("[mesh] OBJ local offset [m]:", CNC_MESH_LOCAL_OFFSET)
    T_w_b = make_SE3(WLD_TO_BASE_R.as_matrix(), WLD_TO_BASE_T)
    motion_backend = create_motion_backend()
    detector_cfg = CONFIG.get("chip_detector", {})
    execution_cfg = CONFIG.get("execution", {})

    if ros_context.is_doosan_robot():
        ros_context.set_robot_mode(ros_context.ROBOT_MODE_AUTONOMOUS)
    motion_backend.move_to_init(INIT_TARGET, CNC_mesh, T_w_b)
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

        motion_backend.move_to_joints(DETECTION_STAGING_TARGET)
        input("[detect] Robot is at detection staging pose. Prepare chips, then press Enter to detect...")

        motion_backend.move_to_init(INIT_TARGET, CNC_mesh, T_w_b)
        time.sleep(3)

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
        if detector_cfg.get("show_pdf_overlay", True):
            visualize_pdf_debug(
                current_rgb_bgr,
                detector.last_depth_diff_mm,
                chip_pdf,
                detector.last_depth_pdf,
                detector.last_rgb_pdf,
                query_pixels,
                alpha=float(detector_cfg.get("pdf_overlay_alpha", 0.55)),
            )
        T_w_cm = get_current_camera_transform(T_w_b)
    else:
        T_w_cm = get_current_camera_transform(T_w_b)
        print("[DEBUG] T_w_cm: ", T_w_cm)
        if CONFIG.get("visualization", {}).get("show_debug_scene", True):
            visualize_debug_scene(
                CNC_mesh,
                T_w_b,
                T_w_cm,
                T_w_cnc,
                axis_len=float(
                    CONFIG.get("visualization", {}).get("trajectory_axis_len", 0.2)
                ),
            )
        snapshot = None
        if not QUERY_PIXELS and CONFIG.get("point_input", {}).get(
            "use_interactive_picker_when_query_pixels_empty", False
        ):
            snapshot = grab_image(IMG_TOPIC)
        query_pixels = get_query_pixels(snapshot)

    sampled_poses = []
    keyframes = []
    skipped_unreachable = 0
    for point_idx, (pu, pv) in enumerate(query_pixels):
        print(f"\n[loop] === Point {point_idx} ===")
        print(f"[main] Picked pixel: ({pu:.1f}, {pv:.1f})")
        T_world_ee = compute_keyframe_from_pixel(pu, pv, T_w_cm, CNC_mesh)
        sampled_poses.append(T_world_ee)
        T_base_ee = mesh_pose_to_base(T_world_ee, T_w_b)
        T_base_ee = T_base_ee.copy()
        if not check_reachable(T_base_ee):
            skipped_unreachable += 1
            print(f"[loop] Skipping point {point_idx}: target pose is unreachable.")
            continue
        keyframes.append(T_world_ee)

    print(
        f"\n[traj] Collected {len(keyframes)} reachable keyframe(s); "
        f"skipped {skipped_unreachable} unreachable target(s)."
    )
    visualize_sampled_poses(
        CNC_mesh,
        sampled_poses,
        axis_len=float(CONFIG.get("visualization", {}).get("trajectory_axis_len", 0.2)),
    )
    if not keyframes:
        raise RuntimeError("[traj] No reachable keyframes found from sampled pixels.")

    keyframes = consistent_rotations(keyframes)

    trajectory = generate_smooth_trajectory(
        keyframes,
        n_interp=50,
        T_w_b=T_w_b,
        cnc_mesh=CNC_mesh,
        motion_backend=motion_backend,
    )
    if trajectory.planned_with_moveit:
        print(
            f"[traj] Prepared {len(trajectory.target_base_pose_groups)} "
            f"keypoint group(s) for online MoveIt planning; "
            f"{len(trajectory.target_poses)} target pose(s) kept for visualization."
        )
    else:
        print(
            f"[traj] Generated one continuous trajectory with "
            f"{len(trajectory)} interpolated guide poses."
        )

    if CONFIG.get("visualization", {}).get("show_trajectory", True):
        visualize_trajectory(
            CNC_mesh,
            trajectory.target_poses or keyframes,
            trajectory,
            axis_len=float(CONFIG.get("visualization", {}).get("trajectory_axis_len", 0.2)),
        )

    if execution_cfg.get("execute_gazebo_first", True):
        print("[exec] Executing the continuous trajectory on the current backend...")
        execute_blowing_trajectory(motion_backend, trajectory)

    if execution_cfg.get("confirm_real_execution", True):
        confirm = input(
            "[exec] Press y to execute the same planned trajectory on the real robot. [y/N]: "
        ).strip().lower()
        if confirm == "y":
            print("[exec] Executing the continuous trajectory on the real robot...")
            execute_blowing_trajectory(motion_backend, trajectory)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
