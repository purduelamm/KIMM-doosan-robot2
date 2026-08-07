#!/usr/bin/env python3
"""Estimate a four-ArUco box pose in the robot base frame from one image."""

import os
import time
from collections.abc import Mapping

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, Image
from tf2_ros import Buffer, TransformListener

from blow_from_mesh_helpers.config import (
    CAMERA_CFG,
    GAZ_TO_OPT_R,
    GAZ_TO_OPT_T,
    L6_TO_CAM_R,
    L6_TO_CAM_T,
)


EXPECTED_MARKER_IDS = (1, 2, 3, 4)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return a homogeneous transform from a rotation and translation."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def transform_from_tf_message(transform_msg) -> np.ndarray:
    """Convert a geometry_msgs Transform into a homogeneous transform."""
    translation = transform_msg.translation
    quaternion = transform_msg.rotation
    rotation = Rotation.from_quat(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    ).as_matrix()
    return make_transform(
        rotation,
        [translation.x, translation.y, translation.z],
    )


def link6_to_camera_transform() -> np.ndarray:
    """Return the calibrated link-6-to-optical-camera transform."""
    link6_to_camera_model = make_transform(
        L6_TO_CAM_R.as_matrix(),
        L6_TO_CAM_T,
    )
    camera_model_to_optical = make_transform(
        GAZ_TO_OPT_R.as_matrix(),
        GAZ_TO_OPT_T,
    )
    return link6_to_camera_model @ camera_model_to_optical


def compose_base_camera_transform(
    base_to_ee: np.ndarray,
    ee_to_camera: np.ndarray | None = None,
) -> np.ndarray:
    """Compose the base-to-camera transform used by blow_from_mesh.py."""
    if ee_to_camera is None:
        ee_to_camera = link6_to_camera_transform()
    return np.asarray(base_to_ee, dtype=float) @ np.asarray(ee_to_camera, dtype=float)


def construct_box_transform(marker_centers: Mapping[int, np.ndarray]) -> np.ndarray:
    """Construct camera-to-box from four marker centers in camera coordinates.

    Marker IDs are corners laid out as 2--1 on the top edge and 4--3 on the
    bottom edge. The box origin is the four-corner centroid. Its +X axis points
    from the left edge toward the right edge, +Y points from the bottom edge
    toward the top edge after orthogonalization, and +Z completes the
    right-handed frame.
    """
    missing = sorted(set(EXPECTED_MARKER_IDS) - set(marker_centers))
    if missing:
        raise ValueError(f"Missing required marker center(s): {missing}")

    centers = {
        marker_id: np.asarray(marker_centers[marker_id], dtype=float).reshape(3)
        for marker_id in EXPECTED_MARKER_IDS
    }
    if not all(np.all(np.isfinite(center)) for center in centers.values()):
        raise ValueError("Marker centers contain non-finite values.")

    origin = np.mean(list(centers.values()), axis=0)
    x_raw = 0.5 * ((centers[1] - centers[2]) + (centers[3] - centers[4]))
    y_raw = 0.5 * ((centers[1] - centers[3]) + (centers[2] - centers[4]))

    epsilon = 1e-9
    x_norm = np.linalg.norm(x_raw)
    if x_norm < epsilon:
        raise ValueError("Degenerate box geometry: left and right edges coincide.")
    x_axis = x_raw / x_norm

    y_orthogonal = y_raw - np.dot(y_raw, x_axis) * x_axis
    y_norm = np.linalg.norm(y_orthogonal)
    if y_norm < epsilon:
        raise ValueError(
            "Degenerate box geometry: box edges are parallel or the top and "
            "bottom edges coincide."
        )
    y_axis = y_orthogonal / y_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < epsilon:
        raise ValueError("Degenerate box geometry: could not construct a surface normal.")
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)

    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("Constructed box orientation is not right-handed.")
    return make_transform(rotation, origin)


def grab_image(node: Node, image_topic: str, timeout_sec: float) -> np.ndarray:
    """Subscribe to an RGB topic, capture one frame, and unsubscribe."""
    latest = {"image": None}
    bridge = CvBridge()

    def compressed_callback(msg: CompressedImage) -> None:
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        bgr_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr_image is not None:
            latest["image"] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    def raw_callback(msg: Image) -> None:
        latest["image"] = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    if image_topic.endswith("/compressed"):
        message_type = CompressedImage
        callback = compressed_callback
    else:
        message_type = Image
        callback = raw_callback

    subscription = node.create_subscription(message_type, image_topic, callback, 10)
    node.get_logger().info(f"Waiting for an RGB frame on {image_topic} ...")
    deadline = time.monotonic() + timeout_sec
    try:
        while latest["image"] is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_sec:.1f}s waiting for {image_topic}."
                )
    finally:
        node.destroy_subscription(subscription)

    image = latest["image"]
    node.get_logger().info(f"Captured RGB frame {image.shape[1]}x{image.shape[0]}.")
    return image


def visualize_camera_image(
    node: Node,
    rgb_image: np.ndarray,
    duration_ms: int,
) -> bool:
    """Show the annotated RGB pose result when a display is available."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        node.get_logger().warning(
            "Camera visualization requested, but no graphical display is available."
        )
        return False
    if duration_ms < 0:
        raise ValueError("visualization_duration_ms must be zero or greater.")

    window_name = "ArUco box pose - estimated coordinate frames"
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    window_created = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.imshow(window_name, bgr_image)
        if duration_ms == 0:
            node.get_logger().info(
                "Showing estimated marker and box frames; "
                "press any key in the image window to continue."
            )
            cv2.waitKey(0)
        else:
            node.get_logger().info(
                "Showing estimated marker and box frames for "
                f"{duration_ms / 1000.0:.1f}s."
            )
            cv2.waitKey(duration_ms)
    except cv2.error as exc:
        node.get_logger().warning(f"Could not display captured camera image: {exc}")
        return False
    finally:
        if window_created:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
    return True


def lookup_transform(
    node: Node,
    source_frame: str,
    target_frame: str,
    timeout_sec: float,
) -> np.ndarray:
    """Return the latest source-from-target TF transform."""
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    deadline = time.monotonic() + timeout_sec
    last_error = None
    node.get_logger().info(
        f"Waiting for TF transform {source_frame} -> {target_frame} ..."
    )
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                stamped_transform = tf_buffer.lookup_transform(
                    source_frame,
                    target_frame,
                    rclpy.time.Time(),
                )
                return transform_from_tf_message(stamped_transform.transform)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                last_error = exc
    finally:
        tf_listener.unregister()

    detail = f" Last TF error: {last_error}" if last_error is not None else ""
    raise TimeoutError(
        f"Timed out after {timeout_sec:.1f}s waiting for transform "
        f"{source_frame} -> {target_frame}.{detail}"
    )


def create_aruco_detector():
    """Create an ArUco detector compatible with old and new OpenCV APIs."""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    if hasattr(aruco, "ArucoDetector"):
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers

    # OpenCV 4.6 exposes DetectorParameters(), but objects made by that newer
    # constructor can crash the legacy detectMarkers() implementation.
    parameters = aruco.DetectorParameters_create()

    def detect(gray_image):
        return aruco.detectMarkers(gray_image, dictionary, parameters=parameters)

    return detect


def detect_marker_poses(
    rgb_image: np.ndarray,
    marker_size: float,
    camera_matrix: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Detect IDs 1-4 and return camera poses and image corners by marker ID."""
    if marker_size <= 0.0:
        raise ValueError("marker_size must be greater than zero.")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was built without the aruco module.")

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = create_aruco_detector()(gray_image)
    if ids is None or not corners:
        raise RuntimeError("No ArUco markers were detected in the captured image.")

    detected_ids = [int(marker_id) for marker_id in np.asarray(ids).reshape(-1)]
    required_indices: dict[int, int] = {}
    duplicate_ids = []
    for index, marker_id in enumerate(detected_ids):
        if marker_id not in EXPECTED_MARKER_IDS:
            continue
        if marker_id in required_indices:
            duplicate_ids.append(marker_id)
        else:
            required_indices[marker_id] = index

    if duplicate_ids:
        raise RuntimeError(
            f"Detected duplicate required marker ID(s): {sorted(set(duplicate_ids))}."
        )
    missing_ids = sorted(set(EXPECTED_MARKER_IDS) - set(required_indices))
    if missing_ids:
        raise RuntimeError(
            f"Missing required ArUco marker ID(s): {missing_ids}; "
            f"detected IDs: {detected_ids}."
        )

    selected_corners = [
        corners[required_indices[marker_id]] for marker_id in EXPECTED_MARKER_IDS
    ]
    distortion = np.zeros(5, dtype=float)
    rotation_vectors, translations, _ = cv2.aruco.estimatePoseSingleMarkers(
        selected_corners,
        marker_size,
        np.asarray(camera_matrix, dtype=float),
        distortion,
    )
    if (
        rotation_vectors is None
        or translations is None
        or len(rotation_vectors) != len(EXPECTED_MARKER_IDS)
        or len(translations) != len(EXPECTED_MARKER_IDS)
    ):
        raise RuntimeError("OpenCV failed to estimate all required marker poses.")

    marker_poses = {}
    marker_corners = {}
    for index, marker_id in enumerate(EXPECTED_MARKER_IDS):
        rotation_vector = np.asarray(rotation_vectors[index], dtype=float).reshape(3)
        translation = np.asarray(translations[index], dtype=float).reshape(3)
        if not (
            np.all(np.isfinite(rotation_vector))
            and np.all(np.isfinite(translation))
        ):
            raise RuntimeError("Estimated marker poses contain non-finite values.")
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        marker_poses[marker_id] = make_transform(rotation_matrix, translation)
        marker_corners[marker_id] = np.asarray(
            selected_corners[index], dtype=np.float32
        ).copy()

    return marker_poses, marker_corners


def detect_marker_centers(
    rgb_image: np.ndarray,
    marker_size: float,
    camera_matrix: np.ndarray,
) -> dict[int, np.ndarray]:
    """Detect IDs 1-4 and return their centers in optical-camera coordinates."""
    marker_poses, _ = detect_marker_poses(rgb_image, marker_size, camera_matrix)
    return {
        marker_id: transform[:3, 3].copy()
        for marker_id, transform in marker_poses.items()
    }


def draw_pose_overlay(
    rgb_image: np.ndarray,
    marker_poses: Mapping[int, np.ndarray],
    marker_corners: Mapping[int, np.ndarray],
    camera_to_box: np.ndarray,
    camera_matrix: np.ndarray,
    marker_size: float,
) -> np.ndarray:
    """Return an RGB image annotated with marker and derived box frames."""
    if marker_size <= 0.0:
        raise ValueError("marker_size must be greater than zero.")
    missing_poses = sorted(set(EXPECTED_MARKER_IDS) - set(marker_poses))
    missing_corners = sorted(set(EXPECTED_MARKER_IDS) - set(marker_corners))
    if missing_poses or missing_corners:
        raise ValueError(
            "Pose overlay requires marker poses and corners for IDs 1-4; "
            f"missing poses: {missing_poses}, missing corners: {missing_corners}."
        )

    camera_matrix = np.asarray(camera_matrix, dtype=float)
    camera_to_box = np.asarray(camera_to_box, dtype=float)
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must be 3x3.")
    if camera_to_box.shape != (4, 4):
        raise ValueError("camera_to_box must be 4x4.")

    distortion = np.zeros(5, dtype=float)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    ordered_corners = [marker_corners[marker_id] for marker_id in EXPECTED_MARKER_IDS]
    marker_ids = np.asarray(EXPECTED_MARKER_IDS, dtype=np.int32).reshape(-1, 1)
    cv2.aruco.drawDetectedMarkers(bgr_image, ordered_corners, marker_ids)

    for marker_id in EXPECTED_MARKER_IDS:
        camera_to_marker = np.asarray(marker_poses[marker_id], dtype=float)
        if camera_to_marker.shape != (4, 4):
            raise ValueError(f"Marker {marker_id} pose must be 4x4.")
        rotation_vector, _ = cv2.Rodrigues(camera_to_marker[:3, :3])
        cv2.drawFrameAxes(
            bgr_image,
            camera_matrix,
            distortion,
            rotation_vector,
            camera_to_marker[:3, 3],
            marker_size / 2.0,
            2,
        )

    box_rotation_vector, _ = cv2.Rodrigues(camera_to_box[:3, :3])
    box_translation = camera_to_box[:3, 3]
    cv2.drawFrameAxes(
        bgr_image,
        camera_matrix,
        distortion,
        box_rotation_vector,
        box_translation,
        marker_size,
        3,
    )
    projected_origin, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=float),
        box_rotation_vector,
        box_translation,
        camera_matrix,
        distortion,
    )
    origin_pixel = tuple(np.rint(projected_origin.reshape(2)).astype(int))
    cv2.putText(
        bgr_image,
        "BOX",
        (origin_pixel[0] + 8, origin_pixel[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def pose_message(transform: np.ndarray, frame_id: str, node: Node) -> PoseStamped:
    """Convert a homogeneous transform into a stamped ROS pose."""
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    message = PoseStamped()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = frame_id
    message.pose.position.x = float(transform[0, 3])
    message.pose.position.y = float(transform[1, 3])
    message.pose.position.z = float(transform[2, 3])
    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])
    return message


def report_pose(node: Node, transform: np.ndarray, frame_id: str) -> None:
    """Log a box transform as a matrix, XYZ, quaternion, and XYZ Euler angles."""
    xyz = transform[:3, 3]
    rotation = Rotation.from_matrix(transform[:3, :3])
    quaternion = rotation.as_quat()
    rpy_degrees = rotation.as_euler("xyz", degrees=True)
    matrix_text = np.array2string(transform, precision=6, suppress_small=True)
    node.get_logger().info(
        f"Box pose with respect to {frame_id}:\n"
        f"T_{frame_id}_box =\n{matrix_text}\n"
        f"XYZ [m] = {np.array2string(xyz, precision=6)}\n"
        f"Quaternion [x y z w] = {np.array2string(quaternion, precision=6)}\n"
        f"RPY [deg] = {np.array2string(rpy_degrees, precision=6)}"
    )


def declare_parameters(node: Node) -> dict[str, object]:
    """Declare and return the estimator's ROS parameters."""
    intrinsics = CAMERA_CFG["intrinsics"]
    defaults = {
        "image_topic": "/camera/camera/color/image_raw",
        # "image_topic": "/camera/image_raw",
        "output_topic": "/aruco_box_pose",
        "marker_size": 0.115,
        "base_frame": "base_link",
        "ee_frame": "link_6",
        "frame_timeout_sec": 5.0,
        "image_timeout_sec": 5.0,
        "visualize_image": True,
        "visualization_duration_ms": 0,
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics.get("fy", intrinsics["fx"])),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
    }
    for name, default in defaults.items():
        node.declare_parameter(name, default)
    return {name: node.get_parameter(name).value for name in defaults}


def run(node: Node) -> None:
    """Capture one frame, estimate the box pose, publish it, and return."""
    parameters = declare_parameters(node)
    image = grab_image(
        node,
        str(parameters["image_topic"]),
        float(parameters["image_timeout_sec"]),
    )
    base_to_link6 = lookup_transform(
        node,
        str(parameters["base_frame"]),
        str(parameters["ee_frame"]),
        float(parameters["frame_timeout_sec"]),
    )
    base_to_camera = compose_base_camera_transform(base_to_link6)
    node.get_logger().info(
        "Calculated optical-camera pose from base-to-link-6 TF and configured "
        "link-to-camera and camera-to-optical transforms."
    )

    camera_matrix = np.array(
        [
            [float(parameters["fx"]), 0.0, float(parameters["cx"])],
            [0.0, float(parameters["fy"]), float(parameters["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    marker_poses, marker_corners = detect_marker_poses(
        image,
        float(parameters["marker_size"]),
        camera_matrix,
    )
    marker_centers = {
        marker_id: transform[:3, 3].copy()
        for marker_id, transform in marker_poses.items()
    }
    marker_ids_text = ", ".join(
        str(marker_id) for marker_id in sorted(marker_centers)
    )
    node.get_logger().info(f"Detected required ArUco markers: {marker_ids_text}")

    camera_to_box = construct_box_transform(marker_centers)
    base_to_box = base_to_camera @ camera_to_box
    base_frame = str(parameters["base_frame"])
    report_pose(node, base_to_box, base_frame)

    publisher = node.create_publisher(PoseStamped, str(parameters["output_topic"]), 10)
    publisher.publish(pose_message(base_to_box, base_frame, node))
    node.get_logger().info(
        f"Published box pose once on {parameters['output_topic']}."
    )
    delivery_deadline = time.monotonic() + 0.5
    while time.monotonic() < delivery_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    if bool(parameters["visualize_image"]):
        annotated_image = draw_pose_overlay(
            image,
            marker_poses,
            marker_corners,
            camera_to_box,
            camera_matrix,
            float(parameters["marker_size"]),
        )
        visualize_camera_image(
            node,
            annotated_image,
            int(parameters["visualization_duration_ms"]),
        )


def main(args=None) -> None:
    """Run the one-shot ArUco box pose estimator."""
    rclpy.init(args=args)
    node = rclpy.create_node("aruco_box_pose_estimator")
    exit_code = 0
    try:
        run(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
        exit_code = 130
    except Exception as exc:
        node.get_logger().error(f"Box pose estimation failed: {exc}")
        exit_code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
