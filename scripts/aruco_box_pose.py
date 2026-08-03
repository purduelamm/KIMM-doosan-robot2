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
    rotation = (L6_TO_CAM_R * GAZ_TO_OPT_R).as_matrix()
    return make_transform(rotation, L6_TO_CAM_T)


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

    The box origin is the four-center centroid. Its +X axis points from marker 4
    to marker 1, +Y points from marker 3 toward marker 2 after orthogonalization,
    and +Z completes the right-handed frame.
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
    x_raw = centers[1] - centers[4]
    y_raw = centers[2] - centers[3]

    epsilon = 1e-9
    x_norm = np.linalg.norm(x_raw)
    if x_norm < epsilon:
        raise ValueError("Degenerate box geometry: markers 1 and 4 coincide.")
    x_axis = x_raw / x_norm

    y_orthogonal = y_raw - np.dot(y_raw, x_axis) * x_axis
    y_norm = np.linalg.norm(y_orthogonal)
    if y_norm < epsilon:
        raise ValueError(
            "Degenerate box geometry: marker ID axes are parallel or markers 2 and 3 coincide."
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
    """Show a captured RGB image in an OpenCV window when a display is available."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        node.get_logger().warning(
            "Camera visualization requested, but no graphical display is available."
        )
        return False
    if duration_ms < 0:
        raise ValueError("visualization_duration_ms must be zero or greater.")

    window_name = "ArUco box pose - captured camera image"
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    window_created = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.imshow(window_name, bgr_image)
        if duration_ms == 0:
            node.get_logger().info(
                "Showing captured camera image; press any key in the image window to continue."
            )
            cv2.waitKey(0)
        else:
            node.get_logger().info(
                f"Showing captured camera image for {duration_ms / 1000.0:.1f}s."
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


def detect_marker_centers(
    rgb_image: np.ndarray,
    marker_size: float,
    camera_matrix: np.ndarray,
) -> dict[int, np.ndarray]:
    """Detect IDs 1-4 and return their centers in optical-camera coordinates."""
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
    _, translations, _ = cv2.aruco.estimatePoseSingleMarkers(
        selected_corners,
        marker_size,
        np.asarray(camera_matrix, dtype=float),
        distortion,
    )
    if translations is None or len(translations) != len(EXPECTED_MARKER_IDS):
        raise RuntimeError("OpenCV failed to estimate all required marker positions.")

    marker_centers = {
        marker_id: np.asarray(translations[index], dtype=float).reshape(3)
        for index, marker_id in enumerate(EXPECTED_MARKER_IDS)
    }
    if not all(np.all(np.isfinite(center)) for center in marker_centers.values()):
        raise RuntimeError("Estimated marker positions contain non-finite values.")
    return marker_centers


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
        "image_topic": "/camera/image_raw",
        "output_topic": "/aruco_box_pose",
        "marker_size": 0.115,
        "base_frame": "base_link",
        "ee_frame": "link_6",
        "frame_timeout_sec": 5.0,
        "image_timeout_sec": 5.0,
        "visualize_image": True,
        "visualization_duration_ms": 2000,
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
    if bool(parameters["visualize_image"]):
        visualize_camera_image(
            node,
            image,
            int(parameters["visualization_duration_ms"]),
        )
    base_to_ee = lookup_transform(
        node,
        str(parameters["base_frame"]),
        str(parameters["ee_frame"]),
        float(parameters["frame_timeout_sec"]),
    )
    base_to_camera = compose_base_camera_transform(base_to_ee)
    node.get_logger().info(
        "Calculated optical-camera pose from the current end-effector TF and calibration."
    )

    camera_matrix = np.array(
        [
            [float(parameters["fx"]), 0.0, float(parameters["cx"])],
            [0.0, float(parameters["fy"]), float(parameters["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    marker_centers = detect_marker_centers(
        image,
        float(parameters["marker_size"]),
        camera_matrix,
    )
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
