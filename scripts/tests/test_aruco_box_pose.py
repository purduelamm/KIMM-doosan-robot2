"""Headless tests for ArUco marker and box-pose visualization."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import aruco_box_pose as pose_estimator


def test_construct_box_transform_uses_marker_corners_as_box_axes():
    expected_rotation = Rotation.from_euler("xyz", [0.25, -0.4, 0.7]).as_matrix()
    expected_translation = np.array([0.3, -0.2, 1.1])
    local_corners = {
        1: np.array([0.4, 0.2, 0.0]),
        2: np.array([-0.4, 0.2, 0.0]),
        3: np.array([0.4, -0.2, 0.0]),
        4: np.array([-0.4, -0.2, 0.0]),
    }
    camera_corners = {
        marker_id: expected_rotation @ corner + expected_translation
        for marker_id, corner in local_corners.items()
    }

    camera_to_box = pose_estimator.construct_box_transform(camera_corners)

    np.testing.assert_allclose(camera_to_box[:3, :3], expected_rotation)
    np.testing.assert_allclose(camera_to_box[:3, 3], expected_translation)


def test_link6_to_camera_transform_composes_both_configured_poses(monkeypatch):
    link6_rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.4])
    optical_rotation = Rotation.from_euler("xyz", [-0.5, 0.1, 0.25])
    link6_translation = np.array([0.12, -0.07, 0.21])
    optical_translation = np.array([0.03, 0.04, -0.02])
    monkeypatch.setattr(pose_estimator, "L6_TO_CAM_R", link6_rotation)
    monkeypatch.setattr(pose_estimator, "L6_TO_CAM_T", link6_translation)
    monkeypatch.setattr(pose_estimator, "GAZ_TO_OPT_R", optical_rotation)
    monkeypatch.setattr(pose_estimator, "GAZ_TO_OPT_T", optical_translation)

    expected = pose_estimator.make_transform(
        link6_rotation.as_matrix(), link6_translation
    ) @ pose_estimator.make_transform(
        optical_rotation.as_matrix(), optical_translation
    )

    np.testing.assert_allclose(pose_estimator.link6_to_camera_transform(), expected)


def test_configured_camera_offset_affects_base_to_box_pose(monkeypatch):
    link6_rotation = Rotation.from_euler("z", 0.35)
    optical_rotation = Rotation.from_euler("y", -0.45)
    link6_translation = np.array([0.08, -0.03, 0.14])
    optical_translation = np.array([0.01, 0.02, -0.04])
    monkeypatch.setattr(pose_estimator, "L6_TO_CAM_R", link6_rotation)
    monkeypatch.setattr(pose_estimator, "L6_TO_CAM_T", link6_translation)
    monkeypatch.setattr(pose_estimator, "GAZ_TO_OPT_R", optical_rotation)
    monkeypatch.setattr(pose_estimator, "GAZ_TO_OPT_T", optical_translation)

    base_to_link6 = pose_estimator.make_transform(
        Rotation.from_euler("x", 0.6).as_matrix(), [0.4, -0.2, 0.7]
    )
    optical_to_box = pose_estimator.make_transform(
        Rotation.from_euler("z", -0.2).as_matrix(), [0.1, 0.05, 0.8]
    )
    expected = (
        base_to_link6
        @ pose_estimator.make_transform(link6_rotation.as_matrix(), link6_translation)
        @ pose_estimator.make_transform(optical_rotation.as_matrix(), optical_translation)
        @ optical_to_box
    )

    base_to_camera = pose_estimator.compose_base_camera_transform(base_to_link6)
    base_to_box = base_to_camera @ optical_to_box

    np.testing.assert_allclose(base_to_box, expected)
    assert not np.allclose(base_to_box, base_to_link6 @ optical_to_box)


def test_detect_marker_poses_retains_rotation_translation_and_corners(monkeypatch):
    detected_ids = np.array([[3], [1], [4], [2]], dtype=np.int32)
    detected_corners = [
        np.full((1, 4, 2), marker_id, dtype=np.float32)
        for marker_id in detected_ids.reshape(-1)
    ]
    monkeypatch.setattr(
        pose_estimator,
        "create_aruco_detector",
        lambda: lambda _gray: (detected_corners, detected_ids, []),
    )

    rotation_vectors = np.array(
        [[[0.0, 0.0, 0.1 * marker_id]] for marker_id in (1, 2, 3, 4)],
        dtype=float,
    )
    translations = np.array(
        [[[marker_id, marker_id + 0.5, marker_id + 1.0]] for marker_id in (1, 2, 3, 4)],
        dtype=float,
    )
    monkeypatch.setattr(
        pose_estimator.cv2.aruco,
        "estimatePoseSingleMarkers",
        lambda *_args: (rotation_vectors, translations, None),
    )

    marker_poses, marker_corners = pose_estimator.detect_marker_poses(
        np.zeros((20, 20, 3), dtype=np.uint8),
        marker_size=0.1,
        camera_matrix=np.eye(3),
    )

    assert set(marker_poses) == set(pose_estimator.EXPECTED_MARKER_IDS)
    for marker_id in pose_estimator.EXPECTED_MARKER_IDS:
        expected_rotation, _ = pose_estimator.cv2.Rodrigues(
            np.array([0.0, 0.0, 0.1 * marker_id])
        )
        np.testing.assert_allclose(marker_poses[marker_id][:3, :3], expected_rotation)
        np.testing.assert_allclose(
            marker_poses[marker_id][:3, 3],
            [marker_id, marker_id + 0.5, marker_id + 1.0],
        )
        np.testing.assert_array_equal(marker_corners[marker_id], marker_id)


def test_detect_marker_centers_compatibility_wrapper(monkeypatch):
    marker_poses = {
        marker_id: pose_estimator.make_transform(np.eye(3), [marker_id, 0.0, 1.0])
        for marker_id in pose_estimator.EXPECTED_MARKER_IDS
    }
    monkeypatch.setattr(
        pose_estimator,
        "detect_marker_poses",
        lambda *_args: (marker_poses, {}),
    )

    centers = pose_estimator.detect_marker_centers(
        np.zeros((1, 1, 3), dtype=np.uint8), 0.1, np.eye(3)
    )

    for marker_id in pose_estimator.EXPECTED_MARKER_IDS:
        np.testing.assert_allclose(centers[marker_id], [marker_id, 0.0, 1.0])


def test_draw_pose_overlay_draws_all_frames_and_preserves_source(monkeypatch):
    rgb_image = np.zeros((240, 320, 3), dtype=np.uint8)
    original_image = rgb_image.copy()
    camera_matrix = np.array(
        [[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]
    )
    marker_poses = {
        marker_id: pose_estimator.make_transform(
            np.eye(3),
            [(marker_id - 2.5) * 0.1, 0.0, 1.0],
        )
        for marker_id in pose_estimator.EXPECTED_MARKER_IDS
    }
    marker_corners = {
        marker_id: np.array(
            [[[40 * marker_id, 80], [40 * marker_id + 20, 80],
              [40 * marker_id + 20, 100], [40 * marker_id, 100]]],
            dtype=np.float32,
        )
        for marker_id in pose_estimator.EXPECTED_MARKER_IDS
    }
    camera_to_box = pose_estimator.make_transform(np.eye(3), [0.0, 0.0, 1.0])

    frame_calls = []
    real_draw_frame_axes = pose_estimator.cv2.drawFrameAxes

    def record_draw_frame_axes(*args):
        frame_calls.append((float(args[5]), int(args[6])))
        return real_draw_frame_axes(*args)

    labels = []
    real_put_text = pose_estimator.cv2.putText

    def record_put_text(*args):
        labels.append(args[1])
        return real_put_text(*args)

    monkeypatch.setattr(pose_estimator.cv2, "drawFrameAxes", record_draw_frame_axes)
    monkeypatch.setattr(pose_estimator.cv2, "putText", record_put_text)

    annotated = pose_estimator.draw_pose_overlay(
        rgb_image,
        marker_poses,
        marker_corners,
        camera_to_box,
        camera_matrix,
        marker_size=0.1,
    )

    np.testing.assert_array_equal(rgb_image, original_image)
    assert np.any(annotated != original_image)
    assert frame_calls == [(0.05, 2)] * 4 + [(0.1, 3)]
    assert labels == ["BOX"]


class _Logger:
    def info(self, _message):
        pass


class _Publisher:
    def __init__(self, events):
        self.events = events

    def publish(self, _message):
        self.events.append("publish")


class _Node:
    def __init__(self, events):
        self.events = events
        self.publisher = _Publisher(events)

    def get_logger(self):
        return _Logger()

    def create_publisher(self, *_args):
        return self.publisher


def _run_with_visualization(monkeypatch, visualize_image):
    events = []
    parameters = {
        "image_topic": "/camera",
        "output_topic": "/box",
        "marker_size": 0.1,
        "base_frame": "base",
        "ee_frame": "ee",
        "frame_timeout_sec": 1.0,
        "image_timeout_sec": 1.0,
        "visualize_image": visualize_image,
        "visualization_duration_ms": 10,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 50.0,
    }
    marker_centers = {
        1: [0.5, 0.5, 1.0],
        2: [-0.5, 0.5, 1.0],
        3: [0.5, -0.5, 1.0],
        4: [-0.5, -0.5, 1.0],
    }
    marker_poses = {
        marker_id: pose_estimator.make_transform(np.eye(3), center)
        for marker_id, center in marker_centers.items()
    }
    marker_corners = {
        marker_id: np.zeros((1, 4, 2), dtype=np.float32)
        for marker_id in marker_centers
    }

    monkeypatch.setattr(pose_estimator, "declare_parameters", lambda _node: parameters)
    monkeypatch.setattr(
        pose_estimator, "grab_image", lambda *_args: np.zeros((2, 2, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(pose_estimator, "lookup_transform", lambda *_args: np.eye(4))
    monkeypatch.setattr(
        pose_estimator, "compose_base_camera_transform", lambda *_args: np.eye(4)
    )
    monkeypatch.setattr(
        pose_estimator,
        "detect_marker_poses",
        lambda *_args: (marker_poses, marker_corners),
    )
    monkeypatch.setattr(pose_estimator, "report_pose", lambda *_args: None)
    monkeypatch.setattr(pose_estimator, "pose_message", lambda *_args: object())
    monkeypatch.setattr(
        pose_estimator,
        "draw_pose_overlay",
        lambda *_args: events.append("overlay") or np.zeros((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        pose_estimator,
        "visualize_camera_image",
        lambda *_args: events.append("display") or True,
    )
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(pose_estimator.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(pose_estimator.rclpy, "spin_once", lambda *_args, **_kwargs: None)

    pose_estimator.run(_Node(events))
    return events


def test_run_displays_overlay_after_publishing(monkeypatch):
    events = _run_with_visualization(monkeypatch, visualize_image=True)

    assert events == ["publish", "overlay", "display"]


def test_run_skips_overlay_when_visualization_is_disabled(monkeypatch):
    events = _run_with_visualization(monkeypatch, visualize_image=False)

    assert events == ["publish"]
