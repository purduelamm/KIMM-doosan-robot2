"""Tests for clean-reference RGB SIFT feature rejection."""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blow_from_mesh_helpers import detection


def keypoint(x, y):
    return cv2.KeyPoint(float(x), float(y), 1.0)


def detector_config(**overrides):
    config = {
        "rgb_fit_scale": 0.5,
        "rgb_num_features": 0,
        "rgb_sift_contrast": 0.02,
        "rgb_sift_edge": 12,
        "rgb_n_components": 5,
        "rgb_include_outliers": True,
        "rgb_outlier_radius": 15,
        "rgb_outlier_probability": 0.8,
        "rgb_ignore_feature_radius": 6.0,
        "rgb_pdf_downsample_factor": 1,
    }
    config.update(overrides)
    return config


class FakeFitRGB:
    keypoint_batches = []
    last_fit_keypoints = None

    def __init__(self, image):
        self.image = image
        self.H, self.W = image.shape[:2]
        self.gmm = None
        self.keypoint_coords = None

    def _sift_detector(self, *_args):
        return self.keypoint_batches.pop(0)

    def _fit_gmm(self, keypoints, n_components=5):
        type(self).last_fit_keypoints = list(keypoints)
        if not keypoints:
            return None, None
        return object(), np.asarray([point.pt for point in keypoints], dtype=np.float32)

    def rgb_gmm_to_pdf(self, **_kwargs):
        return np.ones((self.H, self.W), dtype=np.float32)


def install_fake_rgb_detector(monkeypatch, batches):
    FakeFitRGB.keypoint_batches = [list(batch) for batch in batches]
    FakeFitRGB.last_fit_keypoints = None
    monkeypatch.setattr(
        detection,
        "get_detector_classes",
        lambda: (object, FakeFitRGB),
    )


def test_baseline_features_accumulate_across_frames(monkeypatch):
    install_fake_rgb_detector(
        monkeypatch,
        [[keypoint(1, 2)], [keypoint(3, 4), keypoint(5, 6)]],
    )
    detector = detection.RGBDChipDetector(detector_config())
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    detector.add_rgb_baseline_frame(image)
    detector.add_rgb_baseline_frame(image)

    assert detector.rgb_baseline_frame_count == 2
    np.testing.assert_allclose(
        detector.rgb_baseline_feature_coords,
        [[1, 2], [3, 4], [5, 6]],
    )


def test_radius_filter_rejects_inside_and_boundary_features():
    detector = detection.RGBDChipDetector(
        detector_config(rgb_ignore_feature_radius=6.0)
    )
    detector.rgb_baseline_feature_coords = np.asarray([[10, 10]], dtype=np.float32)
    points = [keypoint(10, 10), keypoint(16, 10), keypoint(17, 10)]

    retained = detector.filter_rgb_baseline_features(points)

    assert [point.pt for point in retained] == [(17.0, 10.0)]


def test_no_baseline_retains_all_current_features():
    detector = detection.RGBDChipDetector(detector_config())
    points = [keypoint(1, 1), keypoint(2, 2)]

    assert detector.filter_rgb_baseline_features(points) == points


def test_all_rejected_features_return_empty_rgb_pdf(monkeypatch):
    install_fake_rgb_detector(monkeypatch, [[keypoint(10, 10)]])
    detector = detection.RGBDChipDetector(detector_config())
    detector.rgb_baseline_feature_coords = np.asarray([[10, 10]], dtype=np.float32)
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    pdf = detector._rgb_pdf(image, (40, 40))

    np.testing.assert_array_equal(pdf, np.zeros((40, 40), dtype=np.float32))
    assert FakeFitRGB.last_fit_keypoints == []
    assert detector.last_rgb_sift_overlay.shape == image.shape


def test_empty_rgb_pdf_does_not_abort_nonempty_depth_detection(monkeypatch):
    install_fake_rgb_detector(monkeypatch, [[keypoint(10, 10)]])
    detector = detection.RGBDChipDetector(detector_config())
    detector.rgb_baseline_feature_coords = np.asarray([[10, 10]], dtype=np.float32)
    monkeypatch.setattr(
        detector,
        "_depth_pdf",
        lambda diff: np.ones(diff.shape, dtype=np.float32),
    )
    monkeypatch.setattr(
        detector,
        "_rgb_saliency_pdf",
        lambda _image, shape: np.zeros(shape, dtype=np.float32),
    )
    reference_depth = np.full((40, 40), 100.0, dtype=np.float32)
    current_depth = np.full((40, 40), 95.0, dtype=np.float32)
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    merged = detector.detect(reference_depth, current_depth, image)

    assert float(merged.sum()) > 0.0
    np.testing.assert_array_equal(
        detector.last_rgb_pdf,
        np.zeros((40, 40), dtype=np.float32),
    )


def test_depth_difference_is_zero_above_baseline_depth_threshold(monkeypatch):
    detector = detection.RGBDChipDetector(
        detector_config(baseline_depth_max_mm=1000.0)
    )
    captured = {}

    def depth_pdf(diff):
        captured["diff"] = diff.copy()
        return np.ones(diff.shape, dtype=np.float32)

    monkeypatch.setattr(detector, "_depth_pdf", depth_pdf)
    monkeypatch.setattr(
        detector,
        "_rgb_pdf",
        lambda _image, shape, feature_allowed_mask=None: np.zeros(
            shape, dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        detector,
        "_rgb_saliency_pdf",
        lambda _image, shape: np.zeros(shape, dtype=np.float32),
    )
    reference_depth = np.asarray(
        [[900.0, 1000.0, 1000.1, 1400.0]], dtype=np.float32
    )
    current_depth = np.asarray(
        [[800.0, 900.0, 900.0, 1300.0]], dtype=np.float32
    )

    detector.detect(
        reference_depth,
        current_depth,
        np.zeros((1, 4, 3), dtype=np.uint8),
    )

    np.testing.assert_allclose(captured["diff"], [[100.0, 100.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        detector.last_depth_diff_mm,
        [[100.0, 100.0, 0.0, 0.0]],
    )


def test_rgb_feature_is_rejected_where_baseline_depth_forces_zero(monkeypatch):
    install_fake_rgb_detector(
        monkeypatch,
        [[keypoint(5, 5), keypoint(15, 5)]],
    )
    detector = detection.RGBDChipDetector(
        detector_config(
            baseline_depth_max_mm=1000.0,
            # The rejected feature at full-resolution x=10 is deliberately
            # outside this depth-fitting ROI. Feature masking must still be
            # applied across the entire aligned RGB-D frame.
            mask_roi={"x_min": 20, "x_max": 40, "y_min": 0, "y_max": 40},
        )
    )
    monkeypatch.setattr(
        detector,
        "_depth_pdf",
        lambda diff: np.ones(diff.shape, dtype=np.float32),
    )
    monkeypatch.setattr(
        detector,
        "_rgb_saliency_pdf",
        lambda _image, shape: np.zeros(shape, dtype=np.float32),
    )
    reference_depth = np.full((40, 40), 900.0, dtype=np.float32)
    reference_depth[10, 10] = 1400.0
    current_depth = np.full((40, 40), 800.0, dtype=np.float32)

    detector.detect(
        reference_depth,
        current_depth,
        np.zeros((40, 40, 3), dtype=np.uint8),
    )

    assert [point.pt for point in FakeFitRGB.last_fit_keypoints] == [(15.0, 5.0)]
    assert detector.last_depth_diff_mm[10, 10] == 0.0
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[10, 10], [0, 0, 0])
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[10, 30], [0, 0, 255])


def test_rgb_gmm_and_overlay_use_only_retained_scaled_features(monkeypatch):
    install_fake_rgb_detector(
        monkeypatch,
        [[keypoint(10, 10), keypoint(30, 20)]],
    )
    detector = detection.RGBDChipDetector(detector_config())
    detector.rgb_baseline_feature_coords = np.asarray([[10, 10]], dtype=np.float32)
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    pdf = detector._rgb_pdf(image, (80, 100))

    assert [point.pt for point in FakeFitRGB.last_fit_keypoints] == [(30.0, 20.0)]
    assert pdf.shape == (80, 100)
    # rgb_fit_scale=0.5 maps fit-image (30, 20) to full-image (60, 40).
    np.testing.assert_array_equal(
        detector.last_rgb_sift_overlay[40, 60],
        [0, 0, 255],
    )


def test_rgb_callback_is_cleared_after_reference_average(monkeypatch):
    grabber = object.__new__(detection.RGBDFrameGrabber)
    grabber.latest_rgb_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    grabber._depth_samples = []
    grabber._collect_depth = False
    grabber._rgb_frame_callback = None
    callback_frames = []

    def fake_spin_once(_node, timeout_sec=0.1):
        grabber._depth_samples.append(np.full((2, 2), 100.0, dtype=np.float32))
        if grabber._rgb_frame_callback is not None:
            grabber._rgb_frame_callback(grabber.latest_rgb_bgr.copy())

    monkeypatch.setattr(detection.rclpy, "spin_once", fake_spin_once)
    averaged = grabber.average_depth(
        2,
        rgb_frame_callback=lambda frame: callback_frames.append(frame),
    )

    np.testing.assert_array_equal(averaged, np.full((2, 2), 100.0))
    assert len(callback_frames) == 3  # Latest frame plus two frames during collection.
    assert grabber._rgb_frame_callback is None

    grabber.average_depth(1)
    assert len(callback_frames) == 3


def test_bundled_configs_match_feature_rejection_defaults():
    for path in (SCRIPTS_DIR / "config").glob("blow_from_mesh*.yaml"):
        with path.open(encoding="utf-8") as stream:
            detector_cfg = yaml.safe_load(stream)["chip_detector"]
        assert detector_cfg["rgb_num_features"] == 0
        assert detector_cfg["rgb_sift_contrast"] == 0.02
        assert detector_cfg["rgb_sift_edge"] == 12
        assert detector_cfg["rgb_include_outliers"] is True
        assert detector_cfg["rgb_ignore_feature_radius"] == 6.0
        assert detector_cfg["baseline_depth_max_mm"] == 1000.0
        baseline_cfg = detector_cfg["baseline"]
        assert baseline_cfg["mode"] in {"capture", "load"}
        assert baseline_cfg["save_directory"] == "data/blow_from_mesh_baselines"
        if baseline_cfg["load_path"] is not None:
            assert isinstance(baseline_cfg["load_path"], str)
            assert baseline_cfg["load_path"]


def test_baseline_archive_round_trip_and_timestamp_history(tmp_path):
    source = detection.RGBDChipDetector(detector_config())
    source.rgb_baseline_feature_coords = np.asarray(
        [[1.5, 2.5], [3.5, 4.5]], dtype=np.float32
    )
    source.rgb_baseline_frame_count = 7
    depth = np.asarray([[0.0, np.nan], [100.0, 200.0]], dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)

    first_path = source.save_baseline(depth, rgb, str(tmp_path), "doosan", "m0609")
    second_path = source.save_baseline(depth, rgb, str(tmp_path), "doosan", "m0609")

    assert first_path != second_path
    assert Path(first_path).is_file()
    assert Path(second_path).is_file()
    assert Path(first_path).name.startswith("baseline_doosan_m0609_")

    loaded = detection.RGBDChipDetector(detector_config())
    loaded_depth = loaded.load_baseline(first_path, depth.copy(), rgb.copy())

    np.testing.assert_array_equal(loaded_depth, depth)
    np.testing.assert_array_equal(
        loaded.rgb_baseline_feature_coords,
        source.rgb_baseline_feature_coords,
    )
    assert loaded.rgb_baseline_frame_count == 7
    assert loaded.last_baseline_metadata["schema_version"] == 1
    assert loaded.last_baseline_metadata["depth_shape"] == [2, 2]
    assert loaded.last_baseline_metadata["rgb_shape"] == [4, 6]
    assert loaded.last_baseline_metadata["rgb_fit_shape"] == [2, 3]


def test_baseline_paths_support_scripts_relative_and_absolute(monkeypatch, tmp_path):
    monkeypatch.setattr(detection, "SCRIPT_DIR", str(tmp_path))

    assert detection.resolve_baseline_path("data/baseline.npz") == str(
        tmp_path / "data" / "baseline.npz"
    )
    absolute = tmp_path / "chosen.npz"
    assert detection.resolve_baseline_path(str(absolute)) == str(absolute)


def write_baseline_archive(path, metadata, depth=None, coords=None, frame_count=1):
    if depth is None:
        depth = np.ones((4, 6), dtype=np.float32)
    if coords is None:
        coords = np.empty((0, 2), dtype=np.float32)
    np.savez_compressed(
        path,
        reference_depth_mm=depth,
        rgb_feature_coords=coords,
        rgb_baseline_frame_count=np.asarray(frame_count, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def baseline_metadata(**overrides):
    metadata = {
        "schema_version": 1,
        "created_utc": "2026-08-09T00:00:00.000000Z",
        "depth_shape": [4, 6],
        "rgb_shape": [4, 6],
        "rgb_fit_shape": [2, 3],
        "sift_settings": {
            "rgb_fit_scale": 0.5,
            "rgb_num_features": 0,
            "rgb_sift_contrast": 0.02,
            "rgb_sift_edge": 12.0,
        },
        "robot_type": "doosan",
        "robot_model": "m0609",
    }
    metadata.update(overrides)
    return metadata


def test_baseline_load_rejects_missing_corrupt_and_wrong_schema(tmp_path):
    detector = detection.RGBDChipDetector(detector_config())
    depth = np.ones((4, 6), dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)

    with np.testing.assert_raises_regex(RuntimeError, "does not exist"):
        detector.load_baseline(str(tmp_path / "missing.npz"), depth, rgb)

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an npz archive")
    with np.testing.assert_raises_regex(RuntimeError, "Could not read"):
        detector.load_baseline(str(corrupt), depth, rgb)

    wrong_schema = tmp_path / "wrong_schema.npz"
    write_baseline_archive(wrong_schema, baseline_metadata(schema_version=99))
    with np.testing.assert_raises_regex(RuntimeError, "uses schema 99"):
        detector.load_baseline(str(wrong_schema), depth, rgb)


def test_baseline_load_rejects_shape_settings_and_malformed_features(tmp_path):
    detector = detection.RGBDChipDetector(detector_config())
    depth = np.ones((4, 6), dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)

    wrong_shape = tmp_path / "wrong_shape.npz"
    write_baseline_archive(wrong_shape, baseline_metadata(depth_shape=[2, 2]))
    with np.testing.assert_raises_regex(RuntimeError, "inconsistent depth metadata"):
        detector.load_baseline(str(wrong_shape), depth, rgb)

    incompatible_dimensions = tmp_path / "incompatible_dimensions.npz"
    write_baseline_archive(incompatible_dimensions, baseline_metadata())
    with np.testing.assert_raises_regex(RuntimeError, "frame dimensions are incompatible"):
        detector.load_baseline(
            str(incompatible_dimensions),
            np.ones((5, 6), dtype=np.float32),
            rgb,
        )

    wrong_settings = tmp_path / "wrong_settings.npz"
    metadata = baseline_metadata()
    metadata["sift_settings"]["rgb_fit_scale"] = 1.0
    write_baseline_archive(wrong_settings, metadata)
    with np.testing.assert_raises_regex(RuntimeError, "SIFT settings are incompatible"):
        detector.load_baseline(str(wrong_settings), depth, rgb)

    malformed = tmp_path / "malformed_features.npz"
    write_baseline_archive(
        malformed,
        baseline_metadata(),
        coords=np.ones((3,), dtype=np.float32),
    )
    with np.testing.assert_raises_regex(RuntimeError, r"numeric \(N, 2\) array"):
        detector.load_baseline(str(malformed), depth, rgb)

    malformed_metadata = tmp_path / "malformed_metadata.npz"
    write_baseline_archive(
        malformed_metadata,
        baseline_metadata(rgb_shape=None),
    )
    with np.testing.assert_raises_regex(RuntimeError, "two positive integers"):
        detector.load_baseline(str(malformed_metadata), depth, rgb)

    wrong_robot = tmp_path / "wrong_robot.npz"
    write_baseline_archive(wrong_robot, baseline_metadata(robot_model="ur10e"))
    with np.testing.assert_raises_regex(RuntimeError, "robot profile is incompatible"):
        detector.load_baseline(
            str(wrong_robot), depth, rgb, robot_type="doosan", robot_model="m0609"
        )


class FakeBaselineGrabber:
    def __init__(self, depth, rgb, fail_capture=False):
        self.depth = depth
        self.latest_depth_mm = depth.copy()
        self.latest_rgb_bgr = rgb.copy()
        self.fail_capture = fail_capture
        self.average_calls = 0
        self.callback_calls = 0

    def average_depth(self, frames_to_average, timeout_sec, rgb_frame_callback=None):
        self.average_calls += 1
        if self.fail_capture:
            raise RuntimeError("capture failed")
        if rgb_frame_callback is not None:
            self.callback_calls += 1
            rgb_frame_callback(self.latest_rgb_bgr.copy())
        return self.depth.copy()


def test_prepare_baseline_capture_saves_only_after_success(monkeypatch, tmp_path):
    depth = np.ones((4, 6), dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    detector = detection.RGBDChipDetector(detector_config())
    grabber = FakeBaselineGrabber(depth, rgb)
    saved = []
    monkeypatch.setattr(detector, "add_rgb_baseline_frame", lambda _rgb: None)
    monkeypatch.setattr(
        detector,
        "save_baseline",
        lambda reference, baseline_rgb, **kwargs: saved.append(
            (reference.copy(), baseline_rgb.copy(), kwargs)
        ),
    )
    cfg = detector_config(
        baseline={"mode": "capture", "save_directory": str(tmp_path)},
        frames_to_average=3,
        reference_timeout_sec=2.0,
    )

    result = detection.prepare_detection_baseline(
        detector, grabber, cfg, rgb, depth, "doosan", "m0609"
    )

    np.testing.assert_array_equal(result, depth)
    assert grabber.average_calls == 1
    assert grabber.callback_calls == 1
    assert len(saved) == 1
    assert saved[0][2]["save_directory"] == str(tmp_path)

    failed_grabber = FakeBaselineGrabber(depth, rgb, fail_capture=True)
    saved.clear()
    with np.testing.assert_raises_regex(RuntimeError, "capture failed"):
        detection.prepare_detection_baseline(
            detector, failed_grabber, cfg, rgb, depth, "doosan", "m0609"
        )
    assert saved == []


def test_prepare_baseline_load_skips_capture(monkeypatch):
    depth = np.ones((4, 6), dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    detector = detection.RGBDChipDetector(detector_config())
    grabber = FakeBaselineGrabber(depth, rgb)
    loaded_paths = []
    monkeypatch.setattr(
        detector,
        "load_baseline",
        lambda path, current_depth_mm, current_rgb_bgr, **_kwargs: (
            loaded_paths.append(path) or depth.copy()
        ),
    )
    cfg = detector_config(
        baseline={"mode": "load", "load_path": "chosen.npz"}
    )

    result = detection.prepare_detection_baseline(
        detector, grabber, cfg, rgb, depth, "doosan", "m0609"
    )

    np.testing.assert_array_equal(result, depth)
    assert loaded_paths == ["chosen.npz"]
    assert grabber.average_calls == 0


def test_prepare_baseline_load_requires_explicit_path():
    depth = np.ones((4, 6), dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    detector = detection.RGBDChipDetector(detector_config())
    grabber = FakeBaselineGrabber(depth, rgb)

    with np.testing.assert_raises_regex(RuntimeError, "load_path must name"):
        detection.prepare_detection_baseline(
            detector,
            grabber,
            detector_config(baseline={"mode": "load", "load_path": None}),
            rgb,
            depth,
            "doosan",
            "m0609",
        )
    assert grabber.average_calls == 0
