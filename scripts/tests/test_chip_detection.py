"""Tests for clean-reference RGB SIFT feature rejection."""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
import pytest
from types import SimpleNamespace


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
        coords = np.asarray([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
        return (object() if len(keypoints) >= 2 else None), coords

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
    def fake_extractor(cfg):
        backbone = cfg.get("rgb_feature_backbone", "sift")
        return SimpleNamespace(
            backbone=backbone,
            device="cpu",
            settings={"rgb_feature_backbone": backbone},
            detect=lambda image: FakeFitRGB(image)._sift_detector(),
        )
    monkeypatch.setattr(detection, "create_rgb_feature_extractor", fake_extractor)


@pytest.mark.parametrize("backbone", ["sift", "superpoint"])
def test_baseline_rejects_only_all_30_frame_features(monkeypatch, backbone):
    batches = []
    for frame in range(30):
        points = [keypoint(10 + (frame % 3) * 0.25, 10)]
        if frame != 15:
            points.append(keypoint(30, 30))
        if frame == 0:
            points.append(keypoint(50, 50))
        batches.append(points)
    batches.append([keypoint(10, 10), keypoint(30, 30), keypoint(50, 50)])
    install_fake_rgb_detector(monkeypatch, batches)
    detector = detection.RGBDChipDetector(detector_config(rgb_feature_backbone=backbone))
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    for frame in range(29):
        detector.add_rgb_baseline_frame(image)
        assert detector.rgb_baseline_feature_coords.shape == (0, 2)
        assert detector.rgb_baseline_frame_count == frame + 1
    detector.add_rgb_baseline_frame(image)
    np.testing.assert_allclose(detector.rgb_baseline_feature_coords, [[10.25, 10]])
    detector.add_rgb_baseline_frame(image)  # A completed baseline stays fixed.
    assert detector.rgb_baseline_frame_count == 30
    detector._rgb_pdf(image, image.shape[:2])
    assert [kp.pt for kp in FakeFitRGB.last_fit_keypoints] == [(30.0, 30.0), (50.0, 50.0)]
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[20, 20], [0, 0, 255])
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[60, 60], [0, 255, 0])
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[100, 100], [0, 255, 0])
    detector.reset_rgb_baseline()
    assert detector.rgb_baseline_frame_count == 0
    assert detector.rgb_baseline_feature_coords.shape == (0, 2)


def complete_baseline(detector, coords):
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    for _ in range(detector.rgb_baseline_stack.target_frames):
        detector.rgb_baseline_stack.add_frame(coords)
    detector.rgb_baseline_feature_coords = detector.rgb_baseline_stack.coordinates
    detector.rgb_baseline_frame_count = detector.rgb_baseline_stack.frame_count


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


@pytest.mark.parametrize("backbone", ["sift", "superpoint"])
def test_all_rejected_features_return_empty_rgb_pdf(monkeypatch, backbone):
    install_fake_rgb_detector(monkeypatch, [[keypoint(10, 10)]])
    detector = detection.RGBDChipDetector(detector_config(rgb_feature_backbone=backbone))
    detector.rgb_baseline_feature_coords = np.asarray([[10, 10]], dtype=np.float32)
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    pdf = detector._rgb_pdf(image, (40, 40))

    np.testing.assert_array_equal(pdf, np.zeros((40, 40), dtype=np.float32))
    assert FakeFitRGB.last_fit_keypoints == []
    assert detector.last_rgb_sift_overlay.shape == image.shape
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[20, 20], [0, 0, 255])


@pytest.mark.parametrize("backbone", ["sift", "superpoint"])
def test_empty_rgb_pdf_does_not_abort_nonempty_depth_detection(monkeypatch, backbone):
    install_fake_rgb_detector(monkeypatch, [[keypoint(10, 10)]])
    detector = detection.RGBDChipDetector(detector_config(rgb_feature_backbone=backbone))
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


@pytest.mark.parametrize("backbone", ["sift", "superpoint"])
def test_rgb_feature_is_rejected_where_baseline_depth_forces_zero(monkeypatch, backbone):
    install_fake_rgb_detector(
        monkeypatch,
        [[keypoint(5, 5), keypoint(15, 5)]],
    )
    detector = detection.RGBDChipDetector(
        detector_config(
            rgb_feature_backbone=backbone,
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
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[10, 10], [0, 0, 255])
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[10, 30], [0, 255, 0])


@pytest.mark.parametrize("backbone", ["sift", "superpoint"])
def test_rgb_gmm_and_overlay_use_only_retained_scaled_features(monkeypatch, backbone):
    install_fake_rgb_detector(
        monkeypatch,
        [[keypoint(10, 10), keypoint(30, 20)]],
    )
    detector = detection.RGBDChipDetector(detector_config(rgb_feature_backbone=backbone))
    detector.rgb_baseline_feature_coords = np.asarray([[10, 10]], dtype=np.float32)
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    pdf = detector._rgb_pdf(image, (80, 100))

    assert [point.pt for point in FakeFitRGB.last_fit_keypoints] == [(30.0, 20.0)]
    assert pdf.shape == (80, 100)
    np.testing.assert_array_equal(detector.last_rgb_sift_overlay[20, 20], [0, 0, 255])
    # rgb_fit_scale=0.5 maps fit-image (30, 20) to full-image (60, 40).
    np.testing.assert_array_equal(
        detector.last_rgb_sift_overlay[40, 60],
        [0, 255, 0],
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
        assert detector_cfg["rgb_baseline_match_radius"] == 3.0
        assert detector_cfg["frames_to_average"] == 30
        assert detector_cfg["rgb_feature_backbone"] == "sift"
        assert detector_cfg["rgb_feature_device"] == "auto"
        assert detector_cfg["rgb_superpoint_max_keypoints"] == 1024
        assert detector_cfg["rgb_superpoint_detection_threshold"] == 0.0005
        assert detector_cfg["rgb_superpoint_nms_radius"] == 4
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
    complete_baseline(source, source.rgb_baseline_feature_coords)
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
    assert loaded.rgb_baseline_frame_count == 30
    assert loaded.last_baseline_metadata["schema_version"] == 3
    assert loaded.last_baseline_metadata["feature_settings"]["rgb_feature_backbone"] == "sift"
    assert loaded.last_baseline_metadata["depth_shape"] == [2, 2]
    assert loaded.last_baseline_metadata["rgb_shape"] == [4, 6]
    assert loaded.last_baseline_metadata["rgb_fit_shape"] == [2, 3]


@pytest.mark.parametrize("count", [0, 29])
def test_incomplete_baseline_cannot_be_saved(tmp_path, count):
    detector = detection.RGBDChipDetector(detector_config())
    for _ in range(count):
        detector.rgb_baseline_stack.add_frame(np.asarray([[1, 1]], dtype=np.float32))
    detector.rgb_baseline_frame_count = count
    with pytest.raises(ValueError, match="incomplete RGB baseline"):
        detector.save_baseline(np.ones((4, 6)), np.zeros((4, 6, 3)), str(tmp_path), "doosan", "m0609")
    assert list(tmp_path.iterdir()) == []


def test_baseline_paths_support_scripts_relative_and_absolute(monkeypatch, tmp_path):
    monkeypatch.setattr(detection, "SCRIPT_DIR", str(tmp_path))

    assert detection.resolve_baseline_path("data/baseline.npz") == str(
        tmp_path / "data" / "baseline.npz"
    )
    absolute = tmp_path / "chosen.npz"
    assert detection.resolve_baseline_path(str(absolute)) == str(absolute)


def write_baseline_archive(path, metadata, depth=None, coords=None, frame_count=30):
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
        "schema_version": 3,
        "created_utc": "2026-08-09T00:00:00.000000Z",
        "depth_shape": [4, 6],
        "rgb_shape": [4, 6],
        "rgb_fit_shape": [2, 3],
        "feature_settings": {
            "rgb_feature_backbone": "sift",
            "rgb_fit_scale": 0.5,
            "rgb_num_features": 0,
            "rgb_sift_contrast": 0.02,
            "rgb_sift_edge": 12.0,
        },
        "baseline_stacking": {
            "frames_to_average": 30,
            "rgb_baseline_match_radius": 3.0,
            "required_presence": "all",
        },
        "robot_type": "doosan",
        "robot_model": "m0609",
    }
    metadata.update(overrides)
    return metadata


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_baseline_requires_recapture(tmp_path, schema):
    detector = detection.RGBDChipDetector(detector_config())
    archive = tmp_path / "legacy.npz"
    write_baseline_archive(archive, baseline_metadata(schema_version=schema))
    with pytest.raises(RuntimeError, match="Recapture.*persistence history"):
        detector.load_baseline(str(archive), np.ones((4, 6)), np.zeros((4, 6, 3)))


@pytest.mark.parametrize("backbone", ["sift", "superpoint"])
@pytest.mark.parametrize("empty", [False, True])
def test_v3_backbone_baseline_roundtrip_and_mismatch(monkeypatch, tmp_path, backbone, empty):
    # Skip model initialization; exercise real configuration and metadata for both backbones.
    from rgb_features import FEATURE_DEFAULTS, SUPERPOINT_MODEL_ID
    original_factory = detection.create_rgb_feature_extractor

    def without_weights(cfg):
        if cfg.get("rgb_feature_backbone", "sift") == "sift":
            return original_factory(cfg)
        values = {**FEATURE_DEFAULTS, **cfg}
        settings = {
            key: values[key] for key in (
                "rgb_feature_backbone", "rgb_superpoint_max_keypoints",
                "rgb_superpoint_detection_threshold", "rgb_superpoint_nms_radius",
            )
        }
        settings["model_identity"] = SUPERPOINT_MODEL_ID
        return SimpleNamespace(backbone="superpoint", device="cpu", settings=settings)

    monkeypatch.setattr(detection, "create_rgb_feature_extractor", without_weights)
    cfg = detector_config(rgb_feature_backbone=backbone)
    source = detection.RGBDChipDetector(cfg)
    coords = np.empty((0, 2), dtype=np.float32) if empty else np.asarray([[1, 2]], dtype=np.float32)
    complete_baseline(source, coords)
    depth = np.ones((16, 16), dtype=np.float32)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    archive = source.save_baseline(depth, rgb, str(tmp_path), "doosan", "m0609")
    loaded = detection.RGBDChipDetector(cfg)
    np.testing.assert_array_equal(loaded.load_baseline(archive, depth, rgb), depth)
    np.testing.assert_array_equal(loaded.rgb_baseline_feature_coords, coords)
    assert loaded.rgb_baseline_frame_count == 30
    assert loaded.rgb_baseline_stack.complete
    assert loaded.last_baseline_metadata["feature_settings"] == source._baseline_feature_settings()

    other = "superpoint" if backbone == "sift" else "sift"
    incompatible = detection.RGBDChipDetector(detector_config(rgb_feature_backbone=other))
    with pytest.raises(RuntimeError, match="Recapture"):
        incompatible.load_baseline(archive, depth, rgb)

    setting = "rgb_sift_contrast" if backbone == "sift" else "rgb_superpoint_detection_threshold"
    changed = detection.RGBDChipDetector({**cfg, setting: 0.1})
    with pytest.raises(RuntimeError, match="Recapture"):
        changed.load_baseline(archive, depth, rgb)

    # Device and inactive-backbone settings do not invalidate saved coordinates.
    inactive = "rgb_superpoint_detection_threshold" if backbone == "sift" else "rgb_sift_contrast"
    compatible = detection.RGBDChipDetector({**cfg, "rgb_feature_device": "cpu", inactive: 0.1})
    compatible.load_baseline(archive, depth, rgb)

    if backbone == "superpoint":
        legacy = tmp_path / "legacy.npz"
        write_baseline_archive(legacy, baseline_metadata(schema_version=2))
        with pytest.raises(RuntimeError, match="Recapture.*persistence history"):
            loaded.load_baseline(str(legacy), np.ones((4, 6)), np.zeros((4, 6, 3)))
        with np.load(archive, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata_json"].item())
        metadata["feature_settings"]["model_identity"] = "different-weights"
        different_model = tmp_path / "different-model.npz"
        write_baseline_archive(different_model, metadata, depth=depth)
        with pytest.raises(RuntimeError, match="Recapture"):
            loaded.load_baseline(str(different_model), depth, rgb)


@pytest.mark.parametrize("backbone", ["sift", "superpoint"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_real_gmm_sparse_and_finite_outputs(monkeypatch, backbone, count):
    from FitGMM import FitRGB_SIFT
    install_fake_rgb_detector(monkeypatch, [[keypoint(3, 4), keypoint(10, 11), keypoint(14, 6)][:count]])
    monkeypatch.setattr(detection, "get_detector_classes", lambda: (object, FitRGB_SIFT))
    detector = detection.RGBDChipDetector(detector_config(rgb_feature_backbone=backbone))
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    pdf = detector._rgb_pdf(image, image.shape[:2])
    assert pdf.shape == (40, 40)
    assert np.isfinite(pdf).all()
    assert (pdf >= 0).all()
    if count < 2:
        assert not pdf.any()
    else:
        assert pdf.any()
    if count:
        np.testing.assert_array_equal(detector.last_rgb_sift_overlay[8, 6], [0, 255, 0])


def test_extraction_failure_clears_stale_rgb_outputs(monkeypatch):
    detector = detection.RGBDChipDetector(detector_config())
    detector.last_rgb_pdf = np.ones((40, 40), dtype=np.float32)
    detector.last_rgb_sift_overlay = np.ones((40, 40, 3), dtype=np.uint8)

    def fail(_image):
        raise RuntimeError("extraction failed")
    monkeypatch.setattr(detector.rgb_feature_extractor, "detect", fail)
    with pytest.raises(RuntimeError, match="extraction failed"):
        detector._rgb_pdf(np.zeros((40, 40, 3), dtype=np.uint8), (40, 40))
    assert detector.last_rgb_pdf is None
    assert detector.last_rgb_sift_overlay is None


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
    metadata["feature_settings"]["rgb_fit_scale"] = 1.0
    write_baseline_archive(wrong_settings, metadata)
    with np.testing.assert_raises_regex(RuntimeError, "RGB feature settings are incompatible"):
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


@pytest.mark.parametrize("change", ["radius", "target", "policy", "count", "missing"])
def test_baseline_load_rejects_incompatible_stacking(tmp_path, change):
    detector = detection.RGBDChipDetector(detector_config())
    metadata = baseline_metadata()
    count = 30
    if change == "radius":
        metadata["baseline_stacking"]["rgb_baseline_match_radius"] = 4.0
    elif change == "target":
        metadata["baseline_stacking"]["frames_to_average"] = 31
    elif change == "policy":
        metadata["baseline_stacking"]["required_presence"] = "any"
    elif change == "count":
        count = 29
    else:
        del metadata["baseline_stacking"]
    archive = tmp_path / "incompatible.npz"
    write_baseline_archive(archive, metadata, frame_count=count)
    with pytest.raises(RuntimeError, match="stacking settings or frame count.*Recapture"):
        detector.load_baseline(str(archive), np.ones((4, 6)), np.zeros((4, 6, 3)))


def frame_grabber():
    grabber = object.__new__(detection.RGBDFrameGrabber)
    grabber.bridge = SimpleNamespace(imgmsg_to_cv2=lambda image, **kwargs: image)
    grabber.depth_unit = "mm"
    grabber.latest_rgb_bgr = np.full((4, 4, 3), 99, dtype=np.uint8)
    grabber._depth_samples = []
    grabber._depth_frame_target = 0
    grabber._collect_depth = False
    grabber._rgb_frame_callback = None
    return grabber


@pytest.mark.parametrize("rgb_first", [False, True])
def test_targeted_capture_uses_fresh_frames_and_waits_for_both_streams(monkeypatch, rgb_first):
    grabber = frame_grabber()
    rgb = [("rgb", i) for i in range(1, 5)]
    depth = [("depth", i) for i in range(1, 5)]
    events = iter(rgb + depth[:3] if rgb_first else depth + rgb[:3])

    def spin(_node, timeout_sec):
        kind, value = next(events)
        if kind == "rgb":
            grabber._rgb_cb(np.full((4, 4, 3), value, dtype=np.uint8))
        else:
            grabber._depth_cb(np.full((4, 4), value, dtype=np.uint16))
    monkeypatch.setattr(detection.rclpy, "spin_once", spin)
    received = []
    averaged = grabber.average_depth(
        3, rgb_frame_callback=lambda frame: received.append(int(frame[0, 0, 0])), rgb_frames_to_collect=3,
    )
    assert received == [1, 2, 3]  # Cached 99 and extra frame 4 never enter the stack.
    np.testing.assert_array_equal(averaged, np.full((4, 4), 2, dtype=np.float32))
    assert len(grabber._depth_samples) == 3
    assert not grabber._collect_depth
    assert grabber._rgb_frame_callback is None
    grabber._rgb_cb(np.full((4, 4, 3), 42, dtype=np.uint8))
    assert received == [1, 2, 3]


def test_frame_grabber_timeout_reports_both_counts_and_discards_samples(monkeypatch):
    grabber = frame_grabber()
    clock = [0.0]

    def spin(_node, timeout_sec):
        grabber._depth_cb(np.ones((4, 4), dtype=np.uint16))
        grabber._rgb_cb(np.ones((4, 4, 3), dtype=np.uint8))
        clock[0] = 1.0
    monkeypatch.setattr(detection.rclpy, "spin_once", spin)
    monkeypatch.setattr(detection.time, "monotonic", lambda: clock[0])
    with pytest.raises(RuntimeError, match="depth=1/3, RGB=1/3"):
        grabber.average_depth(3, timeout_sec=0.5, rgb_frame_callback=lambda frame: None, rgb_frames_to_collect=3)
    assert not grabber._collect_depth
    assert grabber._rgb_frame_callback is None
    assert grabber._depth_samples == []


@pytest.mark.parametrize("with_callback", [False, True])
def test_frame_grabber_preserves_depth_only_callers(monkeypatch, with_callback):
    grabber = frame_grabber()
    values = iter([1, 3])
    monkeypatch.setattr(
        detection.rclpy, "spin_once",
        lambda _node, timeout_sec: grabber._depth_cb(np.full((4, 4), next(values), dtype=np.uint16)),
    )
    received = []
    callback = (lambda frame: received.append(int(frame[0, 0, 0]))) if with_callback else None
    averaged = grabber.average_depth(2, rgb_frame_callback=callback)
    np.testing.assert_array_equal(averaged, np.full((4, 4), 2, dtype=np.float32))
    assert received == ([99] if with_callback else [])
    assert grabber._rgb_frame_callback is None


class FakeBaselineGrabber:
    def __init__(self, depth, rgb, fail_capture=False):
        self.depth = depth
        self.latest_depth_mm = depth.copy()
        self.latest_rgb_bgr = rgb.copy()
        self.fail_capture = fail_capture
        self.average_calls = 0
        self.callback_calls = 0

    def average_depth(self, frames_to_average, timeout_sec, rgb_frame_callback=None, rgb_frames_to_collect=0):
        self.average_calls += 1
        if rgb_frame_callback is not None:
            for _ in range(rgb_frames_to_collect):
                self.callback_calls += 1
                rgb_frame_callback(self.latest_rgb_bgr.copy())
                if self.fail_capture:
                    raise RuntimeError("capture failed")
        return self.depth.copy()


def test_prepare_baseline_capture_saves_only_after_success(monkeypatch, tmp_path):
    depth = np.ones((4, 6), dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    detector = detection.RGBDChipDetector(detector_config(frames_to_average=3, reference_timeout_sec=2.0))
    grabber = FakeBaselineGrabber(depth, rgb)
    saved = []
    monkeypatch.setattr(detector.rgb_feature_extractor, "detect", lambda _rgb: [])
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
    assert grabber.callback_calls == 3
    assert len(saved) == 1
    assert saved[0][2]["save_directory"] == str(tmp_path)

    failed_grabber = FakeBaselineGrabber(depth, rgb, fail_capture=True)
    saved.clear()
    with np.testing.assert_raises_regex(RuntimeError, "capture failed"):
        detection.prepare_detection_baseline(
            detector, failed_grabber, cfg, rgb, depth, "doosan", "m0609"
        )
    assert saved == []
    assert detector.rgb_baseline_frame_count == 0
    assert detector.rgb_baseline_feature_coords.shape == (0, 2)


def test_prepare_baseline_publishes_features_only_after_both_streams(monkeypatch, tmp_path):
    depth = np.ones((4, 6), dtype=np.float32)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    cfg = detector_config(frames_to_average=3, baseline={"mode": "capture", "save_directory": str(tmp_path)})
    detector = detection.RGBDChipDetector(cfg)
    monkeypatch.setattr(detector.rgb_feature_extractor, "detect", lambda image: [keypoint(1, 1)])

    class RGBFirstGrabber(FakeBaselineGrabber):
        def average_depth(self, **kwargs):
            for _ in range(3):
                kwargs["rgb_frame_callback"](self.latest_rgb_bgr)
            assert detector.rgb_baseline_stack.complete
            assert detector.rgb_baseline_feature_coords.shape == (0, 2)
            with pytest.raises(ValueError, match="pending depth capture"):
                detector.save_baseline(self.depth, self.latest_rgb_bgr, str(tmp_path), "doosan", "m0609")
            assert list(tmp_path.iterdir()) == []
            return self.depth.copy()

    detection.prepare_detection_baseline(detector, RGBFirstGrabber(depth, rgb), cfg, rgb, depth, "doosan", "m0609")
    np.testing.assert_array_equal(detector.rgb_baseline_feature_coords, [[1, 1]])
    assert not detector._baseline_capture_in_progress


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
