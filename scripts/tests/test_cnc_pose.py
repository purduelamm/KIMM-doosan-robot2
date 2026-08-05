"""Tests for CNC pose configuration and native-mesh placement."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blow_from_mesh_helpers.config import world_to_cnc_from_config
from blow_from_mesh_helpers.math_utils import make_mesh_world_transform


@pytest.mark.parametrize(
    ("pose_config", "expected_rotation"),
    [
        (
            {
                "euler_xyz": [0.0, 0.0, 90.0],
                "degrees": True,
                "translation": [1.0, 2.0, 3.0],
            },
            Rotation.from_euler("z", 90.0, degrees=True).as_matrix(),
        ),
        (
            {
                "quaternion_xyzw": [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
                "translation": [1.0, 2.0, 3.0],
            },
            Rotation.from_euler("z", 90.0, degrees=True).as_matrix(),
        ),
        (
            {
                "matrix": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                "translation": [1.0, 2.0, 3.0],
            },
            Rotation.from_euler("z", 90.0, degrees=True).as_matrix(),
        ),
    ],
)
def test_world_to_cnc_mapping_supports_existing_rotation_formats(
    pose_config, expected_rotation
):
    rotation, translation = world_to_cnc_from_config(pose_config)

    np.testing.assert_allclose(rotation.as_matrix(), expected_rotation, atol=1e-12)
    np.testing.assert_allclose(translation, [1.0, 2.0, 3.0])


def test_legacy_world_to_cnc_list_uses_identity_rotation():
    rotation, translation = world_to_cnc_from_config([1.0, 2.0, 3.0])

    np.testing.assert_allclose(rotation.as_matrix(), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(translation, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "pose_config",
    [
        [1.0, 2.0],
        {"euler_xyz": [0.0, 0.0, 0.0], "translation": [1.0, 2.0]},
        {"translation": [1.0, 2.0, 3.0]},
    ],
)
def test_invalid_world_to_cnc_pose_has_configuration_specific_error(pose_config):
    with pytest.raises(ValueError, match=r"mesh\.world_to_cnc"):
        world_to_cnc_from_config(pose_config)


def test_mesh_transform_applies_local_offset_before_cnc_pose():
    rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    transform = make_mesh_world_transform(
        rotation,
        t_w_cnc_m=np.array([10.0, 20.0, 30.0]),
        mesh_local_offset_m=np.array([0.5, 0.0, 0.0]),
        mesh_units_per_metre=1000.0,
    )
    native_mesh_point = np.array([1000.0, 0.0, 0.0, 1.0])

    transformed_point = transform @ native_mesh_point

    np.testing.assert_allclose(transformed_point[:3], [10000.0, 21500.0, 30000.0])


def test_identity_mesh_rotation_matches_previous_translation_behavior():
    native_mesh_point = np.array([125.0, -250.0, 375.0, 1.0])
    translation = np.array([1.0, 2.0, 3.0])
    local_offset = np.array([0.1, -0.2, 0.3])
    units_per_metre = 1000.0
    transform = make_mesh_world_transform(
        np.eye(3), translation, local_offset, units_per_metre
    )

    transformed_point = transform @ native_mesh_point
    previous_result = native_mesh_point[:3] + (
        translation + local_offset
    ) * units_per_metre

    np.testing.assert_allclose(transformed_point[:3], previous_result)


def test_bundled_configs_use_identity_cnc_rotation_and_keep_translations():
    expected_translations = {
        "blow_from_mesh_doosan.yaml": [-0.60376, 0.904021, 0.8],
        "blow_from_mesh_ur.yaml": [0.0, 0.0, 0.0],
        "blow_from_mesh.yaml": [0.0, 0.0, 0.0],
    }

    for filename, expected_translation in expected_translations.items():
        with (SCRIPTS_DIR / "config" / filename).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        rotation, translation = world_to_cnc_from_config(
            config["mesh"]["world_to_cnc"]
        )

        np.testing.assert_allclose(rotation.as_matrix(), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(translation, expected_translation)
