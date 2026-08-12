"""Tests for Doosan task-space velocity and acceleration configuration."""

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blow_from_mesh_helpers.motion import (
    format_doosan_task_rate,
    generate_cartesian_guide_trajectory,
    get_current_doosan_pose_safely,
    parse_doosan_task_rate,
    reinterpolate_spline_waypoints,
    repair_unreachable_spline_waypoints,
    reject_unreachable_spline_waypoints,
)
from blow_from_mesh_helpers import motion


def test_scalar_task_rate_remains_supported():
    assert parse_doosan_task_rate(400, "velocity") == 400.0
    assert format_doosan_task_rate(400.0, "mm/s", "deg/s") == "400 mm/s"


def test_initial_keypoint_interpolation_uses_straight_segments():
    poses = []
    for xyz in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]):
        pose = np.eye(4)
        pose[:3, 3] = xyz
        poses.append(pose)

    trajectory = generate_cartesian_guide_trajectory(poses, n_interp=4)
    positions = np.asarray([pose[:3, 3] for pose in trajectory])

    assert len(trajectory) == 9
    np.testing.assert_allclose(positions[:5, 1], 0.0)
    np.testing.assert_allclose(positions[4:, 0], 1.0)
    np.testing.assert_allclose(positions[4], [1.0, 0.0, 0.0])


def test_reachability_reinterpolation_does_not_curve_past_line_segments():
    poses = []
    for xyz in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]):
        pose = np.eye(4)
        pose[:3, 3] = xyz
        poses.append(pose)

    trajectory = reinterpolate_spline_waypoints(poses, waypoint_count=7)
    positions = np.asarray([pose[:3, 3] for pose in trajectory])

    on_first_segment = np.isclose(positions[:, 1], 0.0)
    on_second_segment = np.isclose(positions[:, 0], 1.0)
    assert np.all(on_first_segment | on_second_segment)


def test_linear_angular_task_rate_pair():
    parsed = parse_doosan_task_rate([400, 60], "velocity")

    assert parsed == [400.0, 60.0]
    assert (
        format_doosan_task_rate(parsed, "mm/s", "deg/s")
        == "[400 mm/s, 60 deg/s]"
    )


@pytest.mark.parametrize("value", ([400], [400, 60, 10], [400, 0], [400, float("nan")]))
def test_invalid_task_rate_pair_is_rejected(value):
    with pytest.raises(ValueError):
        parse_doosan_task_rate(value, "velocity")


def test_spline_reachability_checks_every_waypoint_and_removes_only_failures(
    monkeypatch, capsys
):
    poses = []
    for index in range(7):
        pose = np.eye(4)
        pose[0, 3] = float(index)
        poses.append(pose)
    results = iter([True, True, False, True, True, True, False])
    checked = []

    def reachable(pose):
        checked.append(int(pose[0, 3]))
        return next(results)

    monkeypatch.setattr(motion, "check_reachable", reachable)

    retained = reject_unreachable_spline_waypoints(poses)

    assert checked == list(range(7))
    assert [int(pose[0, 3]) for pose in retained] == [0, 1, 3, 4, 5]
    report = capsys.readouterr().out
    assert "waypoint 2" in report
    assert "reason=rejected by reachability" in report


def test_spline_reachability_requires_two_total_waypoints(monkeypatch):
    poses = [np.eye(4) for _ in range(3)]
    results = iter([True, False, False])
    monkeypatch.setattr(motion, "check_reachable", lambda _pose: next(results))

    with pytest.raises(RuntimeError, match="fewer than two reachable"):
        reject_unreachable_spline_waypoints(poses)


def test_doosan_spline_moveit_ik_rejects_unreachable_waypoints(
    monkeypatch, capsys
):
    poses = []
    for index in range(5):
        pose = np.eye(4)
        pose[0, 3] = index * 0.01
        poses.append(pose)
    error_codes = iter([1, -31, 1, 1, 1])
    backend = motion.DoosanSplineMotionBackend()
    monkeypatch.setattr(backend, "_get_ik_client", lambda _name: object())
    monkeypatch.setattr(
        motion,
        "call_service_sync",
        lambda *_args, **_kwargs: type(
            "Response",
            (),
            {
                "error_code": type(
                    "ErrorCode", (), {"val": next(error_codes)}
                )()
            },
        )(),
    )
    monkeypatch.setitem(
        motion.CONFIG["execution"],
        "continuous_spline",
        {
            "moveit_ik_preflight": True,
            "ik_service": "compute_ik",
            "ik_timeout_sec": 0.2,
        },
    )

    reachable, reasons = backend.check_spline_waypoint_kinematics(poses)

    assert reachable == [True, False, True, True, True]
    assert reasons == {1: "rejected by IK: NO_IK_SOLUTION (-31)"}
    assert "all 5 controller spline waypoint(s)" in capsys.readouterr().out


def test_reachability_repair_removes_failures_and_reinterpolates(monkeypatch, capsys):
    poses = []
    for index in range(5):
        pose = np.eye(4)
        pose[0, 3] = index * 0.01
        poses.append(pose)

    monkeypatch.setattr(
        motion,
        "geometric_reachability_failure_reason",
        lambda _pose: None,
    )
    calls = []

    def check_kinematics(candidates):
        calls.append(candidates)
        if len(calls) == 1:
            return [True, True, False, True, True], {2: "rejected by IK: test"}
        return [True] * len(candidates), {}

    repaired = repair_unreachable_spline_waypoints(
        poses,
        max_iterations=3,
        kinematic_check=check_kinematics,
    )

    assert len(calls) == 2
    assert len(repaired) == len(poses)
    output = capsys.readouterr().out
    assert "waypoint 2" in output
    assert "Removing only the 1 rejected waypoint(s)" in output
    assert "passed reachability after 2 iteration(s)" in output


def test_reachability_repair_aborts_at_iteration_limit(monkeypatch):
    poses = []
    for index in range(4):
        pose = np.eye(4)
        pose[0, 3] = index * 0.01
        poses.append(pose)

    monkeypatch.setattr(
        motion,
        "geometric_reachability_failure_reason",
        lambda _pose: None,
    )

    def reject_one(candidates):
        results = [True] * len(candidates)
        results[1] = False
        return results, {1: "rejected by IK: test"}

    with pytest.raises(RuntimeError, match="reachability_max_iterations=2"):
        repair_unreachable_spline_waypoints(
            poses,
            max_iterations=2,
            kinematic_check=reject_one,
        )


def test_empty_doosan_pose_feedback_is_nonfatal(monkeypatch, capsys):
    def empty_feedback():
        raise IndexError("list index out of range")

    monkeypatch.setattr(motion, "get_current_posx", empty_feedback)

    assert get_current_doosan_pose_safely("init") is None
    assert "temporarily unavailable" in capsys.readouterr().out
