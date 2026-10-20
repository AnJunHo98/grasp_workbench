"""Optional robot examples: state subscription/FK and endpoint-only IK contracts."""
import importlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from grasp_workbench.models.robot_reachability import load_robot_adapter
from grasp_workbench.models.robot_ffw import FfwSg2RobotAdapter, _UrdfKinematics


NAMES = tuple(f"arm_{arm}_joint{index}" for arm in ("l", "r") for index in range(1, 8)) + (
    "head_joint1", "head_joint2", "lift_joint")
Q15 = ("lift_joint", *NAMES[:14])


@pytest.fixture
def files(tmp_path):
    assets = tmp_path / "robot"
    (assets / "sources").mkdir(parents=True)
    (assets / "sources/ffw_sg2.urdf").write_text("""<robot name="example">
      <link name="base_link"/><link name="lift_link"/><link name="head_link1"/>
      <link name="head_link2"/><link name="end_effector_l_link"/><link name="end_effector_r_link"/>
      <joint name="lift_joint" type="prismatic"><parent link="base_link"/><child link="lift_link"/>
        <origin xyz="0 0 1"/><axis xyz="0 0 1"/></joint>
      <joint name="head_joint1" type="revolute"><parent link="lift_link"/><child link="head_link1"/>
        <axis xyz="0 1 0"/></joint>
      <joint name="head_joint2" type="revolute"><parent link="head_link1"/><child link="head_link2"/>
        <origin xyz=".1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="arm_l_joint1" type="revolute"><parent link="base_link"/><child link="end_effector_l_link"/>
        <origin xyz="0 .3 1" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="arm_r_joint1" type="revolute"><parent link="base_link"/><child link="end_effector_r_link"/>
        <origin xyz="0 -.3 1" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
    </robot>""")
    calibration = {"parent_frame": "base_link", "length_unit": "m", "transforms": {
        "camera_optical": {"parent_from_sensor": np.eye(4).tolist()}}}
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(calibration))
    config = {"backend": "ffw_sg2", "runtime_manifest": "runtime.json", "robot_asset_dir": "robot",
              "calibration": {"path": "calibration.yaml", "sensor_frame": "camera_optical"}}
    return tmp_path, config


def test_default_robot_modules_import_without_ffw_or_gpu_dependencies():
    script = """
import sys
import grasp_workbench.models.robot_reachability
import grasp_workbench.models.robot_ffw
assert not any(name in sys.modules for name in
 ('torch','curobo','whole_robot_system','robot_worker_runtime','motion_planning_system','object_pose_system'))
"""
    process = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def test_custom_plugin_receives_yaml_and_its_directory_without_builtin_robot_import(monkeypatch, tmp_path):
    received = []
    expected = object()
    monkeypatch.setitem(sys.modules, "custom_robot_plugin", SimpleNamespace(
        create=lambda config, **kwargs: received.append((config, kwargs)) or expected))
    path = tmp_path / "robot.yaml"
    path.write_text("backend: custom_robot_plugin:create\ncustom_setting: user_choice\n")
    assert load_robot_adapter(path) is expected
    assert received == [({"backend": "custom_robot_plugin:create", "custom_setting": "user_choice"},
                         {"config_dir": tmp_path})]


def test_builtin_factory_resolves_paths_from_robot_yaml(files):
    directory, config = files
    path = directory / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    adapter = load_robot_adapter(path)
    assert isinstance(adapter, FfwSg2RobotAdapter)
    assert adapter.asset_dir == directory / "robot"
    assert adapter.runtime_manifest == directory / "runtime.json"
    assert adapter._backend is None


def fake_state_stack(monkeypatch, message):
    class Subscriber:
        def __init__(self):
            self.message, self.closed = message, False

        def take_latest(self):
            value, self.message = self.message, None
            return value

        def close(self):
            self.closed = True

    subscriber, calls = Subscriber(), []
    monkeypatch.setitem(sys.modules, "robot_worker_runtime.transport.runtime", SimpleNamespace(
        EndpointFactory=lambda path: calls.append(path) or SimpleNamespace(
            open_subscriber=lambda channel: calls.append(channel) or subscriber)))
    monkeypatch.setitem(sys.modules, "whole_robot_system.robots.robotis_ai_worker.ffw_sg2.definition",
                        SimpleNamespace(FFW_SG2_Q17_JOINT_NAMES=NAMES, FFW_SG2_FREE_PRRTC_Q15_JOINT_NAMES=Q15))
    monkeypatch.setitem(sys.modules, "whole_robot_system.robots.robotis_ai_worker.ffw_sg2.codec",
                        SimpleNamespace(decode_state=lambda payload: payload))
    return subscriber, calls


def test_read_only_state_snapshot_q17_q15_mapping_hands_and_static_camera(monkeypatch, files):
    directory, config = files
    positions = tuple([0.] * 14 + [.1, .2, .3])
    state = SimpleNamespace(position=positions, sequence=19, timestamp_ns=456)
    subscriber, calls = fake_state_stack(monkeypatch, SimpleNamespace(payload=state))
    adapter = FfwSg2RobotAdapter(config, config_dir=directory)
    adapter.start()
    value = adapter.snapshot()
    assert calls == [directory / "runtime.json", "ffw_sg2.state"]
    assert value["joint_names"] == Q15 and value["joint_positions"] == (.3, *positions[:14])
    assert value["state_sequence"] == 19 and value["state_timestamp_ns"] == 456
    np.testing.assert_array_equal(value["base_from_camera"], np.eye(4))
    np.testing.assert_allclose(value["current_hands"]["left"][:3, 3], (0, .3, 1))
    np.testing.assert_allclose(value["current_hands"]["right"][:3, 3], (0, -.3, 1))
    canonical = value["canonical_gripper_from_tcp"]["left"]
    np.testing.assert_allclose(canonical[:3, 3], (0, 0, .137))
    np.testing.assert_allclose(canonical[:3, :3], ((0, 1, 0), (1, 0, 0), (0, 0, -1)))
    assert adapter.snapshot()["state_sequence"] == 19  # No inferred seed or freshness policy.
    assert adapter._backend is None
    adapter.close()
    assert subscriber.closed


def test_no_robot_feedback_returns_none_instead_of_zero_configuration(monkeypatch, files):
    directory, config = files
    subscriber, _ = fake_state_stack(monkeypatch, None)
    adapter = FfwSg2RobotAdapter(config, config_dir=directory)
    adapter.start()
    assert adapter.snapshot() is None
    adapter.close()
    assert subscriber.closed


def test_head_camera_calibration_is_composed_with_measured_head_and_lift_fk(monkeypatch, files):
    directory, config = files
    calibration_path = directory / "calibration.yaml"
    calibration = yaml.safe_load(calibration_path.read_text())
    calibration["parent_frame"] = "head_link2"
    offset = np.eye(4)
    offset[2, 3] = .05
    calibration["transforms"]["camera_optical"]["parent_from_sensor"] = offset.tolist()
    calibration_path.write_text(yaml.safe_dump(calibration))
    positions = tuple([0.] * 14 + [np.pi / 2, 0., .2])
    state = SimpleNamespace(position=positions, sequence=1, timestamp_ns=2)
    fake_state_stack(monkeypatch, SimpleNamespace(payload=state))
    adapter = FfwSg2RobotAdapter(config, config_dir=directory)
    adapter.start()
    camera = adapter.snapshot()["base_from_camera"]
    np.testing.assert_allclose(camera[:3, 3], (.05, 0., 1.1), atol=1e-8)
    adapter.close()


def test_missing_measured_joint_is_not_synthesized(files):
    directory, _ = files
    fk = _UrdfKinematics(directory / "robot/sources/ffw_sg2.urdf")
    with pytest.raises(KeyError):
        fk.base_from_link({}, "head_link2")


def fake_solver_modules(monkeypatch):
    def variants(pair):
        output = []
        for index in range(8):
            output.append(tuple(SimpleNamespace(rotation=pose.rotation,
                translation=(pose.translation[0] + (index + 1) * .005, *pose.translation[1:])) for pose in pair))
        return tuple(output)

    monkeypatch.setitem(sys.modules, "motion_planning_system.curobo.geometry", SimpleNamespace(
        _mission_target_variants=variants, _tool_frame_for_arm=lambda arm: arm.lower() + "_solver_tool"))
    monkeypatch.setattr(FfwSg2RobotAdapter, "_rigid", staticmethod(lambda matrix: SimpleNamespace(
        rotation=tuple(np.asarray(matrix)[:3, :3].flat), translation=tuple(np.asarray(matrix)[:3, 3]))))
    monkeypatch.setattr(FfwSg2RobotAdapter, "_tool", classmethod(lambda cls, matrix, arm: (matrix, arm)))


def test_single_ik_keeps_old_variants_and_returns_effective_targets_without_execution(monkeypatch, files):
    directory, config = files
    fake_solver_modules(monkeypatch)
    calls = []
    first = SimpleNamespace(q15=tuple([.1] * 15), position_error_m=.001, rotation_error_rad=.01)
    second = SimpleNamespace(q15=tuple([.2] * 15), position_error_m=.002, rotation_error_rad=.02)
    backend = SimpleNamespace(solve_grasp_candidate_approaches=lambda *a, **k: calls.append((a, k)) or ((1, first, second),))
    module = importlib.import_module("grasp_workbench.models.robot_ffw")
    monkeypatch.setattr(module, "_create_ik_backend", lambda *args: backend)
    adapter = FfwSg2RobotAdapter(config, config_dir=directory)
    initial = {"joint_positions": tuple([0.] * 15)}
    result = adapter.solve_single(np.eye(4), np.eye(4), "left", initial)
    assert result["success"] and result["self_collision_checked"]
    assert result["q_pregrasp"] == first.q15 and result["q_grasp"] == second.q15
    assert result["target_variant_index"] == 1 and result["selected_grasp"][0, 3] == .005
    assert len(calls[0][1]["target_variants"][0]) == 9
    assert calls[0][0][0].start_q15 == initial["joint_positions"]
    assert calls[0][1]["scene"] is None
    assert backend._tool_frame == "left_solver_tool" and backend._other_tool_frame == "right_solver_tool"


def test_dual_ik_solves_both_tools_together_and_seeds_grasp_from_pregrasp(monkeypatch, files):
    directory, config = files
    fake_solver_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "motion_planning_system.contracts", SimpleNamespace(
        BimanualIkTarget=lambda *args, **kwargs: SimpleNamespace(args=args, **kwargs)))
    calls = []

    def solve(request, **kwargs):
        calls.append((request, kwargs))
        return (SimpleNamespace(q15=tuple([len(calls) * .1] * 15),
            left_position_error_m=.001, right_position_error_m=.002, relative_translation_error_m=.003,
            left_rotation_error_rad=.01, right_rotation_error_rad=.02, relative_rotation_error_rad=.03),)

    adapter = FfwSg2RobotAdapter(config, config_dir=directory)
    adapter._dual_backend = SimpleNamespace(solve_bimanual_ik=solve)
    result = adapter.solve_dual(*(np.eye(4) for _ in range(4)), {"joint_positions": tuple([0.] * 15)})
    assert result["success"] and result["target_variant_index"] is None
    assert len(calls) == 2 and calls[0][0].position_tolerance_m == .01
    assert calls[1][0].start_q15 == result["q_pregrasp"]
    assert len(result["position_errors_m"]) == 6 and len(result["rotation_errors_rad"]) == 6
    assert set(result["selected_grasp"]) == {"left", "right"}


def test_unreachable_endpoint_does_not_return_a_fake_solution(monkeypatch, files):
    directory, config = files
    fake_solver_modules(monkeypatch)
    adapter = FfwSg2RobotAdapter(config, config_dir=directory)
    adapter._backend = SimpleNamespace(solve_grasp_candidate_approaches=lambda *a, **k: (None,))
    result = adapter.solve_single(np.eye(4), np.eye(4), "right", {"joint_positions": tuple([0.] * 15)})
    assert result == {"success": False, "self_collision_checked": True}


def test_single_dual_shape_caches_remain_separate_and_close_both(monkeypatch, files):
    directory, config = files
    created, destroyed = [], []

    def create(*args):
        identifier = len(created)
        backend = SimpleNamespace(_mission_fallback_ik=SimpleNamespace(
            destroy=lambda: destroyed.append(identifier)))
        created.append(backend)
        return backend

    module = importlib.import_module("grasp_workbench.models.robot_ffw")
    monkeypatch.setattr(module, "_create_ik_backend", create)
    adapter = FfwSg2RobotAdapter(config, config_dir=directory)
    single = adapter._solver()
    dual = adapter._solver(dual=True)
    assert single is not dual
    assert adapter._solver() is single and adapter._solver(dual=True) is dual
    assert len(created) == 2
    adapter.close()
    assert destroyed == [0, 1]
