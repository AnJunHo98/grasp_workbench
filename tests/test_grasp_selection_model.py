"""Object-local grasp selection and read-only endpoint IK with a fake robot."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from grasp_workbench.grasp_assets import grasp_identity, save_grasp_bundle, sha256_file
from grasp_workbench.models.grasp_selection import (
    GraspSelectionPredictor, _topdown_metrics, com_balance_metrics, estimate_mesh_com,
)


def pose(x=0., y=0., z=0., rotation=None):
    value = np.eye(4)
    value[:3, 3] = x, y, z
    if rotation is not None:
        value[:3, :3] = rotation
    return value


def coordination(mode="single", assignments=(("left", "obs-a"),)):
    return {"arm_mode": mode,
            "assignments": [{"arm": arm, "observation_id": observation,
                             "role": "hold", "grasp_region": ""} for arm, observation in assignments],
            "constraints": [], "reason": "test instruction"}


@pytest.fixture
def make_object(tmp_path):
    def make(key="A", *, positions=((-1., 0, 0), (1., 0, 0)), scores=(.8, .6),
             camera=None, scale=1., gripper="test-gripper", symmetric=True, mesh_center=(0, 0, 0), rotations=None):
        trimesh = pytest.importorskip("trimesh")
        mesh = trimesh.creation.box(extents=[2, 2, 2])
        mesh.apply_translation(mesh_center)
        mesh_path = tmp_path / (key + ".ply")
        mesh.export(mesh_path)
        request = SimpleNamespace(asset_id="asset-" + key, mesh_path=str(mesh_path), mesh_scale_to_m=scale)
        metadata = {"config": {"symmetric": symmetric, "sweep_volume": {"offset": [0, 0, .1]}}}
        identity = grasp_identity(request, gripper_id=gripper, gripper_metadata=metadata,
                                  model_metadata={}, inference_settings={})
        grasps = np.array([pose(*position) for position in positions]).reshape(-1, 4, 4)
        if rotations is not None:
            grasps[:, :3, :3] = rotations
        path = save_grasp_bundle(tmp_path / "grasps", identity, grasps, scores,
                                 raw_poses=grasps, raw_scores=scores, sample_points=np.empty((0, 3)),
                                 selected_indices=np.arange(len(grasps)))
        return {"object_id": key, "observation_id": "obs-" + key.lower(), "instance_id": "instance-" + key,
                "bundle_path": path, "mesh_path": str(mesh_path), "mesh_scale_to_m": scale,
                "camera_from_object": None if camera is None else camera.tolist()}
    return make


def request(*objects, mode="single", assignments=(("left", "obs-a"),)):
    return SimpleNamespace(request_id="selection-1", coordination=coordination(mode, assignments), objects=objects)


class Robot:
    def __init__(self, *, connected=True, success=True, gripper_id="test-gripper", transforms=None):
        self.starts = self.closes = 0
        self.single_calls, self.dual_calls = [], []
        self.success, self.answer = success, None
        self.state = None if not connected else {
            "base_from_camera": pose(10, 20, 30), "joint_names": ("j1", "j2"), "joint_positions": (0., 0.),
            "current_hands": {"left": pose(), "right": pose()}, "gripper_id": gripper_id,
            "canonical_gripper_from_tcp": transforms or {"left": pose(0, 0, .2), "right": pose(0, 0, .3)},
        }

    def start(self):
        self.starts += 1

    def close(self):
        self.closes += 1

    def snapshot(self):
        return self.state

    def solve_single(self, pregrasp, grasp, arm, *, seed):
        self.single_calls.append((np.array(pregrasp), np.array(grasp), arm, seed))
        return self._answer()

    def solve_dual(self, left_pregrasp, right_pregrasp, left_grasp, right_grasp, *, seed):
        self.dual_calls.append(tuple(map(np.array, (left_pregrasp, right_pregrasp, left_grasp, right_grasp))) + (seed,))
        return self._answer()

    def _answer(self):
        if self.answer is not None:
            return deepcopy(self.answer)
        return {"success": self.success, "q_pregrasp": (0., 0.), "q_grasp": (0., 0.),
                "position_errors_m": (0.,), "rotation_errors_rad": (0.,),
                "target_variant_index": 0, "self_collision_checked": True}


def test_model_only_uses_explicit_arm_not_object_slot_and_current_pose(make_object):
    obj = make_object("B", camera=pose(2, 3, 4))
    job = request(obj, assignments=(("left", "obs-b"),))
    before = Path(obj["bundle_path"]).with_name("grasps.npz").read_bytes()
    predictor = GraspSelectionPredictor()
    result = predictor.predict(job)
    target = result["candidates"][0]["targets"]["left"]
    assert target["object_id"] == "B" and target["observation_id"] == "obs-b"
    np.testing.assert_allclose(target["camera_from_gripper"], pose(1, 3, 4))
    assert target["base_from_gripper"] is target["base_from_tcp"] is target["base_from_pregrasp"] is None
    assert not result["robot_connected"] and not result["robot_ik_checked"]
    assert not result["self_collision_checked"] and result["objects"]["obs-b"]["base_from_object"] is None
    job.objects[0]["camera_from_object"] = pose(5, 6, 7).tolist()
    updated = predictor.predict(job)
    np.testing.assert_allclose(updated["candidates"][0]["targets"]["left"]["camera_from_gripper"], pose(4, 6, 7))
    assert Path(obj["bundle_path"]).with_name("grasps.npz").read_bytes() == before
    for key in ("scene_collision_checked", "trajectory_checked", "robot_command_published"):
        assert result[key] is False
    json.dumps(result, allow_nan=False)
    predictor.close()


@pytest.mark.parametrize("mode,assignments", [
    ("unknown", ((None, "obs-a"),)), ("single", ((None, "obs-a"),)),
    ("single", (("left", None),)), ("dual", (("left", "obs-a"),)),
])
def test_unknown_or_incomplete_assignments_do_not_guess_arms_or_read_bundles(mode, assignments):
    obj = {"object_id": "A", "observation_id": "obs-a", "instance_id": "instance-a",
           "bundle_path": None, "mesh_path": None, "mesh_scale_to_m": 1., "camera_from_object": None}
    result = GraspSelectionPredictor().predict(request(obj, mode=mode, assignments=assignments))
    assert result["candidates"] == [] and result["reason"]
    assert not result["robot_ik_checked"]


def test_ungrasped_semantic_target_needs_no_bundle(make_object):
    obj = make_object(camera=pose())
    target = {"object_id": "B", "observation_id": "obs-b", "instance_id": "target",
              "bundle_path": None, "mesh_path": None, "mesh_scale_to_m": 1., "camera_from_object": None}
    job = request(obj, target)
    job.coordination["constraints"] = [{
        "reference": {"frame": "object", "observation_id": "obs-b"},
        "moving": {"frame": "left_ee", "observation_id": None}, "behavior": "maintain_relative_pose",
        "phases": ["grasp"], "reference_state": "unknown", "description": "semantic only"}]
    result = GraspSelectionPredictor().predict(job)
    assert len(result["candidates"]) == 2


def test_single_base_tcp_offsets_seed_and_old_reachability_score(make_object):
    obj = make_object(positions=((.2, .3, .4),), scores=(.8,), camera=pose(1, 2, 3))
    rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rx180 = np.diag([1, -1, -1])
    canonical_tcp = pose(0, 0, .137, rz90 @ rx180)
    robot = Robot(transforms={"left": canonical_tcp, "right": canonical_tcp})
    robot.answer = {"success": True, "q_pregrasp": (1., 0.), "q_grasp": (1., 1.),
                    "position_errors_m": (.001,), "rotation_errors_rad": (.02,),
                    "target_variant_index": 1, "self_collision_checked": True}
    predictor = GraspSelectionPredictor(robot_factory=lambda: robot,
                                        settings={"preference": "legacy", "pregrasp_distance_m": .15,
                                                  "grasp_retreat_distance_m": .02,
                                                  "grasp_advance_distance_m": .01})
    predictor.start()
    predictor.start()
    result = predictor.predict(request(obj))
    target = result["candidates"][0]["targets"]["left"]
    base_object = robot.state["base_from_camera"] @ pose(1, 2, 3)
    expected_grasp = base_object @ pose(.2, .3, .4) @ pose(z=-.02) @ canonical_tcp @ pose(z=-.01)
    expected_pregrasp = base_object @ pose(.2, .3, .4) @ pose(z=-.17) @ canonical_tcp
    np.testing.assert_allclose(result["objects"]["obs-a"]["base_from_object"], base_object)
    np.testing.assert_allclose(target["base_from_tcp"], expected_grasp)
    np.testing.assert_allclose(target["base_from_pregrasp"], expected_pregrasp)
    np.testing.assert_allclose(robot.single_calls[0][0], expected_pregrasp)
    np.testing.assert_allclose(robot.single_calls[0][1], expected_grasp)
    assert robot.single_calls[0][2] == "left" and robot.single_calls[0][3] is robot.state
    expected_reachability = .5 / (1 + np.sqrt(2)) + .3 * .85 + .2 / 1.3
    candidate = result["candidates"][0]
    assert candidate["reachability_score"] == pytest.approx(expected_reachability)
    assert candidate["pair_score"] == pytest.approx(.6 * .8 + .4 * expected_reachability)
    assert result["robot_ik_checked"] and result["self_collision_checked"]
    assert candidate["ik_status"] == "solved" and robot.starts == 1
    predictor.close()
    assert robot.closes == 1


def test_selected_single_perturbation_is_reflected_in_all_frames(make_object):
    obj = make_object(positions=((0, 0, 0),), scores=(.8,), camera=pose(1, 2, 3))
    directory = Path(obj["bundle_path"]).parent
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    robot = Robot()
    selected = pose(11.005, 22, 33.2)
    robot.answer = {**robot._answer(), "selected_grasp": selected, "selected_pregrasp": pose(11.005, 22, 33.05),
                    "target_variant_index": 1}
    result = GraspSelectionPredictor(robot_factory=lambda: robot).predict(request(obj))
    target = result["candidates"][0]["targets"]["left"]
    np.testing.assert_allclose(target["base_from_tcp"], selected)
    np.testing.assert_allclose(target["base_from_gripper"], pose(11.005, 22, 33))
    np.testing.assert_allclose(target["camera_from_gripper"], pose(1.005, 2, 3))
    np.testing.assert_allclose(target["object_from_gripper"], pose(.005, 0, 0), atol=1e-12)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original


def test_disconnected_and_wrong_gripper_remain_model_only_but_calibration_geometry_is_kept(make_object):
    obj = make_object(camera=pose(1, 2, 3))
    disconnected = Robot(connected=False)
    result = GraspSelectionPredictor(robot_factory=lambda: disconnected).predict(request(obj))
    assert not result["robot_connected"] and not disconnected.single_calls
    wrong = Robot(gripper_id="different-gripper")
    result = GraspSelectionPredictor(robot_factory=lambda: wrong).predict(request(obj))
    assert result["robot_connected"] and not result["robot_ik_checked"] and not wrong.single_calls
    assert result["reason"] and "gripper" in result["reason"].lower()
    np.testing.assert_allclose(result["objects"]["obs-a"]["base_from_object"], pose(11, 22, 33))
    target = result["candidates"][0]["targets"]["left"]
    np.testing.assert_allclose(target["base_from_gripper"], pose(10, 22, 33))
    assert target["base_from_tcp"] is None


def test_single_failed_ik_and_top_k_do_not_claim_feasibility(make_object):
    obj = make_object(camera=pose())
    robot = Robot(success=False)
    result = GraspSelectionPredictor(robot_factory=lambda: robot, settings={"top_k": 1}).predict(request(obj))
    assert len(robot.single_calls) == 1 and result["robot_ik_checked"]
    assert [candidate["ik_status"] for candidate in result["candidates"]] == ["not_reachable", "not_in_mission_pool"]
    assert all(candidate["reachability_score"] is None for candidate in result["candidates"])


def test_old_com_balance_formula_scale_and_open_mesh_bbox(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    closed = trimesh.creation.box(extents=[2, 4, 6])
    closed.apply_translation([10, 20, 30])
    path = tmp_path / "solid.ply"
    closed.export(path)
    com = estimate_mesh_com(path, .01)
    np.testing.assert_allclose(com["position_m"], [.1, .2, .3])
    assert com["method"] == "uniform_density_volume" and not com["physical_stability_verified"]
    opened = trimesh.Trimesh(vertices=[[10, 20, 30], [12, 20, 30], [10, 24, 30]], faces=[[0, 1, 2]], process=False)
    opened.export(path)
    com = estimate_mesh_com(path, .01)
    np.testing.assert_allclose(com["position_m"], [.11, .22, .3])
    assert com["method"] == "bounding_box_center"
    balance = com_balance_metrics([-1, 0, 0], [1, 0, 0], [0, 1, 4], [0, 0, -1])
    assert balance["projection_fraction"] == pytest.approx(.5)
    assert balance["segment_distance_m"] == pytest.approx(1.)
    assert balance["score"] == pytest.approx(.5)
    assert com_balance_metrics([-1, 0, 0], [1, 0, 0], [0, 0, 0], None)["score"] is None
    assert com_balance_metrics([0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, -1])["score"] is None


def test_same_object_dual_pairs_old_com_and_distance_priority(make_object):
    obj = make_object(positions=((-1, 0, 0), (1, 0, 0)), scores=(.8, .6), camera=pose())
    robot = Robot(success=False)
    robot.state["base_from_camera"] = pose()
    robot.state["current_hands"] = {"left": pose(1, 0, .2), "right": pose(-1, 0, .3)}
    result = GraspSelectionPredictor(robot_factory=lambda: robot).predict(
        request(obj, mode="dual", assignments=(("left", "obs-a"), ("right", "obs-a"))))
    assert len(result["candidates"]) == 2 and len(robot.dual_calls) == 8
    first = result["candidates"][0]
    assert first["targets"]["left"]["index"] == 1 and first["targets"]["right"]["index"] == 0
    assert first["model_score"] == pytest.approx(.6)
    assert first["balance_metrics"]["score"] == pytest.approx(1.)
    assert first["geometry_score"] == pytest.approx(.8)
    assert first["assignment_distance_m"] == pytest.approx(0.)
    assert all(candidate["ik_status"] == "not_reachable" for candidate in result["candidates"])
    assert all(call[-1] is robot.state for call in robot.dual_calls)


def test_dual_symmetry_rotates_canonical_origin_and_preserves_effective_targets(make_object):
    obj = make_object(camera=pose())
    robot = Robot()
    robot.state["base_from_camera"] = pose()
    robot.state["canonical_gripper_from_tcp"] = {"left": pose(.4, 0, .2), "right": pose(.3, 0, .2)}
    count = 0

    def solve(*args, seed):
        nonlocal count
        robot.dual_calls.append(tuple(map(np.array, args)))
        count += 1
        answer = robot._answer()
        answer["success"] = count == 2  # noflip-left / flip-right for the first ordered pair
        return answer

    robot.solve_dual = solve
    result = GraspSelectionPredictor(robot_factory=lambda: robot, settings={"top_k": 1}).predict(
        request(obj, mode="dual", assignments=(("left", "obs-a"), ("right", "obs-a"))))
    assert len(robot.dual_calls) == 4
    candidate = result["candidates"][0]
    assert candidate["ik_status"] == "solved" and candidate["symmetry_variant_index"] == 1
    right = candidate["targets"]["right"]
    expected = pose(1, 0, 0) @ np.diag([-1., -1., 1., 1.])
    np.testing.assert_allclose(right["object_from_gripper"], expected)
    np.testing.assert_allclose(right["base_from_tcp"], expected @ pose(.3, 0, .2))
    # A pi rotation at the TCP origin would incorrectly keep x=1.3 instead of .7.
    assert np.asarray(right["base_from_tcp"])[0, 3] == pytest.approx(.7)
    assert candidate["reachability_score"] == pytest.approx(1.)  # symmetry is not a perturbation
    assert result["candidates"][1]["ik_status"] == "not_checked"


def test_cross_object_dual_has_no_shared_com_and_same_indices_are_allowed(make_object):
    left = make_object("B", positions=((.2, 0, 0),), scores=(.8,), camera=pose(1, 0, 0), symmetric=False)
    right = make_object("A", positions=((.4, 0, 0),), scores=(.4,), camera=pose(0, 2, 0), symmetric=False)
    robot = Robot()
    robot.state["base_from_camera"] = pose()
    result = GraspSelectionPredictor(robot_factory=lambda: robot).predict(
        request(left, right, mode="dual", assignments=(("left", "obs-b"), ("right", "obs-a"))))
    candidate = result["candidates"][0]
    assert len(robot.dual_calls) == 1 and candidate["model_score"] == pytest.approx(.6)
    assert candidate["geometry_score"] == pytest.approx(.6) and candidate["balance_metrics"] is None
    targets = candidate["targets"]
    assert targets["left"]["object_id"] == "B" and targets["right"]["object_id"] == "A"
    np.testing.assert_allclose(robot.dual_calls[0][2], pose(1.2, 0, .2))
    np.testing.assert_allclose(robot.dual_calls[0][3], pose(.4, 2, .3))
    assert candidate["ik_status"] == "solved"


def test_two_instances_can_reuse_one_asset_bundle_without_becoming_one_object(make_object):
    left = make_object(positions=((.2, 0, 0),), scores=(.8,), camera=pose(1, 0, 0))
    right = {**left, "object_id": "B", "observation_id": "obs-b", "instance_id": "instance-B",
             "camera_from_object": pose(0, 2, 0).tolist()}
    result = GraspSelectionPredictor().predict(
        request(left, right, mode="dual", assignments=(("left", "obs-a"), ("right", "obs-b"))))
    assert len(result["candidates"]) == 1 and result["candidates"][0]["balance_metrics"] is None
    targets = result["candidates"][0]["targets"]
    np.testing.assert_allclose(targets["left"]["camera_from_gripper"], pose(1.2, 0, 0))
    np.testing.assert_allclose(targets["right"]["camera_from_gripper"], pose(.2, 2, 0))


def test_mesh_scale_is_not_applied_again_to_metre_grasp_bundle(make_object):
    obj = make_object(positions=((.3, 0, 0),), scores=(.8,), camera=pose(2, 0, 0), scale=.01)
    result = GraspSelectionPredictor().predict(request(obj))
    target = result["candidates"][0]["targets"]["left"]
    np.testing.assert_allclose(target["object_from_gripper"], pose(.3, 0, 0))
    np.testing.assert_allclose(target["camera_from_gripper"], pose(2.3, 0, 0))


def test_pair_search_full_pool_but_only_compact_64_returned(make_object):
    obj = make_object(positions=tuple((float(i), 0, 0) for i in range(9)), scores=(.5,) * 9)
    result = GraspSelectionPredictor().predict(
        request(obj, mode="dual", assignments=(("left", "obs-a"), ("right", "obs-a"))))
    assert result["total_candidate_count"] == 72 and len(result["candidates"]) == 64
    assert all(candidate["targets"]["left"]["index"] != candidate["targets"]["right"]["index"]
               for candidate in result["candidates"])


def test_single_model_pool_is_not_pair_preview_truncated(make_object):
    obj = make_object(positions=tuple((float(i), 0, 0) for i in range(65)), scores=(.5,) * 65)
    result = GraspSelectionPredictor().predict(request(obj))
    assert len(result["candidates"]) == result["total_candidate_count"] == 65


@pytest.mark.parametrize("change", ["mesh", "scale"])
def test_request_preserves_existing_bundle_mesh_and_scale_identity(make_object, change):
    obj = make_object()
    if change == "mesh":
        Path(obj["mesh_path"]).write_bytes(b"changed-geometry")
    else:
        obj["mesh_scale_to_m"] = .01
    with pytest.raises(ValueError, match="mesh/scale"):
        GraspSelectionPredictor().predict(request(obj))


def test_empty_bundle_is_not_a_failed_robot_pose(make_object):
    obj = make_object(positions=(), scores=(), camera=pose())
    robot = Robot()
    result = GraspSelectionPredictor(robot_factory=lambda: robot).predict(request(obj))
    assert result["candidates"] == [] and result["reason"] == "No grasp candidates"
    assert not result["robot_ik_checked"] and not robot.single_calls


@pytest.mark.parametrize("settings", [
    {"top_k": 0}, {"top_k": -1}, {"top_k": True}, {"top_k": 1.5},
    {"pregrasp_distance_m": 0.}, {"pregrasp_distance_m": float("nan")},
    {"pregrasp_distance_m": True}, {"grasp_retreat_distance_m": -.01},
    {"grasp_advance_distance_m": float("inf")}, {"grasp_advance_distance_m": -.01},
])
def test_existing_grasp_profile_numeric_contract_is_preserved(settings):
    with pytest.raises(ValueError):
        GraspSelectionPredictor(settings=settings)


def test_existing_single_top_k_32_limit_does_not_limit_dual_search_wave(make_object):
    obj = make_object()
    predictor = GraspSelectionPredictor(settings={"top_k": 33})
    with pytest.raises(ValueError, match="32"):
        predictor.predict(request(obj))
    result = predictor.predict(request(obj, mode="dual", assignments=(("left", "obs-a"), ("right", "obs-a"))))
    assert result["total_candidate_count"] == 2


def test_topdown_metrics_use_actual_base_approach_not_object_or_tcp_axes():
    assert _topdown_metrics(pose(z=1), pose())["topdown_score"] == 1.
    assert _topdown_metrics(pose(z=1), pose())["approach_angle_deg"] == 0.
    assert _topdown_metrics(pose(), pose(z=1))["approach_angle_deg"] == 180.
    assert _topdown_metrics(pose(), pose(x=1))["topdown_score"] == 0.
    assert _topdown_metrics(pose(), pose())["approach_angle_deg"] is None


def test_default_single_topdown_preselection_before_ik_cap_and_50_30_20_score(make_object):
    down = np.diag([1., -1., -1.])
    obj = make_object(positions=((0, 0, 0), (.1, 0, 0)), scores=(.99, .4), camera=pose(),
                      rotations=(np.eye(3), down))
    robot = Robot()
    robot.state["base_from_camera"] = pose()
    result = GraspSelectionPredictor(robot_factory=lambda: robot, settings={"top_k": 1}).predict(request(obj))
    best = result["candidates"][0]
    assert len(robot.single_calls) == 1 and best["targets"]["left"]["index"] == 1
    assert best["topdown_score"] == pytest.approx(1.) and best["approach_angle_deg"] == pytest.approx(0.)
    assert best["preselection_score"] == pytest.approx((.5 + .3 * .4) / .8)
    assert best["pair_score"] == pytest.approx(.5 + .3 * .4 + .2)
    assert result["candidates"][1]["ik_status"] == "not_in_mission_pool"


def test_topdown_preselection_keeps_old_2cm_10degree_diversity(make_object):
    down = np.diag([1., -1., -1.])
    obj = make_object(positions=((0, 0, 0), (.001, 0, 0), (.1, 0, 0)), scores=(.9, .89, .5), camera=pose(),
                      rotations=(down, down, down))
    robot = Robot(success=False)
    robot.state["base_from_camera"] = pose()
    result = GraspSelectionPredictor(robot_factory=lambda: robot, settings={"top_k": 2}).predict(request(obj))
    assert len(robot.single_calls) == 2
    assert {candidate["targets"]["left"]["index"] for candidate in result["candidates"]
            if candidate["ik_status"] == "not_reachable"} == {0, 2}


def test_topdown_diversity_keeps_near_position_when_rotation_is_distinct(make_object):
    down = np.diag([1., -1., -1.])
    rotations = []
    for degrees in (0., 5., 20.):
        cosine, sine = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
        rotations.append(np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]) @ down)
    obj = make_object(positions=((0, 0, 0),) * 3, scores=(.9, .89, .5), camera=pose(), rotations=rotations)
    robot = Robot(success=False)
    result = GraspSelectionPredictor(robot_factory=lambda: robot, settings={"top_k": 2}).predict(request(obj))
    assert {candidate["targets"]["left"]["index"] for candidate in result["candidates"]
            if candidate["ik_status"] == "not_reachable"} == {0, 2}


def test_topdown_recomputes_final_metrics_from_solver_selected_poses(make_object):
    obj = make_object(positions=((0, 0, 0),), scores=(.8,), camera=pose())
    robot = Robot()
    robot.answer = {**robot._answer(), "selected_pregrasp": pose(11, 20, 30),
                    "selected_grasp": pose(10, 20, 29), "target_variant_index": 1}
    result = GraspSelectionPredictor(robot_factory=lambda: robot).predict(request(obj))
    best = result["candidates"][0]
    assert best["topdown_score"] == pytest.approx(1 / np.sqrt(2))
    assert best["approach_angle_deg"] == pytest.approx(45.)
    assert best["pair_score"] == pytest.approx(.5 / np.sqrt(2) + .3 * .8 + .2 * best["reachability_score"])


def test_robot_snapshot_method_reuses_started_adapter_for_fk(make_object):
    robot = Robot()
    predictor = GraspSelectionPredictor(robot_factory=lambda: robot)
    assert predictor.robot_snapshot() is robot.state
    predictor.predict(request(make_object(camera=pose())))
    assert robot.starts == 1
    predictor.close()


@pytest.mark.parametrize("state,instance,expected_source", [
    ("attached", "instance-A", "grasp_fk"), ("attached", "another-instance", "camera"),
    ("waiting_for_robot_state", "instance-A", "camera"),
])
def test_explicit_matching_attached_fk_base_is_used_without_changing_raw_camera_pose(
    make_object, state, instance, expected_source,
):
    obj = make_object(positions=((.1, .2, .3),), scores=(.8,), camera=pose(1, 2, 3))
    raw_camera = deepcopy(obj["camera_from_object"])
    obj["grasp_fk"] = {"state": state, "instance_id": instance, "base_from_object": pose(4, 5, 6).tolist(),
                        "observed_sequence": 123}
    robot = Robot()
    result = GraspSelectionPredictor(robot_factory=lambda: robot).predict(request(obj))
    expected_base = pose(4, 5, 6) if expected_source == "grasp_fk" else pose(11, 22, 33)
    assert result["objects"]["obs-a"]["pose_source"] == expected_source
    np.testing.assert_allclose(result["objects"]["obs-a"]["base_from_object"], expected_base)
    target = result["candidates"][0]["targets"]["left"]
    np.testing.assert_allclose(target["base_from_tcp"], expected_base @ pose(.1, .2, .3) @ pose(z=.2))
    np.testing.assert_allclose(target["camera_from_gripper"], pose(1.1, 2.2, 3.3))
    assert obj["camera_from_object"] == raw_camera


def test_attached_fk_without_camera_pose_preserves_absent_camera_output_on_selected_ik(make_object):
    obj = make_object(positions=((0, 0, 0),), scores=(.8,))
    obj["grasp_fk"] = {"state": "attached", "instance_id": obj["instance_id"],
                        "base_from_object": pose(4, 5, 6).tolist()}
    robot = Robot()
    robot.answer = {**robot._answer(), "selected_pregrasp": pose(4, 5, 6.05), "selected_grasp": pose(4, 5, 6.2)}
    result = GraspSelectionPredictor(robot_factory=lambda: robot).predict(request(obj))
    assert result["robot_ik_checked"]
    assert result["candidates"][0]["targets"]["left"]["camera_from_gripper"] is None


def test_old_yaml_source_ids_are_preserved_for_exact_topdown_ties(make_object):
    down = np.diag([1., -1., -1.])
    obj = make_object(positions=((0, 0, 0), (.1, 0, 0)), scores=(.8, .8), camera=pose(), rotations=(down, down))
    manifest = Path(obj["bundle_path"])
    metadata = json.loads(manifest.read_text())
    document = {"format": "isaac_grasp", "format_version": 1.0, "grasps": {
        key: {"position": position, "orientation": {"w": 0., "xyz": [1., 0., 0.]}, "confidence": .8}
        for key, position in (("grasp_z", [0, 0, 0]), ("grasp_a", [.1, 0, 0]))}}
    candidates = manifest.with_name("grasps.yaml")
    candidates.write_text(yaml.safe_dump(document, sort_keys=False))
    metadata.update(candidates_file="grasps.yaml", candidates_sha256=sha256_file(candidates))
    manifest.write_text(json.dumps(metadata))
    before = manifest.read_bytes(), candidates.read_bytes()
    obj["bundle_path"] = str(candidates)
    robot = Robot()
    result = GraspSelectionPredictor(robot_factory=lambda: robot, settings={"top_k": 1}).predict(request(obj))
    selected = result["candidates"][0]["targets"]["left"]
    assert selected["index"] == 1 and selected["source_candidate_id"] == "graspgenx:grasp_a"
    assert before == (manifest.read_bytes(), candidates.read_bytes())

