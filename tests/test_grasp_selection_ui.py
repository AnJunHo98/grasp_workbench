"""Manual/semantic selection keeps canonical pairs independent of model generation."""
import json
import os
from copy import deepcopy
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult
from grasp_workbench.ipc.grasp_selection import SelectionResult
from grasp_workbench.ui.camera_window import CameraWindow
from grasp_workbench.ui.grasp_preview import grasp_axis_points, transform_points
from test_grasp_preview import descriptor as descriptor_fixture
from test_graspgenx_ui import FakeReceiver as GraspReceiver, done, deliver as deliver_grasp
from test_pose_mesh_ui import artist_triangles, mesh_path as mesh_path_fixture
from test_pose_rgb_overlay import frame, sam3_result
from test_semantic_session import semantic_result

descriptor, mesh_path = descriptor_fixture, mesh_path_fixture


class FakeReceiver(GraspReceiver):
    def __init__(self, service, *, grasp_selection=False, vlm=False, **kwargs):
        super().__init__(service, **kwargs)
        self.grasp_selection = grasp_selection
        self.selection_requests, self.selection_latest = [], None
        self.vlm_requests, self.vlm_latest = [], None

    def submit_selection(self, request):
        self.selection_requests.append(request)

    def take_selection_latest(self):
        result, self.selection_latest = self.selection_latest, None
        return result

    def submit_vlm(self, request):
        self.vlm_requests.append(request)

    def take_vlm_latest(self):
        result, self.vlm_latest = self.vlm_latest, None
        return result


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def make_window(app, mesh_path, tmp_path):
    windows = []

    def make(**options):
        window = CameraWindow(grasp_selection=True, receiver_factory=FakeReceiver, asset_dir=tmp_path / "assets", **options)
        window.timer.stop()
        for key in ("A", "B"):
            window.pose_mesh_inputs[key].setText(str(mesh_path))
            window.pose_scale_inputs[key].setValue(0.01)
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.receiver.alive = False
        window.close()
    app.processEvents()


def add_grasps(window, key, descriptor=None):
    window.generate_grasp(key)
    result = done(window.receiver.grasp_requests[-1], descriptor)
    result.bundle_path = f"/persistent/{key}/metadata.json"
    deliver_grasp(window, result)
    return result


def selection(request, *, robot=False):
    by_id = {row["observation_id"]: row for row in request.objects}
    targets = {}
    for index, assignment in enumerate(request.coordination["assignments"]):
        obj = by_id[assignment["observation_id"]]
        transform = np.eye(4)
        transform[:3, 3] = (-0.04 if assignment["arm"] == "left" else 0.04, 0.02, 0.01)
        targets[assignment["arm"]] = {"object_id": obj["object_id"], "observation_id": obj["observation_id"],
                                      "index": index, "object_from_gripper": transform.tolist(),
                                      "camera_from_gripper": None, "base_from_gripper": None,
                                      "base_from_tcp": np.eye(4).tolist() if robot else None, "base_from_pregrasp": None}
    candidate = {"candidate_id": "selected-0", "targets": targets, "model_score": 0.9, "geometry_score": 0.8,
                 "pair_score": 0.8, "ik_status": "solved" if robot else "not_checked", "reachability_score": 0.7 if robot else None}
    second = deepcopy(candidate)
    second["candidate_id"] = "selected-1"
    for target in second["targets"].values():
        target["object_from_gripper"][1][3] += 0.03
    return {"arm_mode": request.coordination["arm_mode"], "objects": {key: {"base_from_object": None} for key in by_id},
            "candidates": [candidate, second], "robot_connected": robot, "robot_ik_checked": robot,
            "self_collision_checked": robot, "scene_collision_checked": False, "trajectory_checked": False,
            "robot_command_published": False}


def deliver_selection(window, request, result=None):
    window.receiver.selection_latest = SelectionResult(request.request_id, "done", result or selection(request))
    window.poll_once()


@pytest.mark.parametrize("mode,arms", [("left", {"left"}), ("right", {"right"}), ("dual", {"left", "right"})])
def test_manual_mode_uses_selected_object_bundle_without_fp_or_generation(make_window, descriptor, mode, arms):
    window = make_window()
    source = add_grasps(window, "B", descriptor)
    window.selection_mode_input.setCurrentIndex(window.selection_mode_input.findData(mode))
    window.selection_object_input.setCurrentIndex(1)
    before = len(window.receiver.grasp_requests)
    window.request_grasp_selection()
    request, = window.receiver.selection_requests
    assert {row["arm"] for row in request.coordination["assignments"]} == arms
    assert all(row["observation_id"] == "manual-B" for row in request.coordination["assignments"])
    obj, = request.objects
    assert obj["object_id"] == "B" and obj["bundle_path"] == source.bundle_path and obj["camera_from_object"] is None
    assert "grasps" not in obj and "scores" not in obj
    deliver_selection(window, request)
    assert len(window.receiver.grasp_requests) == before and not window._pose_targets
    assert "geometry-only" in window.selection_status.text() and "로봇 미연결" in window.selection_status.text()
    assert len(window._grasp_artists) == len(arms) and "object coordinates" in window.axes.get_title()
    assert {(key, arm) for key, arm in window._grasp_artists} == {("B", arm) for arm in arms}
    assert {artists["label"].get_text() for artists in window._grasp_artists.values()} == {f"B {arm}" for arm in arms}
    assert "base_from_object" in window.selection_details.toPlainText()


def test_select_pair_changes_two_existing_artists_not_raw_bundle_or_generation(make_window, descriptor):
    window = make_window()
    source = add_grasps(window, "A", descriptor)
    untouched = source.grasps.copy()
    window.selection_mode_input.setCurrentIndex(window.selection_mode_input.findData("dual"))
    window.request_grasp_selection()
    request = window.receiver.selection_requests[-1]
    deliver_selection(window, request)
    artists = dict(window._grasp_artists)
    window.selection_candidate_input.setCurrentIndex(1)
    candidate = window._selected_selection_candidate()
    for arm, target in candidate["targets"].items():
        artist = window._grasp_artists[("A", arm)]["mesh"]
        assert artist is artists[("A", arm)]["mesh"]
        preview = window._grasp_previews["A"][0]
        expected = transform_points(preview.vertices, np.asarray(target["object_from_gripper"]))[preview.faces]
        np.testing.assert_allclose(artist_triangles(artist), expected, atol=1e-7)
    np.testing.assert_array_equal(source.grasps, untouched)
    assert len(window.receiver.grasp_requests) == len(window.receiver.selection_requests) == 1


def test_semantic_source_b_both_arms_preserves_observation_ids_and_ungrasped_target_context(make_window, tmp_path):
    window = make_window(vlm=True)
    window.receiver.latest = frame()
    window.poll_once()
    window.start_vlm()
    result = semantic_result(window.receiver.vlm_requests[-1])
    window.receiver.vlm_latest = result
    window.poll_once()
    add_grasps(window, "B")
    assert all(row["binding_active"] for row in window._semantic_session.record["objects"])
    window.request_grasp_selection()
    request, = window.receiver.selection_requests
    assert request.coordination == result.coordination
    target, source = request.objects
    assert target["observation_id"] == "target-obs" and target["object_id"] == "A" and target["bundle_path"] is None
    assert source["observation_id"] == "source-obs" and source["object_id"] == "B"
    assert {row["observation_id"] for row in request.coordination["assignments"]} == {"source-obs"}
    deliver_selection(window, request)
    assert set(window._grasp_artists) == {("B", "left"), ("B", "right")}
    reference = window._semantic_session.record["grasp_selection"]
    assert reference["objects"][1]["instance_id"] == source["instance_id"]
    saved = window.save_semantic_session(tmp_path / "selection-session.json")
    window.load_semantic_session(saved)
    assert window._selection_result is None and window._semantic_session.record["grasp_selection"] is not None
    assert window._semantic_session.record["grasp_selection"]["selected_candidate"] == reference["selected_candidate"]
    assert len(window.receiver.selection_requests) == 1 and not window._pose_targets


def test_cross_object_semantics_display_one_gripper_per_object_local_view(make_window):
    window = make_window(vlm=True)
    for key in ("A", "B"):
        add_grasps(window, key)
    window.receiver.latest = frame()
    window.poll_once()
    window.start_vlm()
    result = semantic_result(window.receiver.vlm_requests[-1])
    result.coordination["assignments"][0]["observation_id"] = "target-obs"
    window.receiver.vlm_latest = result
    window.poll_once()
    window.request_grasp_selection()
    request = window.receiver.selection_requests[-1]
    deliver_selection(window, request)
    assert set(window._grasp_artists) == {("A", "left")}
    window.grasp_object_input.setCurrentIndex(1)
    assert set(window._grasp_artists) == {("B", "right")}


def test_new_grasp_or_mesh_binding_ignores_previous_selection_completion(make_window):
    window = make_window()
    add_grasps(window, "A")
    window.request_grasp_selection()
    previous = window.receiver.selection_requests[-1]
    window.pose_scale_inputs["A"].setValue(0.02)
    deliver_selection(window, previous)
    assert window._selection_result is None
    add_grasps(window, "A")
    window.request_grasp_selection()
    current = window.receiver.selection_requests[-1]
    deliver_selection(window, previous)
    assert window._selection_result is None
    deliver_selection(window, current)
    assert window._selection_result is not None


def test_selected_effective_grippers_follow_latest_fp_not_frozen_request_camera_poses(make_window, descriptor, monkeypatch):
    window = make_window(foundationpose=True)
    window.prompt_inputs["A"].setText("cup")
    window.apply_prompts()
    source = sam3_result(window._prompts.revision)
    window.receiver.sam3_latest = source
    window.poll_once()
    window.start_pose("A")
    add_grasps(window, "A", descriptor)
    matrix = np.eye(4)
    matrix[:3, 3] = (0.1, -0.1, 1.0)
    item = ObjectPose("A", window._pose_targets["A"].session_id, window._pose_targets["A"].mesh_path, "tracking", matrix)
    window.receiver.pose_latest = PoseResult(source.frame, (item,))
    window.poll_once()
    window.selection_mode_input.setCurrentIndex(window.selection_mode_input.findData("dual"))
    window.request_grasp_selection()
    request = window.receiver.selection_requests[-1]
    frozen = selection(request, robot=True)
    for target in frozen["candidates"][0]["targets"].values():
        target["camera_from_gripper"] = (np.eye(4) * 42).tolist()
    displays = []
    original = window._display_frame

    def record_display(*args, **kwargs):
        displays.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(window, "_display_frame", record_display)
    matrix = matrix.copy()
    matrix[0, 3] += 0.1
    window.receiver.latest = frame(99, (12, 34, 56))
    window.receiver.pose_latest = PoseResult(source.frame, (replace(item, camera_from_object=matrix),))
    deliver_selection(window, request, frozen)
    assert len(displays) == 1 and displays[0].metadata["sequence"] == 99
    assert "IK 요청 snapshot" in window.selection_status.text()
    assert window.image_view.image.pixelColor(0, 0).getRgb()[:3] == (12, 34, 56)
    artists = dict(window._grasp_artists)
    for arm, target in frozen["candidates"][0]["targets"].items():
        transform = matrix @ np.asarray(target["object_from_gripper"])
        axis = grasp_axis_points(transform)
        np.testing.assert_allclose(artists[("A", arm)]["axes"][0].get_data_3d(), np.stack((axis[0], axis[1])).T)
    matrix = matrix.copy()
    matrix[0, 3] += 0.05
    window.receiver.pose_latest = PoseResult(source.frame, (replace(item, camera_from_object=matrix),))
    window.poll_once()
    assert window._grasp_artists[("A", "left")]["mesh"] is artists[("A", "left")]["mesh"]
    assert len(window.receiver.selection_requests) == len(window.receiver.grasp_requests) == 1
    assert "base_from_tcp" in json.dumps(window.selection_reference())


def test_vlm_unknown_is_not_inferred_as_single_or_dual_and_error_does_not_generate(make_window):
    window = make_window(vlm=True)
    window.receiver.latest = frame()
    window.poll_once()
    window.start_vlm()
    result = semantic_result(window.receiver.vlm_requests[-1])
    result.coordination = {"arm_mode": "unknown", "assignments": [], "constraints": [], "reason": "unclear"}
    window.receiver.vlm_latest = result
    window.poll_once()
    window.request_grasp_selection()
    request, = window.receiver.selection_requests
    assert request.coordination["arm_mode"] == "unknown" and not request.coordination["assignments"]
    window.receiver.selection_latest = SelectionResult(request.request_id, "error", error="example backend failure")
    window.poll_once()
    assert "example backend failure" in window.selection_status.text()
    assert not window.receiver.grasp_requests and not window._pose_targets
