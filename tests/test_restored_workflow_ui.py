"""Restored request-only UI flows use fake workers, never robot commands or GPUs."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtCore, QtWidgets

from grasp_workbench.config import load_run_config
from grasp_workbench.grasp_assets import grasp_identity, save_grasp_bundle
from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult
from grasp_workbench.ipc.grasp_selection import SelectionResult
from grasp_workbench.ipc.sam3_refine import RefineResult
from grasp_workbench.snapshot import save_snapshot
from grasp_workbench.ui.camera_window import CameraWindow
from grasp_workbench.ui.mask_editor import MaskCanvas
from test_camera_ui import completed_mesh
from test_grasp_selection_ui import FakeReceiver as BaseReceiver, add_grasps, deliver_selection
from test_graspgenx_ui import done
from test_pose_mesh_ui import artist_triangles, mesh_path as mesh_path_fixture
from test_pose_rgb_overlay import frame, sam3_result
from test_semantic_session import semantic_result

mesh_path = mesh_path_fixture


class FakeReceiver(BaseReceiver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fk_latest, self.refine_latest, self.refine_requests = None, None, []

    def take_fk_latest(self):
        value, self.fk_latest = self.fk_latest, None
        return value

    def submit_refine(self, request):
        self.refine_requests.append(request)

    def take_refine_latest(self):
        value, self.refine_latest = self.refine_latest, None
        return value


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def make_window(app, mesh_path, tmp_path):
    windows = []

    def make(**options):
        window = CameraWindow(receiver_factory=FakeReceiver, asset_dir=tmp_path / "assets", **options)
        window.timer.stop()
        if hasattr(window, "pose_mesh_inputs"):
            for key in ("A", "B"):
                window.pose_mesh_inputs[key].setText(str(mesh_path))
                window.pose_scale_inputs[key].setValue(.01)
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.receiver.alive = False
        window.close()
    app.processEvents()


def observe(window):
    for key in ("A", "B"):
        window.prompt_inputs[key].setText(key)
    window.apply_prompts()
    source = sam3_result(window._prompts.revision)
    window.receiver.sam3_latest, window.receiver.latest = source, frame(99)
    window.poll_once()
    return source


def track(window, source):
    matrix = np.eye(4)
    matrix[2, 3] = 1
    poses = tuple(ObjectPose(key, target.session_id, target.mesh_path, "tracking", matrix.copy())
                  for key, target in window._pose_targets.items())
    window.receiver.pose_latest = PoseResult(source.frame, poses)
    window.poll_once()


def test_manual_source_b_task_requests_pose_then_only_source_grasp_and_selection_once(make_window):
    window = make_window(foundationpose=True, grasp_selection=True)
    source = observe(window)
    source_id = next(row["observation_id"] for row in window._task_rows() if row["slot"] == "B")
    target_id = next(row["observation_id"] for row in window._task_rows() if row["slot"] == "A")
    assert source_id != target_id and source_id not in {"manual-A", "manual-B"}
    window.source_role_input.setCurrentIndex(window.source_role_input.findData(source_id))
    window.target_role_input.setCurrentIndex(window.target_role_input.findData(target_id))
    window.start_task_pipeline()
    assert set(window._pose_targets) == {"A", "B"}
    track(window, source)
    assert [request.object_id for request in window.receiver.grasp_requests] == ["B"]
    request = window.receiver.grasp_requests[0]
    window.receiver.grasp_events.append(replace(done(request), bundle_path="/saved/B/metadata.json"))
    window.poll_once()
    chosen, = window.receiver.selection_requests
    assert len(chosen.objects) == 2
    assert chosen.coordination["assignments"][0]["observation_id"] == source_id
    assert chosen.coordination["assignments"][0]["arm"] == "left"
    deliver_selection(window, chosen)
    for _ in range(4):
        track(window, source)
    assert window._task_flow["phase"] == "done"
    assert len(window.receiver.grasp_requests) == len(window.receiver.selection_requests) == 1
    assert "로봇 명령 없음" in window.task_status.text()


def test_vlm_auto_task_keeps_all_pose_objects_but_generates_source_b_only(make_window):
    window = make_window(vlm=True, foundationpose=True, grasp_selection=True)
    window.receiver.latest = frame()
    window.poll_once()
    window.start_vlm()
    result = semantic_result(window.receiver.vlm_requests[-1])
    window.receiver.vlm_latest = result
    window.poll_once()
    assert window.source_role_input.currentData() == "source-obs"
    window.receiver.sam3_latest = sam3_result(window._prompts.revision)
    window.poll_once()
    assert set(window._pose_targets) == {"A", "B"}
    track(window, window._sam3_result)
    assert [request.object_id for request in window.receiver.grasp_requests] == ["B"]
    request = window.receiver.grasp_requests[0]
    window.receiver.grasp_events.append(replace(done(request), bundle_path="/saved/B/metadata.json"))
    window.poll_once()
    chosen, = window.receiver.selection_requests
    assert chosen.coordination == result.coordination
    assert {item["observation_id"] for item in chosen.coordination["assignments"]} == {"source-obs"}
    assert all(row["binding_active"] for row in window._semantic_session.record["objects"])


def test_manual_role_changes_clear_semantic_arm_assignments_without_guessing_slots(make_window):
    window = make_window(vlm=True, foundationpose=True, grasp_selection=True)
    window.receiver.latest = frame()
    window.poll_once()
    window.start_vlm()
    window.receiver.vlm_latest = semantic_result(window.receiver.vlm_requests[-1])
    window.poll_once()
    window.target_role_input.setCurrentIndex(0)
    window.source_role_input.setCurrentIndex(window.source_role_input.findData("target-obs"))
    record = window._semantic_session.record
    assert record["task_binding"]["source_observation_id"] == "target-obs"
    assert record["task_binding"]["target_observation_id"] is None
    assert record["coordination"] is None and window._task_flow is None
    assert "재분석" in window.task_status.text()
    window.request_grasp_selection()
    assert not window.receiver.selection_requests and "VLM 팔 정보 없음" in window.selection_status.text()


def test_task_reuses_explicit_loaded_source_candidates_and_does_not_retry_errors(make_window):
    window = make_window(foundationpose=True, grasp_selection=True)
    source = observe(window)
    add_grasps(window, "B")
    observation = next(row for row in window._task_rows() if row["slot"] == "B")
    window.source_role_input.setCurrentIndex(window.source_role_input.findData(observation["observation_id"]))
    window.start_task_pipeline()
    track(window, source)
    assert len(window.receiver.grasp_requests) == len(window.receiver.selection_requests) == 1
    request = window.receiver.selection_requests[-1]
    window.receiver.selection_latest = SelectionResult(request.request_id, "error", error="fixture failure")
    window.poll_once()
    for _ in range(3):
        window.poll_once()
    assert window._task_flow["phase"] == "error"
    assert len(window.receiver.selection_requests) == 1


def test_task_without_grasp_worker_remains_pose_only_and_does_not_enable_it(make_window):
    window = make_window(foundationpose=True)
    source = observe(window)
    window.source_role_input.setCurrentIndex(1)
    window.start_task_pipeline()
    track(window, source)
    assert not window.graspgenx and not window.receiver.grasp_requests
    assert window._task_flow["phase"] == "done"


def test_single_selection_is_deep_locked_until_explicit_reselect_or_refresh(make_window):
    window = make_window(grasp_selection=True)
    add_grasps(window, "A")
    window.request_grasp_selection()
    request = window.receiver.selection_requests[-1]
    deliver_selection(window, request)
    pinned = deepcopy(window.selection_reference())
    window._selection_result["candidates"][0]["targets"]["left"]["object_from_gripper"][0][3] = 999
    assert window.selection_reference()["selected_candidate"] == pinned["selected_candidate"]
    deliver_selection(window, request)
    assert window.selection_reference()["selected_candidate"] == pinned["selected_candidate"]
    window.selection_candidate_input.setCurrentIndex(1)
    assert window.selection_reference()["selected_candidate"]["candidate_id"] == "selected-1"
    window.request_grasp_selection()
    assert window._selection_lock is None and len(window.receiver.selection_requests) == 2


def test_snapshot_owned_round_trip_does_not_restore_current_pose_or_request_workers(make_window, tmp_path):
    original = make_window(foundationpose=True, grasp_selection=True)
    source = observe(original)
    original.start_pose("A")
    track(original, source)
    snapshot = original.capture_snapshot()
    source.frame.rgb[:] = 231
    source.objects[0].masks[:] = False
    assert np.all(snapshot.sam3.frame.rgb == 150) and snapshot.sam3.objects[0].masks.any()
    path = original.save_snapshot(tmp_path / "snapshot.npz")
    loaded = make_window(foundationpose=True, grasp_selection=True)
    before = len(loaded.receiver.pose_controls)
    result = loaded.load_snapshot(path)
    assert result.pose_result.objects[0].camera_from_object is not None
    assert not loaded._pose_targets and not loaded._pose_objects
    assert len(loaded.receiver.pose_controls) == before
    assert not loaded.receiver.grasp_requests and not loaded.receiver.selection_requests
    assert "고정 snapshot" in loaded.metadata_label.text()
    loaded.receiver.latest = frame(200, (12, 34, 56))
    loaded.poll_once()
    assert "고정 snapshot" in loaded.metadata_label.text()
    loaded.show_live_camera()
    assert loaded.image_view.image.pixelColor(0, 0).getRgb()[:3] == (12, 34, 56)


def test_saved_fp_seed_uses_its_exact_frame_without_inventing_mask(make_window, tmp_path):
    window = make_window(foundationpose=True)
    frozen = frame(17)
    initial = np.eye(4)
    initial[:3, 3] = (.2, .3, 1.2)
    path = window.pose_mesh_inputs["B"].text()
    saved = save_snapshot(tmp_path / "seed.npz", frame(100),
        pose_result=PoseResult(frozen, (ObjectPose("B", "old", path, "tracking", initial),)),
        objects=({"object_id": "B", "mesh_path": path, "mesh_scale_to_m": .01},))
    window.pose_mesh_inputs["B"].clear()
    window.pose_scale_inputs["B"].setValue(1)
    window.load_snapshot(saved)
    assert not window._pose_targets
    assert window.pose_mesh_inputs["B"].text() == path and window.pose_scale_inputs["B"].value() == .01
    window.pose_seed_inputs["B"].setCurrentIndex(1)
    window.refine_pose("B")
    target = window._pose_targets["B"]
    assert target.frame.metadata["sequence"] == 17 and target.mask is None and target.initialization == "seed"
    np.testing.assert_array_equal(target.initial_pose, initial)
    before = target.session_id
    window.pose_scale_inputs["B"].setValue(.02)
    window.refine_pose("B")
    assert window._pose_targets["B"].session_id == before


def test_sam3d_configured_monocular_texture_and_seed_scale_once(make_window):
    window = make_window(sam3d=True, foundationpose=True,
                         sam3d_depth_source="monocular", sam3d_texture_baking=True)
    observe(window)
    window._task_flow = {"phase": "done"}
    window.generate_mesh("A")
    window._task_flow = None
    request = window.receiver.mesh_requests[-1]
    assert request.depth_source == "monocular" and request.texture_baking
    initial = np.eye(4)
    initial[:3, 3] = (.1, .2, .3)
    result = replace(completed_mesh(request), mesh_path=window.pose_mesh_inputs["A"].text(),
                     camera_from_object=initial, length_unit="monocular_predicted_scale")
    window.receiver.mesh_events.append(result)
    window.poll_once()
    assert "monocular_predicted_scale" in window.mesh_status.text()
    window.pose_scale_inputs["A"].setValue(2.)
    window.refine_pose("A")
    target = window._pose_targets["A"]
    np.testing.assert_allclose(target.initial_pose[:3, 3], initial[:3, 3] * 2)
    assert target.mesh_scale_to_m == 2 and target.frame is window._mesh_bindings["A"]["frame"]
    assert "실측 단위 아님" in window.pose_status.text()


@pytest.mark.parametrize("depth_source,texture_baking", [
    ("depth", False), ("depth", True), ("monocular", False), ("monocular", True),
])
@pytest.mark.parametrize("entry", ["manual", "mesh_to_pose", "task", "vlm"])
def test_sam3d_all_request_paths_use_configured_options(make_window, tmp_path, depth_source, texture_baking, entry):
    path = tmp_path / "config.yaml"
    path.write_text(f"sam3d:\n  depth_source: {depth_source}\n  texture_baking: {str(texture_baking).lower()}\n")
    settings = load_run_config(path)
    window = make_window(sam3d=True, foundationpose=True, vlm=entry == "vlm",
                         sam3d_depth_source=settings["sam3d_depth_source"],
                         sam3d_texture_baking=settings["sam3d_texture_baking"])
    if entry == "vlm":
        window.receiver.latest = frame()
        window.poll_once()
        window.start_vlm()
        window.receiver.vlm_latest = semantic_result(window.receiver.vlm_requests[-1])
        window.poll_once()
        window.receiver.sam3_latest = sam3_result(window._prompts.revision)
        window.poll_once()
    else:
        observe(window)
        if entry == "task":
            window.pose_mesh_inputs["A"].clear()
            window.source_role_input.setCurrentIndex(1)
            window.start_task_pipeline()
        elif entry == "mesh_to_pose":
            window.generate_mesh_and_start_pose("A")
        else:
            window.generate_mesh("A")
    assert window.receiver.mesh_requests
    for request in window.receiver.mesh_requests:
        assert request.depth_source == depth_source and request.texture_baking is texture_baking
    assert not hasattr(window, "mesh_depth_input") and not hasattr(window, "mesh_texture_input")
    assert f"depth_source={depth_source}" in window.mesh_options_label.text()
    assert f"texture_baking={str(texture_baking).lower()}" in window.mesh_options_label.text()


def test_explicit_bundle_load_checks_geometry_without_generation(make_window, tmp_path):
    window = make_window(graspgenx=True)
    request = SimpleNamespace(asset_id="saved-object", mesh_path=window.pose_mesh_inputs["A"].text(), mesh_scale_to_m=.01)
    identity = grasp_identity(request, gripper_id="example", gripper_metadata={}, model_metadata={}, inference_settings={})
    poses, scores = np.eye(4)[None], np.array([.8])
    path = save_grasp_bundle(tmp_path / "grasps", identity, poses, scores,
                            raw_poses=poses, raw_scores=scores, sample_points=np.empty((0, 3)), selected_indices=np.array([0]))
    result = window.load_grasp_bundle_path("A", Path(path).parent)
    assert result.cached and result.bundle_path == path and not window.receiver.grasp_requests
    np.testing.assert_array_equal(window._selected_grasp("A"), poses[0])
    window.pose_scale_inputs["A"].setValue(.02)
    assert window.load_grasp_bundle_path("A", path) is None
    assert not window._grasp_results


def test_mask_canvas_pixel_prompts_preserve_mask_and_distinguish_negative_boxes(app):
    rgb, mask = np.zeros((100, 200, 3), np.uint8), np.ones((100, 200), bool)
    canvas = MaskCanvas(rgb, mask)
    canvas.resize(400, 400)
    canvas.redraw()
    def event(x, y, right=False):
        return SimpleNamespace(position=lambda: QtCore.QPointF(x, y),
            button=lambda: QtCore.Qt.MouseButton.RightButton if right else QtCore.Qt.MouseButton.LeftButton)
    canvas.mousePressEvent(event(100, 150))
    canvas.mousePressEvent(event(200, 200, True))
    canvas.mode = "box"
    canvas.mousePressEvent(event(40, 120, True))
    canvas.mouseReleaseEvent(event(300, 250, True))
    assert canvas.points == [{"xy": [50., 25.], "label": 1}, {"xy": [100., 50.], "label": 0}]
    assert canvas.boxes == [{"xyxy": [20., 10., 150., 75.], "label": False}]
    assert canvas.image_position(QtCore.QPointF(100, 50)) is None
    np.testing.assert_array_equal(canvas.mask, mask)
    assert not rgb.any()


def test_mask_editor_requests_worker_then_updates_frozen_candidates_not_live_prompts(make_window, monkeypatch):
    window = make_window(sam3d=True, foundationpose=True)
    original = observe(window)
    frozen_mask = original.objects[0].masks[0].copy()

    class Dialog:
        def __init__(self, rgb, mask, parent):
            np.testing.assert_array_equal(mask, frozen_mask)
            self.canvas = SimpleNamespace(boxes=[{"xyxy": [1, 2, 5, 6], "label": True}], points=[{"xy": [3, 4], "label": 0}])

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr("grasp_workbench.ui.mask_editor.MaskEditor", Dialog)
    request = window.edit_mask("A")
    assert request.frame.metadata["sequence"] == original.frame.metadata["sequence"]
    assert request.boxes[0]["xyxy"] == [1, 2, 5, 6] and request.points[0]["label"] == 0
    np.testing.assert_array_equal(request.seed_mask, frozen_mask)
    refined = deepcopy(original)
    refined.revision = request.request_id
    refined.objects = (refined.objects[0],)
    refined.objects[0].masks[:, 0, 0] = True
    calls, original_display = [], window._display_frame
    monkeypatch.setattr(window, "_display_frame", lambda *args, **kwargs: (calls.append(args[0]), original_display(*args, **kwargs))[-1])
    before = len(window.receiver.prompts)
    window.receiver.refine_latest = RefineResult(request.request_id, "done", refined)
    window.receiver.latest = frame(200)
    window.poll_once()
    assert len(calls) == 1 and len(window.receiver.prompts) == before
    assert window._snapshot_active and window._snapshot.sam3.revision == request.request_id
    assert {item.object_id for item in window._snapshot.sam3.objects} == {"A", "B"}
    assert window._sam3_result is original
    window.generate_mesh("A")
    generated = window.receiver.mesh_requests[-1]
    assert generated.revision == window._prompts.revision and generated.mask[0, 0]
    assert generated.frame.metadata["sequence"] == request.frame.metadata["sequence"]


def test_fk_attach_updates_base_preview_only_and_detach_restores_local_view(make_window):
    window = make_window(grasp_selection=True)
    add_grasps(window, "B")
    window.selection_object_input.setCurrentIndex(1)
    window.selection_mode_input.setCurrentIndex(window.selection_mode_input.findData("dual"))
    window.request_grasp_selection()
    chosen = window.receiver.selection_requests[-1]
    deliver_selection(window, chosen)
    pinned = deepcopy(window.selection_reference())
    attach = window.attach_grasp_fk()
    assert attach.action == "attach" and attach.attachment == pinned
    obj = chosen.objects[0]
    pose = np.eye(4)
    pose[:3, 3] = (1., 2., 3.)
    row = {**obj, "base_from_object": pose.tolist(), "state": "attached"}
    window.receiver.fk_latest = SelectionResult(attach.request_id, "done", {
        "kind": "grasp_fk", "state": "attached", "objects": {obj["observation_id"]: row}, "robot_command_published": False})
    window.poll_once()
    assert not window._pose_targets and not window._pose_objects
    assert "base_link" in window.axes.get_title() and "Grasp FK" in window.fk_status.text()
    mesh = window._fk_meshes["B"]
    np.testing.assert_allclose(artist_triangles(window._fk_artists["B"]["mesh"]), (mesh.vertices + pose[:3, 3])[mesh.faces])
    assert {key for key in window._fk_artists if isinstance(key, tuple)} == {("B", "left"), ("B", "right")}
    assert window.selection_reference() == pinned
    selected = window._selection_object_record("B", {"observation_id": obj["observation_id"], "instance_id": obj["instance_id"]})
    assert selected["grasp_fk"]["base_from_object"] == pose.tolist()
    detach = window.detach_grasp_fk()
    window.receiver.fk_latest = SelectionResult(detach.request_id, "done", {"kind": "grasp_fk", "state": "detached", "objects": {}})
    window.poll_once()
    assert window._grasp_fk is None and "object coordinates" in window.axes.get_title()
