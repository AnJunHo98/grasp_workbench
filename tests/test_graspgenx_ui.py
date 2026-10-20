"""Grasp UI request binding and canonical transforms; no model or robot process."""
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult
from grasp_workbench.ipc.graspgenx import GraspResult
from grasp_workbench.ui.camera_window import CameraWindow
from grasp_workbench.ui.grasp_preview import grasp_axis_points, transform_points
from test_camera_ui import FakeReceiver as BaseReceiver, completed_mesh
from test_grasp_preview import descriptor as descriptor_fixture
from test_pose_mesh_ui import artist_triangles, mesh_path as mesh_path_fixture
from test_pose_rgb_overlay import frame, sam3_result

descriptor, mesh_path = descriptor_fixture, mesh_path_fixture


class FakeReceiver(BaseReceiver):
    def __init__(self, service, *, graspgenx=False, **kwargs):
        super().__init__(service, **kwargs)
        self.graspgenx, self.grasp_requests, self.grasp_events = graspgenx, [], []

    def submit_grasp(self, request):
        self.grasp_requests.append(request)

    def take_grasp_results(self):
        events, self.grasp_events = self.grasp_events, []
        return events


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, mesh_path):
    window = CameraWindow(graspgenx=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    for key in ("A", "B"):
        window.pose_mesh_inputs[key].setText(str(mesh_path))
        window.pose_scale_inputs[key].setValue(0.01)
    yield window
    window.receiver.alive = False
    window.close()
    app.processEvents()


def done(request, descriptor=None):
    grasps = np.repeat(np.eye(4)[None], 2, axis=0)
    grasps[0, :3, 3] = (0.02, 0.03, 0.04)
    grasps[1, :3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    grasps[1, :3, 3] = (0.05, -0.02, 0.01)
    return GraspResult(request.request_id, request.asset_id, request.object_id, "done", request.mesh_path,
                       request.mesh_scale_to_m, "canonical-gripper", grasps, np.array([0.91, 0.73]),
                       gripper_metadata=descriptor)


def deliver(window, result):
    window.receiver.grasp_events.append(result)
    window.poll_once()


def test_manual_mesh_without_fp_or_sam3_generates_object_local_candidates(window, descriptor):
    assert window.receiver.graspgenx and not window.receiver.foundationpose and not window.receiver.sam3
    assert not window.pose_start_buttons and not window._pose_targets
    window.generate_grasp("A")
    request, = window.receiver.grasp_requests
    assert request.object_id == "A" and request.mesh_scale_to_m == 0.01
    assert not request.regenerate and request.asset_id not in {"A", "B"}
    result = done(request, descriptor)
    deliver(window, result)
    assert window.grasp_inputs["A"].count() == 2 and window.grasp_inputs["A"].currentData() == 0
    assert window.grasp_scores["A"].text() == "score: 0.9100"
    assert "object coordinates" in window.axes.get_title()
    mesh = window._grasp_object_meshes["A"]
    np.testing.assert_allclose(mesh.vertices.min(axis=0), (0.1, 0.2, 0.3), atol=1e-7)
    preview = window._grasp_previews["A"][0]
    mesh_artist = window._grasp_artists["A"]["mesh"]
    np.testing.assert_allclose(artist_triangles(mesh_artist),
                               transform_points(preview.vertices, result.grasps[0])[preview.faces], atol=1e-7)
    window.grasp_inputs["A"].setCurrentIndex(1)
    assert window._grasp_artists["A"]["mesh"] is mesh_artist
    assert window.grasp_scores["A"].text() == "score: 0.7300"
    np.testing.assert_allclose(artist_triangles(mesh_artist),
                               transform_points(preview.vertices, result.grasps[1])[preview.faces], atol=1e-7)


def test_regenerate_replaces_request_but_keeps_geometry_identity_and_ignores_old_completion(window):
    window.generate_grasp("A")
    first = window.receiver.grasp_requests[-1]
    window.generate_grasp("A", regenerate=True)
    second = window.receiver.grasp_requests[-1]
    assert first.request_id != second.request_id and first.asset_id == second.asset_id
    assert second.regenerate
    deliver(window, done(first))
    assert not window._grasp_results
    deliver(window, done(second))
    assert window._grasp_results["A"].request_id == second.request_id


def test_mesh_scale_change_cancels_old_binding_and_never_attaches_old_candidates(window):
    window.generate_grasp("A")
    previous = window.receiver.grasp_requests[-1]
    deliver(window, done(previous))
    window.pose_scale_inputs["A"].setValue(0.02)
    assert "A" not in window._grasp_results and window.grasp_inputs["A"].currentData() is None
    deliver(window, done(previous))
    assert "A" not in window._grasp_results
    window.generate_grasp("A")
    next_request = window.receiver.grasp_requests[-1]
    assert next_request.asset_id != previous.asset_id and next_request.mesh_scale_to_m == 0.02
    deliver(window, replace(done(next_request), mesh_scale_to_m=0.01))
    assert "A" not in window._grasp_results
    deliver(window, done(next_request))
    assert "A" in window._grasp_results


def test_a_b_are_independent_object_slots_with_one_selected_candidate_each(window):
    for key in ("A", "B"):
        window.generate_grasp(key)
        deliver(window, done(window.receiver.grasp_requests[-1]))
    a, b = window.receiver.grasp_requests
    assert a.asset_id == b.asset_id and a.object_id != b.object_id
    window.grasp_inputs["A"].setCurrentIndex(1)
    assert window.grasp_inputs["B"].currentData() == 0
    window.grasp_object_input.setCurrentIndex(0)
    assert "A ·" in window.axes.get_title() and len(window.axes.lines) == 3
    window.grasp_object_input.setCurrentIndex(1)
    assert "B ·" in window.axes.get_title() and len(window.axes.lines) == 3


def test_backend_error_is_shown_without_starting_fp_or_retrying(window):
    window.generate_grasp("A")
    request = window.receiver.grasp_requests[-1]
    deliver(window, replace(done(request), status="error", grasps=None, scores=None, error="backend failed"))
    assert "backend failed" in window.grasp_status.text()
    assert len(window.receiver.grasp_requests) == 1 and not window._pose_targets


def test_fp_pose_compose_selected_canonical_grasp_reuses_artists_and_latest_rgb(app, mesh_path, descriptor, monkeypatch):
    import grasp_workbench.ui.grasp_preview as preview_module

    window = CameraWindow(foundationpose=True, graspgenx=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    try:
        for key in ("A", "B"):
            window.prompt_inputs[key].setText(key)
            window.pose_mesh_inputs[key].setText(str(mesh_path))
            window.pose_scale_inputs[key].setValue(0.01)
        window.apply_prompts()
        source = sam3_result(window._prompts.revision)
        window.receiver.sam3_latest = source
        window.poll_once()
        window.start_pose("A")
        window.generate_grasp("A")
        request = window.receiver.grasp_requests[-1]
        result = done(request, descriptor)
        target = window._pose_targets["A"]
        matrix = np.eye(4)
        matrix[:3, :3] = ((0, 0, 1), (0, 1, 0), (-1, 0, 0))
        matrix[:3, 3] = (-0.1, -0.1, 1.0)
        item = ObjectPose("A", target.session_id, target.mesh_path, "tracking", matrix)
        window.receiver.latest = frame(sequence=99, value=(12, 34, 56))
        window.receiver.pose_latest = PoseResult(source.frame, (item,))
        window.receiver.grasp_events.append(result)
        displays = []
        original = window._display_frame

        def record_display(*args, **kwargs):
            displays.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(window, "_display_frame", record_display)
        window.poll_once()
        assert len(displays) == 1 and displays[0].metadata["sequence"] == 99
        assert window.image_view.image.pixelColor(0, 0).getRgb()[:3] == (12, 34, 56)
        window.grasp_inputs["A"].setCurrentIndex(1)
        artists = window._grasp_artists["A"]
        visual = window._grasp_previews["A"][0]
        expected = matrix @ result.grasps[1]
        np.testing.assert_allclose(artist_triangles(artists["mesh"]),
                                   transform_points(visual.vertices, expected)[visual.faces], atol=1e-7)
        axes = grasp_axis_points(expected)
        np.testing.assert_allclose(artists["axes"][0].get_data_3d(), np.stack((axes[0], axes[1])).T)
        pixels = axes[:, :2] / axes[:, 2:3] * 400 + (160, 120)
        x, y = np.rint((pixels[0] + pixels[1]) / 2).astype(int)
        assert any(window.image_view.image.pixelColor(x + dx, y + dy).getRgb()[:3] == (255, 60, 60)
                   for dx in range(-2, 3) for dy in range(-2, 3))
        monkeypatch.setattr(preview_module, "load_gripper_preview", lambda *_args: pytest.fail("per-frame descriptor reload"))
        matrix = matrix.copy()
        matrix[0, 3] += 0.1
        window.receiver.pose_latest = PoseResult(source.frame, (replace(item, camera_from_object=matrix),))
        window.poll_once()
        assert window._grasp_artists["A"]["mesh"] is artists["mesh"]
        assert len(window.receiver.grasp_requests) == 1
        np.testing.assert_allclose(artist_triangles(artists["mesh"]),
                                   transform_points(visual.vertices, matrix @ result.grasps[1])[visual.faces], atol=1e-7)
        window.receiver.pose_latest = PoseResult(source.frame, (replace(item, status="recovering", camera_from_object=None),))
        window.poll_once()
        assert "A" not in window._grasp_artists
        window.stop_pose("A")
        assert not window._pose_targets and "object coordinates" in window.axes.get_title()
    finally:
        window.receiver.alive = False
        window.close()
        app.processEvents()


def test_active_fp_geometry_is_request_source_and_new_start_changes_binding(app, mesh_path, tmp_path):
    window = CameraWindow(foundationpose=True, graspgenx=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    try:
        window.prompt_inputs["A"].setText("cup")
        window.pose_mesh_inputs["A"].setText(str(mesh_path))
        window.pose_scale_inputs["A"].setValue(0.01)
        window.apply_prompts()
        window.receiver.sam3_latest = sam3_result(window._prompts.revision)
        window.poll_once()
        window.start_pose("A")
        replacement_mesh = tmp_path / "other.ply"
        replacement_mesh.write_bytes(mesh_path.read_bytes())
        window.pose_mesh_inputs["A"].setText(str(replacement_mesh))
        window.pose_scale_inputs["A"].setValue(0.02)
        window.generate_grasp("A")
        request = window.receiver.grasp_requests[-1]
        assert request.mesh_path == str(mesh_path) and request.mesh_scale_to_m == 0.01
        deliver(window, done(request))
        assert "A" in window._grasp_results
        window.start_pose("A")
        assert window._pose_targets["A"].mesh_path == str(replacement_mesh)
        assert "A" not in window._grasp_results
        deliver(window, done(request))
        assert "A" not in window._grasp_results
    finally:
        window.receiver.alive = False
        window.close()
        app.processEvents()


def test_sam3d_mesh_can_feed_grasp_without_fp(app, mesh_path):
    window = CameraWindow(sam3d=True, graspgenx=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    try:
        window.prompt_inputs["A"].setText("cup")
        window.apply_prompts()
        window.receiver.sam3_latest = sam3_result(window._prompts.revision)
        window.poll_once()
        window.generate_mesh("A")
        result = completed_mesh(window.receiver.mesh_requests[-1])
        result.mesh_path = str(mesh_path)
        result.length_unit, result.mesh_scale_to_m = "metre", 1.0
        window.receiver.mesh_events.append(result)
        window.poll_once()
        assert window.pose_mesh_inputs["A"].text() == str(mesh_path)
        window.generate_grasp("A")
        request, = window.receiver.grasp_requests
        assert request.mesh_path == str(mesh_path) and request.mesh_scale_to_m == 1
        assert not window.receiver.foundationpose and not window._pose_targets
    finally:
        window.receiver.alive = False
        window.close()
        app.processEvents()
