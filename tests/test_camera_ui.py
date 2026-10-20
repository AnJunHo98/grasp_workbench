"""Offscreen display tests with a fake receiver; no SDK or iceoryx2 service opens."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ui.camera_window import CameraWindow, RgbImageView


class FakeReceiver:
    def __init__(self, service, *, sam3=False, sam3d=False, foundationpose=False):
        self.service, self.started, self.alive, self.stopped = service, False, False, False
        self.finish_on_stop = True
        self.latest, self.take_count = None, 0
        self.sam3, self.sam3_latest, self.prompts = sam3, None, []
        self.sam3d, self.mesh_requests, self.mesh_events = sam3d, [], []
        self.foundationpose, self.pose_controls, self.pose_latest = foundationpose, [], None
        self.stats = {"ready": True, "error": None, "received_count": 0, "elapsed_s": 0,
                      "last_received_monotonic_s": None}

    def start(self):
        self.started = self.alive = True

    def stop(self):
        self.stopped = True
        if self.finish_on_stop:
            self.alive = False

    def is_alive(self):
        return self.alive

    def take_latest(self):
        self.take_count += 1
        value, self.latest = self.latest, None
        return value

    def snapshot(self):
        return self.stats.copy()

    def set_prompts(self, prompts):
        self.prompts.append(prompts)
        self.sam3_latest = None

    def take_sam3_latest(self):
        result, self.sam3_latest = self.sam3_latest, None
        return result

    def submit_mesh(self, request):
        self.mesh_requests.append(request)

    def take_mesh_results(self):
        events, self.mesh_events = self.mesh_events, []
        return events

    def mesh_preview(self, object_id, asset_id):
        return None

    def set_pose_control(self, control):
        self.pose_controls.append(control)

    def take_pose_latest(self):
        result, self.pose_latest = self.pose_latest, None
        return result


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    window = CameraWindow(service="fake/rgbd", receiver_factory=FakeReceiver)
    window.timer.stop()
    yield window
    window.receiver.finish_on_stop = True
    window.receiver.alive = False
    window.close()
    app.processEvents()


def test_rgb_colors_are_owned_and_aspect_fit_after_resize(app):
    view = RgbImageView()
    view.resize(400, 300)
    view.show()
    app.processEvents()
    rgb = np.array([[[255, 0, 0], [1, 2, 3], [0, 255, 0], [1, 2, 3]],
                    [[0, 0, 255], [1, 2, 3], [255, 255, 0], [1, 2, 3]]], dtype=np.uint8)
    view.set_rgb(rgb[:, ::2])
    rgb[:] = 0
    assert view.image.pixelColor(0, 0).getRgb()[:3] == (255, 0, 0)
    assert view.image.pixelColor(1, 0).getRgb()[:3] == (0, 255, 0)
    assert view.image.pixelColor(0, 1).getRgb()[:3] == (0, 0, 255)
    assert view.pixmap().width() == view.pixmap().height() == 300
    view.resize(500, 200)
    app.processEvents()
    assert view.pixmap().width() == view.pixmap().height() == 200
    view.close()


def test_waiting_is_not_presented_as_camera_success(window, app):
    window.show()
    app.processEvents()
    window.poll_once()
    assert window.receiver.started and window.receiver.service == "fake/rgbd"
    assert "영상 대기" in window.status.text()
    assert "수신 완료" not in window.status.text()
    assert window.image_view.image is None
    assert window.isVisible() and not window.receiver.stopped


def test_latest_frame_only_and_empty_3d_view_without_invented_pose(window):
    receiver = window.receiver
    for sequence in range(1000):
        receiver.latest = CameraFrame(np.full((2, 4, 3), sequence % 256, dtype=np.uint8),
                                      np.ones((2, 4), dtype=np.float32),
                                      {"camera_id": "fake", "sequence": sequence})
    receiver.stats.update(received_count=1000, last_received_monotonic_s=time.monotonic() - 2)
    window.poll_once()
    assert receiver.take_count == 1 and receiver.latest is None
    assert "sequence=999" in window.metadata_label.text()
    assert window.image_view.image.pixelColor(0, 0).red() == 999 % 256
    assert window.axes.name == "3d"
    assert window.axes.get_title() == "Mesh / Pose · Waiting for data"
    assert (window.axes.get_xlabel(), window.axes.get_ylabel(), window.axes.get_zlabel()) == ("X", "Y", "Z")
    assert len(window.axes.lines) == len(window.axes.collections) == 0
    assert not hasattr(window, "fps_curve") and not hasattr(window, "interval_curve")
    assert "마지막 수신 2." in window.status.text()
    window.poll_once()
    assert receiver.take_count == 2 and window.received_count == 1000


def test_close_stops_only_owned_receiver_and_waits_asynchronously(window, app):
    window.receiver.finish_on_stop = False
    window.show()
    app.processEvents()
    window.timer.start()
    window.close()
    assert window.receiver.stopped and window.receiver.alive
    assert window.isVisible() and window.timer.isActive()
    window.receiver.alive = False
    window.poll_once()
    assert not window.isVisible() and not window.timer.isActive()


def test_receiver_error_is_nonzero_and_visible(window):
    window.receiver.stats["error"] = "injected iceoryx2 error"
    window.poll_once()
    assert window.exit_code == 1 and window.receiver.stopped
    assert "injected iceoryx2 error" in window.status.text()


def test_receiver_cleanup_error_is_not_lost_during_asynchronous_close(window):
    window.receiver.finish_on_stop = False
    window.close()
    window.receiver.stats["error"] = "subscriber close failed"
    window.receiver.alive = False
    window.poll_once()
    assert window.exit_code == 1 and "subscriber close failed" in window.status.text()


def test_finite_received_frames_close_successfully(window):
    window.frames = 3
    window.receiver.stats["received_count"] = 3
    window.poll_once()
    assert window.exit_code == 0 and window.receiver.stopped
    assert window.received_count == 3 and "수신 완료" in window.status.text()


@pytest.mark.parametrize("frames,received,expected_code", [(3, 0, 1), (3, 2, 1), (0, 0, 0), (0, 2, 0)])
def test_timeout_preserves_monitor_contract(window, frames, received, expected_code):
    window.frames, window.timeout = frames, 1
    window._started -= 2
    window.receiver.stats["received_count"] = received
    window.poll_once()
    assert window.exit_code == expected_code and window.receiver.stopped
    assert f"수신 {received}" in window.status.text()


def test_ui_worker_help_does_not_import_optional_gui_or_camera_sdk():
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.modules['PyQt6'] = None
sys.modules['matplotlib'] = None
sys.modules['pyzed.sl'] = None
sys.modules['pyrealsense2'] = None
from grasp_workbench.workers.worker_camera_ui import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--service" in result.stdout and "--frames" in result.stdout and "--foundationpose" in result.stdout


@pytest.mark.parametrize("raises", [False, True])
def test_ui_worker_stops_and_joins_receiver_when_event_loop_exits_without_close(app, monkeypatch, raises):
    from grasp_workbench.ui import camera_window
    from grasp_workbench.workers.worker_camera_ui import main

    calls = []

    class Receiver:
        def stop(self):
            calls.append("stop")

        def join(self):
            calls.append("join")

    class Window:
        exit_code = 0
        receiver = Receiver()

        def __init__(self, *args):
            pass

        def show(self):
            pass

    def exec_loop():
        if raises:
            raise RuntimeError("injected event loop failure")
        return 0

    monkeypatch.setattr(camera_window, "CameraWindow", Window)
    monkeypatch.setattr(QtWidgets.QApplication, "exec", staticmethod(exec_loop))
    assert main([]) == (1 if raises else 0)
    assert calls == ["stop", "join"]


@pytest.fixture
def sam3_window(app):
    window = CameraWindow(service="fake/rgbd", sam3=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    yield window
    window.receiver.alive = False
    window.close()
    app.processEvents()


def sam3_result(revision, sequence=7, *, count=2, error=None, include_b=False):
    from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result

    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    masks = np.zeros((count, 60, 80), dtype=bool)
    masks[:, 30:40, 30:40] = True
    boxes = np.tile(np.array([[20, 20, 50, 50]], dtype=np.float32), (count, 1))
    objects = [Sam3ObjectResult("A", "cup", masks, boxes, np.ones(count, dtype=np.float32), error)]
    if include_b:
        objects.append(Sam3ObjectResult("B", "bottle", masks[:1].copy(), boxes[:1].copy(),
                                       np.ones(min(count, 1), dtype=np.float32)))
    frame = CameraFrame(rgb, np.ones((60, 80), dtype=np.float32), {
        "camera_id": "fake", "sequence": sequence,
        "intrinsics": {"fx": 80., "fy": 80., "cx": 40., "cy": 30.},
    })
    return Sam3Result(frame, revision, tuple(objects))


def test_prompt_slots_initially_stopped_omit_blanks_and_use_new_revisions(sam3_window):
    window = sam3_window
    assert window.receiver.sam3 and window.receiver.prompts[-1].objects == ()
    initial = window.receiver.prompts[-1].revision
    window.prompt_inputs["A"].setText("  red cup  ")
    window.prompt_inputs["B"].setText(" ")
    window.apply_prompts()
    applied = window.receiver.prompts[-1]
    assert [(item.object_id, item.text) for item in applied.objects] == [("A", "red cup")]
    assert applied.revision != initial
    window.apply_prompts()
    assert window.receiver.prompts[-1].revision != applied.revision
    window._set_prompts(())
    assert window.receiver.prompts[-1].objects == ()
    assert window.prompt_inputs["A"].text().strip() == "red cup"


def test_prompt_language_hint_and_examples_do_not_translate_or_reject_korean(sam3_window, app):
    window = sam3_window
    window.show()
    app.processEvents()
    assert window.prompt_hint.isVisible()
    assert "입력 문구를 SAM3에 그대로 전달 · 자동 번역 없음" in window.prompt_hint.text()
    for key, example in (("A", "cell phone"), ("B", "orange ball")):
        assert example in window.prompt_inputs[key].placeholderText()
        assert "비우면 제외" in window.prompt_inputs[key].placeholderText()
    window.prompt_inputs["A"].setText("휴대폰")
    window.prompt_inputs["B"].setText("탁구공")
    window.apply_prompts()
    assert [(item.object_id, item.text) for item in window.receiver.prompts[-1].objects] == [("A", "휴대폰"), ("B", "탁구공")]


def test_result_overlay_keeps_source_unchanged_and_uses_latest_camera_rgb(sam3_window):
    window = sam3_window
    window.prompt_inputs["A"].setText("cup")
    window.apply_prompts()
    result = sam3_result(window._prompts.revision)
    untouched = result.frame.rgb.copy()
    window.receiver.sam3_latest = result
    window.poll_once()
    assert "sequence=7" in window.metadata_label.text() and "SAM3 결과의 원본 프레임" in window.metadata_label.text()
    assert "A: 2개 후보" in window.sam3_status.text()
    assert window.image_view.image.pixelColor(35, 35).green() > 0
    overlay = window.image_view.image.copy()
    np.testing.assert_array_equal(result.frame.rgb, untouched)
    newer = CameraFrame(np.full((60, 80, 3), 200, dtype=np.uint8), result.frame.depth_m, {"sequence": 99})
    window.receiver.latest = newer
    window.poll_once()
    assert window.image_view.image != overlay and "sequence=99" in window.metadata_label.text()
    assert "실시간 카메라" in window.metadata_label.text()
    assert "SAM3 overlay sequence=7" in window.metadata_label.text()
    assert window.image_view.image.pixelColor(70, 55).green() == 200
    np.testing.assert_array_equal(result.frame.rgb, untouched)
    window.prompt_inputs["A"].setText("different object")
    window.apply_prompts()
    assert "sequence=99" in window.metadata_label.text()
    assert window.image_view.image.pixelColor(35, 35).green() == 200
    window.receiver.sam3_latest = result
    window.poll_once()
    assert window.image_view.image.pixelColor(35, 35).green() == 200
    assert window._sam3_result is None
    assert len(window.axes.lines) == len(window.axes.collections) == 0


def test_object_errors_and_missing_slots_stay_visible_without_closing(sam3_window):
    window = sam3_window
    window.prompt_inputs["A"].setText("cup")
    window.prompt_inputs["B"].setText("bottle")
    window.apply_prompts()
    window.receiver.sam3_latest = sam3_result(window._prompts.revision, count=0, error="model failed")
    window.poll_once()
    assert "A: 0개 후보" in window.sam3_status.text() and "model failed" in window.sam3_status.text()
    assert "B: 결과 없음" in window.sam3_status.text()
    assert window.exit_code == 0 and not window.receiver.stopped


def test_stop_clears_overlay_and_new_raw_frames_resume(sam3_window):
    window = sam3_window
    window.prompt_inputs["A"].setText("cup")
    window.apply_prompts()
    result = sam3_result(window._prompts.revision)
    window.receiver.sam3_latest = result
    window.poll_once()
    window._set_prompts(())
    assert window.image_view.image.pixelColor(35, 35).green() == 0
    window.receiver.latest = CameraFrame(np.full((60, 80, 3), 123, dtype=np.uint8), result.frame.depth_m,
                                        {"sequence": 100})
    window.poll_once()
    assert window.image_view.image.pixelColor(35, 35).green() == 123
    assert "sequence=100" in window.metadata_label.text()


@pytest.mark.parametrize("feature", ["sam3", "foundationpose"])
def test_sam3_ui_imports_transport_not_model_sdk_or_vlm(app, feature):
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
from PyQt6.QtWidgets import QApplication
from grasp_workbench.ui.camera_window import CameraWindow
class Receiver:
    def __init__(self,*args,**kwargs): pass
    def start(self): pass
    def stop(self): pass
    def is_alive(self): return False
    def set_prompts(self,value): pass
    def set_pose_control(self,value): pass
    def snapshot(self): return {'error':None}
app=QApplication([])
window=CameraWindow(**{sys.argv[1]:True},receiver_factory=Receiver)
assert not any(name.split('.')[0] in {'torch','sam3','openai','pyzed','pyrealsense2','estimater','trimesh'} for name in sys.modules)
window.close()
"""
    result = subprocess.run([sys.executable, "-c", script, feature], cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def sam3d_window(app):
    window = CameraWindow(sam3d=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    window.prompt_inputs["A"].setText("cup")
    window.prompt_inputs["B"].setText("bottle")
    window.apply_prompts()
    yield window
    window.receiver.alive = False
    window.close()
    app.processEvents()


def completed_mesh(request, *, status="done", error=None):
    from grasp_workbench.ipc.sam3d import MeshGeometry, MeshResult

    mesh = MeshGeometry(np.array([[0, 0, 0], [2, 0, 0], [0, 4, 0]], dtype=np.float32),
                        np.array([[0, 1, 2]], dtype=np.uint32),
                        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8))
    return MeshResult(request.request_id, request.asset_id, request.object_id, request.revision, status,
                       mesh=mesh if status == "done" else None, mesh_path="generated/object.glb", error=error,
                       length_unit="metre" if status == "done" else None,
                       camera_from_object=np.eye(4) if status == "done" else None)


def test_mesh_requires_explicit_multiple_candidate_selection_and_freezes_source(sam3d_window):
    window = sam3d_window
    source = sam3_result(window._prompts.revision, sequence=7, include_b=True)
    window.receiver.sam3_latest = source
    window.poll_once()
    assert window.sam3 and window.receiver.sam3d
    assert window.candidate_inputs["A"].currentData() is None
    window.generate_mesh("A")
    assert not window.receiver.mesh_requests
    assert window._mesh_bindings["B"]["candidate_index"] == 0
    window.candidate_inputs["A"].setCurrentIndex(1)
    binding = window._mesh_bindings["A"]
    mask = binding["mask"].copy()
    source.frame.rgb[:] = 222
    source.objects[0].masks[:] = False
    window.generate_mesh("A")
    first = window.receiver.mesh_requests[-1]
    assert first.frame.metadata["sequence"] == 7 and not first.frame.rgb.any()
    np.testing.assert_array_equal(first.mask, mask)
    assert first.asset_id not in {"A", "B", window._mesh_bindings["B"]["asset_id"]}
    window.receiver.sam3_latest = sam3_result(window._prompts.revision, sequence=99, include_b=True)
    window.poll_once()
    assert window._mesh_bindings["A"] is binding
    assert "frame 7" in window.candidate_inputs["A"].currentText()
    assert "frozen frame 7" in window.mesh_status.text()
    window.generate_mesh("A")
    assert len(window.receiver.mesh_requests) == 1
    window.receiver.mesh_events.append(completed_mesh(first))
    window.poll_once()
    assert len(window.axes.collections) == 1
    assert window.axes.get_title() == "A mesh · local preview"
    assert window.axes.get_xlabel() == "X (m)"
    assert "asset 로컬 좌표" in window.mesh_status.text()
    window.generate_mesh("A")
    second = window.receiver.mesh_requests[-1]
    assert second.request_id != first.request_id and second.asset_id == first.asset_id
    assert second.frame is first.frame and second.mask is first.mask


def test_candidate_change_gets_new_asset_and_prompt_edit_invalidates_binding(sam3d_window):
    window = sam3d_window
    window.receiver.sam3_latest = sam3_result(window._prompts.revision)
    window.poll_once()
    window.candidate_inputs["A"].setCurrentIndex(1)
    first = window._mesh_bindings["A"]
    window.candidate_inputs["A"].setCurrentIndex(2)
    second = window._mesh_bindings["A"]
    assert first["asset_id"] != second["asset_id"]
    assert second["candidate_index"] == 1 and second["frame"].metadata["sequence"] == 7
    window.prompt_inputs["A"].setText("new prompt before Apply")
    assert "A" not in window._mesh_bindings and not window.mesh_buttons["A"].isEnabled()
    window.receiver.sam3_latest = sam3_result(window._prompts.revision, count=1)
    window.poll_once()
    assert "A" not in window._mesh_bindings
    assert "Apply 필요" in window.mesh_status.text()


def test_mesh_failure_preserves_other_mesh_and_apply_ignores_old_jobs(sam3d_window):
    window = sam3d_window
    window.receiver.sam3_latest = sam3_result(window._prompts.revision, count=1, include_b=True)
    window.poll_once()
    window.generate_mesh("A")
    window.generate_mesh("B")
    a, b = window.receiver.mesh_requests
    window.receiver.mesh_events.extend([completed_mesh(b), completed_mesh(a, status="error", error="model failed")])
    window.poll_once()
    assert "B" in window._mesh_results and "A" not in window._mesh_results
    assert len(window.axes.collections) == 1 and window.axes.get_title() == "B mesh · local preview"
    assert "model failed" in window.mesh_status.text()
    window.apply_prompts()
    window.receiver.mesh_events.append(completed_mesh(a))
    window.poll_once()
    assert not window._mesh_results and not window._mesh_bindings
    assert len(window.axes.collections) == 0


def test_mesh_status_stays_compact_and_full_output_paths_remain_accessible(sam3d_window):
    window = sam3d_window
    window.receiver.sam3_latest = sam3_result(window._prompts.revision, count=1, include_b=True)
    window.poll_once()
    for key in ("A", "B"):
        window.generate_mesh(key)
    paths = []
    for request in window.receiver.mesh_requests:
        result = completed_mesh(request)
        result.mesh_path = "/external/" + "nested-output/" * 20 + request.asset_id + "/object.ply"
        paths.append(result.mesh_path)
        window.receiver.mesh_events.append(result)
    window.poll_once()
    text = window.mesh_status.text()
    assert "A: Done" in text and "B: Done" in text
    assert "frozen frame 7" in text and "candidate 1" in text
    for path in paths:
        assert path not in text and path in window.mesh_status.toolTip()


@pytest.fixture
def pose_window(app):
    window = CameraWindow(foundationpose=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    for key, text in (("A", "cup"), ("B", "bottle")):
        window.prompt_inputs[key].setText(text)
        window.pose_mesh_inputs[key].setText(f"assets/{key}.glb")
    window.apply_prompts()
    yield window
    window.receiver.alive = False
    window.close()
    app.processEvents()


def test_pose_start_uses_current_sam3_candidate_and_stop_preserves_other_slot(pose_window):
    window = pose_window
    assert window.receiver.foundationpose and window.receiver.sam3 and not window.receiver.sam3d
    assert window.receiver.pose_controls[0].targets == ()
    window.start_pose("A")
    assert len(window.receiver.pose_controls) == 1
    window.receiver.sam3_latest = sam3_result(window._prompts.revision, count=2, include_b=True)
    window.poll_once()
    assert window.candidate_inputs["A"].currentData() is None
    window.candidate_inputs["A"].setCurrentIndex(2)
    current = sam3_result(window._prompts.revision, sequence=99, count=2, include_b=True)
    window.receiver.sam3_latest = current
    window.poll_once()
    assert window.candidate_inputs["A"].currentData() == 1
    window.pose_scale_inputs["A"].setValue(0.001)
    window.start_pose("A")
    a = window.receiver.pose_controls[-1].targets[0]
    assert a.object_id == "A" and a.mesh_path == "assets/A.glb" and a.mesh_scale_to_m == 0.001
    assert a.frame is current.frame and a.frame.metadata["sequence"] == 99
    assert a.prompt_revision == current.revision
    np.testing.assert_array_equal(a.mask, current.objects[0].masks[1])
    window.start_pose("B")
    assert len(window.receiver.pose_controls[-1].targets) == 2
    a, b = window.receiver.pose_controls[-1].targets
    assert a.session_id != b.session_id
    window.start_pose("A")
    replacement, kept_b = window.receiver.pose_controls[-1].targets
    assert replacement.session_id != a.session_id and kept_b is b
    window.stop_pose("A")
    assert window.receiver.pose_controls[-1].targets == (b,)
    window.stop_pose("B")
    assert window.receiver.pose_controls[-1].targets == ()


def test_pose_mesh_choices_display_catalog_names_without_renaming_assets(pose_window):
    catalog_path = Path(__file__).resolve().parents[1] / "assets" / "meshes" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    expected = catalog["objects"] + catalog["table_leg"]
    assert len(expected) == 24
    for key in ("A", "B"):
        choice = pose_window.pose_mesh_choices[key]
        assert choice.currentData() is None and choice.count() == 25
        for index, item in enumerate(expected, 1):
            assert choice.itemText(index) == item["display_name"]
            path = item.get("preferred_mesh", item.get("path"))
            assert choice.itemData(index)["mesh_path"] == str(catalog_path.parent / path)
            assert Path(choice.itemData(index)["mesh_path"]).is_file()


def test_pose_named_tape_and_tuna_choices_fill_paths_scales_independently(pose_window):
    catalog_path = Path(__file__).resolve().parents[1] / "assets" / "meshes" / "catalog.json"
    objects = json.loads(catalog_path.read_text(encoding="utf-8"))["objects"]
    for key, asset_id in (("A", "70b913cc9042458585b173a9a07bb111"),
                          ("B", "e76130be39c44e3282d47be1056760f5")):
        item = next(item for item in objects if item["asset_id"] == asset_id)
        pose_window.pose_scale_inputs[key].setValue(0.001)
        choice = pose_window.pose_mesh_choices[key]
        choice.setCurrentIndex(choice.findText(item["display_name"]))
        assert pose_window.pose_mesh_inputs[key].text() == str(catalog_path.parent / item["preferred_mesh"])
        assert pose_window.pose_scale_inputs[key].value() == item["mesh_scale_to_m"]
    assert pose_window.pose_mesh_inputs["A"].text() != pose_window.pose_mesh_inputs["B"].text()
    assert len(pose_window.receiver.pose_controls) == 1


def test_pose_named_table_mesh_uses_catalog_scale_and_manual_edit_clears_choice(pose_window):
    choice = pose_window.pose_mesh_choices["A"]
    choice.setCurrentIndex(choice.count() - 1)
    selected = choice.currentData()
    assert selected["mesh_path"].endswith("/table_leg/Leg001_colored.obj")
    assert pose_window.pose_mesh_inputs["A"].text() == selected["mesh_path"]
    assert pose_window.pose_scale_inputs["A"].value() == 1
    field = pose_window.pose_mesh_inputs["A"]
    field.setText("/another/mesh.obj")
    field.textEdited.emit(field.text())
    assert choice.currentData() is None
    assert field.text() == "/another/mesh.obj"


def test_pose_results_draw_camera_xyz_rotation_axes_and_preserve_single_registration(pose_window):
    from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult

    window = pose_window
    source = sam3_result(window._prompts.revision, count=1, include_b=True)
    window.receiver.sam3_latest = source
    window.poll_once()
    for key in ("A", "B"):
        window.start_pose(key)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (0.1, -0.2, 0.7)
    a, b = (ObjectPose(key, window._pose_targets[key].session_id, f"assets/{key}.glb", "tracking", pose.copy())
            for key in ("A", "B"))
    window.receiver.pose_latest = PoseResult(source.frame, (b,))
    window.poll_once()
    assert len(window.axes.lines) == 3
    window.receiver.pose_latest = PoseResult(source.frame, (a,))
    window.poll_once()
    assert len(window.axes.lines) == 6 and set(window._pose_objects) == {"A", "B"}
    assert "XYZ(m)=0.100, -0.200, 0.700" in window.pose_status.text()
    assert window.axes.get_xlabel() == "X (m)"
    assert "FP pose sequence=7" in window.metadata_label.text()
    np.testing.assert_allclose(window.axes.lines[-3].get_data_3d(),
                               [[0.1, 0.15], [-0.2, -0.2], [0.7, 0.7]])
    window.stop_pose("A")
    assert len(window.axes.lines) == 3
    window.stop_pose("B")
    window.receiver.pose_latest = PoseResult(source.frame, (a, b))
    window.poll_once()
    assert len(window.axes.lines) == 0 and not window._pose_objects


def test_pose_start_does_not_reuse_frozen_sam3d_frame(app):
    window = CameraWindow(sam3d=True, foundationpose=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    try:
        window.prompt_inputs["A"].setText("cup")
        window.apply_prompts()
        window.receiver.sam3_latest = sam3_result(window._prompts.revision, sequence=7, count=1)
        window.poll_once()
        window.generate_mesh("A")
        window.receiver.mesh_events.append(completed_mesh(window.receiver.mesh_requests[-1]))
        window.poll_once()
        assert window.pose_mesh_inputs["A"].text() == "generated/object.glb"
        current = sam3_result(window._prompts.revision, sequence=99, count=1)
        window.receiver.sam3_latest = current
        window.poll_once()
        assert window._mesh_bindings["A"]["frame"].metadata["sequence"] == 7
        window.start_pose("A")
        assert window.receiver.pose_controls[-1].targets[0].frame is current.frame
        assert window.receiver.pose_controls[-1].targets[0].frame.metadata["sequence"] == 99
    finally:
        window.receiver.alive = False
        window.close()
        app.processEvents()


def test_pose_browse_sets_selected_persistent_mesh_path(pose_window, monkeypatch):
    choice = pose_window.pose_mesh_choices["A"]
    choice.setCurrentIndex(1)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *_args: ("/meshes/cup.glb", "Mesh"))
    pose_window._browse_pose_mesh("A")
    assert pose_window.pose_mesh_inputs["A"].text() == "/meshes/cup.glb"
    assert choice.currentData() is None
