"""Worker navigation is display-only; requests still use the existing controls."""
from copy import deepcopy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtCore, QtWidgets

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result
from grasp_workbench.ipc.sam3_refine import RefineResult
from grasp_workbench.ui.camera_window import CameraWindow
from grasp_workbench.ui.mask_editor import MaskEditor


WORKERS = ("camera", "sam3", "sam3d", "foundationpose", "graspgenx", "grasp_selection", "aprilcube", "recording")


class FakeReceiver:
    def __init__(self, service, **options):
        self.service, self.options = service, options
        self.alive, self.starts, self.stops = False, 0, 0
        self.latest = self.sam3_latest = self.pose_latest = self.vlm_latest = None
        self.refine_latest = self.selection_latest = self.overlay_latest = None
        self.mesh_events, self.grasp_events = [], []
        self.prompts, self.pose_controls, self.aprilcube_controls = [], [], []
        self.mesh_requests, self.grasp_requests, self.selection_requests = [], [], []
        self.vlm_requests, self.refine_requests = [], []

    def start(self):
        self.alive = True
        self.starts += 1

    def stop(self):
        self.alive = False
        self.stops += 1

    def is_alive(self):
        return self.alive

    def snapshot(self):
        return {"ready": True, "error": None, "received_count": 1,
                "last_received_monotonic_s": None}

    def _take(self, name):
        value = getattr(self, name)
        setattr(self, name, [] if isinstance(value, list) else None)
        return value

    def take_latest(self):
        return self._take("latest")

    def take_sam3_latest(self):
        return self._take("sam3_latest")

    def take_pose_latest(self):
        return self._take("pose_latest")

    def take_vlm_latest(self):
        return self._take("vlm_latest")

    def take_refine_latest(self):
        return self._take("refine_latest")

    def take_selection_latest(self):
        return self._take("selection_latest")

    def take_aprilcube_overlay_latest(self):
        return self._take("overlay_latest")

    def take_mesh_results(self):
        return self._take("mesh_events")

    def take_grasp_results(self):
        return self._take("grasp_events")

    def set_prompts(self, value):
        self.prompts.append(value)

    def set_pose_control(self, value):
        self.pose_controls.append(value)

    def send_aprilcube_control(self, value):
        self.aprilcube_controls.append(value)

    def submit_vlm(self, value):
        self.vlm_requests.append(value)

    def submit_mesh(self, value):
        self.mesh_requests.append(value)

    def submit_grasp(self, value):
        self.grasp_requests.append(value)

    def submit_selection(self, value):
        self.selection_requests.append(value)

    def submit_refine(self, value):
        self.refine_requests.append(value)

    def mesh_preview(self, *_args):
        return None


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def make_window(app, tmp_path):
    windows = []

    def make(**options):
        window = CameraWindow(receiver_factory=FakeReceiver, asset_dir=tmp_path / "assets", **options)
        window.timer.stop()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.close()
    app.processEvents()


def select_page(window, key, app):
    page = window.worker_pages[key]
    index = window.worker_stack.indexOf(page)
    window.worker_tabs.setCurrentRow(index)
    app.processEvents()
    assert window.worker_stack.currentWidget() is page
    return page


def buttons(widget):
    return widget.findChildren(QtWidgets.QPushButton)


def writes(receiver):
    return tuple(len(getattr(receiver, name)) for name in (
        "prompts", "pose_controls", "aprilcube_controls", "vlm_requests", "mesh_requests",
        "grasp_requests", "selection_requests", "refine_requests"))


def camera_frame(sequence=12):
    return CameraFrame(np.full((30, 40, 3), (31, 47, 59), dtype=np.uint8),
                       np.ones((30, 40), dtype=np.float32), {
                           "camera_id": "fake", "sequence": sequence,
                           "intrinsics": {"fx": 40., "fy": 40., "cx": 20., "cy": 15.},
                       })


@pytest.mark.parametrize("enabled", [None, "sam3", "sam3d", "foundationpose", "graspgenx", "grasp_selection"])
def test_navigation_reports_requested_workers_not_inferred_frontends(make_window, enabled, app):
    window = make_window(**({enabled: True} if enabled else {}))
    expected = {key: key in {"camera", enabled} for key in WORKERS}
    assert window.worker_enabled == expected
    assert isinstance(window.worker_tabs, QtWidgets.QListWidget)
    assert isinstance(window.worker_stack, QtWidgets.QStackedWidget)
    assert set(window.worker_pages) == set(WORKERS)
    assert window.worker_tabs.count() == window.worker_stack.count() == len(WORKERS)
    for key in WORKERS:
        page = select_page(window, key, app)
        index = window.worker_stack.indexOf(page)
        assert window.worker_tabs.item(index).flags() & QtCore.Qt.ItemFlag.ItemIsEnabled
        assert window.worker_page_contents[key].isEnabled() is expected[key]
        if not expected[key]:
            assert not any(button.isEnabled() for button in buttons(page))
            assert any(label.text().strip() for label in page.findChildren(QtWidgets.QLabel))


def test_vlm_is_above_workspace_not_duplicated_in_worker_pages(make_window, app):
    window = make_window(vlm=True)
    window.resize(1400, 850)
    window.show()
    app.processEvents()
    assert window.vlm_instruction.isVisible()
    assert window.vlm_instruction.isEnabled()
    for page in window.worker_pages.values():
        assert not page.isAncestorOf(window.vlm_instruction)
        assert not page.isAncestorOf(window.vlm_button)
        assert not page.isAncestorOf(window.image_view)
        assert not page.isAncestorOf(window.mesh_view)
    input_y = window.vlm_instruction.mapTo(window.centralWidget(), QtCore.QPoint()).y()
    image_y = window.image_view.mapTo(window.centralWidget(), QtCore.QPoint()).y()
    assert input_y < image_y
    assert "vlm" not in window.worker_pages


def test_navigation_does_not_send_requests_restart_receiver_or_clear_shared_views(make_window, app):
    window = make_window(sam3=True, sam3d=True, foundationpose=True, vlm=True,
                         graspgenx=True, grasp_selection=True)
    window.receiver.latest = frame = camera_frame()
    window.poll_once()
    image = window.image_view.image
    image_key = image.cacheKey()
    line, = window.axes.plot([0, 1], [0, 1], [0, 1])
    view, canvas, receiver, timer = window.image_view, window.mesh_view, window.receiver, window.timer
    before = writes(receiver)
    for key in (*WORKERS, *reversed(WORKERS)):
        select_page(window, key, app)
        assert window.image_view is view and window.mesh_view is canvas
        assert window.image_view.image.cacheKey() == image_key
        assert window._latest_frame is frame and line in window.axes.lines
    assert writes(receiver) == before
    assert window.receiver is receiver and receiver.starts == 1 and receiver.stops == 0
    assert window.timer is timer and not timer.isActive()


def test_snapshot_session_export_actions_are_absent_from_all_pages(make_window):
    window = make_window(sam3=True, sam3d=True, foundationpose=True, vlm=True,
                         graspgenx=True, grasp_selection=True)
    for button in buttons(window):
        text = button.text().lower()
        assert not any(word in text for word in ("snapshot", "session", "export", "현재 프레임 캡처"))
    assert callable(window.capture_snapshot) and callable(window.save_semantic_session)


def test_sam3_only_candidates_refine_and_contextual_return_to_live(make_window, app, monkeypatch):
    window = make_window(sam3=True)
    window.show()
    select_page(window, "sam3", app)
    assert window.live_camera_button.isHidden()
    for key in ("A", "B"):
        window.prompt_inputs[key].setText("tape roll" if key == "A" else "tuna can")
    window.apply_prompts()
    frame = camera_frame()
    observations = tuple(Sam3ObjectResult(key, window.prompt_inputs[key].text(),
        np.ones((1, 30, 40), dtype=bool), np.array([[0, 0, 40, 30]], dtype=np.float32),
        np.ones(1, dtype=np.float32)) for key in ("A", "B"))
    source = Sam3Result(frame, window._prompts.revision, observations)
    window.receiver.latest, window.receiver.sam3_latest = frame, source
    window.poll_once()
    assert set(window.candidate_inputs) == {"A", "B"}
    assert all(combo.currentData() == 0 for combo in window.candidate_inputs.values())
    refine_buttons = [button for button in buttons(window.worker_pages["sam3"]) if "mask 보정" in button.text().lower()]
    assert len(refine_buttons) == 2 and all(button.isEnabled() for button in refine_buttons)

    def accept(dialog):
        dialog.canvas.points.append({"xy": [10, 10], "label": 1})
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(MaskEditor, "exec", accept)
    refine_buttons[0].click()
    request, = window.receiver.refine_requests
    refined = deepcopy(source)
    refined.revision, refined.objects = request.request_id, (source.objects[0],)
    window.receiver.refine_latest = RefineResult(request.request_id, "done", refined)
    window.poll_once()
    assert window._snapshot_active and not window.live_camera_button.isHidden()
    window.receiver.latest = camera_frame(sequence=99)
    window.poll_once()
    before = writes(window.receiver)
    window.live_camera_button.click()
    assert not window._snapshot_active and window.live_camera_button.isHidden()
    assert window._latest_frame.metadata["sequence"] == 99 and writes(window.receiver) == before


def test_selection_only_loads_existing_bundles_from_its_own_page(make_window, app, monkeypatch):
    loads, selections = [], []
    monkeypatch.setattr(CameraWindow, "load_grasp_bundle_path", lambda _window, key: loads.append(key))
    monkeypatch.setattr(CameraWindow, "request_grasp_selection", lambda _window: selections.append("request"))
    window = make_window(grasp_selection=True)
    page = select_page(window, "grasp_selection", app)
    load_buttons = [button for button in buttons(page)
                    if "불러오기" in button.text() and ("bundle" in button.text() or "후보" in button.text())]
    assert len(load_buttons) == 2 and all(button.isEnabled() for button in load_buttons)
    for button in load_buttons:
        button.click()
    assert loads == ["A", "B"]
    assert not any(button.isEnabled() for button in buttons(window.worker_pages["graspgenx"]))
    assert window.selection_button.text() == "후보 선택·다시 평가"
    assert len([button for button in buttons(page) if button.text() == window.selection_button.text()]) == 1
    window.selection_button.click()
    assert selections == ["request"] and not window.receiver.grasp_requests


def test_pose_seed_and_display_fk_controls_start_collapsed(make_window, app):
    window = make_window(foundationpose=True, grasp_selection=True)
    before = writes(window.receiver)
    select_page(window, "foundationpose", app)
    for key in ("A", "B"):
        assert window.pose_advanced[key].isHidden()
        assert window.pose_advanced[key].isAncestorOf(window.pose_seed_inputs[key])
        assert window.worker_pages["foundationpose"].isAncestorOf(window.pose_advanced[key])
        window.pose_advanced_buttons[key].click()
        assert not window.pose_advanced[key].isHidden()
        window.pose_advanced_buttons[key].click()
        assert window.pose_advanced[key].isHidden()
    assert window.selection_advanced.isHidden()
    assert window.selection_advanced.isAncestorOf(window.selection_details)
    fk_buttons = [button for button in buttons(window.selection_advanced) if "파지 관계" in button.text()]
    assert len(fk_buttons) == 2
    select_page(window, "grasp_selection", app)
    window.selection_advanced_button.click()
    assert not window.selection_advanced.isHidden()
    window.selection_advanced_button.click()
    assert window.selection_advanced.isHidden()
    assert writes(window.receiver) == before
    assert window.worker_enabled["sam3"] is False


def test_aprilcube_override_leaves_only_camera_and_april_controls_active(make_window, app):
    window = make_window(aprilcube=True, sam3=True, sam3d=True, foundationpose=True,
                         vlm=True, graspgenx=True, grasp_selection=True)
    assert window.worker_enabled == {key: key in {"camera", "aprilcube"} for key in WORKERS}
    assert window.receiver.options == {"aprilcube": True}
    for key in WORKERS:
        page = select_page(window, key, app)
        if key not in {"camera", "aprilcube"}:
            assert not any(button.isEnabled() for button in buttons(page))
    assert not hasattr(window, "vlm_button") or not window.vlm_button.isEnabled()
    header_fields = window.vlm_header.findChildren(QtWidgets.QLineEdit)
    assert len(header_fields) == 1 and not header_fields[0].isEnabled()
    assert "AprilCube" in header_fields[0].placeholderText()
    assert not any(button.isEnabled() for button in buttons(window.vlm_header))
    assert set(window.aprilcube_xyz_inputs) == set(window.aprilcube_rpy_inputs) == {"A", "B"}
