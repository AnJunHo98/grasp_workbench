"""Optional recording controls use native worker commands, not UI-side writers."""
import os
import sys
from types import ModuleType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.recording import RecordingStatus
from grasp_workbench.ui.camera_window import CameraWindow
from grasp_workbench.workers.worker_camera_ui import build_parser, main


class FakeReceiver:
    def __init__(self, service, **options):
        self.service, self.options = service, options
        self.alive = False
        self.recording_latest = None
        self.controls, self.aprilcube_controls = [], []

    def start(self):
        self.alive = True

    def stop(self):
        self.alive = False

    def is_alive(self):
        return self.alive

    def snapshot(self):
        return {"ready": True, "error": None, "received_count": 0, "last_received_monotonic_s": None}

    def take_latest(self):
        return None

    def take_pose_latest(self):
        return None

    def take_aprilcube_overlay_latest(self):
        return None

    def send_aprilcube_control(self, control):
        self.aprilcube_controls.append(control)

    def send_recording_control(self, control):
        self.controls.append(control)

    def take_recording_latest(self):
        latest, self.recording_latest = self.recording_latest, None
        return latest


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def make_window(app):
    windows = []

    def make(**options):
        window = CameraWindow(receiver_factory=FakeReceiver, **options)
        window.timer.stop()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.close()
    app.processEvents()


def accept(window, state, **options):
    window.receiver.recording_latest = RecordingStatus(state=state, **options)
    window.poll_once()


def enabled(window):
    return {key: button.isEnabled() for key, button in window.recording_buttons.items()}


def test_disabled_recording_tab_does_not_open_endpoints_or_create_controls(make_window):
    window = make_window()
    assert window.receiver.options == {}
    assert window.worker_enabled["recording"] is False
    assert "recording" in window.worker_pages
    assert not window.worker_page_contents["recording"].isEnabled()
    assert not hasattr(window, "recording_buttons")
    window.poll_once()
    assert window.receiver.controls == []


def test_recording_stays_enabled_in_aprilcube_mode_and_startup_does_not_record(make_window):
    window = make_window(aprilcube=True, recording=True, vlm=True, sam3=True, sam3d=True,
                         foundationpose=True, graspgenx=True, grasp_selection=True)
    assert window.receiver.options == {"aprilcube": True, "recording": True}
    assert {key for key, value in window.worker_enabled.items() if value} == {"camera", "aprilcube", "recording"}
    assert window.worker_page_contents["recording"].isEnabled()
    assert window.receiver.controls == []
    assert enabled(window) == {"start": False, "save": False, "discard": False}


def test_buttons_send_start_save_discard_and_show_worker_counts_and_path(make_window):
    window = make_window(recording=True)
    accept(window, "ready", output_dir="runs/recordings/example")
    assert enabled(window) == {"start": True, "save": False, "discard": False}
    window.recording_buttons["start"].click()
    start = window.receiver.controls[-1]
    assert start.action == "start" and start.request_id
    assert not any(enabled(window).values())
    # Old ready/progress must not acknowledge a different pending command.
    accept(window, "ready")
    assert not any(enabled(window).values())
    accept(window, "recording", request_id=start.request_id, output_dir="runs/recordings/example",
           episode_index=2, episode_frames=12, total_frames=72)
    assert enabled(window) == {"start": False, "save": True, "discard": True}
    assert window.recording_counts.text() == "Episode 2 · 12 frames · Total 72 frames"
    assert window.recording_output.text() == "Output · runs/recordings/example"
    window.recording_buttons["save"].click()
    save = window.receiver.controls[-1]
    assert save.action == "save" and save.request_id != start.request_id
    accept(window, "saved", request_id=save.request_id, total_frames=72)
    assert enabled(window)["start"]
    window.recording_buttons["start"].click()
    second = window.receiver.controls[-1]
    accept(window, "recording", request_id=second.request_id)
    assert enabled(window) == {"start": False, "save": False, "discard": True}
    window.recording_buttons["discard"].click()
    discard = window.receiver.controls[-1]
    assert discard.action == "discard" and discard.request_id != save.request_id
    accept(window, "discarded", request_id=discard.request_id)
    assert enabled(window) == {"start": True, "save": False, "discard": False}
    assert [control.action for control in window.receiver.controls] == ["start", "save", "start", "discard"]


def test_empty_recording_episode_can_be_discarded_but_not_saved(make_window):
    window = make_window(recording=True)
    accept(window, "ready")
    window.recording_buttons["start"].click()
    start = window.receiver.controls[-1]
    accept(window, "recording", request_id=start.request_id, episode_frames=0)
    assert enabled(window) == {"start": False, "save": False, "discard": True}
    window.recording_buttons["save"].click()
    assert window.receiver.controls == [start]
    accept(window, "recording", request_id=start.request_id, episode_frames=1, total_frames=1)
    assert enabled(window) == {"start": False, "save": True, "discard": True}


def test_worker_error_is_displayed_and_does_not_claim_recording(make_window):
    window = make_window(recording=True)
    accept(window, "error", error="injected dataset failure")
    assert "injected dataset failure" in window.recording_status.text()
    assert not any(enabled(window).values())


def test_parser_exposes_recording_independently_of_aprilcube():
    args = build_parser().parse_args(["--recording", "--aprilcube", "--vlm"])
    assert args.recording and args.aprilcube and args.vlm


@pytest.mark.parametrize("arguments, expected", [
    (["--recording"], {"recording": True}),
    (["--recording", "--aprilcube", "--vlm", "--foundationpose"], {"aprilcube": True, "recording": True}),
])
def test_ui_worker_retains_recording_after_aprilcube_overrides_models(monkeypatch, arguments, expected):
    calls = []

    class Timer:
        timeout = SimpleNamespace(connect=lambda _callback: None)

        def start(self, *_args):
            pass

        def stop(self):
            pass

    class Window:
        exit_code = 0
        receiver = SimpleNamespace(stop=lambda: None, join=lambda: None)

        def __init__(self, *_args, **options):
            calls.append(options)

        def show(self):
            pass

    qt = ModuleType("PyQt6")
    qt.QtCore = SimpleNamespace(QTimer=Timer)
    qt.QtWidgets = SimpleNamespace(QApplication=SimpleNamespace(instance=lambda: SimpleNamespace(exec=lambda: 0)))
    camera_window = ModuleType("grasp_workbench.ui.camera_window")
    camera_window.CameraWindow = Window
    monkeypatch.setitem(sys.modules, "PyQt6", qt)
    monkeypatch.setitem(sys.modules, "grasp_workbench.ui.camera_window", camera_window)
    assert main(arguments) == 0
    assert calls == [expected]
