"""Worker lifecycle checks with local fakes; no camera or native service opens."""
import ast
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
import pytest

from grasp_workbench.core.worker_base import SingleRateWorker, WorkerContext
from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.workers.registry import WORKERS
from grasp_workbench.workers.worker_camera import CameraWorker
from grasp_workbench.workers.worker_camera_monitor import CameraMonitorWorker


def frame():
    return CameraFrame(np.zeros((2, 3, 3), dtype=np.uint8),
                       np.array([[0.5, 2, np.nan], [np.inf, 0, -1]], dtype=np.float32),
                       {"camera_id": "fake-serial", "sequence": 17, "stream_id": "fake-session",
                        "capture_time_ns": 12345, "intrinsics": {"fx": 100.5}})


class FakeCamera:
    def __init__(self, value):
        self.value, self.opened, self.closed, self.read_count = value, False, False, 0

    def open(self):
        self.opened = True

    def read(self):
        self.read_count += 1
        return self.value

    def close(self):
        self.closed = True


class FakePublisher:
    def __init__(self, _service):
        self.frames, self.closed = [], False

    def publish(self, value):
        self.frames.append(value)
        return 1

    def close(self):
        self.closed = True


class FakeSubscriber:
    def __init__(self, values):
        self.values, self.closed = list(values), False

    def receive_latest(self):
        return self.values.pop(0) if self.values else None

    def close(self):
        self.closed = True


def test_start_failure_still_runs_cleanup_and_restores_signal_handlers():
    calls = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    class Worker(SingleRateWorker):
        def on_start(self):
            calls.append("start")
            raise RuntimeError("start failed")

        def on_stop(self):
            calls.append("stop")

    with pytest.raises(RuntimeError, match="start failed"):
        Worker(WorkerContext("fake")).run()
    assert calls == ["start", "stop"]
    assert all(signal.getsignal(sig) is handler for sig, handler in previous.items())


def test_signal_stops_loop_and_cancellation_wakes_rate_wait():
    previous = signal.getsignal(signal.SIGTERM)

    class Worker(SingleRateWorker):
        def step_once(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    worker = Worker(WorkerContext("signal-test"), hz=0.001)
    started = time.monotonic()
    worker.run()
    assert worker.ctx.stop_event.is_set()
    assert time.monotonic() - started < 1
    assert signal.getsignal(signal.SIGTERM) is previous


def test_local_worker_can_run_without_installing_handlers_from_background_thread():
    class Worker(SingleRateWorker):
        def step_once(self):
            self.ctx.stop()

    worker = Worker(WorkerContext("thread-test"))
    thread = threading.Thread(target=worker.run)
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive() and worker.ctx.stop_event.is_set()


def test_finite_capture_preserves_frame_arrays_and_metadata(capsys):
    value = frame()
    camera, publisher = FakeCamera(value), FakePublisher("fake-service")
    worker = CameraWorker(WorkerContext("camera"), camera, "fake-service", frames=2,
                          publisher_factory=lambda service: publisher)
    worker.run()
    assert camera.opened and camera.closed and publisher.closed
    assert camera.read_count == worker.published_count == 2
    assert all(published is value for published in publisher.frames)
    assert value.metadata["sequence"] == 17 and np.isnan(value.depth_m[0, 2])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["event"] == "camera-open"
    assert records[1]["metadata"] == value.metadata
    assert records[-1] == {"event": "camera-stopped", "published_count": 2}


@pytest.mark.parametrize("failed_operation", ["open", "read", "publisher"])
def test_camera_failures_close_every_initialized_resource(failed_operation):
    camera, publisher = FakeCamera(frame()), FakePublisher("fake-service")

    def fail(*args):
        raise RuntimeError("injected failure")

    if failed_operation in {"open", "read"}:
        setattr(camera, failed_operation, fail)
    factory = fail if failed_operation == "publisher" else lambda service: publisher
    with pytest.raises(RuntimeError, match="injected failure"):
        CameraWorker(WorkerContext("camera"), camera, publisher_factory=factory).run()
    assert camera.closed
    assert publisher.closed == (failed_operation == "read")


def test_capture_cancels_without_any_finite_frame_requirement():
    context = WorkerContext("camera")
    camera, publisher = FakeCamera(frame()), FakePublisher("fake-service")
    original = publisher.publish

    def publish(value):
        context.stop()
        return original(value)

    publisher.publish = publish
    worker = CameraWorker(context, camera, publisher_factory=lambda service: publisher)
    worker.run()
    assert worker.published_count == 1 and camera.closed and publisher.closed


def test_monitor_counts_received_samples_and_summarizes_valid_depth(capsys):
    value = frame()
    subscriber = FakeSubscriber([None, value, value])
    worker = CameraMonitorWorker(WorkerContext("monitor"), frames=2,
                                 subscriber_factory=lambda service: subscriber)
    worker.run()
    assert worker.received_count == 2 and subscriber.closed
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["event"] == "ready"
    summary = records[1]
    assert summary["rgb_shape"] == [2, 3, 3]
    assert summary["depth"] == {"shape": [2, 3], "valid_count": 2,
                                 "min_m": 0.5, "max_m": 2.0, "mean_m": 1.25}
    assert summary["metadata"] == value.metadata and summary["intrinsics"] == {"fx": 100.5}
    assert records[-1] == {"event": "monitor-stopped", "received_count": 2}


@pytest.mark.parametrize("frames,failed", [(2, True), (0, False)])
def test_monitor_timeout_cleans_up_and_distinguishes_unmet_finite_count(frames, failed):
    subscriber = FakeSubscriber([])

    class ExpiredMonitor(CameraMonitorWorker):
        def on_start(self):
            super().on_start()
            self._started -= 2

    worker = ExpiredMonitor(WorkerContext("monitor"), frames=frames, timeout=1,
                            subscriber_factory=lambda service: subscriber)
    if failed:
        with pytest.raises(TimeoutError, match="0/2"):
            worker.run()
    else:
        worker.run()
    assert subscriber.closed


@pytest.mark.parametrize("camera_name", ["zed", "realsense"])
def test_camera_cli_uses_only_selected_sdk_example(monkeypatch, camera_name):
    from grasp_workbench.cameras import realsense, zed
    from grasp_workbench.ipc import camera as ipc
    from grasp_workbench.workers.worker_camera import main

    camera, publisher = FakeCamera(frame()), FakePublisher("fake-service")
    module, attribute = (zed, "ZedCamera") if camera_name == "zed" else (realsense, "RealSenseCamera")
    options = []
    monkeypatch.setattr(module, attribute, lambda **kwargs: options.append(kwargs) or camera)
    monkeypatch.setattr(ipc, "CameraPublisher", lambda service: publisher)
    assert main(["--camera", camera_name, "--serial", "chosen", "--fps", "15", "--frames", "1"]) == 0
    assert options[0]["serial"] == "chosen" and options[0]["fps"] == 15
    assert camera.closed and publisher.closed


def test_entry_points_and_only_iceoryx2_transport_imports():
    from grasp_workbench.workers.worker_camera import build_parser as capture_parser
    from grasp_workbench.workers.worker_camera_monitor import build_parser as monitor_parser

    assert capture_parser().parse_args([]).camera == "zed"
    assert capture_parser().parse_args([]).resolution == "HD720"
    assert monitor_parser().parse_args([]).timeout == 0
    assert set(WORKERS) == {"camera", "camera_monitor", "camera_ui", "data_recorder", "aprilcube", "aprilcube_overlay", "sam3", "sam3d", "foundationpose", "vlm", "graspgenx",
                            "grasp_selection"}
    root = Path(__file__).resolve().parents[1] / "grasp_workbench"
    for relative in ("core/worker_base.py", "workers/worker_camera.py", "workers/worker_camera_monitor.py",
                     "workers/worker_camera_ui.py", "workers/worker_aprilcube.py",
                     "workers/worker_aprilcube_overlay.py", "ui/receiver.py"):
        tree = ast.parse((root / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("multiprocessing") for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("multiprocessing")
