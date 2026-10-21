"""AprilCube publishes poses without waiting for a renderer or display subscriber."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.aprilcube import AprilCubeControl, AprilCubeTarget, services
from grasp_workbench.ipc.foundationpose import services as pose_services
from grasp_workbench.workers.worker_aprilcube import AprilCubeWorker, build_parser
from grasp_workbench.workers import worker_aprilcube as aprilcube_worker
from test_aprilcube_model import frame


class Mailbox:
    def __init__(self):
        self.value, self.closed = None, False

    def receive_latest(self):
        value, self.value = self.value, None
        return value

    def close(self):
        self.closed = True


class Publisher:
    def __init__(self):
        self.results, self.closed, self.refreshes = [], False, 0

    def update_connections(self):
        self.refreshes += 1

    def publish(self, result):
        self.results.append(result)

    def close(self):
        self.closed = True


class Predictor:
    def __init__(self):
        self.started, self.closed, self.frames = False, False, []
        self.timings_ms = {"prepare_ms": .5, "A_ms": 2., "B_ms": 2.5}
        self.diagnostics = {"A": {"n_tags": 3, "predicted": False}, "B": {"n_tags": 0}}
        self.pose = np.eye(4)
        self.pose[:3, 3] = (.1, -.2, .7)

    def start(self):
        self.started = True

    def predict(self, frame):
        self.frames.append(frame)
        return {"A": self.pose.copy(), "B": None}

    def close(self):
        self.closed = True


def worker(frames=0, **options):
    camera, control, output = Mailbox(), Mailbox(), Publisher()
    predictor, calls = Predictor(), []
    instance = AprilCubeWorker(
        WorkerContext("aprilcube-test"), "vendor", "a.json", "b.json", "example/rgbd",
        filter_enabled=False, frames=frames,
        camera_subscriber_factory=lambda service: calls.append(service) or camera,
        control_subscriber_factory=lambda service: calls.append(service) or control,
        result_publisher_factory=lambda service: calls.append(service) or output,
        predictor_factory=lambda *args, **kwargs: calls.append((args, kwargs)) or predictor,
        **options,
    )
    instance.on_start()
    return instance, camera, control, output, predictor, calls


def test_continuous_default_a_b_uses_shared_pose_channel_and_raw_source_frame():
    instance, camera, control, output, predictor, calls = worker(frames=2)
    try:
        assert calls == ["example/rgbd", services("example/rgbd")[0],
                         pose_services("example/rgbd")[1],
                         (("vendor", "a.json", "b.json"),
                          {"filter_enabled": False, "fast": True, "log_timing": False})]
        assert predictor.started
        assert not hasattr(instance, "renderer") and not hasattr(instance, "overlay_publisher")
        instance.step_once()
        assert not output.results
        for sequence in (2, 1):
            source = frame(sequence)
            original = source.rgb.copy()
            camera.value = source
            instance.step_once()
            result = output.results[-1]
            assert result.frame is source
            a, b = result.objects
            assert a.session_id == "aprilcube-A" and b.session_id == "aprilcube-B"
            assert a.mesh_path == b.mesh_path == ""
            assert a.status == "tracking" and b.status == "unavailable"
            assert a.pose_source == b.pose_source == "aprilcube"
            assert a.diagnostics == predictor.diagnostics["A"]
            assert b.diagnostics == predictor.diagnostics["B"]
            np.testing.assert_array_equal(a.camera_from_object, predictor.pose)
            assert b.camera_from_object is None
            np.testing.assert_array_equal(source.rgb, original)
        assert instance.processed_count == 2 and instance.ctx.stop_event.is_set()
    finally:
        instance.on_stop()
    assert all(item.closed for item in (camera, control, output, predictor))


def test_control_replaces_mesh_bindings_without_sam3_or_restarting_detectors():
    instance, camera, control, output, predictor, _ = worker()
    offset = np.eye(4)
    offset[:3, 3] = (.02, .03, -.04)
    selected = AprilCubeTarget("A", "new-session", "asset.ply", .001, offset)
    try:
        control.value = AprilCubeControl((selected,))
        camera.value = frame()
        instance.step_once()
        item, = output.results[-1].objects
        assert item.session_id == "new-session" and item.mesh_path == "asset.ply"
        np.testing.assert_allclose(item.camera_from_object, predictor.pose @ offset)
        control.value, camera.value = AprilCubeControl(()), frame(2)
        instance.step_once()
        assert output.results[-1].objects == ()
        assert len(predictor.frames) == 2
    finally:
        instance.on_stop()


def test_worker_cli_defaults_and_help_without_model_import():
    args = build_parser().parse_args(["--source-dir", "sdk", "--config-a", "a", "--config-b", "b"])
    assert args.frames == 0 and not args.no_filter
    assert args.fast and not args.log_timing
    result = subprocess.run([sys.executable, "-m", "grasp_workbench.workers.worker_aprilcube", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "--config-a" in result.stdout
    assert "--fast" in result.stdout and "--no-fast" in result.stdout and "--log-timing" in result.stdout
    assert "--overlay-alpha" not in result.stdout


@pytest.mark.parametrize("flags, expected_fast, expected_logging", [
    ([], True, False), (["--fast"], True, False), (["--no-fast"], False, False),
    (["--log-timing"], True, True), (["--fast", "--log-timing"], True, True),
])
def test_worker_cli_forwards_fast_and_logging(monkeypatch, flags, expected_fast, expected_logging):
    calls = []

    class StubWorker:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def run(self):
            calls.append("run")

    monkeypatch.setattr(aprilcube_worker, "AprilCubeWorker", StubWorker)
    assert aprilcube_worker.main(["--source-dir", "sdk", "--config-a", "a", "--config-b", "b",
                                  *flags]) == 0
    assert calls[0][1]["fast"] is expected_fast
    assert calls[0][1]["log_timing"] is expected_logging
    assert calls[1] == "run"


@pytest.mark.parametrize("fast", [False, True])
def test_worker_forwards_fast_mode_without_changing_filter(monkeypatch, fast):
    instance, _, _, _, _, calls = worker(fast=fast, log_timing=True)
    try:
        assert calls[-1][1] == {"filter_enabled": False, "fast": fast, "log_timing": True}
    finally:
        instance.on_stop()


@pytest.mark.parametrize("log_timing, timestamp", [(False, True), (True, True), (True, False)])
def test_worker_timing_accounts_for_stages_and_source_age(monkeypatch, log_timing, timestamp):
    instance, camera, _, output, predictor, _ = worker(
        fast=True, log_timing=log_timing)
    clock = SimpleNamespace(ns=10_000_000_000, perf_reads=0, mono_reads=0)
    events = []

    def perf_counter_ns():
        clock.perf_reads += 1
        return clock.ns

    def monotonic_ns():
        clock.mono_reads += 1
        return clock.ns

    def advance(method, duration_ms):
        def call(*args, **kwargs):
            result = method(*args, **kwargs)
            clock.ns += duration_ms * 1_000_000
            return result
        return call

    monkeypatch.setattr(aprilcube_worker, "time", SimpleNamespace(
        perf_counter_ns=perf_counter_ns, monotonic_ns=monotonic_ns))
    monkeypatch.setattr(camera, "receive_latest", advance(camera.receive_latest, 1))
    monkeypatch.setattr(predictor, "predict", advance(predictor.predict, 5))
    monkeypatch.setattr(output, "publish", advance(output.publish, 3))

    def record(event, **values):
        events.append((event, values))
        clock.ns += 100_000_000  # Logging must not inflate the measured worker total.

    monkeypatch.setattr(aprilcube_worker, "log_record", record)
    source = frame(7)
    if timestamp:
        source.metadata["received_monotonic_ns"] = 9_000_000_000
    else:
        del source.metadata["received_monotonic_ns"]
    camera.value = source
    try:
        instance.step_once()
        assert output.results[-1].frame is source
        if log_timing:
            event, values = events.pop()
            assert event == "timing"
            assert values["worker"] == instance.ctx.name and values["sequence"] == 7
            assert values["fast"] is True
            assert values["timings_ms"] == {
                "receive_ms": 1., "predict_ms": 5., "prepare_ms": .5,
                "A_ms": 2., "B_ms": 2.5, "compose_ms": 0., "pose_publish_ms": 3.,
                "total_ms": 9.,
                "source_age_at_pose_ms": 1009. if timestamp else None,
            }
        assert events == []
        assert clock.perf_reads == 5
        assert clock.mono_reads == (1 if log_timing and timestamp else 0)
        assert instance.processed_count == 1
        # Polling with no new source frame must not emit a timing record.
        instance.step_once()
        assert events == [] and instance.processed_count == 1
    finally:
        instance.on_stop()
