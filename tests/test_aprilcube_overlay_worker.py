"""The display worker consumes shared pose results without camera or model inference."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.aprilcube import AprilCubeControl, AprilCubeTarget, services
from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult, services as pose_services
from grasp_workbench.workers import worker_aprilcube_overlay as overlay_worker
from grasp_workbench.workers.worker_aprilcube_overlay import AprilCubeOverlayWorker, build_parser
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


class Renderer:
    def __init__(self):
        self.calls = []

    def render(self, source, targets, poses, alpha):
        self.calls.append((source, targets, poses, alpha))
        return np.full((*source.rgb.shape[:2], 4), 7 if any(
            pose is not None for pose in poses.values()) else 0, np.uint8)


def worker(frames=0, **options):
    results, controls, publisher, renderer = Mailbox(), Mailbox(), Publisher(), Renderer()
    calls = []
    instance = AprilCubeOverlayWorker(
        WorkerContext("aprilcube_overlay"), "example/rgbd", .4, frames,
        result_subscriber_factory=lambda service: calls.append(service) or results,
        control_subscriber_factory=lambda service: calls.append(service) or controls,
        overlay_publisher_factory=lambda service: calls.append(service) or publisher,
        renderer_factory=lambda: renderer, **options)
    instance.on_start()
    return instance, results, controls, publisher, renderer, calls


def object_pose(slot="A", session="aprilcube-A", status="tracking", source="aprilcube",
                pose=None):
    return ObjectPose(slot, session, "mesh.ply", status,
                      np.eye(4) if pose is None else pose, pose_source=source)


def test_worker_receives_shared_pose_channel_and_keeps_raw_source_and_metadata():
    instance, results, controls, publisher, renderer, calls = worker(frames=2)
    try:
        assert calls == [pose_services("example/rgbd")[1], services("example/rgbd")[0],
                         services("example/rgbd")[1]]
        assert instance.hz == 100
        instance.step_once()
        assert publisher.results == [] and renderer.calls == []
        # Neither sequence ordering nor timestamps introduce temporal acceptance gates.
        for sequence in (2, 1):
            source = frame(sequence)
            rgb, depth = source.rgb.copy(), source.depth_m.copy()
            metadata = dict(source.metadata)
            pose = np.eye(4)
            pose[:3, 3] = (.1, -.2, .7)
            results.value = PoseResult(source, (object_pose(pose=pose),))
            instance.step_once()
            rendered_source, targets, poses, alpha = renderer.calls[-1]
            assert rendered_source is source and alpha == .4
            assert [target.session_id for target in targets] == ["aprilcube-A", "aprilcube-B"]
            assert poses["A"] is pose and poses["B"] is None
            assert publisher.results[-1].metadata == metadata
            assert publisher.results[-1].metadata is not source.metadata
            np.testing.assert_array_equal(publisher.results[-1].rgba, 7)
            np.testing.assert_array_equal(source.rgb, rgb)
            np.testing.assert_array_equal(source.depth_m, depth)
            assert source.metadata == metadata
        assert instance.processed_count == 2 and instance.ctx.stop_event.is_set()
    finally:
        instance.on_stop()
    assert all(endpoint.closed for endpoint in (results, controls, publisher))


def test_control_binding_is_used_without_applying_cube_transform_again():
    instance, results, controls, publisher, renderer, _ = worker()
    offset = np.eye(4)
    offset[:3, 3] = (.1, .2, .3)
    selected = AprilCubeTarget("A", "selected", "asset.ply", .001, offset)
    pose = np.eye(4)
    pose[:3, 3] = (.4, .5, .9)
    try:
        controls.value = AprilCubeControl((selected,))
        results.value = PoseResult(frame(), (object_pose(session="selected", pose=pose),))
        instance.step_once()
        assert renderer.calls[-1][1] == (selected,)
        assert renderer.calls[-1][1][0].mesh_scale_to_m == .001
        np.testing.assert_array_equal(renderer.calls[-1][2]["A"], pose)
        assert renderer.calls[-1][2]["A"] is pose
        controls.value = AprilCubeControl(())
        results.value = PoseResult(frame(2), (object_pose(session="selected", pose=pose),))
        instance.step_once()
        assert renderer.calls[-1][1] == () and renderer.calls[-1][2] == {}
        np.testing.assert_array_equal(publisher.results[-1].rgba, 0)
    finally:
        instance.on_stop()


@pytest.mark.parametrize("objects", [(), (object_pose(session="old-session"),),
                         (object_pose(slot="B", session="aprilcube-A"),),
                         (object_pose(status="unavailable"),),
                         (object_pose(status="recovering"),),
                         (object_pose(source="foundationpose"),),
                         (ObjectPose("A", "aprilcube-A", "", "tracking", None,
                                     pose_source="aprilcube"),)])
def test_missing_inactive_or_other_session_pose_clears_display_layer(objects):
    instance, results, _, publisher, renderer, _ = worker()
    try:
        results.value = PoseResult(frame(), objects)
        instance.step_once()
        assert renderer.calls[-1][2] == {"A": None, "B": None}
        np.testing.assert_array_equal(publisher.results[-1].rgba, 0)
    finally:
        instance.on_stop()


def test_controls_are_consumed_without_result_but_do_not_trigger_an_old_frame_render():
    instance, results, controls, publisher, renderer, _ = worker()
    selected = AprilCubeTarget("B", "new-B", "new.ply", .01, np.eye(4))
    try:
        controls.value = AprilCubeControl((selected,))
        instance.step_once()
        assert instance.targets == (selected,)
        assert not publisher.results and not renderer.calls and instance.processed_count == 0
        results.value = PoseResult(frame(), (object_pose("B", "new-B"),))
        instance.step_once()
        instance.step_once()
        assert len(publisher.results) == len(renderer.calls) == instance.processed_count == 1
    finally:
        instance.on_stop()


@pytest.mark.parametrize("logging, timestamp", [(False, True), (True, True), (True, False)])
def test_timing_stages_source_age_and_no_log_on_empty_poll(monkeypatch, logging, timestamp):
    instance, results, _, publisher, renderer, _ = worker(log_timing=logging)
    clock, events = SimpleNamespace(ns=10_000_000_000, mono_reads=0), []

    def monotonic_ns():
        clock.mono_reads += 1
        return clock.ns

    def advance(method, duration_ms):
        def call(*args, **kwargs):
            result = method(*args, **kwargs)
            clock.ns += duration_ms * 1_000_000
            return result
        return call

    monkeypatch.setattr(overlay_worker, "time", SimpleNamespace(
        perf_counter_ns=lambda: clock.ns, monotonic_ns=monotonic_ns))
    monkeypatch.setattr(results, "receive_latest", advance(results.receive_latest, 1))
    monkeypatch.setattr(renderer, "render", advance(renderer.render, 4))
    monkeypatch.setattr(publisher, "publish", advance(publisher.publish, 5))

    def record(event, **values):
        events.append((event, values))
        clock.ns += 100_000_000

    monkeypatch.setattr(overlay_worker, "log_record", record)
    source = frame(7)
    if timestamp:
        source.metadata["received_monotonic_ns"] = 9_000_000_000
    else:
        del source.metadata["received_monotonic_ns"]
    results.value = PoseResult(source, (object_pose(),))
    try:
        instance.step_once()
        if logging:
            assert events.pop() == ("timing", {"worker": "aprilcube_overlay", "sequence": 7,
                                              "timings_ms": {
                                                  "receive_ms": 1., "render_ms": 4.,
                                                  "overlay_publish_ms": 5., "total_ms": 10.,
                                                  "source_age_at_overlay_ms": (
                                                      1010. if timestamp else None)}})
        assert events == [] and clock.mono_reads == int(logging and timestamp)
        instance.step_once()
        assert events == [] and instance.processed_count == 1
    finally:
        instance.on_stop()


def test_cli_help_imports_no_camera_sdk_predictor_or_gpu_backend():
    args = build_parser().parse_args([])
    assert args.frames == 0 and args.overlay_alpha == .65 and not args.log_timing
    result = subprocess.run([sys.executable, "-c", """
import runpy, sys
sys.argv = ['worker_aprilcube_overlay', '--help']
try:
    runpy.run_module('grasp_workbench.workers.worker_aprilcube_overlay', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('cv2', 'torch', 'pyzed', 'pyrealsense2',
                                               'grasp_workbench.models.aprilcube'))
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--overlay-alpha" in result.stdout and "--log-timing" in result.stdout


@pytest.mark.parametrize("flags, expected_logging", [([], False), (["--log-timing"], True)])
def test_cli_forwards_options_and_runs(monkeypatch, flags, expected_logging):
    calls = []

    class StubWorker:
        def __init__(self, context, *args, **kwargs):
            calls.append((context.name, args, kwargs))

        def run(self):
            calls.append("run")

    monkeypatch.setattr(overlay_worker, "AprilCubeOverlayWorker", StubWorker)
    assert overlay_worker.main(["--service", "test/rgbd", "--overlay-alpha", ".3",
                                "--frames", "2", *flags]) == 0
    assert calls == [("aprilcube_overlay", ("test/rgbd", .3, 2),
                      {"log_timing": expected_logging}), "run"]


@pytest.mark.parametrize("flags", [["--frames", "-1"], ["--overlay-alpha", "1.1"],
                                 ["--overlay-alpha", "-.1"]])
def test_cli_rejects_invalid_options_before_starting(monkeypatch, flags):
    monkeypatch.setattr(overlay_worker, "AprilCubeOverlayWorker", lambda *args, **kwargs: pytest.fail(
        "Invalid options must not start a worker"))
    with pytest.raises(SystemExit) as error:
        overlay_worker.main(flags)
    assert error.value.code == 2


def test_cli_reports_worker_failure(monkeypatch, capsys):
    class FailingWorker:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise RuntimeError("render failed")

    monkeypatch.setattr(overlay_worker, "AprilCubeOverlayWorker", FailingWorker)
    assert overlay_worker.main([]) == 1
    assert '"worker": "aprilcube_overlay"' in capsys.readouterr().out
