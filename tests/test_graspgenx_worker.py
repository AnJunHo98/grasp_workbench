"""Lazy independent grasp worker with injected, model-free endpoints."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.graspgenx import GraspRequest, services
from grasp_workbench.workers.worker_graspgenx import GraspGenXWorker, build_parser


class Inbox:
    def __init__(self):
        self.value, self.closed = None, False

    def receive_latest(self):
        value, self.value = self.value, None
        return value

    def close(self):
        self.closed = True


class Output:
    def __init__(self):
        self.results, self.closed, self.refreshes = [], False, 0

    def publish(self, value):
        self.results.append(value)

    def update_connections(self):
        self.refreshes += 1

    def close(self):
        self.closed = True


class Predictor:
    def __init__(self):
        self.starts, self.requests, self.closed = 0, [], False
        self.error = None

    def start(self):
        self.starts += 1

    def predict(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(grasps=np.eye(4)[None], scores=np.array([.9]), bundle_path="bundle.json",
                               gripper_metadata={"control_points": [[0., 0., 0.]]}, cached=request.regenerate is False)

    def close(self):
        self.closed = True


def worker(frames=0):
    inbox, output, predictor, calls = Inbox(), Output(), Predictor(), []

    def make(*args, **kwargs):
        calls.append((args, kwargs))
        return predictor

    instance = GraspGenXWorker(WorkerContext("grasp-test"), "vendor/GraspGenX", checkpoint_dir="weights",
                              assets_dir="grippers", gripper_id="chosen", output_dir="outputs", settings={"seed": 7},
                              service="test/service", frames=frames,
                              subscriber_factory=lambda service: calls.append(service) or inbox,
                              publisher_factory=lambda service: calls.append(service) or output, predictor_factory=make)
    return instance, inbox, output, predictor, calls


def send(instance, inbox, request):
    inbox.value = request
    instance.step_once()


def test_one_lazy_predictor_exact_mesh_request_duplicate_last_result_and_cleanup():
    instance, inbox, output, predictor, calls = worker(frames=2)
    instance.on_start()
    try:
        instance.step_once()
        assert predictor.starts == 0 and calls == list(services("test/service"))
        request = GraspRequest("job", "asset", "A", "object.ply", .001, True)
        send(instance, inbox, request)
        assert [result.status for result in output.results] == ["running", "done"]
        assert predictor.requests == [request]
        assert calls[-1] == (("vendor/GraspGenX",), {"checkpoint_dir": "weights", "assets_dir": "grippers",
                                                   "gripper_id": "chosen", "output_dir": "outputs", "settings": {"seed": 7}})
        done = output.results[-1]
        assert (done.request_id, done.asset_id, done.object_id, done.mesh_path, done.mesh_scale_to_m) == (
            "job", "asset", "A", "object.ply", .001)
        assert done.gripper_id == "chosen" and done.bundle_path == "bundle.json" and done.cached is False
        send(instance, inbox, request)
        assert output.results[-1] is done and len(predictor.requests) == 1
        send(instance, inbox, replace(request, request_id="new", regenerate=False))
        assert len(predictor.requests) == 2 and output.results[-1].cached is True
        assert instance.processed_count == 2 and instance.ctx.stop_event.is_set()
    finally:
        instance.on_stop()
    assert predictor.closed and inbox.closed and output.closed and predictor.starts == 1


def test_actual_predict_failure_reports_error_and_next_request_runs_without_retry():
    instance, inbox, output, predictor, _calls = worker()
    instance.on_start()
    try:
        predictor.error = RuntimeError("injected failure")
        request = GraspRequest("bad", "asset", "A", "object.ply")
        send(instance, inbox, request)
        assert output.results[-1].status == "error" and output.results[-1].error == "RuntimeError: injected failure"
        assert output.results[-1].grasps is None
        instance.step_once()
        assert len(predictor.requests) == 1
        predictor.error = None
        send(instance, inbox, replace(request, request_id="next"))
        assert output.results[-1].status == "done" and len(predictor.requests) == 2
    finally:
        instance.on_stop()


def test_failed_model_start_closes_predictor_and_endpoints():
    instance, inbox, output, predictor, _calls = worker()

    def fail():
        raise RuntimeError("start failed")

    predictor.start = fail
    instance.on_start()
    try:
        send(instance, inbox, GraspRequest("job", "asset", "A", "mesh.ply"))
        assert predictor.closed and instance.predictor is None
        assert output.results[-1].error == "RuntimeError: start failed"
    finally:
        instance.on_stop()
    assert inbox.closed and output.closed


def test_cli_settings_and_heavyweight_lazy_help():
    args = build_parser().parse_args(["--settings", '{"num_grasps": 12}'])
    assert args.settings == {"num_grasps": 12} and args.gripper_id == "franka_panda"
    script = """
import sys
for name in ('torch', 'trimesh', 'graspgen', 'pyzed', 'pyrealsense2', 'grasp_workbench.models.graspgenx'):
    sys.modules[name] = None
from grasp_workbench.workers.worker_graspgenx import main
main(['--help'])
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert "--gripper-id" in completed.stdout and "--checkpoint-dir" in completed.stdout


@pytest.mark.parametrize("settings", ["[]", "null", "3"])
def test_cli_requires_mapping_settings(settings):
    from grasp_workbench.workers.worker_graspgenx import main

    with pytest.raises(SystemExit) as error:
        main(["--settings", settings])
    assert error.value.code == 2
