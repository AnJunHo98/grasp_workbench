"""Selection IPC, independent lifecycle and portable optional YAML wiring."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from grasp_workbench.config import load_run_config
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.grasp_selection import (
    SelectionRequest, SelectionRequestPublisher, SelectionResult, SelectionResultSubscriber,
    decode_request, decode_result, encode_request, encode_result, services,
)
from grasp_workbench.main import build_commands, build_parser
from grasp_workbench.workers.worker_grasp_selection import GraspSelectionWorker
from test_graspgenx_worker import Inbox, Output

ROOT = Path(__file__).resolve().parents[1]


def request():
    return SelectionRequest("selection-1", {"arm_mode": "dual", "assignments": [
        {"arm": arm, "observation_id": "source", "role": "carry", "grasp_region": "body"}
        for arm in ("left", "right")], "constraints": [{
            "reference": {"frame": "object", "observation_id": "context"},
            "moving": {"frame": "object", "observation_id": "source"},
            "behavior": "allow_relative_motion", "phases": ["approach"],
            "reference_state": "observed", "description": "context has no grasp bundle"}], "reason": "same-object dual"}, (
        {"object_id": "A", "observation_id": "context", "instance_id": "instance-A",
         "bundle_path": None, "mesh_path": None, "mesh_scale_to_m": 1., "camera_from_object": None},
        {"object_id": "B", "observation_id": "source", "instance_id": "instance-B",
         "bundle_path": "/assets/cup/metadata.json", "mesh_path": "/assets/cup/object.ply",
         "mesh_scale_to_m": 1., "camera_from_object": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]}))


def test_codec_preserves_obs_roles_ungrasped_context_and_owns_records():
    value = request()
    decoded = decode_request(encode_request(value))
    assert decoded == value and decoded.objects[1]["object_id"] == "B"
    decoded.coordination["assignments"][0]["role"] = "changed"
    assert value.coordination["assignments"][0]["role"] == "carry"
    for status in ("running", "done", "error"):
        result = SelectionResult("selection-1", status, {"robot_command_published": False} if status == "done" else None,
                                 "failed" if status == "error" else None)
        assert decode_result(encode_result(result)) == result


@pytest.mark.parametrize("change", [
    {"coordination": {}}, {"objects": ({"object_id": "B"},)},
    {"request_id": None}, {"objects": (request().objects[1], request().objects[1])},
])
def test_codec_rejects_invalid_contract(change):
    with pytest.raises((ValueError, TypeError, KeyError)):
        encode_request(replace(request(), **change))


def test_lazy_worker_preserves_request_runs_once_and_reports_real_failure():
    inbox, output = Inbox(), Output()

    class Predictor:
        starts, calls, closes = 0, 0, 0

        def start(self):
            self.starts += 1

        def predict(self, value):
            self.calls += 1
            assert value.objects[1]["observation_id"] == "source"
            if self.calls == 2:
                raise RuntimeError("actual backend failure")
            return {"robot_ik_checked": True, "robot_command_published": False}

        def close(self):
            self.closes += 1

    predictor, options = Predictor(), []
    worker = GraspSelectionWorker(WorkerContext("selection"), robot_config="robot.yaml", settings={"top_k": 8},
                                  frames=2, subscriber_factory=lambda _: inbox, publisher_factory=lambda _: output,
                                  predictor_factory=lambda **kwargs: options.append(kwargs) or predictor)
    worker.on_start()
    try:
        worker.step_once()
        assert predictor.starts == 0
        inbox.value = request()
        worker.step_once()
        assert [value.status for value in output.results] == ["running", "done"]
        inbox.value = request()
        worker.step_once()
        assert predictor.calls == 1
        inbox.value = replace(request(), request_id="selection-2")
        worker.step_once()
        assert output.results[-1].status == "error" and "actual backend failure" in output.results[-1].error
        assert worker.ctx.stop_event.is_set() and predictor.calls == 2
        assert options == [{"robot_config": "robot.yaml", "settings": {"top_k": 8}}]
    finally:
        worker.on_stop()
    assert predictor.starts == predictor.closes == 1 and inbox.closed and output.closed


def test_yaml_run_starts_only_requested_stage_and_hides_robot_config_from_ui(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("ui: {enabled: false}\ngrasp_selection: {enabled: true, python: ik-env/bin/python, "
                    "robot_config: robots/example.yaml, settings: {top_k: 8}}\n")
    settings = load_run_config(path)
    commands = build_commands(build_parser(settings).parse_args(["run"]))
    ui, worker, camera = commands
    assert "--grasp-selection" in ui and "worker_camera_ui" in " ".join(ui)
    assert worker[0] == str(tmp_path / "ik-env/bin/python")
    assert worker[worker.index("--robot-config") + 1] == str(tmp_path / "robots/example.yaml")
    assert json.loads(worker[worker.index("--settings") + 1]) == {"top_k": 8}
    assert "--robot-config" not in ui + camera
    assert not any("worker_graspgenx" in " ".join(command) or "worker_vlm" in " ".join(command) for command in commands)


def test_heavyweight_libraries_not_imported_for_help():
    script = """
import sys
for name in ('torch','curobo','trimesh','object_pose_system','ffw_system','pyzed','pyrealsense2'):
    sys.modules[name]=None
from grasp_workbench.workers.worker_grasp_selection import main
main(['--help'])
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0 and "--robot-config" in completed.stdout


def test_native_worker_round_trip_with_only_predictor_injected(tmp_path):
    pytest.importorskip("iceoryx2")
    script = """
import sys
for name in ('torch','curobo','object_pose_system','ffw_system','pyzed','pyrealsense2'):
    sys.modules[name]=None
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.workers.worker_grasp_selection import GraspSelectionWorker
class Predictor:
    def __init__(self,**kwargs):pass
    def start(self):pass
    def predict(self,request):
        assert request.coordination['assignments'][0]['observation_id']=='source'
        assert request.objects[1]['object_id']=='B'
        return {'arm_mode':'dual','robot_connected':False,'robot_ik_checked':False,'robot_command_published':False}
    def close(self):pass
GraspSelectionWorker(WorkerContext('selection-native'),service=sys.argv[1],predictor_factory=Predictor).run()
"""
    service = "selection-native/" + uuid4().hex
    req_service, res_service = services(service)
    publisher, subscriber = SelectionRequestPublisher(req_service), SelectionResultSubscriber(res_service)
    process = None
    log = tmp_path / "worker.log"
    result = None
    try:
        with log.open("w") as stream:
            process = subprocess.Popen([sys.executable, "-B", "-u", "-c", script, service], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT)
            deadline, sent = time.monotonic() + 10, False
            while time.monotonic() < deadline:
                publisher.update_connections()
                if not sent and '"event": "ready"' in log.read_text():
                    publisher.publish(request())
                    sent = True
                incoming = subscriber.receive_latest()
                if incoming is not None and incoming.status in {"done", "error"}:
                    result = incoming
                    break
                assert process.poll() in (None, 0), log.read_text()
                time.sleep(.005)
            assert result is not None and result.status == "done", log.read_text()
            assert result.selection["arm_mode"] == "dual" and result.selection["robot_command_published"] is False
            process.terminate()
            assert process.wait(timeout=5) == 0, log.read_text()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        subscriber.close()
        publisher.close()
