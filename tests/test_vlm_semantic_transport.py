"""Semantic tasks survive a real VLM worker, native IPC, Qt and session export."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher
from grasp_workbench.ui.camera_window import CameraWindow

ROOT = Path(__file__).resolve().parents[1]
WORKER = """
import json,sys
for name in ('openai','PIL','torch','pyzed','pyrealsense2'):
    sys.modules[name]=None
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.vlm import VlmObject
from grasp_workbench.models.vlm import VlmPrediction
from grasp_workbench.workers.worker_vlm import VlmWorker

class Predictor:
    def __init__(self,**kwargs):self.calls=0
    def start(self):pass
    def predict(self,request):
        self.calls+=1
        assert request.frame.metadata['sequence']==37
        assert len(request.policies)==1 and request.policies[0]['policy_id']=='place-example'
        assert 'checkpoint' not in json.dumps(request.policies)
        assert 'private-weight' not in json.dumps(request.policies)
        return VlmPrediction(
            (VlmObject('파란 받침','blue pad',(1,1,9,9),None,'pad-now'),
             VlmObject('빨간 컵','red cup',(12,1,20,9),None,'cup-now')),
            {'source_observation_id':'cup-now','target_observation_id':'pad-now','intent':'컵을 받침 위로 옮기기'},
            {'task_type':'place_on_surface','intent_origin':'explicit','policy_id':'place-example','reason':'작업 일치'},
            {'arm_mode':'dual','assignments':[
                {'arm':'left','observation_id':'cup-now','role':'함께 들기','grasp_region':'왼쪽 몸체'},
                {'arm':'right','observation_id':'cup-now','role':'함께 들기','grasp_region':'오른쪽 몸체'}],
             'constraints':[{'reference':{'frame':'left_ee','observation_id':None},
                             'moving':{'frame':'right_ee','observation_id':None},
                             'behavior':'maintain_relative_pose','phases':['transport'],
                             'reference_state':'양손 파지 확인 이후','description':'운반 중 관계 유지'}],
             'reason':'양팔로 옮기는 지시'})
    def close(self):print(json.dumps({'event':'semantic-model-summary','calls':self.calls}),flush=True)

VlmWorker(WorkerContext('fake-semantic-vlm'),service=sys.argv[1],predictor_factory=Predictor).run()
"""


def test_worker_semantics_map_source_both_arms_by_id_and_export_without_execution(tmp_path):
    pytest.importorskip("iceoryx2")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    catalog = tmp_path / "policies.json"
    catalog.write_text(json.dumps([{
        "policy_id": "place-example", "display_name": "받침 위로 옮기기", "task_type": "place_on_surface",
        "description": "Source를 Target 위로 옮기기", "training_source": {"asset_id": "cup", "name": "컵"},
        "training_target": {"asset_id": "pad", "name": "받침"}, "pose_convention": "target_from_source",
        "transfer_note": "예제 metadata이며 실행 미검증", "checkpoint": "private-weight.bin",
    }], ensure_ascii=False))
    service = "semantic-vlm-native/" + uuid4().hex
    camera = CameraPublisher(service)
    window = CameraWindow(service, vlm=True, policy_catalog=catalog, session_dir=tmp_path / "tasks",
                          asset_dir=tmp_path / "assets")
    window.timer.stop()
    frame = CameraFrame(np.full((16, 24, 3), 40, np.uint8), np.ones((16, 24), np.float32),
                        {"sequence": 37, "camera_id": "semantic-example"})
    log = tmp_path / "worker.log"
    worker = None

    def wait_for(predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            camera.publish(frame)
            app.processEvents()
            window.poll_once()
            assert window.receiver.snapshot()["error"] is None
            if worker is not None:
                assert worker.poll() is None, log.read_text()
            if predicate():
                return
            time.sleep(.005)
        pytest.fail("Timed out waiting for native semantic task results")

    try:
        with log.open("w") as stream:
            worker = subprocess.Popen([sys.executable, "-B", "-u", "-c", WORKER, service], cwd=ROOT,
                                      env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=stream,
                                      stderr=subprocess.STDOUT)
            wait_for(lambda: window.vlm_button.isEnabled() and '"event": "ready"' in log.read_text())
            window.vlm_instruction.setText("빨간 컵을 양팔로 들어 파란 받침 위로 옮겨주세요")
            window.vlm_button.click()
            wait_for(lambda: window._semantic_session is not None)
            record = window._semantic_session.record
            assert record["task_binding"]["source_observation_id"] == "cup-now"
            assert record["task_binding"]["target_observation_id"] == "pad-now"
            assert window.prompt_inputs["A"].text() == "blue pad"
            assert window.prompt_inputs["B"].text() == "red cup"
            assert record["coordination"]["arm_mode"] == "dual"
            assert {item["observation_id"] for item in record["coordination"]["assignments"]} == {"cup-now"}
            assert record["policy_selection"]["policy_id"] == "place-example"
            assert not window._pose_targets and not window._grasp_requests
            saved = tmp_path / "saved-session.json"
            assert window.save_semantic_session(saved) == saved
            request_path = tmp_path / "execution-request.json"
            exported = window.export_semantic_request(request_path)
            assert exported["execution"] == "request_only"
            assert exported["robot_command_published"] is False
            arms = exported["coordination"]["assignments"]
            assert arms[0]["instance_id"] == arms[1]["instance_id"]
            assert arms[0]["instance_id"] == exported["task_binding"]["source_instance_id"]
            assert exported["policy_reference"]["checkpoint"] == str(tmp_path / "private-weight.bin")
            before = dict(window._pose_targets)
            assert window.load_semantic_session(saved) == saved
            assert window._semantic_session.record["coordination"]["arm_mode"] == "dual"
            assert window._pose_targets == before
            frame = CameraFrame(np.full_like(frame.rgb, 80), frame.depth_m, {**frame.metadata, "sequence": 38})
            wait_for(lambda: window._latest_frame.metadata["sequence"] == 38)
    finally:
        window.receiver.stop()
        window.receiver.join(timeout=3)
        window.close()
        app.processEvents()
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        camera.close()
    assert worker.returncode == 0, log.read_text()
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    assert next(row["calls"] for row in rows if row["event"] == "semantic-model-summary") == 1
