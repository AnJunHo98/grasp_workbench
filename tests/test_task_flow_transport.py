"""Explicit Source/Target task advances once through real UI/native workers."""
import json
import os
from pathlib import Path
import signal
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

from grasp_workbench.assets import AssetStore
from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Publisher, Sam3Result
from grasp_workbench.ipc.sam3 import services as sam3_services
from grasp_workbench.ipc.sam3d import MeshResult
from grasp_workbench.ipc.vlm import VlmRequestSubscriber, VlmResultPublisher
from grasp_workbench.ipc.vlm import services as vlm_services
from grasp_workbench.ui.camera_window import CameraWindow
from test_semantic_session import semantic_result

ROOT = Path(__file__).resolve().parents[1]
_COMMON = """
import json, sys
from pathlib import Path
for name in ('torch','openai','pyzed','pyrealsense2','estimater','graspgenx','curobo'):
    sys.modules[name] = None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
"""
_FP = _COMMON + """
from grasp_workbench.workers.worker_foundationpose import FoundationPoseWorker

class Predictor:
    def __init__(self,*args,**kwargs):
        self.registered, self.tracked = [], []
    def start(self):pass
    def register(self,object_id,frame,mask,mesh_path,scale):
        assert frame.metadata['sequence'] == 37 and (frame.rgb == 40).all()
        assert (frame.depth_m == 1).all() and Path(mesh_path).is_file() and scale == 1.
        expected = np.zeros((20,30),bool)
        x = 0 if object_id == 'A' else 15
        expected[:10,x:x+10] = True
        np.testing.assert_array_equal(mask,expected)
        self.registered.append(object_id)
        return self.pose(object_id,37)
    def pose(self,key,sequence):
        matrix = np.eye(4)
        matrix[:3,3] = [(-10 if key == 'A' else 5)/30 + .001*(sequence-37),-5/30,1]
        return matrix
    def track(self,object_id,frame):
        self.tracked.append([object_id,frame.metadata['sequence']])
        return self.pose(object_id,frame.metadata['sequence'])
    def remove(self,object_id):pass
    def close(self):
        print(json.dumps({'event':'fp-model-summary','registered':self.registered,'tracked':self.tracked}),flush=True)

FoundationPoseWorker(WorkerContext('fake-fp'),'unused-backend',service=sys.argv[1],predictor_factory=Predictor).run()
"""
_GRASP = _COMMON + """
from grasp_workbench.models.graspgenx import GraspPrediction
from grasp_workbench.workers.worker_graspgenx import GraspGenXWorker

class Predictor:
    def __init__(self,*args,**kwargs):self.calls=[]
    def start(self):pass
    def predict(self,request):
        assert request.object_id == 'B' and not request.regenerate
        assert Path(request.mesh_path).is_file()
        self.calls.append({'request_id':request.request_id,'object_id':request.object_id})
        grasps = np.tile(np.eye(4),(2,1,1))
        grasps[:,0,3] = [-.03,.03]
        bundle = Path(sys.argv[2])
        bundle.write_text(json.dumps({'robot_command_published':False}))
        return GraspPrediction(grasps,np.array([.9,.8]),str(bundle),{})
    def close(self):
        print(json.dumps({'event':'grasp-model-summary','calls':self.calls}),flush=True)

GraspGenXWorker(WorkerContext('fake-grasp'),service=sys.argv[1],predictor_factory=Predictor).run()
"""
_SELECTION = _COMMON + """
from grasp_workbench.workers.worker_grasp_selection import GraspSelectionWorker

class Predictor:
    def __init__(self,**kwargs):
        assert kwargs['robot_config'] is None
        self.calls=[]
    def start(self):pass
    def predict(self,request):
        assert request.action == 'select' and request.coordination['arm_mode'] == 'dual'
        by_id = {item['observation_id']:item for item in request.objects}
        assert set(by_id) == {'source-obs','target-obs'}
        source, target = by_id['source-obs'], by_id['target-obs']
        assert source['object_id'] == 'B' and source['bundle_path']
        assert target['object_id'] == 'A' and target['bundle_path'] is None
        assert all(item['camera_from_object'] is not None for item in request.objects)
        assert {item['arm'] for item in request.coordination['assignments']} == {'left','right'}
        assert {item['observation_id'] for item in request.coordination['assignments']} == {'source-obs'}
        self.calls.append({'request_id':request.request_id,'source_slot':source['object_id'],
            'target_slot':target['object_id'],'assigned':[item['observation_id'] for item in request.coordination['assignments']]})
        targets = {}
        for index, arm in enumerate(('left','right')):
            canonical = np.eye(4)
            canonical[0,3] = -.03 if arm == 'left' else .03
            targets[arm] = {'object_id':'B','observation_id':'source-obs','index':index,
                'object_from_gripper':canonical.tolist(),'camera_from_gripper':None,'base_from_gripper':None,
                'base_from_tcp':None,'base_from_pregrasp':None}
        return {'arm_mode':'dual','objects':{key:{'base_from_object':None} for key in by_id},
            'candidates':[{'candidate_id':'task-pair','targets':targets,'model_score':.9,'geometry_score':.8,
                'pair_score':.8,'ik_status':'not_checked','reachability_score':None}],
            'robot_connected':False,'robot_ik_checked':False,'self_collision_checked':False,
            'scene_collision_checked':False,'trajectory_checked':False,'robot_command_published':False}
    def close(self):
        print(json.dumps({'event':'selection-model-summary','calls':self.calls}),flush=True)

GraspSelectionWorker(WorkerContext('fake-selection'),service=sys.argv[1],predictor_factory=Predictor).run()
"""


@pytest.mark.parametrize("explicit_task_button", [True, False])
def test_source_b_task_runs_pose_grasp_selection_once_across_live_pose_updates(tmp_path, explicit_task_button):
    pytest.importorskip("iceoryx2")
    pytest.importorskip("trimesh")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = tmp_path / "object.obj"
    source.write_text("v 0 0 0\nv .1 0 0\nv 0 .1 0\nf 1 2 3\n")
    asset_dir = tmp_path / "assets"
    AssetStore(asset_dir).save(MeshResult("saved", "same-asset", "A", "saved", "done",
                                        mesh_path=str(source), length_unit="metre"), "tray", "tray")
    service = "task-flow-native/" + uuid4().hex
    camera = CameraPublisher(service)
    segmentation = Sam3Publisher(sam3_services(service)[1])
    request_service, result_service = vlm_services(service)
    semantic_requests, semantic_results = VlmRequestSubscriber(request_service), VlmResultPublisher(result_service)
    window = CameraWindow(service, vlm=True, foundationpose=True, graspgenx=True, grasp_selection=True,
                          asset_dir=asset_dir, session_dir=tmp_path / "sessions")
    window.timer.stop()
    workers, logs, streams = {}, {}, []
    metadata = {"camera_id": "example", "sequence": 37,
                "intrinsics": {"fx": 30., "fy": 30., "cx": 15., "cy": 10.}}
    frame = CameraFrame(np.full((20, 30, 3), 40, np.uint8), np.ones((20, 30), np.float32), metadata)

    def wait_for(predicate, *, camera_frame=None):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            camera.publish(camera_frame or frame)
            semantic_results.update_connections()
            app.processEvents()
            window.poll_once()
            assert window.receiver.snapshot()["error"] is None
            for name, worker in workers.items():
                assert worker.poll() is None, logs[name].read_text()
            result = predicate()
            if result:
                return result
            time.sleep(.01)
        pytest.fail(f"Timed out waiting for explicit task flow: {window.task_status.text()}")

    try:
        for name, script, args in (("fp", _FP, []), ("grasp", _GRASP, [str(tmp_path / "bundle.json")]),
                                   ("selection", _SELECTION, [])):
            logs[name] = tmp_path / (name + ".log")
            stream = logs[name].open("w")
            streams.append(stream)
            workers[name] = subprocess.Popen([sys.executable, "-B", "-u", "-c", script, service, *args],
                                             cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
                                             stdout=stream, stderr=subprocess.STDOUT)
        wait_for(lambda: window.receiver.snapshot()["ready"]
                 and all('"event": "ready"' in log.read_text() for log in logs.values()))
        window.vlm_instruction.setText("move source onto target")
        window.vlm_button.click()
        request = wait_for(semantic_requests.receive_latest)
        semantic_results.publish(semantic_result(request))
        wait_for(lambda: window._semantic_session is not None)
        assert window.source_role_input.currentData() == "source-obs"
        assert window.target_role_input.currentData() == "target-obs"
        assert not window._pose_targets and not window._grasp_requests
        if explicit_task_button:
            window.task_button.click()
        task_id = window._task_flow["request_id"]
        masks = np.zeros((2, 20, 30), bool)
        masks[0, :10, :10] = True
        masks[1, :10, 15:25] = True
        objects = tuple(Sam3ObjectResult(key, "tray", masks[index:index + 1],
                                        np.array([[x, 0, x + 10, 10]], np.float32), np.array([.9], np.float32))
                        for index, (key, x) in enumerate((("A", 0), ("B", 15))))
        segmentation.publish(Sam3Result(frame, window._prompts.revision, objects))
        wait_for(lambda: window._task_flow is not None and window._task_flow["phase"] == "done")
        assert window._task_flow["request_id"] == task_id
        assert set(window._pose_targets) == set(window._pose_objects) == {"A", "B"}
        assert set(window._grasp_requests) == set(window._grasp_results) == {"B"}
        assert set(window._task_flow["grasp_jobs"]) == {"B"}
        selection_id = window._selection_request.request_id
        grasp_id = window._grasp_requests["B"].request_id
        assert window._task_flow["selection_id"] == selection_id
        assert window._selection_result["robot_command_published"] is False
        assert set(window._selected_selection_candidate()["targets"]) == {"left", "right"}
        assert {target["object_id"] for target in window._selected_selection_candidate()["targets"].values()} == {"B"}
        for sequence in (38, 39, 40):
            newer = CameraFrame(np.full_like(frame.rgb, 80), frame.depth_m, {**metadata, "sequence": sequence})
            wait_for(lambda: window._pose_result is not None
                     and window._pose_result.frame.metadata["sequence"] == sequence, camera_frame=newer)
            assert window._task_flow["phase"] == "done"
            assert window._selection_request.request_id == selection_id
            assert window._grasp_requests["B"].request_id == grasp_id and "A" not in window._grasp_requests
        assert "camera=example · sequence=40 ·" in window.metadata_label.text()
    finally:
        window.receiver.stop()
        window.receiver.join(timeout=3)
        window.close()
        app.processEvents()
        for worker in workers.values():
            if worker.poll() is None:
                worker.send_signal(signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        for stream in streams:
            stream.close()
        for endpoint in (semantic_results, semantic_requests, segmentation, camera):
            endpoint.close()
    summaries = {}
    for name, worker in workers.items():
        assert worker.returncode == 0, logs[name].read_text()
        records = [json.loads(line) for line in logs[name].read_text().splitlines() if line.startswith("{")]
        summaries[name] = next(item for item in records if item["event"].endswith("model-summary"))
    assert sorted(summaries["fp"]["registered"]) == ["A", "B"]
    assert len(summaries["grasp"]["calls"]) == len(summaries["selection"]["calls"]) == 1
    assert summaries["grasp"]["calls"][0] == {"request_id": grasp_id, "object_id": "B"}
    assert summaries["selection"]["calls"][0] == {
        "request_id": selection_id, "source_slot": "B", "target_slot": "A", "assigned": ["source-obs", "source-obs"]}
    assert all([key, 40] in summaries["fp"]["tracked"] for key in ("A", "B"))
