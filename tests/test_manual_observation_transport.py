"""Frozen manual refinement and explicit FP initialization over native worker IPC."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher
from grasp_workbench.ipc.foundationpose import PoseControl, PoseControlPublisher, PoseResultSubscriber, PoseTarget
from grasp_workbench.ipc.foundationpose import services as pose_services
from grasp_workbench.ipc.sam3 import ObjectPrompt, PromptPublisher, PromptSet, Sam3Subscriber
from grasp_workbench.ipc.sam3 import services as sam3_services
from grasp_workbench.ipc.sam3_refine import RefineRequest, RefineRequestPublisher, RefineResultSubscriber
from grasp_workbench.ipc.sam3_refine import services as refine_services

ROOT = Path(__file__).resolve().parents[1]
_COMMON = """
import json, sys
for name in ('torch', 'PIL', 'openai', 'pyzed', 'pyrealsense2', 'estimater', 'sam3'):
    sys.modules[name] = None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext

def refined_mask():
    mask = np.zeros((16,20),bool)
    mask[3:9,3:9] = True
    return mask

def frozen(frame):
    assert frame.metadata['sequence'] == 37
    np.testing.assert_array_equal(frame.rgb,np.arange(960,dtype=np.uint8).reshape(16,20,3))
    depth = np.full((16,20),.75,np.float32)
    depth[0,0] = np.nan
    np.testing.assert_array_equal(frame.depth_m,depth)

def initial_pose():
    matrix = np.eye(4)
    matrix[:3,3] = [.05,-.04,.75]
    return matrix
"""
_SAM3 = _COMMON + """
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result
from grasp_workbench.workers.worker_sam3 import Sam3Worker

class Predictor:
    def __init__(self, *args, **kwargs):
        self.starts, self.continuous, self.refinements = 0, [], []
    def start(self):
        self.starts += 1
    def predict(self, frame, prompts):
        assert prompts.revision == 'continuous-revision'
        assert [(item.object_id,item.text) for item in prompts.objects] == [('A','continuous cup')]
        assert frame.metadata['sequence'] in (100,101)
        assert (frame.rgb == frame.metadata['sequence'] + 101).all()
        self.continuous.append([prompts.revision,frame.metadata['sequence']])
        return Sam3Result(frame,prompts.revision,(Sam3ObjectResult('A','continuous cup',
            refined_mask()[None],np.array([[3,3,9,9]],np.float32),np.array([.8],np.float32)),))
    def refine(self, request):
        frozen(request.frame)
        assert request.object_id == 'B' and request.prompt == 'yellow can'
        assert request.boxes == ({'xyxy':[3,3,9,9],'label':True},
                                  {'xyxy':[12,10,16,14],'label':False})
        assert request.points == ({'xy':[6,6],'label':1},{'xy':[14,12],'label':0})
        seed = np.zeros((16,20),bool)
        seed[4:8,4:8] = True
        np.testing.assert_array_equal(request.seed_mask,seed)
        self.refinements.append(request.request_id)
        return Sam3Result(request.frame,request.request_id,(Sam3ObjectResult('B','yellow can',
            refined_mask()[None],np.array([[3,3,9,9]],np.float32),np.array([.9],np.float32)),))
    def close(self):
        print(json.dumps({'event':'sam3-model-summary','starts':self.starts,
            'continuous':self.continuous,'refinements':self.refinements}),flush=True)

Sam3Worker(WorkerContext('fake-sam3'),'unused-checkpoint',service=sys.argv[1],
    predictor_factory=Predictor).run()
"""
_FP = _COMMON + """
from grasp_workbench.workers.worker_foundationpose import FoundationPoseWorker

class Predictor:
    def __init__(self, *args, **kwargs):
        self.registered, self.seeded, self.tracked = [], [], []
        self.seed_needs_initial_frame = False
    def start(self):
        pass
    def register(self, object_id, frame, mask, mesh_path, scale):
        assert object_id == 'B' and mesh_path == 'registered.ply' and scale == .001
        frozen(frame)
        np.testing.assert_array_equal(mask,refined_mask())
        self.registered.append([object_id,frame.metadata['sequence'],int(mask.sum())])
        return initial_pose()
    def seed(self, object_id, pose, mesh_path, scale):
        assert object_id == 'A' and mesh_path == 'seeded.ply' and scale == 1.
        np.testing.assert_array_equal(pose,initial_pose())
        self.seeded.append(object_id)
        self.seed_needs_initial_frame = True
    def track(self, object_id, frame):
        if object_id == 'A' and self.seed_needs_initial_frame:
            frozen(frame)
            self.seed_needs_initial_frame = False
        else:
            assert frame.metadata['sequence'] == 101
        self.tracked.append([object_id,frame.metadata['sequence']])
        return initial_pose()
    def remove(self, object_id):
        pass
    def close(self):
        print(json.dumps({'event':'fp-model-summary','registered':self.registered,
            'seeded':self.seeded,'tracked':self.tracked}),flush=True)

FoundationPoseWorker(WorkerContext('fake-foundationpose'),'unused-backend',service=sys.argv[1],
    predictor_factory=Predictor).run()
"""


def test_frozen_refinement_and_register_or_seed_keep_source_while_live_stream_advances(tmp_path):
    pytest.importorskip("iceoryx2")
    service = f"manual-observation/{uuid4().hex}"
    prompt_service, sam3_service = sam3_services(service)
    refine_request_service, refine_result_service = refine_services(service)
    control_service, pose_result_service = pose_services(service)
    camera = CameraPublisher(service)
    prompts, segmentation = PromptPublisher(prompt_service), Sam3Subscriber(sam3_service)
    refinement = RefineRequestPublisher(refine_request_service)
    refined = RefineResultSubscriber(refine_result_service)
    control, poses = PoseControlPublisher(control_service), PoseResultSubscriber(pose_result_service)
    publishers = (prompts, refinement, control)
    endpoints = (camera, prompts, segmentation, refinement, refined, control, poses)
    workers, logs, streams = {}, {}, []
    continuous_results, refinement_results, pose_results = [], [], []
    metadata = {"camera_id": "manual-camera", "stream_id": "manual-stream", "sequence": 37,
                "capture_time_ns": 1234567, "intrinsics": {"fx": 30., "fy": 30., "cx": 10., "cy": 8.}}
    source = CameraFrame(np.arange(960, dtype=np.uint8).reshape(16, 20, 3),
                         np.full((16, 20), .75, np.float32), metadata)
    source.depth_m[0, 0] = np.nan

    def live(sequence):
        return CameraFrame(np.full((16, 20, 3), sequence + 101, np.uint8),
                           np.full((16, 20), .9, np.float32), {**metadata, "sequence": sequence})

    def wait_for(predicate, frame=None):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for publisher in publishers:
                publisher.update_connections()
            if frame is not None:
                camera.publish(frame)
            for subscriber, results in ((segmentation, continuous_results), (refined, refinement_results),
                                        (poses, pose_results)):
                result = subscriber.receive_latest()
                if result is not None:
                    results.append(result)
            for name, worker in workers.items():
                assert worker.poll() is None, logs[name].read_text()
            result = predicate()
            if result:
                return result
            time.sleep(.01)
        pytest.fail("Timed out waiting for native manual refinement/pose initialization")

    try:
        for name, script in (("sam3", _SAM3), ("foundationpose", _FP)):
            logs[name] = tmp_path / (name + ".log")
            stream = logs[name].open("w")
            streams.append(stream)
            workers[name] = subprocess.Popen(
                [sys.executable, "-B", "-u", "-c", script, service], cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=stream, stderr=subprocess.STDOUT)
        wait_for(lambda: all('"event": "ready"' in log.read_text() for log in logs.values()))
        prompts.publish(PromptSet("continuous-revision", (ObjectPrompt("A", "continuous cup"),)))
        wait_for(lambda: any(item.frame.metadata["sequence"] == 100 for item in continuous_results), live(100))
        seed_mask = np.zeros((16, 20), bool)
        seed_mask[4:8, 4:8] = True
        request = RefineRequest("manual-refine", "B", "yellow can", source,
                                ({"xyxy": [3, 3, 9, 9], "label": True},
                                 {"xyxy": [12, 10, 16, 14], "label": False}),
                                ({"xy": [6, 6], "label": 1}, {"xy": [14, 12], "label": 0}), seed_mask)
        refinement.publish(request)
        terminal = wait_for(lambda: next((item for item in refinement_results if item.status in {"done", "error"}), None),
                            live(100))
        assert terminal.status == "done", terminal.error
        observation = terminal.result
        assert terminal.request_id == observation.revision == request.request_id
        assert observation.frame.metadata == source.metadata and observation.objects[0].object_id == "B"
        np.testing.assert_array_equal(observation.frame.rgb, source.rgb)
        np.testing.assert_array_equal(observation.frame.depth_m, source.depth_m)
        expected = np.zeros((16, 20), bool)
        expected[3:9, 3:9] = True
        np.testing.assert_array_equal(observation.objects[0].masks[0], expected)
        normal = wait_for(lambda: next((item for item in continuous_results
                                      if item.frame.metadata["sequence"] == 101), None), live(101))
        assert normal.revision == "continuous-revision" and normal.revision != observation.revision
        initial = np.eye(4)
        initial[:3, 3] = [.05, -.04, .75]
        registered = PoseTarget("B", "registered-session", "registered.ply", .001,
                                observation.frame, observation.objects[0].masks[0], request.request_id)
        seeded = PoseTarget("A", "seeded-session", "seeded.ply", 1., source, None,
                            prompt_revision=request.request_id, initial_pose=initial, initialization="seed")
        control.publish(PoseControl((registered, seeded)))
        result = wait_for(lambda: next((item for item in reversed(pose_results)
                                       if item.frame.metadata["sequence"] == 101
                                       and {pose.object_id for pose in item.objects} == {"A", "B"}), None), live(101))
        assert all(item.status == "tracking" for item in result.objects)
        for item in result.objects:
            np.testing.assert_array_equal(item.camera_from_object, initial)
        np.testing.assert_array_equal(source.rgb, np.arange(960, dtype=np.uint8).reshape(16, 20, 3))
        assert seed_mask.sum() == 16
    finally:
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
        for endpoint in reversed(endpoints):
            endpoint.close()
    summaries = {}
    for name, worker in workers.items():
        assert worker.returncode == 0, logs[name].read_text()
        records = [json.loads(line) for line in logs[name].read_text().splitlines() if line.startswith("{")]
        summaries[name] = next(item for item in records if item["event"].endswith("model-summary"))
    assert summaries["sam3"]["starts"] == 1 and summaries["sam3"]["refinements"] == ["manual-refine"]
    assert summaries["sam3"]["continuous"] == [["continuous-revision", 100], ["continuous-revision", 101]]
    assert summaries["foundationpose"]["registered"] == [["B", 37, 36]]
    assert summaries["foundationpose"]["seeded"] == ["A"]
    a_tracks = [sequence for key, sequence in summaries["foundationpose"]["tracked"] if key == "A"]
    assert a_tracks[0] == 37 and 101 in a_tracks
    assert ["B", 101] in summaries["foundationpose"]["tracked"]
