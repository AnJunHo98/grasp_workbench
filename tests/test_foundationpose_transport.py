"""Actual independent FoundationPose process and native IPC, fake GPU backend."""
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
from grasp_workbench.ipc.foundationpose import (
    PoseControl, PoseControlPublisher, PoseResultSubscriber, PoseTarget, services,
)

ROOT = Path(__file__).resolve().parents[1]
_WORKER = """
import json, sys
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.workers.worker_foundationpose import FoundationPoseWorker

class Predictor:
    def __init__(self):
        self.starts = self.closes = 0
        self.registered = []
        self.tracked = []
        self.removed = []
    def start(self):
        self.starts += 1
    def register(self, object_id, frame, mask, mesh_path, scale):
        self.registered.append(object_id)
        assert frame.metadata['sequence'] == 17
        np.testing.assert_array_equal(frame.rgb, np.arange(36,dtype=np.uint8).reshape(3,4,3))
        np.testing.assert_array_equal(frame.depth_m,np.full((3,4),.75,np.float32))
        expected = np.arange(12).reshape(3,4) % 2 == (0 if object_id == 'A' else 1)
        np.testing.assert_array_equal(mask,expected)
        assert mesh_path == object_id + '.glb' and scale == .001
        pose = np.eye(4)
        pose[2,3] = .75
        return pose
    def track(self, object_id, frame):
        self.tracked.append([object_id,frame.metadata['sequence']])
        pose = np.eye(4)
        pose[0,3] = frame.metadata['sequence'] * .001
        pose[1,3] = .01 if object_id == 'A' else .02
        pose[2,3] = .75
        return pose
    def remove(self, object_id):
        self.removed.append(object_id)
        print(json.dumps({'event':'removed','object_id':object_id}),flush=True)
    def close(self):
        self.closes += 1

predictor = Predictor()
worker = FoundationPoseWorker(WorkerContext('fake-foundationpose'),'unused-source',
    service=sys.argv[1],predictor_factory=lambda *a,**k:predictor)
try:
    worker.run()
finally:
    print(json.dumps({'event':'model-summary','starts':predictor.starts,
        'closes':predictor.closes,'registered':predictor.registered,
        'tracked':predictor.tracked,'removed':predictor.removed}),flush=True)
"""


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    pytest.fail("Timed out waiting for FoundationPose IPC")


def test_retained_a_b_register_track_and_explicit_full_snapshot_stop(tmp_path):
    pytest.importorskip("iceoryx2")
    service = f"foundationpose-worker/{uuid4().hex}"
    control_service, result_service = services(service)
    camera = CameraPublisher(service)
    controls = PoseControlPublisher(control_service)
    results = PoseResultSubscriber(result_service)
    source = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
                         np.full((3, 4), .75, np.float32), {"sequence": 17})
    a = PoseTarget("A", "session-A", "A.glb", .001, source,
                   np.arange(12).reshape(3, 4) % 2 == 0)
    b = PoseTarget("B", "session-B", "B.glb", .001, source,
                   np.arange(12).reshape(3, 4) % 2 == 1)
    worker = None
    output = error = ""
    log_path = tmp_path / "worker.log"
    try:
        # Both desired sessions are retained before the worker exists.
        assert controls.publish(PoseControl((a, b))) == 0
        with log_path.open("w") as log:
            worker = subprocess.Popen(
                [sys.executable, "-B", "-u", "-c", _WORKER, service], cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=log,
                stderr=subprocess.PIPE, text=True,
            )

            def next_result(sequence, object_ids):
                def receive():
                    assert worker.poll() is None, log_path.read_text()
                    controls.update_connections()
                    result = results.receive_latest()
                    if result is None or result.frame.metadata["sequence"] != sequence:
                        return None
                    return result if [obj.object_id for obj in result.objects] == object_ids else None

                return wait_for(receive)

            registered = next_result(17, ["B"])
            assert registered.objects[0].status == "tracking"
            for sequence in (18, 19):
                camera.publish(CameraFrame(source.rgb, source.depth_m, {"sequence": sequence}))
                tracked = next_result(sequence, ["A", "B"])
                for index, obj in enumerate(tracked.objects, start=1):
                    assert obj.status == "tracking"
                    assert obj.camera_from_object[0, 3] == sequence * .001
                    assert obj.camera_from_object[1, 3] == index * .01

            controls.publish(PoseControl((b,)))
            wait_for(lambda: '"object_id": "A"' in log_path.read_text())
            camera.publish(CameraFrame(source.rgb, source.depth_m, {"sequence": 20}))
            remaining = next_result(20, ["B"])
            assert remaining.objects[0].session_id == "session-B"
            controls.publish(PoseControl(()))
            wait_for(lambda: '"object_id": "B"' in log_path.read_text())
    finally:
        if worker is not None:
            if worker.poll() is None:
                worker.send_signal(signal.SIGTERM)
            try:
                _, error = worker.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                _, error = worker.communicate()
                pytest.fail("FoundationPose worker did not stop")
            output = log_path.read_text()
        results.close()
        controls.close()
        camera.close()
    assert worker.returncode == 0, f"{output}\n{error}"
    records = [json.loads(line) for line in output.splitlines()]
    summary = next(record for record in records if record["event"] == "model-summary")
    assert summary == {
        "event": "model-summary", "starts": 1, "closes": 1,
        "registered": ["A", "B"],
        "tracked": [["A", 18], ["B", 18], ["A", 19], ["B", 19], ["B", 20]],
        "removed": ["A", "B"],
    }
