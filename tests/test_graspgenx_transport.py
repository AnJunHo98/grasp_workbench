"""Real independent grasp worker + native IPC receiver; no GPU or camera SDK."""
from dataclasses import replace
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

from grasp_workbench.ipc.graspgenx import GraspRequest
from grasp_workbench.ui.receiver import LatestCameraReceiver

ROOT = Path(__file__).resolve().parents[1]
_WORKER = """
import json,sys
for name in ('torch','graspgen','trimesh','pyzed','pyrealsense2'):
    sys.modules[name]=None
from types import SimpleNamespace
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.workers.worker_graspgenx import GraspGenXWorker
class Predictor:
    def __init__(self):
        self.starts=self.closes=self.calls=0
        self.seen=set()
    def start(self):self.starts+=1
    def predict(self,request):
        self.calls+=1
        if request.asset_id=='fail':raise RuntimeError('injected generation failure')
        assert request.mesh_path==request.object_id+'.glb'
        assert request.mesh_scale_to_m==.001
        cached=request.asset_id in self.seen and not request.regenerate
        self.seen.add(request.asset_id)
        poses=np.repeat(np.eye(4)[None],2,axis=0)
        poses[:,0,3]=[.01,.02]
        return SimpleNamespace(grasps=poses,scores=np.asarray([.9,.8]),
            bundle_path=request.asset_id+'/bundle.json',gripper_metadata={'control_points':[[0,0,.1]]},cached=cached)
    def close(self):self.closes+=1
predictor=Predictor()
worker=GraspGenXWorker(WorkerContext('fake-graspgenx'),service=sys.argv[1],
    predictor_factory=lambda *a,**k:predictor)
try:worker.run()
finally:
    assert all(sys.modules[name] is None for name in ('torch','graspgen','trimesh','pyzed','pyrealsense2'))
    print(json.dumps({'event':'summary','starts':predictor.starts,'calls':predictor.calls,
        'closes':predictor.closes,'processed':worker.processed_count}),flush=True)
"""


def _wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.005)
    pytest.fail("Timed out waiting for grasp IPC")


def test_receiver_a_b_late_worker_raw_candidates_last_duplicate_error_and_stop():
    pytest.importorskip("iceoryx2")
    service = "grasp-transport/" + uuid4().hex
    receiver = LatestCameraReceiver(service, graspgenx=True)
    a, b = (GraspRequest("job-" + key, "asset-" + key, key, key + ".glb", .001) for key in ("A", "B"))
    receiver.submit_grasp(a)
    receiver.submit_grasp(b)
    receiver.start()
    child, output, error, events = None, "", "", []
    try:
        _wait_for(lambda: receiver.snapshot()["ready"])
        _wait_for(lambda: receiver._grasp_inflight is not None)
        child = subprocess.Popen([sys.executable, "-B", "-u", "-c", _WORKER, service], cwd=ROOT,
                                 env={**os.environ, "PYTHONPATH": str(ROOT)},
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def terminal(*request_ids):
            def receive():
                assert child.poll() is None
                assert receiver.snapshot()["error"] is None, receiver.snapshot()["error"]
                events.extend(receiver.take_grasp_results())
                matches = {item.request_id: item for item in events
                           if item.request_id in request_ids and item.status in {"done", "error"}}
                return matches if len(matches) == len(request_ids) else None

            return _wait_for(receive)

        completed = terminal(a.request_id, b.request_id)
        assert [item.request_id for item in events if item.status == "done"] == [a.request_id, b.request_id]
        for request in (a, b):
            result = completed[request.request_id]
            assert result.status == "done" and result.cached is False
            assert (result.asset_id, result.object_id, result.mesh_path, result.mesh_scale_to_m) == (
                request.asset_id, request.object_id, request.mesh_path, request.mesh_scale_to_m)
            assert result.gripper_id == "franka_panda" and result.gripper_metadata == {"control_points": [[0, 0, .1]]}
            assert result.bundle_path == request.asset_id + "/bundle.json"
            assert result.grasps.shape == (2, 4, 4) and result.grasps.dtype == np.float64
            assert result.grasps.flags.owndata and result.scores.flags.owndata
            np.testing.assert_array_equal(result.scores, [.9, .8])
            np.testing.assert_array_equal(result.grasps[:, 0, 3], [.01, .02])
        cached_a = replace(a, request_id="cached-a")
        receiver.submit_grasp(cached_a)
        assert terminal(cached_a.request_id)[cached_a.request_id].cached is True
        events.clear()
        receiver.submit_grasp(cached_a)  # Re-publish last request result, no extra prediction.
        assert terminal(cached_a.request_id)[cached_a.request_id].cached is True
        bad = replace(a, request_id="bad", asset_id="fail")
        receiver.submit_grasp(bad)
        failed = terminal(bad.request_id)[bad.request_id]
        assert failed.status == "error" and failed.error == "RuntimeError: injected generation failure"
        assert failed.grasps is None and failed.mesh_path == bad.mesh_path and failed.mesh_scale_to_m == .001
        after = replace(b, request_id="after-error")
        receiver.submit_grasp(after)
        assert terminal(after.request_id)[after.request_id].status == "done"
        assert receiver.snapshot()["received_count"] == 0  # Mesh-only: no camera was opened or published.
    finally:
        receiver.stop()
        receiver.join(timeout=3)
        if child is not None:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
            try:
                output, error = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                output, error = child.communicate()
                pytest.fail("Grasp worker did not stop")
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert child.returncode == 0, f"{output}\n{error}"
    records = [json.loads(line) for line in output.splitlines()]
    assert records[-1] == {"event": "summary", "starts": 1, "calls": 5, "closes": 1, "processed": 5}
