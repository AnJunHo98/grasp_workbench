"""Real IPC/UI receiver with a fake model process; no camera, SDK or GPU."""
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

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.sam3d import MeshRequest
from grasp_workbench.ui.receiver import LatestCameraReceiver

ROOT = Path(__file__).resolve().parents[1]
_WORKER = """
import json, sys
for name in ('torch', 'pyzed', 'pyzed.sl', 'pyrealsense2', 'sam3d_objects'):
    sys.modules[name] = None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.sam3d import MeshGeometry
from grasp_workbench.models.sam3d import Sam3dPrediction
from grasp_workbench.workers.worker_sam3d import Sam3dWorker

class FakePredictor:
    def __init__(self):
        self.calls = self.starts = self.closes = 0
    def start(self):
        self.starts += 1
    def predict(self, frame, mask):
        self.calls += 1
        assert frame.metadata == {'camera_id':'frozen-camera', 'stream_id':'frozen-stream',
            'sequence':37, 'capture_time_ns':1234567, 'timestamp_domain':'device-clock',
            'received_monotonic_ns':987654321, 'intrinsics':{'fx':100., 'cx':2.}}
        np.testing.assert_array_equal(frame.rgb,np.arange(36,dtype=np.uint8).reshape(3,4,3))
        np.testing.assert_array_equal(frame.depth_m,np.full((3,4),.75,dtype=np.float32))
        assert mask.dtype == np.bool_ and mask.shape == (3,4) and mask.any()
        size = float(mask.sum())
        mesh = MeshGeometry(np.array([[0,0,0],[size,0,0],[0,size,0]],dtype=np.float32),
            np.array([[0,1,2]],dtype=np.uint32),
            np.full((3,3),self.calls,dtype=np.uint8))
        pose = np.eye(4)
        pose[:3,3] = [.1,.2,.75]
        return Sam3dPrediction(mesh, pose)
    def close(self):
        self.closes += 1

predictor = FakePredictor()
worker = Sam3dWorker(WorkerContext('fake-sam3d'), 'unused-pipeline.yaml', 'unused-source',
                    sys.argv[2], service=sys.argv[1], predictor_factory=lambda *a,**k: predictor)
try:
    worker.run()
finally:
    assert all(sys.modules[name] is None
               for name in ('torch','pyzed','pyzed.sl','pyrealsense2','sam3d_objects'))
    print(json.dumps({'event':'fake-model-summary', 'calls':predictor.calls,
        'starts':predictor.starts, 'closes':predictor.closes,
        'processed':worker.processed_count}),flush=True)
"""


def _wait_for(predicate, *, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.005)
    pytest.fail("Timed out waiting for the SAM3D transport")


def test_queued_mesh_jobs_late_worker_cache_error_and_clean_shutdown(tmp_path):
    pytest.importorskip("iceoryx2")
    service = f"sam3d-ui-transport/{uuid4().hex}"
    frame = CameraFrame(
        np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        np.full((3, 4), .75, dtype=np.float32),
        {"camera_id": "frozen-camera", "stream_id": "frozen-stream", "sequence": 37,
         "capture_time_ns": 1234567, "timestamp_domain": "device-clock",
         "received_monotonic_ns": 987654321, "intrinsics": {"fx": 100., "cx": 2.}},
    )
    mask_a = np.arange(12).reshape(3, 4) % 2 == 0
    mask_b = np.arange(12).reshape(3, 4) < 4
    a = MeshRequest("job-A", "asset/A:α", "A", "frozen-revision", "red cup", 0, frame, mask_a)
    b = MeshRequest("job-B", "asset/B", "B", "frozen-revision", "blue bowl", 1, frame, mask_b)
    receiver = LatestCameraReceiver(service, sam3d=True)
    receiver.submit_mesh(a)
    receiver.submit_mesh(b)
    receiver.start()
    worker = None
    events = []
    output = error = ""
    try:
        _wait_for(lambda: receiver.snapshot()["ready"])

        def first_job_is_retained():
            with receiver._lock:
                return (receiver._mesh_inflight is not None
                        and receiver._mesh_inflight.request_id == a.request_id
                        and [item.request_id for item in receiver._mesh_queue] == [b.request_id])

        # Request A is already published before the worker/subscriber exists;
        # B must remain queued rather than overwrite A's retained sample.
        _wait_for(first_job_is_retained)
        worker = subprocess.Popen(
            [sys.executable, "-u", "-c", _WORKER, service, str(tmp_path / "mesh-assets")],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        )

        def terminals(*request_ids):
            def collect():
                state = receiver.snapshot()
                assert state["error"] is None, state["error"]
                assert worker.poll() is None, "Fake worker exited before completing requests"
                events.extend(receiver.take_mesh_results())
                matches = {result.request_id: result for result in events
                           if result.status in {"done", "error"} and result.request_id in request_ids}
                return matches if len(matches) == len(request_ids) else None

            return _wait_for(collect)

        completed = terminals(a.request_id, b.request_id)
        assert [item.request_id for item in events if item.status in {"done", "error"}] == ["job-A", "job-B"]
        for request in (a, b):
            result = completed[request.request_id]
            assert result.status == "done", result.error
            assert (result.asset_id, result.object_id, result.revision) == (
                request.asset_id, request.object_id, request.revision)
            assert Path(result.mesh_path).is_file() and Path(result.mesh_path).suffix == ".ply"
            assert result.mesh.vertices.flags.owndata and result.mesh.faces.flags.owndata
            assert result.length_unit == "metre" and result.mesh_scale_to_m == 1.
            np.testing.assert_array_equal(result.camera_from_object[:3, 3], [.1, .2, .75])
        assert completed["job-A"].mesh_path != completed["job-B"].mesh_path

        cached_a = replace(a, request_id="job-A-cache")
        receiver.submit_mesh(cached_a)
        cached = terminals(cached_a.request_id)[cached_a.request_id]
        assert cached.status == "done" and cached.mesh_path == completed["job-A"].mesh_path
        np.testing.assert_array_equal(cached.mesh.vertices, completed["job-A"].mesh.vertices)
        np.testing.assert_array_equal(cached.mesh.faces, completed["job-A"].mesh.faces)
        np.testing.assert_array_equal(cached.mesh.colors, completed["job-A"].mesh.colors)
        np.testing.assert_array_equal(cached.camera_from_object, completed["job-A"].camera_from_object)

        empty = replace(a, request_id="job-empty", asset_id="empty-asset", mask=np.zeros((3, 4), bool))
        receiver.submit_mesh(empty)
        rejected = terminals(empty.request_id)[empty.request_id]
        assert rejected.status == "error" and rejected.mesh is None
        assert "foreground" in rejected.error
        assert worker.poll() is None

        # A terminal error releases the in-flight slot; valid cached work still runs.
        cached_b = replace(b, request_id="job-B-after-error")
        receiver.submit_mesh(cached_b)
        recovered = terminals(cached_b.request_id)[cached_b.request_id]
        assert recovered.status == "done" and recovered.mesh_path == completed["job-B"].mesh_path
        np.testing.assert_array_equal(recovered.mesh.colors, completed["job-B"].mesh.colors)
        assert receiver.snapshot()["received_count"] == 0  # No camera publisher was needed.
    finally:
        receiver.stop()
        receiver.join(timeout=3)
        if worker is not None:
            if worker.poll() is None:
                worker.send_signal(signal.SIGTERM)
            try:
                output, error = worker.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                output, error = worker.communicate()
                pytest.fail(f"Fake model worker did not stop on SIGTERM: {output}\n{error}")
    assert not receiver.is_alive() and receiver.snapshot()["ready"] is False
    assert receiver.snapshot()["error"] is None
    assert worker.returncode == 0, f"stdout:\n{output}\nstderr:\n{error}"
    records = [json.loads(line) for line in output.splitlines()]
    summaries = [item for item in records if item.get("event") == "fake-model-summary"]
    assert summaries == [{"event": "fake-model-summary", "calls": 2, "starts": 1, "closes": 1,
                          "processed": 5}]
