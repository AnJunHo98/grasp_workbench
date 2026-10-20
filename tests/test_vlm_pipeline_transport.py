"""One VLM button through real native worker processes, with model-only fakes."""
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
from grasp_workbench.ipc.sam3d import MeshResult
from grasp_workbench.ui.camera_window import CameraWindow

ROOT = Path(__file__).resolve().parents[1]
_COMMON = """
import json, sys
from pathlib import Path
for name in ('torch', 'PIL', 'openai', 'pyzed', 'pyrealsense2', 'estimater', 'sam3d_objects'):
    sys.modules[name] = None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext

def check_source(frame):
    assert frame.metadata['sequence'] == 37
    np.testing.assert_array_equal(frame.rgb,np.arange(4608,dtype=np.uint8).reshape(32,48,3))
    np.testing.assert_array_equal(frame.depth_m,np.full((32,48),.75,np.float32))

def expected_mask(key):
    mask = np.zeros((32,48),bool)
    x = 4 if key == 'A' else 32
    mask[8:16,x:x+8] = True
    return mask

def camera_pose(key):
    pose = np.eye(4)
    pose[:3,3] = ((8 if key == 'A' else 36)-24)*.75/40, (12-16)*.75/40, .75
    return pose
"""
_VLM_WORKER = _COMMON + """
from dataclasses import fields
from grasp_workbench.ipc.vlm import VlmObject, VlmRequest
from grasp_workbench.workers.worker_vlm import VlmWorker

class Predictor:
    def __init__(self, **kwargs):
        assert kwargs['api_key_file'] is None
        self.calls = 0
    def start(self):
        pass
    def predict(self, request):
        check_source(request.frame)
        assert request.instruction == '테이프와 새 공'
        assert {field.name for field in fields(VlmRequest)} == {'request_id','frame','instruction','assets','policies'}
        assert request.policies == ()
        assert all(set(asset) == {'asset_id','name','prompt'} for asset in request.assets)
        assert any(asset['asset_id'] == 'known-tape' for asset in request.assets)
        self.calls += 1
        return (VlmObject('테이프','brown packing tape roll',(4,8,12,16),'known-tape'),
                VlmObject('새 공','orange ball',(32,8,40,16),None))
    def close(self):
        print(json.dumps({'event':'vlm-model-summary','calls':self.calls}),flush=True)

VlmWorker(WorkerContext('fake-vlm'), service=sys.argv[1], predictor_factory=Predictor).run()
"""
_SAM3_WORKER = _COMMON + """
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result
from grasp_workbench.workers.worker_sam3 import Sam3Worker

class Predictor:
    def __init__(self, *args, **kwargs):
        self.sequences = []
    def start(self):
        pass
    def predict(self, frame, prompts):
        assert [(item.object_id,item.text) for item in prompts.objects] == [
            ('A','brown packing tape roll'),('B','orange ball')]
        if frame.metadata['sequence'] == 37:
            check_source(frame)
        self.sequences.append(frame.metadata['sequence'])
        a, b = expected_mask('A'), expected_mask('B')
        objects = (Sam3ObjectResult('A',prompts.objects[0].text,np.stack((b,a)),
            np.array([[32,8,40,16],[4,8,12,16]],np.float32),np.array([.99,.1],np.float32)),
            Sam3ObjectResult('B',prompts.objects[1].text,b[None],
            np.array([[32,8,40,16]],np.float32),np.array([.9],np.float32)))
        return Sam3Result(frame,prompts.revision,objects)
    def close(self):
        print(json.dumps({'event':'sam3-model-summary','sequences':self.sequences}),flush=True)

Sam3Worker(WorkerContext('fake-sam3'),'unused-checkpoint',service=sys.argv[1],
    predictor_factory=Predictor).run()
"""
_MESH_WORKER = _COMMON + """
from grasp_workbench.ipc.sam3d import MeshGeometry
from grasp_workbench.models.sam3d import Sam3dPrediction
from grasp_workbench.workers.worker_sam3d import Sam3dWorker

class Predictor:
    def __init__(self, *args, **kwargs):
        self.mask_sizes = []
    def start(self):
        pass
    def predict(self, frame, mask):
        check_source(frame)
        np.testing.assert_array_equal(mask,expected_mask('B'))
        self.mask_sizes.append(int(mask.sum()))
        mesh = MeshGeometry(np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]],np.float32),
            np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.uint32))
        return Sam3dPrediction(mesh,camera_pose('B'))
    def close(self):
        print(json.dumps({'event':'mesh-model-summary','mask_sizes':self.mask_sizes}),flush=True)

Sam3dWorker(WorkerContext('fake-sam3d'),'unused-config','unused-source',sys.argv[2],
    service=sys.argv[1],predictor_factory=Predictor).run()
"""
_POSE_WORKER = _COMMON + """
from grasp_workbench.workers.worker_foundationpose import FoundationPoseWorker

class Predictor:
    def __init__(self, *args, **kwargs):
        self.registered, self.tracked = [], []
    def start(self):
        pass
    def register(self, object_id, frame, mask, mesh_path, scale):
        check_source(frame)
        np.testing.assert_array_equal(mask,expected_mask(object_id))
        assert Path(mesh_path).is_file() and Path(mesh_path).suffix == '.ply'
        assert scale == 1.
        self.registered.append({'object_id':object_id,'sequence':frame.metadata['sequence'],
            'mask_size':int(mask.sum()),'mesh_path':mesh_path,'scale':scale})
        return camera_pose(object_id)
    def track(self, object_id, frame):
        self.tracked.append([object_id,frame.metadata['sequence']])
        return camera_pose(object_id)
    def remove(self, object_id):
        pass
    def close(self):
        print(json.dumps({'event':'pose-model-summary','registered':self.registered,
            'tracked':self.tracked}),flush=True)

FoundationPoseWorker(WorkerContext('fake-foundationpose'),'unused-source',service=sys.argv[1],
    predictor_factory=Predictor).run()
"""


def test_vlm_button_routes_known_and_generated_assets_through_native_workers(tmp_path):
    pytest.importorskip("iceoryx2")
    pytest.importorskip("trimesh")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = tmp_path / "known.ply"
    source.write_text("""ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
0.1 0 0
0 0.1 0
3 0 1 2
""", encoding="utf-8")
    asset_dir = tmp_path / "assets"
    known = AssetStore(asset_dir).save(
        MeshResult("saved", "known-tape", "A", "saved", "done", mesh_path=str(source),
                   mesh_scale_to_m=1., length_unit="metre"), "테이프", "brown packing tape roll")
    service = f"vlm-full-ui/{uuid4().hex}"
    camera = CameraPublisher(service)
    window = CameraWindow(service, vlm=True, sam3d=True, foundationpose=True, asset_dir=asset_dir)
    window.timer.stop()
    workers, streams, logs = {}, [], {}
    frame = CameraFrame(np.arange(4608, dtype=np.uint8).reshape(32, 48, 3),
                        np.full((32, 48), .75, np.float32), {
                            "sequence": 37, "camera_id": "example-camera", "stream_id": "example-stream",
                            "intrinsics": {"fx": 40., "fy": 40., "cx": 24., "cy": 16.},
                        })

    def wait_for(predicate, *, stream_frame=None):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if stream_frame is not None:
                camera.publish(stream_frame)
            app.processEvents()
            window.poll_once()
            assert window.receiver.snapshot()["error"] is None
            for name, worker in workers.items():
                assert worker.poll() is None, logs[name].read_text()
            value = predicate()
            if value:
                return value
            time.sleep(.01)
        pytest.fail("Timed out waiting for VLM → SAM3 → existing/generated mesh → FoundationPose")

    try:
        wait_for(lambda: window.receiver.snapshot()["ready"])
        for name, script, args in (
            ("vlm", _VLM_WORKER, []), ("sam3", _SAM3_WORKER, []),
            ("sam3d", _MESH_WORKER, [str(tmp_path / "generated")]),
            ("foundationpose", _POSE_WORKER, []),
        ):
            logs[name] = tmp_path / (name + ".log")
            stream = logs[name].open("w")
            streams.append(stream)
            workers[name] = subprocess.Popen(
                [sys.executable, "-B", "-u", "-c", script, service, *args], cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=stream, stderr=subprocess.STDOUT)
        wait_for(lambda: all('"event": "ready"' in log.read_text() for log in logs.values()))
        wait_for(lambda: window.vlm_button.isEnabled(), stream_frame=frame)
        window.vlm_instruction.setText("테이프와 새 공")
        window.vlm_button.click()  # The only pipeline control: no manual Apply/candidate/mesh/FP Start.
        wait_for(lambda: set(window._pose_targets) == {"A", "B"}
                 and set(window._pose_mesh_artists) == {"A", "B"}, stream_frame=frame)
        assert window.prompt_inputs["A"].text() == "brown packing tape roll"
        assert window.prompt_inputs["B"].text() == "orange ball"
        assert window._pose_targets["A"].mesh_path == known["mesh_path"]
        assert set(window._mesh_results) == {"B"}
        generated = window._mesh_results["B"]
        for key, x in (("A", 4), ("B", 32)):
            target = window._pose_targets[key]
            expected = np.zeros((32, 48), bool)
            expected[8:16, x:x + 8] = True
            assert target.frame.metadata["sequence"] == 37 and target.mesh_scale_to_m == 1.
            np.testing.assert_array_equal(target.frame.rgb, frame.rgb)
            np.testing.assert_array_equal(target.frame.depth_m, frame.depth_m)
            np.testing.assert_array_equal(target.mask, expected)
        assert generated.length_unit == "metre" and generated.mesh_scale_to_m == 1.
        saved = AssetStore(asset_dir).get(generated.asset_id)
        assert saved["name"] == "새 공" and saved["prompt"] == "orange ball"
        assert Path(saved["mesh_path"]).read_bytes() == Path(generated.mesh_path).read_bytes()
        assert (asset_dir / generated.asset_id / "manifest.json").is_file()
        newer = CameraFrame(np.full_like(frame.rgb, 221), frame.depth_m, {**frame.metadata, "sequence": 38})
        wait_for(lambda: window._pose_result is not None
                 and window._pose_result.frame.metadata["sequence"] == 38
                 and set(window._pose_objects) == {"A", "B"}
                 and window._sam3_result.frame.metadata["sequence"] == 38, stream_frame=newer)
        assert all(item.status == "tracking" for item in window._pose_objects.values())
        assert len(window.axes.collections) == 2 and len(window.axes.lines) == 6
        assert window.image_view.image.pixelColor(0, 31).getRgb()[:3] == (221, 221, 221)
        assert "camera=example-camera · sequence=38 ·" in window.metadata_label.text()
        assert "실시간 카메라" in window.metadata_label.text() and "FP pose sequence=38" in window.metadata_label.text()
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
        camera.close()
    assert not window.receiver.is_alive()
    summaries = {}
    for name, worker in workers.items():
        assert worker.returncode == 0, logs[name].read_text()
        records = [json.loads(line) for line in logs[name].read_text().splitlines() if line.startswith("{")]
        summaries[name] = next(record for record in records if record["event"].endswith("model-summary"))
    assert summaries["vlm"]["calls"] == 1
    assert 37 in summaries["sam3"]["sequences"] and 38 in summaries["sam3"]["sequences"]
    assert summaries["sam3d"]["mask_sizes"] == [64]
    registered = summaries["foundationpose"]["registered"]
    assert {item["object_id"] for item in registered} == {"A", "B"} and len(registered) == 2
    assert all(item["sequence"] == 37 and item["mask_size"] == 64 for item in registered)
    assert ["A", 38] in summaries["foundationpose"]["tracked"]
    assert ["B", 38] in summaries["foundationpose"]["tracked"]
