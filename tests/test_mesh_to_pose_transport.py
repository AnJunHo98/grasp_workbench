"""New-mesh UI branch across actual SAM3D/FP processes; fake model backends."""
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

from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Publisher, Sam3Result, services
from grasp_workbench.ui.camera_window import CameraWindow

ROOT = Path(__file__).resolve().parents[1]
_MESH_WORKER = """
import json, sys
for name in ('torch', 'pyzed', 'pyrealsense2', 'sam3d_objects'):
    sys.modules[name] = None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.sam3d import MeshGeometry
from grasp_workbench.models.sam3d import Sam3dPrediction
from grasp_workbench.workers.worker_sam3d import Sam3dWorker

class Predictor:
    def __init__(self):
        self.calls = []
    def start(self):
        pass
    def predict(self, frame, mask):
        assert frame.metadata['sequence'] == 37
        np.testing.assert_array_equal(frame.rgb,np.arange(3072,dtype=np.uint8).reshape(32,32,3))
        np.testing.assert_array_equal(frame.depth_m,np.full((32,32),.75,np.float32))
        self.calls.append(int(mask.sum()))
        mesh = MeshGeometry(np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]],np.float32),
            np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.uint32))
        pose = np.eye(4)
        pose[2,3] = .75
        return Sam3dPrediction(mesh, pose)
    def close(self):
        print(json.dumps({'event':'mesh-model-summary','mask_sizes':self.calls}),flush=True)

predictor = Predictor()
Sam3dWorker(WorkerContext('fake-sam3d'),'unused-config','unused-source',sys.argv[2],
    service=sys.argv[1],predictor_factory=lambda *a,**k:predictor).run()
"""
_POSE_WORKER = """
import json, sys
from pathlib import Path
for name in ('torch', 'pyzed', 'pyrealsense2', 'estimater'):
    sys.modules[name] = None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.workers.worker_foundationpose import FoundationPoseWorker

class Predictor:
    def __init__(self):
        self.registered, self.tracked = [], []
    def start(self):
        pass
    def register(self, object_id, frame, mask, mesh_path, scale):
        assert frame.metadata['sequence'] == 37
        np.testing.assert_array_equal(frame.rgb,np.arange(3072,dtype=np.uint8).reshape(32,32,3))
        np.testing.assert_array_equal(frame.depth_m,np.full((32,32),.75,np.float32))
        expected = np.zeros((32,32),bool)
        expected[4:12,4:12] = True
        if object_id == 'B':
            expected[12:20,12:20] = True
        np.testing.assert_array_equal(mask,expected)
        assert Path(mesh_path).is_file() and Path(mesh_path).suffix == '.ply'
        assert Path(mesh_path).read_bytes().startswith(b'ply\\nformat binary_little_endian')
        self.registered.append({'object_id':object_id,'mesh_path':mesh_path,
            'scale':scale,'mask_size':int(mask.sum())})
        pose = np.eye(4)
        pose[2,3] = .75
        return pose
    def track(self, object_id, frame):
        self.tracked.append([object_id,frame.metadata['sequence']])
        pose = np.eye(4)
        pose[0,3] = .01 if object_id == 'A' else -.01
        pose[2,3] = .75
        return pose
    def remove(self, object_id):
        pass
    def close(self):
        print(json.dumps({'event':'pose-model-summary','registered':self.registered,
            'tracked':self.tracked}),flush=True)

predictor = Predictor()
FoundationPoseWorker(WorkerContext('fake-foundationpose'),'unused-source',
    service=sys.argv[1],predictor_factory=lambda *a,**k:predictor).run()
"""


def test_a_b_new_mesh_buttons_generate_then_register_and_track(tmp_path):
    pytest.importorskip("iceoryx2")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    service = f"mesh-to-pose-ui/{uuid4().hex}"
    camera = CameraPublisher(service)
    segmentation = Sam3Publisher(services(service)[1])
    window = CameraWindow(service, sam3d=True, foundationpose=True)
    workers, streams, logs = [], [], []
    frame = CameraFrame(
        np.arange(3072, dtype=np.uint8).reshape(32, 32, 3),
        np.full((32, 32), .75, np.float32),
        {"sequence": 37, "camera_id": "example-camera", "stream_id": "example-stream",
         "intrinsics": {"fx": 20., "fy": 20., "cx": 16., "cy": 16.}},
    )

    def wait_for(predicate, *, stream_frame=None):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if stream_frame is not None:
                camera.publish(stream_frame)
            app.processEvents()
            window.poll_once()
            assert window.receiver.snapshot()["error"] is None
            for worker, log in zip(workers, logs):
                assert worker.poll() is None, log.read_text()
            value = predicate()
            if value:
                return value
            time.sleep(.005)
        pytest.fail("Timed out waiting for the new-mesh UI/native-worker pipeline")

    try:
        wait_for(lambda: window.receiver.snapshot()["ready"])
        for name, script, args in (
            ("sam3d", _MESH_WORKER, [str(tmp_path / "meshes")]),
            ("foundationpose", _POSE_WORKER, []),
        ):
            log = tmp_path / (name + ".log")
            stream = log.open("w")
            logs.append(log)
            streams.append(stream)
            workers.append(subprocess.Popen(
                [sys.executable, "-B", "-u", "-c", script, service, *args], cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=stream,
                stderr=subprocess.STDOUT,
            ))
        wait_for(lambda: all('"event": "ready"' in log.read_text() for log in logs))
        window.prompt_inputs["A"].setText("brown packing tape roll")
        window.prompt_inputs["B"].setText("yellow tuna can")
        window.apply_prompts()
        masks = np.zeros((2, 32, 32), bool)
        masks[:, 4:12, 4:12] = True
        masks[1, 12:20, 12:20] = True
        objects = tuple(Sam3ObjectResult(
            key, window.prompt_inputs[key].text(), masks[index:index + 1],
            np.array([[4, 4, 20 if index else 12, 20 if index else 12]], np.float32),
            np.array([.99], np.float32),
        ) for index, key in enumerate(("A", "B")))
        camera.publish(frame)
        segmentation.publish(Sam3Result(frame, window._prompts.revision, objects))
        wait_for(lambda: len(window._mesh_bindings) == 2)
        for key, scale in (("A", .08), ("B", .12)):
            assert window.pose_mesh_inputs[key].text() == ""
            window.pose_scale_inputs[key].setValue(scale)
            assert window.mesh_pose_buttons[key].isEnabled()
            window.mesh_pose_buttons[key].click()
        assert window._pose_targets == {}  # No FP register before generation completes.
        wait_for(lambda: set(window._pose_targets) == {"A", "B"})
        paths = {key: window._mesh_results[key].mesh_path for key in ("A", "B")}
        assert paths["A"] != paths["B"]
        for key in ("A", "B"):
            target = window._pose_targets[key]
            assert target.mesh_path == paths[key] == window.pose_mesh_inputs[key].text()
            assert target.mesh_scale_to_m == window.pose_scale_inputs[key].value() == 1.0
            assert target.prompt_revision == window._prompts.revision
            assert window._mesh_results[key].length_unit == "metre"
            np.testing.assert_allclose(window._mesh_results[key].camera_from_object[:3,3], [0,0,.75])
            assert target.frame.metadata["sequence"] == 37
            np.testing.assert_array_equal(target.mask, masks[("A", "B").index(key)])
        tracking_frame = CameraFrame(frame.rgb, frame.depth_m, {**frame.metadata, "sequence": 38})
        wait_for(lambda: window._pose_result is not None
                 and window._pose_result.frame.metadata["sequence"] == 38
                 and {item.object_id for item in window._pose_result.objects} == {"A", "B"},
                 stream_frame=tracking_frame)
        for key, x in (("A", .01), ("B", -.01)):
            assert window._pose_objects[key].status == "tracking"
            pose = window._pose_objects[key].camera_from_object
            np.testing.assert_allclose(pose[:3, 3], [x, 0, .75])
        assert "실시간 카메라" in window.metadata_label.text()
        assert "FP pose sequence=38" in window.metadata_label.text()
        assert "SAM3 overlay sequence=37" in window.metadata_label.text()
        assert window.image_view.image is not None
    finally:
        window.receiver.stop()
        window.receiver.join(timeout=3)
        window.close()
        app.processEvents()
        for worker in workers:
            if worker.poll() is None:
                worker.send_signal(signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        for stream in streams:
            stream.close()
        segmentation.close()
        camera.close()
    assert not window.receiver.is_alive()
    for worker, log in zip(workers, logs):
        assert worker.returncode == 0, log.read_text()
    mesh_records = [json.loads(line) for line in logs[0].read_text().splitlines() if line.startswith("{")]
    assert next(item for item in mesh_records if item["event"] == "mesh-model-summary")["mask_sizes"] == [64, 128]
    pose_records = [json.loads(line) for line in logs[1].read_text().splitlines() if line.startswith("{")]
    summary = next(item for item in pose_records if item["event"] == "pose-model-summary")
    assert summary["registered"] == [
        {"object_id": "A", "mesh_path": paths["A"], "scale": 1.0, "mask_size": 64},
        {"object_id": "B", "mesh_path": paths["B"], "scale": 1.0, "mask_size": 128},
    ]
    assert ["A", 38] in summary["tracked"] and ["B", 38] in summary["tracked"]
