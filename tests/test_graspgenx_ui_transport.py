"""Real Qt/iceoryx2/independent grasp worker; only grasp inference is synthetic."""
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
from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult, PoseResultPublisher, services
from grasp_workbench.ui.camera_window import CameraWindow

ROOT = Path(__file__).resolve().parents[1]
WORKER = """
import json,sys
for name in ('torch','graspgenx','pyzed','pyrealsense2'):
    sys.modules[name]=None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.models.graspgenx import GraspPrediction
from grasp_workbench.workers.worker_graspgenx import GraspGenXWorker

class Predictor:
    def __init__(self,*args,**kwargs):self.calls=[]
    def start(self):pass
    def predict(self,request):
        self.calls.append({'slot':request.object_id,'scale':request.mesh_scale_to_m,
                           'regenerate':request.regenerate,'mesh':request.mesh_path})
        poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[.01,.03,.07]
        return GraspPrediction(poses,np.array([.9,.3,.6]),'synthetic-bundle.json',{})
    def close(self):print(json.dumps({'event':'grasp-model-summary','calls':self.calls}),flush=True)

GraspGenXWorker(WorkerContext('fake-graspgenx'),service=sys.argv[1],predictor_factory=Predictor).run()
"""


def test_two_mesh_buttons_generate_independently_then_follow_fp_without_reinference(tmp_path):
    pytest.importorskip("iceoryx2")
    pytest.importorskip("trimesh")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    service = "grasp-ui-native/" + uuid4().hex
    camera = CameraPublisher(service)
    pose_publisher = PoseResultPublisher(services(service)[1])
    window = CameraWindow(service, foundationpose=True, graspgenx=True, asset_dir=tmp_path / "assets")
    frame = CameraFrame(np.full((48, 64, 3), 40, np.uint8), np.ones((48, 64), np.float32),
                        {"camera_id": "example", "sequence": 1,
                         "intrinsics": {"fx": 60., "fy": 60., "cx": 32., "cy": 24.}})
    paths = {}
    for key in ("A", "B"):
        paths[key] = tmp_path / (key + ".obj")
        paths[key].write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    log = tmp_path / "worker.log"
    worker = None

    def wait_for(predicate, pose=None):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            camera.publish(frame)
            pose_publisher.update_connections()
            if pose is not None:
                pose_publisher.publish(pose)
            app.processEvents()
            window.poll_once()
            assert window.receiver.snapshot()["error"] is None
            if worker is not None:
                assert worker.poll() is None, log.read_text()
            if predicate():
                return
            time.sleep(.005)
        pytest.fail("Timed out waiting for the native grasp UI pipeline")

    try:
        with log.open("w") as stream:
            worker = subprocess.Popen([sys.executable, "-B", "-u", "-c", WORKER, service], cwd=ROOT,
                                      env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=stream,
                                      stderr=subprocess.STDOUT)
            wait_for(lambda: window.receiver.snapshot()["ready"] and '"event": "ready"' in log.read_text())
            for key, scale in (("A", .1), ("B", .2)):
                window.pose_mesh_inputs[key].setText(str(paths[key]))
                window.pose_scale_inputs[key].setValue(scale)
                window.grasp_buttons[(key, False)].click()
            wait_for(lambda: set(window._grasp_results) == {"A", "B"})
            assert not window._pose_targets  # Generation does not need FP or SAM3.
            for key in ("A", "B"):
                assert window.grasp_inputs[key].count() == 3
                np.testing.assert_allclose(window._grasp_results[key].scores, [.9, .3, .6])
                assert window._grasp_previews[key][0] is None  # Descriptor-free axes remain available.
            window.grasp_inputs["A"].setCurrentIndex(2)
            np.testing.assert_allclose(window._selected_grasp("A")[:3, 3], [.07, 0, 0])
            for key in ("A", "B"):
                window.start_pose(key, frame=frame, mask=np.ones((48, 64), bool), prompt_revision="example")
            poses = []
            for key, x in (("A", .1), ("B", -.1)):
                pose = np.eye(4)
                pose[:3, 3] = [x, .02, .8]
                target = window._pose_targets[key]
                poses.append(ObjectPose(key, target.session_id, target.mesh_path, "tracking", pose))
            result = PoseResult(frame, tuple(poses))
            wait_for(lambda: set(window._pose_objects) == {"A", "B"}, pose=result)
            assert set(window._grasp_artists) == {"A", "B"}
            np.testing.assert_allclose((poses[0].camera_from_object @ window._selected_grasp("A"))[:3, 3],
                                       [.17, .02, .8])
            frame = CameraFrame(np.full_like(frame.rgb, 80), frame.depth_m, {**frame.metadata, "sequence": 2})
            wait_for(lambda: window._latest_frame.metadata["sequence"] == 2)
            assert "FP pose sequence=1" in window.metadata_label.text()
            assert "sequence=2" in window.metadata_label.text()
            window.grasp_buttons[("A", True)].click()
            wait_for(lambda: "A" in window._grasp_results)
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
        pose_publisher.close()
        camera.close()
    assert worker.returncode == 0, log.read_text()
    records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    calls = next(item["calls"] for item in records if item["event"] == "grasp-model-summary")
    assert calls == [{"slot": key, "scale": scale, "regenerate": regenerate, "mesh": str(paths[key])}
                     for key, scale, regenerate in (("A", .1, False), ("B", .2, False), ("A", .1, True))]
