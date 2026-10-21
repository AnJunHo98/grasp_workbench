"""Actual independent worker and native iceoryx2, with a fake AprilCube SDK boundary."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.aprilcube import (
    AprilCubeControl, AprilCubeControlPublisher, AprilCubeOverlaySubscriber,
    AprilCubeTarget, services,
)
from grasp_workbench.ipc.camera import CameraPublisher
from grasp_workbench.ipc.foundationpose import PoseResultSubscriber, services as pose_services
from test_aprilcube_model import frame

ROOT = Path(__file__).resolve().parents[1]
_WORKER = """
import sys
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.workers.worker_aprilcube import AprilCubeWorker

class Predictor:
    diagnostics = {}
    def start(self):
        pass
    def predict(self, frame):
        pose = np.eye(4)
        pose[:3,3] = [.1,.2,.7]
        return {'A':pose,'B':None}
    def close(self):
        pass

AprilCubeWorker(WorkerContext('fake-aprilcube'),'unused','a','b',service=sys.argv[1],
    predictor_factory=lambda *a,**k:Predictor()).run()
"""
_OVERLAY = """
from pathlib import Path
import sys
import time
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.workers.worker_aprilcube_overlay import AprilCubeOverlayWorker

class Renderer:
    def render(self, frame, targets, poses, alpha):
        assert targets[0].mesh_scale_to_m == .001
        np.testing.assert_allclose(poses['A'][:3,3], [.11,.18,.73])
        print('render-start', flush=True)
        deadline = time.monotonic() + 10
        while not Path(sys.argv[2]).exists():
            assert time.monotonic() < deadline, 'Test did not release renderer'
            time.sleep(.005)
        return np.broadcast_to(np.array([11,22,33,166],np.uint8),
                               (*frame.rgb.shape[:2],4)).copy()

AprilCubeOverlayWorker(WorkerContext('fake-aprilcube-overlay'),service=sys.argv[1],
    renderer_factory=Renderer).run()
"""


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    pytest.fail("Timed out waiting for AprilCube IPC")


def test_process_accepts_retained_bindings_and_publishes_common_pose_plus_rgba(tmp_path):
    pytest.importorskip("iceoryx2")
    service = "aprilcube-test/" + uuid4().hex
    camera, controls = CameraPublisher(service), AprilCubeControlPublisher(services(service)[0])
    results, overlays = PoseResultSubscriber(pose_services(service)[1]), AprilCubeOverlaySubscriber(services(service)[1])
    transform = np.eye(4)
    transform[:3, 3] = [.01, -.02, .03]
    selected = AprilCubeTarget("A", "native-session", "mesh.obj", .001, transform)
    workers = []
    log_path = tmp_path / "worker.log"
    overlay_log_path = tmp_path / "overlay.log"
    release_renderer = tmp_path / "release-renderer"
    try:
        assert controls.publish(AprilCubeControl((selected,))) == 0
        with log_path.open("w") as log, overlay_log_path.open("w") as overlay_log:
            for script, output, extra in ((_OVERLAY, overlay_log, [str(release_renderer)]),
                                          (_WORKER, log, [])):
                workers.append(subprocess.Popen(
                    [sys.executable, "-B", "-u", "-c", script, service, *extra], cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=output,
                    stderr=subprocess.STDOUT, text=True))
            wait_for(lambda: all('"event": "ready"' in path.read_text()
                                 for path in (log_path, overlay_log_path)))
            controls.update_connections()
            source = frame(7)
            assert camera.publish(source) == 1

            def receive_pose():
                assert all(worker.poll() is None for worker in workers), (
                    log_path.read_text() + overlay_log_path.read_text())
                return results.receive_latest()

            result = wait_for(receive_pose)
            wait_for(lambda: "render-start" in overlay_log_path.read_text())
            # The renderer is held inside its own process. Pose publication must
            # continue without waiting for its RGBA output or acknowledgement.
            assert camera.publish(frame(8)) == 1
            second = wait_for(receive_pose)
            assert second.frame.metadata["sequence"] == 8
            assert overlays.receive_latest() is None
            release_renderer.touch()
            layer = wait_for(overlays.receive_latest)
            item, = result.objects
            assert item.object_id == "A" and item.session_id == "native-session"
            assert item.pose_source == "aprilcube" and item.status == "tracking"
            np.testing.assert_allclose(item.camera_from_object[:3, 3], [.11, .18, .73])
            assert result.frame.metadata == source.metadata
            assert layer.metadata in (source.metadata, second.frame.metadata)
            np.testing.assert_array_equal(result.frame.rgb, source.rgb)
            np.testing.assert_array_equal(result.frame.depth_m, source.depth_m)
            np.testing.assert_array_equal(layer.rgba[0, 0], [11, 22, 33, 166])
            # Decoded arrays remain usable after endpoint release.
        for worker in workers:
            worker.send_signal(signal.SIGTERM)
            worker.wait(timeout=5)
            assert worker.returncode == 0, log_path.read_text() + overlay_log_path.read_text()
    finally:
        release_renderer.touch()
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)
        for endpoint in (overlays, results, controls, camera):
            endpoint.close()
    np.testing.assert_array_equal(layer.rgba[0, 0], [11, 22, 33, 166])
