"""Synthetic UI transport checks: real iceoryx2, no camera SDK or device access."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

_PUBLISHER = """
import json, sys, time
import numpy as np
from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher
publisher = CameraPublisher(sys.argv[1])
rgb = np.zeros((3, 4, 3), dtype=np.uint8)
depth = np.zeros((3, 4), dtype=np.float32)
metadata = {'camera_id':'synthetic-ui', 'stream_id':sys.argv[1], 'sequence':0,
            'capture_time_ns':1000, 'timestamp_domain':'synthetic_device',
            'received_monotonic_ns':0,
            'intrinsics':{'fx':100., 'fy':101., 'cx':2., 'cy':1.5}}
frame = CameraFrame(rgb, depth, metadata)
try:
    deadline = time.monotonic() + 10
    while publisher.publish(frame) == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError('No UI subscriber connected')
        time.sleep(.02)
    for sequence in range(24):
        rgb[:] = (sequence, 40, 200)
        depth[:] = np.float32(.5 + sequence / 100)
        metadata.update(sequence=sequence, capture_time_ns=1000 + sequence * 1000,
                        received_monotonic_ns=time.monotonic_ns())
        last_send = time.monotonic()
        publisher.publish(frame)
        time.sleep(.02)
    print(json.dumps({'last_send_monotonic_s':last_send}), flush=True)
    time.sleep(.15)
finally:
    publisher.close()
"""


def _environment():
    return {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}


def _wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    pytest.fail("Timed out waiting for the native UI receiver")


def test_native_receiver_latest_slot_owned_arrays_and_stop():
    pytest.importorskip("iceoryx2")
    from grasp_workbench.ui.receiver import LatestCameraReceiver

    service = f"ui-receiver-test/{uuid4().hex}"
    receiver = LatestCameraReceiver(service)
    receiver.start()
    result = None
    try:
        _wait_for(lambda: receiver.snapshot()["ready"])
        assert receiver.take_latest() is None
        publisher = subprocess.run(
            [sys.executable, "-c", _PUBLISHER, service], cwd=ROOT,
            env=_environment(), capture_output=True, text=True, timeout=15,
        )
        assert publisher.returncode == 0, publisher.stderr
        last_send = json.loads(publisher.stdout)["last_send_monotonic_s"]
        _wait_for(lambda: (
            receiver.snapshot()["last_received_monotonic_s"] or 0
        ) >= last_send)
        stats = receiver.snapshot()
        assert stats["ready"] and stats["error"] is None
        assert stats["received_count"] >= 3
        assert stats["elapsed_s"] > 0
        assert last_send <= stats["last_received_monotonic_s"] <= time.monotonic()
        result = receiver.take_latest()
        assert result is not None and result.metadata["sequence"] == 23
        assert receiver.take_latest() is None, "UI input must be one latest slot, not a frame queue"
        assert result.metadata["camera_id"] == "synthetic-ui"
        assert result.metadata["stream_id"] == service
        assert result.metadata["capture_time_ns"] == 24_000
        assert result.metadata["timestamp_domain"] == "synthetic_device"
        assert result.metadata["intrinsics"]["fx"] == 100.0
        assert result.rgb.flags.owndata and result.depth_m.flags.owndata
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert not receiver.is_alive()
    assert receiver.snapshot()["ready"] is False
    # Publisher and subscriber are closed; no borrowed native memory may remain.
    np.testing.assert_array_equal(result.rgb, np.full((3, 4, 3), (23, 40, 200), dtype=np.uint8))
    np.testing.assert_array_equal(result.depth_m, np.full((3, 4), .73, dtype=np.float32))


def test_offscreen_ui_worker_receives_native_synthetic_frames_without_camera_sdk(tmp_path):
    pytest.importorskip("iceoryx2")
    if any(importlib.util.find_spec(package) is None for package in ("PyQt6", "matplotlib")):
        pytest.skip("Install the optional UI packages to run the offscreen worker smoke")
    service = f"ui-worker-test/{uuid4().hex}"
    runtime = tmp_path / "qt-runtime"
    runtime.mkdir(mode=0o700)
    environment = {**_environment(), "QT_QPA_PLATFORM": "offscreen",
                   "XDG_RUNTIME_DIR": str(runtime)}
    script = """
import runpy, sys
class NoCameraSdk:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyzed', 'pyrealsense2'}:
            raise AssertionError('UI worker must not import a camera SDK')
sys.meta_path.insert(0, NoCameraSdk())
sys.argv = ['worker_camera_ui', '--service', sys.argv[1], '--frames', '3', '--timeout', '5']
runpy.run_module('grasp_workbench.workers.worker_camera_ui', run_name='__main__')
"""
    ui = subprocess.Popen([sys.executable, "-c", script, service], cwd=ROOT, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        publisher = subprocess.run(
            [sys.executable, "-c", _PUBLISHER, service], cwd=ROOT,
            env=environment, capture_output=True, text=True, timeout=15,
        )
        output, error = ui.communicate(timeout=10)
        assert ui.returncode == 0, f"stdout:\n{output}\nstderr:\n{error}"
        assert publisher.returncode == 0, publisher.stderr
    finally:
        if ui.poll() is None:
            ui.kill()
            ui.communicate()
