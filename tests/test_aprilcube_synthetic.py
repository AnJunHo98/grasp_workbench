"""Real pinned SDK, synthetic printed markers, native worker and pose/overlay IPC."""
import importlib
import json
import os
from pathlib import Path
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
from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher
from grasp_workbench.ipc.foundationpose import PoseResultSubscriber, services as pose_services

ROOT = Path(__file__).resolve().parents[1]


def _scene(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    if not (ROOT / "third_party/aprilcube/src/aprilcube").is_dir():
        pytest.skip("Initialize the AprilCube submodule for the real SDK check")
    monkeypatch.syspath_prepend(str(ROOT / "third_party/aprilcube/src"))
    sdk = importlib.import_module("aprilcube.detect")
    matrix = np.array([[1100., 0, 640.], [0, 1100., 360.], [0, 0, 1.]])
    image = np.full((720, 1280), 255, np.uint8)
    rvec, truth = np.array([.12, .22, .04]), {}
    square = np.array([[0, 0], [179, 0], [179, 179], [0, 179]], np.float32)
    for slot, x in (("A", -120.), ("B", 120.)):
        config, faces = sdk.load_cube_config(str(ROOT / "examples/aprilcube" / f"cube_{slot.lower()}.json"))
        corners = sdk.build_tag_corner_map(config)
        dictionary = cv2.aruco.getPredefinedDictionary(config.dict_id)
        translation = np.array([x, 0, 700.])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], translation / 1000
        truth[slot] = pose
        for marker_id in faces["-Z"]:
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 180)
            pixels = cv2.projectPoints(corners[marker_id], rvec, translation, matrix, np.zeros(5))[0]
            transform = cv2.getPerspectiveTransform(square, pixels.reshape(4, 2).astype(np.float32))
            warped = cv2.warpPerspective(marker, transform, (1280, 720),
                                         flags=cv2.INTER_NEAREST, borderValue=255)
            image = np.minimum(image, warped)
    frame = CameraFrame(np.repeat(image[:, :, None], 3, axis=2), np.full(image.shape, .7, np.float32), {
        "camera_id": "synthetic-aprilcube", "stream_id": "synthetic", "sequence": 7,
        "received_monotonic_ns": time.monotonic_ns(),
        "intrinsics": {"fx": 1100., "fy": 1100., "cx": 640., "cy": 360.,
                       "distortion_model": "zed_rectified", "coefficients": [.3, .2, .1, 0, 0]},
    })
    return frame, truth


def _wait(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    pytest.fail("Timed out waiting for the real AprilCube worker")


@pytest.mark.parametrize("fast", [False, True], ids=["standard", "fast"])
def test_real_sdk_a_b_poses_and_mesh_layers_over_native_ipc(tmp_path, monkeypatch, fast):
    pytest.importorskip("iceoryx2")
    pytest.importorskip("trimesh")
    frame, truth = _scene(monkeypatch)
    original_rgb, original_depth = frame.rgb.copy(), frame.depth_m.copy()
    mesh = tmp_path / "virtual_rectangle.obj"
    mesh.write_text("v -30 -20 0\nv 30 -20 0\nv 30 20 0\nv -30 20 0\nf 1 2 3\nf 1 3 4\n")
    original_mesh = mesh.read_bytes()
    service = "aprilcube-real/" + uuid4().hex
    camera, controls = CameraPublisher(service), AprilCubeControlPublisher(services(service)[0])
    poses = PoseResultSubscriber(pose_services(service)[1])
    overlays = AprilCubeOverlaySubscriber(services(service)[1])
    targets = []
    for slot in ("A", "B"):
        mapping = np.eye(4)
        mapping[:3, 3] = (.01, -.02, -.07)
        targets.append(AprilCubeTarget(slot, "real-" + slot, str(mesh), .001, mapping))
    workers = []
    log_path = tmp_path / "worker.log"
    overlay_log_path = tmp_path / "overlay.log"
    try:
        controls.publish(AprilCubeControl(tuple(targets)))
        with log_path.open("w") as log, overlay_log_path.open("w") as overlay_log:
            workers.append(subprocess.Popen([
                sys.executable, "-B", "-u", "-m", "grasp_workbench.workers.worker_aprilcube_overlay",
                "--service", service, *(["--log-timing"] if fast else []),
            ], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=overlay_log, stderr=overlay_log))
            workers.append(subprocess.Popen([
                sys.executable, "-B", "-u", "-m", "grasp_workbench.workers.worker_aprilcube",
                "--service", service, "--source-dir", str(ROOT / "third_party/aprilcube"),
                "--config-a", str(ROOT / "examples/aprilcube/cube_a.json"),
                "--config-b", str(ROOT / "examples/aprilcube/cube_b.json"),
                "--no-filter",
                *(["--log-timing"] if fast else ["--no-fast"]),
            ], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=log, stderr=log))
            _wait(lambda: all('"event": "ready"' in path.read_text()
                              for path in (log_path, overlay_log_path)))
            controls.update_connections()
            assert camera.publish(frame) == 1
            result, overlay = _wait(poses.receive_latest), _wait(overlays.receive_latest)
            for worker in workers:
                worker.terminate()
                worker.wait(timeout=5)
                assert worker.returncode == 0, log_path.read_text() + overlay_log_path.read_text()
        records = [json.loads(line) for line in log_path.read_text().splitlines()
                   if line.startswith("{")]
        timings = [record for record in records if record["event"] == "timing"]
        assert len(timings) == (1 if fast else 0)
        if fast:
            assert timings[0]["worker"] == "aprilcube" and timings[0]["sequence"] == 7
            assert timings[0]["fast"] is True
            assert all(value >= 0 for value in timings[0]["timings_ms"].values())
        overlay_records = [json.loads(line) for line in overlay_log_path.read_text().splitlines()
                           if line.startswith("{")]
        overlay_timings = [record for record in overlay_records if record["event"] == "timing"]
        assert len(overlay_timings) == (1 if fast else 0)
        if fast:
            assert overlay_timings[0]["worker"] == "aprilcube_overlay"
            assert overlay_timings[0]["sequence"] == 7
            assert all(value >= 0 for value in overlay_timings[0]["timings_ms"].values())
        assert result.frame.metadata == frame.metadata == overlay.metadata
        np.testing.assert_array_equal(result.frame.rgb, original_rgb)
        np.testing.assert_array_equal(result.frame.depth_m, original_depth)
        assert {item.object_id for item in result.objects} == {"A", "B"}
        for item, target in zip(result.objects, targets):
            assert item.pose_source == "aprilcube" and item.status == "tracking"
            assert item.session_id == target.session_id and item.mesh_path == str(mesh)
            expected = truth[item.object_id] @ target.cube_from_object
            np.testing.assert_allclose(item.camera_from_object[:3, 3], expected[:3, 3], atol=.01)
            np.testing.assert_allclose(item.camera_from_object[:3, :3], expected[:3, :3], atol=.05)
        assert overlay.rgba.shape == (720, 1280, 4)
        assert np.count_nonzero(overlay.rgba[:, :, 3]) > 5000
        assert set(np.unique(overlay.rgba[:, :, 3])) == {0, 166}
        assert not overlay.rgba[0, 0].any()
        np.testing.assert_array_equal(frame.rgb, original_rgb)
        np.testing.assert_array_equal(frame.depth_m, original_depth)
        assert mesh.read_bytes() == original_mesh
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)
        for endpoint in (overlays, poses, controls, camera):
            endpoint.close()
