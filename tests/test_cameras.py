"""Fake SDK checks only: no camera devices are opened by this test suite."""
from pathlib import Path
from types import SimpleNamespace as NS
import os
import subprocess
import sys

import numpy as np
import pytest

from grasp_workbench.cameras import realsense, zed


def fake_zed():
    sdk = NS(cameras=[], bgr=np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8),
             depth=np.array([[1.5, np.nan], [np.inf, 0.0]], dtype=np.float32),
             status="SUCCESS", serial=12345)
    sdk.ERROR_CODE = NS(SUCCESS="SUCCESS")
    sdk.RESOLUTION = NS(HD720="HD720")
    sdk.UNIT, sdk.DEPTH_MODE = NS(METER="METER"), NS(PERFORMANCE="PERFORMANCE")
    sdk.VIEW, sdk.MEASURE = NS(LEFT_BGR="LEFT_BGR"), NS(DEPTH="DEPTH")
    sdk.TIME_REFERENCE = NS(IMAGE="IMAGE")
    sdk.RuntimeParameters = lambda: NS()

    class Options:
        def set_from_serial_number(self, serial):
            self.serial = serial

    class Mat:
        def get_data(self):
            return self.data

    class Camera:
        @staticmethod
        def get_device_list():
            return [NS(serial_number=12345, camera_model="ZED-M", camera_state="AVAILABLE")]

        def __init__(self):
            sdk.cameras.append(self)
            self.closed, self.grabs = 0, 0

        def open(self, options):
            self.options = options
            return sdk.status

        def get_camera_information(self):
            calibration = NS(fx=5.0, fy=6.0, cx=0.5, cy=0.75, disto=np.zeros(5))
            return NS(serial_number=sdk.serial, camera_configuration=NS(
                resolution=NS(width=2, height=2), calibration_parameters=NS(left_cam=calibration)))

        def grab(self, runtime):
            self.grabs += 1
            return sdk.status

        def get_timestamp(self, reference):
            assert reference == "IMAGE"
            return NS(get_nanoseconds=lambda: 7000 + self.grabs)

        def retrieve_image(self, mat, view):
            assert view == "LEFT_BGR"
            mat.data = sdk.bgr
            return sdk.status

        def retrieve_measure(self, mat, measure):
            assert measure == "DEPTH"
            mat.data = sdk.depth
            return sdk.status

        def close(self):
            self.closed += 1

    sdk.InitParameters, sdk.Mat, sdk.Camera = Options, Mat, Camera
    return sdk


def fake_realsense():
    sdk = NS(pipelines=[], rgb=np.full((2, 2, 3), (1, 2, 3), dtype=np.uint8),
             depth=np.array([[1000, 0], [2000, 65535]], dtype=np.uint16),
             serial="rs-7", failure=None, aligned=False)
    sdk.camera_info = NS(serial_number="serial", name="name")
    sdk.stream, sdk.format = NS(color="color", depth="depth"), NS(rgb8="rgb8", z16="z16")
    device = NS(get_info=lambda key: sdk.serial if key == "serial" else "RealSense D405",
                first_depth_sensor=lambda: NS(get_depth_scale=lambda: 0.001))
    sdk.context = lambda: NS(query_devices=lambda: [device])
    calibration = NS(width=2, height=2, fx=4.0, fy=5.0, ppx=0.5, ppy=0.75,
                     model="brown_conrady", coeffs=[0.1, 0.2, 0, 0, 0])

    def frame(data, timestamp, domain):
        return NS(get_data=lambda: data, get_timestamp=lambda: timestamp,
                  get_frame_timestamp_domain=lambda: domain,
                  profile=NS(as_video_stream_profile=lambda: NS(get_intrinsics=lambda: calibration)))

    source_depth = frame(sdk.depth, 11.25, "hardware_clock")
    original = NS(get_depth_frame=lambda: source_depth)
    color = frame(sdk.rgb, 12.5, "system_time")
    aligned = NS(get_color_frame=lambda: color,
                 get_depth_frame=lambda: frame(sdk.depth, 11.25, "hardware_clock"))

    class Options:
        def __init__(self):
            self.streams = []

        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, *args):
            self.streams.append(args)

    class Pipeline:
        def __init__(self):
            self.stopped = 0
            sdk.pipelines.append(self)

        def start(self, options):
            self.options = options
            if sdk.failure == "start":
                raise RuntimeError("start failure")
            return NS(get_device=lambda: device)

        def wait_for_frames(self):
            if sdk.failure == "read":
                raise RuntimeError("disconnected")
            return original

        def stop(self):
            self.stopped += 1

    def align(stream):
        assert stream == "color"

        def process(frames):
            assert frames is original
            sdk.aligned = True
            return aligned

        return NS(process=process)

    sdk.config, sdk.pipeline, sdk.align = Options, Pipeline, align
    return sdk


def test_camera_imports_and_constructors_do_not_import_sdk():
    code = """
import sys
from grasp_workbench.cameras.zed import ZedCamera
from grasp_workbench.cameras.realsense import RealSenseCamera
ZedCamera(); RealSenseCamera()
assert 'pyzed.sl' not in sys.modules
assert 'pyrealsense2' not in sys.modules
"""
    root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": root, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module,factory,camera_class", [
    (zed, fake_zed, zed.ZedCamera), (realsense, fake_realsense, realsense.RealSenseCamera),
])
def test_discovery_only_enumerates(module, factory, camera_class, monkeypatch):
    sdk = factory()
    monkeypatch.setattr(module, "_sdk", lambda: sdk)
    found = camera_class.discover()
    assert found[0]["camera_id"]
    assert found[0]["model"]
    assert getattr(sdk, "cameras", getattr(sdk, "pipelines", [])) == []


def test_zed_rgb_metres_timestamps_and_owned_buffers(monkeypatch):
    sdk = fake_zed()
    monkeypatch.setattr(zed, "_sdk", lambda: sdk)
    camera = zed.ZedCamera(serial="12345")
    camera.open()
    owned = sdk.cameras[-1]
    assert owned.options.serial == 12345
    assert owned.options.coordinate_units == "METER"
    assert owned.options.depth_mode == "PERFORMANCE"
    first = camera.read()
    assert first.metadata["sequence"] == 0
    assert first.metadata["camera_id"] == "12345"
    assert first.metadata["capture_time_ns"] == 7001
    assert first.metadata["timestamp_domain"] == "zed_image"
    assert first.metadata["depth_aligned_to_color"]
    assert first.metadata["intrinsics"]["fx"] == 5.0
    assert first.metadata["received_monotonic_ns"] > 0
    np.testing.assert_array_equal(first.rgb[0, 0], [30, 20, 10])
    np.testing.assert_array_equal(first.depth_m, [[1.5, np.nan], [np.inf, 0]])
    sdk.bgr[:] = 99
    sdk.depth[:] = 88
    assert first.rgb[0, 0, 0] == 30 and first.depth_m[0, 0] == 1.5
    assert first.rgb.flags.c_contiguous and first.depth_m.flags.c_contiguous
    assert camera.read().metadata["sequence"] == 1
    stream = first.metadata["stream_id"]
    camera.open()
    assert owned.closed == 1
    new = camera.read()
    assert new.metadata["sequence"] == 0 and new.metadata["stream_id"] != stream
    camera.close()
    camera.close()
    assert sdk.cameras[-1].closed == 1


@pytest.mark.parametrize("failure", ["open", "read", "serial"])
def test_zed_failure_closes_owned_camera(failure, monkeypatch):
    sdk = fake_zed()
    monkeypatch.setattr(zed, "_sdk", lambda: sdk)
    camera = zed.ZedCamera(serial="12345")
    if failure == "open":
        sdk.status = "CAMERA_NOT_DETECTED"
    elif failure == "serial":
        sdk.serial = 777
    if failure == "read":
        camera.open()
        sdk.status = "CAMERA_DISCONNECTED"
        action = camera.read
    else:
        action = camera.open
    with pytest.raises(RuntimeError):
        action()
    assert sdk.cameras[-1].closed == 1
    camera.close()
    assert sdk.cameras[-1].closed == 1


def test_realsense_alignment_metres_original_timestamps_and_ownership(monkeypatch):
    sdk = fake_realsense()
    monkeypatch.setattr(realsense, "_sdk", lambda: sdk)
    camera = realsense.RealSenseCamera(serial="rs-7", width=2, height=2)
    camera.open()
    pipeline = sdk.pipelines[-1]
    assert pipeline.options.serial == "rs-7"
    assert pipeline.options.streams == [("color", 2, 2, "rgb8", 30), ("depth", 2, 2, "z16", 30)]
    first = camera.read()
    assert sdk.aligned
    np.testing.assert_array_equal(first.rgb[0, 0], [1, 2, 3])
    np.testing.assert_allclose(first.depth_m, [[1, 0], [2, 65.535]], rtol=1e-6)
    assert first.depth_m.dtype == np.float32
    assert first.metadata["capture_time_ns"] == 12_500_000
    assert first.metadata["timestamp_domain"] == "system_time"
    assert first.metadata["depth_capture_time_ns"] == 11_250_000
    assert first.metadata["depth_timestamp_domain"] == "hardware_clock"
    assert first.metadata["intrinsics"]["coefficients"] == [0.1, 0.2, 0, 0, 0]
    assert first.metadata["camera_id"] == "rs-7"
    assert first.metadata["depth_aligned_to_color"]
    sdk.rgb[:] = 99
    sdk.depth[:] = 999
    assert first.rgb[0, 0, 0] == 1 and first.depth_m[0, 0] == 1
    assert camera.read().metadata["sequence"] == 1
    old_stream = first.metadata["stream_id"]
    camera.open()
    assert pipeline.stopped == 1
    assert camera.read().metadata["stream_id"] != old_stream
    assert camera.read().metadata["sequence"] == 1
    camera.close()
    camera.close()
    assert sdk.pipelines[-1].stopped == 1


@pytest.mark.parametrize("failure", ["start", "read", "serial"])
def test_realsense_failure_closes_owned_pipeline(failure, monkeypatch):
    sdk = fake_realsense()
    monkeypatch.setattr(realsense, "_sdk", lambda: sdk)
    camera = realsense.RealSenseCamera(serial="rs-7")
    if failure == "serial":
        sdk.serial = "other-device"
    elif failure == "start":
        sdk.failure = "start"
    if failure == "read":
        camera.open()
        sdk.failure = "read"
        action = camera.read
    else:
        action = camera.open
    with pytest.raises(RuntimeError):
        action()
    assert sdk.pipelines[-1].stopped == 1
    camera.close()
    assert sdk.pipelines[-1].stopped == 1


@pytest.mark.parametrize("module", [zed, realsense])
def test_missing_sdk_message_is_install_guidance_not_installer(module, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="Install.*SDK"):
        module.discover()
