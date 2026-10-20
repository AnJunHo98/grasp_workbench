"""Minimal direct ZED SDK RGB-D connection example; no SDK import until used."""
from contextlib import suppress
import importlib
import time
import uuid

import numpy as np


def _sdk():
    try:
        return importlib.import_module("pyzed.sl")
    except (ImportError, OSError) as error:
        raise RuntimeError("Install the ZED SDK and its matching pyzed binding in the camera environment") from error


def discover():
    sl = _sdk()
    return [{"camera_id": str(device.serial_number), "model": str(device.camera_model),
             "state": str(device.camera_state)} for device in sl.Camera.get_device_list()]


class ZedCamera:
    discover = staticmethod(discover)

    def __init__(self, serial=None, resolution="HD720", fps=30):
        self.serial, self.resolution, self.fps = serial, resolution, fps
        self._camera = None

    def open(self):
        self.close()
        self._sl = sl = _sdk()
        self._camera = sl.Camera()
        try:
            options = sl.InitParameters()
            options.camera_resolution = getattr(sl.RESOLUTION, self.resolution.upper())
            options.camera_fps = self.fps
            options.coordinate_units = sl.UNIT.METER
            options.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            if self.serial is not None:
                options.set_from_serial_number(int(self.serial))
            self._success(self._camera.open(options), "open")
            info = self._camera.get_camera_information()
            self._serial = str(info.serial_number)
            if self.serial is not None and self._serial != str(self.serial):
                raise RuntimeError("ZED opened a different serial than requested")
            configuration = info.camera_configuration
            calibration = configuration.calibration_parameters.left_cam
            self._intrinsics = {
                "width": configuration.resolution.width, "height": configuration.resolution.height,
                "fx": float(calibration.fx), "fy": float(calibration.fy),
                "cx": float(calibration.cx), "cy": float(calibration.cy),
                "distortion_model": "zed_rectified",
                "coefficients": np.asarray(calibration.disto).reshape(-1).tolist(),
            }
            self._image, self._depth = sl.Mat(), sl.Mat()
            self._runtime = sl.RuntimeParameters()
            self._stream_id, self._sequence = uuid.uuid4().hex, 0
        except Exception:
            with suppress(Exception):
                self.close()
            raise

    def _success(self, status, operation):
        if status != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED {operation} failed: {status}")

    def read(self):
        if self._camera is None:
            raise RuntimeError("Open the ZED camera before reading")
        try:
            sl = self._sl
            self._success(self._camera.grab(self._runtime), "grab")
            captured = int(self._camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            self._success(self._camera.retrieve_image(self._image, sl.VIEW.LEFT_BGR), "image retrieval")
            self._success(self._camera.retrieve_measure(self._depth, sl.MEASURE.DEPTH), "depth retrieval")
            received = time.monotonic_ns()
            bgr, depth = self._image.get_data(), self._depth.get_data()
            rgb = np.array(bgr[:, :, ::-1], dtype=np.uint8, copy=True, order="C")
            depth_m = np.array(depth, dtype=np.float32, copy=True, order="C")
            metadata = {
                "camera_id": self._serial, "sequence": self._sequence,
                "stream_id": self._stream_id, "capture_time_ns": captured,
                "timestamp_domain": "zed_image", "received_monotonic_ns": received,
                "intrinsics": dict(self._intrinsics),
                "camera_frame": f"zed_{self._serial}_left_optical",
                "depth_aligned_to_color": True, "depth_source": "zed_sdk",
                "depth_capture_time_ns": captured, "depth_timestamp_domain": "zed_image",
                "depth_scale_to_m": 1.0,
            }
            from ..ipc.camera import CameraFrame

            frame = CameraFrame(rgb, depth_m, metadata)
            self._sequence += 1
            return frame
        except Exception:
            with suppress(Exception):
                self.close()
            raise

    def close(self):
        camera, self._camera = self._camera, None
        if camera is not None:
            camera.close()
