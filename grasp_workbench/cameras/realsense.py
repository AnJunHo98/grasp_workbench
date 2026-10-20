"""Minimal direct RealSense SDK aligned RGB-D connection example."""
from contextlib import suppress
import importlib
import time
import uuid

import numpy as np


def _sdk():
    try:
        return importlib.import_module("pyrealsense2")
    except (ImportError, OSError) as error:
        raise RuntimeError("Install the RealSense SDK and pyrealsense2 in the camera environment") from error


def discover():
    rs = _sdk()
    return [{"camera_id": device.get_info(rs.camera_info.serial_number),
             "model": device.get_info(rs.camera_info.name)} for device in rs.context().query_devices()]


def _timestamp(frame):
    return int(round(frame.get_timestamp() * 1_000_000))


class RealSenseCamera:
    discover = staticmethod(discover)

    def __init__(self, serial=None, width=640, height=480, fps=30):
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self._pipeline = None

    def open(self):
        self.close()
        rs = _sdk()
        self._pipeline = rs.pipeline()
        try:
            options = rs.config()
            if self.serial is not None:
                options.enable_device(str(self.serial))
            options.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            options.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            profile = self._pipeline.start(options)
            device = profile.get_device()
            self._serial = device.get_info(rs.camera_info.serial_number)
            if self.serial is not None and self._serial != str(self.serial):
                raise RuntimeError("RealSense opened a different serial than requested")
            self._depth_scale = float(device.first_depth_sensor().get_depth_scale())
            if not np.isfinite(self._depth_scale) or self._depth_scale <= 0:
                raise RuntimeError("RealSense returned an invalid depth scale")
            self._align = rs.align(rs.stream.color)
            self._stream_id, self._sequence = uuid.uuid4().hex, 0
        except Exception:
            with suppress(Exception):
                self.close()
            raise

    def read(self):
        if self._pipeline is None:
            raise RuntimeError("Open the RealSense camera before reading")
        try:
            original = self._pipeline.wait_for_frames()
            source_depth = original.get_depth_frame()
            frames = self._align.process(original)
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            received = time.monotonic_ns()
            if not color or not depth or not source_depth:
                raise RuntimeError("RealSense frameset has no aligned color/depth pair")
            rgb = np.array(color.get_data(), dtype=np.uint8, copy=True, order="C")
            depth_m = np.array(depth.get_data(), dtype=np.float32, copy=True, order="C")
            depth_m *= self._depth_scale
            calibration = color.profile.as_video_stream_profile().get_intrinsics()
            metadata = {
                "camera_id": self._serial, "sequence": self._sequence,
                "stream_id": self._stream_id, "capture_time_ns": _timestamp(color),
                "timestamp_domain": str(color.get_frame_timestamp_domain()),
                "received_monotonic_ns": received,
                "intrinsics": {
                    "width": calibration.width, "height": calibration.height,
                    "fx": float(calibration.fx), "fy": float(calibration.fy),
                    "cx": float(calibration.ppx), "cy": float(calibration.ppy),
                    "distortion_model": str(calibration.model),
                    "coefficients": list(calibration.coeffs),
                },
                "camera_frame": f"realsense_{self._serial}_color_optical",
                "depth_aligned_to_color": True, "depth_source": "realsense_z16",
                "depth_capture_time_ns": _timestamp(source_depth),
                "depth_timestamp_domain": str(source_depth.get_frame_timestamp_domain()),
                "depth_scale_to_m": self._depth_scale,
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
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            pipeline.stop()
