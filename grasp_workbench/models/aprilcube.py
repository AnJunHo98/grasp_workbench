"""Official AprilCube detectors in the raw camera optical frame, in metres."""
import importlib
from pathlib import Path
import sys
import time

import numpy as np


def camera_parameters(frame):
    intrinsics = frame.metadata["intrinsics"]
    matrix = np.asarray(((intrinsics["fx"], 0, intrinsics["cx"]),
                         (0, intrinsics["fy"], intrinsics["cy"]), (0, 0, 1)), dtype=float)
    # ZED LEFT is already rectified: its original calibration coefficients must
    # not be applied a second time to either detection or display projection.
    distortion = np.zeros(5) if intrinsics.get("distortion_model") == "zed_rectified" else np.asarray(
        intrinsics.get("coefficients", [0., 0., 0., 0., 0.]), dtype=float,
    )
    return matrix, distortion


class _FrameMarkerDetector:
    """Share one SDK marker pass without sharing cube tracking state."""
    def __init__(self, detector, log_timing):
        self.detector, self.log_timing = detector, log_timing
        self.reset()

    def reset(self):
        self.result = None
        self.elapsed_ms = 0.

    def detectMarkers(self, image):
        if self.result is None:
            started = time.perf_counter_ns() if self.log_timing else None
            self.result = self.detector.detectMarkers(image)
            if self.log_timing:
                self.elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return self.result


class OfficialAprilCubePredictor:
    def __init__(self, source_dir, config_a, config_b, *, filter_enabled=True, fast=True,
                 log_timing=False):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.configs = {"A": str(Path(config_a).expanduser().resolve()),
                        "B": str(Path(config_b).expanduser().resolve())}
        self.filter_enabled = filter_enabled
        self.fast, self.log_timing = fast, log_timing
        self.timings_ms = {}
        self.diagnostics = {}
        self._backend, self._calibration = None, None
        self._detectors = {}
        self._marker_detectors = {}

    def start(self):
        sys.path.insert(0, str(self.source_dir / "src"))
        self._backend = importlib.import_module("aprilcube")

    def predict(self, frame):
        self.timings_ms = {}
        self.diagnostics = {}
        started = time.perf_counter_ns() if self.log_timing else None
        matrix, distortion = camera_parameters(frame)
        calibration = (frame.metadata.get("camera_id"), frame.metadata.get("stream_id"),
                       frame.rgb.shape[:2], tuple(matrix.flat), tuple(distortion))
        if calibration != self._calibration:
            self._detectors = {object_id: self._backend.detector(
                config, matrix, dist_coeffs=distortion, enable_filter=self.filter_enabled, fast=self.fast,
            ) for object_id, config in self.configs.items()}
            self._marker_detectors = {}
            for estimator in self._detectors.values():
                for name in ("detector", "fallback_detector"):
                    key = (estimator.config.dict_id, name)
                    if key not in self._marker_detectors:
                        self._marker_detectors[key] = _FrameMarkerDetector(
                            getattr(estimator, name), self.log_timing)
                    setattr(estimator, name, self._marker_detectors[key])
            self._calibration = calibration
        for detector in self._marker_detectors.values():
            detector.reset()
        bgr = np.ascontiguousarray(frame.rgb[:, :, ::-1])
        received = frame.metadata.get("received_monotonic_ns")
        timestamp = None if received is None else received / 1e9
        if self.log_timing:
            self.timings_ms["prepare_ms"] = (time.perf_counter_ns() - started) / 1e6
        poses = {}
        for object_id, detector in self._detectors.items():
            started = time.perf_counter_ns() if self.log_timing else None
            result = detector.process_frame(bgr, timestamp=timestamp)
            diagnostic = {key: int(result[key]) for key in ("n_tags", "n_inliers") if key in result}
            for key in ("success", "predicted"):
                if key in result:
                    diagnostic[key] = bool(result[key])
            if "tag_ids" in result:
                diagnostic["tag_ids"] = [int(value) for value in result["tag_ids"]]
            if "visible_faces" in result:
                diagnostic["visible_faces"] = sorted(result["visible_faces"])
            if "reproj_error" in result:
                error = float(result["reproj_error"])
                diagnostic["reproj_error"] = error if np.isfinite(error) else None
            self.diagnostics[object_id] = diagnostic
            pose = None
            if result["success"]:
                pose = np.array(result["T"], dtype=float, copy=True)
                pose[:3, 3] *= .001  # Official cube dimensions and pose translation are millimetres.
            poses[object_id] = pose
            if self.log_timing:
                self.timings_ms[object_id + "_ms"] = (time.perf_counter_ns() - started) / 1e6
        if self.log_timing:
            # Decode time is already included in A_ms/B_ms, not an extra stage.
            self.timings_ms["detect_ms"] = sum(item.elapsed_ms for item in self._marker_detectors.values())
        return poses

    def close(self):
        self._detectors.clear()
        self._marker_detectors.clear()
        self._backend, self._calibration = None, None
