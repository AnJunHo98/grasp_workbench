"""Local LeRobot RGB-D episodes; no camera SDK, inference, or Hub upload."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ipc.camera import _parts
from .ipc.foundationpose import PoseResult

REPO_ID = "local/grasp_workbench"
TASK = "Observe objects with an RGB-D camera"


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _features(height, width):
    features = {
        "observation.images.rgb": {
            "dtype": "image", "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.depth_m": {
            "dtype": "image", "shape": (1, height, width),
            "names": ["channels", "height", "width"],
            "info": {"is_depth_map": True, "depth_unit": "m"},
        },
        "observation.camera_metadata": {"dtype": "string", "shape": (1,), "names": None},
        "observation.objects": {"dtype": "string", "shape": (1,), "names": None},
    }
    for slot in ("A", "B"):
        features[f"observation.pose.{slot}"] = {
            "dtype": "float64", "shape": (16,),
            "names": [f"m{row}{column}" for row in range(4) for column in range(4)],
        }
        features[f"observation.pose.{slot}_valid"] = {
            "dtype": "bool", "shape": (1,), "names": ["pose_available"],
        }
    return features


def _object_record(item):
    return {
        "object_id": item.object_id, "session_id": item.session_id,
        "mesh_path": item.mesh_path, "status": item.status,
        "camera_from_object": (None if item.camera_from_object is None
                               else np.asarray(item.camera_from_object).tolist()),
        "pose_source": item.pose_source, "error": item.error,
        "visual_mismatch_count": item.visual_mismatch_count,
        "diagnostics": item.diagnostics,
    }


def _dataset_frame(result):
    frame = result.frame
    # Match the application's camera codec, including metric float32 depth.
    _parts(frame)
    objects = {item.object_id: item for item in result.objects}
    values = {
        "observation.images.rgb": np.ascontiguousarray(frame.rgb.transpose(2, 0, 1)),
        "observation.depth_m": np.ascontiguousarray(frame.depth_m[None, ...]),
        "observation.camera_metadata": _json(frame.metadata),
        "observation.objects": _json([_object_record(item) for item in result.objects]),
        "task": TASK,
    }
    for slot in ("A", "B"):
        item = objects.get(slot)
        valid = item is not None and item.camera_from_object is not None
        values[f"observation.pose.{slot}"] = (
            np.asarray(item.camera_from_object, dtype=np.float64).reshape(16).copy()
            if valid else np.zeros(16, dtype=np.float64)
        )
        values[f"observation.pose.{slot}_valid"] = np.array([valid], dtype=np.bool_)
    return values


class LeRobotRecorder:
    """One local dataset with explicitly started, saved, or discarded episodes.

    LeRobot stores RGB as lossless PNG and metric float32 depth as lossless TIFF
    in image-backed Parquet rows. Load with ``depth_output_unit="m"`` to avoid
    LeRobot's default metres-to-millimetres conversion. LeRobot's ``timestamp``
    is the nominal sample index / fps; original times stay in camera_metadata.
    """

    def __init__(self, output_dir, fps=30, run_metadata=None, *, dataset_factory=None):
        if type(fps) is not int or fps <= 0:
            raise ValueError("Recording fps must be a positive integer")
        self.output_dir, self.fps = Path(output_dir), fps
        # Only connection/AprilCube settings belong here, never a complete app config.
        allowed = {"source", "camera", "aprilcube"}
        self.run_metadata = {key: value for key, value in (run_metadata or {}).items()
                             if key in allowed}
        _json(self.run_metadata)
        if dataset_factory is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset_factory = LeRobotDataset.create
        self.dataset_factory = dataset_factory
        self.dataset = None
        self.shape = None
        self.recording = False
        self.closed = False
        self.episode_index = self.episode_frames = self.total_frames = 0

    def start(self):
        if self.closed or self.recording:
            raise RuntimeError("Recorder is closed or already recording")
        self.recording = True

    def _create_dataset(self, frame):
        height, width = frame.rgb.shape[:2]
        self.dataset = self.dataset_factory(
            repo_id=REPO_ID, root=self.output_dir, fps=self.fps,
            features=_features(height, width), use_videos=False,
            image_writer_processes=0, image_writer_threads=0, video_backend="pyav",
        )
        self.shape = (height, width)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "recording.json").write_text(_json({
            "schema_version": 1, "format": "lerobot", "repo_id": REPO_ID,
            "depth_unit": "m", "pose_frame": "camera_optical",
            "pose_layout": "row_major_4x4", "run": self.run_metadata,
        }), encoding="utf-8")

    def add_frame(self, result: PoseResult):
        if self.closed or not self.recording:
            raise RuntimeError("Start recording before adding frames")
        values = _dataset_frame(result)
        if self.dataset is None:
            self._create_dataset(result.frame)
        if result.frame.rgb.shape[:2] != self.shape:
            raise ValueError("Camera image dimensions changed within the dataset")
        self.dataset.add_frame(values)
        self.episode_frames += 1
        self.total_frames += 1

    def save(self):
        if self.closed or not self.recording:
            raise RuntimeError("No recording episode is active")
        if not self.episode_frames:
            raise RuntimeError("Cannot save an empty recording episode")
        self.dataset.save_episode()
        self.episode_index += 1
        self.episode_frames = 0
        self.recording = False

    def discard(self):
        if self.closed or not self.recording:
            raise RuntimeError("No recording episode is active")
        if self.dataset is not None:
            self.dataset.clear_episode_buffer()
        self.total_frames -= self.episode_frames
        self.episode_frames = 0
        self.recording = False

    def close(self):
        if self.closed:
            return
        try:
            if self.recording:
                if self.episode_frames:
                    self.save()
                else:
                    self.discard()
        finally:
            try:
                if self.dataset is not None:
                    self.dataset.finalize()
            finally:
                self.closed = True
