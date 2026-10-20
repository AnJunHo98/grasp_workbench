"""Owned RGB-D/mask/pose snapshots for explicit offline review, never robot commands."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .ipc.camera import CameraFrame, decode_frame, encode_frame
from .ipc.foundationpose import PoseResult, decode_result as decode_pose, encode_result as encode_pose
from .ipc.sam3 import Sam3Result, decode_result as decode_sam3, encode_result as encode_sam3


@dataclass
class WorkbenchSnapshot:
    frame: CameraFrame
    sam3: Sam3Result | None = None
    pose_result: PoseResult | None = None
    objects: tuple[dict, ...] = ()
    semantic: dict | None = None


def save_snapshot(path, frame, *, sam3=None, pose_result=None, objects=(), semantic=None):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"schema_version": 1, "objects": objects, "semantic": semantic},
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.frombuffer(metadata, np.uint8),
                            frame=np.frombuffer(encode_frame(frame), np.uint8),
                            sam3=np.frombuffer(b"" if sam3 is None else encode_sam3(sam3), np.uint8),
                            poses=np.frombuffer(b"" if pose_result is None else encode_pose(pose_result), np.uint8))
    return path


def load_snapshot(path):
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].tobytes())
        if metadata["schema_version"] != 1:
            raise ValueError("Unsupported workbench snapshot")
        frame = decode_frame(data["frame"].tobytes())
        sam3 = decode_sam3(data["sam3"].tobytes()) if data["sam3"].size else None
        poses = decode_pose(data["poses"].tobytes()) if data["poses"].size else None
        return WorkbenchSnapshot(frame, sam3, poses, tuple(metadata["objects"]), metadata["semantic"])
