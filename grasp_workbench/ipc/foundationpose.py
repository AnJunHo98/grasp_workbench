"""FoundationPose session controls and camera-frame poses over iceoryx2."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from .camera import CameraFrame, decode_frame, encode_frame
from .sam3 import _BytePublisher, _ByteSubscriber

_LENGTH = struct.Struct("<I")


@dataclass
class PoseTarget:
    object_id: str
    session_id: str
    mesh_path: str
    mesh_scale_to_m: float
    frame: CameraFrame
    mask: np.ndarray | None
    prompt_revision: str = ""
    initial_pose: np.ndarray | None = None
    initialization: str = "register"


@dataclass
class PoseControl:
    targets: tuple[PoseTarget, ...]


@dataclass
class ObjectPose:
    object_id: str
    session_id: str
    mesh_path: str
    status: str
    camera_from_object: np.ndarray | None = None
    error: str | None = None
    visual_mismatch_count: int = 0


@dataclass
class PoseResult:
    frame: CameraFrame
    objects: tuple[ObjectPose, ...]


def services(base):
    return base + "/foundationpose/control", base + "/foundationpose/results"


def _identity(item):
    fields = {name: getattr(item, name) for name in ("object_id", "session_id", "mesh_path")}
    if not all(isinstance(value, str) for value in fields.values()):
        raise ValueError("Pose object/session IDs and mesh path must be strings")
    return fields


def _header(kind, **fields):
    data = json.dumps({"schema_version": 1, "kind": kind, **fields},
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return _LENGTH.pack(len(data)) + data


def _read(payload, kind):
    view = memoryview(payload).cast("B")
    if len(view) < _LENGTH.size:
        raise ValueError("Truncated pose header")
    offset = _LENGTH.size + _LENGTH.unpack_from(view)[0]
    if offset > len(view):
        raise ValueError("Truncated pose metadata")
    record = json.loads(bytes(view[_LENGTH.size:offset]))
    if not isinstance(record, dict) or (
        type(record.get("schema_version")) is not int or record["schema_version"] != 1
        or record.get("kind") != kind or not isinstance(record.get("objects"), list)
    ):
        raise ValueError("Unsupported pose schema or metadata")
    for item in record["objects"]:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(name), str) for name in ("object_id", "session_id", "mesh_path")
        ):
            raise ValueError("Invalid pose identity")
    return record, view, offset


def encode_control(control: PoseControl) -> bytes:
    objects, arrays = [], []
    for target in control.targets:
        identity = _identity(target)
        frame = encode_frame(target.frame)
        mask = None if target.mask is None else np.asarray(target.mask)
        if mask is not None and (mask.dtype != np.bool_ or mask.shape != target.frame.rgb.shape[:2]):
            raise ValueError("Pose mask must be bool HxW matching its source frame")
        if isinstance(target.mesh_scale_to_m, bool) or not isinstance(target.mesh_scale_to_m, (float, int)):
            raise ValueError("Mesh scale must be a number")
        if not isinstance(target.prompt_revision, str):
            raise ValueError("Prompt revision must be a string")
        if target.initialization not in {"register", "seed"}:
            raise ValueError("Pose initialization must be register or seed")
        initial_pose = None if target.initial_pose is None else np.asarray(target.initial_pose, dtype=float)
        if initial_pose is not None and (initial_pose.shape != (4, 4) or not np.isfinite(initial_pose).all()):
            raise ValueError("Initial object pose must be a finite 4x4 matrix")
        if target.initialization == "seed" and initial_pose is None:
            raise ValueError("Seed initialization needs an initial object pose")
        if target.initialization == "register" and mask is None:
            raise ValueError("Mask registration needs an object mask")
        objects.append({**identity, "mesh_scale_to_m": target.mesh_scale_to_m,
                        "frame_size": len(frame), "prompt_revision": target.prompt_revision,
                        "initial_pose": None if initial_pose is None else initial_pose.tolist(),
                        "initialization": target.initialization, "has_mask": mask is not None})
        arrays.extend((frame, b"" if mask is None else mask.tobytes(order="C")))
    return b"".join((_header("pose_control", objects=objects, mask_encoding="bool8"), *arrays))


def decode_control(payload) -> PoseControl:
    record, view, offset = _read(payload, "pose_control")
    if record.get("mask_encoding") != "bool8":
        raise ValueError("Unsupported pose mask encoding")
    targets = []
    for item in record["objects"]:
        size, scale = item.get("frame_size"), item.get("mesh_scale_to_m")
        if (type(size) is not int or size <= 0 or offset + size > len(view)
                or isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not isinstance(item.get("prompt_revision", ""), str)):
            raise ValueError("Invalid pose frame size or mesh scale")
        frame = decode_frame(view[offset:offset + size])
        offset += size
        height, width = frame.rgb.shape[:2]
        if type(item.get("has_mask", True)) is not bool:
            raise ValueError("Invalid pose mask flag")
        count = height * width if item.get("has_mask", True) else 0
        if offset + count > len(view):
            raise ValueError("Truncated pose mask")
        mask = np.frombuffer(view, np.bool_, count, offset).reshape(height, width).copy() if count else None
        offset += count
        target = PoseTarget(item["object_id"], item["session_id"], item["mesh_path"],
                                  scale, frame, mask, item.get("prompt_revision", ""),
                                  None if item.get("initial_pose") is None else np.asarray(item["initial_pose"], dtype=float),
                                  item.get("initialization", "register"))
        if (target.initialization not in {"register", "seed"}
                or target.initialization == "register" and mask is None
                or target.initialization == "seed" and target.initial_pose is None
                or target.initial_pose is not None and (target.initial_pose.shape != (4, 4)
                                                       or not np.isfinite(target.initial_pose).all())):
            raise ValueError("Invalid pose initialization")
        targets.append(target)
    if offset != len(view):
        raise ValueError("Unexpected pose control payload bytes")
    return PoseControl(tuple(targets))


def _pose_record(item):
    identity = _identity(item)
    if not isinstance(item.status, str) or (item.error is not None and not isinstance(item.error, str)):
        raise ValueError("Pose status/error must be strings")
    if type(item.visual_mismatch_count) is not int or item.visual_mismatch_count < 0:
        raise ValueError("Visual mismatch count must be a nonnegative integer")
    pose = None
    if item.camera_from_object is not None:
        pose = np.asarray(item.camera_from_object)
        if pose.shape != (4, 4) or pose.dtype.kind not in "fiu":
            raise ValueError("Camera-from-object pose must be a numeric 4x4 matrix")
        pose = pose.tolist()
    return {**identity, "status": item.status, "camera_from_object": pose, "error": item.error,
            "visual_mismatch_count": item.visual_mismatch_count}


def encode_result(result: PoseResult) -> bytes:
    frame = encode_frame(result.frame)
    return _header("pose_result", objects=[_pose_record(item) for item in result.objects],
                   frame_size=len(frame)) + frame


def decode_result(payload) -> PoseResult:
    record, view, offset = _read(payload, "pose_result")
    size = record.get("frame_size")
    if type(size) is not int or size <= 0 or offset + size != len(view):
        raise ValueError("Invalid pose source frame length")
    frame = decode_frame(view[offset:])
    objects = []
    for item in record["objects"]:
        pose = item.get("camera_from_object")
        if pose is not None:
            pose = np.array(pose, dtype=np.float64)
        obj = ObjectPose(item["object_id"], item["session_id"], item["mesh_path"],
                         item.get("status"), pose, item.get("error"),
                         item.get("visual_mismatch_count", 0))
        _pose_record(obj)
        objects.append(obj)
    return PoseResult(frame, tuple(objects))


class _PosePublisher(_BytePublisher):
    def update_connections(self):
        if self._publisher is None:
            raise RuntimeError("Pose publisher is closed")
        return self._publisher.update_connections()


class PoseControlPublisher(_PosePublisher):
    def publish(self, control: PoseControl):
        return self._publish(encode_control(control))


class PoseControlSubscriber(_ByteSubscriber):
    def receive_latest(self) -> PoseControl | None:
        return self._receive(decode_control)


class PoseResultPublisher(_PosePublisher):
    def publish(self, result: PoseResult):
        return self._publish(encode_result(result))


class PoseResultSubscriber(_ByteSubscriber):
    def receive_latest(self) -> PoseResult | None:
        return self._receive(decode_result)
