"""AprilCube mesh bindings and display-only RGBA layers over iceoryx2."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from .sam3 import _BytePublisher, _ByteSubscriber

_LENGTH = struct.Struct("<I")


@dataclass
class AprilCubeTarget:
    object_id: str
    session_id: str
    mesh_path: str
    mesh_scale_to_m: float
    cube_from_object: np.ndarray


@dataclass
class AprilCubeControl:
    targets: tuple[AprilCubeTarget, ...]


@dataclass
class AprilCubeOverlay:
    metadata: dict
    rgba: np.ndarray


def services(base):
    return base + "/aprilcube/control", base + "/aprilcube/overlay"


def _header(kind, **fields):
    encoded = json.dumps({"schema_version": 1, "kind": kind, **fields},
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return _LENGTH.pack(len(encoded)) + encoded


def _read(payload, kind):
    view = memoryview(payload).cast("B")
    if len(view) < _LENGTH.size:
        raise ValueError("Truncated AprilCube header")
    offset = _LENGTH.size + _LENGTH.unpack_from(view)[0]
    if offset > len(view):
        raise ValueError("Truncated AprilCube metadata")
    record = json.loads(bytes(view[_LENGTH.size:offset]))
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or record.get("kind") != kind):
        raise ValueError("Unsupported AprilCube schema")
    return record, view, offset


def _target_record(target):
    if not all(isinstance(getattr(target, key), str)
               for key in ("object_id", "session_id", "mesh_path")):
        raise ValueError("AprilCube object/session IDs and mesh path must be strings")
    scale = target.mesh_scale_to_m
    if (isinstance(scale, bool) or not isinstance(scale, (float, int))
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("Mesh scale must be positive metres per mesh unit")
    pose = np.asarray(target.cube_from_object, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Cube-from-object must be a finite 4x4 matrix")
    return {"object_id": target.object_id, "session_id": target.session_id,
            "mesh_path": target.mesh_path, "mesh_scale_to_m": scale,
            "cube_from_object": pose.tolist()}


def encode_control(control: AprilCubeControl) -> bytes:
    return _header("aprilcube_control", targets=[_target_record(item) for item in control.targets])


def decode_control(payload) -> AprilCubeControl:
    record, view, offset = _read(payload, "aprilcube_control")
    if offset != len(view) or not isinstance(record.get("targets"), list):
        raise ValueError("Invalid AprilCube control payload")
    targets = []
    for item in record["targets"]:
        if not isinstance(item, dict):
            raise ValueError("AprilCube target must be an object")
        target = AprilCubeTarget(item.get("object_id"), item.get("session_id"),
                                 item.get("mesh_path"), item.get("mesh_scale_to_m"),
                                 np.array(item.get("cube_from_object"), dtype=float))
        _target_record(target)
        targets.append(target)
    return AprilCubeControl(tuple(targets))


def encode_overlay(overlay: AprilCubeOverlay) -> bytes:
    rgba = np.asarray(overlay.rgba)
    if (rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4
            or not rgba.shape[0] or not rgba.shape[1]):
        raise ValueError("AprilCube overlay must be uint8 HxWx4 RGBA")
    if not isinstance(overlay.metadata, dict):
        raise ValueError("AprilCube overlay metadata must be an object")
    return _header("aprilcube_overlay", metadata=overlay.metadata, width=rgba.shape[1],
                   height=rgba.shape[0], encoding="rgba8_straight") + rgba.tobytes(order="C")


def decode_overlay(payload) -> AprilCubeOverlay:
    record, view, offset = _read(payload, "aprilcube_overlay")
    width, height = record.get("width"), record.get("height")
    if (any(type(value) is not int or value <= 0 for value in (width, height))
            or record.get("encoding") != "rgba8_straight"
            or not isinstance(record.get("metadata"), dict)
            or offset + width * height * 4 != len(view)):
        raise ValueError("Invalid AprilCube overlay dimensions or encoding")
    rgba = np.frombuffer(view, np.uint8, width * height * 4, offset).reshape(height, width, 4).copy()
    return AprilCubeOverlay(record["metadata"], rgba)


class _AprilCubePublisher(_BytePublisher):
    def update_connections(self):
        if self._publisher is None:
            raise RuntimeError("AprilCube publisher is closed")
        return self._publisher.update_connections()


class AprilCubeControlPublisher(_AprilCubePublisher):
    def publish(self, control: AprilCubeControl):
        return self._publish(encode_control(control))


class AprilCubeControlSubscriber(_ByteSubscriber):
    def receive_latest(self) -> AprilCubeControl | None:
        return self._receive(decode_control)


class AprilCubeOverlayPublisher(_AprilCubePublisher):
    def publish(self, overlay: AprilCubeOverlay):
        return self._publish(encode_overlay(overlay))


class AprilCubeOverlaySubscriber(_ByteSubscriber):
    def receive_latest(self) -> AprilCubeOverlay | None:
        return self._receive(decode_overlay)
