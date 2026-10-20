"""Object-local grasp requests and raw float64 candidates over iceoryx2."""
from dataclasses import dataclass
import json
import struct

import numpy as np

from .sam3 import _BytePublisher, _ByteSubscriber

_LENGTH = struct.Struct("<I")


@dataclass
class GraspRequest:
    request_id: str
    asset_id: str
    object_id: str
    mesh_path: str
    mesh_scale_to_m: float = 1.0
    regenerate: bool = False


@dataclass
class GraspResult:
    request_id: str
    asset_id: str
    object_id: str
    status: str
    mesh_path: str = ""
    mesh_scale_to_m: float = 1.0
    gripper_id: str = ""
    grasps: np.ndarray | None = None
    scores: np.ndarray | None = None
    bundle_path: str | None = None
    gripper_metadata: dict | None = None
    error: str | None = None
    cached: bool = False


def services(base):
    return base + "/graspgenx/requests", base + "/graspgenx/results"


def _identity(value):
    fields = {key: getattr(value, key) for key in ("request_id", "asset_id", "object_id", "mesh_path", "mesh_scale_to_m")}
    if not all(isinstance(fields[key], str) for key in ("request_id", "asset_id", "object_id", "mesh_path")):
        raise ValueError("Grasp identities and mesh path must be strings")
    if isinstance(fields["mesh_scale_to_m"], bool) or not isinstance(fields["mesh_scale_to_m"], (int, float)):
        raise ValueError("Grasp mesh scale must be numeric")
    return fields


def _header(kind, **fields):
    encoded = json.dumps({"schema_version": 1, "kind": kind, **fields},
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return _LENGTH.pack(len(encoded)) + encoded


def _read(payload, kind):
    view = memoryview(payload).cast("B")
    if len(view) < _LENGTH.size:
        raise ValueError("Truncated grasp header")
    offset = _LENGTH.size + _LENGTH.unpack_from(view)[0]
    if offset > len(view):
        raise ValueError("Truncated grasp metadata")
    record = json.loads(bytes(view[_LENGTH.size:offset]))
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or record.get("kind") != kind):
        raise ValueError("Unsupported grasp schema")
    return record, view, offset


def encode_request(request):
    if type(request.regenerate) is not bool:
        raise ValueError("Grasp regenerate must be boolean")
    return _header("grasp_request", **_identity(request), regenerate=request.regenerate)


def decode_request(payload):
    record, view, offset = _read(payload, "grasp_request")
    if offset != len(view):
        raise ValueError("Unexpected grasp request payload bytes")
    request = GraspRequest(*(record.get(key) for key in
                            ("request_id", "asset_id", "object_id", "mesh_path", "mesh_scale_to_m", "regenerate")))
    encode_request(request)
    return request


def _result_metadata(result):
    fields = _identity(result)
    if result.status not in {"running", "done", "error"} or not isinstance(result.gripper_id, str):
        raise ValueError("Invalid grasp status or gripper ID")
    if any(value is not None and not isinstance(value, str) for value in (result.bundle_path, result.error)):
        raise ValueError("Grasp bundle path and error must be strings or null")
    if result.gripper_metadata is not None and not isinstance(result.gripper_metadata, dict):
        raise ValueError("Gripper metadata must be a mapping or null")
    if type(result.cached) is not bool:
        raise ValueError("Grasp cached must be boolean")
    return {**fields, "status": result.status, "gripper_id": result.gripper_id,
            "bundle_path": result.bundle_path, "gripper_metadata": result.gripper_metadata,
            "error": result.error, "cached": result.cached}


def encode_result(result):
    fields = _result_metadata(result)
    if result.grasps is None and result.scores is None:
        return _header("grasp_result", **fields, count=None, float_encoding="float64_le")
    grasps, scores = np.asarray(result.grasps), np.asarray(result.scores)
    if (grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or scores.shape != (len(grasps),)
            or grasps.dtype.kind not in "fiu" or scores.dtype.kind not in "fiu"):
        raise ValueError("Grasps and scores must be numeric Nx4x4 and N arrays")
    return b"".join((_header("grasp_result", **fields, count=len(grasps), float_encoding="float64_le"),
                     np.asarray(grasps, dtype="<f8").tobytes(order="C"),
                     np.asarray(scores, dtype="<f8").tobytes(order="C")))


def decode_result(payload):
    record, view, offset = _read(payload, "grasp_result")
    count = record.get("count")
    if record.get("float_encoding") != "float64_le" or (count is not None and (type(count) is not int or count < 0)):
        raise ValueError("Invalid grasp array encoding or count")
    expected = 0 if count is None else count * 17 * 8
    if len(view) != offset + expected:
        raise ValueError("Grasp result byte length does not match candidate count")
    grasps = scores = None
    if count is not None:
        grasps = np.frombuffer(view, "<f8", count * 16, offset).reshape(count, 4, 4).copy()
        scores = np.frombuffer(view, "<f8", count, offset + count * 16 * 8).copy()
    result = GraspResult(*(record.get(key) for key in ("request_id", "asset_id", "object_id", "status")),
                         mesh_path=record.get("mesh_path"), mesh_scale_to_m=record.get("mesh_scale_to_m"),
                         gripper_id=record.get("gripper_id"), grasps=grasps, scores=scores,
                         bundle_path=record.get("bundle_path"), gripper_metadata=record.get("gripper_metadata"),
                         error=record.get("error"), cached=record.get("cached"))
    _result_metadata(result)
    return result


class _GraspPublisher(_BytePublisher):
    def update_connections(self):
        if self._publisher is None:
            raise RuntimeError("Grasp publisher is closed")
        return self._publisher.update_connections()


class GraspRequestPublisher(_GraspPublisher):
    def publish(self, request):
        return self._publish(encode_request(request))


class GraspRequestSubscriber(_ByteSubscriber):
    def receive_latest(self):
        return self._receive(decode_request)


class GraspResultPublisher(_GraspPublisher):
    def publish(self, result):
        return self._publish(encode_result(result))


class GraspResultSubscriber(_ByteSubscriber):
    def receive_latest(self):
        return self._receive(decode_result)
