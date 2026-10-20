"""One-shot box/point refinement on an exact owned RGB-D/mask via iceoryx2."""
from dataclasses import dataclass
import json
import struct

import numpy as np

from .camera import CameraFrame, decode_frame, encode_frame
from .sam3 import Sam3Result, _BytePublisher, _ByteSubscriber, decode_result as decode_sam3, encode_result as encode_sam3

_LENGTH = struct.Struct("<I")


@dataclass
class RefineRequest:
    request_id: str
    object_id: str
    prompt: str
    frame: CameraFrame
    boxes: tuple[dict, ...] = ()
    points: tuple[dict, ...] = ()
    seed_mask: np.ndarray | None = None


@dataclass
class RefineResult:
    request_id: str
    status: str
    result: Sam3Result | None = None
    error: str | None = None


def services(base):
    return base + "/sam3/refine/requests", base + "/sam3/refine/results"


def _header(kind, **fields):
    if not isinstance(fields["request_id"], str):
        raise ValueError("Refinement request ID must be a string")
    payload = json.dumps({"schema_version": 1, "kind": kind, **fields}, ensure_ascii=False, allow_nan=False).encode()
    return _LENGTH.pack(len(payload)) + payload


def _read(payload, kind):
    data = memoryview(payload).cast("B")
    if len(data) < 4:
        raise ValueError("Truncated refinement header")
    offset = 4 + _LENGTH.unpack_from(data)[0]
    if offset > len(data):
        raise ValueError("Truncated refinement metadata")
    record = json.loads(bytes(data[4:offset]))
    if record.get("schema_version") != 1 or record.get("kind") != kind:
        raise ValueError("Unsupported refinement schema")
    return record, data, offset


def encode_request(request):
    if not isinstance(request.object_id, str) or not isinstance(request.prompt, str):
        raise ValueError("Refinement object and prompt must be strings")
    for box in request.boxes:
        xyxy = np.asarray(box["xyxy"], dtype=float)
        if (xyxy.shape != (4,) or not np.isfinite(xyxy).all() or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]
                or type(box.get("label", True)) is not bool):
            raise ValueError("Box prompts require finite original-pixel xyxy and a boolean label")
    for point in request.points:
        xy = np.asarray(point["xy"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or point["label"] not in (0, 1):
            raise ValueError("Point prompts require finite original-pixel xy and a 0/1 label")
    frame = encode_frame(request.frame)
    mask = None if request.seed_mask is None else np.asarray(request.seed_mask)
    if mask is not None and (mask.dtype != np.bool_ or mask.shape != request.frame.rgb.shape[:2]):
        raise ValueError("Refinement seed must match the original RGB-D frame")
    return b"".join((_header("refine_request", request_id=request.request_id, object_id=request.object_id,
                             prompt=request.prompt, boxes=request.boxes, points=request.points,
                             frame_size=len(frame), has_seed=mask is not None), frame,
                      b"" if mask is None else mask.tobytes(order="C")))


def decode_request(payload):
    record, view, offset = _read(payload, "refine_request")
    size = record["frame_size"]
    if type(size) is not int or size <= 0 or offset + size > len(view) or type(record["has_seed"]) is not bool:
        raise ValueError("Invalid refinement frame size or seed flag")
    frame = decode_frame(view[offset:offset + size])
    offset += size
    count = frame.rgb.shape[0] * frame.rgb.shape[1] if record["has_seed"] else 0
    if len(view) != offset + count:
        raise ValueError("Refinement mask byte length does not match its frame")
    mask = np.frombuffer(view, np.bool_, count, offset).reshape(frame.rgb.shape[:2]).copy() if count else None
    request = RefineRequest(record["request_id"], record["object_id"], record["prompt"], frame,
                            tuple(record["boxes"]), tuple(record["points"]), mask)
    encode_request(request)
    return request


def encode_result(result):
    if (result.status not in {"running", "done", "error"}
            or (result.status == "done") != (result.result is not None)
            or result.error is not None and not isinstance(result.error, str)):
        raise ValueError("Invalid refinement result")
    payload = b"" if result.result is None else encode_sam3(result.result)
    return _header("refine_result", request_id=result.request_id, status=result.status,
                   result_size=len(payload), error=result.error) + payload


def decode_result(payload):
    record, view, offset = _read(payload, "refine_result")
    size = record["result_size"]
    if type(size) is not int or size < 0 or len(view) != offset + size:
        raise ValueError("Invalid refinement result byte length")
    result = RefineResult(record["request_id"], record["status"], decode_sam3(view[offset:]) if size else None, record["error"])
    encode_result(result)
    return result


class _RefinePublisher(_BytePublisher):
    def update_connections(self):
        return self._publisher.update_connections()


class RefineRequestPublisher(_RefinePublisher):
    def publish(self, request):
        return self._publish(encode_request(request))


class RefineRequestSubscriber(_ByteSubscriber):
    def receive_latest(self):
        return self._receive(decode_request)


class RefineResultPublisher(_RefinePublisher):
    def publish(self, result):
        return self._publish(encode_result(result))


class RefineResultSubscriber(_ByteSubscriber):
    def receive_latest(self):
        return self._receive(decode_result)
