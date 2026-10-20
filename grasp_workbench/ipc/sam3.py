"""SAM3 prompt/result transport: exact source RGB-D and raw, owned mask arrays.

Construct, use and close each native endpoint in its owning worker/thread.
"""
from __future__ import annotations

import ctypes
import json
import struct
from dataclasses import dataclass

import numpy as np

from .camera import CameraFrame, _service, decode_frame, encode_frame

_LENGTH = struct.Struct("<I")


@dataclass
class ObjectPrompt:
    object_id: str
    text: str


@dataclass
class PromptSet:
    revision: str
    objects: tuple[ObjectPrompt, ...]


@dataclass
class Sam3ObjectResult:
    object_id: str
    text: str
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    error: str | None = None


@dataclass
class Sam3Result:
    frame: CameraFrame
    revision: str
    objects: tuple[Sam3ObjectResult, ...]


def services(camera_service):
    return camera_service + "/sam3/prompts", camera_service + "/sam3/results"


def _identity(object_id, text):
    if not isinstance(object_id, str) or not isinstance(text, str):
        raise ValueError("Object ID and text must be strings")
    return {"object_id": object_id, "text": text}


def _header(kind, revision, objects, **fields):
    if not isinstance(revision, str):
        raise ValueError("Prompt revision must be a string")
    record = {"schema_version": 1, "kind": kind, "revision": revision,
              "objects": objects, **fields}
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return _LENGTH.pack(len(encoded)) + encoded


def _read_header(payload, kind):
    view = memoryview(payload).cast("B")
    if len(view) < _LENGTH.size:
        raise ValueError("Truncated SAM3 header")
    offset = _LENGTH.size + _LENGTH.unpack_from(view)[0]
    if offset > len(view):
        raise ValueError("Truncated SAM3 metadata")
    record = json.loads(bytes(view[_LENGTH.size:offset]))
    if not isinstance(record, dict) or (
        type(record.get("schema_version")) is not int or record["schema_version"] != 1
        or record.get("kind") != kind or not isinstance(record.get("revision"), str)
        or not isinstance(record.get("objects"), list)
    ):
        raise ValueError("Unsupported SAM3 schema or metadata")
    for item in record["objects"]:
        if not isinstance(item, dict):
            raise ValueError("Object metadata must be an object")
        _identity(item.get("object_id"), item.get("text"))
    return record, view, offset


def encode_prompts(prompts: PromptSet) -> bytes:
    return _header("sam3_prompts", prompts.revision,
                   [_identity(item.object_id, item.text) for item in prompts.objects])


def decode_prompts(payload) -> PromptSet:
    record, view, offset = _read_header(payload, "sam3_prompts")
    if offset != len(view):
        raise ValueError("Unexpected prompt payload bytes")
    return PromptSet(record["revision"], tuple(
        ObjectPrompt(item["object_id"], item["text"]) for item in record["objects"]))


def encode_result(result: Sam3Result) -> bytes:
    frame = encode_frame(result.frame)
    height, width = result.frame.rgb.shape[:2]
    objects, arrays = [], []
    for item in result.objects:
        identity = _identity(item.object_id, item.text)
        masks, boxes, scores = map(np.asarray, (item.masks, item.boxes, item.scores))
        if masks.dtype != np.bool_ or masks.ndim != 3 or masks.shape[1:] != (height, width):
            raise ValueError("Masks must be bool NxHxW arrays matching the original frame")
        count = masks.shape[0]
        if (boxes.dtype.kind != "f" or boxes.dtype.itemsize != 4 or boxes.shape != (count, 4)
                or scores.dtype.kind != "f" or scores.dtype.itemsize != 4 or scores.shape != (count,)):
            raise ValueError("Boxes/scores must be float32 Nx4/N arrays matching the mask count")
        if item.error is not None and not isinstance(item.error, str):
            raise ValueError("Object error must be a string or None")
        objects.append({**identity, "count": count, "error": item.error})
        arrays.extend((masks.tobytes(order="C"), boxes.astype("<f4", copy=False).tobytes(order="C"),
                       scores.astype("<f4", copy=False).tobytes(order="C")))
    header = _header("sam3_result", result.revision, objects, frame_size=len(frame),
                     mask_encoding="bool8", float_encoding="float32_le")
    return b"".join((header, frame, *arrays))


def decode_result(payload) -> Sam3Result:
    record, view, offset = _read_header(payload, "sam3_result")
    size = record.get("frame_size")
    if (type(size) is not int or size <= 0 or offset + size > len(view)
            or record.get("mask_encoding") != "bool8"
            or record.get("float_encoding") != "float32_le"):
        raise ValueError("Invalid source frame length or result encoding")
    frame = decode_frame(view[offset:offset + size])
    offset += size
    height, width = frame.rgb.shape[:2]
    for item in record["objects"]:
        if type(item.get("count")) is not int or item["count"] < 0:
            raise ValueError("Mask count must be a nonnegative integer")
        if item.get("error") is not None and not isinstance(item["error"], str):
            raise ValueError("Object error must be a string or None")
    expected = offset + sum(item["count"] * (height * width + 20) for item in record["objects"])
    if expected != len(view):
        raise ValueError("Result byte length does not match object mask counts")
    objects = []
    for item in record["objects"]:
        count = item["count"]
        masks = np.frombuffer(view, np.bool_, count * height * width, offset)
        masks = masks.reshape(count, height, width).copy()
        offset += count * height * width
        boxes = np.frombuffer(view, "<f4", count * 4, offset).reshape(count, 4).astype(np.float32, copy=True)
        offset += count * 16
        scores = np.frombuffer(view, "<f4", count, offset).astype(np.float32, copy=True)
        offset += count * 4
        objects.append(Sam3ObjectResult(item["object_id"], item["text"], masks, boxes, scores, item.get("error")))
    return Sam3Result(frame, record["revision"], tuple(objects))


class _BytePublisher:
    def __init__(self, service):
        iox2, self._node, self._service = _service(service)
        self._publisher = (self._service.publisher_builder().initial_max_slice_len(1)
                           .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo).create())

    def _publish(self, payload):
        if self._publisher is None:
            raise RuntimeError("SAM3 publisher is closed")
        sample = self._publisher.loan_slice_uninit(len(payload))
        ctypes.memmove(sample.payload().as_ptr(), payload, len(payload))
        return sample.assume_init().send()

    def close(self):
        if self._publisher is not None:
            self._publisher.delete()
        self._publisher = self._service = self._node = None


class _ByteSubscriber:
    def __init__(self, service):
        _iox2, self._node, self._service = _service(service)
        self._subscriber = self._service.subscriber_builder().buffer_size(2).create()

    def _receive(self, decoder):
        if self._subscriber is None:
            raise RuntimeError("SAM3 subscriber is closed")
        latest = None
        try:
            for _ in range(2):
                sample = self._subscriber.receive()
                if sample is None:
                    break
                if latest is not None:
                    latest.delete()
                latest = sample
            if latest is None:
                return None
            payload = latest.payload()
            data = (ctypes.c_uint8 * payload.len()).from_address(payload.as_ptr())
            return decoder(data)
        finally:
            if latest is not None:
                latest.delete()

    def close(self):
        if self._subscriber is not None:
            self._subscriber.delete()
        self._subscriber = self._service = self._node = None


class PromptPublisher(_BytePublisher):
    def publish(self, prompts: PromptSet):
        return self._publish(encode_prompts(prompts))

    def update_connections(self):
        """Deliver retained prompts to subscribers that connected after publish."""
        if self._publisher is None:
            raise RuntimeError("SAM3 publisher is closed")
        return self._publisher.update_connections()


class PromptSubscriber(_ByteSubscriber):
    def receive_latest(self) -> PromptSet | None:
        return self._receive(decode_prompts)


class Sam3Publisher(_BytePublisher):
    def publish(self, result: Sam3Result):
        return self._publish(encode_result(result))


class Sam3Subscriber(_ByteSubscriber):
    def receive_latest(self) -> Sam3Result | None:
        return self._receive(decode_result)
