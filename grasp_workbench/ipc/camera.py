"""Raw RGB-D frames over iceoryx2; no camera SDK or model imports."""

from __future__ import annotations

import ctypes
import importlib
import json
import struct
from dataclasses import dataclass

import numpy as np

_LENGTH = struct.Struct("<I")


@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    metadata: dict


def _parts(frame: CameraFrame):
    rgb, depth = np.asarray(frame.rgb), np.asarray(frame.depth_m)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must be a uint8 HxWx3 array")
    height, width = rgb.shape[:2]
    if not height or not width:
        raise ValueError("Image dimensions must be positive")
    if depth.dtype.kind != "f" or depth.dtype.itemsize != 4 or depth.shape != (height, width):
        raise ValueError("Depth must be a float32 HxW array matching RGB")
    if not isinstance(frame.metadata, dict):
        raise TypeError("Frame metadata must be a JSON object")
    envelope = {
        "schema_version": 1, "width": width, "height": height,
        "rgb_encoding": "rgb8", "depth_encoding": "float32_le_m",
        "metadata": frame.metadata,
    }
    metadata = json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return (_LENGTH.pack(len(metadata)) + metadata,
            np.ascontiguousarray(rgb), np.ascontiguousarray(depth, dtype="<f4"))


def encode_frame(frame: CameraFrame) -> bytes:
    """Pure codec helper. Native publication fills its loan directly instead."""
    header, rgb, depth = _parts(frame)
    return b"".join((header, memoryview(rgb).cast("B"), memoryview(depth).cast("B")))


def decode_frame(payload) -> CameraFrame:
    """Decode into owned arrays, which remain valid after an IPC sample is released."""
    view = memoryview(payload).cast("B")
    if len(view) < _LENGTH.size:
        raise ValueError("Truncated frame header")
    metadata_size = _LENGTH.unpack_from(view)[0]
    offset = _LENGTH.size + metadata_size
    if offset > len(view):
        raise ValueError("Truncated frame metadata")
    envelope = json.loads(bytes(view[_LENGTH.size:offset]))
    if not isinstance(envelope, dict) or (
        type(envelope.get("schema_version")) is not int or envelope["schema_version"] != 1
        or envelope.get("rgb_encoding") != "rgb8"
        or envelope.get("depth_encoding") != "float32_le_m"
    ):
        raise ValueError("Unsupported frame schema or encoding")
    width, height = envelope.get("width"), envelope.get("height")
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Invalid frame dimensions")
    if not isinstance(envelope.get("metadata"), dict):
        raise ValueError("Frame metadata must be a JSON object")
    rgb_size = width * height * 3
    if len(view) != offset + width * height * 7:
        raise ValueError("Frame byte length does not match image dimensions")
    rgb = np.frombuffer(view, np.uint8, width * height * 3, offset).reshape(height, width, 3).copy()
    depth = np.frombuffer(view, "<f4", width * height, offset + rgb_size)
    return CameraFrame(rgb, depth.reshape(height, width).astype(np.float32, copy=True),
                       envelope["metadata"])


def _service(service_name):
    iox2 = importlib.import_module("iceoryx2")
    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
    service = (
        node.service_builder(iox2.ServiceName.new(service_name))
        .publish_subscribe(iox2.Slice[ctypes.c_uint8])
        .history_size(1).subscriber_max_buffer_size(2).enable_safe_overflow(True)
        .open_or_create()
    )
    return iox2, node, service


class CameraPublisher:
    def __init__(self, service_name: str):
        iox2, self._node, self._service = _service(service_name)
        self._publisher = (
            self._service.publisher_builder().initial_max_slice_len(1)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo).create()
        )

    def publish(self, frame: CameraFrame) -> int:
        """Copy capture arrays into an exact-sized loan; return delivered receivers."""
        if self._publisher is None:
            raise RuntimeError("Camera publisher is closed")
        header, rgb, depth = _parts(frame)
        sample = self._publisher.loan_slice_uninit(len(header) + rgb.nbytes + depth.nbytes)
        address = sample.payload().as_ptr()
        ctypes.memmove(address, header, len(header))
        ctypes.memmove(address + len(header), rgb.ctypes.data, rgb.nbytes)
        ctypes.memmove(address + len(header) + rgb.nbytes, depth.ctypes.data, depth.nbytes)
        return sample.assume_init().send()

    def close(self):
        if self._publisher is not None:
            self._publisher.delete()
        self._publisher = self._service = self._node = None


class CameraSubscriber:
    def __init__(self, service_name: str):
        _iox2, self._node, self._service = _service(service_name)
        self._subscriber = self._service.subscriber_builder().buffer_size(2).create()

    def receive_latest(self) -> CameraFrame | None:
        if self._subscriber is None:
            raise RuntimeError("Camera subscriber is closed")
        latest = None
        # Bound work even while a fast publisher continues producing frames.
        for _ in range(2):
            sample = self._subscriber.receive()
            if sample is None:
                break
            if latest is not None:
                latest.delete()
            latest = sample
        if latest is None:
            return None
        try:
            payload = latest.payload()
            data = (ctypes.c_uint8 * payload.len()).from_address(payload.as_ptr())
            return decode_frame(data)
        finally:
            latest.delete()

    def close(self):
        if self._subscriber is not None:
            self._subscriber.delete()
        self._subscriber = self._service = self._node = None
