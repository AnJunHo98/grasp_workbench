"""Small recorder controls and progress messages over iceoryx2."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .sam3 import _BytePublisher, _ByteSubscriber

_ACTIONS = {"start", "save", "discard"}
_STATES = {"ready", "recording", "saved", "discarded", "error"}


@dataclass
class RecordingControl:
    request_id: str
    action: str


@dataclass
class RecordingStatus:
    request_id: str = ""
    state: str = "ready"
    output_dir: str = ""
    episode_index: int = 0
    episode_frames: int = 0
    total_frames: int = 0
    error: str | None = None


def services(base):
    return base + "/recording/control", base + "/recording/status"


def _encode(kind, value):
    return json.dumps({"schema_version": 1, "kind": kind, **asdict(value)},
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read(payload, kind):
    record = json.loads(bytes(payload))
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or record.get("kind") != kind):
        raise ValueError("Unsupported recording schema")
    return record


def _control(value):
    if (not isinstance(value.request_id, str)
            or not isinstance(value.action, str) or value.action not in _ACTIONS):
        raise ValueError("Recording control needs a string request ID and start/save/discard action")


def _status(value):
    if (not isinstance(value.request_id, str) or not isinstance(value.output_dir, str)
            or not isinstance(value.state, str) or value.state not in _STATES
            or value.error is not None and not isinstance(value.error, str)):
        raise ValueError("Invalid recording status")
    if any(type(getattr(value, key)) is not int or getattr(value, key) < 0
           for key in ("episode_index", "episode_frames", "total_frames")):
        raise ValueError("Recording frame/episode counts must be nonnegative integers")


def encode_control(control: RecordingControl) -> bytes:
    _control(control)
    return _encode("recording_control", control)


def decode_control(payload) -> RecordingControl:
    record = _read(payload, "recording_control")
    control = RecordingControl(record.get("request_id"), record.get("action"))
    _control(control)
    return control


def encode_status(status: RecordingStatus) -> bytes:
    _status(status)
    return _encode("recording_status", status)


def decode_status(payload) -> RecordingStatus:
    record = _read(payload, "recording_status")
    status = RecordingStatus(**{key: record.get(key) for key in RecordingStatus.__dataclass_fields__})
    _status(status)
    return status


class _RecordingPublisher(_BytePublisher):
    def update_connections(self):
        if self._publisher is None:
            raise RuntimeError("Recording publisher is closed")
        return self._publisher.update_connections()


class RecordingControlPublisher(_RecordingPublisher):
    def publish(self, control: RecordingControl):
        return self._publish(encode_control(control))


class RecordingControlSubscriber(_ByteSubscriber):
    def receive_latest(self) -> RecordingControl | None:
        return self._receive(decode_control)


class RecordingStatusPublisher(_RecordingPublisher):
    def publish(self, status: RecordingStatus):
        return self._publish(encode_status(status))


class RecordingStatusSubscriber(_ByteSubscriber):
    def receive_latest(self) -> RecordingStatus | None:
        return self._receive(decode_status)
