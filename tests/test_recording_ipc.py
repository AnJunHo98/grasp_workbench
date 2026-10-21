"""Recorder commands/progress use owned native iceoryx2 messages."""
import json
import time
from uuid import uuid4

import pytest

from grasp_workbench.ipc.recording import (
    RecordingControl, RecordingControlPublisher, RecordingControlSubscriber,
    RecordingStatus, RecordingStatusPublisher, RecordingStatusSubscriber,
    decode_control, decode_status, encode_control, encode_status, services,
)


@pytest.mark.parametrize("action", ["start", "save", "discard"])
def test_control_roundtrip(action):
    value = RecordingControl("request", action)
    assert decode_control(encode_control(value)) == value
    assert services("camera") == ("camera/recording/control", "camera/recording/status")


@pytest.mark.parametrize("state", ["ready", "recording", "saved", "discarded", "error"])
def test_status_roundtrip(state):
    value = RecordingStatus("request", state, "runs/datasets/run", 2, 3, 14,
                            "write failed" if state == "error" else None)
    payload = bytearray(encode_status(value))
    decoded = decode_status(payload)
    payload[:] = b"x" * len(payload)
    assert decoded == value


@pytest.mark.parametrize("value", [RecordingControl(None, "start"), RecordingControl("r", "bad"),
                                   RecordingControl("r", []), RecordingControl("r", None)])
def test_invalid_control(value):
    with pytest.raises(ValueError):
        encode_control(value)


@pytest.mark.parametrize("fields", [
    {"state": "bad"}, {"state": []}, {"output_dir": None}, {"request_id": 3},
    {"episode_index": -1}, {"episode_frames": True}, {"total_frames": 2.5}, {"error": 4},
])
def test_invalid_status(fields):
    with pytest.raises(ValueError):
        encode_status(RecordingStatus(**fields))


@pytest.mark.parametrize("decoder,kind", [(decode_control, "recording_control"),
                                          (decode_status, "recording_status")])
def test_rejects_wrong_or_truncated_schema(decoder, kind):
    for payload in (b"", b"{", b"[]", json.dumps({"schema_version": True, "kind": kind}).encode(),
                    json.dumps({"schema_version": 1, "kind": "wrong"}).encode(),
                    json.dumps({"schema_version": 1, "kind": kind}).encode()):
        with pytest.raises(ValueError):
            decoder(payload)


def test_native_control_status_and_late_subscriber_retained_ready():
    base = "test/recording/" + uuid4().hex
    control_service, status_service = services(base)
    control_pub = RecordingControlPublisher(control_service)
    control_sub = RecordingControlSubscriber(control_service)
    status_pub = RecordingStatusPublisher(status_service)
    status_sub = None
    try:
        ready = RecordingStatus(output_dir="runs/datasets/run")
        status_pub.publish(ready)
        status_sub = RecordingStatusSubscriber(status_service)

        def receive(endpoint, publisher):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                publisher.update_connections()
                value = endpoint.receive_latest()
                if value is not None:
                    return value
                time.sleep(.005)
            raise AssertionError("Native recording message was not received")

        assert receive(status_sub, status_pub) == ready
        command = RecordingControl("native-start", "start")
        control_pub.update_connections()
        control_pub.publish(command)
        assert receive(control_sub, control_pub) == command
        progress = RecordingStatus("native-start", "recording", "runs/datasets/run", 0, 4, 4)
        status_pub.publish(progress)
        assert receive(status_sub, status_pub) == progress
    finally:
        for endpoint in (status_sub, status_pub, control_sub, control_pub):
            if endpoint is not None:
                endpoint.close()
