"""Raw SAM3 codecs and device-free IPC; no SDK, model, VLM or camera access."""
import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.sam3 import (
    ObjectPrompt, PromptPublisher, PromptSet, PromptSubscriber, Sam3ObjectResult, Sam3Result, Sam3Subscriber,
    decode_prompts, decode_result, encode_prompts, encode_result, services,
)


def result():
    frame = CameraFrame(
        np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        np.array([[np.nan, np.inf, 0, 1], [.5, 2, 3, 4], [5, 6, 7, -np.inf]], dtype=np.float32),
        {"camera_id": "camera-source", "stream_id": "original-stream", "sequence": 37,
         "capture_time_ns": 1234567, "timestamp_domain": "original-device-clock",
         "received_monotonic_ns": 99887766, "intrinsics": {"fx": 100.0, "cx": 2.0},
         "extra": {"text": "원본 관측"}},
    )
    objects = (
        Sam3ObjectResult("물체/left ID", "red cup", np.arange(24).reshape(2, 3, 4) % 3 == 0,
                         np.array([[0, 0, 3, 2], [1, 1, 4, 3]], dtype=np.float32),
                         np.array([.9, .5], dtype=np.float32)),
        Sam3ObjectResult("B:any id", "blue bowl", np.empty((0, 3, 4), dtype=bool),
                         np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32),
                         "No candidate for this object"),
    )
    return Sam3Result(frame, "revision-α", objects)


def rewrite_header(payload, changes):
    length = struct.unpack_from("<I", payload)[0]
    header = json.loads(payload[4:4 + length])
    if callable(changes):
        changes(header)
    else:
        header.update(changes)
    encoded = json.dumps(header).encode()
    return struct.pack("<I", len(encoded)) + encoded + payload[4 + length:]


def assert_result_same(actual, original):
    assert actual.revision == original.revision
    assert actual.frame.metadata == original.frame.metadata
    np.testing.assert_array_equal(actual.frame.rgb, original.frame.rgb)
    np.testing.assert_array_equal(actual.frame.depth_m, original.frame.depth_m)
    assert len(actual.objects) == len(original.objects)
    for got, expected in zip(actual.objects, original.objects):
        assert (got.object_id, got.text, got.error) == (expected.object_id, expected.text, expected.error)
        for name in ("masks", "boxes", "scores"):
            np.testing.assert_array_equal(getattr(got, name), getattr(expected, name))
            assert getattr(got, name).flags.owndata
            assert getattr(got, name).flags.c_contiguous
        assert got.masks.dtype == np.bool_
        assert got.boxes.dtype == got.scores.dtype == np.float32


def test_service_names_and_empty_prompt_clear():
    assert services("camera/rgbd") == ("camera/rgbd/sam3/prompts", "camera/rgbd/sam3/results")
    prompts = PromptSet("clear-revision", ())
    assert decode_prompts(encode_prompts(prompts)) == prompts


def test_arbitrary_prompt_ids_text_and_revision_are_preserved():
    prompts = PromptSet("revision-α", (ObjectPrompt("ID / 한글", "  red cup  "),
                                     ObjectPrompt("B:not-an-enum", "blue bowl")))
    assert decode_prompts(encode_prompts(prompts)) == prompts


def test_result_exact_frame_all_candidates_empty_error_and_owned_arrays():
    original = result()
    payload = bytearray(encode_result(original))
    decoded = decode_result(payload)
    payload[:] = b"\x00" * len(payload)
    assert_result_same(decoded, original)
    assert decoded.frame.rgb.flags.owndata and decoded.frame.depth_m.flags.owndata
    assert decoded.objects[1].masks.shape == (0, 3, 4)


def test_noncontiguous_masks_and_float32_endianness_are_normalized():
    original = result()
    item = original.objects[0]
    item.masks = np.repeat(item.masks, 2, axis=2)[:, :, ::2]
    item.boxes = item.boxes.astype(">f4")
    item.scores = item.scores.astype(">f4")
    assert not item.masks.flags.c_contiguous
    assert_result_same(decode_result(encode_result(original)), original)


def test_result_with_no_objects_keeps_original_camera_frame():
    original = result()
    original.objects = ()
    assert_result_same(decode_result(encode_result(original)), original)


@pytest.mark.parametrize("field,value", [
    ("masks", np.ones((2, 3, 4), dtype=np.uint8)),
    ("masks", np.ones((2, 4, 3), dtype=bool)),
    ("masks", np.ones((3, 4), dtype=bool)),
    ("boxes", np.ones((2, 4), dtype=np.float64)),
    ("boxes", np.ones((1, 4), dtype=np.float32)),
    ("boxes", np.ones((2, 5), dtype=np.float32)),
    ("scores", np.ones(2, dtype=np.float64)),
    ("scores", np.ones((2, 1), dtype=np.float32)),
    ("scores", np.ones(1, dtype=np.float32)),
    ("error", {"not": "a string"}),
    ("object_id", 1), ("text", None),
])
def test_invalid_object_dtype_shape_count_or_metadata_is_rejected(field, value):
    original = result()
    setattr(original.objects[0], field, value)
    with pytest.raises(ValueError):
        encode_result(original)


@pytest.mark.parametrize("value", [1, None, {}])
def test_revision_must_be_string(value):
    with pytest.raises(ValueError):
        encode_prompts(PromptSet(value, ()))
    original = result()
    original.revision = value
    with pytest.raises(ValueError):
        encode_result(original)


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"kind": "other"},
    {"revision": 2}, {"objects": {}}, {"objects": [None]},
    {"objects": [{"object_id": "a", "text": None}]},
])
def test_invalid_prompt_header(change):
    encoded = encode_prompts(PromptSet("one", (ObjectPrompt("id", "cup"),)))
    with pytest.raises(ValueError):
        decode_prompts(rewrite_header(encoded, change))


@pytest.mark.parametrize("change", [
    {"frame_size": -1}, {"frame_size": True}, {"frame_size": 10**12},
    {"mask_encoding": "uint8"}, {"float_encoding": "float64"},
    lambda header: header["objects"][0].update(count=-1),
    lambda header: header["objects"][0].update(count=True),
    lambda header: header["objects"][0].update(count=100),
    lambda header: header["objects"][0].update(error=4),
])
def test_invalid_result_lengths_encoding_or_counts(change):
    with pytest.raises(ValueError):
        decode_result(rewrite_header(encode_result(result()), change))


@pytest.mark.parametrize("decoder", [decode_prompts, decode_result])
@pytest.mark.parametrize("payload", [b"", b"\x01\x00", struct.pack("<I", 100) + b"{}"])
def test_truncated_header_is_rejected(decoder, payload):
    with pytest.raises(ValueError):
        decoder(payload)


def test_trailing_truncated_and_wrong_message_kind_are_rejected():
    prompts = encode_prompts(PromptSet("one", ()))
    encoded = encode_result(result())
    for decoder, payload in ((decode_prompts, prompts + b"extra"),
                             (decode_result, encoded + b"extra"),
                             (decode_result, encoded[:-1]),
                             (decode_prompts, encoded), (decode_result, prompts)):
        with pytest.raises(ValueError):
            decoder(payload)


def test_subscriber_drains_only_two_and_deletes_native_samples_after_owned_decode():
    class Sample:
        def __init__(self, sequence):
            value = result()
            value.frame.metadata["sequence"] = sequence
            payload = encode_result(value)
            self.data = ctypes.create_string_buffer(payload)
            self.size, self.deleted = len(payload), False

        def payload(self):
            return SimpleNamespace(len=lambda: self.size, as_ptr=lambda: ctypes.addressof(self.data))

        def delete(self):
            self.deleted = True
            ctypes.memset(ctypes.addressof(self.data), 0, self.size)

    samples = [Sample(index) for index in range(4)]
    queue = list(samples)
    subscriber = Sam3Subscriber.__new__(Sam3Subscriber)
    subscriber._subscriber = SimpleNamespace(receive=lambda: queue.pop(0), delete=lambda: None)
    decoded = subscriber.receive_latest()
    assert decoded.frame.metadata["sequence"] == 1
    assert len(queue) == 2 and samples[0].deleted and samples[1].deleted
    assert decoded.frame.rgb[0, 0, 1] == 1
    assert decoded.objects[0].masks.any()
    subscriber.close()
    subscriber.close()
    with pytest.raises(RuntimeError, match="closed"):
        subscriber.receive_latest()


def test_codecs_import_without_native_sdk_or_model_packages():
    script = """
import sys
for name in ('iceoryx2', 'pyzed.sl', 'pyrealsense2', 'torch'):
    sys.modules[name] = None
from grasp_workbench.ipc.sam3 import encode_prompts, decode_prompts, PromptSet
assert decode_prompts(encode_prompts(PromptSet('clear', ()))).objects == ()
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr


def test_prompt_connection_refresh_calls_native_and_rejects_closed_publisher():
    calls = []
    publisher = PromptPublisher.__new__(PromptPublisher)
    publisher._publisher = SimpleNamespace(update_connections=lambda: calls.append("update"),
                                           delete=lambda: calls.append("delete"))
    publisher.update_connections()
    assert calls == ["update"]
    publisher.close()
    with pytest.raises(RuntimeError, match="closed"):
        publisher.update_connections()
    assert calls == ["update", "delete"]


def test_native_prompt_history_reaches_late_subscriber_after_connection_refresh():
    pytest.importorskip("iceoryx2")
    service = f"sam3-late-prompts/{uuid4().hex}"
    publisher, subscriber = PromptPublisher(service), None
    prompts = PromptSet("published-before-subscriber", (ObjectPrompt("A/any-ID", "red cup"),))
    try:
        assert publisher.publish(prompts) == 0
        subscriber = PromptSubscriber(service)
        assert subscriber.receive_latest() is None
        deadline = time.monotonic() + 2
        received = None
        while received is None and time.monotonic() < deadline:
            publisher.update_connections()
            received = subscriber.receive_latest()
            time.sleep(.005)
        assert received == prompts
    finally:
        if subscriber is not None:
            subscriber.close()
        publisher.close()


def test_native_two_process_prompt_and_result_roundtrip():
    pytest.importorskip("iceoryx2")
    root = Path(__file__).resolve().parents[1]
    service = f"sam3-test/{uuid4().hex}"
    environment = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    receiver_script = """
import json, sys, time
import numpy as np
from grasp_workbench.ipc.sam3 import PromptSubscriber, Sam3Subscriber, services
prompts_service, result_service = services(sys.argv[1])
prompts, results = PromptSubscriber(prompts_service), Sam3Subscriber(result_service)
try:
    p = r = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (p is None or r is None):
        p = prompts.receive_latest() or p
        r = results.receive_latest() or r
        time.sleep(.005)
    assert p.revision == r.revision == 'native-revision'
    assert p.objects[0].object_id == r.objects[0].object_id == 'arbitrary/ID'
    assert p.objects[0].text == r.objects[0].text == 'red cup'
    assert r.frame.metadata == {'camera_id':'original', 'stream_id':'source-stream',
        'sequence':39, 'capture_time_ns':123456, 'timestamp_domain':'device',
        'received_monotonic_ns':987654, 'intrinsics':{'fx':100., 'cx':2.}}
    np.testing.assert_array_equal(r.frame.rgb, np.arange(36,dtype=np.uint8).reshape(3,4,3))
    np.testing.assert_array_equal(r.frame.depth_m, np.full((3,4),.75,dtype=np.float32))
    np.testing.assert_array_equal(r.objects[0].masks, np.ones((1,3,4),dtype=bool))
    np.testing.assert_array_equal(r.objects[0].boxes, [[0,0,4,3]])
    np.testing.assert_array_equal(r.objects[0].scores, [.75])
finally:
    prompts.close(); results.close()
assert r.frame.rgb.flags.owndata and r.objects[0].masks.flags.owndata
assert r.frame.rgb[0,0,1] == 1 and r.objects[0].masks.all()
print(json.dumps({'revision':r.revision, 'source_sequence':r.frame.metadata['sequence']}))
"""
    publisher_script = """
import sys,time
import numpy as np
from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.sam3 import *
prompt_name,result_name=services(sys.argv[1])
pub, out=PromptPublisher(prompt_name),Sam3Publisher(result_name)
prompts=PromptSet('native-revision',(ObjectPrompt('arbitrary/ID','red cup'),))
frame=CameraFrame(np.arange(36,dtype=np.uint8).reshape(3,4,3),
    np.full((3,4),.75,dtype=np.float32),
    {'camera_id':'original','stream_id':'source-stream','sequence':39,'capture_time_ns':123456,
     'timestamp_domain':'device','received_monotonic_ns':987654,'intrinsics':{'fx':100.,'cx':2.}})
result=Sam3Result(frame,prompts.revision,(Sam3ObjectResult('arbitrary/ID','red cup',
    np.ones((1,3,4),dtype=bool),np.array([[0,0,4,3]],dtype=np.float32),np.array([.75],dtype=np.float32)),))
try:
    deadline=time.monotonic()+10
    while True:
        a,b=pub.publish(prompts),out.publish(result)
        if a and b: break
        if time.monotonic()>deadline: raise RuntimeError('No matching SAM3 subscribers')
        time.sleep(.02)
    time.sleep(.15)
finally:
    pub.close();out.close()
"""
    receiver = subprocess.Popen([sys.executable, "-c", receiver_script, service], cwd=root,
                                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        publisher = subprocess.run([sys.executable, "-c", publisher_script, service], cwd=root,
                                   env=environment, capture_output=True, text=True, timeout=15)
        output, error = receiver.communicate(timeout=15)
        assert publisher.returncode == 0, publisher.stderr
        assert receiver.returncode == 0, error
        assert json.loads(output) == {"revision": "native-revision", "source_sequence": 39}
    finally:
        if receiver.poll() is None:
            receiver.kill()
            receiver.communicate()
