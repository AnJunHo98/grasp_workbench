"""Pure raw-frame codec checks and a real, device-free two-process IPC smoke."""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame, decode_frame, encode_frame


def frame(height=3, width=4):
    return CameraFrame(
        np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3),
        np.arange(height * width, dtype=np.float32).reshape(height, width) / 10,
        {"camera_id": "test-camera", "sequence": 42, "stream_id": "capture-epoch",
         "capture_time_ns": 123456789, "timestamp_domain": "device_clock",
         "received_monotonic_ns": 987654321,
         "intrinsics": {"fx": 100.0, "fy": 101.0, "cx": 2.0, "cy": 1.5},
         "extra": {"description": "원본 프레임", "enabled": True}},
    )


@pytest.mark.parametrize("height,width", [(3, 4), (5, 7)])
def test_exact_raw_roundtrip_and_owned_arrays(height, width):
    original = frame(height, width)
    encoded = bytearray(encode_frame(original))
    metadata_size = struct.unpack_from("<I", encoded)[0]
    envelope = json.loads(encoded[4:4 + metadata_size])
    assert envelope == {
        "schema_version": 1, "width": width, "height": height,
        "rgb_encoding": "rgb8", "depth_encoding": "float32_le_m",
        "metadata": original.metadata,
    }
    assert len(encoded) == 4 + metadata_size + width * height * 7
    assert encoded[4 + metadata_size:] == original.rgb.tobytes() + original.depth_m.tobytes()
    decoded = decode_frame(encoded)
    encoded[:] = b"\x00" * len(encoded)
    np.testing.assert_array_equal(decoded.rgb, original.rgb)
    np.testing.assert_array_equal(decoded.depth_m, original.depth_m)
    assert decoded.metadata == original.metadata
    assert decoded.rgb.flags.owndata and decoded.depth_m.flags.owndata
    assert decoded.rgb.dtype == np.uint8 and decoded.depth_m.dtype == np.float32


def test_noncontiguous_and_big_endian_depth_are_sent_as_contiguous_little_endian():
    original = frame(5, 8)
    original.rgb = original.rgb[:, ::2]
    original.depth_m = original.depth_m[:, ::2].astype(">f4")
    original.depth_m[0, :3] = [np.nan, np.inf, -np.inf]
    decoded = decode_frame(encode_frame(original))
    np.testing.assert_array_equal(decoded.rgb, original.rgb)
    np.testing.assert_array_equal(decoded.depth_m, original.depth_m)
    assert decoded.rgb.flags.c_contiguous and decoded.depth_m.flags.c_contiguous


@pytest.mark.parametrize("field,value", [
    ("rgb", np.zeros((3, 4, 3), dtype=np.float32)),
    ("rgb", np.zeros((3, 4), dtype=np.uint8)),
    ("rgb", np.zeros((0, 4, 3), dtype=np.uint8)),
    ("depth_m", np.zeros((3, 4), dtype=np.float64)),
    ("depth_m", np.zeros((4, 3), dtype=np.float32)),
    ("metadata", []),
    ("metadata", {"non_json": object()}),
    ("metadata", {"non_json": float("nan")}),
])
def test_invalid_frame_is_not_encoded(field, value):
    original = frame()
    setattr(original, field, value)
    with pytest.raises((TypeError, ValueError)):
        encode_frame(original)


@pytest.mark.parametrize("payload", [b"", b"\x01\x00\x00", struct.pack("<I", 100) + b"{}"])
def test_truncated_header_or_metadata_is_rejected(payload):
    with pytest.raises(ValueError):
        decode_frame(payload)


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"width": 0}, {"height": False},
    {"width": "4"}, {"rgb_encoding": "bgr8"}, {"depth_encoding": "uint16_mm"},
    {"metadata": []},
])
def test_invalid_envelope_is_rejected(change):
    encoded = encode_frame(frame())
    size = struct.unpack_from("<I", encoded)[0]
    envelope = json.loads(encoded[4:4 + size])
    envelope.update(change)
    metadata = json.dumps(envelope).encode()
    modified = struct.pack("<I", len(metadata)) + metadata + encoded[4 + size:]
    with pytest.raises(ValueError):
        decode_frame(modified)


@pytest.mark.parametrize("suffix", [b"x", None])
def test_raw_payload_length_is_exact(suffix):
    encoded = encode_frame(frame())
    modified = encoded + suffix if suffix is not None else encoded[:-1]
    with pytest.raises(ValueError, match="byte length"):
        decode_frame(modified)


def test_codec_import_does_not_require_native_iceoryx2():
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.modules['iceoryx2'] = None
from grasp_workbench.ipc.camera import CameraFrame, encode_frame, decode_frame
assert 'iceoryx2' in sys.modules and sys.modules['iceoryx2'] is None
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_real_two_process_rgbd_publish_subscribe():
    pytest.importorskip("iceoryx2")
    root = Path(__file__).resolve().parents[1]
    name = f"camera-test/{uuid4().hex}"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    subscriber_script = """
import json, sys, time
import numpy as np
from grasp_workbench.ipc.camera import CameraSubscriber
subscriber = CameraSubscriber(sys.argv[1])
try:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = subscriber.receive_latest()
        if result is not None:
            np.testing.assert_array_equal(result.rgb, np.arange(36, dtype=np.uint8).reshape(3,4,3))
            np.testing.assert_array_equal(result.depth_m, np.full((3,4), .75, dtype=np.float32))
            print(json.dumps(result.metadata), flush=True)
            break
        time.sleep(.01)
    else:
        raise RuntimeError('No RGB-D sample received')
finally:
    subscriber.close()
"""
    publisher_script = """
import sys, time
import numpy as np
from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher
publisher = CameraPublisher(sys.argv[1])
frame = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3,4,3),
                    np.full((3,4), .75, dtype=np.float32),
                    {'camera_id':'synthetic', 'stream_id':'smoke', 'sequence':7,
                     'intrinsics':{'fx':100., 'fy':100., 'cx':2., 'cy':1.5}})
try:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if publisher.publish(frame) > 0:
            time.sleep(.1)
            break
        time.sleep(.02)
    else:
        raise RuntimeError('No subscriber connected')
finally:
    publisher.close()
"""
    subscriber = subprocess.Popen([sys.executable, "-c", subscriber_script, name], cwd=root,
                                  env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True)
    try:
        publisher = subprocess.run([sys.executable, "-c", publisher_script, name], cwd=root,
                                   env=environment, capture_output=True, text=True, timeout=15)
        output, error = subscriber.communicate(timeout=15)
        assert publisher.returncode == 0, publisher.stderr
        assert subscriber.returncode == 0, error
        metadata = json.loads(output)
        assert metadata["camera_id"] == "synthetic" and metadata["sequence"] == 7
        assert metadata["stream_id"] == "smoke"
    finally:
        if subscriber.poll() is None:
            subscriber.kill()
            subscriber.communicate()
