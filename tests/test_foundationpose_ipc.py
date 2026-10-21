"""FoundationPose controls/results preserve raw source frames and owned arrays."""
from dataclasses import replace
import json
import struct
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.foundationpose import (
    ObjectPose, PoseControl, PoseControlPublisher, PoseControlSubscriber,
    PoseResult, PoseResultPublisher, PoseResultSubscriber, PoseTarget,
    decode_control, decode_result, encode_control, encode_result, services,
)


def target(object_id="A"):
    frame = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
                        np.array([[np.nan, np.inf, 0, 1], [2, 3, 4, 5], [6, 7, 8, 9]], np.float32),
                        {"camera_id": "camera-α", "sequence": 7,
                         "intrinsics": {"fx": 100, "fy": 101, "cx": 2, "cy": 1}})
    return PoseTarget(object_id, "session-" + object_id, "/meshes/물체.glb", 0.001,
                      frame, np.arange(12).reshape(3, 4) % 2 == 0, "revision-α")


def control():
    return PoseControl((target(), target("B")))


def result():
    return PoseResult(target().frame, (
        ObjectPose("A", "session-A", "/meshes/물체.glb", "tracking", np.eye(4)),
        ObjectPose("B", "session-B", "/meshes/b.ply", "error", error="RuntimeError: failed"),
    ))


def assert_frame(actual, expected):
    assert actual.metadata == expected.metadata
    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    np.testing.assert_array_equal(actual.depth_m, expected.depth_m)
    assert actual.rgb.flags.owndata and actual.depth_m.flags.owndata


def assert_control(actual, expected):
    assert len(actual.targets) == len(expected.targets)
    for decoded, original in zip(actual.targets, expected.targets):
        for name in ("object_id", "session_id", "mesh_path", "mesh_scale_to_m", "prompt_revision"):
            assert getattr(decoded, name) == getattr(original, name)
        assert_frame(decoded.frame, original.frame)
        np.testing.assert_array_equal(decoded.mask, original.mask)
        assert decoded.mask.dtype == np.bool_ and decoded.mask.flags.owndata


def assert_result(actual, expected):
    assert_frame(actual.frame, expected.frame)
    assert len(actual.objects) == len(expected.objects)
    for decoded, original in zip(actual.objects, expected.objects):
        for name in ("object_id", "session_id", "mesh_path", "status", "error", "visual_mismatch_count", "pose_source", "diagnostics"):
            assert getattr(decoded, name) == getattr(original, name)
        if original.camera_from_object is None:
            assert decoded.camera_from_object is None
        else:
            np.testing.assert_array_equal(decoded.camera_from_object, original.camera_from_object)
            assert decoded.camera_from_object.flags.owndata


def rewrite_header(payload, changes):
    size = struct.unpack_from("<I", payload)[0]
    record = json.loads(payload[4:4 + size])
    record.update(changes)
    metadata = json.dumps(record).encode()
    return struct.pack("<I", len(metadata)) + metadata + payload[4 + size:]


def test_control_exact_frames_masks_a_b_and_owned_arrays():
    assert services("example/rgbd") == (
        "example/rgbd/foundationpose/control", "example/rgbd/foundationpose/results")
    original = control()
    original.targets[1].frame.metadata["sequence"] = 8
    payload = bytearray(encode_control(original))
    decoded = decode_control(payload)
    payload[:] = b"\x00" * len(payload)
    assert_control(decoded, original)


def test_empty_control_and_noncontiguous_or_empty_mask_are_representable():
    assert decode_control(encode_control(PoseControl(()))).targets == ()
    item = target()
    item.mask = np.repeat(item.mask, 2, axis=1)[:, ::2]
    assert not item.mask.flags.c_contiguous
    assert_control(decode_control(encode_control(PoseControl((item,)))), PoseControl((item,)))
    item.mask[:] = False
    assert not decode_control(encode_control(PoseControl((item,)))).targets[0].mask.any()


def test_result_exact_frame_pose_and_error_without_geometry_quality_policy():
    original = result()
    # Wire format checks only the matrix shape, not rigidity, jump or confidence.
    original.objects[0].camera_from_object = np.arange(16, dtype=np.float32).reshape(4, 4)
    payload = bytearray(encode_result(original))
    decoded = decode_result(payload)
    payload[:] = b"\x00" * len(payload)
    assert_result(decoded, original)


def test_aprilcube_source_uses_the_same_pose_result_contract():
    original = result()
    original.objects[0].pose_source = "aprilcube"
    decoded = decode_result(encode_result(original))
    assert_result(decoded, original)
    assert decoded.objects[0].pose_source == "aprilcube"
    assert decoded.objects[1].pose_source == "foundationpose"


def test_pose_result_preserves_optional_sdk_diagnostics_without_changing_status():
    original = result()
    original.objects[0].diagnostics = {"n_tags": 0, "predicted": True,
                                       "tag_ids": [], "reproj_error": None}
    decoded = decode_result(encode_result(original))
    assert_result(decoded, original)
    assert decoded.objects[0].status == "tracking"
    assert decoded.objects[1].diagnostics is None


def test_legacy_pose_result_without_source_defaults_to_foundationpose():
    payload = encode_result(result())
    size = struct.unpack_from("<I", payload)[0]
    objects = json.loads(payload[4:4 + size])["objects"]
    for item in objects:
        del item["pose_source"]
        del item["diagnostics"]
    decoded = decode_result(rewrite_header(payload, {"objects": objects}))
    assert all(item.pose_source == "foundationpose" for item in decoded.objects)
    assert all(item.diagnostics is None for item in decoded.objects)


@pytest.mark.parametrize("field,value", [
    ("object_id", 1), ("session_id", None), ("mesh_path", []), ("mesh_scale_to_m", "metres"),
    ("mask", np.ones((3, 4), np.uint8)), ("mask", np.ones((4, 3), bool)),
])
def test_invalid_control_wire_fields_rejected(field, value):
    item = target()
    setattr(item, field, value)
    with pytest.raises(ValueError):
        encode_control(PoseControl((item,)))


@pytest.mark.parametrize("changes", [
    {"camera_from_object": np.eye(3)}, {"camera_from_object": np.full((4, 4), "bad")},
    {"status": None}, {"error": {}}, {"pose_source": None}, {"diagnostics": []},
])
def test_invalid_result_wire_fields_rejected(changes):
    original = result()
    original.objects = (replace(original.objects[0], **changes),)
    with pytest.raises(ValueError):
        encode_result(original)


@pytest.mark.parametrize("decoder,encoder,value", [
    (decode_control, encode_control, control), (decode_result, encode_result, result),
])
def test_truncation_extra_bytes_and_unknown_schema_rejected(decoder, encoder, value):
    payload = encoder(value())
    for bad in (b"", b"\x00", payload[:8], payload[:-1], payload + b"extra",
                rewrite_header(payload, {"schema_version": 2}),
                rewrite_header(payload, {"schema_version": True}),
                rewrite_header(payload, {"objects": None})):
        with pytest.raises(ValueError):
            decoder(bad)


@pytest.mark.parametrize("publisher_type,subscriber_type,value,assert_same", [
    (PoseControlPublisher, PoseControlSubscriber, control, assert_control),
    (PoseResultPublisher, PoseResultSubscriber, result, assert_result),
])
def test_native_retained_late_join_and_arrays_outlive_endpoints(
    publisher_type, subscriber_type, value, assert_same
):
    pytest.importorskip("iceoryx2")
    service = f"foundationpose-codec/{uuid4().hex}"
    publisher = publisher_type(service)
    subscriber = None
    try:
        original = value()
        assert publisher.publish(original) == 0
        subscriber = subscriber_type(service)
        deadline, decoded = time.monotonic() + 3, None
        while time.monotonic() < deadline and decoded is None:
            publisher.update_connections()
            decoded = subscriber.receive_latest()
            if decoded is None:
                time.sleep(0.01)
        assert decoded is not None
        assert_same(decoded, original)
        assert subscriber.receive_latest() is None
    finally:
        if subscriber is not None:
            subscriber.close()
            subscriber.close()
        publisher.close()
        publisher.close()
    assert_same(decoded, original)
    with pytest.raises(RuntimeError):
        publisher.update_connections()
