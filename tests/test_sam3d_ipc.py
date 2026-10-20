"""Raw geometry/selection codecs and native transport without model imports."""
from dataclasses import replace
import json
import struct
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.sam3d import (
    MeshGeometry, MeshRequest, MeshRequestPublisher, MeshRequestSubscriber,
    MeshResult, MeshResultPublisher, MeshResultSubscriber, decode_request,
    decode_result, encode_request, encode_result, services,
)


def request():
    frame = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
                        np.array([[np.nan, np.inf, 0, 1], [2, 3, 4, 5], [6, 7, 8, -np.inf]], np.float32),
                        {"camera_id": "original-camera", "stream_id": "stream-α", "sequence": 19,
                         "capture_time_ns": 123, "received_monotonic_ns": 456,
                         "timestamp_domain": "device-clock", "intrinsics": {"fx": 100, "cx": 2}})
    return MeshRequest("job/요청", "asset/../A", "object-한글", "revision-α", "a red cup", 2,
                       frame, np.arange(12).reshape(3, 4) % 2 == 0)


def result(colors=True):
    mesh = MeshGeometry(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                        np.array([[0, 1, 2]], np.uint32),
                        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8) if colors else None)
    return MeshResult("job/요청", "asset/../A", "object-한글", "revision-α", "done",
                      mesh, "/output/asset.ply", mesh_scale_to_m=1., length_unit="metre",
                      camera_from_object=np.array([[0., -1, 0, .1], [1, 0, 0, -.2],
                                                   [0, 0, 1, .75], [0, 0, 0, 1]]))


def rewrite_header(payload, changes):
    size = struct.unpack_from("<I", payload)[0]
    record = json.loads(payload[4:4 + size])
    record.update(changes)
    metadata = json.dumps(record).encode()
    return struct.pack("<I", len(metadata)) + metadata + payload[4 + size:]


def assert_request(actual, expected):
    for name in ("request_id", "asset_id", "object_id", "revision", "prompt", "candidate_index",
                 "depth_source", "texture_baking"):
        assert getattr(actual, name) == getattr(expected, name)
    assert actual.frame.metadata == expected.frame.metadata
    for array, original in ((actual.frame.rgb, expected.frame.rgb),
                            (actual.frame.depth_m, expected.frame.depth_m), (actual.mask, expected.mask)):
        np.testing.assert_array_equal(array, original)
        assert array.flags.owndata and array.flags.c_contiguous


def assert_result(actual, expected):
    for name in ("request_id", "asset_id", "object_id", "revision", "status", "mesh_path", "error",
                 "mesh_scale_to_m", "length_unit"):
        assert getattr(actual, name) == getattr(expected, name)
    if expected.camera_from_object is None:
        assert actual.camera_from_object is None
    else:
        np.testing.assert_array_equal(actual.camera_from_object, expected.camera_from_object)
        assert actual.camera_from_object.flags.owndata
    if expected.mesh is None:
        assert actual.mesh is None
        return
    for name in ("vertices", "faces", "colors"):
        array, original = getattr(actual.mesh, name), getattr(expected.mesh, name)
        if original is None:
            assert array is None
        else:
            np.testing.assert_array_equal(array, original)
            assert array.flags.owndata and array.flags.c_contiguous
    assert actual.mesh.vertices.dtype == np.float32
    assert actual.mesh.faces.dtype == np.uint32


def test_service_names_request_exact_frame_mask_and_owned_buffers():
    assert services("example/rgbd") == ("example/rgbd/sam3d/requests", "example/rgbd/sam3d/results")
    original = request()
    payload = bytearray(encode_request(original))
    decoded = decode_request(payload)
    payload[:] = b"\x00" * len(payload)
    assert_request(decoded, original)
    assert decoded.mask.dtype == np.bool_


def test_noncontiguous_mask_and_empty_foreground_are_representable():
    original = request()
    original.mask = np.repeat(original.mask, 2, axis=1)[:, ::2]
    assert not original.mask.flags.c_contiguous
    assert_request(decode_request(encode_request(original)), original)
    original.mask[:] = False
    assert not decode_request(encode_request(original)).mask.any()


@pytest.mark.parametrize("source,texture", [("depth", False), ("depth", True), ("monocular", False), ("monocular", True)])
def test_source_texture_options_round_trip(source, texture):
    original = replace(request(), depth_source=source, texture_baking=texture)
    assert_request(decode_request(encode_request(original)), original)


@pytest.mark.parametrize("field,value", [("depth_source", "guessed"), ("texture_baking", 1)])
def test_source_texture_options_validate_wire_types(field, value):
    original = request()
    setattr(original, field, value)
    with pytest.raises(ValueError):
        encode_request(original)


@pytest.mark.parametrize("colors", [False, True])
def test_result_exact_geometry_and_owned_buffers(colors):
    original = result(colors)
    payload = bytearray(encode_result(original))
    decoded = decode_result(payload)
    payload[:] = b"\x00" * len(payload)
    assert_result(decoded, original)


def test_noncontiguous_and_big_endian_geometry_are_normalized():
    original = result()
    original.mesh.vertices = np.repeat(original.mesh.vertices, 2, axis=1)[:, ::2].astype(">f4")
    original.mesh.faces = original.mesh.faces.astype(">u4")
    original.mesh.colors = np.repeat(original.mesh.colors, 2, axis=1)[:, ::2]
    assert_result(decode_result(encode_result(original)), original)


def test_optional_metric_metadata_does_not_change_legacy_geometry_only_results():
    original = replace(result(), camera_from_object=None, length_unit=None)
    assert_result(decode_result(encode_result(original)), original)


@pytest.mark.parametrize("status,error", [("running", None), ("error", "RuntimeError: one object failed")])
def test_running_and_error_have_no_geometry(status, error):
    original = replace(result(), status=status, mesh=None, mesh_path=None, error=error)
    assert_result(decode_result(encode_result(original)), original)


@pytest.mark.parametrize("field,value", [
    ("request_id", 1), ("asset_id", None), ("object_id", []), ("revision", False),
    ("prompt", None), ("candidate_index", -1), ("candidate_index", True), ("candidate_index", 0.5),
    ("mask", np.ones((3, 4), np.uint8)), ("mask", np.ones((4, 3), bool)),
])
def test_invalid_request_rejected(field, value):
    original = request()
    setattr(original, field, value)
    with pytest.raises(ValueError):
        encode_request(original)


@pytest.mark.parametrize("field,value", [
    ("vertices", np.ones((3, 3), np.float64)), ("vertices", np.ones((3, 2), np.float32)),
    ("vertices", np.empty((0, 3), np.float32)), ("vertices", np.full((3, 3), np.nan, np.float32)),
    ("faces", np.array([[0, 1, 2]], np.int32)), ("faces", np.array([[0, 1, 3]], np.uint32)),
    ("faces", np.empty((0, 3), np.uint32)), ("faces", np.ones((1, 2), np.uint32)),
    ("colors", np.ones((3, 3), np.float32)), ("colors", np.ones((2, 3), np.uint8)),
])
def test_invalid_geometry_rejected(field, value):
    original = result()
    setattr(original.mesh, field, value)
    with pytest.raises(ValueError):
        encode_result(original)


@pytest.mark.parametrize("changes", [
    {"status": "pending"}, {"status": "error"}, {"status": "running"},
    {"mesh": None}, {"mesh_path": 2}, {"error": {}}, {"asset_id": 1},
    {"length_unit": 1}, {"camera_from_object": np.eye(3)},
    {"camera_from_object": np.full((4, 4), np.nan)},
])
def test_invalid_result_states_and_metadata_rejected(changes):
    with pytest.raises(ValueError):
        encode_result(replace(result(), **changes))


@pytest.mark.parametrize("decoder,encoder,value", [
    (decode_request, encode_request, request), (decode_result, encode_result, result),
])
def test_truncation_extraneous_bytes_and_unknown_schema_rejected(decoder, encoder, value):
    payload = encoder(value())
    for bad in (b"", b"\x00", payload[:7], payload[:-1], payload + b"extra",
                rewrite_header(payload, {"schema_version": 2}),
                rewrite_header(payload, {"schema_version": True}),
                rewrite_header(payload, {"asset_id": None})):
        with pytest.raises(ValueError):
            decoder(bad)


@pytest.mark.parametrize("changes", [
    {"frame_size": -1}, {"frame_size": True}, {"mask_encoding": "png"},
    {"candidate_index": -1}, {"candidate_index": True}, {"prompt": 1},
])
def test_invalid_request_wire_metadata_rejected(changes):
    with pytest.raises(ValueError):
        decode_request(rewrite_header(encode_request(request()), changes))


@pytest.mark.parametrize("changes", [
    {"vertex_count": -1}, {"vertex_count": True}, {"vertex_count": 0}, {"face_count": 0},
    {"has_colors": 1}, {"vertex_encoding": "float64_le"}, {"face_encoding": "int32_le"},
    {"color_encoding": "bgr8"}, {"status": "error"}, {"status": "unknown"}, {"error": 4},
    {"length_unit": 1}, {"camera_from_object": [[1, 0], [0, 1]]},
])
def test_invalid_result_wire_metadata_rejected(changes):
    with pytest.raises(ValueError):
        decode_result(rewrite_header(encode_result(result()), changes))


def test_corrupt_wire_coordinates_and_indices_are_rejected():
    payload = bytearray(encode_result(result()))
    offset = 4 + struct.unpack_from("<I", payload)[0]
    struct.pack_into("<f", payload, offset, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        decode_result(payload)
    payload = bytearray(encode_result(result()))
    struct.pack_into("<I", payload, offset + 3 * 3 * 4, 3)
    with pytest.raises(ValueError, match="indices"):
        decode_result(payload)


@pytest.mark.parametrize("publisher_type,subscriber_type,value,assert_same", [
    (MeshRequestPublisher, MeshRequestSubscriber, request, assert_request),
    (MeshResultPublisher, MeshResultSubscriber, result, assert_result),
])
def test_native_late_join_retained_data_refresh_ownership_and_close(
    publisher_type, subscriber_type, value, assert_same
):
    pytest.importorskip("iceoryx2")
    service = f"sam3d-codec/{uuid4().hex}"
    publisher = publisher_type(service)
    subscriber = None
    try:
        original = value()
        assert publisher.publish(original) == 0
        # A subscriber joining after the only publication needs refresh, not a
        # second application request. All decoded arrays outlive the SDK loan.
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
    with pytest.raises(RuntimeError):
        publisher.publish(original)
    with pytest.raises(RuntimeError):
        subscriber.receive_latest()
