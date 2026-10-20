"""Pure grasp codecs: no model, GPU, robot or camera is required."""
from dataclasses import replace
import json
import struct

import numpy as np
import pytest

from grasp_workbench.ipc.graspgenx import (
    GraspRequest, GraspResult, decode_request, decode_result, encode_request, encode_result, services,
)


def result():
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, 0, 3] = [.1, .2, .3]
    return GraspResult("job", "asset/α", "A", "done", "objects/object.ply", .001, "franka_panda",
                        poses, np.array([.9, .8, .7], np.float64), "grasps/bundle.json",
                        {"control_points": [[0., .1, .2]], "offset": .05}, cached=True)


def test_request_roundtrip_preserves_identity_scale_and_regenerate():
    request = GraspRequest("job", "asset/α", "B", "/some/object.glb", .001, True)
    assert decode_request(encode_request(request)) == request
    assert services("example") == ("example/graspgenx/requests", "example/graspgenx/results")


def test_raw_float64_result_roundtrip_owned_arrays_and_metadata():
    original = result()
    payload = bytearray(encode_result(original))
    size = struct.unpack_from("<I", payload)[0]
    header = json.loads(payload[4:4 + size])
    assert header["float_encoding"] == "float64_le"
    assert len(payload) == 4 + size + 3 * 17 * 8
    decoded = decode_result(payload)
    for key in ("request_id", "asset_id", "object_id", "status", "mesh_path", "mesh_scale_to_m",
                "gripper_id", "bundle_path", "gripper_metadata", "error", "cached"):
        assert getattr(decoded, key) == getattr(original, key)
    np.testing.assert_array_equal(decoded.grasps, original.grasps)
    np.testing.assert_array_equal(decoded.scores, original.scores)
    assert decoded.grasps.dtype == decoded.scores.dtype == np.float64
    assert decoded.grasps.flags.owndata and decoded.scores.flags.owndata
    payload[:] = b"\0" * len(payload)
    np.testing.assert_array_equal(decoded.grasps, original.grasps)


@pytest.mark.parametrize("status", ["running", "error"])
def test_status_without_arrays_roundtrip(status):
    value = GraspResult("job", "asset", "A", status, error="failure" if status == "error" else None)
    assert decode_result(encode_result(value)) == value


def test_empty_candidates_are_preserved_as_arrays_not_missing():
    value = replace(result(), grasps=np.empty((0, 4, 4)), scores=np.empty(0))
    decoded = decode_result(encode_result(value))
    assert decoded.grasps.shape == (0, 4, 4) and decoded.scores.shape == (0,)


@pytest.mark.parametrize("fields", [
    {"grasps": np.zeros((3, 3, 3))}, {"scores": np.zeros(2)}, {"grasps": None}, {"scores": None},
    {"grasps": np.full((3, 4, 4), "x")}, {"status": "unknown"}, {"cached": 1},
    {"gripper_metadata": []}, {"mesh_scale_to_m": True}, {"request_id": 1},
])
def test_result_contract_rejects_malformed_fields(fields):
    with pytest.raises(ValueError):
        encode_result(replace(result(), **fields))


@pytest.mark.parametrize("fields", [{"regenerate": 1}, {"object_id": None}, {"mesh_scale_to_m": "m"}])
def test_request_contract_rejects_malformed_fields(fields):
    with pytest.raises(ValueError):
        encode_request(replace(GraspRequest("job", "asset", "A", "mesh.ply"), **fields))


@pytest.mark.parametrize("delta", [-1, 1])
def test_result_payload_size_must_match_arrays(delta):
    data = encode_result(result())
    malformed = data[:-1] if delta < 0 else data + b"x"
    with pytest.raises(ValueError, match="byte length"):
        decode_result(malformed)


@pytest.mark.parametrize("payload", [b"", b"abc", struct.pack("<I", 100) + b"{}"])
def test_truncated_headers_rejected(payload):
    with pytest.raises(ValueError, match="Truncated"):
        decode_result(payload)


def test_request_does_not_accept_trailing_payload():
    with pytest.raises(ValueError, match="Unexpected"):
        decode_request(encode_request(GraspRequest("job", "asset", "A", "mesh.ply")) + b"x")
