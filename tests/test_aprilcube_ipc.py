"""Owned pose bindings and straight-alpha display layers, without camera mutation."""
import json
import struct

import numpy as np
import pytest

from grasp_workbench.ipc.aprilcube import (
    AprilCubeControl, AprilCubeOverlay, AprilCubeTarget,
    decode_control, decode_overlay, encode_control, encode_overlay, services,
)


def target():
    transform = np.eye(4)
    transform[:3, 3] = (.01, -.02, .03)
    return AprilCubeTarget("A", "session", "mesh.obj", .001, transform)


def test_control_roundtrip_preserves_mesh_scale_transform_and_empty_selection():
    original = target()
    result = decode_control(encode_control(AprilCubeControl((original,))))
    decoded, = result.targets
    assert decoded.object_id == "A" and decoded.session_id == "session"
    assert decoded.mesh_path == "mesh.obj" and decoded.mesh_scale_to_m == .001
    np.testing.assert_array_equal(decoded.cube_from_object, original.cube_from_object)
    assert not np.shares_memory(decoded.cube_from_object, original.cube_from_object)
    assert decode_control(encode_control(AprilCubeControl(()))).targets == ()
    assert services("camera") == ("camera/aprilcube/control", "camera/aprilcube/overlay")


def test_overlay_roundtrip_is_straight_rgba_owned_and_metadata_preserved():
    pixels = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)[:, ::-1]
    payload = bytearray(encode_overlay(AprilCubeOverlay({"sequence": 7}, pixels)))
    result = decode_overlay(payload)
    payload[:] = b"\0" * len(payload)
    assert result.metadata == {"sequence": 7}
    np.testing.assert_array_equal(result.rgba, pixels)
    assert result.rgba.flags.owndata


@pytest.mark.parametrize("scale", [0, -.001, float("nan"), True])
def test_invalid_scale(scale):
    value = target()
    value.mesh_scale_to_m = scale
    with pytest.raises(ValueError, match="Mesh scale"):
        encode_control(AprilCubeControl((value,)))


def test_invalid_transform_and_overlay_payload_are_rejected():
    value = target()
    value.cube_from_object = np.eye(3)
    with pytest.raises(ValueError, match="4x4"):
        encode_control(AprilCubeControl((value,)))
    with pytest.raises(ValueError, match="RGBA"):
        encode_overlay(AprilCubeOverlay({}, np.zeros((2, 3, 3), dtype=np.uint8)))
    payload = encode_overlay(AprilCubeOverlay({}, np.zeros((2, 3, 4), dtype=np.uint8)))
    for bad in (payload[:2], payload[:-1], payload + b"x"):
        with pytest.raises(ValueError):
            decode_overlay(bad)
    data = json.dumps({"schema_version": 1, "kind": "wrong"}).encode()
    with pytest.raises(ValueError, match="schema"):
        decode_control(struct.pack("<I", len(data)) + data)
