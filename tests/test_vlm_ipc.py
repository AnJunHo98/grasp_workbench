"""Owned raw RGB-D semantic IPC; no SDK, provider or credentials needed."""
from dataclasses import replace
import json
import struct
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.vlm import (
    VlmObject, VlmRequest, VlmRequestPublisher, VlmRequestSubscriber,
    VlmResult, VlmResultPublisher, VlmResultSubscriber,
    decode_request, decode_result, encode_request, encode_result, services,
)


def request():
    frame = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
                        np.array([[0, np.nan, np.inf, 1], [2, 3, 4, 5], [6, 7, 8, 9]], np.float32),
                        {"camera_id": "zed", "sequence": 7, "intrinsics": {"fx": 100, "cx": 2}})
    return VlmRequest("request-한글", frame, "테이프와 참치캔", (
        {"asset_id": "tuna", "name": "참치캔", "prompt": "yellow tuna can"},))


def result():
    value = request()
    return VlmResult(value.request_id, value.frame, "done", (
        VlmObject("테이프", "brown packing tape roll", (0, 0, 2, 3), None),
        VlmObject("참치캔", "yellow tuna can", None, "tuna"),), model="example-model")


def policy():
    return {"policy_id": "pour-1", "display_name": "Pour liquid", "task_type": "pour",
            "description": "Move source toward target", "training_source": {"asset_id": "source", "name": "cup"},
            "training_target": {"asset_id": "target", "name": "bowl"}, "pose_convention": "target_from_source",
            "transfer_note": "Semantic reference only"}


def semantic_result():
    original = result()
    return replace(original,
                   objects=(replace(original.objects[0], observation_id="current-tool"),
                            replace(original.objects[1], observation_id="current-target")),
                   task_binding={"source_observation_id": "current-tool", "target_observation_id": "current-target", "intent": "pour"},
                   policy_selection={"task_type": "pour", "intent_origin": "explicit", "policy_id": "pour-1", "reason": "requested"},
                   coordination={"arm_mode": "dual", "assignments": [
                       {"arm": "left", "observation_id": "current-target", "role": "support", "grasp_region": "side"},
                       {"arm": "right", "observation_id": "current-tool", "role": "pour", "grasp_region": "handle"}],
                       "constraints": [{"reference": {"frame": "object", "observation_id": "current-target"},
                                        "moving": {"frame": "object", "observation_id": "current-tool"},
                                        "behavior": "allow_relative_motion", "phases": ["interaction"],
                                        "reference_state": "current observation", "description": "semantic intent"}],
                       "reason": "user intent"})


def assert_same(actual, expected):
    assert actual.request_id == expected.request_id
    assert actual.frame.metadata == expected.frame.metadata
    for name in ("rgb", "depth_m"):
        array = getattr(actual.frame, name)
        np.testing.assert_array_equal(array, getattr(expected.frame, name))
        assert array.flags.owndata
    if isinstance(expected, VlmRequest):
        assert actual.instruction == expected.instruction and actual.assets == expected.assets
        assert actual.policies == expected.policies
    else:
        assert (actual.status, actual.objects, actual.error, actual.model) == (
            expected.status, expected.objects, expected.error, expected.model)
        assert (actual.task_binding, actual.policy_selection, actual.coordination) == (
            expected.task_binding, expected.policy_selection, expected.coordination)


def rewrite(payload, **fields):
    size = struct.unpack_from("<I", payload)[0]
    record = json.loads(payload[4:4 + size])
    record.update(fields)
    header = json.dumps(record).encode()
    return struct.pack("<I", len(header)) + header + payload[4 + size:]


@pytest.mark.parametrize("factory,encoder,decoder", [
    (request, encode_request, decode_request), (result, encode_result, decode_result),
    (semantic_result, encode_result, decode_result),
])
def test_exact_frame_metadata_unicode_and_owned_raw_arrays(factory, encoder, decoder):
    original = factory()
    payload = bytearray(encoder(original))
    assert b"base64" not in payload and b"image/png" not in payload
    decoded = decoder(payload)
    payload[:] = b"\0" * len(payload)
    assert_same(decoded, original)


def test_request_only_transports_public_catalog_fields():
    original = request()
    original.assets[0]["private_path"] = "/private/checkpoint"
    payload = encode_request(original)
    assert b"private_path" not in payload and b"checkpoint" not in payload
    assert set(decode_request(payload).assets[0]) == {"asset_id", "name", "prompt"}


@pytest.mark.parametrize("status,error", [("running", None), ("done", None), ("error", "RuntimeError")])
def test_empty_objects_and_statuses_roundtrip(status, error):
    original = replace(result(), status=status, objects=(), error=error)
    assert_same(decode_result(encode_result(original)), original)


@pytest.mark.parametrize("encoder,decoder,factory", [
    (encode_request, decode_request, request), (encode_result, decode_result, result),
])
def test_truncated_unknown_schema_and_extra_bytes_are_rejected(encoder, decoder, factory):
    payload = encoder(factory())
    for bad in (b"", b"\0", payload[:7], payload[:-1], payload + b"extra",
                rewrite(payload, schema_version=2), rewrite(payload, schema_version=True),
                rewrite(payload, request_id=None), rewrite(payload, frame_size=True)):
        with pytest.raises((ValueError, TypeError)):
            decoder(bad)


@pytest.mark.parametrize("changes", [
    {"request_id": 1}, {"instruction": None}, {"assets": ({"asset_id": "missing-fields"},)},
])
def test_invalid_requests(changes):
    with pytest.raises(ValueError):
        encode_request(replace(request(), **changes))


@pytest.mark.parametrize("changes", [
    {"status": "pending"}, {"status": "running"}, {"error": 123}, {"model": []},
    {"objects": (VlmObject("", "prompt", None, None),)},
    {"objects": (VlmObject("label", "", None, None),)},
    {"objects": (VlmObject("label", "prompt", (1, 1, 0, 2), None),)},
    {"objects": (VlmObject("label", "prompt", (0, 0, 5, 2), None),)},
    {"objects": (VlmObject("label", "prompt", (0, 0, np.nan, 2), None),)},
    {"objects": (VlmObject("label", "prompt", None, 1),)},
    {"objects": (VlmObject("label", "prompt", None, None),) * 3},
])
def test_invalid_results(changes):
    with pytest.raises(ValueError):
        encode_result(replace(result(), **changes))


@pytest.mark.parametrize("publisher_type,subscriber_type,factory", [
    (VlmRequestPublisher, VlmRequestSubscriber, request),
    (VlmResultPublisher, VlmResultSubscriber, result),
    (VlmResultPublisher, VlmResultSubscriber, semantic_result),
])
def test_native_late_join_retention_ownership_and_cleanup(publisher_type, subscriber_type, factory):
    pytest.importorskip("iceoryx2")
    service = "vlm-test/" + uuid4().hex
    publisher, subscriber = publisher_type(service), None
    decoded = None
    original = factory()
    try:
        assert publisher.publish(original) == 0
        subscriber = subscriber_type(service)
        deadline = time.monotonic() + 3
        while decoded is None and time.monotonic() < deadline:
            publisher.update_connections()
            decoded = subscriber.receive_latest()
            if decoded is None:
                time.sleep(.01)
        assert decoded is not None
        assert_same(decoded, original)
    finally:
        if subscriber is not None:
            subscriber.close()
        publisher.close()
    assert_same(decoded, original)
    with pytest.raises(RuntimeError):
        publisher.update_connections()


def test_service_names():
    assert services("camera") == ("camera/vlm/requests", "camera/vlm/results")


def test_observation_ids_roundtrip_without_inventing_legacy_roles():
    original = replace(result(), objects=(VlmObject("tape", "brown tape roll", None, None, "current-A"),))
    decoded = decode_result(encode_result(original))
    assert decoded.objects[0].observation_id == "current-A"
    assert decoded.task_binding is decoded.policy_selection is decoded.coordination is None


def test_policy_catalog_whitelist_strips_private_paths_and_nested_training_sidecars():
    entry = policy()
    entry.update({"checkpoint_path": "/private/model.pth", "private_path": "/private/policy"})
    entry["training_source"]["checkpoint"] = "/private/source"
    value = replace(request(), policies=(entry,))
    payload = encode_request(value)
    assert b"/private" not in payload and b"checkpoint" not in payload and b"private_path" not in payload
    decoded = decode_request(payload)
    assert set(decoded.policies[0]) == set(policy())
    assert decoded.policies[0]["training_source"] == {"asset_id": "source", "name": "cup"}
    assert entry["checkpoint_path"] == "/private/model.pth"  # Caller-owned input was not mutated.


def test_both_arms_can_reference_one_current_observation_without_slot_role_invention():
    value = semantic_result()
    value.coordination["assignments"][0]["observation_id"] = "current-tool"
    decoded = decode_result(encode_result(value))
    assert [item["observation_id"] for item in decoded.coordination["assignments"]] == ["current-tool", "current-tool"]
    assert_same(decoded, value)


@pytest.mark.parametrize("field", ["task_binding", "policy_selection", "coordination"])
def test_partial_semantic_payload_is_rejected_instead_of_filling_missing_roles(field):
    with pytest.raises(ValueError):
        encode_result(replace(semantic_result(), **{field: None}))


@pytest.mark.parametrize("field", ["task", "assignment", "constraint"])
def test_semantic_relations_cannot_reference_observations_from_another_request(field):
    value = semantic_result()
    if field == "task":
        value.task_binding["source_observation_id"] = "previous-observation"
    elif field == "assignment":
        value.coordination["assignments"][0]["observation_id"] = "previous-observation"
    else:
        value.coordination["constraints"][0]["reference"]["observation_id"] = "previous-observation"
    with pytest.raises(ValueError):
        encode_result(value)


@pytest.mark.parametrize("ids", [("", "current-target"), ("duplicate", "duplicate")])
def test_full_semantics_need_unique_nonempty_current_observation_ids(ids):
    value = semantic_result()
    value.objects = tuple(replace(obj, observation_id=identity) for obj, identity in zip(value.objects, ids))
    with pytest.raises(ValueError, match="unique current"):
        encode_result(value)


def test_legacy_wire_without_observation_policy_or_semantic_fields_remains_readable():
    for original, encoder, decoder in ((request(), encode_request, decode_request), (result(), encode_result, decode_result)):
        payload = encoder(original)
        size = struct.unpack_from("<I", payload)[0]
        record = json.loads(payload[4:4 + size])
        for field in ("policies", "task_binding", "policy_selection", "coordination"):
            record.pop(field, None)
        for obj in record.get("objects", []):
            obj.pop("observation_id", None)
        header = json.dumps(record).encode()
        decoded = decoder(struct.pack("<I", len(header)) + header + payload[4 + size:])
        assert_same(decoded, original)
