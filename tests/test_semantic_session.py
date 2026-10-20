"""Latest semantic observation identities and request-only export, no robot dependencies."""
import json
from types import SimpleNamespace

import numpy as np

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.vlm import VlmObject, VlmResult
from grasp_workbench.semantic_session import SemanticSession, build_execution_request


def task_fields():
    return {"source_observation_id": "source-obs", "target_observation_id": "target-obs", "intent": "move source onto target"}


def coordination_fields():
    return {"arm_mode": "dual", "assignments": [
        {"arm": "left", "observation_id": "source-obs", "role": "support", "grasp_region": "left handle"},
        {"arm": "right", "observation_id": "source-obs", "role": "move", "grasp_region": "right handle"}],
        "constraints": [
            {"reference": {"frame": "left_ee", "observation_id": None},
             "moving": {"frame": "right_ee", "observation_id": None}, "behavior": "maintain_relative_pose",
             "phases": ["transport"], "reference_state": "after both grasps", "description": "Keep two hands together"},
            {"reference": {"frame": "object", "observation_id": "target-obs"},
             "moving": {"frame": "object", "observation_id": "source-obs"}, "behavior": "allow_relative_motion",
             "phases": ["approach"], "reference_state": "current observation", "description": "Approach target"}],
        "reason": "Both hands carry the source, not one arm per visible object"}


def semantic_result(request=None):
    if request is None:
        request = SimpleNamespace(request_id="request", frame=CameraFrame(
            np.zeros((20, 30, 3), dtype=np.uint8), np.ones((20, 30), dtype=np.float32),
            {"camera_id": "fake", "sequence": 7, "api_key_file": "must-not-copy"}))
    return VlmResult(request.request_id, request.frame, "done", (
        VlmObject("target", "tray", (0, 0, 10, 10), "same-asset", "target-obs"),
        VlmObject("source", "tray", (15, 0, 25, 10), "same-asset", "source-obs")),
        model="fake-vlm", task_binding=task_fields(),
        policy_selection={"task_type": "place_on_surface", "intent_origin": "explicit", "policy_id": "place-v1", "reason": "same task"},
        coordination=coordination_fields())


def test_explicit_source_second_and_same_asset_distinct_instances_not_arm_slot_inference():
    session = SemanticSession.from_result(semantic_result(), "carry tray")
    target, source = session.record["objects"]
    assert target["slot"] == "A" and source["slot"] == "B"
    assert target["known_asset_id"] == source["known_asset_id"]
    assert target["instance_id"] != source["instance_id"]
    request = build_execution_request(session.record)
    assert request["task_binding"]["source_instance_id"] == source["instance_id"]
    assert request["task_binding"]["target_instance_id"] == target["instance_id"]
    assert [item["instance_id"] for item in request["coordination"]["assignments"]] == [source["instance_id"]] * 2
    reference = request["coordination"]["constraints"][1]
    assert reference["reference"]["instance_id"] == target["instance_id"]
    assert reference["moving"]["instance_id"] == source["instance_id"]
    assert request["execution"] == "request_only" and request["robot_command_published"] is False
    assert "api_key_file" not in json.dumps(session.record)


def test_instruction_stale_and_binding_invalidation_keep_semantic_arm_information():
    session = SemanticSession.from_result(semantic_result(), "carry tray")
    original = json.loads(json.dumps(session.record["coordination"]))
    session.instruction_edited("pour instead")
    assert session.record["coordination_instruction_stale"] and session.record["coordination"] == original
    session.invalidate_bindings("manual mesh changed", ("B",))
    assert session.object_for_slot("A") is not None and session.object_for_slot("B") is None
    assert session.record["objects"][1]["last_slot"] == "B" and session.record["bindings_stale"]
    assert session.record["task_binding"] == task_fields() and session.record["coordination"] == original


def test_save_load_preserves_references_but_never_restores_current_observation_binding(tmp_path):
    session = SemanticSession.from_result(semantic_result(), "carry tray")
    row = session.record["objects"][1]
    row["geometry"] = {"asset_id": "same-asset", "mesh_path": "/assets/tray.ply", "mesh_scale_to_m": 1}
    row["foundationpose"] = {"session_id": "pose", "status": "tracking", "camera_from_object": np.eye(4).tolist()}
    row["grasp"] = {"bundle_path": "/grasp/bundle.npz", "selected_index": 3, "score": 0.9}
    path = session.save(tmp_path / "latest.json")
    loaded = SemanticSession.load(path)
    source = loaded.record["objects"][1]
    assert source["instance_id"] == row["instance_id"]
    assert source["geometry"] == row["geometry"] and source["foundationpose"] == row["foundationpose"]
    assert source["grasp"] == row["grasp"] and not source["binding_active"] and source["slot"] is None
    assert loaded.record["provenance"]["source"] == "loaded_session" and loaded.record["bindings_stale"]
    request = build_execution_request(loaded.record)
    assert request["coordination"]["assignments"][0]["instance_id"] == source["instance_id"]
    assert request["robot_command_published"] is False


def test_legacy_semantic_none_has_observation_mapping_but_does_not_invent_arm_mode():
    result = semantic_result()
    result.task_binding = result.policy_selection = result.coordination = None
    session = SemanticSession.from_result(result, "find objects")
    assert session.record["coordination"] is None and session.record["task_binding"] is None
    assert session.record["objects"][0]["slot"] == "A"
    assert build_execution_request(session.record)["coordination"] is None


def test_selected_grasp_reachability_snapshot_round_trips_without_becoming_a_robot_command(tmp_path):
    session = SemanticSession.from_result(semantic_result(), "carry tray")
    selected = {"request_id": "selection", "robot_ik_checked": True,
                "selected_candidate": {"candidate_id": "left:1-right:2", "ik_status": "solved"},
                "robot_command_published": False}
    session.record["grasp_selection"] = selected
    loaded = SemanticSession.load(session.save(tmp_path / "session.json"))
    exported = build_execution_request(loaded.record)
    assert loaded.record["grasp_selection"] == exported["grasp_selection"] == selected
    assert exported["execution"] == "request_only" and exported["robot_command_published"] is False
    assert all(not row["binding_active"] for row in loaded.record["objects"])
