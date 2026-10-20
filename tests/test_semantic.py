"""Restored task/policy/arm semantics are metadata, not numeric robot commands."""
import json

import pytest

from grasp_workbench.semantic import (
    coordination_schema, public_policy_catalog, task_schema, unresolved_coordination,
    validate_coordination, validate_policy_selection, validate_task_binding,
)


def policy_catalog():
    return [{"policy_id": "place-on-tray", "display_name": "트레이 정리", "task_type": "place_on_surface",
             "description": "물체를 트레이에 놓기", "training_source": {"asset_id": "cup", "name": "컵"},
             "training_target": {"asset_id": "tray", "name": "트레이"},
             "pose_convention": "target_from_source", "transfer_note": "새 물체 전이 미검증"}]


def semantic_payload(objects=()):
    objects = [{**item, "observation_id": item.get("observation_id", f"obs_{index + 1}")}
               for index, item in enumerate(objects)]
    return {"objects": objects,
            "task_binding": {"source_observation_id": objects[0]["observation_id"] if objects else None,
                             "target_observation_id": None, "intent": "물체 파지" if objects else ""},
            "policy_selection": {"task_type": "grasp" if objects else "unknown", "intent_origin": "explicit" if objects else "unclear",
                                 "policy_id": None, "reason": ""},
            "coordination": unresolved_coordination()}


def coordination():
    return {"arm_mode": "dual", "assignments": [
        {"arm": "left", "observation_id": "obs_1", "role": "왼쪽 지지", "grasp_region": "왼쪽 가장자리"},
        {"arm": "right", "observation_id": "obs_1", "role": "오른쪽 지지", "grasp_region": "오른쪽 가장자리"}],
        "constraints": [{"reference": {"frame": "left_ee", "observation_id": None},
                         "moving": {"frame": "right_ee", "observation_id": None},
                         "behavior": "maintain_relative_pose", "phases": ["운반"],
                         "reference_state": "양손 파지 확인 후", "description": "운반 중 상대 자세 유지 요청"}],
        "reason": "양손으로 같은 물체 운반"}


def test_same_object_dual_arm_and_relation_semantics_are_owned_not_execution():
    original = coordination()
    decoded = validate_coordination(original, {"obs_1"})
    assert decoded == original
    original["assignments"][0]["grasp_region"] = "changed"
    original["constraints"][0]["phases"].append("changed")
    assert decoded["assignments"][0]["grasp_region"] == "왼쪽 가장자리"
    assert decoded["constraints"][0]["phases"] == ["운반"]
    assert "pose" not in decoded and "tolerance" not in decoded and "robot_command" not in decoded


def test_unknown_coordination_is_preserved_without_single_or_dual_inference():
    assert validate_coordination(unresolved_coordination(), ()) == unresolved_coordination()
    original = unresolved_coordination()
    original["assignments"] = [{"arm": None, "observation_id": None, "role": "", "grasp_region": ""}]
    assert validate_coordination(original, ()) == original


@pytest.mark.parametrize("change", ["duplicate-arm", "unknown-object", "non-object-reference", "numeric-region",
                                    "unknown-frame", "numeric-phases", "extra-pose", "invalid-mode"])
def test_previous_coordination_reference_and_shape_checks(change):
    value = coordination()
    if change == "duplicate-arm":
        value["assignments"][1]["arm"] = "left"
    elif change == "unknown-object":
        value["assignments"][0]["observation_id"] = "missing"
    elif change == "non-object-reference":
        value["constraints"][0]["reference"]["observation_id"] = "obs_1"
    elif change == "numeric-region":
        value["assignments"][0]["grasp_region"] = [0, 1, 2]
    elif change == "unknown-frame":
        value["constraints"][0]["moving"]["frame"] = "robot_base"
    elif change == "numeric-phases":
        value["constraints"][0]["phases"] = [1]
    elif change == "extra-pose":
        value["constraints"][0]["pose"] = [0, 0, 1]
    else:
        value["arm_mode"] = "auto"
    with pytest.raises(ValueError):
        validate_coordination(value, {"obs_1"})


def test_task_roles_are_observation_ids_not_arms_and_grasp_only_has_null_target():
    binding = {"source_observation_id": "obs_2", "target_observation_id": "obs_1", "intent": "컵을 트레이에 놓기"}
    assert validate_task_binding(binding, {"obs_1", "obs_2"}) == binding
    binding["target_observation_id"] = None
    assert validate_task_binding(binding, {"obs_1", "obs_2"}) == binding
    assert validate_task_binding(semantic_payload()["task_binding"], ()) == semantic_payload()["task_binding"]
    binding["source_observation_id"] = "missing"
    with pytest.raises(ValueError):
        validate_task_binding(binding, {"obs_1", "obs_2"})
    binding.update(source_observation_id="obs_1", target_observation_id="obs_1")
    with pytest.raises(ValueError):
        validate_task_binding(binding, {"obs_1"})


def test_public_policy_catalog_removes_checkpoint_and_nested_private_metadata():
    original = policy_catalog()
    original[0]["checkpoint"] = "/private/model.pth"
    original[0]["training_source"]["path"] = "/private/source.npz"
    public = public_policy_catalog(original)
    assert public == policy_catalog() and "/private" not in json.dumps(public)
    original[0]["training_source"]["name"] = "changed"
    assert public[0]["training_source"]["name"] == "컵"
    with pytest.raises(ValueError):
        public_policy_catalog(public + public)


def test_selection_is_supplied_policy_id_and_matching_task_type_only():
    value = {"task_type": "place_on_surface", "intent_origin": "inferred", "policy_id": "place-on-tray", "reason": "정리 의도"}
    assert validate_policy_selection(value, policy_catalog()) == value
    for changes in ({"policy_id": "invented"}, {"task_type": "pour"}, {"intent_origin": "automatic"}):
        with pytest.raises(ValueError):
            validate_policy_selection({**value, **changes}, policy_catalog())
    assert validate_policy_selection(semantic_payload()["policy_selection"], ()) == semantic_payload()["policy_selection"]


def test_strict_full_schema_covers_roles_policy_and_coordination_with_null_policy_for_empty_catalog():
    schema = task_schema(["cup"], policy_catalog())
    assert set(schema["required"]) == {"objects", "task_binding", "policy_selection", "coordination"}
    assert schema["properties"]["objects"]["maxItems"] == 2
    assert "observation_id" in schema["properties"]["objects"]["items"]["required"]
    assert schema["properties"]["policy_selection"]["properties"]["policy_id"]["enum"] == ["place-on-tray", None]
    assert task_schema([])["properties"]["policy_selection"]["properties"]["policy_id"]["enum"] == [None]
    assert schema["properties"]["coordination"] == coordination_schema()
    assert schema["additionalProperties"] is False
    assert "checkpoint" not in json.dumps(schema)
