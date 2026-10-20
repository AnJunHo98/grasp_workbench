"""Semantic arm/relation intent, not numeric constraints or execution authorization."""

from __future__ import annotations

from collections.abc import Collection

_FIELDS = {"arm_mode", "assignments", "constraints", "reason"}
_ASSIGNMENT_FIELDS = {"arm", "observation_id", "role", "grasp_region"}
_ENDPOINT_FIELDS = {"frame", "observation_id"}
_CONSTRAINT_FIELDS = {
    "reference", "moving", "behavior", "phases", "reference_state", "description",
}
_ARM_MODES = ("single", "dual", "unknown")
_FRAMES = ("left_ee", "right_ee", "object", "unknown")
_BEHAVIORS = ("maintain_relative_pose", "allow_relative_motion", "unknown")
_POLICY_FIELDS = ("policy_id", "display_name", "task_type", "description", "training_source",
                  "training_target", "pose_convention", "transfer_note")


def unresolved_coordination() -> dict:
    """Unknown intent remains representable without assigning arms or adding constraints."""
    return {"arm_mode": "unknown", "assignments": [], "constraints": [], "reason": ""}


def public_policy_catalog(entries) -> list[dict]:
    """Publish the previous semantic catalog fields, never checkpoint paths or sidecars."""
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Policy catalog entries must be objects")
        public = {}
        for key in _POLICY_FIELDS:
            value = entry.get(key)
            if key in ("training_source", "training_target"):
                if not isinstance(value, dict) or any(not isinstance(value.get(field), str) or not value[field].strip()
                                                       for field in ("asset_id", "name")):
                    raise ValueError("Policy training references require asset_id and name")
                public[key] = {field: value[field] for field in ("asset_id", "name")}
            else:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Policy catalog fields must be nonempty strings")
                public[key] = value
        if public["policy_id"] in seen:
            raise ValueError("Duplicate policy_id")
        seen.add(public["policy_id"])
        result.append(public)
    return result


def validate_task_binding(value, observation_ids) -> dict:
    _fields(value, {"source_observation_id", "target_observation_id", "intent"}, "task binding")
    for name in ("source_observation_id", "target_observation_id"):
        _reference(value[name], observation_ids)
    if not isinstance(value["intent"], str):
        raise ValueError("Invalid task intent")
    if value["target_observation_id"] is not None and value["source_observation_id"] == value["target_observation_id"]:
        raise ValueError("A relation uses a source and a distinct target instance")
    return dict(value)


def validate_policy_selection(value, policies=None) -> dict:
    _fields(value, {"task_type", "intent_origin", "policy_id", "reason"}, "policy selection")
    if not isinstance(value["task_type"], str) or not value["task_type"].strip():
        raise ValueError("Invalid policy task type")
    if value["intent_origin"] not in ("explicit", "inferred", "unclear") or not isinstance(value["reason"], str):
        raise ValueError("Invalid policy intent origin or reason")
    policy_id = value["policy_id"]
    if policy_id is not None and (not isinstance(policy_id, str) or not policy_id.strip()):
        raise ValueError("Invalid policy reference")
    if policies is not None and policy_id is not None:
        selected = next((entry for entry in policies if entry["policy_id"] == policy_id), None)
        if selected is None or selected["task_type"] != value["task_type"]:
            raise ValueError("Selected policy does not match the supplied catalog and task type")
    return dict(value)


def _fields(value: object, fields: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Invalid semantic {name} fields")


def _reference(value: object, observation_ids: Collection[str]) -> None:
    if value is not None and (not isinstance(value, str) or value not in observation_ids):
        raise ValueError("Coordination refers to an unknown current observation")


def _endpoint(value: object, observation_ids: Collection[str]) -> dict:
    _fields(value, _ENDPOINT_FIELDS, "constraint endpoint")
    if value["frame"] not in _FRAMES:
        raise ValueError("Invalid semantic constraint frame")
    _reference(value["observation_id"], observation_ids)
    if value["frame"] != "object" and value["observation_id"] is not None:
        raise ValueError("Only an object endpoint can reference an observation")
    return dict(value)


def validate_coordination(value: object, observation_ids: Collection[str]) -> dict:
    """Check structural/current-frame references without making runtime control decisions.

    Both arms may hold the same observation. Unknown arms, objects, phases and reference
    states are valid; actual poses, tolerances and attachment state are resolved locally.
    """
    _fields(value, _FIELDS, "coordination")
    if value["arm_mode"] not in _ARM_MODES or not isinstance(value["reason"], str):
        raise ValueError("Invalid semantic arm mode or coordination reason")
    if not isinstance(value["assignments"], list) or not isinstance(value["constraints"], list):
        raise ValueError("Semantic assignments and constraints must be lists")
    assignments, arms = [], set()
    for item in value["assignments"]:
        _fields(item, _ASSIGNMENT_FIELDS, "arm assignment")
        arm = item["arm"]
        if arm not in ("left", "right", None):
            raise ValueError("Invalid semantic arm assignment")
        if arm is not None:
            if arm in arms:
                raise ValueError("Duplicate semantic arm assignment")
            arms.add(arm)
        _reference(item["observation_id"], observation_ids)
        if any(not isinstance(item[name], str) for name in ("role", "grasp_region")):
            raise ValueError("Invalid semantic arm role or grasp region")
        assignments.append(dict(item))
    constraints = []
    for item in value["constraints"]:
        _fields(item, _CONSTRAINT_FIELDS, "relative constraint")
        if item["behavior"] not in _BEHAVIORS:
            raise ValueError("Invalid semantic relative constraint behavior")
        if not isinstance(item["phases"], list) or any(
            not isinstance(phase, str) for phase in item["phases"]
        ):
            raise ValueError("Invalid semantic constraint phases")
        if any(not isinstance(item[name], str) for name in ("reference_state", "description")):
            raise ValueError("Invalid semantic constraint reference state or description")
        constraints.append({
            **item, "reference": _endpoint(item["reference"], observation_ids),
            "moving": _endpoint(item["moving"], observation_ids), "phases": list(item["phases"]),
        })
    return {"arm_mode": value["arm_mode"], "assignments": assignments,
            "constraints": constraints, "reason": value["reason"]}


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def coordination_schema() -> dict:
    """Strict output schema with semantic-only fields and explicit unknown values."""
    string = {"type": "string"}
    observation = {"type": ["string", "null"]}
    endpoint = _object({
        "frame": {"type": "string", "enum": list(_FRAMES)}, "observation_id": observation,
    })
    return _object({
        "arm_mode": {"type": "string", "enum": list(_ARM_MODES)},
        "assignments": {"type": "array", "items": _object({
            "arm": {"type": ["string", "null"], "enum": ["left", "right", None]},
            "observation_id": observation, "role": string, "grasp_region": string,
        })},
        "constraints": {"type": "array", "items": _object({
            "reference": endpoint, "moving": endpoint,
            "behavior": {"type": "string", "enum": list(_BEHAVIORS)},
            "phases": {"type": "array", "items": string},
            "reference_state": string, "description": string,
        })},
        "reason": string,
    })


def task_schema(asset_ids, policies=()) -> dict:
    """Full semantic output shared by the API and CLI, retaining A/B's two-object scope."""
    string, observation = {"type": "string"}, {"type": ["string", "null"]}
    return _object({
        "objects": {"type": "array", "maxItems": 2, "items": _object({
            "observation_id": string, "label": string, "segmentation_prompt": string,
            "bbox_xyxy": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                     {"type": "null"}]},
            "known_asset_id": {"type": ["string", "null"], "enum": [*asset_ids, None]},
        })},
        "task_binding": _object({"source_observation_id": observation, "target_observation_id": observation, "intent": string}),
        "policy_selection": _object({
            "task_type": string, "intent_origin": {"type": "string", "enum": ["explicit", "inferred", "unclear"]},
            "policy_id": {"type": ["string", "null"], "enum": [*[entry["policy_id"] for entry in policies], None]},
            "reason": string,
        }),
        "coordination": coordination_schema(),
    })
