"""Optional semantic policy metadata; checkpoint paths stay local and are not loaded."""
import json
from pathlib import Path

from .semantic import public_policy_catalog as public_policy_catalog

_PUBLIC_FIELDS = ("policy_id", "display_name", "task_type", "description", "training_source",
                  "training_target", "pose_convention", "transfer_note")


def load_policy_catalog(path=None):
    if path is None:
        return []
    path = Path(path).expanduser().resolve()
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Policy catalog must be a list")
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Policy catalog entries must be objects")
        for key in (*_PUBLIC_FIELDS, "checkpoint"):
            value = entry.get(key)
            if key in ("training_source", "training_target"):
                if not isinstance(value, dict) or any(not isinstance(value.get(name), str)
                                                     or not value[name].strip() for name in ("asset_id", "name")):
                    raise ValueError(f"Policy {key} requires asset_id and name")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"Policy {key} must be a nonempty string")
        if entry["policy_id"] in seen:
            raise ValueError(f"Duplicate policy_id: {entry['policy_id']}")
        seen.add(entry["policy_id"])
        result.append({**entry, "checkpoint": str((path.parent / entry["checkpoint"]).resolve())})
    return result


def resolve_policy(entries, policy_id, task_type):
    if policy_id is None:
        return None
    for entry in entries:
        if entry["policy_id"] == policy_id:
            if entry["task_type"] != task_type:
                raise ValueError(f"Policy {policy_id} does not match task_type {task_type}")
            return entry.copy()
    raise ValueError(f"Unknown policy_id: {policy_id}")
