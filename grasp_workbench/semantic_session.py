"""Latest semantic task snapshot and observation-instance references; no robot commands."""
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

_FRAME_FIELDS = ("camera_id", "stream_id", "sequence", "capture_time_ns", "timestamp_domain", "intrinsics")
_RECORD_FIELDS = ("schema_version", "session_id", "instruction", "semantic_instruction", "provenance", "objects",
                  "task_binding", "policy_selection", "coordination", "policy_reference", "bindings_stale",
                  "coordination_instruction_stale", "stale_reason", "robot_command_published", "grasp_selection")
_OBJECT_FIELDS = ("observation_id", "instance_id", "slot", "last_slot", "binding_active", "label",
                  "segmentation_prompt", "bbox_xyxy", "known_asset_id", "geometry", "foundationpose", "grasp")


def frame_reference(frame):
    return {key: deepcopy(frame.metadata[key]) for key in _FRAME_FIELDS if key in frame.metadata}


def write_record(path, record):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


class SemanticSession:
    def __init__(self, record):
        self.record = record

    @classmethod
    def from_result(cls, result, instruction):
        objects = []
        for slot, item in zip(("A", "B"), result.objects):
            objects.append({"observation_id": item.observation_id or f"obs_{len(objects) + 1}",
                            "instance_id": uuid4().hex, "slot": slot, "last_slot": slot, "binding_active": True,
                            "label": item.label, "segmentation_prompt": item.segmentation_prompt,
                            "bbox_xyxy": item.bbox_xyxy, "known_asset_id": item.known_asset_id,
                            "geometry": None, "foundationpose": None, "grasp": None})
        return cls({"schema_version": 1, "session_id": uuid4().hex, "instruction": instruction,
                    "semantic_instruction": instruction,
                    "provenance": {"source": "vlm", "request_id": result.request_id,
                                   "model": result.model, "frame": frame_reference(result.frame)},
                    "objects": objects, "task_binding": deepcopy(result.task_binding),
                    "policy_selection": deepcopy(result.policy_selection), "coordination": deepcopy(result.coordination),
                    "policy_reference": None, "bindings_stale": False, "coordination_instruction_stale": False,
                    "stale_reason": "", "robot_command_published": False, "grasp_selection": None})

    def object_for_slot(self, slot):
        return next((row for row in self.record["objects"] if row["binding_active"] and row["slot"] == slot), None)

    def invalidate_bindings(self, reason, slots=("A", "B")):
        changed = False
        for row in self.record["objects"]:
            if row["slot"] in slots and row["binding_active"]:
                row["last_slot"], row["slot"], row["binding_active"] = row["slot"], None, False
                changed = True
        if changed:
            self.record["bindings_stale"], self.record["stale_reason"] = True, reason

    def instruction_edited(self, instruction):
        self.record["instruction"] = instruction
        self.record["coordination_instruction_stale"] = True
        self.record["stale_reason"] = "instruction edited · semantic reanalysis needed"

    def save(self, path):
        return write_record(path, self.record)

    @classmethod
    def load(cls, path):
        source = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(source, dict) or source.get("schema_version") != 1 or not isinstance(source.get("objects"), list):
            raise ValueError("Unsupported semantic session")
        record = {key: deepcopy(source[key]) for key in _RECORD_FIELDS if key in source}
        record["objects"] = [{key: deepcopy(row[key]) for key in _OBJECT_FIELDS if key in row}
                             for row in source["objects"]]
        for row in record["objects"]:
            row["last_slot"] = row.get("slot") or row.get("last_slot")
            row["slot"], row["binding_active"] = None, False
        record["provenance"] = {**record.get("provenance", {}), "source": "loaded_session",
                                "session_path": str(Path(path).resolve())}
        record["bindings_stale"], record["robot_command_published"] = True, False
        record["stale_reason"] = "saved semantic snapshot · not a current camera observation"
        return cls(record)


def build_execution_request(record):
    """Map explicit observation IDs to local instances; never infer arms from A/B slots."""
    rows = record["objects"]
    instances = {row["observation_id"]: row["instance_id"] for row in rows}
    binding = deepcopy(record.get("task_binding"))
    if binding is not None:
        for role in ("source", "target"):
            binding[f"{role}_instance_id"] = instances.get(binding.get(f"{role}_observation_id"))
    coordination = deepcopy(record.get("coordination"))
    if coordination is not None:
        for assignment in coordination["assignments"]:
            assignment["instance_id"] = instances.get(assignment["observation_id"])
        for constraint in coordination["constraints"]:
            for name in ("reference", "moving"):
                endpoint = constraint[name]
                endpoint["instance_id"] = instances.get(endpoint["observation_id"]) if endpoint["frame"] == "object" else None
    return {"schema_version": 1, "session_id": record["session_id"], "execution": "request_only",
            "robot_command_published": False, "instruction": record.get("instruction", ""),
            "semantic_instruction": record.get("semantic_instruction", ""), "provenance": deepcopy(record.get("provenance")),
            "objects": [{key: deepcopy(row[key]) for key in ("observation_id", "instance_id", "slot", "binding_active",
                                                             "geometry", "foundationpose", "grasp") if key in row} for row in rows],
            "task_binding": binding, "policy_selection": deepcopy(record.get("policy_selection")),
            "policy_reference": deepcopy(record.get("policy_reference")), "coordination": coordination,
            "grasp_selection": deepcopy(record.get("grasp_selection")),
            "bindings_stale": record.get("bindings_stale", False),
            "coordination_instruction_stale": record.get("coordination_instruction_stale", False)}
