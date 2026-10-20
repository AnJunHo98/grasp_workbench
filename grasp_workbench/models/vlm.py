"""Small OpenAI semantic boundary: current RGB plus a public text asset catalog."""
import base64
from dataclasses import dataclass
import io
import json
from pathlib import Path

from ..ipc.vlm import VlmObject, object_from_record, public_assets
from ..semantic import public_policy_catalog, task_schema, validate_coordination, validate_policy_selection, validate_task_binding

DEFAULT_MODEL = "gpt-6.1-sol"
_INSTRUCTIONS = (
    "Identify at most two distinct visible objects relevant to the user's instruction. "
    "Return objects in the order requested, for local A/B slots; do not repeat the same visible instance. "
    "For each object give a label. segmentation_prompt for SAM3 must be a short English noun phrase, "
    "about 2-6 words: object category plus simple color/material, e.g. 'red mug' or 'blue plastic bottle'. "
    "Do not use full sentences, relative location, support objects, or brand/writing details in segmentation_prompt. "
    "bbox_xyxy is [left, top, right, bottom] in ORIGINAL IMAGE PIXELS, "
    "not normalized coordinates; use null when a region cannot be identified. "
    "Choose known_asset_id only from an appropriate reference in the supplied asset catalog, "
    "otherwise null. Similar labels are not proof of the same asset or physical instance. "
    "Catalog references are suggestions, not proof of identity or metric geometry. "
    "Each object has a distinct nonempty observation_id local to this image, not a persistent tracking ID. "
    "Multiple visible instances may use the same known_asset_id. "
    "Return task_binding with source_observation_id for the object to grasp or move, "
    "target_observation_id for a distinct reference object, and intent. "
    "For a grasp-only task use source and null target even if other objects are visible. "
    "Choose roles from visual/task meaning, never catalog or object list order. "
    "If a role is unresolved use null. If nothing matches return objects: [] and null task references. "
    "Return policy_selection with task_type, intent_origin (explicit, inferred, unclear), policy_id and reason. "
    "An instruction such as '정리해라' with a tray and an object outside it may imply place_on_surface; "
    "describe this inference in task_binding.intent, or mark intent_origin unclear when competing meanings remain. "
    "Choose policy_id only from supplied policies matching task_type and intended operation; "
    "a shared target does not make placement equivalent to pouring. Use null when no policy matches. "
    "Different source geometry can be only an unverified semantic transfer candidate; apply no coordinate correction. "
    "Do not generate, train or invent a fallback policy. "
    "Also return coordination with arm_mode (single, dual, unknown), assignments, constraints and reason. "
    "Source/Target are object task roles, never left/right arm aliases. Assign left/right arms to current "
    "observation IDs with a role and visual grasp_region; both arms may be assigned to the same object. "
    "Leave arm/object references null when unresolved; do not substitute a single/dual guess for unknown. "
    "A constraint has reference and moving endpoints (frame left_ee, right_ee, object or unknown; "
    "observation_id only for object endpoints), behavior, phases, reference_state and description. "
    "For a rigid object held by both hands, maintain_relative_pose during transport refers to the state "
    "after both grasps are confirmed, not an invented current arrangement. Separate objects may instead "
    "need their object-to-object relation maintained. Supporting while inserting or rotating may require "
    "allow_relative_motion rather than a rigid lock. Scope constraints to relevant transport/insertion phases, "
    "not grasp/release without evidence. Leave phases/reference_state empty and behavior unknown when unresolved; "
    "assignments/constraints may be empty. Explain uncertainty in reason/description. "
    "Descriptions other than segmentation_prompt should be concise Korean. "
    "Do not infer masks, numeric poses, transformations, depth, scale, tolerances, axes, joint values, "
    "robot state or control parameters. grasp_region and constraints are semantic descriptions only, "
    "not grasp filters, numeric constraints, planner results, robot commands or execution authorization."
)


class VlmError(RuntimeError):
    """Only an exception type name crosses the SDK boundary, never its message."""


@dataclass
class VlmPrediction:
    objects: tuple[VlmObject, ...]
    task_binding: dict
    policy_selection: dict
    coordination: dict


def _schema(asset_ids, policies=()):
    return task_schema(asset_ids, policies)


def _parse_objects(output_text, frame, asset_ids):
    payload = json.loads(output_text)
    if (not isinstance(payload, dict) or set(payload) != {"objects"}
            or not isinstance(payload["objects"], list) or len(payload["objects"]) > 2):
        raise ValueError("Invalid VLM objects")
    objects = tuple(object_from_record(item, frame) for item in payload["objects"])
    if any(item.known_asset_id not in [None, *asset_ids] for item in objects):
        raise ValueError("Unknown VLM asset reference")
    return objects


def _parse_prediction(output_text, frame, asset_ids, policies=()):
    payload = json.loads(output_text)
    if (not isinstance(payload, dict) or set(payload) != {"objects", "task_binding", "policy_selection", "coordination"}
            or not isinstance(payload["objects"], list) or len(payload["objects"]) > 2):
        raise ValueError("Invalid semantic task fields")
    objects, observation_ids = [], set()
    for item in payload["objects"]:
        if not isinstance(item, dict) or set(item) != {"observation_id", "label", "segmentation_prompt", "bbox_xyxy", "known_asset_id"}:
            raise ValueError("Invalid semantic object fields")
        observation_id = item["observation_id"]
        if not isinstance(observation_id, str) or not observation_id.strip() or observation_id in observation_ids:
            raise ValueError("Invalid or duplicate observation reference")
        observation_ids.add(observation_id)
        obj = object_from_record(item, frame)
        if obj.known_asset_id not in [None, *asset_ids]:
            raise ValueError("Unknown VLM asset reference")
        objects.append(obj)
    return VlmPrediction(tuple(objects), validate_task_binding(payload["task_binding"], observation_ids),
                         validate_policy_selection(payload["policy_selection"], policies),
                         validate_coordination(payload["coordination"], observation_ids))


class OfficialVlmPredictor:
    def __init__(self, model=DEFAULT_MODEL, api_key_file=None, *, client_factory=None):
        self.model, self.api_key_file = model, api_key_file
        self._client_factory, self._client = client_factory, None

    def start(self):
        if self._client is not None:
            return
        try:
            factory = self._client_factory
            if factory is None:
                from openai import OpenAI

                factory = OpenAI
            options = {"max_retries": 0}
            if self.api_key_file is not None:
                options["api_key"] = Path(self.api_key_file).read_text(encoding="utf-8").strip()
            self._client = factory(**options)
        except Exception as error:
            raise VlmError(type(error).__name__) from None

    def predict(self, request):
        if self._client is None:
            raise RuntimeError("Start VLM before prediction")
        try:
            from PIL import Image

            assets = public_assets(request.assets)
            policies = public_policy_catalog(request.policies)
            asset_ids = list(dict.fromkeys(asset["asset_id"] for asset in assets))
            height, width = request.frame.rgb.shape[:2]
            buffer = io.BytesIO()
            Image.fromarray(request.frame.rgb).save(buffer, format="PNG")
            image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            response = self._client.responses.create(
                model=self.model, instructions=_INSTRUCTIONS, store=False,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": json.dumps({
                        "instruction": request.instruction, "image_size": {"width": width, "height": height},
                        "assets": assets, "policies": policies,
                    }, ensure_ascii=False)},
                    {"type": "input_image", "image_url": image_url},
                ]}],
                text={"format": {"type": "json_schema", "name": "visible_objects",
                                 "schema": _schema(asset_ids, policies), "strict": True}},
            )
            if response.status != "completed":
                raise ValueError("VLM response did not complete")
            return _parse_prediction(response.output_text, request.frame, asset_ids, policies)
        except Exception as error:
            raise VlmError(type(error).__name__) from None

    def close(self):
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as error:
                raise VlmError(type(error).__name__) from None
