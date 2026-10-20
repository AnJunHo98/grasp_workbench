"""On-demand semantic recognition with exact RGB-D frames over raw iceoryx2."""
from dataclasses import asdict, dataclass
import json
import math
import struct

from .camera import CameraFrame, decode_frame, encode_frame
from .sam3 import _BytePublisher, _ByteSubscriber

_LENGTH = struct.Struct("<I")


@dataclass
class VlmRequest:
    request_id: str
    frame: CameraFrame
    instruction: str
    assets: tuple[dict, ...]
    policies: tuple[dict, ...] = ()


@dataclass
class VlmObject:
    label: str
    segmentation_prompt: str
    bbox_xyxy: tuple[float, float, float, float] | None
    known_asset_id: str | None
    observation_id: str = ""


@dataclass
class VlmResult:
    request_id: str
    frame: CameraFrame
    status: str
    objects: tuple[VlmObject, ...] = ()
    error: str | None = None
    model: str | None = None
    task_binding: dict | None = None
    policy_selection: dict | None = None
    coordination: dict | None = None


def services(base):
    return base + "/vlm/requests", base + "/vlm/results"


def public_assets(assets):
    result = []
    for asset in assets:
        if not isinstance(asset, dict) or any(not isinstance(asset.get(key), str)
                                             for key in ("asset_id", "name", "prompt")):
            raise ValueError("Invalid public asset catalog")
        result.append({key: asset[key] for key in ("asset_id", "name", "prompt")})
    return result


def public_policies(policies):
    from ..semantic import public_policy_catalog

    if not isinstance(policies, (tuple, list)):
        raise ValueError("Policy catalog must be a list or tuple")
    return public_policy_catalog(policies)


def object_from_record(record, frame):
    fields = {"label", "segmentation_prompt", "bbox_xyxy", "known_asset_id"}
    if not isinstance(record, dict) or set(record) not in (fields, fields | {"observation_id"}):
        raise ValueError("Invalid VLM object fields")
    if any(not isinstance(record[key], str) or not record[key].strip() for key in ("label", "segmentation_prompt")):
        raise ValueError("VLM object labels/prompts must be nonempty strings")
    asset_id, box = record["known_asset_id"], record["bbox_xyxy"]
    observation_id = record.get("observation_id", "")
    if not isinstance(observation_id, str):
        raise ValueError("VLM observation ID must be a string")
    if asset_id is not None and not isinstance(asset_id, str):
        raise ValueError("Invalid VLM asset reference")
    if box is not None:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in box)):
            raise ValueError("VLM box must contain four finite pixel coordinates")
        height, width = frame.rgb.shape[:2]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError("VLM box must lie inside the original image")
        box = tuple(map(float, box))
    return VlmObject(record["label"], record["segmentation_prompt"], box, asset_id, observation_id)


def _encode(kind, value, **fields):
    if not isinstance(value.request_id, str):
        raise ValueError("VLM request ID must be a string")
    frame = encode_frame(value.frame)
    header = json.dumps({"schema_version": 1, "kind": kind, "request_id": value.request_id,
                         "frame_size": len(frame), **fields}, ensure_ascii=False, allow_nan=False).encode()
    return _LENGTH.pack(len(header)) + header + frame


def _decode(payload, kind):
    view = memoryview(payload).cast("B")
    if len(view) < 4:
        raise ValueError("Truncated VLM header")
    offset = 4 + _LENGTH.unpack_from(view)[0]
    if offset > len(view):
        raise ValueError("Truncated VLM metadata")
    record = json.loads(bytes(view[4:offset]))
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or record.get("kind") != kind
            or not isinstance(record.get("request_id"), str)
            or type(record.get("frame_size")) is not int or record["frame_size"] <= 0
            or len(view) != offset + record["frame_size"]):
        raise ValueError("Invalid VLM schema or frame length")
    return record, decode_frame(view[offset:])


def encode_request(request):
    if not isinstance(request.instruction, str):
        raise ValueError("VLM instruction must be a string")
    return _encode("vlm_request", request, instruction=request.instruction, assets=public_assets(request.assets),
                   policies=public_policies(request.policies))


def decode_request(payload):
    record, frame = _decode(payload, "vlm_request")
    if not isinstance(record.get("instruction"), str) or not isinstance(record.get("assets"), list):
        raise ValueError("Invalid VLM request fields")
    return VlmRequest(record["request_id"], frame, record["instruction"], tuple(public_assets(record["assets"])),
                      tuple(public_policies(record.get("policies", []))))


def _result_fields(status, objects, error, model, frame):
    if (status not in ("running", "done", "error") or not isinstance(objects, (tuple, list))
            or len(objects) > 2 or (status != "done" and objects)
            or any(value is not None and not isinstance(value, str) for value in (error, model))):
        raise ValueError("Invalid VLM result fields")
    return tuple(object_from_record(item, frame) for item in objects)


def _semantic_fields(objects, task_binding, policy_selection, coordination):
    if task_binding is policy_selection is coordination is None:
        return {"task_binding": None, "policy_selection": None, "coordination": None}
    from ..semantic import validate_coordination, validate_policy_selection, validate_task_binding

    observation_ids = [item.observation_id for item in objects]
    if (any(not observation_id.strip() for observation_id in observation_ids)
            or len(set(observation_ids)) != len(observation_ids)):
        raise ValueError("Semantic results require unique current observation IDs")
    return {"task_binding": validate_task_binding(task_binding, observation_ids),
            "policy_selection": validate_policy_selection(policy_selection),
            "coordination": validate_coordination(coordination, observation_ids)}


def encode_result(result):
    objects = [asdict(item) for item in result.objects]
    parsed = _result_fields(result.status, objects, result.error, result.model, result.frame)
    semantics = _semantic_fields(parsed, result.task_binding, result.policy_selection, result.coordination)
    return _encode("vlm_result", result, status=result.status, objects=objects,
                   error=result.error, model=result.model, **semantics)


def decode_result(payload):
    record, frame = _decode(payload, "vlm_result")
    objects = _result_fields(record.get("status"), record.get("objects"), record.get("error"), record.get("model"), frame)
    semantics = _semantic_fields(objects, record.get("task_binding"), record.get("policy_selection"), record.get("coordination"))
    return VlmResult(record["request_id"], frame, record["status"], objects,
                     record.get("error"), record.get("model"), **semantics)


class _VlmPublisher(_BytePublisher):
    def update_connections(self):
        if self._publisher is None:
            raise RuntimeError("VLM publisher is closed")
        return self._publisher.update_connections()


class VlmRequestPublisher(_VlmPublisher):
    def publish(self, request):
        return self._publish(encode_request(request))


class VlmRequestSubscriber(_ByteSubscriber):
    def receive_latest(self):
        return self._receive(decode_request)


class VlmResultPublisher(_VlmPublisher):
    def publish(self, result):
        return self._publish(encode_result(result))


class VlmResultSubscriber(_ByteSubscriber):
    def receive_latest(self):
        return self._receive(decode_result)
