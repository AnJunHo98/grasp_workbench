"""Compact grasp-selection snapshots over iceoryx2; no robot commands or images."""
from dataclasses import dataclass
import json

from .sam3 import _BytePublisher, _ByteSubscriber


@dataclass
class SelectionRequest:
    request_id: str
    coordination: dict
    objects: tuple[dict, ...]
    action: str = "select"
    attachment: dict | None = None


@dataclass
class SelectionResult:
    request_id: str
    status: str
    selection: dict | None = None
    error: str | None = None


def services(base):
    return base + "/grasp_selection/requests", base + "/grasp_selection/results"


def _encode(kind, **fields):
    if not isinstance(fields["request_id"], str):
        raise ValueError("Selection request ID must be a string")
    return json.dumps({"schema_version": 1, "kind": kind, **fields},
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _decode(payload, kind):
    record = json.loads(bytes(payload))
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or record.get("kind") != kind):
        raise ValueError("Unsupported grasp selection schema")
    return record


def encode_request(request):
    from ..semantic import validate_coordination

    if request.action not in ("select", "attach", "detach"):
        raise ValueError("Unknown grasp selection action")
    if request.attachment is not None and not isinstance(request.attachment, dict):
        raise ValueError("Grasp attachment must be a record or null")
    if request.action != "select":
        if request.action == "attach" and request.attachment is None:
            raise ValueError("Attach requires the selected grasp reference")
        return _encode("selection_request", request_id=request.request_id,
                       coordination=request.coordination, objects=request.objects,
                       action=request.action, attachment=request.attachment)
    if not isinstance(request.objects, (tuple, list)) or any(not isinstance(item, dict) for item in request.objects):
        raise ValueError("Selection objects must be records")
    for item in request.objects:
        for name in ("object_id", "observation_id", "instance_id"):
            if not isinstance(item.get(name), str) or not item[name]:
                raise ValueError(f"Selection object requires {name}")
        for name in ("bundle_path", "mesh_path"):
            if item.get(name) is not None and not isinstance(item[name], str):
                raise ValueError(f"Selection {name} must be a string or null")
    ids = [item["observation_id"] for item in request.objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Selection observation IDs must be unique")
    coordination = validate_coordination(request.coordination, ids)
    return _encode("selection_request", request_id=request.request_id,
                   coordination=coordination, objects=request.objects,
                   action=request.action, attachment=request.attachment)


def decode_request(payload):
    record = _decode(payload, "selection_request")
    request = SelectionRequest(record["request_id"], record["coordination"], tuple(record["objects"]),
                               record.get("action", "select"), record.get("attachment"))
    encode_request(request)
    return request


def encode_result(result):
    if (result.status not in {"running", "done", "error"}
            or (result.selection is not None and not isinstance(result.selection, dict))
            or (result.error is not None and not isinstance(result.error, str))):
        raise ValueError("Invalid grasp selection result")
    return _encode("selection_result", request_id=result.request_id, status=result.status,
                   selection=result.selection, error=result.error)


def decode_result(payload):
    record = _decode(payload, "selection_result")
    result = SelectionResult(*(record[name] for name in ("request_id", "status", "selection", "error")))
    encode_result(result)
    return result


class _SelectionPublisher(_BytePublisher):
    def update_connections(self):
        if self._publisher is None:
            raise RuntimeError("Selection publisher is closed")
        return self._publisher.update_connections()


class SelectionRequestPublisher(_SelectionPublisher):
    def publish(self, request):
        return self._publish(encode_request(request))


class SelectionRequestSubscriber(_ByteSubscriber):
    def receive_latest(self):
        return self._receive(decode_request)


class SelectionResultPublisher(_SelectionPublisher):
    def publish(self, result):
        return self._publish(encode_result(result))


class SelectionResultSubscriber(_ByteSubscriber):
    def receive_latest(self):
        return self._receive(decode_result)
