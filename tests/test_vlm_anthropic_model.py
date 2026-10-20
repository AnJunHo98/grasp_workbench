"""Synthetic clients and credentials only; never contact the Anthropic API."""
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.models.vlm import VlmError, VlmPrediction, _schema
from grasp_workbench.models.vlm_anthropic import AnthropicVlmPredictor
from test_semantic import coordination, policy_catalog, semantic_payload
from test_vlm_model import object_record, request


class Client:
    def __init__(self, payload=None):
        self.calls, self.closed = [], False
        self.error, self.close_error = None, None
        self.messages = self
        self.response = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(
            type="text", text=json.dumps(payload if payload is not None else semantic_payload([object_record()])))])

    def create(self, **options):
        self.calls.append(options)
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


def started(client):
    predictor = AnthropicVlmPredictor("example-claude", client_factory=lambda **options: client)
    predictor.start()
    return predictor


def test_lazy_start_exact_rgb_png_public_catalog_and_supported_raw_schema():
    image = pytest.importorskip("PIL.Image")
    client, factories = Client(), []
    predictor = AnthropicVlmPredictor("example-claude", client_factory=lambda **options: factories.append(options) or client)
    assert not factories
    predictor.start()
    predictor.start()
    assert factories == [{"max_retries": 0}]
    value = request()
    prediction = predictor.predict(value)
    assert isinstance(prediction, VlmPrediction)
    assert prediction.objects[0].observation_id == "obs_1"
    assert prediction.objects[0].bbox_xyxy == (0., 0., 3., 2.)
    assert prediction.objects[0].known_asset_id == "saved-tuna"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert set(call) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert call["model"] == "example-claude" and call["max_tokens"] > 0
    assert "English noun phrase" in call["system"]
    assert "not grasp filters" in call["system"]
    assert len(call["messages"]) == 1 and call["messages"][0]["role"] == "user"
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[1]["type"] == "text"
    source = content[0]["source"]
    assert set(source) == {"type", "media_type", "data"}
    assert source["type"] == "base64" and source["media_type"] == "image/png"
    np.testing.assert_array_equal(np.array(image.open(io.BytesIO(base64.b64decode(source["data"])))), value.frame.rgb)
    assert json.loads(content[1]["text"]) == {
        "instruction": value.instruction, "image_size": {"width": 4, "height": 3},
        "assets": [{"asset_id": "saved-tuna", "name": "참치캔", "prompt": "yellow tuna can"}], "policies": [],
    }
    for private in ("private-camera-context", "/local/mesh", "request-7", '"depth"'):
        assert private not in json.dumps(call)
    form = call["output_config"]["format"]
    assert set(form) == {"type", "schema"} and form["type"] == "json_schema"
    schema = form["schema"]
    assert set(schema["required"]) == {"objects", "task_binding", "policy_selection", "coordination"}
    objects = schema["properties"]["objects"]
    assert "maxItems" not in objects and "two" in objects["description"]
    bbox = objects["items"]["properties"]["bbox_xyxy"]["anyOf"][0]
    assert "minItems" not in bbox and "maxItems" not in bbox and "four" in bbox["description"]
    assert objects["items"]["properties"]["known_asset_id"] == {
        "type": ["string", "null"], "enum": ["saved-tuna", None]}
    assert schema["properties"]["task_binding"]["properties"]["target_observation_id"]["type"] == ["string", "null"]
    assert _schema(["saved-tuna"])["properties"]["objects"]["maxItems"] == 2
    assert _schema([])["properties"]["objects"]["items"]["properties"]["bbox_xyxy"]["anyOf"][0]["minItems"] == 4
    predictor.close()
    predictor.close()
    assert client.closed


def test_full_source_b_policy_and_same_object_dual_semantics_preserved():
    payload = semantic_payload([object_record(), dict(object_record(), label="컵", segmentation_prompt="white cup")])
    payload["task_binding"] = {"source_observation_id": "obs_2", "target_observation_id": "obs_1", "intent": "컵을 트레이에 놓기"}
    payload["policy_selection"] = {"task_type": "place_on_surface", "intent_origin": "inferred",
                                   "policy_id": "place-on-tray", "reason": "정리 요청"}
    payload["coordination"] = coordination()
    for item in payload["coordination"]["assignments"]:
        item["observation_id"] = "obs_2"
    client, value = Client(payload), request()
    policies = policy_catalog()
    policies[0]["checkpoint"] = "/private/policy.pth"
    policies[0]["training_source"]["path"] = "/private/source.npz"
    value.policies = tuple(policies)
    predictor = started(client)
    result = predictor.predict(value)
    assert [item.observation_id for item in result.objects] == ["obs_1", "obs_2"]
    assert result.task_binding == payload["task_binding"]
    assert result.policy_selection == payload["policy_selection"]
    assert result.coordination == payload["coordination"]
    assert "/private/" not in json.dumps(client.calls)
    text = json.loads(client.calls[0]["messages"][0]["content"][1]["text"])
    assert text["policies"] == policy_catalog()
    predictor.close()


def test_empty_result_unknown_semantics_and_null_asset_catalog():
    client, value = Client(semantic_payload()), request()
    value.assets = ()
    predictor = started(client)
    result = predictor.predict(value)
    assert result.objects == () and result.task_binding["source_observation_id"] is None
    assert result.coordination["arm_mode"] == "unknown" and result.coordination["assignments"] == []
    schema = client.calls[0]["output_config"]["format"]["schema"]
    assert schema["properties"]["objects"]["items"]["properties"]["known_asset_id"]["enum"] == [None]
    predictor.close()


def test_synthetic_file_key_read_only_at_start_and_not_copied_or_printed(tmp_path, capsys):
    path, options, client = tmp_path / "synthetic-key.txt", [], Client()
    predictor = AnthropicVlmPredictor("example", path, client_factory=lambda **value: options.append(value) or client)
    assert not path.exists()
    path.write_text("  synthetic-anthropic-key \n")
    predictor.start()
    assert options == [{"max_retries": 0, "api_key": "synthetic-anthropic-key"}]
    assert "synthetic-anthropic-key" not in repr(predictor.__dict__)
    assert capsys.readouterr() == ("", "")
    predictor.close()


def test_lazy_sdk_uses_default_env_auth_without_manual_key_loading(monkeypatch):
    options, client = [], Client()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-sdk-env-key")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda **value: options.append(value) or client))
    predictor = AnthropicVlmPredictor("example")
    assert not options
    predictor.start()
    assert options == [{"max_retries": 0}]
    predictor.close()


def test_import_and_construction_need_no_sdk_image_or_gpu_module():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'anthropic', 'openai', 'PIL', 'torch'}:
            raise AssertionError('Unexpected eager import: ' + fullname)
sys.meta_path.insert(0, Block())
from grasp_workbench.models.vlm_anthropic import AnthropicVlmPredictor
AnthropicVlmPredictor('example')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stage", ["constructor", "key-file", "api", "close"])
def test_failures_only_expose_exception_type_without_retry(stage, tmp_path, capsys):
    client = Client()
    if stage == "constructor":
        def factory(**options):
            raise RuntimeError("synthetic-sensitive-detail")
        predictor = AnthropicVlmPredictor("example", client_factory=factory)
        operation, expected = predictor.start, "RuntimeError"
    elif stage == "key-file":
        predictor = AnthropicVlmPredictor("example", tmp_path / "missing", client_factory=lambda **options: client)
        operation, expected = predictor.start, "FileNotFoundError"
    else:
        predictor = started(client)
        if stage == "api":
            client.error = RuntimeError("synthetic-sensitive-detail")
            def operation():
                return predictor.predict(request())
        else:
            client.close_error = RuntimeError("synthetic-sensitive-detail")
            operation = predictor.close
        expected = "RuntimeError"
    with pytest.raises(VlmError) as error:
        operation()
    assert str(error.value) == expected and error.value.__suppress_context__
    assert error.value.__cause__ is None
    assert len(client.calls) == (1 if stage == "api" else 0)
    assert capsys.readouterr() == ("", "")
    predictor.close()


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", "tool_use", "pause_turn", None])
def test_only_completed_end_turn_can_become_a_prediction(reason):
    client = Client()
    client.response.stop_reason = reason
    predictor = started(client)
    with pytest.raises(VlmError, match="^ValueError$"):
        predictor.predict(request())
    assert len(client.calls) == 1
    predictor.close()


def test_text_blocks_are_joined_and_non_text_content_is_not_parsed():
    client = Client()
    text = client.response.content[0].text
    client.response.content = [SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text[:20]),
                               SimpleNamespace(type="text", text=text[20:])]
    predictor = started(client)
    assert predictor.predict(request()).objects[0].observation_id == "obs_1"
    predictor.close()


@pytest.mark.parametrize("payload", [
    semantic_payload([object_record(), object_record(), object_record()]),
    semantic_payload([dict(object_record(), bbox_xyxy=[0, 1, 2])]),
    semantic_payload([dict(object_record(), bbox_xyxy=[0, 0, 99, 99])]),
    semantic_payload([dict(object_record(), known_asset_id="unknown")]),
    semantic_payload([dict(object_record(), segmentation_prompt="")]),
    semantic_payload([dict(object_record(), pose=[0, 0, 1])]),
    semantic_payload([dict(object_record(), observation_id="same"), dict(object_record(), observation_id="same")]),
    dict(semantic_payload([object_record()]), task_binding={"source_observation_id": "missing", "target_observation_id": None, "intent": ""}),
    dict(semantic_payload(), policy_selection={"task_type": "grasp", "intent_origin": "explicit", "policy_id": "unknown", "reason": ""}),
    {"objects": []}, [], "not a JSON object",
])
def test_common_parser_still_enforces_semantic_and_removed_array_constraints(payload):
    client = Client(payload)
    predictor = started(client)
    with pytest.raises(VlmError, match="^ValueError$"):
        predictor.predict(request())
    assert len(client.calls) == 1
    predictor.close()


@pytest.mark.parametrize("content", [[], [SimpleNamespace(type="text", text="")], [SimpleNamespace(type="text", text="not JSON")]])
def test_empty_or_invalid_text_is_sanitized(content):
    client = Client()
    client.response.content = content
    predictor = started(client)
    with pytest.raises(VlmError, match="^JSONDecodeError$"):
        predictor.predict(request())
    predictor.close()


def test_prediction_requires_start():
    with pytest.raises(RuntimeError, match="Start VLM before prediction"):
        AnthropicVlmPredictor("example").predict(request())
