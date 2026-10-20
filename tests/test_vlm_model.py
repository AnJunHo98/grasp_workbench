"""Fake Responses client only: never load a real credential or call the API."""
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.vlm import VlmRequest
from grasp_workbench.models.vlm import OfficialVlmPredictor, VlmError
from test_semantic import coordination, policy_catalog, semantic_payload


def request():
    frame = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3), np.full((3, 4), 9.9, np.float32),
                        {"camera_id": "private-camera-context", "sequence": 3})
    return VlmRequest("request-7", frame, "테이프와 참치캔", (
        {"asset_id": "saved-tuna", "name": "참치캔", "prompt": "yellow tuna can", "private_path": "/local/mesh"},))


def object_record():
    return {"label": "참치캔", "segmentation_prompt": "yellow tuna can",
            "bbox_xyxy": [0, 0, 3, 2], "known_asset_id": "saved-tuna"}


class Client:
    def __init__(self):
        self.calls, self.closed = [], False
        self.response = SimpleNamespace(status="completed", output_text=json.dumps(semantic_payload([object_record()])))
        self.error = None
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


def test_lazy_client_once_exact_png_public_catalog_english_prompt_and_structured_output():
    image = pytest.importorskip("PIL.Image")
    client, factories = Client(), []
    predictor = OfficialVlmPredictor("example-model", client_factory=lambda **options: factories.append(options) or client)
    assert not factories
    predictor.start()
    predictor.start()
    assert factories == [{"max_retries": 0}]
    value = request()
    prediction = predictor.predict(value)
    objects = prediction.objects
    assert len(client.calls) == 1 and objects[0].known_asset_id == "saved-tuna"
    assert objects[0].bbox_xyxy == (0., 0., 3., 2.)
    assert objects[0].observation_id == "obs_1"
    assert prediction.task_binding["source_observation_id"] == "obs_1"
    assert prediction.policy_selection["policy_id"] is None
    assert prediction.coordination["arm_mode"] == "unknown"
    call = client.calls[0]
    assert call["model"] == "example-model" and "English" in call["instructions"]
    for phrase in ("noun phrase", "2-6 words", "category", "color/material", "full sentences",
                   "relative location", "support objects", "brand/writing details", "red mug", "blue plastic bottle"):
        assert phrase in call["instructions"]
    assert call["store"] is False
    content = call["input"][0]["content"]
    text = json.loads(content[0]["text"])
    assert text == {"instruction": "테이프와 참치캔", "image_size": {"width": 4, "height": 3},
                    "assets": [{"asset_id": "saved-tuna", "name": "참치캔", "prompt": "yellow tuna can"}], "policies": []}
    assert "private-camera-context" not in json.dumps(call) and "/local/mesh" not in json.dumps(call)
    assert "depth" not in text and "frame" not in text and "request_id" not in text
    prefix, encoded = content[1]["image_url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    np.testing.assert_array_equal(np.array(image.open(io.BytesIO(base64.b64decode(encoded)))), value.frame.rgb)
    form = call["text"]["format"]
    assert form["type"] == "json_schema" and form["strict"] is True
    schema = form["schema"]["properties"]["objects"]
    assert schema["maxItems"] == 2
    assert schema["items"]["properties"]["known_asset_id"]["enum"] == ["saved-tuna", None]
    assert set(form["schema"]["required"]) == {"objects", "task_binding", "policy_selection", "coordination"}
    predictor.close()
    predictor.close()
    assert client.closed


def test_synthetic_key_file_is_only_read_at_start_not_stored_or_printed(tmp_path, capsys):
    # This file is test-owned synthetic text, not the application's real key file.
    path = tmp_path / "synthetic-key.txt"
    predictor = OfficialVlmPredictor(api_key_file=path, client_factory=lambda **options: Client())
    assert not path.exists() and "synthetic-content" not in repr(predictor)
    path.write_text("  synthetic-content  \n")
    predictor.start()
    assert "synthetic-content" not in repr(predictor.__dict__)
    assert capsys.readouterr() == ("", "")
    predictor.close()


def test_sdk_constructor_error_only_exposes_exception_type():
    def broken(**options):
        raise RuntimeError("synthetic-sensitive-message")

    predictor = OfficialVlmPredictor(client_factory=broken)
    with pytest.raises(VlmError) as error:
        predictor.start()
    assert str(error.value) == "RuntimeError" and error.value.__suppress_context__


def test_api_exception_has_no_message_or_automatic_retry(capsys):
    client = Client()
    client.error = RuntimeError("synthetic-sensitive-message")
    predictor = OfficialVlmPredictor(client_factory=lambda **options: client)
    predictor.start()
    with pytest.raises(VlmError) as error:
        predictor.predict(request())
    assert str(error.value) == "RuntimeError" and len(client.calls) == 1
    assert capsys.readouterr() == ("", "")
    predictor.close()


@pytest.mark.parametrize("payload", [
    semantic_payload([object_record(), object_record(), object_record()]),
    semantic_payload([dict(object_record(), known_asset_id="not-in-catalog")]),
    semantic_payload([dict(object_record(), bbox_xyxy=[0, 0, 99, 99])]),
    semantic_payload([dict(object_record(), segmentation_prompt="")]),
    semantic_payload([dict(object_record(), pose=[0, 0, 1])]),
    {"objects": None}, {"unexpected": []}, [], {"objects": [object_record()]},
])
def test_invalid_semantic_outputs_do_not_become_object_bindings(payload):
    client = Client()
    client.response.output_text = json.dumps(payload)
    predictor = OfficialVlmPredictor(client_factory=lambda **options: client)
    predictor.start()
    with pytest.raises(VlmError):
        predictor.predict(request())
    assert len(client.calls) == 1
    predictor.close()


@pytest.mark.parametrize("status,output", [("incomplete", ""), ("completed", ""), ("completed", "not JSON")])
def test_incomplete_refusal_or_malformed_response_is_a_sanitized_error(status, output):
    client = Client()
    client.response.status, client.response.output_text = status, output
    predictor = OfficialVlmPredictor(client_factory=lambda **options: client)
    predictor.start()
    with pytest.raises(VlmError) as error:
        predictor.predict(request())
    assert str(error.value) in ("ValueError", "JSONDecodeError")
    predictor.close()


def test_empty_catalog_and_no_visible_objects_are_supported():
    client = Client()
    client.response.output_text = json.dumps(semantic_payload())
    predictor = OfficialVlmPredictor(client_factory=lambda **options: client)
    predictor.start()
    value = request()
    value.assets = ()
    prediction = predictor.predict(value)
    assert prediction.objects == () and prediction.task_binding["source_observation_id"] is None
    assert prediction.coordination["arm_mode"] == "unknown"
    predictor.close()


def test_full_source_target_policy_and_same_source_dual_arm_metadata_are_returned():
    client = Client()
    value = request()
    value.policies = tuple(policy_catalog())
    value.policies[0]["checkpoint"] = "/private/policy.pth"
    value.policies[0]["training_source"]["private_path"] = "/private/training.npz"
    data = semantic_payload([object_record(), object_record()])
    data["task_binding"] = {"source_observation_id": "obs_1", "target_observation_id": "obs_2", "intent": "물체를 트레이에 놓기"}
    data["policy_selection"] = {"task_type": "place_on_surface", "intent_origin": "inferred",
                                "policy_id": "place-on-tray", "reason": "정리 지시를 배치로 해석"}
    data["coordination"] = coordination()
    client.response.output_text = json.dumps(data)
    predictor = OfficialVlmPredictor(client_factory=lambda **options: client)
    predictor.start()
    prediction = predictor.predict(value)
    assert prediction.task_binding == data["task_binding"]
    assert prediction.policy_selection == data["policy_selection"]
    assert prediction.coordination == data["coordination"]
    assert [item.known_asset_id for item in prediction.objects] == ["saved-tuna", "saved-tuna"]
    call = client.calls[0]
    public = json.loads(call["input"][0]["content"][0]["text"])["policies"]
    assert public == policy_catalog()
    assert "/private" not in json.dumps(call)
    instructions = call["instructions"]
    for text in ("Source/Target", "grasp-only", "place_on_surface", "grasp_region", "allow_relative_motion",
                 "maintain_relative_pose", "reference_state", "semantic descriptions only", "concise Korean"):
        assert text in instructions
    assert call["text"]["format"]["schema"]["properties"]["policy_selection"]["properties"]["policy_id"]["enum"] == ["place-on-tray", None]
    predictor.close()


@pytest.mark.parametrize("change", ["duplicate-observation", "empty-observation", "missing-observation", "unknown-source",
                                    "unknown-arm-object", "unknown-policy", "mismatched-policy-task", "extra-numeric-constraint"])
def test_real_backend_strict_full_semantics_checks_current_ids_and_supplied_policies(change):
    value, client = request(), Client()
    value.policies = tuple(policy_catalog())
    data = semantic_payload([object_record(), object_record()])
    if change == "duplicate-observation":
        data["objects"][1]["observation_id"] = "obs_1"
    elif change == "empty-observation":
        data["objects"][1]["observation_id"] = ""
    elif change == "missing-observation":
        data["objects"][1].pop("observation_id")
    elif change == "unknown-source":
        data["task_binding"]["source_observation_id"] = "not-visible"
    elif change == "unknown-arm-object":
        data["coordination"] = coordination()
        data["coordination"]["assignments"][0]["observation_id"] = "not-visible"
    elif change in ("unknown-policy", "mismatched-policy-task"):
        data["policy_selection"]["policy_id"] = "invented" if change == "unknown-policy" else "place-on-tray"
    else:
        data["coordination"] = coordination()
        data["coordination"]["constraints"][0]["pose"] = [0, 0, 1]
    client.response.output_text = json.dumps(data)
    predictor = OfficialVlmPredictor(client_factory=lambda **options: client)
    predictor.start()
    with pytest.raises(VlmError, match="^ValueError$"):
        predictor.predict(value)
    assert len(client.calls) == 1
    predictor.close()


def test_model_import_constructor_and_worker_help_do_not_load_sdk_or_keys():
    script = """
import sys
for name in ('openai', 'PIL', 'torch', 'iceoryx2'):
    sys.modules[name] = None
from grasp_workbench.models.vlm import OfficialVlmPredictor
OfficialVlmPredictor(api_key_file='file-that-does-not-exist')
from grasp_workbench.workers.worker_vlm import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert all(sys.modules[name] is None for name in ('openai','PIL','torch','iceoryx2'))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout and "--api-key-file" in result.stdout
