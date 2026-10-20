"""Codex adapter tests use a fake runner only: no CLI, login or inference calls."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.vlm import VlmRequest
from grasp_workbench.models.vlm import VlmError
from grasp_workbench.models.vlm_codex import CodexCliPredictor
from test_semantic import coordination, policy_catalog, semantic_payload


def request():
    return VlmRequest("codex-job", CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
                                              np.full((3, 4), .75, np.float32),
                                              {"camera_id": "private-camera", "sequence": 7}),
                      "테이프와 참치캔", ({"asset_id": "tuna", "name": "참치캔", "prompt": "yellow tuna can",
                                        "private_path": "/private/mesh.glb"},))


def payload():
    return semantic_payload([
        {"label": "테이프", "segmentation_prompt": "brown tape roll", "bbox_xyxy": [0, 0, 2, 3],
         "known_asset_id": None},
        {"label": "참치캔", "segmentation_prompt": "yellow tuna can", "bbox_xyxy": None,
         "known_asset_id": "tuna"},
    ])


def test_one_shot_argv_exact_png_public_catalog_schema_and_temp_cleanup(capsys):
    image = pytest.importorskip("PIL.Image")
    value, calls, directories = request(), [], []

    def run(command, **options):
        calls.append((command, options))
        directory = Path(options["cwd"])
        directories.append(directory)
        assert directory.is_dir() and directory.parent != Path.cwd()
        assert command[:2] == ["custom-codex", "exec"] and command[-1] == "-"
        assert command[command.index("-m") + 1] == "example-model"
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert command[command.index("--disable") + 1] == "shell_tool"
        assert command[command.index("-c") + 1] == 'web_search="disabled"'
        for flag in ("--ephemeral", "--skip-git-repo-check", "--ignore-user-config"):
            assert flag in command
        assert options["capture_output"] is True and options["text"] is True
        assert "timeout" not in options and "env" not in options
        image_path = Path(command[command.index("--image") + 1])
        schema_path = Path(command[command.index("--output-schema") + 1])
        result_path = Path(command[command.index("--output-last-message") + 1])
        assert image_path.parent == schema_path.parent == result_path.parent == directory
        np.testing.assert_array_equal(np.array(image.open(image_path)), value.frame.rgb)
        schema = json.loads(schema_path.read_text())
        assert schema["properties"]["objects"]["maxItems"] == 2
        assert schema["properties"]["objects"]["items"]["properties"]["known_asset_id"]["enum"] == ["tuna", None]
        instructions, raw = options["input"].split("\n", 1)
        for text in ("English noun phrase", "2-6 words", "Do not use tools"):
            assert text in instructions
        assert json.loads(raw) == {"instruction": "테이프와 참치캔", "image_size": {"width": 4, "height": 3},
                                   "assets": [{"asset_id": "tuna", "name": "참치캔", "prompt": "yellow tuna can"}], "policies": []}
        assert "private-camera" not in raw and "/private/mesh.glb" not in raw
        assert "api_key" not in raw and "depth" not in raw and "request_id" not in raw
        result_path.write_text(json.dumps(payload()), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="synthetic-sensitive-stdout", stderr="synthetic-sensitive-stderr")

    predictor = CodexCliPredictor(model="example-model", executable="custom-codex", runner=run)
    predictor.start()
    assert not calls
    prediction = predictor.predict(value)
    objects = prediction.objects
    predictor.close()
    assert len(calls) == 1 and all(not directory.exists() for directory in directories)
    assert [item.segmentation_prompt for item in objects] == ["brown tape roll", "yellow tuna can"]
    assert [item.known_asset_id for item in objects] == [None, "tuna"]
    assert objects[0].bbox_xyxy == (0., 0., 2., 3.)
    assert [item.observation_id for item in objects] == ["obs_1", "obs_2"]
    assert prediction.task_binding["source_observation_id"] == "obs_1"
    assert prediction.coordination["arm_mode"] == "unknown" and prediction.policy_selection["policy_id"] is None
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("mode,expected", [
    ("nonzero", "CodexProcessError"), ("exception", "FileNotFoundError"),
    ("invalid-json", "JSONDecodeError"), ("missing-result", "FileNotFoundError"),
    ("unknown-asset", "ValueError"), ("too-many-objects", "ValueError"),
    ("invalid-box", "ValueError"),
])
def test_failed_cli_or_invalid_output_is_sanitized_without_retries_and_cleans_temp(mode, expected, capsys):
    calls, directories = [], []

    def run(command, **options):
        calls.append(command)
        directories.append(Path(options["cwd"]))
        if mode == "exception":
            raise FileNotFoundError("synthetic-sensitive-message")
        result = Path(command[command.index("--output-last-message") + 1])
        data = payload()
        if mode == "unknown-asset":
            data["objects"][0]["known_asset_id"] = "unknown"
        elif mode == "too-many-objects":
            data["objects"].append(data["objects"][0])
        elif mode == "invalid-box":
            data["objects"][0]["bbox_xyxy"] = [0, 0, 100, 100]
        if mode not in ("missing-result", "nonzero"):
            result.write_text("synthetic-sensitive-message" if mode == "invalid-json" else json.dumps(data))
        return SimpleNamespace(returncode=1 if mode == "nonzero" else 0,
                               stdout="synthetic-sensitive-stdout", stderr="synthetic-sensitive-stderr")

    predictor = CodexCliPredictor(runner=run)
    predictor.start()
    with pytest.raises(VlmError) as error:
        predictor.predict(request())
    predictor.close()
    assert str(error.value) == expected and len(calls) == 1
    assert all(not directory.exists() for directory in directories)
    assert capsys.readouterr() == ("", "")


def test_no_visible_objects_and_empty_catalog_are_valid():
    def run(command, **options):
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(semantic_payload()))
        return SimpleNamespace(returncode=0)

    value = request()
    value.assets = ()
    predictor = CodexCliPredictor(runner=run)
    predictor.start()
    prediction = predictor.predict(value)
    assert prediction.objects == () and prediction.coordination["arm_mode"] == "unknown"
    predictor.close()


def test_cli_restores_full_semantic_prediction_and_only_public_policy_inputs():
    value, calls = request(), []
    value.policies = tuple(policy_catalog())
    value.policies[0]["checkpoint"] = "/private/policy.pth"
    data = payload()
    data["task_binding"] = {"source_observation_id": "obs_2", "target_observation_id": "obs_1", "intent": "물체 배치"}
    data["policy_selection"] = {"task_type": "place_on_surface", "intent_origin": "inferred",
                                "policy_id": "place-on-tray", "reason": "정리 지시"}
    data["coordination"] = coordination()

    def run(command, **options):
        calls.append((command, options))
        public = json.loads(options["input"].split("\n", 1)[1])["policies"]
        assert public == policy_catalog() and "/private" not in options["input"]
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert set(schema["required"]) == {"objects", "task_binding", "policy_selection", "coordination"}
        assert schema["properties"]["policy_selection"]["properties"]["policy_id"]["enum"] == ["place-on-tray", None]
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(data))
        return SimpleNamespace(returncode=0)

    predictor = CodexCliPredictor(runner=run)
    predictor.start()
    prediction = predictor.predict(value)
    predictor.close()
    assert len(calls) == 1 and prediction.task_binding == data["task_binding"]
    assert prediction.policy_selection == data["policy_selection"] and prediction.coordination == data["coordination"]


@pytest.mark.parametrize("executable", ["./fakecodex", "~/fakecodex", "./linkedcodex"])
def test_relative_and_home_executable_paths_survive_temporary_working_directory(tmp_path, monkeypatch, executable):
    # Execute only this synthetic stub, not the real CLI or an inference backend.
    stub = tmp_path / "fakecodex"
    stub.write_text(f"#!{sys.executable}\n" + """
from pathlib import Path
import sys
args = sys.argv[1:]
""" + f"Path(args[args.index('--output-last-message') + 1]).write_text({json.dumps(semantic_payload())!r})\n", encoding="utf-8")
    stub.chmod(0o755)
    (tmp_path / "linkedcodex").symlink_to(stub)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    predictor = CodexCliPredictor(executable=executable)
    predictor.start()
    assert predictor.predict(request()).objects == ()
    predictor.close()
    assert predictor.executable == str(tmp_path / ("linkedcodex" if executable.endswith("linkedcodex") else "fakecodex"))


def test_import_and_lifecycle_do_not_load_sdk_or_read_auth():
    script = """
import sys
for name in ('openai', 'PIL', 'torch', 'iceoryx2'):
    sys.modules[name] = None
from grasp_workbench.models.vlm_codex import CodexCliPredictor
predictor = CodexCliPredictor(executable='nonexistent-not-executed')
predictor.start()
predictor.close()
assert all(sys.modules[name] is None for name in ('openai', 'PIL', 'torch', 'iceoryx2'))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=10)
    assert result.returncode == 0, result.stderr
