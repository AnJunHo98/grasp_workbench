"""YAML tests use temporary files only, never a developer's private config."""
from pathlib import Path
import subprocess
import sys

import pytest

from grasp_workbench import config


@pytest.fixture
def shared(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", path)
    return path


def test_fixed_defaults_and_null_python(shared):
    assert config.load_run_config() == {
        "camera": "zed", "serial": None, "resolution": "HD720", "fps": 30,
        "service": "grasp_workbench/camera/rgbd", "ui": True, "sam3": False,
        "sam3_python": sys.executable, "sam3_checkpoint": None, "sam3_device": "cuda",
        "sam3d": False, "sam3d_python": sys.executable,
        "sam3d_source_dir": str(config.ROOT / "third_party/sam-3d-objects"),
        "sam3d_config_path": None, "sam3d_output_dir": str(config.ROOT / "runs/meshes"),
        "sam3d_seed": 42, "sam3d_depth_source": "depth", "sam3d_texture_baking": False,
        "foundationpose": False, "foundationpose_python": sys.executable,
        "foundationpose_source_dir": str(config.ROOT / "third_party/FoundationPose"),
        "foundationpose_device": "cuda:0", "foundationpose_registration_iterations": 5,
        "foundationpose_tracking_iterations": 1,
        "vlm": False, "vlm_python": sys.executable, "vlm_model": "gpt-6.1-sol",
        "vlm_backend": "openai_api", "vlm_codex_executable": "codex",
        "vlm_api_key_file": None, "asset_dir": str(config.ROOT / "runs/assets"),
        "vlm_policy_catalog": None, "vlm_session_dir": str(config.ROOT / "runs/tasks"),
        "graspgenx": False, "graspgenx_python": sys.executable,
        "graspgenx_source_dir": str(config.ROOT / "third_party/GraspGenX"),
        "graspgenx_checkpoint_dir": None, "graspgenx_assets_dir": None,
        "graspgenx_gripper_id": "franka_panda", "graspgenx_output_dir": str(config.ROOT / "runs/grasps"),
        "graspgenx_settings": {},
        "grasp_selection": False, "grasp_selection_python": sys.executable,
        "grasp_selection_robot_config": None, "grasp_selection_settings": {},
        "frames": 0, "timeout": 0.0,
    }


def test_local_partial_override_retains_shared_and_static_fields(shared):
    shared.write_text("camera: {type: realsense, fps: 15}\nipc: {service: shared/rgbd}\n"
                      "sam3: {device: cpu}\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text(
        "camera: {serial: 12345}\nui: {enabled: false}\nsam3: {enabled: true}\n",
        encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["camera"] == "realsense" and loaded["fps"] == 15
    assert loaded["serial"] == "12345" and loaded["resolution"] == "HD720"
    assert loaded["service"] == "shared/rgbd"
    assert loaded["ui"] is False and loaded["sam3"] is True and loaded["sam3_device"] == "cpu"


def test_default_path_does_not_depend_on_current_directory(shared, tmp_path, monkeypatch):
    shared.write_text("camera: {fps: 17}\nsam3: {checkpoint: models/sam3.pt}\n", encoding="utf-8")
    elsewhere = tmp_path / "different-cwd"
    elsewhere.mkdir()
    (elsewhere / "config.yaml").write_text("camera: {fps: 99}\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    loaded = config.load_run_config()
    assert loaded["fps"] == 17
    assert loaded["sam3_checkpoint"] == str(shared.parent / "models/sam3.pt")


def test_explicit_file_ignores_default_and_its_own_local_overlay(shared, tmp_path):
    shared.write_text("camera: {fps: 99}\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("not: valid\n", encoding="utf-8")
    directory = tmp_path / "explicit"
    directory.mkdir()
    selected = directory / "selected.yaml"
    selected.write_text("camera: {fps: 12}\n", encoding="utf-8")
    (directory / "config.local.yaml").write_text("also: invalid\n", encoding="utf-8")
    assert config.load_run_config(selected)["fps"] == 12
    # Explicitly selecting the normal default file still disables its overlay.
    assert config.load_run_config(shared)["fps"] == 99


def test_paths_resolve_relative_to_defining_yaml_and_preserve_unset_shared_path(shared):
    shared.write_text("sam3: {python: .venv-sam3/bin/python, checkpoint: models/shared.pt}\n",
                      encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("sam3: {checkpoint: local/models.pt}\n", encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["sam3_python"] == str(shared.parent / ".venv-sam3/bin/python")
    assert loaded["sam3_checkpoint"] == str(shared.parent / "local/models.pt")


@pytest.mark.parametrize("python", ["python3", "python3.12"])
def test_bare_python_command_remains_on_path(shared, python):
    shared.write_text(f"sam3: {{python: {python}}}\n", encoding="utf-8")
    assert config.load_run_config()["sam3_python"] == python


def test_tilde_expands_and_absolute_paths_remain_absolute(shared, tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    shared.write_text("sam3: {python: ~/env/bin/python, checkpoint: ~/models/sam3.pt}\n", encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["sam3_python"] == str(home / "env/bin/python")
    assert loaded["sam3_checkpoint"] == str(home / "models/sam3.pt")
    absolute = tmp_path / "other" / "checkpoint.pt"
    shared.write_text(f"sam3: {{checkpoint: {absolute}}}\n", encoding="utf-8")
    assert config.load_run_config()["sam3_checkpoint"] == str(absolute)


def test_local_null_resets_python_and_checkpoint(shared):
    shared.write_text("sam3: {python: .venv/bin/python, checkpoint: models/sam3.pt}\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("sam3: {python: null, checkpoint: null}\n", encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["sam3_python"] == sys.executable and loaded["sam3_checkpoint"] is None


def test_python_symlink_keeps_venv_executable_path(shared):
    base = shared.parent / "base-python"
    base.write_text("not-executed", encoding="utf-8")
    venv_python = shared.parent / ".venv-sam3/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base)
    shared.write_text("sam3: {python: .venv-sam3/bin/python}\n", encoding="utf-8")
    assert config.load_run_config()["sam3_python"] == str(venv_python)


def test_sam3d_paths_are_yaml_relative_and_python_preserves_venv_symlink(shared, tmp_path, monkeypatch):
    base = tmp_path / "base-python"
    base.write_text("not-executed", encoding="utf-8")
    python = shared.parent / ".venv-sam3d/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base)
    shared.write_text("""
sam3d:
  enabled: true
  python: .venv-sam3d/bin/python
  source_dir: external/sam-3d-objects
  config_path: weights/pipeline.yaml
  output_dir: outputs/mesh assets
  seed: 17
""", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = config.load_run_config(shared)
    assert loaded["sam3d"] is True and loaded["sam3d_python"] == str(python)
    assert loaded["sam3d_source_dir"] == str(shared.parent / "external/sam-3d-objects")
    assert loaded["sam3d_config_path"] == str(shared.parent / "weights/pipeline.yaml")
    assert loaded["sam3d_output_dir"] == str(shared.parent / "outputs/mesh assets")
    assert loaded["sam3d_seed"] == 17


def test_sam3d_partial_local_override_retains_paths_and_null_source_uses_portable_default(shared):
    shared.write_text("sam3d: {source_dir: vendor/source, config_path: weights/pipeline.yaml, seed: 7}\n",
                      encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("sam3d: {enabled: true, source_dir: null, seed: 0}\n",
                                                    encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["sam3d"] is True and loaded["sam3d_seed"] == 0
    assert loaded["sam3d_source_dir"] == str(config.ROOT / "third_party/sam-3d-objects")
    assert loaded["sam3d_config_path"] == str(shared.parent / "weights/pipeline.yaml")
    assert loaded["sam3d_output_dir"] == str(config.ROOT / "runs/meshes")


def test_sam3d_defaults_do_not_depend_on_explicit_yaml_directory(shared):
    loaded = config.load_run_config(shared)
    assert loaded["sam3d_python"] == sys.executable
    assert loaded["sam3d_source_dir"] == str(config.ROOT / "third_party/sam-3d-objects")
    assert loaded["sam3d_output_dir"] == str(config.ROOT / "runs/meshes")


def test_sam3d_depth_and_texture_local_options_override_shared_values(shared):
    shared.write_text("sam3d: {depth_source: depth, texture_baking: true, seed: 17}\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text(
        "sam3d: {depth_source: monocular, texture_baking: false}\n", encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["sam3d_depth_source"] == "monocular"
    assert loaded["sam3d_texture_baking"] is False
    assert loaded["sam3d_seed"] == 17 and loaded["sam3d"] is False


def test_foundationpose_paths_are_yaml_relative_and_preserve_python_symlink(shared, monkeypatch):
    base = shared.parent / "base-python"
    base.write_text("not-executed", encoding="utf-8")
    python = shared.parent / ".venv-foundationpose/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base)
    shared.write_text("""
foundationpose:
  enabled: true
  python: .venv-foundationpose/bin/python
  source_dir: vendor/FoundationPose
  device: cuda:2
  registration_iterations: 7
  tracking_iterations: 2
""", encoding="utf-8")
    elsewhere = shared.parent / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = config.load_run_config(shared)
    assert loaded["foundationpose"] is True
    assert loaded["foundationpose_python"] == str(python)
    assert loaded["foundationpose_source_dir"] == str(shared.parent / "vendor/FoundationPose")
    assert loaded["foundationpose_device"] == "cuda:2"
    assert loaded["foundationpose_registration_iterations"] == 7
    assert loaded["foundationpose_tracking_iterations"] == 2


def test_foundationpose_local_overlay_retains_fields_and_resets_null_paths(shared):
    shared.write_text("foundationpose: {python: local/python, source_dir: local/source, device: cpu, "
                      "registration_iterations: 9}\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text(
        "foundationpose: {enabled: true, python: null, source_dir: null, tracking_iterations: 3}\n",
        encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["foundationpose"] is True
    assert loaded["foundationpose_python"] == sys.executable
    assert loaded["foundationpose_source_dir"] == str(config.ROOT / "third_party/FoundationPose")
    assert loaded["foundationpose_device"] == "cpu"
    assert loaded["foundationpose_registration_iterations"] == 9
    assert loaded["foundationpose_tracking_iterations"] == 3


def test_foundationpose_bare_python_and_portable_default_source(shared):
    shared.write_text("foundationpose: {enabled: true, python: python3}\n", encoding="utf-8")
    loaded = config.load_run_config(shared)
    assert loaded["foundationpose_python"] == "python3"
    assert loaded["foundationpose_source_dir"] == str(config.ROOT / "third_party/FoundationPose")


def test_vlm_paths_are_yaml_relative_and_python_keeps_venv_symlink(shared):
    python = shared.parent / ".venv-vlm/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    shared.write_text("""
vlm:
  enabled: true
  python: .venv-vlm/bin/python
  model: example-model
  api_key_file: .secrets/provider.key
  asset_dir: objects/library
""", encoding="utf-8")
    loaded = config.load_run_config(shared)
    assert loaded["vlm"] is True and loaded["vlm_python"] == str(python)
    assert loaded["vlm_model"] == "example-model"
    assert loaded["vlm_api_key_file"] == str(shared.parent / ".secrets/provider.key")
    assert loaded["asset_dir"] == str(shared.parent / "objects/library")
    assert not Path(loaded["vlm_api_key_file"]).exists()  # Configuration does not open credentials.


def test_vlm_partial_local_overlay_preserves_model_and_paths(shared):
    shared.write_text("vlm: {python: python3, model: configured, api_key_file: keys/shared, asset_dir: library}\n",
                      encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("vlm: {enabled: true, api_key_file: null}\n",
                                                    encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["vlm"] is True and loaded["vlm_python"] == "python3"
    assert loaded["vlm_model"] == "configured" and loaded["vlm_api_key_file"] is None
    assert loaded["asset_dir"] == str(shared.parent / "library")


def test_unset_vlm_assets_use_portable_repo_default_for_explicit_config(shared):
    assert config.load_run_config(shared)["asset_dir"] == str(config.ROOT / "runs/assets")


def test_codex_cli_settings_preserve_bare_command_and_model(shared):
    shared.write_text("vlm: {enabled: true, backend: codex_cli, codex_executable: codex, model: configured}\n",
                      encoding="utf-8")
    loaded = config.load_run_config(shared)
    assert loaded["vlm_backend"] == "codex_cli"
    assert loaded["vlm_codex_executable"] == "codex"
    assert loaded["vlm_model"] == "configured"


def test_anthropic_settings_preserve_model_and_yaml_relative_key_path(shared):
    shared.write_text("vlm: {enabled: true, backend: anthropic_api, model: configured-claude, "
                      "api_key_file: keys/anthropic.key}\n", encoding="utf-8")
    loaded = config.load_run_config(shared)
    assert loaded["vlm_backend"] == "anthropic_api" and loaded["vlm"] is True
    assert loaded["vlm_model"] == "configured-claude"
    assert loaded["vlm_api_key_file"] == str(shared.parent / "keys/anthropic.key")
    assert not Path(loaded["vlm_api_key_file"]).exists()


def test_backend_change_does_not_automatically_change_model(shared):
    shared.write_text("vlm: {backend: anthropic_api}\n", encoding="utf-8")
    loaded = config.load_run_config(shared)
    assert loaded["vlm_model"] == "gpt-6.1-sol"
    assert loaded["vlm_api_key_file"] is None


def test_codex_executable_path_is_yaml_relative_and_preserves_symlink(shared):
    executable = shared.parent / "tools/codex"
    executable.parent.mkdir()
    executable.symlink_to("/not-executed/codex")
    shared.write_text("vlm: {backend: codex_cli, codex_executable: tools/codex}\n", encoding="utf-8")
    assert config.load_run_config(shared)["vlm_codex_executable"] == str(executable)


def test_codex_partial_local_override_preserves_other_vlm_settings(shared):
    shared.write_text("vlm: {enabled: true, model: shared-model, api_key_file: keys/shared}\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("vlm: {backend: codex_cli, codex_executable: tools/local-codex}\n",
                                                    encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["vlm_backend"] == "codex_cli"
    assert loaded["vlm_codex_executable"] == str(shared.parent / "tools/local-codex")
    assert loaded["vlm_model"] == "shared-model"


def test_graspgenx_paths_settings_and_python_symlink(shared):
    executable = shared.parent / ".venv-graspgenx/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    shared.write_text("""
graspgenx:
  enabled: true
  python: .venv-graspgenx/bin/python
  source_dir: vendor/GraspGenX
  checkpoint_dir: weights/grasp
  assets_dir: examples/grippers
  gripper_id: custom_gripper
  output_dir: outputs/grasps
  settings: {num_grasps: 42, nested: {enabled: true}, scales: [1, 2]}
""", encoding="utf-8")
    loaded = config.load_run_config(shared)
    assert loaded["graspgenx"] is True and loaded["graspgenx_python"] == str(executable)
    for field, relative in (("source_dir", "vendor/GraspGenX"), ("checkpoint_dir", "weights/grasp"),
                            ("assets_dir", "examples/grippers"), ("output_dir", "outputs/grasps")):
        assert loaded["graspgenx_" + field] == str(shared.parent / relative)
    assert loaded["graspgenx_gripper_id"] == "custom_gripper"
    assert loaded["graspgenx_settings"] == {"num_grasps": 42, "nested": {"enabled": True}, "scales": [1, 2]}


def test_graspgenx_partial_overlay_and_mutable_defaults_are_independent(shared):
    config.load_run_config(shared)["graspgenx_settings"]["not-shared"] = True
    assert config.load_run_config(shared)["graspgenx_settings"] == {}
    shared.write_text("graspgenx: {python: python3, source_dir: vendor/source, settings: {num_grasps: 8}}\n",
                      encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("graspgenx: {enabled: true, source_dir: null}\n", encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["graspgenx"] is True and loaded["graspgenx_python"] == "python3"
    assert loaded["graspgenx_source_dir"] == str(config.ROOT / "third_party/GraspGenX")
    assert loaded["graspgenx_output_dir"] == str(config.ROOT / "runs/grasps")
    assert loaded["graspgenx_settings"] == {"num_grasps": 8}


def test_vlm_semantic_paths_are_yaml_relative_without_reading_catalog(shared):
    shared.write_text("vlm: {policy_catalog: catalogs/policies.json, session_dir: outputs/tasks}\n", encoding="utf-8")
    loaded = config.load_run_config(shared)
    assert loaded["vlm_policy_catalog"] == str(shared.parent / "catalogs/policies.json")
    assert loaded["vlm_session_dir"] == str(shared.parent / "outputs/tasks")
    assert not Path(loaded["vlm_policy_catalog"]).exists()


def test_vlm_semantic_partial_overlay_resets_optional_catalog_only(shared):
    shared.write_text("vlm: {policy_catalog: policies.json, session_dir: tasks}\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("vlm: {policy_catalog: null}\n", encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["vlm_policy_catalog"] is None
    assert loaded["vlm_session_dir"] == str(shared.parent / "tasks")


def test_zero_limits_and_false_boolean_are_preserved(shared):
    shared.write_text("ui: {enabled: false}\nsam3: {enabled: false}\nrun: {frames: 0, timeout: 0}\n",
                      encoding="utf-8")
    loaded = config.load_run_config()
    assert loaded["ui"] is False and loaded["sam3"] is False
    assert loaded["frames"] == 0 and loaded["timeout"] == 0


@pytest.mark.parametrize("text", [
    "unknown_section: {}", "camera: {fpps: 30}", "sam3: {checkpont: path.pt}",
    "[camera, ui]", "camera: []", "camera: null", "camera: {type: usb}",
    "camera: {fps: 0}", "camera: {fps: -1}", "camera: {fps: 1.5}", "camera: {fps: true}",
    "camera: {serial: true}", "camera: {serial: 1.5}", "camera: {serial: ''}",
    "camera: {resolution: ''}", "camera: {resolution: 720}",
    "ui: {enabled: 1}", "ui: {enabled: 'false'}", "sam3: {enabled: null}",
    "sam3: {python: 3}", "sam3: {checkpoint: []}", "sam3: {device: ''}",
    "ipc: {service: ''}", "run: {frames: -1}", "run: {frames: false}",
    "run: {timeout: -1}", "run: {timeout: true}", "run: {timeout: .nan}",
    "run: {timeout: .inf}", "run: {timeout: '-1'}",
    "sam3d: {enable: true}", "sam3d: {enabled: 1}", "sam3d: {python: 3}",
    "sam3d: {source_dir: []}", "sam3d: {config_path: 7}", "sam3d: {output_dir: null}",
    "sam3d: {output_dir: ''}", "sam3d: {seed: false}", "sam3d: {seed: 1.5}",
    "sam3d: {depth_source: auto}", "sam3d: {depth_source: null}", "sam3d: {depth_source: false}",
    "sam3d: {texture_baking: null}", "sam3d: {texture_baking: 1}", "sam3d: {texture_baking: 'false'}",
    "foundationpose: {enable: true}", "foundationpose: {enabled: 1}",
    "foundationpose: {python: 3}", "foundationpose: {source_dir: []}",
    "foundationpose: {device: ''}", "foundationpose: {registration_iterations: false}",
    "foundationpose: {tracking_iterations: 1.5}",
    "vlm: {enabled: 1}", "vlm: {python: 3}", "vlm: {model: ''}",
    "vlm: {api_key_file: []}", "vlm: {asset_dir: null}", "vlm: {asset_dir: ''}",
    "vlm: {backend: unknown}", "vlm: {backend: null}", "vlm: {backend: true}",
    "vlm: {codex_executable: null}", "vlm: {codex_executable: ''}", "vlm: {codex_executable: []}",
    "vlm: {policy_catalog: []}", "vlm: {session_dir: null}", "vlm: {session_dir: ''}",
    "graspgenx: {enabled: 1}", "graspgenx: {python: []}", "graspgenx: {source_dir: 1}",
    "graspgenx: {checkpoint_dir: 1}", "graspgenx: {assets_dir: []}", "graspgenx: {gripper_id: ''}",
    "graspgenx: {output_dir: null}", "graspgenx: {settings: []}", "graspgenx: {settings: null}",
])
def test_structural_type_and_numeric_errors(shared, text):
    shared.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        config.load_run_config(shared)


@pytest.mark.parametrize("text", ["camera: [broken\n", "\x00"])
def test_syntax_error_is_value_error_with_file_path(shared, text):
    shared.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML") as error:
        config.load_run_config(shared)
    assert str(shared) in str(error.value)


def test_unsafe_yaml_tags_are_rejected(shared):
    shared.write_text("!!python/object/apply:os.system ['not-executed']", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        config.load_run_config(shared)


def test_missing_explicit_file_preserves_io_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_run_config(tmp_path / "missing.yaml")


def test_loader_does_not_require_camera_model_or_ui_packages():
    script = """
import sys
for name in ('iceoryx2', 'pyzed.sl', 'pyrealsense2', 'torch', 'PyQt6'):
    sys.modules[name] = None
from grasp_workbench.config import load_run_config
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
