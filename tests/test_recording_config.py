"""Recorder config/launch checks never start hardware or load LeRobot."""
import json
import sys

import pytest

from grasp_workbench import config
from grasp_workbench.main import build_commands, build_parser


def _load(tmp_path, contents):
    path = tmp_path / "settings.yaml"
    path.write_text(contents, encoding="utf-8")
    return path, config.load_run_config(path)


def _option(command, name):
    return command[command.index(name) + 1]


def test_recording_paths_are_yaml_relative_and_do_not_resolve_python_symlink(tmp_path):
    python = tmp_path / ".venv-recorder/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    _path, loaded = _load(tmp_path, """
recording:
  enabled: true
  python: .venv-recorder/bin/python
  output_dir: saved/datasets
  fps: 15
  source: camera
""")
    assert loaded["recording"] is True
    assert loaded["recording_python"] == str(python)
    assert loaded["recording_output_dir"] == str(tmp_path / "saved/datasets")
    assert loaded["recording_fps"] == 15 and loaded["recording_source"] == "camera"


def test_recording_partial_local_override_and_null_python(tmp_path, monkeypatch):
    shared = tmp_path / "config.yaml"
    shared.write_text("recording: {python: python3, output_dir: recordings, fps: 15}\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text(
        "recording: {enabled: true, python: null, source: camera}\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", shared)
    loaded = config.load_run_config()
    assert loaded["recording"] is True and loaded["recording_python"] == sys.executable
    assert loaded["recording_source"] == "camera" and loaded["recording_fps"] == 15
    assert loaded["recording_output_dir"] == str(tmp_path / "recordings")


def test_recording_default_output_directory_is_repository_relative(tmp_path):
    _path, loaded = _load(tmp_path, "{}\n")
    assert loaded["recording"] is False and loaded["recording_python"] == sys.executable
    assert loaded["recording_output_dir"] == str(config.ROOT / "runs/datasets")
    assert loaded["recording_fps"] == 30 and loaded["recording_source"] == "pose"


@pytest.mark.parametrize("options", [
    "enabled: 1", "enabled: 'true'", "python: ''", "python: 1", "output_dir: null",
    "output_dir: ''", "fps: 0", "fps: -1", "fps: 1.5", "fps: true", "source: sam3", "source: null",
])
def test_recording_rejects_invalid_yaml_options(tmp_path, options):
    with pytest.raises(ValueError, match="recording\\."):
        _load(tmp_path, f"recording: {{{options}}}\n")


@pytest.mark.parametrize("aprilcube", [False, True])
def test_recording_is_independent_and_forces_ui_without_enabling_models(tmp_path, aprilcube):
    _path, loaded = _load(tmp_path, f"""
ui: {{enabled: false}}
recording: {{enabled: true, python: python3, source: camera, fps: 15}}
aprilcube: {{enabled: {str(aprilcube).lower()}}}
""")
    args = build_parser(loaded).parse_args(["run"])
    commands = build_commands(args)
    modules = [_option(command, "-m").rsplit(".", 1)[-1] for command in commands]
    assert modules == (["worker_camera_ui", "worker_data_recorder", "worker_aprilcube",
                        "worker_aprilcube_overlay", "worker_camera"] if aprilcube
                       else ["worker_camera_ui", "worker_data_recorder", "worker_camera"])
    assert "--recording" in commands[0]
    recorder = commands[1]
    assert recorder[0] == "python3"
    assert _option(recorder, "--source") == "camera" and _option(recorder, "--fps") == "15"
    assert _option(recorder, "--output-dir") == str(config.ROOT / "runs/datasets")
    assert all(_option(command, "--service") == args.service for command in commands)
    assert not any(loaded[name] for name in config.APRILCUBE_DISABLED_WORKERS)
    assert "--frames" not in recorder and "--timeout" not in recorder


def test_recording_cli_overrides_yaml_settings(tmp_path):
    path, loaded = _load(tmp_path, "recording: {python: ignored, output_dir: ignored, fps: 15, source: camera}\n")
    args = build_parser(loaded).parse_args([
        "run", "--config", str(path), "--recording", "--recording-python", "recorder-python",
        "--recording-output-dir", "/example/datasets", "--recording-fps", "24", "--recording-source", "pose",
    ])
    ui, recorder, _camera = build_commands(args)
    assert "--recording" in ui and recorder[0] == "recorder-python"
    assert _option(recorder, "--output-dir") == "/example/datasets"
    assert _option(recorder, "--fps") == "24" and _option(recorder, "--source") == "pose"


def test_recording_cli_source_choices_are_explicit():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--recording-source", "sam3"])


def test_recording_preserves_explicit_model_flags_without_enabling_other_models(tmp_path):
    _path, loaded = _load(tmp_path, "recording: {enabled: true}\nvlm: {enabled: true}\n")
    assert loaded["recording"] is True and loaded["vlm"] is True
    assert not any(loaded[name] for name in ("sam3", "sam3d", "foundationpose", "graspgenx", "grasp_selection"))


def test_recording_run_metadata_uses_only_public_whitelisted_settings(tmp_path):
    _path, loaded = _load(tmp_path, """
camera: {type: zed, serial: private-serial, resolution: HD1080, fps: 15}
recording: {enabled: true}
aprilcube: {enabled: true, fast: false, enable_filter: false, config_a: private/cube.json}
vlm: {api_key_file: private/provider-key, policy_catalog: private/catalog.json}
""")
    commands = build_commands(build_parser(loaded).parse_args(["run"]))
    recorder = commands[1]
    metadata = _option(recorder, "--run-metadata-json")
    assert json.loads(metadata) == {
        "camera": {"type": "zed", "resolution": "HD1080", "fps": 15},
        "aprilcube": {"enabled": True, "fast": False, "enable_filter": False},
    }
    assert "private" not in metadata and "serial" not in metadata
