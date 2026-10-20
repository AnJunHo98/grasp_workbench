import sys
from types import SimpleNamespace

import pytest

from grasp_workbench import config
from grasp_workbench import main as entrypoint


@pytest.fixture
def supervisor(monkeypatch):
    state = SimpleNamespace(commands=[], runs=0, exit_code=19)

    class FakeSupervisor:
        def __init__(self, commands):
            state.commands.append(commands)

        def run(self):
            state.runs += 1
            return state.exit_code

    def forbidden_spawn(*_args, **_kwargs):
        pytest.fail("configuration tests must not start worker processes")

    monkeypatch.setattr(entrypoint, "WorkerSupervisor", FakeSupervisor)
    monkeypatch.setattr(entrypoint.subprocess, "Popen", forbidden_spawn)
    return state


def _write_config(tmp_path, contents):
    path = tmp_path / "settings.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def _commands(supervisor):
    assert supervisor.runs == 1
    assert len(supervisor.commands) == 1
    return supervisor.commands[0]


def _module(command):
    return command[command.index("-m") + 1].rsplit(".", 1)[-1]


def _option(command, name):
    return command[command.index(name) + 1]


def test_yaml_settings_create_three_workers_with_config_relative_paths(
    tmp_path, monkeypatch, supervisor
):
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    path = _write_config(settings_dir, """
camera:
  type: zed
  serial: 12345678
  resolution: HD1080
  fps: 15
ipc:
  service: example/config/rgbd
ui:
  enabled: true
sam3:
  enabled: true
  python: model-env/bin/python
  checkpoint: weights/sam3.pt
  device: cuda:1
run:
  frames: 7
  timeout: 4.5
""")
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code

    ui, model, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, model, camera)] == [
        "worker_camera_ui", "worker_sam3", "worker_camera"
    ]
    assert ui[0] == camera[0] == sys.executable
    assert model[0] == str(settings_dir / "model-env/bin/python")
    assert _option(model, "--checkpoint") == str(settings_dir / "weights/sam3.pt")
    assert _option(model, "--device") == "cuda:1"
    assert _option(camera, "--camera") == "zed"
    assert _option(camera, "--serial") == "12345678"
    assert _option(camera, "--resolution") == "HD1080"
    assert _option(camera, "--fps") == "15"
    assert _option(ui, "--frames") == "7"
    assert _option(ui, "--timeout") == "4.5"
    assert "--sam3" in ui
    assert all(_option(command, "--service") == "example/config/rgbd"
               for command in (ui, model, camera))
    assert all("--config" not in command for command in (ui, model, camera))
    assert "--checkpoint" not in ui and "--checkpoint" not in camera
    assert "--frames" not in model and "--frames" not in camera


def test_optional_cli_flags_override_yaml_settings(tmp_path, supervisor):
    path = _write_config(tmp_path, """
camera: {type: zed, serial: from-yaml, resolution: HD1080, fps: 30}
ipc: {service: example/yaml}
ui: {enabled: false}
sam3: {enabled: false, python: yaml-env/bin/python, checkpoint: yaml.pt, device: cuda}
run: {frames: 2, timeout: 10}
""")
    model_python = str(tmp_path / "cli-env/bin/python")
    checkpoint = str(tmp_path / "cli.pt")

    assert entrypoint.main([
        "run", "--config", str(path), "--camera", "realsense", "--serial", "cli-serial",
        "--resolution", "HD720", "--fps", "12", "--service", "example/cli",
        "--ui", "--sam3", "--sam3-python", model_python,
        "--sam3-checkpoint", checkpoint, "--sam3-device", "cpu",
        "--frames", "9", "--timeout", "2.25",
    ]) == supervisor.exit_code

    ui, model, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, model, camera)] == [
        "worker_camera_ui", "worker_sam3", "worker_camera"
    ]
    assert model[0] == model_python
    assert _option(model, "--checkpoint") == checkpoint
    assert _option(model, "--device") == "cpu"
    assert _option(camera, "--camera") == "realsense"
    assert _option(camera, "--serial") == "cli-serial"
    assert _option(camera, "--resolution") == "HD720"
    assert _option(camera, "--fps") == "12"
    assert _option(ui, "--frames") == "9"
    assert _option(ui, "--timeout") == "2.25"
    assert "--sam3" in ui
    assert all(_option(command, "--service") == "example/cli"
               for command in (ui, model, camera))


def test_camera_only_yaml_skips_ui_and_model_without_checkpoint(tmp_path, supervisor):
    path = _write_config(tmp_path, """
camera: {type: realsense, serial: null, fps: 6}
ui: {enabled: false}
sam3: {enabled: false, checkpoint: null}
run: {frames: 3, timeout: 2}
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code

    monitor, camera = _commands(supervisor)
    assert [_module(command) for command in (monitor, camera)] == [
        "worker_camera_monitor", "worker_camera"
    ]
    assert _option(camera, "--camera") == "realsense"
    assert _option(camera, "--fps") == "6"
    assert "--serial" not in camera
    assert _option(monitor, "--frames") == "3"
    assert _option(monitor, "--timeout") == "2.0"
    assert "--sam3" not in monitor


@pytest.mark.parametrize("contents", [
    "camera: [\n",
    "- not-a-settings-mapping\n",
    "run: {frames: many}\n",
])
def test_invalid_yaml_fails_before_constructing_supervisor(
    tmp_path, supervisor, capsys, contents
):
    path = _write_config(tmp_path, contents)
    with pytest.raises(SystemExit) as error:
        entrypoint.main(["run", "--config", str(path)])

    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert supervisor.commands == []
    assert supervisor.runs == 0


def test_missing_config_fails_before_constructing_supervisor(tmp_path, supervisor):
    with pytest.raises(SystemExit) as error:
        entrypoint.main(["run", "--config", str(tmp_path / "missing.yaml")])
    assert error.value.code == 2
    assert supervisor.commands == []
    assert supervisor.runs == 0


@pytest.mark.parametrize("argv", [["run"], None])
def test_run_without_extra_flags_uses_loaded_defaults(monkeypatch, supervisor, argv):
    loaded = []
    defaults = {
        "camera": "zed", "serial": "configured-camera", "resolution": "HD720", "fps": 20,
        "service": "example/defaults", "ui": True, "sam3": True,
        "sam3_python": "/configured/model-python", "sam3_checkpoint": "/configured/sam3.pt",
        "sam3_device": "cuda:2", "frames": 11, "timeout": 8.0,
    }

    def load_run_config(path):
        loaded.append(path)
        return defaults

    monkeypatch.setattr(config, "load_run_config", load_run_config)
    monkeypatch.setattr(sys, "argv", ["grasp-workbench", "run"])

    assert entrypoint.main(argv) == supervisor.exit_code
    assert loaded == [None]
    ui, model, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, model, camera)] == [
        "worker_camera_ui", "worker_sam3", "worker_camera"
    ]
    assert model[0] == "/configured/model-python"
    assert _option(model, "--checkpoint") == "/configured/sam3.pt"
    assert _option(model, "--device") == "cuda:2"
    assert _option(camera, "--serial") == "configured-camera"
    assert _option(camera, "--fps") == "20"
    assert _option(ui, "--frames") == "11"
    assert _option(ui, "--timeout") == "8.0"
    assert all(_option(command, "--service") == "example/defaults"
               for command in (ui, model, camera))


def test_yaml_sam3_and_sam3d_create_four_independent_workers(tmp_path, monkeypatch, supervisor):
    path = _write_config(tmp_path, """
camera: {type: zed, serial: 12345}
ipc: {service: example/mesh/rgbd}
ui: {enabled: false}
sam3: {enabled: true, python: sam3-env/bin/python, checkpoint: sam3.pt}
sam3d:
  enabled: true
  python: sam3d-env/bin/python
  source_dir: vendor/sam-3d-objects
  config_path: weights/pipeline.yaml
  output_dir: generated/meshes
  seed: 18
""")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, sam3, sam3d, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, sam3, sam3d, camera)] == [
        "worker_camera_ui", "worker_sam3", "worker_sam3d", "worker_camera"]
    assert "--sam3" in ui and "--sam3d" in ui
    assert sam3d[0] == str(tmp_path / "sam3d-env/bin/python")
    assert _option(sam3d, "--source-dir") == str(tmp_path / "vendor/sam-3d-objects")
    assert _option(sam3d, "--config-path") == str(tmp_path / "weights/pipeline.yaml")
    assert _option(sam3d, "--output-dir") == str(tmp_path / "generated/meshes")
    assert _option(sam3d, "--seed") == "18"
    assert all(_option(command, "--service") == "example/mesh/rgbd"
               for command in (ui, sam3, sam3d, camera))
    assert "--config-path" not in sam3 and "--checkpoint" not in sam3d


def test_sam3d_does_not_force_sam3_or_require_its_checkpoint(tmp_path, supervisor):
    path = _write_config(tmp_path, """
ui: {enabled: false}
sam3: {enabled: false, checkpoint: null}
sam3d: {enabled: true, config_path: weights/pipeline.yaml}
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, mesh, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, mesh, camera)] == [
        "worker_camera_ui", "worker_sam3d", "worker_camera"]
    assert "--sam3d" in ui and "--sam3" not in ui
    assert _option(mesh, "--source-dir") == str(config.ROOT / "third_party/sam-3d-objects")
    assert _option(mesh, "--output-dir") == str(config.ROOT / "runs/meshes")


def test_sam3d_enabled_requires_config_path_before_spawning(tmp_path, supervisor, capsys):
    path = _write_config(tmp_path, "sam3d: {enabled: true, config_path: null}\n")
    with pytest.raises(SystemExit) as error:
        entrypoint.main(["run", "--config", str(path)])
    assert error.value.code == 2 and "sam3d.config_path" in capsys.readouterr().err
    assert supervisor.commands == []


def test_sam3d_cli_overrides_yaml_without_other_model_changes(tmp_path, supervisor):
    path = _write_config(tmp_path, "sam3d: {enabled: false}\n")
    assert entrypoint.main(["run", "--config", str(path), "--sam3d", "--sam3d-python", "python3",
                            "--sam3d-source-dir", "/example/source", "--sam3d-config-path", "/example/pipeline.yaml",
                            "--sam3d-output-dir", "/example/output", "--sam3d-seed", "99"]) == supervisor.exit_code
    ui, mesh, camera = _commands(supervisor)
    assert mesh[0] == "python3" and "--sam3d" in ui and "--sam3" not in ui
    assert _option(mesh, "--config-path") == "/example/pipeline.yaml"
    assert _option(mesh, "--source-dir") == "/example/source"
    assert _option(mesh, "--output-dir") == "/example/output"
    assert _option(mesh, "--seed") == "99"


def test_sam3d_yaml_depth_and_texture_are_ui_request_settings_only(tmp_path, supervisor):
    path = _write_config(tmp_path, """
sam3d:
  enabled: true
  config_path: weights/pipeline.yaml
  depth_source: monocular
  texture_baking: true
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, mesh, camera = _commands(supervisor)
    assert _option(ui, "--sam3d-depth-source") == "monocular"
    assert "--sam3d-texture-baking" in ui
    assert "--sam3" not in ui
    for command in (mesh, camera):
        assert "--sam3d-depth-source" not in command
        assert "--sam3d-texture-baking" not in command


def test_sam3d_request_cli_options_override_yaml(tmp_path, supervisor):
    path = _write_config(tmp_path, "sam3d: {enabled: true, config_path: pipeline.yaml, depth_source: monocular}\n")
    assert entrypoint.main(["run", "--config", str(path), "--sam3d-depth-source", "depth",
                            "--sam3d-texture-baking"]) == supervisor.exit_code
    ui, _mesh, _camera = _commands(supervisor)
    assert _option(ui, "--sam3d-depth-source") == "depth"
    assert "--sam3d-texture-baking" in ui


def test_yaml_foundationpose_starts_independent_worker_with_ui_and_yaml_paths(tmp_path, supervisor):
    path = _write_config(tmp_path, """
ui: {enabled: false}
sam3: {enabled: true, checkpoint: weights/sam3.pt}
ipc: {service: example/pose/rgbd}
foundationpose:
  enabled: true
  python: pose-env/bin/python
  source_dir: vendor/FoundationPose
  device: cuda:1
  registration_iterations: 8
  tracking_iterations: 2
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, sam3, pose, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, sam3, pose, camera)] == [
        "worker_camera_ui", "worker_sam3", "worker_foundationpose", "worker_camera"]
    assert "--sam3" in ui and "--foundationpose" in ui
    assert pose[0] == str(tmp_path / "pose-env/bin/python")
    assert _option(pose, "--source-dir") == str(tmp_path / "vendor/FoundationPose")
    assert _option(pose, "--device") == "cuda:1"
    assert _option(pose, "--registration-iterations") == "8"
    assert _option(pose, "--tracking-iterations") == "2"
    assert all(_option(command, "--service") == "example/pose/rgbd"
               for command in (ui, sam3, pose, camera))
    assert "--frames" not in pose and "--checkpoint" not in pose


def test_foundationpose_does_not_force_other_models_or_require_their_paths(tmp_path, supervisor):
    path = _write_config(tmp_path, """
ui: {enabled: false}
sam3: {enabled: false, checkpoint: null}
sam3d: {enabled: false, config_path: null}
foundationpose: {enabled: true}
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, pose, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, pose, camera)] == [
        "worker_camera_ui", "worker_foundationpose", "worker_camera"]
    assert "--foundationpose" in ui and "--sam3" not in ui and "--sam3d" not in ui
    assert pose[0] == sys.executable
    assert _option(pose, "--source-dir") == str(config.ROOT / "third_party/FoundationPose")
    assert _option(pose, "--device") == "cuda:0"
    assert _option(pose, "--registration-iterations") == "5"
    assert _option(pose, "--tracking-iterations") == "1"


def test_foundationpose_cli_overrides_yaml_without_model_runtime_checks(tmp_path, supervisor):
    path = _write_config(tmp_path, "foundationpose: {enabled: false, device: cuda:3}\n")
    assert entrypoint.main([
        "run", "--config", str(path), "--foundationpose", "--foundationpose-python", "python3",
        "--foundationpose-source-dir", "/example/FoundationPose", "--foundationpose-device", "cpu",
        "--foundationpose-registration-iterations", "4", "--foundationpose-tracking-iterations", "2",
    ]) == supervisor.exit_code
    ui, pose, camera = _commands(supervisor)
    assert pose[0] == "python3" and "--foundationpose" in ui
    assert _option(pose, "--source-dir") == "/example/FoundationPose"
    assert _option(pose, "--device") == "cpu"
    assert _option(pose, "--registration-iterations") == "4"
    assert _option(pose, "--tracking-iterations") == "2"


def test_foundationpose_and_sam3d_are_independently_selectable(tmp_path, supervisor):
    path = _write_config(tmp_path, """
sam3: {enabled: true, checkpoint: weights/sam3.pt}
sam3d: {enabled: true, config_path: weights/pipeline.yaml}
foundationpose: {enabled: true}
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    commands = _commands(supervisor)
    assert [_module(command) for command in commands] == [
        "worker_camera_ui", "worker_sam3", "worker_sam3d", "worker_foundationpose", "worker_camera"]
    assert all(flag in commands[0] for flag in ("--sam3", "--sam3d", "--foundationpose"))


def test_vlm_yaml_starts_only_requested_worker_and_keeps_key_path_out_of_ui(tmp_path, supervisor):
    path = _write_config(tmp_path, """
ui: {enabled: false}
ipc: {service: example/vlm/rgbd}
vlm:
  enabled: true
  python: vlm-env/bin/python
  model: configured-model
  api_key_file: .secrets/provider.key
  asset_dir: objects/assets
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, vlm, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, vlm, camera)] == [
        "worker_camera_ui", "worker_vlm", "worker_camera"]
    assert vlm[0] == str(tmp_path / "vlm-env/bin/python")
    assert _option(vlm, "--model") == "configured-model"
    key_path = str(tmp_path / ".secrets/provider.key")
    assert _option(vlm, "--api-key-file") == key_path
    assert "--vlm" in ui and _option(ui, "--asset-dir") == str(tmp_path / "objects/assets")
    assert "--api-key-file" not in ui and key_path not in ui and key_path not in camera
    assert all(flag not in ui for flag in ("--sam3", "--sam3d", "--foundationpose"))
    assert all(_option(command, "--service") == "example/vlm/rgbd" for command in (ui, vlm, camera))


def test_vlm_cli_overrides_and_no_credentials_path_is_required_by_supervisor(tmp_path, supervisor):
    path = _write_config(tmp_path, "vlm: {enabled: false}\n")
    assert entrypoint.main(["run", "--config", str(path), "--vlm", "--vlm-python", "python3",
                            "--vlm-model", "cli-model", "--asset-dir", "/example/assets"]) == supervisor.exit_code
    ui, vlm, camera = _commands(supervisor)
    assert vlm[0] == "python3" and _option(vlm, "--model") == "cli-model"
    assert "--api-key-file" not in vlm and "--vlm" in ui
    assert _option(ui, "--asset-dir") == "/example/assets"


def test_vlm_key_file_cli_override_reaches_only_model_worker(tmp_path, supervisor):
    path = _write_config(tmp_path, "vlm: {enabled: true, api_key_file: keys/yaml}\n")
    assert entrypoint.main(["run", "--config", str(path), "--vlm-api-key-file", "/local/key-file"]) == supervisor.exit_code
    ui, vlm, camera = _commands(supervisor)
    assert _option(vlm, "--api-key-file") == "/local/key-file"
    assert "/local/key-file" not in ui and "/local/key-file" not in camera


def test_all_models_are_selected_independently_with_vlm(tmp_path, supervisor):
    path = _write_config(tmp_path, """
sam3: {enabled: true, checkpoint: weights/sam3.pt}
sam3d: {enabled: true, config_path: weights/pipeline.yaml}
foundationpose: {enabled: true}
vlm: {enabled: true}
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    commands = _commands(supervisor)
    assert [_module(command) for command in commands] == [
        "worker_camera_ui", "worker_sam3", "worker_sam3d", "worker_foundationpose", "worker_vlm", "worker_camera"]
    assert all(flag in commands[0] for flag in ("--sam3", "--sam3d", "--foundationpose", "--vlm"))


def test_codex_backend_yaml_forwards_model_executable_but_not_api_key_file(tmp_path, supervisor):
    path = _write_config(tmp_path, """
vlm:
  enabled: true
  backend: codex_cli
  codex_executable: tools/codex
  model: same-configured-model
  api_key_file: .secrets/unused-by-codex
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, vlm, camera = _commands(supervisor)
    assert _option(vlm, "--backend") == "codex_cli"
    assert _option(vlm, "--codex-executable") == str(tmp_path / "tools/codex")
    assert _option(vlm, "--model") == "same-configured-model"
    assert "--api-key-file" not in vlm
    assert all("unused-by-codex" not in value for command in (ui, vlm, camera) for value in command)
    assert "--backend" not in ui and "--codex-executable" not in ui


def test_codex_backend_cli_overrides_yaml_and_preserves_bare_executable(tmp_path, supervisor):
    path = _write_config(tmp_path, "vlm: {enabled: true, backend: openai_api, api_key_file: keys/ignored}\n")
    assert entrypoint.main(["run", "--config", str(path), "--vlm-backend", "codex_cli",
                            "--vlm-codex-executable", "codex-local", "--vlm-model", "cli-model"]) == supervisor.exit_code
    _ui, vlm, _camera = _commands(supervisor)
    assert _option(vlm, "--backend") == "codex_cli"
    assert _option(vlm, "--codex-executable") == "codex-local"
    assert _option(vlm, "--model") == "cli-model"
    assert "--api-key-file" not in vlm


@pytest.mark.parametrize("backend", ["openai_api", "anthropic_api"])
def test_api_backend_cli_override_keeps_api_key_routing(tmp_path, supervisor, backend):
    path = _write_config(tmp_path, "vlm: {enabled: true, backend: codex_cli, api_key_file: keys/api}\n")
    assert entrypoint.main(["run", "--config", str(path), "--vlm-backend", backend]) == supervisor.exit_code
    _ui, vlm, _camera = _commands(supervisor)
    assert _option(vlm, "--backend") == backend
    assert _option(vlm, "--api-key-file") == str(tmp_path / "keys/api")


def test_anthropic_backend_yaml_only_starts_selected_vlm_and_routes_key_path(tmp_path, supervisor):
    path = _write_config(tmp_path, """
ui: {enabled: false}
vlm:
  enabled: true
  backend: anthropic_api
  model: configured-claude
  api_key_file: keys/anthropic.key
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, vlm, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, vlm, camera)] == [
        "worker_camera_ui", "worker_vlm", "worker_camera"]
    assert _option(vlm, "--backend") == "anthropic_api"
    assert _option(vlm, "--model") == "configured-claude"
    key_path = str(tmp_path / "keys/anthropic.key")
    assert _option(vlm, "--api-key-file") == key_path
    assert "--codex-executable" not in vlm
    assert key_path not in ui and key_path not in camera
    assert all(flag not in ui for flag in ("--sam3", "--sam3d", "--foundationpose", "--graspgenx", "--grasp-selection"))


def test_invalid_vlm_backend_cli_rejects_before_starting_workers(tmp_path, supervisor):
    path = _write_config(tmp_path, "{}\n")
    with pytest.raises(SystemExit) as error:
        entrypoint.main(["run", "--config", str(path), "--vlm-backend", "unknown"])
    assert error.value.code == 2 and supervisor.commands == []


def test_graspgenx_runs_standalone_without_forcing_perception_workers(tmp_path, supervisor):
    path = _write_config(tmp_path, """
ui: {enabled: false}
ipc: {service: example/grasp}
graspgenx:
  enabled: true
  python: grasp-env/bin/python
  source_dir: vendor/GraspGenX
  checkpoint_dir: weights/grasp
  assets_dir: assets/grippers
  gripper_id: custom
  output_dir: generated/grasps
  settings: {num_grasps: 12, seed: 7}
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, grasp, camera = _commands(supervisor)
    assert [_module(command) for command in (ui, grasp, camera)] == [
        "worker_camera_ui", "worker_graspgenx", "worker_camera"]
    assert "--graspgenx" in ui
    assert all(flag not in ui for flag in ("--sam3", "--sam3d", "--foundationpose", "--vlm"))
    assert grasp[0] == str(tmp_path / "grasp-env/bin/python")
    assert _option(grasp, "--source-dir") == str(tmp_path / "vendor/GraspGenX")
    assert _option(grasp, "--checkpoint-dir") == str(tmp_path / "weights/grasp")
    assert _option(grasp, "--assets-dir") == str(tmp_path / "assets/grippers")
    assert _option(grasp, "--output-dir") == str(tmp_path / "generated/grasps")
    assert _option(grasp, "--gripper-id") == "custom"
    assert entrypoint.json.loads(_option(grasp, "--settings")) == {"num_grasps": 12, "seed": 7}
    assert all(_option(command, "--service") == "example/grasp" for command in (ui, grasp, camera))


def test_graspgenx_cli_overrides_yaml(tmp_path, supervisor):
    path = _write_config(tmp_path, "graspgenx: {enabled: false}\n")
    assert entrypoint.main(["run", "--config", str(path), "--graspgenx", "--graspgenx-python", "python3",
                            "--graspgenx-source-dir", "/custom/source", "--graspgenx-checkpoint-dir", "/custom/checkpoints",
                            "--graspgenx-assets-dir", "/custom/assets", "--graspgenx-output-dir", "/custom/output",
                            "--graspgenx-gripper-id", "chosen", "--graspgenx-settings", '{"seed": 99}']) == supervisor.exit_code
    ui, grasp, _camera = _commands(supervisor)
    assert "--graspgenx" in ui and grasp[0] == "python3"
    assert _option(grasp, "--source-dir") == "/custom/source"
    assert _option(grasp, "--checkpoint-dir") == "/custom/checkpoints"
    assert _option(grasp, "--assets-dir") == "/custom/assets"
    assert _option(grasp, "--output-dir") == "/custom/output"
    assert _option(grasp, "--gripper-id") == "chosen"
    assert entrypoint.json.loads(_option(grasp, "--settings")) == {"seed": 99}


def test_graspgenx_non_mapping_cli_settings_rejected(tmp_path, supervisor):
    path = _write_config(tmp_path, "{}\n")
    with pytest.raises(SystemExit) as error:
        entrypoint.main(["run", "--config", str(path), "--graspgenx", "--graspgenx-settings", "[]"])
    assert error.value.code == 2 and supervisor.commands == []


def test_vlm_semantic_local_paths_are_forwarded_only_to_ui_not_model_workers(tmp_path, supervisor):
    path = _write_config(tmp_path, """
vlm:
  enabled: true
  backend: codex_cli
  api_key_file: .secrets/not-for-cli
  policy_catalog: policies/registry.json
  session_dir: outputs/task_sessions
""")
    assert entrypoint.main(["run", "--config", str(path)]) == supervisor.exit_code
    ui, vlm, camera = _commands(supervisor)
    catalog, sessions = str(tmp_path / "policies/registry.json"), str(tmp_path / "outputs/task_sessions")
    assert _option(ui, "--policy-catalog") == catalog
    assert _option(ui, "--session-dir") == sessions
    for command in (vlm, camera):
        assert catalog not in command and sessions not in command
        assert "--policy-catalog" not in command and "--session-dir" not in command
        assert "--api-key-file" not in command


def test_vlm_semantic_cli_paths_override_yaml_without_reading_files(tmp_path, supervisor):
    path = _write_config(tmp_path, "vlm: {policy_catalog: yaml.json, session_dir: yaml-tasks}\n")
    assert entrypoint.main(["run", "--config", str(path), "--vlm-policy-catalog", "/local/policies.json",
                            "--vlm-session-dir", "/local/tasks"]) == supervisor.exit_code
    ui, _camera = _commands(supervisor)
    assert _option(ui, "--policy-catalog") == "/local/policies.json"
    assert _option(ui, "--session-dir") == "/local/tasks"
