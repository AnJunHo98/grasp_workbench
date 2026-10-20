import subprocess
import sys
import time
import signal
from types import SimpleNamespace

import pytest

from grasp_workbench.main import WorkerSupervisor, build_commands, build_parser


def test_selected_camera_arguments_and_independent_workers():
    args = build_parser().parse_args(["run", "--camera", "zed", "--serial", "123", "--frames", "3"])
    monitor, camera = build_commands(args)
    assert monitor[0] == camera[0] == sys.executable
    assert "grasp_workbench.workers.worker_camera_monitor" in monitor
    assert "grasp_workbench.workers.worker_camera" in camera
    assert camera[camera.index("--serial") + 1] == "123"
    assert monitor[monitor.index("--frames") + 1] == "3"
    assert "--frames" not in camera


def test_ui_replaces_console_receiver_not_camera():
    args = build_parser().parse_args(["run", "--ui", "--camera", "zed", "--frames", "5"])
    ui, camera = build_commands(args)
    assert "grasp_workbench.workers.worker_camera_ui" in ui
    assert "grasp_workbench.workers.worker_camera" in camera
    assert ui[ui.index("--frames") + 1] == "5"


def test_standalone_ui_needs_no_camera_options():
    args = build_parser().parse_args(["ui", "--service", "example/rgbd"])
    assert args.command == "ui" and args.service == "example/rgbd"
    assert not hasattr(args, "camera")


def test_sam3_uses_one_independent_model_process_and_enables_ui():
    args = build_parser().parse_args(["run", "--sam3", "--sam3-python", "/example/model-python",
                                     "--sam3-checkpoint", "/example/sam3.pt"])
    ui, model, camera = build_commands(args)
    assert "grasp_workbench.workers.worker_camera_ui" in ui and "--sam3" in ui
    assert model[0] == "/example/model-python"
    assert model.count("grasp_workbench.workers.worker_sam3") == 1
    assert model[model.index("--checkpoint") + 1] == "/example/sam3.pt"
    assert "grasp_workbench.workers.worker_camera" in camera
    assert all(command[command.index("--service") + 1] == args.service for command in (ui, model, camera))


def test_sam3_checkpoint_is_only_needed_when_starting_model(monkeypatch, tmp_path):
    from grasp_workbench.main import main

    monkeypatch.delenv("SAM3_CHECKPOINT", raising=False)
    config = tmp_path / "test.yaml"
    config.write_text("sam3:\n  enabled: false\n  checkpoint: null\n")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(config), "--sam3"])
    assert error.value.code == 2
    assert build_parser().parse_args(["run"]).sam3 is False
    assert build_parser().parse_args(["ui", "--sam3"]).sam3 is True


def test_standalone_sam3_ui_only_connects_existing_services(monkeypatch):
    from grasp_workbench.main import main
    from grasp_workbench.workers import worker_camera_ui

    calls = []
    monkeypatch.setattr(worker_camera_ui, "main", lambda argv: calls.append(argv) or 0)
    assert main(["ui", "--sam3", "--service", "example/rgbd"]) == 0
    assert calls == [["--service", "example/rgbd", "--frames", "0", "--timeout", "0", "--sam3"]]


def test_standalone_sam3d_ui_only_connects_existing_services(monkeypatch):
    from grasp_workbench.main import main
    from grasp_workbench.workers import worker_camera_ui

    calls = []
    monkeypatch.setattr(worker_camera_ui, "main", lambda argv: calls.append(argv) or 0)
    assert main(["ui", "--sam3d", "--service", "example/rgbd"]) == 0
    assert calls == [["--service", "example/rgbd", "--frames", "0", "--timeout", "0", "--sam3d",
                      "--sam3d-depth-source", "depth"]]
    assert build_parser().parse_args(["run"]).sam3d is False


def test_standalone_foundationpose_ui_only_connects_existing_services(monkeypatch):
    from grasp_workbench.main import main
    from grasp_workbench.workers import worker_camera_ui

    calls = []
    monkeypatch.setattr(worker_camera_ui, "main", lambda argv: calls.append(argv) or 0)
    assert main(["ui", "--foundationpose", "--service", "example/rgbd"]) == 0
    assert calls == [["--service", "example/rgbd", "--frames", "0", "--timeout", "0", "--foundationpose"]]
    assert build_parser().parse_args(["run"]).foundationpose is False


def test_standalone_vlm_ui_forwards_only_service_flags_and_asset_directory(monkeypatch):
    from grasp_workbench.main import main
    from grasp_workbench.workers import worker_camera_ui

    calls = []
    monkeypatch.setattr(worker_camera_ui, "main", lambda argv: calls.append(argv) or 0)
    assert main(["ui", "--vlm", "--asset-dir", "/example/assets", "--service", "example/rgbd"]) == 0
    assert calls == [["--service", "example/rgbd", "--frames", "0", "--timeout", "0", "--vlm",
                      "--asset-dir", "/example/assets"]]
    assert build_parser().parse_args(["run"]).vlm is False


def test_standalone_grasp_ui_only_connects_existing_worker(monkeypatch):
    from grasp_workbench.main import main
    from grasp_workbench.workers import worker_camera_ui

    calls = []
    monkeypatch.setattr(worker_camera_ui, "main", lambda argv: calls.append(argv) or 0)
    assert main(["ui", "--graspgenx", "--service", "example/grasp"]) == 0
    assert calls == [["--service", "example/grasp", "--frames", "0", "--timeout", "0", "--graspgenx"]]


def test_supervisor_stops_sibling_after_worker_completes():
    supervisor = WorkerSupervisor([[sys.executable, "-c", "import time;time.sleep(0.15)"],
                                   [sys.executable, "-c", "import time;time.sleep(20)"]])
    started = time.monotonic()
    assert supervisor.run() == 0
    assert time.monotonic() - started < 4
    assert len({process.pid for process in supervisor.processes}) == 2
    assert all(process.poll() is not None for process in supervisor.processes)


def test_supervisor_reports_worker_failure():
    supervisor = WorkerSupervisor([[sys.executable, "-c", "raise SystemExit(7)"],
                                   [sys.executable, "-c", "import time;time.sleep(20)"]])
    assert supervisor.run() == 7
    assert all(process.poll() is not None for process in supervisor.processes)


def test_failed_spawn_cleans_already_started_worker(monkeypatch):
    real_popen = subprocess.Popen
    calls = []

    def popen(*args, **kwargs):
        if calls:
            raise FileNotFoundError("missing second worker")
        process = real_popen(*args, **kwargs)
        calls.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    supervisor = WorkerSupervisor([[sys.executable, "-c", "import time;time.sleep(20)"], ["missing"]])
    with pytest.raises(FileNotFoundError):
        supervisor.run()
    assert calls[0].poll() is not None


def test_exited_leader_does_not_leave_its_process_group(monkeypatch):
    signals = []
    monkeypatch.setattr("grasp_workbench.main.os.killpg", lambda pid, sig: signals.append((pid, sig)))
    supervisor = WorkerSupervisor([])
    supervisor.processes = [SimpleNamespace(pid=12345, wait=lambda **_kwargs: 0)]
    supervisor.close()
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
