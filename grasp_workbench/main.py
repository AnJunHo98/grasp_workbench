"""Discover a camera or run independent capture/subscriber workers."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from threading import Event


class WorkerSupervisor:
    """Manage processes only; camera data stays on iceoryx2, not in the parent."""

    def __init__(self, commands):
        self.commands = commands
        self.processes = []
        self.stop_event = Event()

    def stop(self, *_args):
        self.stop_event.set()

    def run(self):
        previous = {sig: signal.signal(sig, self.stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for command in self.commands:
                self.processes.append(subprocess.Popen(command, start_new_session=True))
            while not self.stop_event.is_set():
                statuses = [process.poll() for process in self.processes]
                if any(code is not None for code in statuses):
                    return next((code for code in statuses if code not in (None, 0)), 0)
                self.stop_event.wait(0.05)
            return 0
        finally:
            self.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def close(self):
        for process in self.processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        for process in self.processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            # The leader can exit before an SDK/helper descendant. Release the
            # entire group owned by this supervisor, not just the leader PID.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def build_commands(args):
    from .workers.registry import WORKERS

    common = ["--service", args.service]
    camera = [sys.executable, "-u", "-m", WORKERS["camera"].module,
              *common, "--camera", args.camera, "--fps", str(args.fps),
              "--resolution", args.resolution]
    if args.serial is not None:
        camera += ["--serial", args.serial]
    receiver = "camera_ui" if (args.ui or args.sam3 or args.sam3d or args.foundationpose
                               or args.vlm or args.graspgenx or args.grasp_selection) else "camera_monitor"
    monitor = [sys.executable, "-u", "-m", WORKERS[receiver].module,
               *common, "--frames", str(args.frames), "--timeout", str(args.timeout)]
    commands = [monitor]
    if receiver == "camera_ui" and args.asset_dir is not None:
        monitor += ["--asset-dir", args.asset_dir]
    if receiver == "camera_ui":
        for name in ("policy_catalog", "session_dir"):
            value = getattr(args, "vlm_" + name)
            if value is not None:
                monitor += ["--" + name.replace("_", "-"), value]
    if args.sam3:
        monitor.append("--sam3")
        commands.append([args.sam3_python, "-u", "-m", WORKERS["sam3"].module,
                         *common, "--checkpoint", args.sam3_checkpoint,
                         "--device", args.sam3_device])
    if args.sam3d:
        monitor += ["--sam3d", "--sam3d-depth-source", args.sam3d_depth_source]
        if args.sam3d_texture_baking:
            monitor.append("--sam3d-texture-baking")
        mesh = [args.sam3d_python, "-u", "-m", WORKERS["sam3d"].module,
                *common, "--config-path", args.sam3d_config_path, "--seed", str(args.sam3d_seed)]
        if args.sam3d_source_dir is not None:
            mesh += ["--source-dir", args.sam3d_source_dir]
        if args.sam3d_output_dir is not None:
            mesh += ["--output-dir", args.sam3d_output_dir]
        commands.append(mesh)
    if args.foundationpose:
        monitor.append("--foundationpose")
        pose = [args.foundationpose_python, "-u", "-m", WORKERS["foundationpose"].module,
                *common, "--device", args.foundationpose_device,
                "--registration-iterations", str(args.foundationpose_registration_iterations),
                "--tracking-iterations", str(args.foundationpose_tracking_iterations)]
        if args.foundationpose_source_dir is not None:
            pose += ["--source-dir", args.foundationpose_source_dir]
        commands.append(pose)
    if args.vlm:
        monitor.append("--vlm")
        vlm = [args.vlm_python, "-u", "-m", WORKERS["vlm"].module,
               *common, "--model", args.vlm_model, "--backend", args.vlm_backend]
        if args.vlm_backend == "codex_cli":
            vlm += ["--codex-executable", args.vlm_codex_executable]
        elif args.vlm_api_key_file is not None:
            vlm += ["--api-key-file", args.vlm_api_key_file]
        commands.append(vlm)
    if args.graspgenx:
        monitor.append("--graspgenx")
        grasp = [args.graspgenx_python, "-u", "-m", WORKERS["graspgenx"].module,
                 *common, "--gripper-id", args.graspgenx_gripper_id,
                 "--output-dir", args.graspgenx_output_dir,
                 "--settings", json.dumps(args.graspgenx_settings, ensure_ascii=False)]
        for name in ("source_dir", "checkpoint_dir", "assets_dir"):
            value = getattr(args, "graspgenx_" + name)
            if value is not None:
                grasp += ["--" + name.replace("_", "-"), value]
        commands.append(grasp)
    if args.grasp_selection:
        monitor.append("--grasp-selection")
        selection = [args.grasp_selection_python, "-u", "-m", WORKERS["grasp_selection"].module,
                     *common, "--settings", json.dumps(args.grasp_selection_settings, ensure_ascii=False)]
        if args.grasp_selection_robot_config is not None:
            selection += ["--robot-config", args.grasp_selection_robot_config]
        commands.append(selection)
    return [*commands, camera]


def build_parser(run_defaults=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("cameras", help="List devices without opening a camera")
    discover.add_argument("--camera", choices=("zed", "realsense"), default="zed")
    run = commands.add_parser("run", help="Run workers from YAML settings")
    run.add_argument("--config", help="YAML settings; defaults to config.yaml + optional config.local.yaml")
    run.add_argument("--camera", choices=("zed", "realsense"), default="zed")
    run.add_argument("--serial", help="Optional camera serial; otherwise SDK default")
    run.add_argument("--resolution", default="HD720", help="ZED resolution enum")
    run.add_argument("--fps", type=int, default=30)
    run.add_argument("--service", default="grasp_workbench/camera/rgbd")
    run.add_argument("--frames", type=int, default=0, help="Stop after N received frames; 0 runs continuously")
    run.add_argument("--timeout", type=float, default=0, help="Receiver timeout in seconds; 0 waits continuously")
    run.add_argument("--ui", action="store_true", help="Use the independent Qt image/plot subscriber")
    run.add_argument("--sam3", action="store_true", help="Enable UI A/B prompts and an independent SAM3 worker")
    run.add_argument("--sam3-checkpoint", default=os.environ.get("SAM3_CHECKPOINT"),
                     help="Override YAML sam3.checkpoint")
    run.add_argument("--sam3-python", default=os.environ.get("SAM3_PYTHON", sys.executable),
                     help="Override YAML sam3.python")
    run.add_argument("--sam3-device", default="cuda", help="Override YAML sam3.device")
    run.add_argument("--sam3d", action="store_true", help="Enable mesh requests and an independent SAM3D worker")
    run.add_argument("--sam3d-python", default=sys.executable, help="Override YAML sam3d.python")
    run.add_argument("--sam3d-source-dir", help="Override YAML sam3d.source_dir")
    run.add_argument("--sam3d-config-path", help="Override YAML sam3d.config_path")
    run.add_argument("--sam3d-output-dir", help="Override YAML sam3d.output_dir")
    run.add_argument("--sam3d-seed", type=int, default=42, help="Override YAML sam3d.seed")
    run.add_argument("--sam3d-depth-source", choices=("depth", "monocular"), default="depth",
                     help="Override YAML sam3d.depth_source for UI mesh requests")
    run.add_argument("--sam3d-texture-baking", action="store_true", help="Override YAML sam3d.texture_baking")
    run.add_argument("--foundationpose", action="store_true", help="Enable an independent FoundationPose worker")
    run.add_argument("--foundationpose-python", default=sys.executable, help="Override YAML foundationpose.python")
    run.add_argument("--foundationpose-source-dir", help="Override YAML foundationpose.source_dir")
    run.add_argument("--foundationpose-device", default="cuda:0", help="Override YAML foundationpose.device")
    run.add_argument("--foundationpose-registration-iterations", type=int, default=5,
                     help="Override YAML foundationpose.registration_iterations")
    run.add_argument("--foundationpose-tracking-iterations", type=int, default=1,
                     help="Override YAML foundationpose.tracking_iterations")
    run.add_argument("--vlm", action="store_true", help="Enable an independent VLM request worker")
    run.add_argument("--vlm-python", default=sys.executable, help="Override YAML vlm.python")
    run.add_argument("--vlm-model", default="gpt-6.1-sol", help="Override YAML vlm.model")
    run.add_argument("--vlm-backend", choices=("openai_api", "codex_cli", "anthropic_api"), default="openai_api",
                     help="Override YAML vlm.backend")
    run.add_argument("--vlm-codex-executable", default="codex", help="Override YAML vlm.codex_executable")
    run.add_argument("--vlm-api-key-file", help="Override YAML vlm.api_key_file; read only by the VLM worker")
    run.add_argument("--asset-dir", help="Override YAML vlm.asset_dir for the UI asset library")
    run.add_argument("--vlm-policy-catalog", help="Override YAML vlm.policy_catalog for public semantic policies")
    run.add_argument("--vlm-session-dir", help="Override YAML vlm.session_dir for UI task sessions")
    run.add_argument("--graspgenx", action="store_true", help="Enable an independent mesh-only GraspGenX worker")
    run.add_argument("--graspgenx-python", default=sys.executable, help="Override YAML graspgenx.python")
    run.add_argument("--graspgenx-source-dir", help="Override YAML graspgenx.source_dir")
    run.add_argument("--graspgenx-checkpoint-dir", help="Override YAML graspgenx.checkpoint_dir")
    run.add_argument("--graspgenx-assets-dir", help="Override YAML graspgenx.assets_dir")
    run.add_argument("--graspgenx-gripper-id", default="franka_panda", help="Override YAML graspgenx.gripper_id")
    run.add_argument("--graspgenx-output-dir", default="runs/grasps", help="Override YAML graspgenx.output_dir")
    run.add_argument("--graspgenx-settings", type=json.loads, default={}, help="Override YAML graspgenx.settings as JSON")
    run.add_argument("--grasp-selection", action="store_true", help="Enable an independent grasp selection/reachability worker")
    run.add_argument("--grasp-selection-python", default=sys.executable, help="Override YAML grasp_selection.python")
    run.add_argument("--grasp-selection-robot-config", help="Override YAML grasp_selection.robot_config")
    run.add_argument("--grasp-selection-settings", type=json.loads, default={}, help="Override YAML grasp_selection.settings as JSON")
    run.set_defaults(**(run_defaults or {}))
    ui = commands.add_parser("ui", help="Subscribe and display an existing camera service; no camera opened")
    ui.add_argument("--service", default="grasp_workbench/camera/rgbd")
    ui.add_argument("--frames", type=int, default=0)
    ui.add_argument("--timeout", type=float, default=0)
    ui.add_argument("--sam3", action="store_true", help="Control/receive an existing SAM3 worker")
    ui.add_argument("--sam3d", action="store_true", help="Request meshes from an existing SAM3D worker")
    ui.add_argument("--sam3d-depth-source", choices=("depth", "monocular"), default="depth")
    ui.add_argument("--sam3d-texture-baking", action="store_true")
    ui.add_argument("--foundationpose", action="store_true", help="Control/receive an existing FoundationPose worker")
    ui.add_argument("--vlm", action="store_true", help="Request objects from an existing VLM worker")
    ui.add_argument("--asset-dir", help="UI asset library directory")
    ui.add_argument("--vlm-policy-catalog", help="UI semantic policy catalog JSON file")
    ui.add_argument("--vlm-session-dir", help="UI task session directory")
    ui.add_argument("--graspgenx", action="store_true", help="Request object-local grasps from an existing worker")
    ui.add_argument("--grasp-selection", action="store_true", help="Request grasp selection from an existing worker")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        from .config import load_run_config

        try:
            defaults = load_run_config(args.config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        parser = build_parser(defaults)
        args = parser.parse_args(argv)
        if args.sam3 and not args.sam3_checkpoint:
            parser.error("SAM3 needs sam3.checkpoint in YAML or --sam3-checkpoint PATH")
        if args.sam3d and not args.sam3d_config_path:
            parser.error("SAM3D needs sam3d.config_path in YAML or --sam3d-config-path PATH")
        if args.graspgenx and not isinstance(args.graspgenx_settings, dict):
            parser.error("--graspgenx-settings must be a JSON object")
        if args.grasp_selection and not isinstance(args.grasp_selection_settings, dict):
            parser.error("--grasp-selection-settings must be a JSON object")
    if args.command == "cameras":
        if args.camera == "zed":
            from .cameras.zed import discover
        else:
            from .cameras.realsense import discover
        print(json.dumps(discover(), indent=2))
        return 0
    if args.command == "ui":
        from .workers.worker_camera_ui import main as ui_main

        return ui_main(["--service", args.service, "--frames", str(args.frames),
                        "--timeout", str(args.timeout), *(["--sam3"] if args.sam3 else []),
                        *(["--sam3d", "--sam3d-depth-source", args.sam3d_depth_source] if args.sam3d else []),
                        *(["--sam3d-texture-baking"] if args.sam3d and args.sam3d_texture_baking else []),
                        *(["--foundationpose"] if args.foundationpose else []),
                        *(["--vlm"] if args.vlm else []),
                        *(["--graspgenx"] if args.graspgenx else []),
                        *(["--grasp-selection"] if args.grasp_selection else []),
                        *(["--policy-catalog", args.vlm_policy_catalog] if args.vlm_policy_catalog is not None else []),
                        *(["--session-dir", args.vlm_session_dir] if args.vlm_session_dir is not None else []),
                        *(["--asset-dir", args.asset_dir] if args.asset_dir is not None else [])])
    return WorkerSupervisor(build_commands(args)).run()


if __name__ == "__main__":
    raise SystemExit(main())
