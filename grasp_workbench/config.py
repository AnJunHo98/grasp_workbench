"""Small YAML defaults for camera/SAM examples, without runtime imports."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
_DEFAULTS = {
    "camera": {"type": "zed", "serial": None, "resolution": "HD720", "fps": 30},
    "ipc": {"service": "grasp_workbench/camera/rgbd"},
    "ui": {"enabled": True},
    "recording": {"enabled": False, "python": None, "output_dir": "runs/datasets", "fps": 30, "source": "pose"},
    "aprilcube": {"enabled": False, "python": None, "source_dir": None,
                  "config_a": "examples/aprilcube/cube_a.json", "config_b": "examples/aprilcube/cube_b.json",
                  "enable_filter": True, "fast": True, "log_timing": False, "overlay_alpha": 0.65},
    "sam3": {"enabled": False, "python": None, "checkpoint": None, "device": "cuda"},
    "sam3d": {"enabled": False, "python": None, "source_dir": None,
              "config_path": None, "output_dir": "runs/meshes", "seed": 42,
              "depth_source": "depth", "texture_baking": False},
    "foundationpose": {"enabled": False, "python": None, "source_dir": None,
                       "device": "cuda:0", "registration_iterations": 5, "tracking_iterations": 1},
    "vlm": {"enabled": False, "python": None, "model": "gpt-6.1-sol", "backend": "openai_api",
            "codex_executable": "codex", "api_key_file": None, "asset_dir": "runs/assets",
            "policy_catalog": None, "session_dir": "runs/tasks"},
    "graspgenx": {"enabled": False, "python": None, "source_dir": None, "checkpoint_dir": None,
                 "assets_dir": None, "gripper_id": "franka_panda", "output_dir": "runs/grasps", "settings": {}},
    "grasp_selection": {"enabled": False, "python": None, "robot_config": None, "settings": {}},
    "run": {"frames": 0, "timeout": 0.0},
}
APRILCUBE_DISABLED_WORKERS = ("vlm", "sam3", "sam3d", "foundationpose", "graspgenx", "grasp_selection")


def apply_worker_mode(options):
    """Apply the exclusive AprilCube mode to runtime options, never to YAML files."""
    if options.get("aprilcube", False):
        for name in APRILCUBE_DISABLED_WORKERS:
            options[name] = False
        options["ui"] = True
    return options


def _value(section, key, value, directory):
    field = f"{section}.{key}"
    if section in {"graspgenx", "grasp_selection"} and key == "settings":
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a mapping")
        return value
    if field == "camera.serial":
        if value is None:
            return None
        if type(value) not in (str, int) or value == "":
            raise ValueError(f"{field} must be a string, integer or null")
        return str(value)
    if section in {"recording", "aprilcube", "sam3", "sam3d", "foundationpose", "vlm", "graspgenx", "grasp_selection"} and key in {
        "python", "checkpoint", "source_dir", "config_path", "output_dir", "api_key_file", "asset_dir", "codex_executable",
        "checkpoint_dir", "assets_dir",
        "policy_catalog", "session_dir",
        "robot_config",
        "config_a", "config_b",
    }:
        if value is None and key not in {"output_dir", "asset_dir", "codex_executable", "session_dir", "config_a", "config_b"}:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string or null")
        if key not in {"python", "codex_executable"} or "/" in value or "\\" in value or value.startswith((".", "~")):
            path = Path(value).expanduser()
            path = path if path.is_absolute() else directory / path
            return os.path.abspath(path) if key in {"python", "codex_executable"} else str(path.resolve())
        return value
    default = _DEFAULTS[section][key]
    if type(default) is bool:
        if type(value) is not bool:
            raise ValueError(f"{field} must be a boolean")
    elif type(default) is int:
        restriction = "> 0" if key == "fps" else ">= 0" if field == "run.frames" else ""
        if (type(value) is not int or (key == "fps" and value <= 0)
                or (field == "run.frames" and value < 0)):
            raise ValueError(f"{field} must be an integer {restriction}".rstrip())
    elif type(default) is float:
        if type(value) not in (int, float):
            raise ValueError(f"{field} must be a finite number >= 0")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError(f"{field} must be a finite number >= 0") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be a finite number >= 0")
    elif not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    if field == "camera.type" and value not in {"zed", "realsense"}:
        raise ValueError("camera.type must be zed or realsense")
    if field == "sam3d.depth_source" and value not in {"depth", "monocular"}:
        raise ValueError("sam3d.depth_source must be depth or monocular")
    if field == "recording.source" and value not in {"camera", "pose"}:
        raise ValueError("recording.source must be camera or pose")
    if field == "vlm.backend" and value not in {"openai_api", "codex_cli", "anthropic_api"}:
        raise ValueError("vlm.backend must be openai_api, codex_cli or anthropic_api")
    if field == "aprilcube.overlay_alpha" and value > 1:
        raise ValueError("aprilcube.overlay_alpha must be a finite number between 0 and 1")
    return value


def _merge(path, sections):
    try:
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML ({getattr(error, 'problem', str(error))})") from error
    if record is None:
        record = {}
    if not isinstance(record, dict):
        raise ValueError(f"{path}: configuration must be a mapping")
    unknown = set(record) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"{path}: unknown sections: {', '.join(sorted(map(str, unknown)))}")
    for section, options in record.items():
        if not isinstance(options, dict):
            raise ValueError(f"{path}: {section} must be a mapping")
        unknown = set(options) - set(_DEFAULTS[section])
        if unknown:
            raise ValueError(f"{path}: unknown {section} keys: {', '.join(sorted(map(str, unknown)))}")
        for key, value in options.items():
            sections[section][key] = _value(section, key, value, path.parent)


def load_run_config(path=None) -> dict:
    """Return flat CLI defaults; only implicit shared config uses a local overlay."""
    selected = Path(path if path is not None else DEFAULT_CONFIG_PATH).expanduser().resolve()
    sections = {name: dict(options) for name, options in _DEFAULTS.items()}
    for name in ("config_a", "config_b"):
        sections["aprilcube"][name] = str((ROOT / sections["aprilcube"][name]).resolve())
    sections["sam3d"]["output_dir"] = str((ROOT / "runs/meshes").resolve())
    sections["vlm"]["asset_dir"] = str((ROOT / "runs/assets").resolve())
    sections["vlm"]["session_dir"] = str((ROOT / "runs/tasks").resolve())
    sections["graspgenx"]["output_dir"] = str((ROOT / "runs/grasps").resolve())
    sections["recording"]["output_dir"] = str((ROOT / "runs/datasets").resolve())
    sections["graspgenx"]["settings"] = {}
    sections["grasp_selection"]["settings"] = {}
    _merge(selected, sections)
    local = selected.with_name("config.local.yaml")
    if path is None and local.exists():
        _merge(local, sections)
    camera, sam3, sam3d = sections["camera"], sections["sam3"], sections["sam3d"]
    foundationpose = sections["foundationpose"]
    vlm = sections["vlm"]
    graspgenx = sections["graspgenx"]
    selection = sections["grasp_selection"]
    aprilcube = sections["aprilcube"]
    recording = sections["recording"]
    return apply_worker_mode({
        "camera": camera["type"], "serial": camera["serial"],
        "resolution": camera["resolution"], "fps": camera["fps"],
        "service": sections["ipc"]["service"], "ui": sections["ui"]["enabled"],
        "recording": recording["enabled"], "recording_python": recording["python"] or sys.executable,
        "recording_output_dir": recording["output_dir"], "recording_fps": recording["fps"],
        "recording_source": recording["source"],
        "aprilcube": aprilcube["enabled"], "aprilcube_python": aprilcube["python"] or sys.executable,
        "aprilcube_source_dir": aprilcube["source_dir"] or str((ROOT / "third_party/aprilcube").resolve()),
        "aprilcube_config_a": aprilcube["config_a"], "aprilcube_config_b": aprilcube["config_b"],
        "aprilcube_enable_filter": aprilcube["enable_filter"], "aprilcube_overlay_alpha": aprilcube["overlay_alpha"],
        "aprilcube_fast": aprilcube["fast"], "aprilcube_log_timing": aprilcube["log_timing"],
        "sam3": sam3["enabled"], "sam3_python": sam3["python"] or sys.executable,
        "sam3_checkpoint": sam3["checkpoint"], "sam3_device": sam3["device"],
        "sam3d": sam3d["enabled"], "sam3d_python": sam3d["python"] or sys.executable,
        "sam3d_source_dir": sam3d["source_dir"] or str((ROOT / "third_party/sam-3d-objects").resolve()),
        "sam3d_config_path": sam3d["config_path"], "sam3d_output_dir": sam3d["output_dir"],
        "sam3d_seed": sam3d["seed"], "sam3d_depth_source": sam3d["depth_source"],
        "sam3d_texture_baking": sam3d["texture_baking"],
        "foundationpose": foundationpose["enabled"],
        "foundationpose_python": foundationpose["python"] or sys.executable,
        "foundationpose_source_dir": foundationpose["source_dir"] or str((ROOT / "third_party/FoundationPose").resolve()),
        "foundationpose_device": foundationpose["device"],
        "foundationpose_registration_iterations": foundationpose["registration_iterations"],
        "foundationpose_tracking_iterations": foundationpose["tracking_iterations"],
        "vlm": vlm["enabled"], "vlm_python": vlm["python"] or sys.executable,
        "vlm_model": vlm["model"], "vlm_api_key_file": vlm["api_key_file"],
        "vlm_backend": vlm["backend"], "vlm_codex_executable": vlm["codex_executable"],
        "asset_dir": vlm["asset_dir"],
        "vlm_policy_catalog": vlm["policy_catalog"], "vlm_session_dir": vlm["session_dir"],
        "graspgenx": graspgenx["enabled"], "graspgenx_python": graspgenx["python"] or sys.executable,
        "graspgenx_source_dir": graspgenx["source_dir"] or str((ROOT / "third_party/GraspGenX").resolve()),
        "graspgenx_checkpoint_dir": graspgenx["checkpoint_dir"], "graspgenx_assets_dir": graspgenx["assets_dir"],
        "graspgenx_gripper_id": graspgenx["gripper_id"], "graspgenx_output_dir": graspgenx["output_dir"],
        "graspgenx_settings": graspgenx["settings"],
        "grasp_selection": selection["enabled"],
        "grasp_selection_python": selection["python"] or sys.executable,
        "grasp_selection_robot_config": selection["robot_config"],
        "grasp_selection_settings": selection["settings"],
        "frames": sections["run"]["frames"], "timeout": sections["run"]["timeout"],
    })
