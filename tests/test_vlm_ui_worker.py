"""CLI delegation only: no real Qt window, API, key file or native IPC."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from grasp_workbench.workers.worker_camera_ui import build_parser, main


@pytest.mark.parametrize("argv,options", [
    (["--vlm"], {"sam3": True, "vlm": True}),
    (["--vlm", "--asset-dir", "/example/assets"],
     {"sam3": True, "vlm": True, "asset_dir": "/example/assets"}),
    (["--vlm", "--sam3d", "--foundationpose"],
     {"sam3": True, "sam3d": True, "sam3d_depth_source": "depth", "sam3d_texture_baking": False,
      "foundationpose": True, "vlm": True}),
    (["--sam3d", "--sam3d-depth-source", "monocular", "--sam3d-texture-baking"],
     {"sam3": True, "sam3d": True, "sam3d_depth_source": "monocular", "sam3d_texture_baking": True}),
    (["--asset-dir", "/example/assets"], {"asset_dir": "/example/assets"}),
    (["--graspgenx"], {"graspgenx": True}),
    (["--vlm", "--policy-catalog", "/example/policies.json", "--session-dir", "/example/tasks"],
     {"sam3": True, "vlm": True, "policy_catalog": "/example/policies.json", "session_dir": "/example/tasks"}),
])
def test_ui_worker_passes_vlm_and_asset_options_without_loading_models_or_keys(monkeypatch, argv, options):
    calls = []

    class Timer:
        timeout = SimpleNamespace(connect=lambda _callback: None)

        def start(self, *_args):
            pass

        def stop(self):
            pass

    class Window:
        exit_code = 0
        receiver = SimpleNamespace(stop=lambda: calls.append("stop"), join=lambda: calls.append("join"))

        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def show(self):
            pass

    qt = ModuleType("PyQt6")
    app = SimpleNamespace(exec=lambda: 0)
    qt.QtCore = SimpleNamespace(QTimer=Timer)
    qt.QtWidgets = SimpleNamespace(QApplication=SimpleNamespace(instance=lambda: app))
    camera_window = ModuleType("grasp_workbench.ui.camera_window")
    camera_window.CameraWindow = Window
    monkeypatch.setitem(sys.modules, "PyQt6", qt)
    monkeypatch.setitem(sys.modules, "grasp_workbench.ui.camera_window", camera_window)
    assert main(["--service", "example/rgbd", *argv]) == 0
    assert calls == [(("example/rgbd", 0, 0), options), "stop", "join"]


def test_ui_parser_exposes_no_vlm_key_options():
    args = build_parser().parse_args(["--vlm", "--asset-dir", "library"])
    assert args.vlm and args.asset_dir == "library"
    assert not hasattr(args, "api_key_file") and not hasattr(args, "vlm_api_key_file")
