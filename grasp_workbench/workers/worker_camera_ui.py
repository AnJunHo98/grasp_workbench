"""Run a separate optional Qt process subscribed to the camera iceoryx2 service."""
import argparse
import math
import signal
import sys
from threading import Event

from ..core.worker_base import log_record
from .worker_camera import DEFAULT_SERVICE


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--frames", type=int, default=0, help="Received frames before closing; 0 runs continuously")
    parser.add_argument("--timeout", type=float, default=0,
                        help="Maximum seconds; an unmet finite frame count is an error; 0 waits")
    parser.add_argument("--sam3", action="store_true", help="A/B text prompts and masks from a separate SAM3 worker")
    parser.add_argument("--sam3d", action="store_true", help="Selected A/B mask to mesh from a separate SAM3D worker")
    parser.add_argument("--sam3d-depth-source", choices=("depth", "monocular"), default="depth")
    parser.add_argument("--sam3d-texture-baking", action="store_true")
    parser.add_argument("--foundationpose", action="store_true", help="A/B mesh registration and tracking in a separate worker")
    parser.add_argument("--aprilcube", action="store_true", help="AprilCube pose/mesh overlay only; overrides all model flags")
    parser.add_argument("--vlm", action="store_true", help="Request object prompts from a separate VLM worker")
    parser.add_argument("--asset-dir", help="UI asset library directory")
    parser.add_argument("--policy-catalog", help="Local semantic policy catalog JSON file")
    parser.add_argument("--session-dir", help="UI task session directory")
    parser.add_argument("--graspgenx", action="store_true", help="Request object-local grasps from a separate worker")
    parser.add_argument("--grasp-selection", action="store_true", help="Select grasps and display optional robot-frame reachability")
    parser.add_argument("--recording", action="store_true", help="Control a separate LeRobot recording worker")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0 or args.timeout < 0 or not math.isfinite(args.timeout):
        parser.error("--frames and finite --timeout must be nonnegative")
    try:
        from PyQt6 import QtCore, QtWidgets
        from ..ui.camera_window import CameraWindow
    except ImportError as error:
        log_record("error", worker="camera_ui", error=f"Install the optional UI dependencies: {error}")
        return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    stopped = Event()
    previous = {sig: signal.signal(sig, lambda *_args: stopped.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    timer = QtCore.QTimer()
    window = None
    try:
        options = {"sam3": True} if args.sam3 else {}
        if args.sam3d:
            options.update(sam3d=True, sam3d_depth_source=args.sam3d_depth_source,
                           sam3d_texture_baking=args.sam3d_texture_baking)
        if args.foundationpose:
            options["foundationpose"] = True
        if args.vlm:
            options["vlm"] = True
        if args.graspgenx:
            options["graspgenx"] = True
        if args.grasp_selection:
            options["grasp_selection"] = True
        if args.asset_dir is not None:
            options["asset_dir"] = args.asset_dir
        if args.policy_catalog is not None:
            options["policy_catalog"] = args.policy_catalog
        if args.session_dir is not None:
            options["session_dir"] = args.session_dir
        if args.aprilcube:
            options = {"aprilcube": True}
        if args.recording:
            options["recording"] = True
        window = CameraWindow(args.service, args.frames, args.timeout, **options)
        timer.timeout.connect(lambda: window.close() if stopped.is_set() else None)
        timer.start(50)
        window.show()
        app.exec()
        return window.exit_code
    except Exception as error:
        log_record("error", worker="camera_ui", error=f"{type(error).__name__}: {error}")
        return 1
    finally:
        timer.stop()
        if window is not None:
            window.receiver.stop()
            window.receiver.join()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
