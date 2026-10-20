"""Capture RGB-D with the selected SDK and publish unchanged frames on iceoryx2."""
import argparse
import time

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record

DEFAULT_SERVICE = "grasp_workbench/camera/rgbd"


class CameraWorker(SingleRateWorker):
    def __init__(self, context, camera, service=DEFAULT_SERVICE, frames=0, *, publisher_factory=None):
        super().__init__(context)
        self.camera, self.service, self.frames = camera, service, frames
        self.publisher_factory, self.publisher = publisher_factory, None
        self.published_count, self._last_log = 0, None

    def on_start(self):
        self.camera.open()
        if self.publisher_factory is None:
            from ..ipc.camera import CameraPublisher

            self.publisher_factory = CameraPublisher
        self.publisher = self.publisher_factory(self.service)
        log_record("camera-open", worker=self.ctx.name, camera=type(self.camera).__name__, service=self.service)

    def step_once(self):
        frame = self.camera.read()
        receivers = self.publisher.publish(frame)
        self.published_count += 1
        now = time.monotonic()
        if self._last_log is None or now - self._last_log >= 1:
            log_record("published", published_count=self.published_count,
                       receivers=receivers, metadata=frame.metadata)
            self._last_log = now
        if self.frames and self.published_count >= self.frames:
            self.ctx.stop()

    def on_stop(self):
        try:
            if self.publisher is not None:
                self.publisher.close()
        finally:
            try:
                self.camera.close()
            finally:
                log_record("camera-stopped", published_count=self.published_count)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("zed", "realsense"), default="zed")
    parser.add_argument("--serial")
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--resolution", default="HD720", help="ZED resolution enum")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=0, help="Published frames before stopping; 0 runs continuously")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0 or args.fps <= 0:
        parser.error("--frames must be nonnegative and --fps must be positive")
    if args.camera == "zed":
        from ..cameras.zed import ZedCamera

        camera = ZedCamera(serial=args.serial, resolution=args.resolution, fps=args.fps)
    else:
        from ..cameras.realsense import RealSenseCamera

        camera = RealSenseCamera(serial=args.serial, fps=args.fps)
    try:
        CameraWorker(WorkerContext("camera"), camera, args.service, args.frames).run()
    except Exception as error:
        log_record("error", worker="camera", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
