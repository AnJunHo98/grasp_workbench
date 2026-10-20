"""Independent RGB-D subscriber; stdout contains summaries, not image payloads."""
import argparse
import time

import numpy as np

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from .worker_camera import DEFAULT_SERVICE


class CameraMonitorWorker(SingleRateWorker):
    def __init__(self, context, service=DEFAULT_SERVICE, frames=0, timeout=0, *, subscriber_factory=None):
        super().__init__(context, hz=100)
        self.service, self.frames, self.timeout = service, frames, timeout
        self.subscriber_factory, self.subscriber = subscriber_factory, None
        self.received_count, self._last_log = 0, None

    def on_start(self):
        if self.subscriber_factory is None:
            from ..ipc.camera import CameraSubscriber

            self.subscriber_factory = CameraSubscriber
        self.subscriber = self.subscriber_factory(self.service)
        self._started = time.monotonic()
        log_record("ready", worker=self.ctx.name, service=self.service)

    def step_once(self):
        now = time.monotonic()
        if self.timeout and now - self._started >= self.timeout:
            if self.frames and self.received_count < self.frames:
                raise TimeoutError(f"Received {self.received_count}/{self.frames} frames before timeout")
            self.ctx.stop()
            return
        frame = self.subscriber.receive_latest()
        if frame is None:
            return
        self.received_count += 1
        if self._last_log is None or now - self._last_log >= 1 or self.received_count == self.frames:
            valid = frame.depth_m[np.isfinite(frame.depth_m) & (frame.depth_m > 0)]
            log_record("frame", received_count=self.received_count, rgb_shape=list(frame.rgb.shape),
                       depth={"shape": list(frame.depth_m.shape), "valid_count": int(valid.size),
                              "min_m": float(valid.min()) if valid.size else None,
                              "max_m": float(valid.max()) if valid.size else None,
                              "mean_m": float(valid.mean()) if valid.size else None},
                       intrinsics=frame.metadata.get("intrinsics"), metadata=frame.metadata)
            self._last_log = now
        if self.frames and self.received_count >= self.frames:
            self.ctx.stop()

    def on_stop(self):
        try:
            if self.subscriber is not None:
                self.subscriber.close()
        finally:
            log_record("monitor-stopped", received_count=self.received_count)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--frames", type=int, default=0, help="Received frames before stopping; 0 runs continuously")
    parser.add_argument("--timeout", type=float, default=0,
                        help="Maximum seconds; unmet finite frame count is an error; 0 waits continuously")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0 or args.timeout < 0 or not np.isfinite(args.timeout):
        parser.error("--frames and finite --timeout must be nonnegative")
    try:
        CameraMonitorWorker(WorkerContext("camera_monitor"), args.service, args.frames, args.timeout).run()
    except Exception as error:
        log_record("error", worker="camera_monitor", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
