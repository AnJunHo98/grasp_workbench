"""Render AprilCube display layers independently of pose estimation."""
import argparse
import time

import numpy as np

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from ..ipc.aprilcube import AprilCubeOverlay, AprilCubeTarget, services
from ..ipc.foundationpose import services as pose_services
from .worker_camera import DEFAULT_SERVICE


class AprilCubeOverlayWorker(SingleRateWorker):
    def __init__(self, context, service=DEFAULT_SERVICE, overlay_alpha=.65, frames=0, *,
                 log_timing=False, result_subscriber_factory=None,
                 control_subscriber_factory=None, overlay_publisher_factory=None,
                 renderer_factory=None):
        super().__init__(context, hz=100)
        self.service, self.overlay_alpha, self.frames = service, overlay_alpha, frames
        self.log_timing = log_timing
        self.result_subscriber_factory = result_subscriber_factory
        self.control_subscriber_factory = control_subscriber_factory
        self.overlay_publisher_factory = overlay_publisher_factory
        self.renderer_factory = renderer_factory
        self.result_subscriber = self.control_subscriber = self.overlay_publisher = None
        self.renderer = None
        self.targets = tuple(AprilCubeTarget(slot, "aprilcube-" + slot, "", 1., np.eye(4))
                             for slot in ("A", "B"))
        self.processed_count = 0

    def on_start(self):
        from ..ipc.aprilcube import AprilCubeControlSubscriber, AprilCubeOverlayPublisher
        from ..ipc.foundationpose import PoseResultSubscriber
        from ..models.aprilcube_overlay import MeshOverlayRenderer

        self.result_subscriber = (self.result_subscriber_factory or PoseResultSubscriber)(
            pose_services(self.service)[1])
        self.control_subscriber = (self.control_subscriber_factory or AprilCubeControlSubscriber)(
            services(self.service)[0])
        self.overlay_publisher = (self.overlay_publisher_factory or AprilCubeOverlayPublisher)(
            services(self.service)[1])
        self.renderer = (self.renderer_factory or MeshOverlayRenderer)()
        log_record("ready", worker=self.ctx.name, service=self.service)

    def step_once(self):
        started = time.perf_counter_ns()
        self.overlay_publisher.update_connections()
        control = self.control_subscriber.receive_latest()
        if control is not None:
            self.targets = control.targets
        result = self.result_subscriber.receive_latest()
        if result is None:
            return
        received = time.perf_counter_ns()
        objects = {(item.object_id, item.session_id): item for item in result.objects
                   if item.pose_source == "aprilcube" and item.status == "tracking"}
        poses = {}
        for target in self.targets:
            item = objects.get((target.object_id, target.session_id))
            poses[target.object_id] = None if item is None else item.camera_from_object
        rgba = self.renderer.render(result.frame, self.targets, poses, self.overlay_alpha)
        rendered = time.perf_counter_ns()
        self.overlay_publisher.publish(AprilCubeOverlay(dict(result.frame.metadata), rgba))
        published = time.perf_counter_ns()
        if self.log_timing:
            source_time = result.frame.metadata.get("received_monotonic_ns")
            source_age = ((time.monotonic_ns() - source_time) / 1e6
                          if source_time is not None else None)
            log_record("timing", worker=self.ctx.name,
                       sequence=result.frame.metadata.get("sequence"), timings_ms={
                           "receive_ms": (received - started) / 1e6,
                           "render_ms": (rendered - received) / 1e6,
                           "overlay_publish_ms": (published - rendered) / 1e6,
                           "total_ms": (published - started) / 1e6,
                           "source_age_at_overlay_ms": source_age,
                       })
        self.processed_count += 1
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()

    def on_stop(self):
        for endpoint in (self.overlay_publisher, self.control_subscriber,
                         self.result_subscriber):
            if endpoint is not None:
                endpoint.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--overlay-alpha", type=float, default=.65)
    parser.add_argument("--log-timing", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if not 0 <= args.overlay_alpha <= 1:
        parser.error("--overlay-alpha must be in [0, 1]")
    try:
        AprilCubeOverlayWorker(WorkerContext("aprilcube_overlay"), args.service,
                               args.overlay_alpha, args.frames,
                               log_timing=args.log_timing).run()
    except Exception as error:
        log_record("error", worker="aprilcube_overlay", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
