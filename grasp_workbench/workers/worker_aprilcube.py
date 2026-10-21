"""Continuous AprilCube pose estimation; display rendering is a separate worker."""
import argparse
import time

import numpy as np

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from ..ipc.aprilcube import AprilCubeTarget, services
from ..ipc.foundationpose import ObjectPose, PoseResult, services as pose_services
from .worker_camera import DEFAULT_SERVICE


class AprilCubeWorker(SingleRateWorker):
    def __init__(self, context, source_dir, config_a, config_b, service=DEFAULT_SERVICE,
                 filter_enabled=True, frames=0, *, fast=True, log_timing=False,
                 camera_subscriber_factory=None, control_subscriber_factory=None,
                 result_publisher_factory=None, predictor_factory=None):
        super().__init__(context, hz=100)
        self.source_dir, self.config_a, self.config_b = source_dir, config_a, config_b
        self.service, self.filter_enabled = service, filter_enabled
        self.frames = frames
        self.fast, self.log_timing = fast, log_timing
        self.camera_subscriber_factory = camera_subscriber_factory
        self.control_subscriber_factory = control_subscriber_factory
        self.result_publisher_factory = result_publisher_factory
        self.predictor_factory = predictor_factory
        self.camera = self.control_subscriber = self.publisher = None
        self.predictor = None
        self.targets = tuple(AprilCubeTarget(slot, "aprilcube-" + slot, "", 1., np.eye(4))
                             for slot in ("A", "B"))
        self.processed_count = 0

    def on_start(self):
        from ..ipc.aprilcube import AprilCubeControlSubscriber
        from ..ipc.camera import CameraSubscriber
        from ..ipc.foundationpose import PoseResultPublisher
        from ..models.aprilcube import OfficialAprilCubePredictor

        self.camera = (self.camera_subscriber_factory or CameraSubscriber)(self.service)
        self.control_subscriber = (self.control_subscriber_factory or AprilCubeControlSubscriber)(
            services(self.service)[0])
        self.publisher = (self.result_publisher_factory or PoseResultPublisher)(pose_services(self.service)[1])
        self.predictor = (self.predictor_factory or OfficialAprilCubePredictor)(
            self.source_dir, self.config_a, self.config_b, filter_enabled=self.filter_enabled,
            fast=self.fast, log_timing=self.log_timing)
        self.predictor.start()
        log_record("ready", worker=self.ctx.name, service=self.service)

    def step_once(self):
        started = time.perf_counter_ns()
        self.publisher.update_connections()
        control = self.control_subscriber.receive_latest()
        if control is not None:
            self.targets = control.targets
        frame = self.camera.receive_latest()
        if frame is None:
            return
        received = time.perf_counter_ns()
        cube_poses = self.predictor.predict(frame)
        predicted = time.perf_counter_ns()
        objects = []
        for target in self.targets:
            cube_pose = cube_poses.get(target.object_id)
            pose = None if cube_pose is None else cube_pose @ target.cube_from_object
            objects.append(ObjectPose(target.object_id, target.session_id, target.mesh_path,
                                      "unavailable" if pose is None else "tracking", pose,
                                      pose_source="aprilcube",
                                      diagnostics=self.predictor.diagnostics.get(target.object_id)))
        composed = time.perf_counter_ns()
        self.publisher.publish(PoseResult(frame, tuple(objects)))
        pose_published = time.perf_counter_ns()
        source_time = frame.metadata.get("received_monotonic_ns")
        if self.log_timing:
            pose_age = ((time.monotonic_ns() - source_time) / 1e6
                        if source_time is not None else None)
            log_record("timing", worker=self.ctx.name, sequence=frame.metadata.get("sequence"),
                       fast=self.fast, timings_ms={
                           **self.predictor.timings_ms,
                           "receive_ms": (received - started) / 1e6,
                           "predict_ms": (predicted - received) / 1e6,
                           "compose_ms": (composed - predicted) / 1e6,
                           "pose_publish_ms": (pose_published - composed) / 1e6,
                           "total_ms": (pose_published - started) / 1e6,
                           "source_age_at_pose_ms": pose_age,
                       })
        self.processed_count += 1
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()

    def on_stop(self):
        for endpoint in (self.predictor, self.publisher,
                         self.control_subscriber, self.camera):
            if endpoint is not None:
                endpoint.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--no-filter", action="store_true")
    fast = parser.add_mutually_exclusive_group()
    fast.add_argument("--fast", action="store_true", default=True)
    fast.add_argument("--no-fast", dest="fast", action="store_false")
    parser.add_argument("--log-timing", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    try:
        AprilCubeWorker(WorkerContext("aprilcube"), args.source_dir, args.config_a, args.config_b,
                        args.service, not args.no_filter, args.frames,
                        fast=args.fast, log_timing=args.log_timing).run()
    except Exception as error:
        log_record("error", worker="aprilcube", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
