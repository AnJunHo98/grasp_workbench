"""On-demand mesh-only GraspGenX generation in an independent worker."""
import argparse
import json

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from ..ipc.graspgenx import GraspResult, services
from .worker_camera import DEFAULT_SERVICE


class GraspGenXWorker(SingleRateWorker):
    def __init__(self, context, source_dir=None, *, checkpoint_dir=None, assets_dir=None,
                 gripper_id="franka_panda", output_dir="runs/grasps", settings=None,
                 service=DEFAULT_SERVICE, frames=0, subscriber_factory=None,
                 publisher_factory=None, predictor_factory=None):
        super().__init__(context, hz=100)
        self.source_dir, self.checkpoint_dir, self.assets_dir = source_dir, checkpoint_dir, assets_dir
        self.gripper_id, self.output_dir = gripper_id, output_dir
        self.settings = {} if settings is None else settings
        self.service, self.frames = service, frames
        self.subscriber_factory, self.publisher_factory = subscriber_factory, publisher_factory
        self.predictor_factory = predictor_factory
        self.subscriber = self.publisher = self.predictor = None
        self.processed_count, self._last_result = 0, None

    def on_start(self):
        from ..ipc.graspgenx import GraspRequestSubscriber, GraspResultPublisher

        request_service, result_service = services(self.service)
        self.subscriber = (self.subscriber_factory or GraspRequestSubscriber)(request_service)
        self.publisher = (self.publisher_factory or GraspResultPublisher)(result_service)
        log_record("ready", worker=self.ctx.name, service=self.service)

    def _start_predictor(self):
        if self.predictor is not None:
            return
        if self.predictor_factory is None:
            from ..models.graspgenx import OfficialGraspGenXPredictor

            self.predictor_factory = OfficialGraspGenXPredictor
        predictor = self.predictor_factory(
            self.source_dir, checkpoint_dir=self.checkpoint_dir, assets_dir=self.assets_dir,
            gripper_id=self.gripper_id, output_dir=self.output_dir, settings=self.settings,
        )
        try:
            predictor.start()
        except Exception:
            predictor.close()
            raise
        self.predictor = predictor

    def _result(self, request, status, **fields):
        return GraspResult(request.request_id, request.asset_id, request.object_id, status,
                           mesh_path=request.mesh_path, mesh_scale_to_m=request.mesh_scale_to_m,
                           gripper_id=self.gripper_id, **fields)

    def step_once(self):
        self.publisher.update_connections()
        request = self.subscriber.receive_latest()
        if request is None:
            return
        if self._last_result is not None and request.request_id == self._last_result.request_id:
            self.publisher.publish(self._last_result)
            return
        self.publisher.publish(self._result(request, "running"))
        try:
            self._start_predictor()
            prediction = self.predictor.predict(request)
            result = self._result(request, "done", grasps=prediction.grasps, scores=prediction.scores,
                                  bundle_path=prediction.bundle_path, gripper_metadata=prediction.gripper_metadata,
                                  cached=prediction.cached)
        except Exception as error:
            result = self._result(request, "error", error=f"{type(error).__name__}: {error}")
        self._last_result = result
        self.publisher.publish(result)
        self.processed_count += 1
        log_record("graspgenx-result", request_id=result.request_id, object_id=result.object_id,
                   status=result.status, cached=result.cached, error=result.error)
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()

    def on_stop(self):
        try:
            if self.predictor is not None:
                self.predictor.close()
        finally:
            try:
                if self.publisher is not None:
                    self.publisher.close()
            finally:
                if self.subscriber is not None:
                    self.subscriber.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--assets-dir")
    parser.add_argument("--gripper-id", default="franka_panda")
    parser.add_argument("--output-dir", default="runs/grasps")
    parser.add_argument("--settings", type=json.loads, default={})
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--frames", type=int, default=0, help="Terminal requests; 0 waits continuously")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not isinstance(args.settings, dict):
        parser.error("--settings must be a JSON object")
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    try:
        GraspGenXWorker(WorkerContext("graspgenx"), args.source_dir, checkpoint_dir=args.checkpoint_dir,
                        assets_dir=args.assets_dir, gripper_id=args.gripper_id, output_dir=args.output_dir,
                        settings=args.settings, service=args.service, frames=args.frames).run()
    except Exception as error:
        log_record("error", worker="graspgenx", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
