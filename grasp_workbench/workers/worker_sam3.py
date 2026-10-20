"""One SAM3 model, independent text prompts, latest RGB-D frames via iceoryx2."""

import argparse
from contextlib import ExitStack
import math

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from .worker_camera import DEFAULT_SERVICE


class Sam3Worker(SingleRateWorker):
    def __init__(
        self, context, checkpoint, service=DEFAULT_SERVICE, device="cuda",
        confidence_threshold=0.5, frames=0, *, camera_subscriber_factory=None,
        prompt_subscriber_factory=None, publisher_factory=None, predictor_factory=None,
        refine_subscriber_factory=None, refine_publisher_factory=None,
    ):
        super().__init__(context, hz=100)
        self.checkpoint, self.service, self.device = checkpoint, service, device
        self.confidence_threshold, self.frames = confidence_threshold, frames
        self.camera_subscriber_factory = camera_subscriber_factory
        self.prompt_subscriber_factory = prompt_subscriber_factory
        self.publisher_factory, self.predictor_factory = publisher_factory, predictor_factory
        self.refine_subscriber_factory, self.refine_publisher_factory = refine_subscriber_factory, refine_publisher_factory
        self.camera = self.prompt_subscriber = self.publisher = self.predictor = None
        self.refine_subscriber = self.refine_publisher = None
        self.prompts = self.latest_frame = self._processed_key = None
        self.processed_count = 0

    def on_start(self):
        from ..ipc.camera import CameraSubscriber
        from ..ipc.sam3 import PromptSubscriber, Sam3Publisher, services
        from ..ipc.sam3_refine import RefineRequestSubscriber, RefineResultPublisher, services as refine_services

        prompt_service, result_service = services(self.service)
        self.camera = (self.camera_subscriber_factory or CameraSubscriber)(self.service)
        self.prompt_subscriber = (self.prompt_subscriber_factory or PromptSubscriber)(prompt_service)
        self.publisher = (self.publisher_factory or Sam3Publisher)(result_service)
        request_service, result_service = refine_services(self.service)
        self.refine_subscriber = (self.refine_subscriber_factory or RefineRequestSubscriber)(request_service)
        self.refine_publisher = (self.refine_publisher_factory or RefineResultPublisher)(result_service)
        log_record("ready", worker=self.ctx.name, service=self.service)

    def _start_predictor(self):
        if self.predictor is None:
            if self.predictor_factory is None:
                from ..models.sam3 import OfficialSam3Predictor

                self.predictor_factory = OfficialSam3Predictor
            self.predictor = self.predictor_factory(
                self.checkpoint, device=self.device, confidence_threshold=self.confidence_threshold,
            )
            self.predictor.start()

    def _refine(self, request):
        from ..ipc.sam3_refine import RefineResult

        self.refine_publisher.update_connections()
        self.refine_publisher.publish(RefineResult(request.request_id, "running"))
        try:
            self._start_predictor()
            result = RefineResult(request.request_id, "done", self.predictor.refine(request))
        except Exception as error:
            result = RefineResult(request.request_id, "error", error=f"{type(error).__name__}: {error}")
        self.refine_publisher.publish(result)
        self.processed_count += 1
        log_record("sam3-refine-result", request_id=request.request_id, status=result.status, error=result.error)
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()

    def step_once(self):
        request = self.refine_subscriber.receive_latest()
        if request is not None:
            self._refine(request)
            return
        incoming = self.prompt_subscriber.receive_latest()
        if incoming is not None:
            self.prompts = incoming
        frame = self.camera.receive_latest()
        if frame is not None:
            self.latest_frame = frame
        if self.prompts is None or not self.prompts.objects or self.latest_frame is None:
            return
        if self.predictor is None:
            self._start_predictor()
            # Loading may take seconds; sample fresh commands/frame next iteration.
            return
        frame, prompts = self.latest_frame, self.prompts
        key = (*tuple(frame.metadata.get(name) for name in ("camera_id", "stream_id", "sequence")),
               prompts.revision)
        if key == self._processed_key:
            return
        result = self.predictor.predict(frame, prompts)
        self._processed_key = key
        incoming = self.prompt_subscriber.receive_latest()
        if incoming is not None:
            self.prompts = incoming
        if self.ctx.stop_event.is_set() or self.prompts.revision != prompts.revision:
            return
        self.publisher.publish(result)
        self.processed_count += 1
        log_record("sam3-result", processed_count=self.processed_count,
                   revision=result.revision, metadata=frame.metadata,
                   objects=[{"object_id": obj.object_id, "text": obj.text,
                             "candidate_count": len(obj.scores), "error": obj.error}
                            for obj in result.objects])
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()

    def on_stop(self):
        with ExitStack() as cleanup:
            for endpoint in (self.camera, self.prompt_subscriber, self.publisher,
                             self.refine_subscriber, self.refine_publisher, self.predictor):
                if endpoint is not None:
                    cleanup.callback(endpoint.close)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=0, help="Published results; 0 runs continuously")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0 or not math.isfinite(args.confidence_threshold) or not 0 <= args.confidence_threshold <= 1:
        parser.error("--frames must be nonnegative and --confidence-threshold must be in [0, 1]")
    try:
        Sam3Worker(WorkerContext("sam3"), args.checkpoint, args.service, args.device,
                   args.confidence_threshold, args.frames).run()
    except Exception as error:
        log_record("error", worker="sam3", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
