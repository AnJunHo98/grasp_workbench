"""On-demand semantic recognition through the selected API or Codex CLI backend."""
import argparse

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from ..ipc.vlm import VlmResult, services
from ..models.vlm import DEFAULT_MODEL, VlmError
from .worker_camera import DEFAULT_SERVICE


class VlmWorker(SingleRateWorker):
    def __init__(self, context, model=DEFAULT_MODEL, api_key_file=None, service=DEFAULT_SERVICE,
                 frames=0, *, backend="openai_api", codex_executable="codex",
                 subscriber_factory=None, publisher_factory=None, predictor_factory=None):
        super().__init__(context, hz=100)
        self.model, self.api_key_file, self.service, self.frames = model, api_key_file, service, frames
        self.backend, self.codex_executable = backend, codex_executable
        self.subscriber_factory, self.publisher_factory = subscriber_factory, publisher_factory
        self.predictor_factory = predictor_factory
        self.subscriber = self.publisher = self.predictor = None
        self.processed_count = 0
        self._last_result = None

    def on_start(self):
        from ..ipc.vlm import VlmRequestSubscriber, VlmResultPublisher

        request_service, result_service = services(self.service)
        self.subscriber = (self.subscriber_factory or VlmRequestSubscriber)(request_service)
        self.publisher = (self.publisher_factory or VlmResultPublisher)(result_service)
        log_record("ready", worker=self.ctx.name, service=self.service)

    def _start_predictor(self):
        if self.predictor is not None:
            return
        options = {"model": self.model}
        if self.backend == "codex_cli":
            options["executable"] = self.codex_executable
        elif self.backend in {"openai_api", "anthropic_api"}:
            options["api_key_file"] = self.api_key_file
        else:
            raise ValueError("Unknown VLM backend")
        if self.predictor_factory is None:
            if self.backend == "codex_cli":
                from ..models.vlm_codex import CodexCliPredictor

                self.predictor_factory = CodexCliPredictor
            elif self.backend == "openai_api":
                from ..models.vlm import OfficialVlmPredictor

                self.predictor_factory = OfficialVlmPredictor
            elif self.backend == "anthropic_api":
                from ..models.vlm_anthropic import AnthropicVlmPredictor

                self.predictor_factory = AnthropicVlmPredictor
        predictor = self.predictor_factory(**options)
        try:
            predictor.start()
        except Exception:
            predictor.close()
            raise
        self.predictor = predictor

    def step_once(self):
        self.publisher.update_connections()
        request = self.subscriber.receive_latest()
        if request is None:
            return
        if self._last_result is not None and request.request_id == self._last_result.request_id:
            self.publisher.publish(self._last_result)
            return
        self.publisher.publish(VlmResult(request.request_id, request.frame, "running", model=self.model))
        try:
            self._start_predictor()
            prediction = self.predictor.predict(request)
            if hasattr(prediction, "objects"):
                result = VlmResult(request.request_id, request.frame, "done", tuple(prediction.objects), model=self.model,
                                   task_binding=prediction.task_binding, policy_selection=prediction.policy_selection,
                                   coordination=prediction.coordination)
            else:
                result = VlmResult(request.request_id, request.frame, "done", tuple(prediction), model=self.model)
        except Exception as error:
            kind = str(error) if isinstance(error, VlmError) else type(error).__name__
            result = VlmResult(request.request_id, request.frame, "error", error=kind, model=self.model)
        self._last_result = result
        self.publisher.publish(result)
        self.processed_count += 1
        log_record("vlm-result", request_id=result.request_id, status=result.status, error=result.error, model=self.model)
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("openai_api", "codex_cli", "anthropic_api"), default="openai_api")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--api-key-file")
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--frames", type=int, default=0, help="Terminal requests; 0 waits continuously")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    try:
        VlmWorker(WorkerContext("vlm"), args.model, args.api_key_file, args.service, args.frames,
                  backend=args.backend, codex_executable=args.codex_executable).run()
    except Exception as error:
        kind = str(error) if isinstance(error, VlmError) else type(error).__name__
        log_record("error", worker="vlm", error=kind)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
