"""Select single/dual grasps and optionally check robot-frame endpoint reachability."""
import argparse
import json

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from ..ipc.grasp_selection import SelectionResult, services
from ..models.grasp_fk import GraspFkTracker
from .worker_camera import DEFAULT_SERVICE


class GraspSelectionWorker(SingleRateWorker):
    def __init__(self, context, *, robot_config=None, settings=None, service=DEFAULT_SERVICE,
                 frames=0, subscriber_factory=None, publisher_factory=None, predictor_factory=None):
        super().__init__(context, hz=100)
        self.robot_config, self.settings = robot_config, settings or {}
        self.service, self.frames = service, frames
        self.subscriber_factory, self.publisher_factory = subscriber_factory, publisher_factory
        self.predictor_factory = predictor_factory
        self.subscriber = self.publisher = self.predictor = None
        self.processed_count, self._last_result = 0, None
        self._grasp_fk = GraspFkTracker()
        self._attachment_id, self._attachment_sequence = None, None

    def on_start(self):
        from ..ipc.grasp_selection import SelectionRequestSubscriber, SelectionResultPublisher

        request_service, result_service = services(self.service)
        self.subscriber = (self.subscriber_factory or SelectionRequestSubscriber)(request_service)
        self.publisher = (self.publisher_factory or SelectionResultPublisher)(result_service)
        log_record("ready", worker=self.ctx.name, service=self.service)

    def _start_predictor(self):
        if self.predictor is not None:
            return
        if self.predictor_factory is None:
            from ..models.grasp_selection import GraspSelectionPredictor

            self.predictor_factory = GraspSelectionPredictor
        predictor = self.predictor_factory(robot_config=self.robot_config, settings=self.settings)
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
            self._poll_attachment()
            return
        if self._last_result is not None and request.request_id == self._last_result.request_id:
            self.publisher.publish(self._last_result)
            self._poll_attachment()
            return
        self.publisher.publish(SelectionResult(request.request_id, "running"))
        try:
            if request.action == "detach":
                self._attachment_id = None
                selection = self._grasp_fk.detach()
            else:
                self._start_predictor()
                if request.action == "attach":
                    self._grasp_fk.attach(request.attachment)
                    self._attachment_id = request.request_id
                    selection = self._estimate_attachment()
                else:
                    selection = self.predictor.predict(request)
            result = SelectionResult(request.request_id, "done", selection)
        except Exception as error:
            result = SelectionResult(request.request_id, "error", error=f"{type(error).__name__}: {error}")
        self._last_result = result
        self.publisher.publish(result)
        self.processed_count += 1
        log_record("grasp-selection-result", request_id=result.request_id, status=result.status, error=result.error)
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()
        if request.action == "select":
            self._poll_attachment()

    def _estimate_attachment(self):
        snapshot = self.predictor.robot_snapshot()
        self._attachment_sequence = None if snapshot is None else snapshot.get("state_sequence")
        return self._grasp_fk.estimate(snapshot)

    def _poll_attachment(self):
        if self._attachment_id is None:
            return
        snapshot = self.predictor.robot_snapshot()
        sequence = None if snapshot is None else snapshot.get("state_sequence")
        if sequence == self._attachment_sequence:
            return
        self._attachment_sequence = sequence
        self.publisher.publish(SelectionResult(self._attachment_id, "done", self._grasp_fk.estimate(snapshot)))

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
    parser.add_argument("--robot-config", help="Optional local robot state/calibration/endpoint IK YAML")
    parser.add_argument("--settings", type=json.loads, default={})
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--frames", type=int, default=0, help="Terminal requests; 0 waits continuously")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0 or not isinstance(args.settings, dict):
        parser.error("--frames must be nonnegative and --settings must be a JSON object")
    try:
        GraspSelectionWorker(WorkerContext("grasp_selection"), robot_config=args.robot_config,
                             settings=args.settings, service=args.service, frames=args.frames).run()
    except Exception as error:
        log_record("error", worker="grasp_selection", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
