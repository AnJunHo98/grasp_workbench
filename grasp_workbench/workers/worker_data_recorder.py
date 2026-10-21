"""Record exact received camera/pose frames into local LeRobot episodes."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from ..ipc.foundationpose import PoseResult, services as pose_services
from ..ipc.recording import RecordingControl, RecordingStatus, services
from ..recording import LeRobotRecorder
from .worker_camera import DEFAULT_SERVICE


class DataRecorderWorker(SingleRateWorker):
    def __init__(self, context, output_dir, service=DEFAULT_SERVICE, fps=30,
                 source="pose", frames=0, auto_start=False, *, run_metadata=None,
                 source_subscriber_factory=None, control_subscriber_factory=None,
                 status_publisher_factory=None, recorder_factory=None):
        super().__init__(context, hz=100)
        if source not in {"camera", "pose"}:
            raise ValueError("Recording source must be camera or pose")
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
        self.output_dir = Path(output_dir) / run
        self.service, self.fps, self.source = service, fps, source
        self.frames, self.auto_start = frames, auto_start
        self.run_metadata = {**(run_metadata or {}), "source": source}
        self.source_subscriber_factory = source_subscriber_factory
        self.control_subscriber_factory = control_subscriber_factory
        self.status_publisher_factory = status_publisher_factory
        self.recorder_factory = recorder_factory
        self.subscriber = self.control_subscriber = self.publisher = self.recorder = None
        self.request_id = ""
        self.processed_count = 0

    def _publish(self, state, error=None):
        self.publisher.publish(RecordingStatus(
            request_id=self.request_id, state=state, output_dir=str(self.output_dir),
            episode_index=self.recorder.episode_index,
            episode_frames=self.recorder.episode_frames,
            total_frames=self.recorder.total_frames, error=error,
        ))

    def on_start(self):
        from ..ipc.camera import CameraSubscriber
        from ..ipc.foundationpose import PoseResultSubscriber
        from ..ipc.recording import RecordingControlSubscriber, RecordingStatusPublisher

        factory = (self.source_subscriber_factory
                   or (PoseResultSubscriber if self.source == "pose" else CameraSubscriber))
        source_service = pose_services(self.service)[1] if self.source == "pose" else self.service
        self.subscriber = factory(source_service)
        control_service, status_service = services(self.service)
        self.control_subscriber = (self.control_subscriber_factory or RecordingControlSubscriber)(control_service)
        self.publisher = (self.status_publisher_factory or RecordingStatusPublisher)(status_service)
        self.recorder = (self.recorder_factory or LeRobotRecorder)(
            self.output_dir, self.fps, run_metadata=self.run_metadata)
        self._publish("ready")
        log_record("ready", worker=self.ctx.name, source=self.source,
                   output_dir=str(self.output_dir))
        if self.auto_start:
            self._command(RecordingControl("auto-start", "start"))

    def _command(self, command):
        self.request_id = command.request_id
        try:
            if command.action == "start":
                self.recorder.start()
                state = "recording"
            elif command.action == "save":
                self.recorder.save()
                state = "saved"
            elif command.action == "discard":
                self.recorder.discard()
                state = "discarded"
            else:
                raise ValueError("Unknown recording action")
            self._publish(state)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self._publish("error", message)
            log_record("error", worker=self.ctx.name, error=message)

    def step_once(self):
        self.publisher.update_connections()
        command = self.control_subscriber.receive_latest()
        if command is not None:
            self._command(command)
        result = self.subscriber.receive_latest()
        if result is None or not self.recorder.recording:
            return
        if self.source == "camera":
            result = PoseResult(result, ())
        try:
            self.recorder.add_frame(result)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self._publish("error", message)
            raise
        self.processed_count += 1
        self._publish("recording")
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()

    def on_stop(self):
        try:
            if self.recorder is not None:
                saved = self.recorder.recording and self.recorder.episode_frames > 0
                self.recorder.close()
                if saved and self.publisher is not None:
                    self._publish("saved")
                log_record("stopped", worker=self.ctx.name,
                           output_dir=str(self.output_dir),
                           episodes=self.recorder.episode_index,
                           total_frames=self.recorder.total_frames)
        finally:
            for endpoint in (self.publisher, self.control_subscriber, self.subscriber):
                if endpoint is not None:
                    endpoint.close()


def _metadata(value):
    try:
        record = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(record, dict):
        raise argparse.ArgumentTypeError("Recording metadata must be a JSON object")
    return record


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--source", choices=("camera", "pose"), default="pose")
    parser.add_argument("--run-metadata-json", type=_metadata, default=None,
                        help="Public camera/AprilCube settings only; never the complete config")
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop after this many recorded frames (0: unlimited)")
    parser.add_argument("--auto-start", action="store_true",
                        help="Start an episode immediately, for a finite collection check")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.frames < 0:
        parser.error("--fps must be positive and --frames must be nonnegative")
    try:
        DataRecorderWorker(WorkerContext("data_recorder"), args.output_dir, args.service,
                           args.fps, args.source, args.frames, args.auto_start,
                           run_metadata=args.run_metadata_json).run()
    except Exception as error:
        log_record("error", worker="data_recorder", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
