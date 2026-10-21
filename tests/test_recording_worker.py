"""Independent recorder lifecycle uses native-message contracts, not camera SDKs."""
import json
import subprocess
import sys

import pytest

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.foundationpose import PoseResult, services as pose_services
from grasp_workbench.ipc.recording import RecordingControl, services
from grasp_workbench.recording import LeRobotRecorder
from grasp_workbench.workers import worker_data_recorder
from grasp_workbench.workers.worker_data_recorder import DataRecorderWorker, build_parser
from test_recording import Dataset, result


class Mailbox:
    def __init__(self):
        self.value, self.closed = None, False

    def receive_latest(self):
        value, self.value = self.value, None
        return value

    def close(self):
        self.closed = True


class Publisher:
    def __init__(self):
        self.statuses, self.closed, self.refreshes = [], False, 0

    def update_connections(self):
        self.refreshes += 1

    def publish(self, value):
        self.statuses.append(value)

    def close(self):
        self.closed = True


def worker(tmp_path, **options):
    source, control, output, data = Mailbox(), Mailbox(), Publisher(), Dataset()
    calls = []

    def writer_factory(path, fps, run_metadata):
        calls.append((path, fps, run_metadata))
        return LeRobotRecorder(path, fps, run_metadata, dataset_factory=lambda **kwargs: data)

    instance = DataRecorderWorker(
        WorkerContext("recorder-test"), tmp_path, service="test/camera", fps=30,
        source_subscriber_factory=lambda service: calls.append(service) or source,
        control_subscriber_factory=lambda service: calls.append(service) or control,
        status_publisher_factory=lambda service: calls.append(service) or output,
        recorder_factory=writer_factory, **options)
    instance.on_start()
    return instance, source, control, output, data, calls


def test_pose_source_start_save_discard_and_status_request_ids(tmp_path):
    instance, source, control, output, data, calls = worker(tmp_path)
    try:
        assert calls[:3] == [pose_services("test/camera")[1], *services("test/camera")]
        assert calls[-1] == (instance.output_dir, 30, {"source": "pose"})
        assert instance.output_dir.parent == tmp_path
        assert instance.hz == 100
        assert output.statuses[-1].state == "ready"
        source.value = result()
        instance.step_once()
        assert instance.processed_count == 0 and data.frames == []
        control.value, source.value = RecordingControl("start-1", "start"), result()
        instance.step_once()
        assert output.statuses[-1].request_id == "start-1"
        assert output.statuses[-1].state == "recording"
        assert output.statuses[-1].episode_frames == output.statuses[-1].total_frames == 1
        stored, = data.frames
        assert json.loads(stored["observation.objects"])[1]["status"] == "unavailable"
        source.value = result(4)
        instance.step_once()
        assert output.statuses[-1].episode_frames == 2
        control.value, source.value = RecordingControl("save-1", "save"), result(5)
        instance.step_once()
        status = output.statuses[-1]
        assert status.state == "saved" and status.request_id == "save-1"
        assert status.episode_index == 1 and status.episode_frames == 0 and status.total_frames == 2
        assert len(data.episodes) == 1 and len(data.episodes[0]) == 2
        control.value, source.value = RecordingControl("start-2", "start"), result(6)
        instance.step_once()
        control.value = RecordingControl("discard-2", "discard")
        instance.step_once()
        status = output.statuses[-1]
        assert status.state == "discarded" and status.request_id == "discard-2"
        assert status.episode_index == 1 and status.total_frames == 2
        assert not data.frames and data.discards == 1
        assert output.refreshes == 6
    finally:
        instance.on_stop()
    assert all(item.closed for item in (source, control, output))
    assert data.finalizes == 1


def test_camera_source_wraps_exact_frame_without_constructing_poses(tmp_path):
    instance, source, control, output, data, calls = worker(tmp_path, source="camera")
    try:
        assert calls[0] == "test/camera"
        frame = result().frame
        source.value, control.value = frame, RecordingControl("capture", "start")
        instance.step_once()
        stored, = data.frames
        assert json.loads(stored["observation.camera_metadata"]) == frame.metadata
        assert json.loads(stored["observation.objects"]) == []
        assert not stored["observation.pose.A_valid"].item()
        assert not stored["observation.pose.B_valid"].item()
    finally:
        instance.on_stop()
    assert len(data.episodes) == 1 and output.statuses[-1].state == "saved"


def test_finite_auto_start_smoke_saves_active_episode_on_stop(tmp_path):
    instance, source, _, output, data, _ = worker(tmp_path, frames=2, auto_start=True,
                                               run_metadata={"aprilcube": {"fast": True}})
    try:
        assert output.statuses[-1].state == "recording"
        assert output.statuses[-1].request_id == "auto-start"
        assert instance.recorder.run_metadata == {"source": "pose", "aprilcube": {"fast": True}}
        for sequence in (1, 2):
            source.value = result(sequence)
            instance.step_once()
        assert instance.processed_count == 2 and instance.ctx.stop_event.is_set()
    finally:
        instance.on_stop()
    assert len(data.episodes) == 1 and len(data.episodes[0]) == 2
    assert data.finalizes == 1 and output.statuses[-1].state == "saved"
    assert output.statuses[-1].total_frames == 2


def test_source_pose_reference_is_not_replaced_by_new_camera_frame(tmp_path):
    instance, source, control, _, _, _ = worker(tmp_path)
    seen = []
    exact = result()
    instance.recorder.add_frame = seen.append
    try:
        source.value, control.value = exact, RecordingControl("start", "start")
        instance.step_once()
        assert seen == [exact] and isinstance(seen[0], PoseResult)
        assert seen[0].frame is exact.frame
    finally:
        instance.on_stop()


def test_command_errors_report_status_and_write_errors_propagate(tmp_path):
    instance, source, control, output, data, _ = worker(tmp_path)
    try:
        control.value = RecordingControl("bad-save", "save")
        instance.step_once()
        assert output.statuses[-1].state == "error"
        assert output.statuses[-1].request_id == "bad-save"
        assert "active" in output.statuses[-1].error
        control.value, source.value = RecordingControl("start", "start"), result()
        instance.step_once()

        def fail(frame):
            raise OSError("disk full")

        data.add_frame = fail
        source.value = result(4)
        with pytest.raises(OSError, match="disk full"):
            instance.step_once()
        assert output.statuses[-1].state == "error" and "disk full" in output.statuses[-1].error
        assert instance.processed_count == 1
    finally:
        instance.on_stop()


def test_stop_closes_native_endpoints_if_dataset_finalize_fails(tmp_path):
    instance, source, control, output, data, _ = worker(tmp_path, auto_start=True)
    source.value = result()
    instance.step_once()

    def fail():
        raise OSError("close failed")

    data.finalize = fail
    with pytest.raises(OSError, match="close failed"):
        instance.on_stop()
    assert all(item.closed for item in (source, control, output))


def test_cli_help_does_not_need_lerobot_or_camera_sdk():
    completed = subprocess.run([sys.executable, "-m", "grasp_workbench.workers.worker_data_recorder",
                                "--help"], capture_output=True, text=True)
    assert completed.returncode == 0
    assert "--output-dir" in completed.stdout and "--source" in completed.stdout
    assert "--auto-start" in completed.stdout and "--run-metadata-json" in completed.stdout
    args = build_parser().parse_args(["--output-dir", "runs/datasets"])
    assert args.source == "pose" and args.fps == 30 and args.frames == 0 and not args.auto_start


def test_cli_passes_selected_source_settings_and_metadata(monkeypatch):
    calls = []

    class StubWorker:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def run(self):
            calls.append("run")

    monkeypatch.setattr(worker_data_recorder, "DataRecorderWorker", StubWorker)
    assert worker_data_recorder.main([
        "--output-dir", "runs/datasets", "--service", "demo/camera", "--source", "camera",
        "--fps", "15", "--frames", "5", "--auto-start",
        "--run-metadata-json", '{"camera":{"fps":15}}',
    ]) == 0
    assert calls[0][0][1:] == ("runs/datasets", "demo/camera", 15, "camera", 5, True)
    assert calls[0][1] == {"run_metadata": {"camera": {"fps": 15}}}
    assert calls[1] == "run"


@pytest.mark.parametrize("flags", [["--fps", "0"], ["--frames", "-1"],
                                   ["--run-metadata-json", "[]"],
                                   ["--run-metadata-json", "not-json"]])
def test_invalid_cli_arguments(flags):
    with pytest.raises(SystemExit) as error:
        worker_data_recorder.main(["--output-dir", "runs/datasets", *flags])
    assert error.value.code == 2


def test_cli_worker_error_returns_failure(monkeypatch, capsys):
    class FailingWorker:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise OSError("write failed")

    monkeypatch.setattr(worker_data_recorder, "DataRecorderWorker", FailingWorker)
    assert worker_data_recorder.main(["--output-dir", "runs/datasets"]) == 1
    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "error" and "write failed" in record["error"]
