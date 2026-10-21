"""Actual LeRobot storage behind an independent worker and native iceoryx2."""
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.foundationpose import PoseResultPublisher, services as pose_services
from grasp_workbench.ipc.recording import RecordingStatusSubscriber, services
from test_recording import result


def test_independent_worker_records_native_pose_frames_and_finalizes(tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    service = "recording-process-test/" + uuid4().hex
    publisher = PoseResultPublisher(pose_services(service)[1])
    status_subscriber = RecordingStatusSubscriber(services(service)[1])
    process = None
    latest = None
    sources = [result(sequence) for sequence in (41, 43, 48)]
    sources[0].objects[0].diagnostics = {"n_tags": 2, "predicted": False}
    try:
        with (tmp_path / "worker.log").open("w+") as output:
            process = subprocess.Popen([
                sys.executable, "-u", "-m", "grasp_workbench.workers.worker_data_recorder",
                "--service", service, "--output-dir", str(tmp_path / "datasets"),
                "--source", "pose", "--frames", "3", "--auto-start",
            ], cwd=Path(__file__).resolve().parents[1], stdout=output, stderr=subprocess.STDOUT)

            def receive_until(predicate):
                nonlocal latest
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    publisher.update_connections()
                    status = status_subscriber.receive_latest()
                    if status is not None:
                        latest = status
                    if latest is not None and predicate(latest):
                        return
                    if process.poll() is not None:
                        output.seek(0)
                        pytest.fail(output.read())
                    time.sleep(.005)
                output.seek(0)
                pytest.fail("Recorder status timeout: " + output.read())

            receive_until(lambda status: status.state == "recording")
            for index, source in enumerate(sources, start=1):
                publisher.publish(source)
                receive_until(lambda status: status.total_frames >= index)
            assert process.wait(timeout=45) == 0
        dataset = LeRobotDataset("local/grasp_workbench", root=latest.output_dir,
                                depth_output_unit="m", video_backend="pyav")
        assert len(dataset) == 3 and dataset.num_episodes == 1
        for index, source in enumerate(sources):
            row = dataset.hf_dataset.with_format(None)[index]
            np.testing.assert_array_equal(np.asarray(row["observation.images.rgb"]), source.frame.rgb)
            np.testing.assert_array_equal(np.asarray(row["observation.depth_m"]).view(np.uint32),
                                          source.frame.depth_m.view(np.uint32))
            assert json.loads(row["observation.camera_metadata"]) == source.frame.metadata
            objects = json.loads(row["observation.objects"])
            assert objects[0]["diagnostics"] == source.objects[0].diagnostics
            assert objects[1]["status"] == "unavailable"
            assert not row["observation.pose.B_valid"]
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=45)
        publisher.close()
        status_subscriber.close()
