"""Native independent selection-worker FK stream with fake measured robot states."""
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.grasp_selection import (
    SelectionRequest, SelectionRequestPublisher, SelectionResultSubscriber, services,
)
from test_grasp_fk import reference

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import sys
import numpy as np
for name in ('torch','curobo','object_pose_system','ffw_system','pyzed','pyrealsense2'):
    sys.modules[name]=None
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.workers.worker_grasp_selection import GraspSelectionWorker
class Predictor:
    def __init__(self,**kwargs):self.sequence=0
    def start(self):pass
    def close(self):pass
    def robot_snapshot(self):
        self.sequence+=1
        hands={}
        for arm,x in [('left',1),('right',2)]:
            matrix=np.eye(4);matrix[0,3]=x;matrix[1,3]=self.sequence*.01
            hands[arm]=matrix
        return {'state_sequence':self.sequence,'state_timestamp_ns':self.sequence*100,
                'current_hands':hands}
GraspSelectionWorker(WorkerContext('fk-native'),service=sys.argv[1],predictor_factory=Predictor).run()
"""


def test_native_attachment_stream_tracks_measured_state_changes_without_reselecting(tmp_path):
    pytest.importorskip("iceoryx2")
    service = "fk-native/" + uuid4().hex
    request_service, result_service = services(service)
    publisher, subscriber = SelectionRequestPublisher(request_service), SelectionResultSubscriber(result_service)
    process, log = None, tmp_path / "worker.log"
    try:
        with log.open("w") as stream:
            process = subprocess.Popen([sys.executable, "-B", "-u", "-c", SCRIPT, service],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            deadline, sent, sequences = time.monotonic() + 10, False, []
            while time.monotonic() < deadline:
                publisher.update_connections()
                if not sent and '"event": "ready"' in log.read_text():
                    publisher.publish(SelectionRequest("attach-native", {}, (), "attach", reference()))
                    sent = True
                value = subscriber.receive_latest()
                if value is not None and value.status == "done":
                    assert value.request_id == "attach-native"
                    item = value.selection["objects"]["obs-A"]
                    sequence = item["observed_sequence"]
                    sequences.append(sequence)
                    expected = np.eye(4)
                    expected[1, 3] = sequence * .01
                    np.testing.assert_allclose(item["base_from_object"], expected)
                    assert not value.selection["robot_command_published"]
                    if len(sequences) >= 3:
                        break
                assert process.poll() is None, log.read_text()
                time.sleep(.005)
            assert len(sequences) >= 3 and sequences == sorted(set(sequences))
            publisher.publish(SelectionRequest("detach-native", {}, (), "detach"))
            deadline, detached = time.monotonic() + 5, False
            while time.monotonic() < deadline:
                publisher.update_connections()
                value = subscriber.receive_latest()
                if value is not None and value.request_id == "detach-native" and value.status == "done":
                    assert value.selection["state"] == "detached"
                    detached = True
                    break
                time.sleep(.005)
            assert detached, log.read_text()
            time.sleep(.05)
            assert subscriber.receive_latest() is None
            process.terminate()
            assert process.wait(timeout=5) == 0, log.read_text()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        subscriber.close()
        publisher.close()
    records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    assert [item["request_id"] for item in records if item["event"] == "grasp-selection-result"] == [
        "attach-native", "detach-native"]
