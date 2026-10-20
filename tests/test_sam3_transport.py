"""Real iceoryx2 controls/results across a separate, model-free SAM3 worker."""
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher
from grasp_workbench.ipc.sam3 import (
    ObjectPrompt, PromptPublisher, PromptSet, Sam3Subscriber, services,
)


WORKER = """
import sys
sys.modules['torch'] = None
sys.modules['sam3'] = None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result
from grasp_workbench.workers.worker_sam3 import Sam3Worker
class Predictor:
    def start(self):
        print('FAKE_MODEL_START', flush=True)
    def predict(self, frame, prompts):
        h, w = frame.rgb.shape[:2]
        objects = []
        for i, prompt in enumerate(prompts.objects):
            mask = np.zeros((1, h, w), dtype=bool)
            mask[:, :, i::2] = True
            objects.append(Sam3ObjectResult(prompt.object_id, prompt.text, mask,
                np.array([[i, 0, w, h]], np.float32), np.array([.9], np.float32)))
        return Sam3Result(frame, prompts.revision, tuple(objects))
    def close(self):
        print('FAKE_MODEL_CLOSE', flush=True)
worker = Sam3Worker(WorkerContext('sam3-test'), 'unused.pt', service=sys.argv[1], frames=3,
                   predictor_factory=lambda *args, **kwargs: Predictor())
worker.run()
assert worker.processed_count == 3
"""


def test_two_prompts_continuous_frames_stop_and_revision_across_processes():
    service = "grasp_workbench/test/sam3_worker_" + uuid.uuid4().hex
    prompt_service, result_service = services(service)
    control, camera, subscriber = PromptPublisher(prompt_service), CameraPublisher(service), Sam3Subscriber(result_service)
    child = subprocess.Popen([sys.executable, "-B", "-u", "-c", WORKER, service],
                             cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)

    def publish_frame(sequence):
        camera.publish(CameraFrame(np.full((3, 4, 3), sequence * 20, np.uint8),
                                   np.ones((3, 4), np.float32),
                                   {"camera_id": "test", "stream_id": "stream", "sequence": sequence}))

    def wait_result(revision, sequence=None):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            control.update_connections()
            if sequence is not None:
                publish_frame(sequence)
            value = subscriber.receive_latest()
            if value is not None and value.revision == revision:
                if sequence is None or value.frame.metadata["sequence"] == sequence:
                    return value
            assert child.poll() is None, "Worker exited before delivering expected result"
            time.sleep(.01)
        raise AssertionError("No expected SAM3 result")

    try:
        control.publish(PromptSet("first", (ObjectPrompt("A", "red cup"), ObjectPrompt("B", "blue bowl"))))
        publish_frame(0)
        first = wait_result("first", 0)
        assert [(obj.object_id, obj.text) for obj in first.objects] == [("A", "red cup"), ("B", "blue bowl")]
        assert not np.array_equal(first.objects[0].masks, first.objects[1].masks)
        publish_frame(1)
        second = wait_result("first", 1)
        assert np.all(second.frame.rgb == 20)
        control.publish(PromptSet("stopped", ()))
        publish_frame(2)
        time.sleep(.2)
        assert subscriber.receive_latest() is None
        assert child.poll() is None
        control.publish(PromptSet("changed", (ObjectPrompt("A", "green bottle"),)))
        changed = wait_result("changed")
        assert [(obj.object_id, obj.text) for obj in changed.objects] == [("A", "green bottle")]
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr
        assert stdout.count("FAKE_MODEL_START") == stdout.count("FAKE_MODEL_CLOSE") == 1
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
        subscriber.close()
        camera.close()
        control.close()
