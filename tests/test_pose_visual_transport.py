"""Independent FP worker, native RGB-D/SAM3/pose IPC and a model-free backend."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame, CameraPublisher, encode_frame
from grasp_workbench.ipc.foundationpose import (
    PoseControl, PoseControlPublisher, PoseResultSubscriber, PoseTarget, services,
)
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Publisher, Sam3Result, services as sam3_services

ROOT = Path(__file__).resolve().parents[1]
_WORKER = """
import hashlib, json, sys
for name in ('torch', 'estimater', 'pyzed', 'pyrealsense2', 'object_pose_system'):
    sys.modules[name] = None
import numpy as np
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.camera import encode_frame
from grasp_workbench.workers.worker_foundationpose import FoundationPoseWorker

class Predictor:
    def __init__(self):
        self.poses = {}
        self.starts = self.closes = 0
    def start(self):
        self.starts += 1
    def register(self, object_id, frame, mask, mesh_path, scale):
        assert mesh_path == object_id + '.glb' and scale == .001
        rows, columns = np.nonzero(mask)
        intrinsics = frame.metadata['intrinsics']
        z = float(np.median(frame.depth_m[mask]))
        pose = np.eye(4)
        pose[:3,3] = [(columns.mean()-intrinsics['cx'])*z/intrinsics['fx'],
                      (rows.mean()-intrinsics['cy'])*z/intrinsics['fy'], z]
        self.poses[object_id] = pose
        print(json.dumps({'event':'register','object_id':object_id,
            'sequence':frame.metadata['sequence'],
            'frame_sha256':hashlib.sha256(encode_frame(frame)).hexdigest(),
            'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()}),flush=True)
        return pose.copy()
    def track(self, object_id, frame):
        print(json.dumps({'event':'track','object_id':object_id,
            'sequence':frame.metadata['sequence']}),flush=True)
        if object_id == 'A' and frame.metadata.get('fail_a'):
            raise RuntimeError('injected A track failure')
        pose = self.poses[object_id].copy()
        if object_id == 'A' and frame.metadata.get('drift_a'):
            pose[0,3] += .08  # Below the previous 0.25 m translation jump limit.
        return pose
    def remove(self, object_id):
        self.poses.pop(object_id,None)
        print(json.dumps({'event':'remove','object_id':object_id}),flush=True)
    def close(self):
        self.closes += 1

predictor = Predictor()
worker = FoundationPoseWorker(WorkerContext('visual-transport'),'unused-source',
    service=sys.argv[1],predictor_factory=lambda *a,**k:predictor)
try:
    worker.run()
finally:
    assert all(sys.modules[name] is None for name in
        ('torch','estimater','pyzed','pyrealsense2','object_pose_system'))
    print(json.dumps({'event':'closed','starts':predictor.starts,'closes':predictor.closes}),flush=True)
"""


def _frame(sequence, depth=1., **metadata):
    return CameraFrame(np.full((80, 240, 3), sequence % 256, np.uint8),
                       np.full((80, 240), depth, np.float32),
                       {"camera_id": "test-camera", "stream_id": "test-stream", "sequence": sequence,
                        "intrinsics": {"fx": 1000., "fy": 1000., "cx": 120., "cy": 40.}, **metadata})


def _mask(box):
    x1, y1, x2, y2 = box
    mask = np.zeros((80, 240), bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _observation(object_id, boxes, scores):
    return Sam3ObjectResult(object_id, "red cup" if object_id == "A" else "blue bowl",
                            np.stack([_mask(box) for box in boxes]) if boxes else np.empty((0, 80, 240), bool),
                            np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32))


def _wait_for(predicate, message, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    pytest.fail(message)


def test_native_seed_association_visual_recovery_and_tracking_exception(tmp_path):
    pytest.importorskip("iceoryx2")
    service = f"pose-visual-transport/{uuid4().hex}"
    controls = PoseControlPublisher(services(service)[0])
    results = PoseResultSubscriber(services(service)[1])
    camera = CameraPublisher(service)
    sam3 = Sam3Publisher(sam3_services(service)[1])
    a_box, b_box = (20, 30, 40, 50), (160, 30, 180, 50)
    source = _frame(17)
    log_path, error_path = tmp_path / "worker.log", tmp_path / "worker.err"
    child = None

    def records(event):
        return [record for line in log_path.read_text().splitlines()
                if (record := json.loads(line)).get("event") == event]

    def receive(sequence, object_ids, status=None):
        def poll():
            assert child.poll() is None, log_path.read_text() + error_path.read_text()
            controls.update_connections()
            result = results.receive_latest()
            if result is None or result.frame.metadata["sequence"] != sequence:
                return None
            if [obj.object_id for obj in result.objects] != object_ids:
                return None
            if status is not None and result.objects[0].status != status:
                return None
            return result

        return _wait_for(poll, f"No pose result for frame {sequence}, slots {object_ids}")

    def track(sequence, **metadata):
        camera.publish(_frame(sequence, **metadata))
        return receive(sequence, ["A", "B"])

    def observe(sequence, object_id, boxes, scores, *, revision="selected", depth=1.):
        frame = _frame(sequence, depth)
        sam3.publish(Sam3Result(frame, revision, (_observation(object_id, boxes, scores),)))
        return frame

    try:
        with log_path.open("w") as log, error_path.open("w") as error:
            child = subprocess.Popen([sys.executable, "-B", "-u", "-c", _WORKER, service],
                                     cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
                                     stdout=log, stderr=error)
            _wait_for(lambda: records("ready"), "FP worker did not become ready")
            controls.publish(PoseControl(tuple(
                PoseTarget(key, "session-" + key, key + ".glb", .001, source, _mask(box), "selected")
                for key, box in (("A", a_box), ("B", b_box)))))
            receive(17, ["B"], "tracking")
            assert all(obj.status == "tracking" for obj in track(18).objects)

            # The selected instance wins by bbox overlap, not score or current index.
            for sequence, key, boxes, scores in (
                (31, "A", [b_box, a_box], [.99, .5]),
                (32, "A", [a_box, b_box], [.4, .99]),
                (33, "B", [a_box, b_box], [.99, .5]),
                (34, "B", [b_box, a_box], [.4, .99]),
            ):
                observe(sequence, key, boxes, scores)
                current = receive(sequence, [key]).objects[0]
                assert current.status == "tracking" and current.visual_mismatch_count == 0
            sam3.publish(Sam3Result(_frame(35), "selected", tuple(
                _observation(key, [], []) for key in ("A", "B"))))
            assert all(obj.status == "tracking" for obj in track(36).objects)
            assert len(records("register")) == 2

            drifted = track(40, drift_a=True)
            for count in range(1, 5):
                observe(40 + count, "A", [b_box, a_box], [.99, .5])
                current = receive(40 + count, ["A"]).objects[0]
                assert current.status == "tracking" and current.visual_mismatch_count == count
                assert current.error == "visual_mismatch:bbox"
                # B continues tracking through A's repeated failed visual checks.
                paired = track(50 + count, drift_a=True)
                assert paired.objects[1].status == "tracking"
                assert paired.objects[0].visual_mismatch_count == count
            assert len(records("register")) == 2
            recovered_source = observe(99, "A", [b_box, a_box], [.99, .5], depth=1.02)
            recovered = receive(99, ["A"], "tracking")
            assert recovered.objects[0].visual_mismatch_count == 0
            assert recovered.objects[0].error is None
            assert not np.array_equal(recovered.objects[0].camera_from_object, drifted.objects[0].camera_from_object)
            np.testing.assert_array_equal(recovered.frame.rgb, recovered_source.rgb)
            np.testing.assert_array_equal(recovered.frame.depth_m, recovered_source.depth_m)
            assert all(obj.status == "tracking" for obj in track(100).objects)

            failed = track(150, fail_a=True)
            assert failed.objects[0].status == "recovering"
            assert failed.objects[0].camera_from_object is None
            assert failed.objects[0].error == "RuntimeError: injected A track failure"
            assert failed.objects[1].status == "tracking"
            waiting = track(151)
            assert waiting.objects[0].status == "recovering" and waiting.objects[0].camera_from_object is None
            assert len(records("register")) == 3
            observe(152, "A", [a_box], [.9], revision="previous-prompt")
            time.sleep(.1)
            ignored = track(153)
            assert ignored.objects[0].status == "recovering" and ignored.objects[0].camera_from_object is None
            assert ignored.objects[1].status == "tracking" and len(records("register")) == 3
            fresh_box = (25, 30, 45, 50)
            fresh_source = observe(154, "A", [fresh_box], [.9], depth=.95)
            fresh = receive(154, ["A"], "tracking")
            assert fresh.objects[0].error is None and fresh.objects[0].visual_mismatch_count == 0
            assert all(obj.status == "tracking" for obj in track(155).objects)

            registered = records("register")
            expected = (("A", source, a_box), ("B", source, b_box),
                        ("A", recovered_source, a_box), ("A", fresh_source, fresh_box))
            assert len(registered) == len(expected)
            for event, (key, frame, box) in zip(registered, expected):
                assert event == {"event": "register", "object_id": key, "sequence": frame.metadata["sequence"],
                                 "frame_sha256": hashlib.sha256(encode_frame(frame)).hexdigest(),
                                 "mask_sha256": hashlib.sha256(_mask(box).tobytes()).hexdigest()}
            tracked = [(event["object_id"], event["sequence"]) for event in records("track")]
            assert ("A", 150) in tracked and ("A", 151) not in tracked and ("A", 153) not in tracked
            assert all(("B", sequence) in tracked for sequence in (150, 151, 153, 155))
            controls.publish(PoseControl(()))
            _wait_for(lambda: len(records("remove")) == 2, "A/B sessions did not stop")
    finally:
        if child is not None:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                pytest.fail("FP visual transport worker did not stop")
        sam3.close()
        results.close()
        controls.close()
        camera.close()
    assert child.returncode == 0, log_path.read_text() + error_path.read_text()
    assert records("closed") == [{"event": "closed", "starts": 1, "closes": 1}]
