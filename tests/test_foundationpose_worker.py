"""Minimal explicit session registration and continuous A/B tracking, no GPU."""
from dataclasses import replace
import subprocess
import sys

import numpy as np

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.foundationpose import PoseControl, PoseTarget, services
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result, services as sam3_services
from grasp_workbench.workers.worker_foundationpose import FoundationPoseWorker, build_parser


def frame(sequence=1):
    return CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
                       np.ones((3, 4), dtype=np.float32),
                       {"camera_id": "camera", "sequence": sequence,
                        "intrinsics": {"fx": 100, "fy": 101, "cx": 2, "cy": 1}})


def target(object_id="A", session_id=None, sequence=1):
    return PoseTarget(object_id, session_id or "session-" + object_id,
                      object_id + ".glb", 0.001, frame(sequence), np.ones((3, 4), bool), "revision")


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
        self.results, self.closed, self.refreshes = [], False, 0

    def update_connections(self):
        self.refreshes += 1

    def publish(self, result):
        self.results.append(result)
        return 1

    def close(self):
        self.closed = True


class Predictor:
    def __init__(self):
        self.starts, self.registers, self.tracks, self.removed = 0, [], [], []
        self.closed, self.registration_error, self.tracking_error = False, None, None
        self.registration_pose = np.eye(4)
        self.registration_pose[2, 3] = 1

    def start(self):
        self.starts += 1

    def register(self, object_id, source, mask, mesh_path, mesh_scale_to_m):
        self.registers.append((object_id, source, mask, mesh_path, mesh_scale_to_m))
        if self.registration_error == object_id:
            raise RuntimeError("registration failed")
        return self.registration_pose.copy()

    def track(self, object_id, source):
        self.tracks.append((object_id, source))
        if self.tracking_error == object_id:
            raise RuntimeError("tracking failed")
        pose = np.eye(4)
        pose[0, 3] = source.metadata["sequence"] * .001
        pose[2, 3] = 1
        return pose

    def remove(self, object_id):
        self.removed.append(object_id)

    def close(self):
        self.closed = True


def worker(frames=0, predictor=None):
    camera, control, output, predictor = Mailbox(), Mailbox(), Publisher(), predictor or Predictor()
    factories = []

    def make_predictor(*args, **kwargs):
        factories.append((args, kwargs))
        return predictor

    instance = FoundationPoseWorker(
        WorkerContext("foundationpose-test"), "vendor/source", device="cuda:1",
        registration_iterations=7, tracking_iterations=2, service="example/rgbd", frames=frames,
        camera_subscriber_factory=lambda service: factories.append(service) or camera,
        control_subscriber_factory=lambda service: factories.append(service) or control,
        result_publisher_factory=lambda service: factories.append(service) or output,
        sam3_subscriber_factory=lambda service: factories.append(service) or Mailbox(),
        predictor_factory=make_predictor,
    )
    instance.on_start()
    return instance, camera, control, output, predictor, factories


def test_lazy_register_exact_selection_once_then_track_both_common_frames_and_cleanup():
    instance, camera, control, output, predictor, factories = worker(frames=2)
    first, second = target(), target("B", sequence=2)
    try:
        camera.value = frame()
        instance.step_once()
        assert not predictor.starts and not output.results
        assert factories == ["example/rgbd", *services("example/rgbd"), sam3_services("example/rgbd")[1]]
        control.value = PoseControl((first, second))
        instance.step_once()
        assert predictor.starts == 1 and len(predictor.registers) == 2
        assert factories[-1] == (("vendor/source",), {
            "device": "cuda:1", "registration_iterations": 7, "tracking_iterations": 2})
        for registered, selected in zip(predictor.registers, (first, second)):
            assert registered[0] == selected.object_id
            assert registered[1] is selected.frame and registered[2] is selected.mask
            assert registered[3:] == (selected.mesh_path, selected.mesh_scale_to_m)
        assert output.results[0].frame is first.frame
        assert output.results[1].frame is second.frame
        control.value = PoseControl((first, second))
        # Process every received frame even if its metadata repeats or goes back.
        for sequence in (4, 1):
            incoming = frame(sequence)
            camera.value = incoming
            instance.step_once()
            latest = output.results[-1]
            assert latest.frame is incoming
            assert [item.object_id for item in latest.objects] == ["A", "B"]
            assert all(item.camera_from_object[0, 3] == sequence * .001 for item in latest.objects)
        assert len(predictor.registers) == 2 and len(predictor.tracks) == 4
        assert instance.processed_count == 2 and instance.ctx.stop_event.is_set()
    finally:
        instance.on_stop()
    assert camera.closed and control.closed and output.closed and predictor.closed


def test_control_is_full_desired_snapshot_and_new_session_registers_again():
    instance, camera, control, output, predictor, _ = worker()
    first, second = target(), target("B")
    try:
        control.value = PoseControl((first, second))
        instance.step_once()
        control.value = PoseControl((second,))
        camera.value = frame(2)
        instance.step_once()
        assert predictor.removed == ["A"]
        assert [item.object_id for item in output.results[-1].objects] == ["B"]
        control.value = PoseControl((replace(second, session_id="new-B"),))
        instance.step_once()
        assert predictor.removed == ["A", "B"]
        assert len(predictor.registers) == 3
        assert output.results[-1].objects[0].session_id == "new-B"
        control.value = PoseControl(())
        camera.value = frame(3)
        prior = len(output.results)
        instance.step_once()
        assert predictor.removed == ["A", "B", "B"]
        assert not instance.objects and len(output.results) == prior
    finally:
        instance.on_stop()


def test_tracking_exception_recovers_only_on_new_sam3_evidence_and_other_slot_continues():
    instance, camera, control, output, predictor, _ = worker()
    first, second = target(), target("B")
    try:
        control.value = PoseControl((first, second))
        instance.step_once()
        predictor.tracking_error = "A"
        for sequence in (2, 3):
            control.value = PoseControl((first, second))
            camera.value = frame(sequence)
            instance.step_once()
            a, b = output.results[-1].objects
            assert a.status == "recovering" and a.camera_from_object is None
            assert a.error == "RuntimeError: tracking failed"
            assert b.status == "tracking"
        assert [name for name, _ in predictor.tracks] == ["A", "B", "B"]
        assert len(predictor.registers) == 2
        predictor.tracking_error = None
        control.value = PoseControl((replace(first, session_id="explicit-new-A"), second))
        instance.step_once()
        assert len(predictor.registers) == 3
        assert output.results[-1].objects[0].status == "tracking"
    finally:
        instance.on_stop()


def test_registration_exception_waits_for_new_sam3_evidence_and_other_object_tracks():
    predictor = Predictor()
    predictor.registration_error = "A"
    instance, camera, control, output, predictor, _ = worker(predictor=predictor)
    selection = PoseControl((target(), target("B")))
    try:
        control.value = selection
        instance.step_once()
        assert output.results[0].objects[0].error == "RuntimeError: registration failed"
        control.value, camera.value = selection, frame(2)
        instance.step_once()
        assert len(predictor.registers) == 2
        assert [name for name, _ in predictor.tracks] == ["B"]
    finally:
        instance.on_stop()


def test_model_load_failure_is_status_error_and_explicit_new_session_can_start():
    predictor = Predictor()
    normal_start = predictor.start

    def fail_start():
        predictor.starts += 1
        raise RuntimeError("model loading failed")

    predictor.start = fail_start
    instance, camera, control, output, predictor, _ = worker(predictor=predictor)
    try:
        selection = PoseControl((target(),))
        control.value = selection
        instance.step_once()
        assert predictor.starts == 1 and not predictor.registers
        assert output.results[-1].objects[0].error == "RuntimeError: model loading failed"
        assert instance.predictor is None and predictor.closed
        control.value = selection
        instance.step_once()
        assert predictor.starts == 1
        predictor.start = normal_start
        control.value, camera.value = PoseControl((target(session_id="new-session"),)), frame(2)
        instance.step_once()
        assert predictor.starts == 2 and len(predictor.registers) == 1 and len(predictor.tracks) == 1
        assert output.results[-1].objects[0].status == "tracking"
    finally:
        instance.on_stop()
    assert predictor.closed


def visual_frame(sequence=1, depth=1.):
    return CameraFrame(np.zeros((6, 8, 3), np.uint8), np.full((6, 8), depth, np.float32),
                       {"sequence": sequence, "intrinsics": {"fx": 100, "fy": 100, "cx": 4, "cy": 3}})


def visual_target(object_id="A"):
    return replace(target(object_id), frame=visual_frame(), mask=np.ones((6, 8), bool))


def evidence(sequence=2, *, object_id="A", revision="revision", empty=False, depth=1.):
    count = 0 if empty else 1
    return Sam3Result(visual_frame(sequence, depth), revision, (
        Sam3ObjectResult(object_id, "object", np.ones((count, 6, 8), bool),
                         np.tile(np.array([[0, 0, 8, 6]], np.float32), (count, 1)),
                         np.ones(count, np.float32)),
    ))


def test_visual_mismatch_confirmations_reset_on_match_and_detector_miss_does_not_stop_track():
    instance, camera, control, output, predictor, _ = worker()
    try:
        control.value = PoseControl((visual_target(),))
        instance.step_once()
        instance.objects["A"].camera_from_object[0, 3] = 1
        for sequence in (2, 3):
            instance.sam3_subscriber.value = evidence(sequence)
            instance.step_once()
        assert output.results[-1].objects[0].visual_mismatch_count == 2
        assert output.results[-1].objects[0].status == "tracking"
        instance.sam3_subscriber.value = evidence(4, empty=True)
        camera.value = visual_frame(4)
        instance.step_once()
        assert len(predictor.tracks) == 1 and instance.objects["A"].status == "tracking"
        assert instance.objects["A"].visual_mismatch_count == 2
        instance.sam3_subscriber.value = evidence(5)
        instance.step_once()
        assert instance.objects["A"].visual_mismatch_count == 0
        assert len(predictor.registers) == 1
    finally:
        instance.on_stop()


def test_five_mismatches_recover_and_reregister_exact_sam3_frame_mask_then_validate():
    instance, camera, control, output, predictor, _ = worker()
    try:
        control.value = PoseControl((visual_target(), visual_target("B")))
        instance.step_once()
        instance.objects["A"].camera_from_object[0, 3] = 1
        predictor.registration_pose[2, 3] = 2  # New registration must pass depth gate too.
        for sequence in range(2, 7):
            instance.sam3_subscriber.value = evidence(sequence)
            instance.step_once()
        current = instance.objects["A"]
        assert current.status == "recovering" and current.camera_from_object is None
        assert current.visual_mismatch_count == 5 and current.error == "visual_mismatch:depth"
        assert len(predictor.registers) == 3  # A/B initially, A recovery on fifth mismatch.
        assert instance.objects["B"].status == "tracking"
        predictor.registration_pose[2, 3] = 1
        fresh = evidence(7)
        instance.sam3_subscriber.value = fresh
        instance.step_once()
        last = predictor.registers[-1]
        assert last[1] is fresh.frame and np.shares_memory(last[2], fresh.objects[0].masks)
        np.testing.assert_array_equal(last[2], fresh.objects[0].masks[0])
        assert instance.objects["A"].status == "tracking"
        assert instance.objects["A"].visual_mismatch_count == 0
    finally:
        instance.on_stop()


def test_track_exception_reregisters_when_next_matching_revision_observation_arrives():
    instance, camera, control, output, predictor, _ = worker()
    try:
        control.value = PoseControl((visual_target(),))
        instance.step_once()
        predictor.tracking_error = "A"
        camera.value = visual_frame(2)
        instance.step_once()
        assert instance.objects["A"].status == "recovering"
        instance.sam3_subscriber.value = evidence(3, revision="different-revision")
        instance.step_once()
        assert len(predictor.registers) == 1
        predictor.tracking_error = None
        fresh = evidence(4)
        instance.sam3_subscriber.value = fresh
        instance.step_once()
        current = instance.objects["A"]
        assert len(predictor.registers) == 2 and predictor.registers[-1][1] is fresh.frame
        np.testing.assert_array_equal(predictor.registers[-1][2], fresh.objects[0].masks[0])
        assert current.status == "tracking" and current.visual_mismatch_count == 0
    finally:
        instance.on_stop()


def test_pending_candidate_is_promoted_and_registered_on_fifth_visual_mismatch():
    instance, camera, control, output, predictor, _ = worker()
    try:
        source = CameraFrame(np.zeros((6, 160, 3), np.uint8), np.ones((6, 160), np.float32),
                             {"intrinsics": {"fx": 100, "fy": 100, "cx": 4, "cy": 3}})
        seed = np.zeros((6, 160), bool)
        seed[:, :8] = True
        control.value = PoseControl((replace(target(), frame=source, mask=seed),))
        instance.step_once()
        predictor.registration_pose[0, 3] = .8  # Recovered origin projects into the new box.
        moved = np.zeros((1, 6, 160), bool)
        moved[:, :, 80:88] = True
        for _ in range(5):
            fresh = Sam3Result(source, "revision", (
                Sam3ObjectResult("A", "object", moved, np.array([[80, 0, 88, 6]], np.float32),
                                 np.ones(1, np.float32)),
            ))
            instance.sam3_subscriber.value = fresh
            instance.step_once()
        assert len(predictor.registers) == 2
        assert instance.objects["A"].status == "tracking"
        assert instance.objects["A"].visual_mismatch_count == 0
        assert instance._associations["A"].accepted_box == (80., 0., 88., 6.)
    finally:
        instance.on_stop()


def test_same_sam3_batch_publishes_a_b_once_with_empty_degenerate_and_duplicate_masks_excluded():
    instance, camera, control, output, predictor, _ = worker()
    try:
        source = visual_frame()
        left, right = np.zeros((6, 8), bool), np.zeros((6, 8), bool)
        left[:, :4], right[:, 4:] = True, True
        control.value = PoseControl((replace(visual_target(), mask=left),
                                     replace(visual_target("B"), mask=right)))
        instance.step_once()
        objects = (
            Sam3ObjectResult("A", "first", np.array([np.zeros_like(left), left, left]),
                             np.array([[0, 0, 4, 6], [0, 0, 0, 6], [-2, -2, 4, 9]], np.float32),
                             np.ones(3, np.float32)),
            Sam3ObjectResult("B", "second", np.array([left, right]),
                             np.array([[0, 0, 4, 6], [4, 0, 8, 6]], np.float32),
                             np.ones(2, np.float32)),
        )
        before = len(output.results)
        instance.sam3_subscriber.value = Sam3Result(source, "revision", objects)
        instance.step_once()
        assert len(output.results) == before + 1
        assert [item.object_id for item in output.results[-1].objects] == ["A", "B"]
        assert instance._associations["A"].accepted_box == (0., 0., 4., 6.)
        assert instance._associations["B"].accepted_box == (4., 0., 8., 6.)
    finally:
        instance.on_stop()


def test_parser_and_module_help_do_not_import_camera_or_gpu_backends():
    args = build_parser().parse_args(["--source-dir", "vendor/source"])
    assert args.device == "cuda:0" and args.registration_iterations == 5 and args.tracking_iterations == 1
    script = """
import sys
from grasp_workbench.workers.worker_foundationpose import build_parser
build_parser().parse_args(['--source-dir', 'source'])
assert not any(name in sys.modules for name in ('torch', 'pyzed', 'pyrealsense2', 'estimater'))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
