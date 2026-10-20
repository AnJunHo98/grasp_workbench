"""Frozen display attachments, measured FK and independent worker streaming."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.grasp_selection import SelectionRequest, decode_request, encode_request
from grasp_workbench.models.grasp_fk import GraspFkTracker, _average
from grasp_workbench.workers.worker_grasp_selection import GraspSelectionWorker


def pose(x=0., y=0., z=0., angle=0.):
    matrix = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    matrix[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    matrix[:3, 3] = x, y, z
    return matrix


def reference(*, cross=False):
    return {"selected_candidate": {"targets": {
        "left": {"object_id": "A", "observation_id": "obs-A", "instance_id": "instance-A",
                 "base_from_tcp": pose(1).tolist()},
        "right": {"object_id": "B" if cross else "A", "observation_id": "obs-B" if cross else "obs-A",
                  "instance_id": "instance-B" if cross else "instance-A", "base_from_tcp": pose(2).tolist()},
    }}, "object_poses": {"obs-A": {"base_from_object": pose().tolist()},
                          "obs-B": {"base_from_object": pose(10).tolist()}},
        "objects": [{"camera_from_object": pose(7, 8, 9).tolist()}]}


def snapshot(sequence=1, *, left=None, right=None):
    return {"state_sequence": sequence, "state_timestamp_ns": sequence * 100,
            "current_hands": {"left": pose(1) if left is None else left,
                              "right": pose(2) if right is None else right}}


def test_freezes_selected_relations_not_measured_contact_and_preserves_camera_and_input():
    original = reference()
    before = deepcopy(original)
    tracker = GraspFkTracker()
    tracker.attach(original)
    value = tracker.estimate(snapshot(2, left=pose(1, 2, 3), right=pose(2, 2, 3)))
    item = value["objects"]["obs-A"]
    np.testing.assert_allclose(item["base_from_object"], pose(0, 2, 3))
    assert item["state"] == "attached" and item["observed_sequence"] == 2
    assert item["translation_disagreement_m"] == item["rotation_disagreement_rad"] == 0
    assert original == before and not value["robot_command_published"]
    original["selected_candidate"]["targets"]["left"]["base_from_tcp"][0][3] = 900
    assert tracker.objects["obs-A"]["object_from_tcp_by_arm"]["left"][0][3] == 1


def test_two_arm_average_uses_old_quaternion_hemisphere_and_reports_disagreement():
    tracker = GraspFkTracker()
    tracker.attach(reference())
    left_object, right_object = pose(0, 1, 0, np.deg2rad(170)), pose(2, 1, 0, np.deg2rad(-170))
    state = snapshot(3, left=left_object @ pose(1), right=right_object @ pose(2))
    item = tracker.estimate(state)["objects"]["obs-A"]
    np.testing.assert_allclose(item["base_from_object"], pose(1, 1, 0, np.pi), atol=1e-12)
    assert item["translation_disagreement_m"] == pytest.approx(2)
    assert item["rotation_disagreement_rad"] == pytest.approx(np.deg2rad(20))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotation_average_supports_each_half_turn_axis_without_linear_matrix_average(axis):
    matrix = np.eye(4)
    matrix[:3, :3] = -np.eye(3)
    matrix[axis, axis] = 1
    np.testing.assert_allclose(_average([matrix, matrix]), matrix, atol=1e-12)


def test_cross_object_attachments_are_not_averaged_together():
    tracker = GraspFkTracker()
    tracker.attach(reference(cross=True))
    value = tracker.estimate(snapshot(4, left=pose(4), right=pose(5)))
    np.testing.assert_allclose(value["objects"]["obs-A"]["base_from_object"], pose(3))
    np.testing.assert_allclose(value["objects"]["obs-B"]["base_from_object"], pose(13))
    assert all(item["translation_disagreement_m"] == 0 for item in value["objects"].values())


def test_missing_robot_feedback_or_hand_leaves_effective_pose_unavailable_then_detach():
    tracker = GraspFkTracker()
    tracker.attach(reference())
    for state in (None, {"current_hands": snapshot()["current_hands"]},
                  {**snapshot(), "current_hands": {"left": pose(1)}}):
        value = tracker.estimate(state)
        assert value["state"] == "waiting_for_robot_state"
        assert value["objects"]["obs-A"]["base_from_object"] is None
    assert tracker.detach() == {"kind": "grasp_fk", "state": "detached", "objects": {},
                                "robot_command_published": False}
    assert not tracker.objects


def test_fixing_relation_requires_selected_base_object_and_tcp_not_default_identity():
    tracker = GraspFkTracker()
    value = reference()
    value["object_poses"]["obs-A"]["base_from_object"] = None
    with pytest.raises(ValueError, match="base-frame"):
        tracker.attach(value)
    assert not tracker.objects


@pytest.mark.parametrize("action", ["attach", "detach"])
def test_control_codecs_preserve_action_and_own_frozen_attachment(action):
    request = SelectionRequest("control", {}, (), action, reference() if action == "attach" else None)
    decoded = decode_request(encode_request(request))
    assert decoded == request
    if action == "attach":
        decoded.attachment["selected_candidate"]["targets"]["left"]["base_from_tcp"][0][3] = 99
        assert request.attachment["selected_candidate"]["targets"]["left"]["base_from_tcp"][0][3] == 1


class Mailbox:
    def __init__(self):
        self.value = None

    def receive_latest(self):
        value, self.value = self.value, None
        return value

    def close(self):
        pass


class Publisher:
    def __init__(self):
        self.values = []

    def update_connections(self):
        pass

    def publish(self, value):
        self.values.append(value)

    def close(self):
        pass


def test_same_worker_polls_state_after_attachment_without_reselection_and_detaches():
    inbox, output, state = Mailbox(), Publisher(), snapshot()
    predictor = SimpleNamespace(start=lambda: None, close=lambda: None, robot_snapshot=lambda: state,
                                predict=lambda request: {"kind": "selection", "camera_raw": "unchanged"})
    worker = GraspSelectionWorker(WorkerContext("fk-test"), subscriber_factory=lambda _: inbox,
                                  publisher_factory=lambda _: output, predictor_factory=lambda **kwargs: predictor)
    worker.on_start()
    try:
        worker.step_once()
        assert not output.values
        inbox.value = SelectionRequest("attach-1", {}, (), "attach", reference())
        worker.step_once()
        assert output.values[-1].selection["state"] == "attached"
        before = len(output.values)
        worker.step_once()
        assert len(output.values) == before
        state = snapshot(2, left=pose(1, 3), right=pose(2, 3))
        worker.step_once()
        latest = output.values[-1]
        assert latest.request_id == "attach-1" and latest.selection["kind"] == "grasp_fk"
        np.testing.assert_allclose(latest.selection["objects"]["obs-A"]["base_from_object"], pose(0, 3))
        inbox.value = SelectionRequest("detach-1", {}, (), "detach")
        worker.step_once()
        assert output.values[-1].selection["state"] == "detached"
        state, before = snapshot(3), len(output.values)
        worker.step_once()
        assert len(output.values) == before and worker.processed_count == 2
    finally:
        worker.on_stop()


def test_first_missing_state_waiting_event_is_once_and_detach_needs_no_predictor():
    inbox, output = Mailbox(), Publisher()
    constructions = []
    predictor = SimpleNamespace(start=lambda: None, close=lambda: None, robot_snapshot=lambda: None)
    worker = GraspSelectionWorker(WorkerContext("fk-test"), subscriber_factory=lambda _: inbox,
                                  publisher_factory=lambda _: output,
                                  predictor_factory=lambda **kwargs: constructions.append(1) or predictor)
    worker.on_start()
    try:
        inbox.value = SelectionRequest("detach-initial", {}, (), "detach")
        worker.step_once()
        assert not constructions and output.values[-1].selection["state"] == "detached"
        inbox.value = SelectionRequest("attach-waiting", {}, (), "attach", reference())
        worker.step_once()
        assert output.values[-1].selection["state"] == "waiting_for_robot_state"
        before = len(output.values)
        for _ in range(3):
            worker.step_once()
        assert len(output.values) == before
    finally:
        worker.on_stop()
