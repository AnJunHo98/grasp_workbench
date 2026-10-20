"""Explicit existing/generated mesh branches with fake worker endpoints."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ui.camera_window import CameraWindow
from test_camera_ui import FakeReceiver, completed_mesh, sam3_result


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    window = CameraWindow(sam3d=True, foundationpose=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    window.prompt_inputs["A"].setText("cup")
    window.prompt_inputs["B"].setText("bottle")
    window.apply_prompts()
    yield window
    window.receiver.alive = False
    window.close()
    app.processEvents()


def receive_candidates(window, *, count=1, sequence=7):
    result = sam3_result(window._prompts.revision, sequence=sequence, count=count, include_b=True)
    window.receiver.sam3_latest = result
    window.poll_once()
    return result


def test_generate_and_start_buttons_follow_candidate_and_busy_state(window):
    for key in ("A", "B"):
        assert window.pose_start_buttons[key].text() == "기존 mesh로 Start"
        assert "생성 후 Start" in window.mesh_pose_buttons[key].text()
        assert not window.mesh_pose_buttons[key].isEnabled()
    receive_candidates(window)
    assert all(button.isEnabled() for button in window.mesh_pose_buttons.values())
    window.mesh_pose_buttons["A"].click()
    assert not window.mesh_buttons["A"].isEnabled()
    assert not window.mesh_pose_buttons["A"].isEnabled()
    assert window.mesh_pose_buttons["B"].isEnabled()


def test_existing_mesh_start_does_not_generate_mesh(window):
    source = receive_candidates(window)
    window.pose_mesh_inputs["A"].setText("assets/existing.ply")
    window.pose_start_buttons["A"].click()
    assert not window.receiver.mesh_requests
    assert not window._mesh_pose_jobs
    target, = window.receiver.pose_controls[-1].targets
    assert target.mesh_path == "assets/existing.ply" and target.frame is source.frame
    np.testing.assert_array_equal(target.mask, source.objects[0].masks[0])


def test_generated_mesh_overrides_named_mesh_and_starts_with_its_frozen_source(window):
    receive_candidates(window)
    window.pose_mesh_choices["A"].setCurrentIndex(1)
    previous_path = window.pose_mesh_inputs["A"].text()
    window.pose_scale_inputs["A"].setValue(0.001)
    control_count = len(window.receiver.pose_controls)
    window.mesh_pose_buttons["A"].click()
    request, = window.receiver.mesh_requests
    assert window._mesh_pose_jobs["A"] == request.request_id
    assert len(window.receiver.pose_controls) == control_count
    assert window.pose_mesh_inputs["A"].text() == previous_path

    receive_candidates(window, sequence=99)
    window._sam3_result = None
    result = completed_mesh(request)
    result.mesh_path = "generated/new-A.ply"
    window.receiver.mesh_events.append(result)
    window.poll_once()

    target, = window.receiver.pose_controls[-1].targets
    assert len(window.receiver.pose_controls) == control_count + 1
    assert target.mesh_path == window.pose_mesh_inputs["A"].text() == "generated/new-A.ply"
    assert window.pose_mesh_choices["A"].currentData() is None
    assert target.mesh_scale_to_m == window.pose_scale_inputs["A"].value() == 1.0
    assert target.frame is request.frame and target.frame.metadata["sequence"] == 7
    assert target.mask is request.mask
    assert target.prompt_revision == request.revision
    assert not window._mesh_pose_jobs
    assert window.mesh_buttons["A"].isEnabled() and window.mesh_pose_buttons["A"].isEnabled()


def test_generated_a_and_b_start_independently_on_their_completions(window):
    receive_candidates(window)
    for key in ("A", "B"):
        window.mesh_pose_buttons[key].click()
    a, b = window.receiver.mesh_requests
    window.receiver.mesh_events.append(completed_mesh(b))
    window.poll_once()
    assert set(window._pose_targets) == {"B"}
    assert window._pose_targets["B"].frame is b.frame
    assert window._mesh_pose_jobs == {"A": a.request_id}
    control_count = len(window.receiver.pose_controls)
    window.receiver.mesh_events.append(completed_mesh(a, status="running"))
    window.poll_once()
    assert len(window.receiver.pose_controls) == control_count
    window.receiver.mesh_events.append(completed_mesh(a))
    window.poll_once()
    assert set(window._pose_targets) == {"A", "B"}
    assert window._pose_targets["A"].frame is a.frame
    assert not window._mesh_pose_jobs


def test_generate_mesh_only_does_not_start_foundationpose(window):
    receive_candidates(window)
    control_count = len(window.receiver.pose_controls)
    window.mesh_buttons["A"].click()
    request, = window.receiver.mesh_requests
    window.receiver.mesh_events.append(completed_mesh(request))
    window.poll_once()
    assert not window._mesh_pose_jobs and not window._pose_targets
    assert len(window.receiver.pose_controls) == control_count
    assert window._mesh_results["A"].request_id == request.request_id


def test_mesh_only_completion_fills_metric_scale_and_preserves_initial_pose(window):
    receive_candidates(window)
    window.pose_scale_inputs["A"].setValue(0.11)
    window.mesh_buttons["A"].click()
    request, = window.receiver.mesh_requests
    result = completed_mesh(request)
    result.camera_from_object[:3, 3] = [0.1, 0.2, 0.7]
    window.receiver.mesh_events.append(result)
    window.poll_once()
    assert not window._pose_targets
    assert window.pose_mesh_inputs["A"].text() == result.mesh_path
    assert window.pose_scale_inputs["A"].value() == 1.0
    np.testing.assert_array_equal(window._mesh_results["A"].camera_from_object[:3, 3], [0.1, 0.2, 0.7])
    assert window.axes.get_xlabel() == "X (m)"


def test_mesh_only_completion_does_not_change_an_existing_mesh_scale(window):
    receive_candidates(window)
    window.pose_mesh_inputs["A"].setText("existing.obj")
    window.pose_scale_inputs["A"].setValue(0.001)
    window.mesh_buttons["A"].click()
    request, = window.receiver.mesh_requests
    window.receiver.mesh_events.append(completed_mesh(request))
    window.poll_once()
    assert window.pose_mesh_inputs["A"].text() == "existing.obj"
    assert window.pose_scale_inputs["A"].value() == 0.001


def test_stop_cancels_pending_generate_and_start_without_discarding_generated_mesh(window):
    receive_candidates(window)
    window.mesh_pose_buttons["A"].click()
    request, = window.receiver.mesh_requests
    window.stop_pose("A")
    control_count = len(window.receiver.pose_controls)
    window.receiver.mesh_events.append(completed_mesh(request))
    window.poll_once()
    assert not window._mesh_pose_jobs and not window._pose_targets
    assert len(window.receiver.pose_controls) == control_count
    assert "A" in window._mesh_results


@pytest.mark.parametrize("action", ["apply", "prompt_edit", "candidate_change", "existing_start"])
def test_new_user_action_cancels_pending_automatic_start(window, action):
    receive_candidates(window, count=2)
    window.candidate_inputs["A"].setCurrentIndex(1)
    window.pose_mesh_inputs["A"].setText("assets/existing.ply")
    window.mesh_pose_buttons["A"].click()
    request, = window.receiver.mesh_requests
    if action == "apply":
        window.apply_prompts()
    elif action == "prompt_edit":
        window.prompt_inputs["A"].setText("different cup")
    elif action == "candidate_change":
        window.candidate_inputs["A"].setCurrentIndex(2)
    else:
        window.pose_start_buttons["A"].click()
        assert window._pose_targets["A"].mesh_path == "assets/existing.ply"
    control_count = len(window.receiver.pose_controls)
    assert "A" not in window._mesh_pose_jobs
    window.receiver.mesh_events.append(completed_mesh(request))
    window.poll_once()
    assert len(window.receiver.pose_controls) == control_count
    assert window.pose_mesh_inputs["A"].text() == "assets/existing.ply"


def test_generation_error_ends_chain_without_pose_start_or_automatic_retry(window):
    receive_candidates(window)
    window.mesh_pose_buttons["A"].click()
    request, = window.receiver.mesh_requests
    control_count = len(window.receiver.pose_controls)
    window.receiver.mesh_events.append(completed_mesh(request, status="error", error="generation failed"))
    window.poll_once()
    window.poll_once()
    assert "generation failed" in window.mesh_status.text()
    assert not window._mesh_pose_jobs and not window._pose_targets
    assert len(window.receiver.pose_controls) == control_count
    assert len(window.receiver.mesh_requests) == 1
    assert window.mesh_pose_buttons["A"].isEnabled()
