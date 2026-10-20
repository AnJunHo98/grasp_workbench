"""Previous bbox-selection behavior with unchanged source candidate indices."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from grasp_workbench.models.sam3_association import Sam3Association


def test_defaults_match_previous_temporal_association():
    association = Sam3Association()
    assert association.temporal_iou_threshold == .1
    assert association.maximum_area_ratio == 2.5
    assert association.switch_confirm_frames == 3
    assert association.maximum_center_jump_ratio == .75
    assert association.track_memory_frames == 15


def test_initial_selection_uses_score_and_preserves_source_indices_and_arrays():
    boxes = np.array([[0, 0, 10, 10], [30, 30, 40, 40], [60, 60, 70, 70]], dtype=np.float32)
    scores = np.array([.2, .9, .9], dtype=np.float32)
    original_boxes, original_scores = boxes.copy(), scores.copy()
    association = Sam3Association()
    result = association.select(boxes, scores)
    assert result.decision == "initial" and result.pending is None
    assert result.candidate.index == 1
    assert result.candidate.box == (30, 30, 40, 40)
    assert association.accepted_box == result.candidate.box
    np.testing.assert_array_equal(boxes, original_boxes)
    np.testing.assert_array_equal(scores, original_scores)


def test_explicit_selected_bbox_seed_takes_precedence_over_score():
    association = Sam3Association((30, 30, 40, 40))
    result = association.select([[0, 0, 10, 10], [31, 31, 41, 41]], [.99, .6])
    assert result.decision == "associated"
    assert result.candidate.index == 1
    assert result.candidate.box == association.accepted_box == (31, 31, 41, 41)


def test_equal_iou_is_resolved_by_score_then_source_order():
    association = Sam3Association((0, 0, 10, 10))
    result = association.select([[0, 0, 10, 10]] * 3, [.5, .8, .8])
    assert result.candidate.index == 1


@pytest.mark.parametrize("width,decision", [(25, "associated"), (25.1, "pending_jump")])
def test_area_ratio_uses_existing_boundary(width, decision):
    association = Sam3Association((0, 0, 10, 10))
    result = association.select([[0, 0, width, 10]], [.9])
    assert result.decision == decision
    assert (result.candidate is not None) == (decision == "associated")


def test_pending_jumps_never_auto_promote_after_confirmation():
    association = Sam3Association((0, 0, 10, 10))
    for count in (1, 2, 3, 4):
        result = association.select([[100, 100, 110, 110]], [.9])
        assert result.candidate is None and result.pending.index == 0
        assert association.accepted_box == (0, 0, 10, 10)
        assert association.pending_count == count
        assert result.decision == ("pending_jump" if count < 3 else "pending_jump_verification")


def test_explicit_pending_promotion_requires_confirmation_and_returns_latest_candidate():
    association = Sam3Association((0, 0, 10, 10))
    assert association.accept_pending() is None
    for offset in (100, 101):
        association.select([[offset, 0, offset + 10, 10]], [.9])
        assert association.accept_pending() is None
    selected = association.select([[102, 0, 112, 10]], [.7]).pending
    assert association.accept_pending() == selected
    assert association.accepted_box == (102, 0, 112, 10)
    assert association.pending_box is None and association.pending_count == 0
    assert association.accept_pending() is None
    assert association.select([[103, 0, 113, 10]], [.8]).decision == "associated"


def test_nonoverlap_pending_uses_previous_center_distance_and_keeps_original_pending_box():
    association = Sam3Association((0, 0, 10, 10))
    association.select([[100, 0, 110, 10]], [.9])
    association.select([[110, 0, 120, 10]], [.9])
    assert association.pending_count == 2
    assert association.pending_box == (100, 0, 110, 10)
    association.select([[120, 0, 130, 10]], [.9])
    assert association.pending_count == 1
    assert association.pending_box == (120, 0, 130, 10)


def test_inconsistent_pending_area_restarts_confirmation():
    association = Sam3Association((0, 0, 10, 10))
    association.select([[100, 100, 110, 110]], [.9])
    association.select([[100, 100, 130, 110]], [.9])
    assert association.pending_count == 1
    assert association.pending_box == (100, 100, 130, 110)


def test_missing_candidate_resets_pending_and_keeps_existing_15_frame_memory():
    association = Sam3Association((0, 0, 10, 10))
    association.select([[100, 100, 110, 110]], [.9])
    for count in range(1, 16):
        result = association.select([], [])
        assert result.decision == "missing" and result.candidate is result.pending is None
        assert association.missed_frames == count
        assert association.accepted_box == (0, 0, 10, 10)
        assert association.pending_box is None and association.pending_count == 0
    association.select([], [])
    assert association.accepted_box is None
    assert association.select([[100, 100, 110, 110]], [.9]).decision == "initial"
    assert association.missed_frames == 0


def test_return_to_accepted_bbox_clears_pending():
    association = Sam3Association((0, 0, 10, 10))
    association.select([[100, 100, 110, 110]], [.9])
    result = association.select([[1, 1, 11, 11]], [.9])
    assert result.decision == "associated"
    assert association.pending_box is None and association.pending_count == 0
    assert association.accept_pending() is None


def test_reset_clears_selection_and_can_seed_new_user_selection():
    association = Sam3Association((0, 0, 10, 10))
    association.select([[100, 100, 110, 110]], [.9])
    association.reset((50, 50, 60, 60))
    assert association.accepted_box == (50, 50, 60, 60)
    assert association.pending_count == association.missed_frames == 0
    assert association.accept_pending() is None
    association.reset()
    assert association.accepted_box is None


def test_a_b_associations_keep_independent_seeds_without_reordering_candidates():
    a, b = Sam3Association((0, 0, 10, 10)), Sam3Association((100, 100, 110, 110))
    boxes, scores = [[101, 101, 111, 111], [1, 1, 11, 11]], [.9, .6]
    assert a.select(boxes, scores).candidate.index == 1
    assert b.select(boxes, scores).candidate.index == 0
    assert a.accepted_box != b.accepted_box


def test_helper_import_has_no_model_camera_or_old_application_dependency():
    script = """
import sys
for name in ('torch', 'numpy', 'sam3', 'iceoryx2', 'pyzed', 'pyrealsense2', 'object_pose_system'):
    sys.modules[name] = None
from grasp_workbench.models.sam3_association import Sam3Association
assert Sam3Association((0, 0, 10, 10)).select([[1, 1, 11, 11]], [.8]).candidate.index == 0
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
