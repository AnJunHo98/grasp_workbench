"""Select a continuous SAM3 candidate without changing the displayed candidate order."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    index: int
    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class AssociationSelection:
    candidate: Candidate | None
    pending: Candidate | None
    decision: str


def _area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def _box_iou(left, right):
    intersection = _area((max(left[0], right[0]), max(left[1], right[1]),
                          min(left[2], right[2]), min(left[3], right[3])))
    union = _area(left) + _area(right) - intersection
    return intersection / union if union > 0 else 0.


def _area_ratio(left, right):
    left_area, right_area = _area(left), _area(right)
    return max(left_area, right_area) / min(left_area, right_area) if min(left_area, right_area) > 0 else math.inf


class Sam3Association:
    """The previous bbox association; pending jumps need explicit pose verification."""

    def __init__(self, seed_box=None, *, temporal_iou_threshold=.1, maximum_area_ratio=2.5,
                 switch_confirm_frames=3, maximum_center_jump_ratio=.75, track_memory_frames=15):
        self.temporal_iou_threshold = temporal_iou_threshold
        self.maximum_area_ratio = maximum_area_ratio
        self.switch_confirm_frames = switch_confirm_frames
        self.maximum_center_jump_ratio = maximum_center_jump_ratio
        self.track_memory_frames = track_memory_frames
        self.reset(seed_box)

    def reset(self, seed_box=None):
        self.accepted_box = tuple(seed_box) if seed_box is not None else None
        self.pending_box = self._pending_candidate = None
        self.pending_count = self.missed_frames = 0

    def select(self, boxes, scores):
        candidates = [Candidate(index, tuple(map(float, box)), float(score))
                      for index, (box, score) in enumerate(zip(boxes, scores))]
        if not candidates:
            self.missed_frames += 1
            self.pending_box = self._pending_candidate = None
            self.pending_count = 0
            if self.missed_frames > self.track_memory_frames:
                self.accepted_box = None
            return AssociationSelection(None, None, "missing")
        self.missed_frames = 0
        if self.accepted_box is None:
            selected = max(candidates, key=lambda value: value.score)
            self.accepted_box = selected.box
            return AssociationSelection(selected, None, "initial")
        selected = max(candidates, key=lambda value: (_box_iou(self.accepted_box, value.box), value.score))
        if (_box_iou(self.accepted_box, selected.box) >= self.temporal_iou_threshold
                and _area_ratio(self.accepted_box, selected.box) <= self.maximum_area_ratio):
            self.accepted_box = selected.box
            self.pending_box = self._pending_candidate = None
            self.pending_count = 0
            return AssociationSelection(selected, None, "associated")
        if self.pending_box is not None and self._plausible(self.pending_box, selected.box):
            self.pending_count += 1
        else:
            self.pending_box, self.pending_count = selected.box, 1
        self._pending_candidate = selected
        decision = "pending_jump_verification" if self.pending_count >= self.switch_confirm_frames else "pending_jump"
        return AssociationSelection(None, selected, decision)

    def _plausible(self, previous, current):
        if _area_ratio(previous, current) > self.maximum_area_ratio:
            return False
        if _box_iou(previous, current) >= self.temporal_iou_threshold:
            return True
        previous_center = ((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2)
        current_center = ((current[0] + current[2]) / 2, (current[1] + current[3]) / 2)
        diagonal = max(math.hypot(previous[2] - previous[0], previous[3] - previous[1]), 1.)
        return math.dist(previous_center, current_center) <= self.maximum_center_jump_ratio * diagonal

    def accept_pending(self):
        if self._pending_candidate is None or self.pending_count < self.switch_confirm_frames:
            return None
        selected = self._pending_candidate
        self.accepted_box = selected.box
        self.pending_box = self._pending_candidate = None
        self.pending_count = 0
        return selected
