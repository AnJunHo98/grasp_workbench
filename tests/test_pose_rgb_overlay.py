"""Camera-space pose axes drawn on the SAM3 RGB overlay, without model SDKs."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result
from grasp_workbench.ui.camera_window import CameraWindow


class FakeReceiver:
    def __init__(self, *_args, **_kwargs):
        self.latest = self.sam3_latest = self.pose_latest = None
        self.alive = False

    def start(self):
        self.alive = True

    def stop(self):
        self.alive = False

    def is_alive(self):
        return self.alive

    def snapshot(self):
        return {"ready": True, "error": None, "received_count": 1,
                "last_received_monotonic_s": None}

    def take_latest(self):
        result, self.latest = self.latest, None
        return result

    def take_sam3_latest(self):
        result, self.sam3_latest = self.sam3_latest, None
        return result

    def take_pose_latest(self):
        result, self.pose_latest = self.pose_latest, None
        return result

    def set_prompts(self, _prompts):
        pass

    def set_pose_control(self, _control):
        pass


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    window = CameraWindow(foundationpose=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    for key in ("A", "B"):
        window.prompt_inputs[key].setText(key)
        window.pose_mesh_inputs[key].setText(f"{key}.glb")
    window.apply_prompts()
    window.receiver.latest = frame()
    window.receiver.sam3_latest = sam3_result(window._prompts.revision)
    window.poll_once()
    for key in ("A", "B"):
        window.start_pose(key)
    yield window
    window.close()
    app.processEvents()


def frame(sequence=42, value=(30, 40, 50)):
    rgb = np.empty((240, 320, 3), dtype=np.uint8)
    rgb[:] = value
    return CameraFrame(rgb, np.ones((240, 320), dtype=np.float32), {
        "camera_id": "fake", "sequence": sequence,
        "intrinsics": {"fx": 400., "fy": 400., "cx": 160., "cy": 120.},
    })


def sam3_result(revision, sequence=7):
    objects = []
    for key, y in (("A", 20), ("B", 60)):
        mask = np.zeros((1, 240, 320), dtype=bool)
        mask[:, y + 5:y + 15, 25:35] = True
        objects.append(Sam3ObjectResult(key, key, mask,
                                       np.array([[20, y, 40, y + 20]], dtype=np.float32),
                                       np.ones(1, dtype=np.float32)))
    return Sam3Result(frame(sequence, (150, 150, 150)), revision, tuple(objects))


def pose(window, key, *, matrix=True):
    transform = np.eye(4, dtype=np.float32)
    diagonal = np.sqrt(.5)
    transform[:3, :3] = ((1, 0, 0), (0, diagonal, -diagonal), (0, diagonal, diagonal))
    transform[:3, 3] = (-.2 if key == "A" else .2, 0, 1)
    return ObjectPose(key, window._pose_targets[key].session_id, f"{key}.glb",
                      "tracking", transform if matrix else None)


def color_at(image, x, y):
    return image.pixelColor(x, y).getRgb()[:3]


def assert_axis_pixels(image, source, item):
    intrinsics = source.metadata["intrinsics"]
    matrix = item.camera_from_object
    camera_points = matrix[:3, 3] + np.vstack((np.zeros(3), np.eye(3) * .05)) @ matrix[:3, :3].T
    pixels = camera_points[:, :2] / camera_points[:, 2:3]
    pixels = pixels * (intrinsics["fx"], intrinsics["fy"]) + (intrinsics["cx"], intrinsics["cy"])
    for end, color in zip(pixels[1:], ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        x, y = np.rint((pixels[0] + end) / 2).astype(int)
        assert any(color_at(image, x + dx, y + dy) == color
                   for dx in range(-2, 3) for dy in range(-2, 3)), (item.object_id, color, x, y)


def test_projects_both_pose_axes_over_masks_boxes_without_mutating_inputs(window):
    source = frame()
    segmentation = window._sam3_result
    objects = (pose(window, "A"), pose(window, "B"))
    rgb_before = source.rgb.copy()
    sam3_rgb_before = segmentation.frame.rgb.copy()
    masks_before = [item.masks.copy() for item in segmentation.objects]
    poses_before = [item.camera_from_object.copy() for item in objects]
    window._display_frame(source, segmentation, objects)
    image = window.image_view.image
    assert color_at(image, 300, 220) == (30, 40, 50)
    assert color_at(image, 30, 30) == (40, 110, 85)
    assert color_at(image, 20, 35) == (57, 217, 138)
    assert color_at(image, 30, 70) == (120, 96, 63)
    assert color_at(image, 20, 75) == (255, 180, 84)
    for item in objects:
        assert_axis_pixels(image, source, item)
    np.testing.assert_array_equal(source.rgb, rgb_before)
    np.testing.assert_array_equal(segmentation.frame.rgb, sam3_rgb_before)
    for item, before in zip(segmentation.objects, masks_before):
        np.testing.assert_array_equal(item.masks, before)
    for item, before in zip(objects, poses_before):
        np.testing.assert_array_equal(item.camera_from_object, before)
    source.rgb[:] = 0
    assert color_at(image, 300, 220) == (30, 40, 50)


def test_pose_update_keeps_live_camera_rgb_and_latest_sam3_overlay(window):
    source = frame(sequence=42)
    objects = (pose(window, "A"), pose(window, "B"))
    window.receiver.pose_latest = PoseResult(frame(sequence=5, value=(200, 200, 200)), objects)
    window.poll_once()
    image = window.image_view.image
    assert color_at(image, 300, 220) == (30, 40, 50)
    assert color_at(image, 30, 30) == (40, 110, 85)
    for item in objects:
        assert_axis_pixels(image, source, item)
    text = window.metadata_label.text()
    assert "camera=fake · sequence=42 ·" in text and "실시간 카메라" in text
    assert "FP pose sequence=5" in text and "SAM3 overlay sequence=7" in text
    assert "SAM3 결과의 원본 프레임" not in text


def test_independent_sam3_update_redraws_on_live_camera_and_preserves_axes(window):
    source = frame(sequence=42)
    objects = (pose(window, "A"), pose(window, "B"))
    window.receiver.pose_latest = PoseResult(source, objects)
    window.poll_once()
    segmentation = sam3_result(window._prompts.revision, sequence=900)
    segmentation.objects[0].masks[:] = False
    segmentation.objects[0].masks[:, 180:190, 25:35] = True
    segmentation.objects[0].boxes[:] = (20, 175, 40, 195)
    window.receiver.sam3_latest = segmentation
    window.poll_once()
    image = window.image_view.image
    assert color_at(image, 300, 220) == (30, 40, 50)
    assert color_at(image, 30, 30) == (30, 40, 50)
    assert color_at(image, 30, 185) == (40, 110, 85)
    assert color_at(image, 20, 185) == (57, 217, 138)
    for item in objects:
        assert_axis_pixels(image, source, item)
    assert "sequence=42" in window.metadata_label.text()
    assert "900" in window.metadata_label.text()


def test_partial_pose_result_preserves_latest_axes_for_both_objects(window):
    source = frame()
    a, b = pose(window, "A"), pose(window, "B")
    window.receiver.pose_latest = PoseResult(source, (a, b))
    window.poll_once()
    assert_axis_pixels(window.image_view.image, source, a)
    window.receiver.pose_latest = PoseResult(frame(sequence=43), (b,))
    window.poll_once()
    image = window.image_view.image
    assert_axis_pixels(image, source, a)
    assert_axis_pixels(image, source, b)
    assert set(window._pose_objects) == {"A", "B"}
    assert "camera=fake · sequence=42 ·" in window.metadata_label.text()


def test_restarting_only_object_keeps_frame_label_without_old_session_axes(window):
    window.stop_pose("B")
    source = frame(sequence=42)
    window.receiver.pose_latest = PoseResult(source, (pose(window, "A"),))
    window.poll_once()
    window.start_pose("A")
    assert color_at(window.image_view.image, 90, 120) == (30, 40, 50)
    assert "sequence=42" in window.metadata_label.text()
    assert "실시간 카메라" in window.metadata_label.text()
    assert "SAM3 결과의 원본 프레임" not in window.metadata_label.text()


def test_none_pose_leaves_mask_and_rgb_visible_without_axes(window):
    source = frame()
    window._display_frame(source, window._sam3_result, (pose(window, "A", matrix=False),))
    assert color_at(window.image_view.image, 30, 30) == (40, 110, 85)
    assert color_at(window.image_view.image, 90, 120) == (30, 40, 50)


def test_sam3_mismatch_count_and_recovering_clear_only_the_affected_pose(window):
    a, b = pose(window, "A"), pose(window, "B")
    a.visual_mismatch_count = 4
    window.receiver.pose_latest = PoseResult(frame(), (a, b))
    window.poll_once()
    assert "SAM3 불일치 4/5" in window.pose_status.text()
    assert len(window.axes.lines) == 6
    recovering = ObjectPose("A", a.session_id, a.mesh_path, "recovering",
                            error="visual_mismatch:bbox", visual_mismatch_count=5)
    window.receiver.pose_latest = PoseResult(frame(sequence=43), (recovering,))
    window.poll_once()
    assert "A: recovering" in window.pose_status.text()
    assert len(window.axes.lines) == 3
    assert color_at(window.image_view.image, 90, 120) == (30, 40, 50)
    assert_axis_pixels(window.image_view.image, frame(sequence=43), b)
    assert color_at(window.image_view.image, 30, 30) == (40, 110, 85)


def test_apply_removes_old_masks_but_keeps_current_tracking_axes(window):
    source = frame()
    objects = (pose(window, "A"), pose(window, "B"))
    window.receiver.pose_latest = PoseResult(source, objects)
    window.poll_once()
    window.prompt_inputs["A"].setText("new prompt")
    window.apply_prompts()
    image = window.image_view.image
    assert color_at(image, 30, 30) == (30, 40, 50)
    assert color_at(image, 20, 35) == (30, 40, 50)
    for item in objects:
        assert_axis_pixels(image, source, item)
    assert window._sam3_result is None


def test_stopping_pose_removes_only_its_axes_and_keeps_live_camera(window):
    source = frame()
    a, b = pose(window, "A"), pose(window, "B")
    window.receiver.pose_latest = PoseResult(source, (a, b))
    window.poll_once()
    window.stop_pose("A")
    image = window.image_view.image
    assert color_at(image, 90, 120) == (30, 40, 50)
    assert color_at(image, 30, 30) == (40, 110, 85)
    assert_axis_pixels(image, source, b)
    window.stop_pose("B")
    image = window.image_view.image
    assert color_at(image, 300, 220) == (30, 40, 50)
    assert color_at(image, 90, 120) == (30, 40, 50)
    assert "camera=fake · sequence=42 ·" in window.metadata_label.text()


def test_sam3_only_display_overlays_latest_camera_and_keeps_camera_moving(app):
    window = CameraWindow(sam3=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    try:
        window.prompt_inputs["A"].setText("A")
        window.apply_prompts()
        result = sam3_result(window._prompts.revision)
        window.receiver.latest = frame(sequence=42)
        window.receiver.sam3_latest = result
        window.poll_once()
        assert color_at(window.image_view.image, 300, 220) == (30, 40, 50)
        assert color_at(window.image_view.image, 30, 30) == (40, 110, 85)
        assert "camera=fake · sequence=42 ·" in window.metadata_label.text()
        assert "SAM3 overlay sequence=7" in window.metadata_label.text()
        window.receiver.latest = frame(sequence=43, value=(60, 70, 80))
        window.poll_once()
        assert color_at(window.image_view.image, 300, 220) == (60, 70, 80)
        assert color_at(window.image_view.image, 30, 30) == (58, 128, 103)
        assert "camera=fake · sequence=43 ·" in window.metadata_label.text()
    finally:
        window.close()
        app.processEvents()


def test_sam3_source_frame_is_fallback_before_first_camera_frame(app):
    window = CameraWindow(sam3=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    try:
        window.prompt_inputs["A"].setText("A")
        window.apply_prompts()
        window.receiver.sam3_latest = sam3_result(window._prompts.revision)
        window.poll_once()
        assert color_at(window.image_view.image, 300, 220) == (150, 150, 150)
        assert "camera=fake · sequence=7 ·" in window.metadata_label.text()
    finally:
        window.close()
        app.processEvents()


def test_camera_keeps_advancing_during_sparse_pose_updates(window):
    objects = (pose(window, "A"), pose(window, "B"))
    for index in range(30):
        sequence = 100 + index
        current = frame(sequence=sequence, value=(index, 50, 60))
        window.receiver.latest = current
        if index % 10 == 0:
            window.receiver.pose_latest = PoseResult(frame(sequence=index, value=(200, 200, 200)), objects)
        window.poll_once()
        image = window.image_view.image
        assert color_at(image, 300, 220) == (index, 50, 60)
        assert f"camera=fake · sequence={sequence} ·" in window.metadata_label.text()
        for item in objects:
            assert_axis_pixels(image, current, item)


def test_mixed_camera_sam3_and_pose_updates_render_rgb_once_per_poll(window, monkeypatch):
    calls = []
    display_frame = window._display_frame

    def capture_display(*args, **kwargs):
        calls.append(args[0])
        return display_frame(*args, **kwargs)

    monkeypatch.setattr(window, "_display_frame", capture_display)
    current = frame(sequence=100)
    window.receiver.latest = current
    window.receiver.sam3_latest = sam3_result(window._prompts.revision, sequence=80)
    window.receiver.pose_latest = PoseResult(frame(sequence=60), (pose(window, "A"), pose(window, "B")))
    window.poll_once()
    assert len(calls) == 1 and calls[0] is current
    assert "camera=fake · sequence=100 ·" in window.metadata_label.text()
    assert "FP pose sequence=60" in window.metadata_label.text()
    assert "SAM3 overlay sequence=80" in window.metadata_label.text()
