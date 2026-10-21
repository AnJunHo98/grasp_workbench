"""AprilCube display-only controls and latest-camera RGB composition."""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.aprilcube import AprilCubeOverlay
from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult
from grasp_workbench.ui.camera_window import CameraWindow


class FakeReceiver:
    def __init__(self, service, **options):
        self.options, self.controls = options, []
        self.latest = self.pose_latest = self.overlay_latest = None
        self.alive = False

    def start(self):
        self.alive = True

    def stop(self):
        self.alive = False

    def join(self):
        pass

    def is_alive(self):
        return self.alive

    def snapshot(self):
        return {"ready": True, "error": None, "received_count": 1, "last_received_monotonic_s": None}

    def send_aprilcube_control(self, control):
        self.controls.append(control)

    def take_latest(self):
        latest, self.latest = self.latest, None
        return latest

    def take_pose_latest(self):
        latest, self.pose_latest = self.pose_latest, None
        return latest

    def take_aprilcube_overlay_latest(self):
        latest, self.overlay_latest = self.overlay_latest, None
        return latest


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    window = CameraWindow(aprilcube=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    yield window
    window.close()
    app.processEvents()


def frame(sequence=42, value=40):
    return CameraFrame(np.full((30, 40, 3), value, dtype=np.uint8), np.ones((30, 40), dtype=np.float32), {
        "camera_id": "fake", "sequence": sequence,
        "intrinsics": {"fx": 40., "fy": 40., "cx": 20., "cy": 15.},
    })


def pose(window, key, *, missing=False, source="aprilcube"):
    target = window._pose_targets[key]
    transform = np.eye(4)
    transform[:3, 3] = (-.1 if key == "A" else .1, 0, 1)
    return ObjectPose(key, target.session_id, target.mesh_path, "unavailable" if missing else "tracking",
                      None if missing else transform, pose_source=source)


def color_at(window, x=0, y=0):
    return window.image_view.image.pixelColor(x, y).getRgb()[:3]


def test_aprilcube_direct_window_overrides_normal_flags_without_model_widgets(app, tmp_path):
    window = CameraWindow(aprilcube=True, sam3=True, sam3d=True, foundationpose=True, vlm=True,
                          graspgenx=True, grasp_selection=True, policy_catalog=str(tmp_path / "missing.json"),
                          receiver_factory=FakeReceiver)
    window.timer.stop()
    try:
        assert window.receiver.options == {"aprilcube": True}
        assert not any((window.sam3, window.sam3d, window.foundationpose, window.vlm,
                        window.graspgenx, window.grasp_selection))
        assert not hasattr(window, "prompt_inputs") and not hasattr(window, "vlm_button")
        assert not window.pose_start_buttons and not window.pose_seed_inputs
        assert len(window.receiver.controls[-1].targets) == 2
        for target in window.receiver.controls[-1].targets:
            assert target.object_id in {"A", "B"} and target.mesh_path == ""
            assert target.mesh_scale_to_m == 1
            np.testing.assert_array_equal(target.cube_from_object, np.eye(4))
        assert window.pose_mesh_choices["A"].count() > 1
        assert "AprilCube" in window.pose_status.text()
    finally:
        window.close()
        app.processEvents()


def test_control_apply_publishes_object_offset_rpy_and_scaled_plot_mesh_once(window, tmp_path):
    pytest.importorskip("trimesh")
    path = tmp_path / "mesh.obj"
    path.write_text("v 10 20 30\nv 14 20 30\nv 10 26 30\nf 1 2 3\n", encoding="utf-8")
    original = path.read_bytes()
    old_a, old_b = window._pose_targets["A"], window._pose_targets["B"]
    window.pose_mesh_inputs["A"].setText(str(path))
    window.pose_scale_inputs["A"].setValue(.01)
    for field, value in zip(window.aprilcube_xyz_inputs["A"], (.1, -.2, .3)):
        field.setValue(value)
    for field, value in zip(window.aprilcube_rpy_inputs["A"], (0, 0, 90)):
        field.setValue(value)
    target = window.apply_aprilcube_target("A")
    assert target.session_id != old_a.session_id and window._pose_targets["B"] is old_b
    assert target.mesh_path == str(path) and target.mesh_scale_to_m == .01
    np.testing.assert_allclose(target.cube_from_object[:3, 3], (.1, -.2, .3))
    np.testing.assert_allclose(target.cube_from_object[:3, :3], ((0, -1, 0), (1, 0, 0), (0, 0, 1)), atol=1e-10)
    assert window.receiver.controls[-1].targets == (target, old_b)
    local_mesh = window._pose_meshes["A"][1]
    expected = np.array(((.1, .2, .3), (.14, .2, .3), (.1, .26, .3)))
    np.testing.assert_allclose(sorted(map(tuple, local_mesh.vertices)), sorted(map(tuple, expected)), atol=1e-7)
    item = pose(window, "A")
    window.receiver.pose_latest = PoseResult(frame(), (item, pose(window, "B")))
    window.poll_once()
    artist = window._pose_mesh_artists["A"]
    triangles = np.asarray(artist._faces) if hasattr(artist, "_faces") else artist._vec[:3].T.reshape(-1, 3, 3)
    expected = local_mesh.vertices @ item.camera_from_object[:3, :3].T + item.camera_from_object[:3, 3]
    np.testing.assert_allclose(triangles, expected[local_mesh.faces])
    assert len(window.axes.lines) == 6 and path.read_bytes() == original
    assert "AprilCube" in window.axes.get_title()


def test_worker_overlay_composes_once_with_latest_raw_camera_and_never_old_pose_rgb(window, monkeypatch):
    window._pose_targets["A"].mesh_path = "selected-mesh.glb"
    source = frame(42, value=40)
    layer = np.zeros((30, 40, 4), dtype=np.uint8)
    layer[0, 0] = (200, 100, 0, 128)
    overlay = AprilCubeOverlay({"sequence": 7}, layer)
    calls = []
    display = window._display_frame

    def record(*args, **kwargs):
        calls.append(args[0])
        return display(*args, **kwargs)

    monkeypatch.setattr(window, "_display_frame", record)
    window.receiver.latest = source
    window.receiver.pose_latest = PoseResult(frame(3, value=230), (pose(window, "A"), pose(window, "B")))
    window.receiver.overlay_latest = overlay
    window.poll_once()
    assert calls == [source]
    expected = (40 * (1 - 128 / 255) + np.array((200, 100, 0)) * 128 / 255).astype(np.uint8)
    assert color_at(window) == tuple(expected) and color_at(window, 39, 29) == (40, 40, 40)
    np.testing.assert_array_equal(source.rgb, 40)
    np.testing.assert_array_equal(layer[0, 0], (200, 100, 0, 128))
    assert "camera=fake · sequence=42" in window.metadata_label.text()
    assert "AprilCube pose sequence=3" in window.metadata_label.text()
    assert "AprilCube overlay sequence=7" in window.metadata_label.text()
    window.receiver.latest = frame(43, value=80)
    window.poll_once()
    assert color_at(window, 39, 29) == (80, 80, 80)
    expected = (80 * (1 - 128 / 255) + np.array((200, 100, 0)) * 128 / 255).astype(np.uint8)
    assert color_at(window) == tuple(expected)
    snapshot = window.capture_snapshot()
    np.testing.assert_array_equal(snapshot.frame.rgb, 80)
    assert color_at(window) == (80, 80, 80)


def test_transparent_layer_and_unavailable_pose_clear_display_and_no_pose_rgb_fallback(window):
    window._pose_targets["A"].mesh_path = "selected-mesh.glb"
    window.receiver.pose_latest = PoseResult(frame(2, value=230), (pose(window, "A"),))
    window.poll_once()
    assert window.image_view.image is None
    assert len(window.axes.lines) == 3
    window.receiver.latest = frame()
    window.receiver.overlay_latest = AprilCubeOverlay({}, np.full((30, 40, 4), 255, dtype=np.uint8))
    window.poll_once()
    assert color_at(window) == (255, 255, 255)
    window.receiver.overlay_latest = AprilCubeOverlay({}, np.zeros((30, 40, 4), dtype=np.uint8))
    window.receiver.pose_latest = PoseResult(frame(3, value=230), (pose(window, "A", missing=True),))
    window.poll_once()
    assert color_at(window) == (40, 40, 40) and not window.axes.lines


def axes_frame(sequence=42, value=40, **calibration):
    return CameraFrame(np.full((120, 160, 3), value, dtype=np.uint8),
                       np.ones((120, 160), dtype=np.float32), {
                           "camera_id": "fake", "sequence": sequence,
                           "intrinsics": {"fx": 200., "fy": 200., "cx": 80., "cy": 60., **calibration},
                       })


def test_axes_only_uses_latest_plot_pose_not_stale_rgba_over_latest_camera(window, monkeypatch):
    source = axes_frame(42)
    layer = np.full((120, 160, 4), (200, 100, 0, 255), dtype=np.uint8)
    window.receiver.latest = source
    window.receiver.overlay_latest = AprilCubeOverlay({"sequence": 7}, layer)
    first = PoseResult(axes_frame(3, value=230), (pose(window, "A"), pose(window, "B")))
    window.receiver.pose_latest = first
    calls = []
    display = window._display_frame
    draws = []
    draw = window.image_view._draw_image

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return display(*args, **kwargs)

    def record_draw():
        draws.append(window.image_view.image.cacheKey())
        return draw()

    monkeypatch.setattr(window, "_display_frame", record)
    monkeypatch.setattr(window.image_view, "_draw_image", record_draw)
    window.poll_once()
    assert len(calls) == len(draws) == 1
    assert calls[0][0][0] is source and calls[0][0][2] == first.objects
    assert color_at(window) == (40, 40, 40)
    assert color_at(window, 66, 60) == (255, 0, 0)
    assert color_at(window, 60, 66) == (0, 255, 0)
    assert color_at(window, 106, 60) == (255, 0, 0)
    for item in first.objects:
        x, y, z = item.camera_from_object[:3, 3]
        np.testing.assert_allclose(window._pose_axis_artists[item.object_id][0].get_data_3d(),
                                   ((x, x + .05), (y, y), (z, z)))
    assert "camera=fake · sequence=42" in window.metadata_label.text()
    assert "AprilCube pose sequence=3" in window.metadata_label.text()
    assert "AprilCube overlay sequence=" not in window.metadata_label.text()
    np.testing.assert_array_equal(source.rgb, 40)
    np.testing.assert_array_equal(layer, np.full_like(layer, (200, 100, 0, 255)))

    # The overlay worker has not emitted anything new. Pose-only arrival must
    # nevertheless move both the RGB axes and the existing 3D artists.
    moved = pose(window, "A")
    moved.camera_from_object[:3, 3] = (-.2, 0, 1)
    latest = PoseResult(axes_frame(9, value=250), (moved, pose(window, "B", missing=True)))
    window.receiver.pose_latest = latest
    window.poll_once()
    assert len(calls) == len(draws) == 2 and calls[-1][0][0] is source
    assert calls[-1][0][2] == latest.objects
    assert color_at(window, 46, 60) == (255, 0, 0)
    assert color_at(window, 66, 60) == (40, 40, 40)
    assert color_at(window, 106, 60) == (40, 40, 40)
    np.testing.assert_allclose(window._pose_axis_artists["A"][0].get_data_3d(),
                               ((-.2, -.15), (0, 0), (1, 1)))
    assert "B" not in window._pose_axis_artists
    assert "camera=fake · sequence=42" in window.metadata_label.text()
    assert "AprilCube pose sequence=9" in window.metadata_label.text()
    assert "AprilCube overlay sequence=" not in window.metadata_label.text()

    window.receiver.latest = axes_frame(43, value=80)
    window.poll_once()
    assert len(calls) == len(draws) == 3
    assert color_at(window, 159, 119) == (80, 80, 80)
    assert color_at(window, 46, 60) == (255, 0, 0)
    assert "camera=fake · sequence=43" in window.metadata_label.text()
    assert "AprilCube pose sequence=9" in window.metadata_label.text()
    snapshot = window.capture_snapshot()
    np.testing.assert_array_equal(snapshot.frame.rgb, 80)
    assert color_at(window, 46, 60) == (80, 80, 80)
    assert "고정 snapshot" in window.metadata_label.text()


def test_axes_only_without_pose_ignores_stale_rgba_and_unavailable_clears_axes(window):
    source = axes_frame()
    window.receiver.latest = source
    window.receiver.overlay_latest = AprilCubeOverlay({"sequence": 7},
                                                       np.full((120, 160, 4), 255, dtype=np.uint8))
    window.poll_once()
    for x, y in ((0, 0), (66, 60), (60, 66), (106, 60), (159, 119)):
        assert color_at(window, x, y) == (40, 40, 40)
    assert "AprilCube overlay sequence=" not in window.metadata_label.text()
    assert not window.axes.lines
    window.receiver.pose_latest = PoseResult(axes_frame(3), (pose(window, "A"),))
    window.poll_once()
    assert color_at(window, 66, 60) == (255, 0, 0)
    window.receiver.pose_latest = PoseResult(axes_frame(4), (pose(window, "A", missing=True),))
    window.poll_once()
    assert color_at(window, 66, 60) == (40, 40, 40) and not window.axes.lines


def test_axes_only_skips_complete_rgb_axes_when_an_endpoint_is_behind_camera(window):
    window.receiver.latest = axes_frame()
    item = pose(window, "A")
    item.camera_from_object[:3, 3] = (0, 0, .025)
    item.camera_from_object[:3, :3] = ((0, 0, 1), (0, 1, 0), (-1, 0, 0))
    window.receiver.pose_latest = PoseResult(axes_frame(3), (item,))
    window.poll_once()
    # The origin is visible, but the X endpoint crosses the camera plane.
    # Like the existing overlay renderer, omit the whole RGB axis drawing.
    for x, y in ((0, 0), (80, 60), (120, 60), (80, 90), (159, 119)):
        assert color_at(window, x, y) == (40, 40, 40)
    assert len(window._pose_axis_artists["A"]) == 3
    assert "AprilCube pose sequence=3" in window.metadata_label.text()


@pytest.mark.parametrize(("model", "coefficients", "red_x"), (
    ("zed_rectified", (4., 0., 0., 0., 0.), 126),
    ("brown_conrady", (0., 0., 0., 0., 0.), 126),
    ("brown_conrady", (4., 0., 0., 0., 0.), 139),
))
def test_axes_only_projection_respects_rectified_and_distorted_camera(window, model, coefficients, red_x):
    pytest.importorskip("cv2")
    window.receiver.latest = axes_frame(distortion_model=model, coefficients=coefficients)
    item = pose(window, "A")
    item.camera_from_object[:3, 3] = (.2, 0, 1)
    window.receiver.pose_latest = PoseResult(axes_frame(3), (item,))
    window.poll_once()
    assert color_at(window, red_x, 60) == (255, 0, 0)
    if red_x == 126:
        assert color_at(window, 136, 60) == (40, 40, 40)


def test_pose_source_and_control_session_bindings_reject_unrelated_result(window):
    item = pose(window, "A", source="foundationpose")
    assert not window._accept_pose_result(PoseResult(frame(), (item,)))
    item.pose_source, item.session_id = "aprilcube", "obsolete-session"
    assert not window._accept_pose_result(PoseResult(frame(), (item,)))
    assert not window._pose_objects


def test_aprilcube_ui_cli_discards_normal_mode_flags(app, monkeypatch):
    from grasp_workbench.ui import camera_window
    from grasp_workbench.workers import worker_camera_ui

    calls = []
    receiver = FakeReceiver("test")

    def create_window(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(show=lambda: None, exit_code=0, receiver=receiver, close=lambda: None)

    monkeypatch.setattr(camera_window, "CameraWindow", create_window)
    monkeypatch.setattr(QtWidgets.QApplication, "exec", lambda self: 0)
    assert worker_camera_ui.main(["--aprilcube", "--sam3", "--sam3d", "--foundationpose", "--vlm",
                                  "--graspgenx", "--grasp-selection", "--policy-catalog", "missing.json"]) == 0
    assert calls[0][1] == {"aprilcube": True}
