"""Display-only mesh geometry follows each FoundationPose session."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult
from grasp_workbench.ui.camera_window import CameraWindow
from test_camera_ui import FakeReceiver, completed_mesh, sam3_result


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def mesh_path(tmp_path):
    pytest.importorskip("trimesh")
    path = tmp_path / "uncentered.ply"
    path.write_text("""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar int vertex_indices
end_header
10 20 30
14 20 30
10 26 30
10 20 38
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
""", encoding="utf-8")
    return path


@pytest.fixture
def window(app, mesh_path):
    window = CameraWindow(foundationpose=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    for key in ("A", "B"):
        window.prompt_inputs[key].setText(key)
        window.pose_mesh_inputs[key].setText(str(mesh_path))
        window.pose_scale_inputs[key].setValue(0.01)
    window.apply_prompts()
    window.receiver.sam3_latest = sam3_result(window._prompts.revision, count=1, include_b=True)
    window.poll_once()
    yield window
    window.receiver.alive = False
    window.close()
    app.processEvents()


def tracking_pose(window, key, *, x=0.4):
    target = window._pose_targets[key]
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    matrix[:3, 3] = (x, -0.2, 1)
    return ObjectPose(key, target.session_id, target.mesh_path, "tracking", matrix)


def receive_poses(window, *objects):
    window.receiver.pose_latest = PoseResult(window._sam3_result.frame, tuple(objects))
    window.poll_once()


def artist_triangles(artist):
    # Matplotlib 3.11 stores faces directly; earlier supported versions use a homogeneous vector.
    if hasattr(artist, "_faces"):
        return np.asarray(artist._faces)
    return artist._vec[:3].T.reshape(-1, 3, 3)


def visible_mesh(window, key):
    artist = window._pose_mesh_artists.get(key)
    return artist is not None and artist.axes is window.axes and artist.get_visible()


def sorted_vertices(vertices):
    return np.asarray(sorted(map(tuple, vertices)))


def test_mesh_uses_original_coordinates_scaled_once_then_camera_pose(window, mesh_path):
    file_before = mesh_path.read_bytes()
    window.start_pose("A")
    target, = window.receiver.pose_controls[-1].targets
    assert target.mesh_path == str(mesh_path) and target.mesh_scale_to_m == 0.01
    session_id, mesh = window._pose_meshes["A"]
    assert session_id == target.session_id
    original = np.array([[10, 20, 30], [14, 20, 30], [10, 26, 30], [10, 20, 38]], dtype=np.float32)
    np.testing.assert_allclose(sorted_vertices(mesh.vertices), sorted_vertices(original * 0.01), atol=1e-7)
    vertices_before = mesh.vertices.copy()
    item = tracking_pose(window, "A")
    receive_poses(window, item)
    artist = window._pose_mesh_artists["A"]
    transformed = mesh.vertices @ item.camera_from_object[:3, :3].T + item.camera_from_object[:3, 3]
    np.testing.assert_allclose(artist_triangles(artist), transformed[mesh.faces], atol=1e-7)
    np.testing.assert_array_equal(mesh.vertices, vertices_before)
    assert mesh_path.read_bytes() == file_before
    assert visible_mesh(window, "A") and len(window.axes.lines) == 3


def test_pose_updates_reuse_cached_mesh_artist_and_keep_view_rotation(window, monkeypatch):
    import grasp_workbench.ui.mesh_preview as module

    calls = []
    original = module.load_pose_mesh

    def record_load(path, scale, **kwargs):
        calls.append((path, scale))
        return original(path, scale, **kwargs)

    monkeypatch.setattr(module, "load_pose_mesh", record_load)
    window.start_pose("A")
    receive_poses(window, tracking_pose(window, "A"))
    artist = window._pose_mesh_artists["A"]
    local_mesh = window._pose_meshes["A"][1]
    window.axes.view_init(elev=17, azim=-23)
    for x in (0.42, 0.44, 0.46):
        item = tracking_pose(window, "A", x=x)
        if x == 0.44:
            item.camera_from_object[:3, :3] = np.eye(3)
        receive_poses(window, item)
        assert window._pose_mesh_artists["A"] is artist
        assert window._pose_meshes["A"][1] is local_mesh
        assert (window.axes.elev, window.axes.azim) == (17, -23)
        expected = local_mesh.vertices @ item.camera_from_object[:3, :3].T + item.camera_from_object[:3, 3]
        np.testing.assert_allclose(artist_triangles(artist), expected[local_mesh.faces], atol=1e-7)
    assert calls == [(window._pose_targets["A"].mesh_path, 0.01)]
    assert len(window.axes.collections) == 1 and len(window.axes.lines) == 3


def test_partial_pose_results_keep_other_mesh_and_recovery_hides_only_affected_slot(window):
    for key in ("A", "B"):
        window.start_pose(key)
    a, b = tracking_pose(window, "A"), tracking_pose(window, "B", x=-0.4)
    receive_poses(window, a, b)
    b_artist = window._pose_mesh_artists["B"]
    local_a = window._pose_meshes["A"][1]
    receive_poses(window, tracking_pose(window, "A", x=0.45))
    assert visible_mesh(window, "A") and visible_mesh(window, "B")
    assert window._pose_mesh_artists["B"] is b_artist and len(window.axes.lines) == 6
    recovering = ObjectPose("A", a.session_id, a.mesh_path, "recovering", error="visual_mismatch:bbox")
    receive_poses(window, recovering)
    assert not visible_mesh(window, "A") and visible_mesh(window, "B")
    assert len(window.axes.lines) == 3
    receive_poses(window, tracking_pose(window, "A", x=0.5))
    assert visible_mesh(window, "A") and visible_mesh(window, "B")
    assert window._pose_meshes["A"][1] is local_a
    assert window._pose_mesh_artists["B"] is b_artist and len(window.axes.lines) == 6


def test_stop_removes_only_its_mesh_and_axes(window):
    for key in ("A", "B"):
        window.start_pose(key)
    receive_poses(window, tracking_pose(window, "A"), tracking_pose(window, "B", x=-0.4))
    b_artist = window._pose_mesh_artists["B"]
    window.stop_pose("A")
    assert not visible_mesh(window, "A") and visible_mesh(window, "B")
    assert window._pose_mesh_artists["B"] is b_artist and len(window.axes.lines) == 3
    window.stop_pose("B")
    assert not visible_mesh(window, "A") and not visible_mesh(window, "B")
    assert len(window.axes.lines) == 0


def test_preview_load_error_is_visible_but_does_not_block_pose_tracking(window, tmp_path):
    broken = tmp_path / "broken.ply"
    broken.write_text("invalid ply file", encoding="utf-8")
    window.pose_mesh_inputs["A"].setText(str(broken))
    window.start_pose("A")
    target, = window.receiver.pose_controls[-1].targets
    assert target.mesh_path == str(broken)
    assert "preview" in (window.pose_status.text() + window.pose_status.toolTip()).lower()
    receive_poses(window, tracking_pose(window, "A"))
    assert len(window.axes.lines) == 3 and not visible_mesh(window, "A")
    assert window._pose_objects["A"].status == "tracking"


def test_mesh_only_plot_keeps_local_preview_without_pose_targets(app):
    window = CameraWindow(sam3d=True, foundationpose=True, receiver_factory=FakeReceiver)
    window.timer.stop()
    try:
        window.prompt_inputs["A"].setText("cup")
        window.apply_prompts()
        window.receiver.sam3_latest = sam3_result(window._prompts.revision, count=1)
        window.poll_once()
        window.generate_mesh("A")
        result = completed_mesh(window.receiver.mesh_requests[-1])
        window.receiver.mesh_events.append(result)
        window.poll_once()
        assert not window._pose_targets
        assert window.axes.get_title() == "A mesh · local preview"
        assert len(window.axes.collections) == 1
        np.testing.assert_allclose(artist_triangles(window.axes.collections[0]), result.mesh.vertices[result.mesh.faces])
    finally:
        window.receiver.alive = False
        window.close()
        app.processEvents()
