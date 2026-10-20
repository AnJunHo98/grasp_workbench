"""Display simplification keeps a surface and runs off the Qt/main thread."""
from threading import Event, get_ident
from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest

from grasp_workbench.ipc.sam3d import MeshGeometry, MeshResult
from grasp_workbench.ui.mesh_preview import preview_mesh
from grasp_workbench.ui.receiver import LatestCameraReceiver


def icosphere(subdivisions=3):
    t = (1 + np.sqrt(5)) / 2
    vertices = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                         [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(subdivisions):
        points, mids, refined = list(vertices), {}, []

        def midpoint(a, b):
            key = tuple(sorted((a, b)))
            if key not in mids:
                value = (vertices[a] + vertices[b]) / 2
                mids[key] = len(points)
                points.append(value / np.linalg.norm(value))
            return mids[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        vertices, faces = np.asarray(points), np.asarray(refined)
    colors = np.arange(len(vertices) * 3, dtype=np.uint8).reshape(-1, 3)
    return MeshGeometry(vertices.astype(np.float32), faces.astype(np.uint32), colors)


def area(mesh):
    triangles = mesh.vertices[mesh.faces]
    return np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2


def test_small_mesh_is_unchanged_and_does_not_import_simplifier(monkeypatch):
    mesh = icosphere(0)
    monkeypatch.setitem(sys.modules, "fast_simplification", None)
    assert preview_mesh(mesh) is mesh


def test_reduced_sphere_is_closed_and_covers_surface_not_sparse_deleted_faces():
    pytest.importorskip("fast_simplification")
    mesh = icosphere()
    vertices, faces, colors = mesh.vertices.copy(), mesh.faces.copy(), mesh.colors.copy()
    preview = preview_mesh(mesh, max_faces=120)
    assert len(preview.faces) <= 120 < len(mesh.faces)
    assert preview.vertices.dtype == np.float32 and preview.faces.dtype == np.uint32
    assert area(preview) > 0.9 * area(mesh)
    np.testing.assert_allclose(preview.vertices.min(axis=0), mesh.vertices.min(axis=0), atol=0.08)
    np.testing.assert_allclose(preview.vertices.max(axis=0), mesh.vertices.max(axis=0), atol=0.08)
    edges = np.sort(np.concatenate((preview.faces[:, [0, 1]], preview.faces[:, [1, 2]], preview.faces[:, [2, 0]])), axis=1)
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    expected = np.broadcast_to(colors.mean(axis=0).astype(np.uint8), preview.colors.shape)
    np.testing.assert_array_equal(preview.colors, expected)
    for original, unchanged in ((mesh.vertices, vertices), (mesh.faces, faces), (mesh.colors, colors)):
        np.testing.assert_array_equal(original, unchanged)


def test_reduced_planar_grid_preserves_area_and_bounds_without_stride_holes():
    pytest.importorskip("fast_simplification")
    size = 35
    x, y = np.meshgrid(np.linspace(0, 2, size), np.linspace(0, 3, size))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(size * size))).astype(np.float32)
    a = (np.arange(size - 1)[:, None] * size + np.arange(size - 1)).ravel()
    faces = np.vstack((np.column_stack((a, a + 1, a + size)),
                       np.column_stack((a + 1, a + size + 1, a + size)))).astype(np.uint32)
    mesh = MeshGeometry(vertices, faces)
    preview = preview_mesh(mesh, max_faces=80)
    assert len(preview.faces) <= 80 and preview.colors is None
    assert area(preview) > 0.95 * area(mesh)
    np.testing.assert_allclose(preview.vertices.min(axis=0), [0, 0, 0], atol=0.01)
    np.testing.assert_allclose(preview.vertices.max(axis=0), [2, 3, 0], atol=0.01)


@pytest.mark.parametrize("change_prompts", [False, True])
def test_receiver_prepares_preview_on_owner_thread_outside_lock_and_preserves_full_mesh(monkeypatch, change_prompts):
    import grasp_workbench.ui.mesh_preview as module
    from grasp_workbench.ipc.sam3 import PromptSet

    mesh, entered, release = icosphere(0), Event(), Event()
    result = MeshResult("request", "asset", "A", "old", "done", mesh=mesh, mesh_path="full.ply")
    calls = []

    def prepare(original):
        assert original is mesh
        calls.append(get_ident())
        entered.set()
        assert release.wait(timeout=2)
        return mesh

    monkeypatch.setattr(module, "preview_mesh", prepare)

    class Endpoint:
        def __init__(self, service):
            self.service, self.sent = service, False
            calls.append(get_ident())

        def publish(self, request):
            calls.append(get_ident())

        def update_connections(self):
            calls.append(get_ident())

        def receive_latest(self):
            calls.append(get_ident())
            if self.service.endswith("sam3d/results") and not self.sent:
                self.sent = True
                return result
            return None

        def close(self):
            calls.append(get_ident())

    receiver = LatestCameraReceiver("test", sam3d=True, subscriber_factory=Endpoint,
                                    prompt_publisher_factory=Endpoint, result_subscriber_factory=Endpoint,
                                    mesh_request_publisher_factory=Endpoint, mesh_result_subscriber_factory=Endpoint)
    receiver.submit_mesh(SimpleNamespace(request_id="request", revision="old"))
    receiver.start()
    try:
        assert entered.wait(timeout=2)
        # The UI can still read status while CPU simplification waits outside the lock.
        assert receiver.snapshot()["ready"]
        assert receiver.mesh_preview("A", "asset") is None
        if change_prompts:
            receiver.set_prompts(PromptSet("new", ()))
        release.set()
        deadline = time.monotonic() + 2
        while receiver._mesh_inflight is not None and time.monotonic() < deadline:
            time.sleep(0.001)
        assert receiver._mesh_inflight is None
        assert receiver.mesh_preview("A", "asset") is (None if change_prompts else mesh)
        assert receiver.mesh_preview("A", "different-asset") is None
        assert receiver.take_mesh_results()[0].mesh is mesh and result.mesh_path == "full.ply"
        receiver.set_prompts(PromptSet("after", ()))
        assert receiver.mesh_preview("A", "asset") is None
    finally:
        release.set()
        receiver.stop()
        receiver.join(timeout=2)
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert set(calls) == {receiver.ident} and receiver.ident != get_ident()
