"""Pose plot mesh loading matches FP geometry without loading any models."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from grasp_workbench.ui.mesh_preview import load_pose_mesh


def triangle(trimesh):
    return trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]],
        vertex_colors=[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], process=False,
    )


def sorted_rows(array):
    array = np.asarray(array)
    return array[np.lexsort(tuple(array[:, index] for index in reversed(range(array.shape[1]))))]


def test_ply_applies_scale_once_preserves_rgb_and_does_not_modify_file_or_source(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    asset = triangle(trimesh)
    vertices, faces, colors = asset.vertices.copy(), asset.faces.copy(), asset.visual.vertex_colors.copy()
    path = tmp_path / "millimetre.ply"
    asset.export(path)
    original_file = path.read_bytes()
    mesh = load_pose_mesh(path, .001)
    np.testing.assert_allclose(mesh.vertices, vertices * .001)
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(mesh.colors, colors[:, :3])
    assert mesh.vertices.dtype == np.float32 and mesh.faces.dtype == np.uint32
    assert mesh.colors.dtype == np.uint8 and mesh.vertices.flags.owndata and mesh.colors.flags.owndata
    assert path.read_bytes() == original_file
    np.testing.assert_array_equal(asset.vertices, vertices)
    np.testing.assert_array_equal(asset.faces, faces)
    np.testing.assert_array_equal(asset.visual.vertex_colors, colors)


def test_glb_scene_transforms_and_scaled_combined_geometry_match_foundationpose(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    asset = triangle(trimesh)
    scene = trimesh.Scene()
    transform = np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]])
    scene.add_geometry(asset, node_name="rotated", transform=transform)
    scene.add_geometry(asset.copy(), node_name="translated", transform=trimesh.transformations.translation_matrix([1, 2, 3]))
    path = tmp_path / "scene.glb"
    scene.export(path)
    original_file = path.read_bytes()
    mesh = load_pose_mesh(path, .1)
    expected = trimesh.load(path, force="mesh")
    expected.apply_scale(.1)
    np.testing.assert_allclose(mesh.vertices, expected.vertices)
    np.testing.assert_array_equal(mesh.faces, expected.faces)
    np.testing.assert_array_equal(mesh.colors, expected.visual.vertex_colors[:, :3])
    coordinates = [[4, 5, 6], [4, 6, 6], [3, 5, 6], [1, 2, 3], [2, 2, 3], [1, 3, 3]]
    np.testing.assert_allclose(sorted_rows(mesh.vertices), sorted_rows(np.asarray(coordinates) * .1))
    assert path.read_bytes() == original_file and len(mesh.faces) == 2


def test_obj_geometry_uses_the_same_scale_as_foundationpose(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    path = tmp_path / "object.obj"
    triangle(trimesh).export(path)
    mesh = load_pose_mesh(path, .25)
    expected = trimesh.load(path, force="mesh")
    expected.apply_scale(.25)
    np.testing.assert_allclose(mesh.vertices, expected.vertices)
    np.testing.assert_array_equal(mesh.faces, expected.faces)


def test_textured_glb_loads_geometry_without_baking_texture_or_vertex_colors(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    image = pytest.importorskip("PIL.Image")
    asset = triangle(trimesh)
    asset.visual = trimesh.visual.texture.TextureVisuals(
        uv=[[0, 0], [1, 0], [0, 1]], image=image.new("RGB", (2, 2), (200, 100, 50)),
    )
    path = tmp_path / "textured.glb"
    asset.export(path)
    loaded = trimesh.load(path, force="mesh")
    assert isinstance(loaded.visual, trimesh.visual.texture.TextureVisuals)
    assert not hasattr(loaded.visual, "vertex_colors")
    mesh = load_pose_mesh(path, .25)
    np.testing.assert_allclose(mesh.vertices, loaded.vertices * .25)
    np.testing.assert_array_equal(mesh.faces, loaded.faces)
    assert mesh.colors is None


def test_large_ply_is_surface_simplified_to_requested_face_budget(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("fast_simplification")
    asset = trimesh.creation.icosphere(subdivisions=3)
    asset.visual.vertex_colors = [25, 50, 75, 255]
    path = tmp_path / "dense.ply"
    asset.export(path)
    original_file = path.read_bytes()
    mesh = load_pose_mesh(path, .1, max_faces=100)
    assert 0 < len(mesh.faces) <= 100 < len(asset.faces)
    np.testing.assert_array_equal(mesh.colors, np.broadcast_to([25, 50, 75], mesh.colors.shape))
    np.testing.assert_allclose(mesh.vertices.min(axis=0), asset.vertices.min(axis=0) * .1, atol=.01)
    np.testing.assert_allclose(mesh.vertices.max(axis=0), asset.vertices.max(axis=0) * .1, atol=.01)
    assert path.read_bytes() == original_file


def test_loader_does_not_import_models_and_missing_trimesh_is_not_hidden(tmp_path):
    script = """
import sys
for name in ('torch', 'sam3d_objects', 'estimater', 'trimesh'):
    sys.modules[name] = None
from grasp_workbench.ui.mesh_preview import load_pose_mesh
try:
    load_pose_mesh('unused.ply', 1.)
except ImportError:
    pass
else:
    raise AssertionError('Missing trimesh must propagate')
assert all(sys.modules[name] is None for name in ('torch', 'sam3d_objects', 'estimater'))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=10)
    assert result.returncode == 0, result.stderr
