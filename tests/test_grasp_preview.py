"""Canonical gripper previews preserve metric coordinates and recorded provenance."""
import hashlib

import numpy as np
import pytest

from grasp_workbench.ui.grasp_preview import (
    asset_identity, grasp_axis_points, load_gripper_preview, mesh_binding, transform_points,
)


@pytest.fixture
def descriptor(tmp_path):
    pytest.importorskip("trimesh")
    config = tmp_path / "gripper.yaml"
    config.write_text("base_rotation: [1, 2, 3]\n", encoding="utf-8")
    visual = tmp_path / "vis_mesh.obj"
    visual.write_text("v 0.01 0.02 0.03\nv 0.04 0.02 0.03\nv 0.01 0.06 0.03\nf 1 2 3\n", encoding="utf-8")
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {"config_path": str(config), "config_sha256": sha(config),
            "base_rotation": [0, 0, 180],
            "files": [{"relative_path": visual.name, "sha256": sha(visual)}]}


def test_identity_uses_catalog_or_exact_geometry_not_prompt_slot(tmp_path):
    path = tmp_path / "object.ply"
    canonical = mesh_binding(path, 0.001)
    assert canonical == (str(path.resolve()), 0.001)
    assert asset_identity(path, 0.001) == asset_identity(tmp_path / "." / path.name, 0.001)
    assert asset_identity(path, 0.001) != asset_identity(path, 1.0)
    assert asset_identity(path, 0.001) != asset_identity(tmp_path / "other.ply", 0.001)
    catalog = ({"asset_id": "known-cup", "mesh_path": str(path), "mesh_scale_to_m": 0.001},)
    assert asset_identity(path, 0.001, catalog) == "known-cup"


def test_preview_geometry_is_already_canonical_metres_no_base_rotation_or_recentering(descriptor):
    mesh, note = load_gripper_preview(descriptor)
    assert mesh is not None and "canonical" in note
    expected = np.array([[0.01, 0.02, 0.03], [0.04, 0.02, 0.03], [0.01, 0.06, 0.03]])
    np.testing.assert_allclose(sorted(map(tuple, mesh.vertices)), sorted(map(tuple, expected)), atol=1e-7)
    pose = np.eye(4)
    pose[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    pose[:3, 3] = (0.1, 0.2, 0.3)
    np.testing.assert_allclose(transform_points(mesh.vertices, pose), mesh.vertices @ pose[:3, :3].T + pose[:3, 3])
    axes = grasp_axis_points(pose)
    np.testing.assert_allclose(axes[0], (0.1, 0.2, 0.3))
    np.testing.assert_allclose(axes[1], (0.1, 0.23, 0.3))


@pytest.mark.parametrize("change", ["config", "geometry", "missing", "unrecorded"])
def test_changed_or_missing_descriptor_keeps_axes_only(descriptor, change):
    from pathlib import Path

    if change == "config":
        Path(descriptor["config_path"]).write_text("changed", encoding="utf-8")
    elif change == "geometry":
        (Path(descriptor["config_path"]).parent / "vis_mesh.obj").write_text("changed", encoding="utf-8")
    elif change == "missing":
        descriptor = None
    else:
        descriptor["files"] = []
    mesh, note = load_gripper_preview(descriptor)
    assert mesh is None and "axes only" in note
