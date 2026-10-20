"""Persistent grasp geometry does not depend on a camera frame or robot state."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.grasp_assets import (
    cache_key, grasp_identity, load_grasp_bundle, save_grasp_bundle, validate_arrays,
)


@pytest.fixture
def identity(tmp_path):
    mesh = tmp_path / "object.ply"
    mesh.write_text("synthetic-mesh")
    return grasp_identity(SimpleNamespace(asset_id="saved-object", mesh_path=str(mesh), mesh_scale_to_m=.01),
                          gripper_id="test-gripper", gripper_metadata={"config_sha256": "gripper-hash"},
                          model_metadata={"checkpoints": ["weights-hash"]}, inference_settings={"seed": 42})


def test_npz_bundle_roundtrip_preserves_object_local_order_scores_and_pool(tmp_path, identity):
    grasps = np.repeat(np.eye(4)[None], 3, axis=0)
    grasps[:, 0, 3] = [.1, -.2, .3]
    scores = np.array([.8, .2, .9])
    points = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    selected = np.array([2, 0])
    path = save_grasp_bundle(tmp_path / "grasps", identity, grasps[selected], scores[selected],
                             raw_poses=grasps, raw_scores=scores, sample_points=points, selected_indices=selected)
    assert Path(path).parent == tmp_path / "grasps" / cache_key(identity)
    decoded, confidence, metadata = load_grasp_bundle(path)
    np.testing.assert_array_equal(decoded, grasps[selected])
    np.testing.assert_array_equal(confidence, scores[selected])
    assert decoded.flags.owndata and confidence.flags.owndata
    assert metadata["pose_convention"] == "object_from_canonical_gripper"
    assert metadata["candidate_frame"] == "canonical_gripper_base"
    assert metadata["object_frame"] == "saved-object_body" and metadata["candidate_count"] == 2
    assert not metadata["scene_collision_checked"] and not metadata["robot_ik_checked"]
    assert not metadata["robot_command_published"]
    with np.load(Path(path).with_name("candidate_pool.npz"), allow_pickle=False) as pool:
        for key, value in (("grasps", grasps), ("scores", scores), ("sample_points", points), ("selected_indices", selected)):
            np.testing.assert_array_equal(pool[key], value)


@pytest.mark.parametrize("field", ["asset_id", "asset_geometry", "gripper_id", "gripper_metadata", "model_metadata", "inference_settings"])
def test_each_previous_cache_identity_component_changes_the_key(identity, field):
    changed = deepcopy(identity)
    changed[field] = "changed" if isinstance(changed[field], str) else {"changed": True}
    assert cache_key(changed) != cache_key(identity)


def test_file_content_and_scale_change_geometry_identity(tmp_path):
    mesh = tmp_path / "mesh.ply"
    mesh.write_text("first")
    value = SimpleNamespace(asset_id="object", mesh_path=mesh, mesh_scale_to_m=1.)
    options = {"gripper_id": "g", "gripper_metadata": {}, "model_metadata": {}, "inference_settings": {}}
    first = grasp_identity(value, **options)
    mesh.write_text("second")
    assert grasp_identity(value, **options)["asset_geometry"]["mesh_sha256"] != first["asset_geometry"]["mesh_sha256"]
    value.mesh_scale_to_m = .001
    assert grasp_identity(value, **options)["asset_geometry"]["mesh_scale_to_m"] == .001


def test_empty_candidates_are_supported_and_no_pickle_is_written(tmp_path, identity):
    path = save_grasp_bundle(tmp_path, identity, [], [], raw_poses=[], raw_scores=[],
                             sample_points=np.empty((0, 3)), selected_indices=[])
    poses, scores, metadata = load_grasp_bundle(path)
    assert poses.shape == (0, 4, 4) and scores.shape == (0,) and metadata["max_score"] is None


@pytest.mark.parametrize("poses,scores", [
    (np.eye(4), [1]), (np.full((1, 4, 4), np.nan), [1]), (np.eye(4)[None], []),
    (np.eye(4)[None], [np.nan]), (np.eye(4)[None], [1.1]), (np.eye(4)[None], [-.1]),
    (np.ones((1, 4, 4)), [.5]),
])
def test_previous_shape_rigid_pose_and_confidence_contract(poses, scores):
    with pytest.raises(ValueError):
        validate_arrays(poses, scores)
