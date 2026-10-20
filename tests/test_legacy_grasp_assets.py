"""Read the original Isaac-YAML bundles without changing their provenance/files."""
from copy import deepcopy
import json

import numpy as np
import pytest
import yaml

from grasp_workbench.grasp_assets import cache_key, load_grasp_bundle, sha256_file


@pytest.fixture
def legacy_bundle(tmp_path):
    directory = tmp_path / "saved-bundle"
    directory.mkdir()
    document = {"format": "isaac_grasp", "format_version": 1.0, "grasps": {
        "grasp_7": {"position": [.1, -.2, .3], "orientation": {"w": 1., "xyz": [0., 0., 0.]}, "confidence": .7},
        "grasp_2": {"position": [-.4, .5, .6], "orientation": {"w": 0., "xyz": [0., 0., 2.]}, "confidence": .9},
    }}
    candidates = directory / "grasps.yaml"
    candidates.write_text(yaml.safe_dump(document, sort_keys=False))
    identity = {"asset_id": "saved-object", "asset_geometry": {"mesh_sha256": "recorded-geometry-sha",
                "mesh_scale_to_m": .01, "object_frame": "saved_body", "length_unit": "metre"},
                "gripper_id": "ffw_sg2_rh_p12_rn", "gripper_metadata": {"config_sha256": "gripper-sha",
                "config": {"symmetric": True}, "files": [{"relative_path": "vis_mesh.obj", "sha256": "visual-sha"}]},
                "model_metadata": {"source_revision": "recorded-revision", "checkpoint_sha256": "weights-sha"},
                "inference_settings": {"num_grasps": 1000, "topk_num_grasps": 100, "seed": 42}}
    metadata = {"schema_version": 1, **identity, "cache_key": cache_key(identity),
                "object_frame": "saved_body", "candidate_frame": "canonical_gripper_base",
                "pose_convention": "object_from_canonical_gripper", "candidate_count": 2, "max_score": .9,
                "candidates_file": "grasps.yaml", "candidates_sha256": sha256_file(candidates),
                "scene_collision_checked": False, "robot_ik_checked": False, "robot_command_published": False}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory, metadata


@pytest.mark.parametrize("name", [None, "metadata.json", "grasps.yaml"])
def test_original_directory_manifest_and_isaac_yaml_return_same_raw_order_without_writes(legacy_bundle, name):
    directory, expected = legacy_bundle
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    poses, scores, metadata = load_grasp_bundle(directory if name is None else directory / name)
    np.testing.assert_allclose(poses[:, :3, 3], [[.1, -.2, .3], [-.4, .5, .6]])
    np.testing.assert_allclose(poses[0, :3, :3], np.eye(3))
    np.testing.assert_allclose(poses[1, :3, :3], np.diag([-1., -1., 1.]))
    np.testing.assert_allclose(scores, [.7, .9])
    assert metadata == expected and poses.flags.owndata and scores.flags.owndata
    assert "candidate_pool_count" not in metadata and "candidate_pool_file" not in metadata
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_legacy_identity_and_candidate_file_hash_are_still_checked(legacy_bundle):
    directory, metadata = legacy_bundle
    changed = deepcopy(metadata)
    changed["asset_geometry"]["mesh_scale_to_m"] = 1.
    (directory / "metadata.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="provenance"):
        load_grasp_bundle(directory)
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory / "grasps.yaml").write_text("changed")
    with pytest.raises(ValueError, match="provenance"):
        load_grasp_bundle(directory)


def test_standalone_yaml_does_not_invent_geometry_or_gripper_provenance(tmp_path):
    path = tmp_path / "grasps.yaml"
    path.write_text("format: isaac_grasp\nformat_version: 1.0\ngrasps: {}\n")
    with pytest.raises(FileNotFoundError):
        load_grasp_bundle(path)

