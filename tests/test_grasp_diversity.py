"""CPU-only checks for candidate coverage without changing generated geometry."""

import numpy as np

from grasp_workbench.models.grasp_diversity import select_diverse_grasps


def rotation_y(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


def make_poses(count):
    return np.repeat(np.eye(4)[None], count, axis=0)


def test_low_score_unique_pose_survives_high_score_cluster():
    poses = make_poses(5)
    poses[:, 0, 3] = (0, 0.001, 0.002, 0.003, 0.1)
    assert select_diverse_grasps(poses, [1, 0.99, 0.98, 0.97, 0.1], 2).tolist() == [0, 4]


def test_translation_and_rotation_are_both_covered():
    poses = make_poses(4)
    poses[1, 0, 3] = 0.04
    poses[2, :3, :3] = rotation_y(90)
    poses[3, 0, 3] = 0.001
    poses[3, :3, :3] = rotation_y(2)
    assert select_diverse_grasps(poses, [1, 0.9, 0.8, 0.7], 3).tolist() == [0, 2, 1]


def test_selection_does_not_change_poses_or_scores():
    poses = make_poses(3)
    poses[:, 0, 3] = (0, 0.04, 0.08)
    scores = np.array([0.1, 0.9, 0.3])
    original_poses, original_scores = poses.copy(), scores.copy()
    selected = select_diverse_grasps(poses, scores, 2)
    assert selected.tolist() == [1, 2]
    np.testing.assert_array_equal(poses, original_poses)
    np.testing.assert_array_equal(scores, original_scores)


def test_common_world_transform_does_not_change_selection():
    poses = make_poses(4)
    for i, degrees in enumerate((0, 27, 70, 150)):
        poses[i, :3, :3] = rotation_y(degrees)
        poses[i, :3, 3] = (i * 0.021, i * 0.003, i * 0.017)
    world = np.eye(4)
    world[:3, :3] = rotation_y(41)
    world[:3, 3] = (0.3, -0.4, 0.2)
    settings = {"reference_point_m": (0.01, 0.02, 0.1), "symmetric": True}
    scores = [0.8, 0.9, 0.5, 0.7]
    np.testing.assert_array_equal(
        select_diverse_grasps(poses, scores, 3, **settings),
        select_diverse_grasps(world @ poses, scores, 3, **settings),
    )


def test_local_z_half_turn_symmetry_preserves_different_approach():
    poses = make_poses(3)
    poses[1, :3, :3] = np.diag((-1, -1, 1))
    poses[2, :3, :3] = rotation_y(180)
    scores = [1, 0.9, 0.8]
    assert select_diverse_grasps(poses, scores, 3).tolist() == [0, 1, 2]
    assert select_diverse_grasps(poses, scores, 3, symmetric=True).tolist() == [0, 2]


def test_reference_point_avoids_artificial_gripper_base_distance():
    poses = make_poses(2)
    reference = np.array((0, 0, 0.1))
    poses[1, :3, :3] = rotation_y(10)
    poses[1, :3, 3] = reference - poses[1, :3, :3] @ reference
    assert select_diverse_grasps(poses, [1, 0.9], 2).tolist() == [0, 1]
    assert select_diverse_grasps(
        poses, [1, 0.9], 2, reference_point_m=reference,
    ).tolist() == [0]


def test_near_duplicates_are_not_used_to_pad_limit():
    poses = make_poses(6)
    poses[:, 0, 3] = (0, 0.001, 0.002, 0.1, 0.101, 0.102)
    selected = select_diverse_grasps(poses, [1, 0.9, 0.8, 0.7, 0.6, 0.5], 100)
    assert selected.tolist() == [0, 5]


def test_coverage_ties_use_score_then_original_index():
    poses = make_poses(4)
    poses[:, 0, 3] = (0, 0.1, -0.1, 0.1)
    assert select_diverse_grasps(poses, [1, 0.5, 0.6, 0.5], 3).tolist() == [0, 2, 1]
    assert select_diverse_grasps(poses, [1, 0.6, 0.6, 0.6], 2).tolist() == [0, 1]


def test_empty_and_zero_limit():
    assert select_diverse_grasps([], [], 10).shape == (0,)
    assert select_diverse_grasps(make_poses(1), [1], 0).shape == (0,)
