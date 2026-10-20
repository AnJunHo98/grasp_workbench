"""Deterministic object-local grasp coverage, ported from the previous worker."""
import numpy as np


def select_diverse_grasps(poses, scores, limit, *, reference_point_m=(0., 0., 0.), symmetric=False,
                         translation_threshold_m=.01, rotation_threshold_deg=15.):
    """Keep quality first, then farthest SE(3) coverage with stable score/index ties."""
    poses, scores = np.asarray(poses, dtype=float), np.asarray(scores, dtype=float)
    if limit <= 0 or len(poses) == 0:
        return np.empty(0, dtype=np.int64)
    if poses.shape != (len(poses), 4, 4) or scores.shape != (len(poses),):
        raise ValueError("Expected poses (N, 4, 4) and scores (N,)")
    if translation_threshold_m <= 0 or rotation_threshold_deg <= 0:
        raise ValueError("Diversity thresholds must be positive")
    order = np.argsort(-scores, kind="stable")
    rotations = poses[order, :3, :3]
    points = rotations @ np.asarray(reference_point_m, dtype=float) + poses[order, :3, 3]
    nearest, selected, candidate = np.full(len(poses), np.inf), [], 0
    for _ in range(min(int(limit), len(poses))):
        selected.append(int(order[candidate]))
        traces = np.einsum("nij,ij->n", rotations, rotations[candidate])
        if symmetric:
            equivalent = rotations[candidate] * np.array((-1., -1., 1.))
            traces = np.maximum(traces, np.einsum("nij,ij->n", rotations, equivalent))
        angles = np.arccos(np.clip((traces - 1.) / 2., -1., 1.))
        distances = np.maximum(np.linalg.norm(points - points[candidate], axis=1) / translation_threshold_m,
                               angles / np.deg2rad(rotation_threshold_deg))
        nearest = np.minimum(nearest, distances)
        nearest[candidate] = -np.inf
        candidate = int(np.argmax(nearest))
        if nearest[candidate] < 1.:
            break
    return np.asarray(selected, dtype=np.int64)
