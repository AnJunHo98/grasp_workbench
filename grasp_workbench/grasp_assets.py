"""Persistent object-local grasp bundles, independent of live camera poses."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def grasp_identity(request, *, gripper_id, gripper_metadata, model_metadata, inference_settings):
    return {
        "asset_id": request.asset_id,
        "asset_geometry": {"mesh_sha256": sha256_file(request.mesh_path),
                           "mesh_scale_to_m": request.mesh_scale_to_m,
                           "object_frame": request.asset_id + "_body", "length_unit": "metre"},
        "gripper_id": gripper_id, "gripper_metadata": gripper_metadata,
        "model_metadata": model_metadata, "inference_settings": inference_settings,
    }


def validate_arrays(grasps, scores):
    poses, confidence = np.asarray(grasps, np.float64), np.asarray(scores, np.float64)
    if poses.shape == (0,):
        poses = poses.reshape(0, 4, 4)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("Grasp poses must be finite with shape (N, 4, 4)")
    if (confidence.shape != (len(poses),) or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1))):
        raise ValueError("Grasp scores must be finite in [0, 1] with shape (N,)")
    if len(poses):
        rotations = poses[:, :3, :3]
        if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-4)):
            raise ValueError("Grasp poses must be rigid transforms")
    return poses.copy(), confidence.copy()


def _read_isaac_yaml(path):
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document.get("format") != "isaac_grasp" or document.get("format_version") != 1.0:
        raise ValueError("Expected an Isaac grasp candidate file")
    poses, scores = [], []
    for candidate in document["grasps"].values():
        orientation = candidate["orientation"]
        q = np.asarray([orientation["w"], *orientation["xyz"]], dtype=float)
        position = np.asarray(candidate["position"], dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or not np.dot(q, q) or position.shape != (3,):
            raise ValueError("Invalid Isaac grasp pose")
        w, x, y, z = q
        s = 2. / np.dot(q, q)
        pose = np.eye(4)
        pose[:3, :3] = [[1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
                       [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
                       [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]]
        pose[:3, 3] = position
        poses.append(pose)
        scores.append(candidate["confidence"])
    return validate_arrays(poses, scores)


def load_grasp_bundle(path):
    """Read native NPZ or the original metadata-backed Isaac YAML, without writes."""
    path = Path(path)
    path = path / "metadata.json" if path.is_dir() else path
    if path.suffix.lower() in {".yaml", ".yml"}:
        path = path.parent / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (metadata["schema_version"] != 1 or metadata["candidate_frame"] != "canonical_gripper_base"
            or metadata["pose_convention"] != "object_from_canonical_gripper"):
        raise ValueError("Unsupported grasp bundle format")
    identity = {key: metadata[key] for key in (
        "asset_id", "asset_geometry", "gripper_id", "gripper_metadata", "model_metadata", "inference_settings")}
    candidates = path.parent / metadata["candidates_file"]
    if cache_key(identity) != metadata["cache_key"] or sha256_file(candidates) != metadata["candidates_sha256"]:
        raise ValueError("Grasp bundle provenance does not match its data")
    if candidates.suffix.lower() in {".yaml", ".yml"}:
        grasps, scores = _read_isaac_yaml(candidates)
    else:
        with np.load(candidates, allow_pickle=False) as data:
            grasps, scores = validate_arrays(data["grasps"], data["scores"])
    if len(grasps) != metadata["candidate_count"]:
        raise ValueError("Grasp bundle candidate count does not match its data")
    return grasps, scores, metadata


def save_grasp_bundle(output_dir, identity, grasps, scores, *, raw_poses, raw_scores, sample_points, selected_indices):
    poses, confidence = validate_arrays(grasps, scores)
    key = cache_key(identity)
    directory = Path(output_dir) / key
    directory.mkdir(parents=True, exist_ok=True)
    candidates = directory / "grasps.npz"
    np.savez_compressed(candidates, grasps=poses, scores=confidence)
    np.savez_compressed(directory / "candidate_pool.npz", grasps=raw_poses, scores=raw_scores,
                        sample_points=sample_points, selected_indices=selected_indices)
    metadata = {
        "schema_version": 1, **identity, "cache_key": key,
        "object_frame": identity["asset_geometry"]["object_frame"],
        "candidate_frame": "canonical_gripper_base", "pose_convention": "object_from_canonical_gripper",
        "candidate_count": len(poses), "max_score": float(confidence.max()) if len(poses) else None,
        "candidates_file": candidates.name, "candidates_sha256": sha256_file(candidates),
        "candidate_pool_file": "candidate_pool.npz", "candidate_pool_count": len(raw_poses),
        "scene_collision_checked": False, "robot_ik_checked": False, "robot_command_published": False,
    }
    path = directory / "metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return str(path)
