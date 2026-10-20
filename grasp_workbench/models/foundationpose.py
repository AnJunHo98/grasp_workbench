"""Official FoundationPose registration and tracking, with one session per object."""
import importlib
from pathlib import Path
import sys

import numpy as np


def _pose_from_output(value):
    """The previous UI adapter's SE(3) checks, including its TF32 tolerance."""
    pose = np.asarray(value, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    if (not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0., 0., 0., 1.), atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-3, rtol=0)):
        raise ValueError("FoundationPose returned an invalid SE(3) matrix")
    return pose.copy()


def visual_pose_mismatch_reason(frame, pose, mask, box, *, bbox_margin_px=40,
                                depth_margin_m=.25, depth_min_pixels=16):
    """The previous UI's projected-origin bbox and mask depth min/max gates."""
    x, y, z = pose[:3, 3]
    if z <= 1e-6:
        return "visual_mismatch:bbox"
    intrinsics = frame.metadata["intrinsics"]
    u = intrinsics["fx"] * x / z + intrinsics["cx"]
    v = intrinsics["fy"] * y / z + intrinsics["cy"]
    x1, y1, x2, y2 = box
    if not (x1 - bbox_margin_px <= u <= x2 + bbox_margin_px
            and y1 - bbox_margin_px <= v <= y2 + bbox_margin_px):
        return "visual_mismatch:bbox"
    if mask.shape != frame.depth_m.shape:
        return "visual_mismatch:mask_shape"
    values = frame.depth_m[mask & np.isfinite(frame.depth_m) & (frame.depth_m > 0)]
    if values.size >= depth_min_pixels and not (
        float(values.min()) - depth_margin_m <= z <= float(values.max()) + depth_margin_m
    ):
        return "visual_mismatch:depth"
    return None


def _arrays(frame):
    intrinsics = frame.metadata["intrinsics"]
    matrix = np.asarray(((intrinsics["fx"], 0, intrinsics["cx"]),
                         (0, intrinsics["fy"], intrinsics["cy"]), (0, 0, 1)), dtype=np.float32)
    return np.ascontiguousarray(frame.rgb), np.ascontiguousarray(frame.depth_m), matrix


class OfficialFoundationPosePredictor:
    def __init__(self, source_dir, *, device="cuda:0", registration_iterations=5, tracking_iterations=1):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.device = device
        self.registration_iterations = registration_iterations
        self.tracking_iterations = tracking_iterations
        self._sessions = {}
        self._last_poses = {}
        self._backend = self._torch = self._scorer = self._refiner = self._glctx = None

    def start(self):
        sys.path.insert(0, str(self.source_dir))
        self._torch = importlib.import_module("torch")
        self._torch.cuda.set_device(self.device)
        self._backend = importlib.import_module("estimater")
        with self._torch.inference_mode():
            self._scorer = self._backend.ScorePredictor()
            self._refiner = self._backend.PoseRefinePredictor()
            self._glctx = self._backend.dr.RasterizeCudaContext(self.device)

    def _session(self, object_id, mesh_path, mesh_scale_to_m):
        import trimesh

        session = self._sessions.get(object_id)
        if session is None:
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.apply_scale(mesh_scale_to_m)
            debug_dir = Path(__file__).resolve().parents[2] / "runs/foundationpose" / object_id
            session = self._backend.FoundationPose(
                model_pts=np.asarray(mesh.vertices, dtype=np.float32),
                model_normals=np.asarray(mesh.vertex_normals, dtype=np.float32), mesh=mesh,
                scorer=self._scorer, refiner=self._refiner, glctx=self._glctx,
                debug=0, debug_dir=str(debug_dir),
            )
            # NumPy float64 radius otherwise makes the upstream crop offsets double.
            session.diameter = float(session.diameter)
            self._sessions[object_id] = session
        return session

    def seed(self, object_id, pose, mesh_path, mesh_scale_to_m):
        pose = _pose_from_output(pose)
        with self._torch.inference_mode():
            session = self._session(object_id, mesh_path, mesh_scale_to_m)
            centered_from_mesh = session.get_tf_to_centered_mesh()
            if hasattr(centered_from_mesh, "detach"):
                centered_from_mesh = centered_from_mesh.detach().cpu().numpy()
            session.pose_last = self._torch.as_tensor(
                pose @ np.linalg.inv(np.asarray(centered_from_mesh)), dtype=self._torch.float32, device=self.device,
            )
        self._last_poses[object_id] = pose

    def register(self, object_id, frame, mask, mesh_path, mesh_scale_to_m):
        rgb, depth, matrix = _arrays(frame)
        with self._torch.inference_mode():
            session = self._session(object_id, mesh_path, mesh_scale_to_m)
            pose = session.register(K=matrix, rgb=rgb, depth=depth, ob_mask=mask,
                                    iteration=self.registration_iterations)
        pose = _pose_from_output(pose)
        self._last_poses[object_id] = pose
        return pose

    def track(self, object_id, frame):
        rgb, depth, matrix = _arrays(frame)
        with self._torch.inference_mode():
            pose = self._sessions[object_id].track_one(
                rgb=rgb, depth=depth, K=matrix, iteration=self.tracking_iterations,
            )
        pose = _pose_from_output(pose)
        previous = self._last_poses[object_id]
        if np.linalg.norm(pose[:3, 3] - previous[:3, 3]) > .25:
            self._sessions[object_id].pose_last = None
            raise ValueError("FoundationPose translation jump exceeded 0.250 m")
        self._last_poses[object_id] = pose
        return pose

    def remove(self, object_id):
        self._last_poses.pop(object_id, None)
        session = self._sessions.pop(object_id, None)
        if session is not None and session.mesh_path is not None:
            Path(session.mesh_path).unlink(missing_ok=True)

    def close(self):
        for object_id in tuple(self._sessions):
            self.remove(object_id)
        self._scorer = self._refiner = self._glctx = self._backend = self._torch = None
