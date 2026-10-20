"""Official SAM3D with measured/monocular pointmaps and explicit local units."""
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import importlib
import os
from pathlib import Path
import sys

import numpy as np

from ..ipc.sam3d import MeshGeometry, validate_mesh


_OPENCV_FROM_PYTORCH3D = np.diag([-1., -1., 1.])
# Upstream to_glb applies this row-vector axis change to canonical vertices.
_GLB_ROW_AXIS_TRANSFORM = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
MOGE_MODEL_REVISION = "979e84da9415762c30e6c0cf8dc0962896c793df"


@dataclass
class Sam3dPrediction:
    mesh: MeshGeometry
    camera_from_object: np.ndarray
    textured_glb: bytes | None = None
    length_unit: str = "metre"


@contextmanager
def _gsplat_texture_rendering():
    """Use the old adapter's supported gsplat observations without vendor edits."""
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

    original = render_utils.render_frames

    def render_frames(sample, extrinsics, intrinsics, options=None, **kwargs):
        return original(sample, extrinsics, intrinsics, {**(options or {}), "backend": "gsplat"}, **kwargs)

    render_utils.render_frames = render_frames
    try:
        yield
    finally:
        render_utils.render_frames = original


def depth_to_pointmap(depth_m, camera_k):
    """Backproject optical z-depth in metres to left/up/forward coordinates."""
    depth, matrix = np.asarray(depth_m, np.float32), np.asarray(camera_k, np.float64)
    rows, cols = np.indices(depth.shape)
    rays = np.stack((cols, rows, np.ones_like(cols)), axis=-1) @ np.linalg.inv(matrix).T
    valid = np.isfinite(depth) & (depth > 0)
    points = rays * (np.where(valid, depth, np.nan) / rays[..., 2])[..., None]
    return (points @ _OPENCV_FROM_PYTORCH3D.T).astype(np.float32)


def metric_vertices_and_pose(glb_vertices, rotation_wxyz, translation, scale):
    """Undo GLB display axes, bake output scale once, retain a rigid camera pose."""
    scale = np.asarray(scale, np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    canonical = np.asarray(glb_vertices, np.float64) @ _GLB_ROW_AXIS_TRANSFORM.T
    w, x, y, z = q = np.asarray(rotation_wxyz, np.float64).reshape(4)
    s = 2. / np.dot(q, q)
    rotation = np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])
    pose = np.eye(4)
    pose[:3, :3] = _OPENCV_FROM_PYTORCH3D @ rotation.T
    pose[:3, 3] = _OPENCV_FROM_PYTORCH3D @ np.asarray(translation, np.float64).reshape(3)
    return (canonical * scale).astype(np.float32), pose


def _numpy(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


class OfficialSam3dPredictor:
    def __init__(self, config_path, *, source_dir=None, seed=42):
        self.config_path = Path(config_path).expanduser().resolve()
        self.source_dir = Path(source_dir or Path(__file__).resolve().parents[2]
                               / "third_party/sam-3d-objects").expanduser().resolve()
        self.seed = seed
        self._pipeline = None

    def start(self):
        if self._pipeline is not None:
            return
        # The official notebook's Inference wrapper uses the same pipeline/run
        # options, but also imports optional Gradio/Kaolin notebook viewers.
        if str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
        importlib.import_module("sam3d_objects.pipeline.inference_pipeline_pointmap")
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config = OmegaConf.load(self.config_path)
        config.workspace_dir = str(self.config_path.parent)
        config.compile_model = False
        config.rendering_engine = "pytorch3d"
        config.depth_model = None
        self._pipeline = instantiate(config)

    def _monocular_depth(self):
        if self._pipeline.depth_model is None:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf

            config = OmegaConf.load(self.config_path)
            config.workspace_dir = str(self.config_path.parent)
            config.depth_model.model.revision = MOGE_MODEL_REVISION
            self._pipeline.depth_model = instantiate(config.depth_model)

    def predict(self, frame, mask, *, depth_source="depth", texture_baking=False):
        if self._pipeline is None:
            raise RuntimeError("Start SAM3D before prediction")
        rgb, mask = np.asarray(frame.rgb), np.asarray(mask)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM3D requires uint8 RGB")
        if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2] or not mask.any():
            raise ValueError("SAM3D requires a nonempty bool mask matching the source RGB")
        if depth_source not in ("depth", "monocular") or type(texture_baking) is not bool:
            raise ValueError("SAM3D requires depth/monocular source and bool texture_baking")
        import torch

        pointmap = None
        if depth_source == "depth":
            intrinsics = frame.metadata["intrinsics"]
            matrix = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                               [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]])
            pointmap = torch.from_numpy(depth_to_pointmap(frame.depth_m, matrix))
        else:
            self._monocular_depth()
        rgba = np.concatenate((rgb, (mask.astype(np.uint8) * 255)[..., None]), axis=-1)
        renderer = _gsplat_texture_rendering() if texture_baking else nullcontext()
        with torch.inference_mode(), renderer:
            output = self._pipeline.run(
                rgba, seed=self.seed, pointmap=pointmap, stage1_only=False,
                with_mesh_postprocess=False, with_texture_baking=texture_baking,
                with_layout_postprocess=False, use_vertex_color=not texture_baking,
            )
        asset = output.get("glb")
        if asset is None:
            raise ValueError("SAM3D returned no mesh")
        vertices, pose = metric_vertices_and_pose(
            asset.vertices, _numpy(output["rotation"]), _numpy(output["translation"]), _numpy(output["scale"]),
        )
        faces = np.asarray(asset.faces, dtype=np.uint32).copy()
        visual = getattr(asset, "visual", None)
        raw_colors = getattr(visual, "vertex_colors", None)
        colors = None if raw_colors is None else np.asarray(raw_colors, dtype=np.uint8)[:, :3].copy()
        mesh = MeshGeometry(vertices, faces, colors)
        validate_mesh(mesh)
        textured_glb = None
        if texture_baking:
            converted = asset.copy()
            converted.vertices = vertices
            textured_glb = converted.export(file_type="glb")
        return Sam3dPrediction(mesh, pose, textured_glb,
                              "metre" if depth_source == "depth" else "monocular_predicted_scale")

    def close(self):
        self._pipeline = None
