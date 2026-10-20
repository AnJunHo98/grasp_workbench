"""Geometry-only GraspGenX inference with the previous worker's sampling and cache."""
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ..grasp_assets import cache_key, grasp_identity, load_grasp_bundle, save_grasp_bundle, sha256_file, validate_arrays
from . import grasp_diversity


@dataclass
class GraspPrediction:
    grasps: np.ndarray
    scores: np.ndarray
    bundle_path: str
    gripper_metadata: dict
    cached: bool = False


def sample_object_points(mesh_path, scale_to_m, count, seed):
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(scale_to_m)
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def _virtual_gripper_descriptor(config, scale):
    sweep = config["sweep_volume"]
    descriptor = {target: (np.asarray(sweep[source], np.float32) * scale).tolist()
                  for target, source in (("extents_open", "extents"), ("offset_open", "offset"),
                                         ("extents_mid", "extents2"), ("offset_mid", "offset2"))}
    descriptor["gripper_type"] = {"parallel_2f": 0, "revolute_2f": 1, "revolute_3f": 2}[config["type"]]
    descriptor["fingertip_depth"] = float(config["fingertip"][-1]) * scale
    return descriptor


def _position_clip_compat(sampler, clip_range):
    generator = sampler.model.grasp_generator
    if clip_range == 1. or hasattr(generator, "position_clip_sample_range"):
        return
    scheduler = generator.noise_scheduler_pos
    scheduler.register_to_config(clip_sample=False, prediction_type="sample")
    original_step = scheduler.step

    def step(model_output, timestep, sample, **kwargs):
        alpha = scheduler.alphas_cumprod[timestep].to(sample.device)
        predicted = (sample - (1 - alpha).sqrt() * model_output) / alpha.sqrt()
        return original_step(model_output=predicted.clamp(-clip_range, clip_range),
                             timestep=timestep, sample=sample, **kwargs)

    scheduler.step = step


class OfficialGraspGenXPredictor:
    def __init__(self, source_dir=None, *, checkpoint_dir=None, assets_dir=None, gripper_id="franka_panda",
                 output_dir="runs/grasps", settings=None):
        self.source_dir = Path(source_dir or Path(__file__).resolve().parents[2] / "third_party/GraspGenX").expanduser().resolve()
        self.checkpoint_dir = Path(checkpoint_dir or self.source_dir / "ext/graspgenx_checkpoints/release").expanduser().resolve()
        self.assets_dir = Path(assets_dir or self.source_dir / "assets").expanduser().resolve()
        self.output_dir, self.gripper_id = Path(output_dir).expanduser().resolve(), gripper_id
        self.options = dict(settings or {})
        self._cfg = self._sampler = self._sampler_class = None
        self.gripper_metadata = self.model_metadata = self.settings = None

    def start(self):
        if self._cfg is not None:
            return
        if str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))
        from graspgenx.utils.checkpoint_io import load_model_cfg
        from graspgenx.x_grippers import resolve_gripper_asset_dir
        from omegaconf import OmegaConf

        cfg = load_model_cfg(str(self.checkpoint_dir / "gen"), str(self.checkpoint_dir / "dis"),
                             self.options.get("gen_pth"), self.options.get("dis_pth"))
        clip_range = float(self.options.get("position_clip_sample_range", 1.))
        cfg.diffusion.position_clip_sample_range = clip_range
        descriptor_dir = Path(resolve_gripper_asset_dir(self.gripper_id, str(self.assets_dir)))
        config_path = descriptor_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        virtual_scale = float(self.options.get("virtual_geometry_scale", 1.))
        virtual_descriptor = _virtual_gripper_descriptor(config, virtual_scale) if virtual_scale != 1. else None
        geometry_files = [descriptor_dir / name for name in ("config.json", "gripper.urdf", "vis_mesh.obj", "coll_mesh.obj")]
        geometry_files.extend(path for path in (descriptor_dir / "meshes").rglob("*") if path.is_file())
        self.gripper_metadata = {
            "config_path": str(config_path), "config_sha256": sha256_file(config_path), "config": config,
            "canonical_from_native": config.get("base_rotation", np.eye(4).tolist()),
            "collision_mesh_status": "actual" if (descriptor_dir / "coll_mesh.obj").is_file() else "dummy_fallback",
            "files": [{"relative_path": str(path.relative_to(descriptor_dir)), "sha256": sha256_file(path)}
                      for path in sorted(geometry_files) if path.is_file()],
        }
        checkpoints = []
        for role, checkpoint in (("generator", cfg.eval.gen_checkpoint), ("discriminator", cfg.eval.dis_checkpoint)):
            path = Path(checkpoint)
            checkpoints.append({"role": role, "path": str(path), "sha256": sha256_file(path),
                                "config_path": str(path.parent / "config.yaml"),
                                "config_sha256": sha256_file(path.parent / "config.yaml")})
        revision = subprocess.run(["git", "-C", str(self.source_dir), "rev-parse", "HEAD"],
                                  text=True, capture_output=True, check=False)
        self.model_metadata = {
            "source_dir": str(self.source_dir), "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "generator_source_sha256": sha256_file(self.source_dir / "graspgenx/models/generator.py"),
            "worker_sha256": sha256_file(Path(__file__)),
            "candidate_selection_source_sha256": sha256_file(Path(grasp_diversity.__file__)),
            "checkpoints": checkpoints, "diffusion": OmegaConf.to_container(cfg.diffusion, resolve=True),
            "discriminator": OmegaConf.to_container(cfg.discriminator, resolve=True),
            "torch_version": importlib.metadata.version("torch"), "device": "cuda", "precision": "fp32", "tensorrt": False,
            "virtual_geometry_scale": virtual_scale, "virtual_gripper_descriptor": virtual_descriptor,
            "position_clip_compat": "instance_scheduler_epsilon_to_clamped_x0_v1",
        }
        if virtual_scale != 1.:
            self.model_metadata.update({
                "virtual_sampler_route": "from_sweep_volume", "virtual_sampler_uses_physical_meshes": False,
                "virtual_sampler_source_sha256": {relative: sha256_file(self.source_dir / relative) for relative in (
                    "graspgenx/grasp_server.py", "graspgenx/x_grippers.py", "graspgenx/serving/types.py")},
            })
        self.settings = {
            "planner": "diffusion", "sampling_frame": "object_local", "point_cloud_centered_by_adapter": False,
            "position_clip_sample_range": clip_range, "virtual_geometry_scale": virtual_scale,
            "virtual_gripper_descriptor": virtual_descriptor,
            "num_sample_points": int(self.options.get("num_sample_points", cfg.data.num_points)),
            "num_grasps": int(self.options.get("num_grasps", 1000)),
            "grasp_threshold": float(self.options.get("grasp_threshold", -1.)),
            "topk_num_grasps": int(self.options.get("topk_num_grasps", 100)),
            "min_grasps": int(self.options.get("min_grasps", 40)), "max_tries": int(self.options.get("max_tries", 6)),
            "remove_outliers": bool(self.options.get("remove_outliers", False)), "seed": int(self.options.get("seed", 42)),
            "candidate_selection": {"method": "farthest_point_se3_v1", "translation_threshold_m": .01,
                                    "rotation_threshold_deg": 15., "reference_point_m": config.get("fingertip", [0., 0., 0.]),
                                    "symmetric": bool(config.get("symmetric", False))},
        }
        self._cfg = cfg

    def _infer(self, points):
        import torch

        np.random.seed(self.settings["seed"])
        torch.manual_seed(self.settings["seed"])
        virtual_scale = self.settings["virtual_geometry_scale"]
        if self._sampler is None:
            from graspgenx.grasp_server import GraspGenXSampler

            self._sampler_class = GraspGenXSampler
            self._sampler = (GraspGenXSampler(self._cfg, self.gripper_id, assets_dir=str(self.assets_dir)) if virtual_scale == 1.
                             else GraspGenXSampler.from_sweep_volume(self._cfg, self.settings["virtual_gripper_descriptor"]))
            _position_clip_compat(self._sampler, self.settings["position_clip_sample_range"])
        options = {key: self.settings[key] for key in (
            "num_grasps", "grasp_threshold", "min_grasps", "max_tries", "remove_outliers")}
        options["topk_num_grasps"] = self.settings["num_grasps"]
        poses, scores = self._sampler_class.run_inference(points * virtual_scale, self._sampler, **options)
        poses, scores = poses.detach().cpu().numpy(), scores.detach().cpu().numpy()
        if virtual_scale != 1. and len(poses):
            poses = poses.copy()
            poses[:, :3, 3] /= virtual_scale
        return validate_arrays(poses, scores)

    def predict(self, request):
        if self._cfg is None:
            raise RuntimeError("Start GraspGenX before prediction")
        identity = grasp_identity(request, gripper_id=self.gripper_id, gripper_metadata=self.gripper_metadata,
                                  model_metadata=self.model_metadata, inference_settings=self.settings)
        path = self.output_dir / cache_key(identity) / "metadata.json"
        if path.is_file() and not request.regenerate:
            try:
                poses, scores, metadata = load_grasp_bundle(path)
                return GraspPrediction(poses, scores, str(path), metadata["gripper_metadata"], True)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        points = sample_object_points(request.mesh_path, request.mesh_scale_to_m,
                                      self.settings["num_sample_points"], self.settings["seed"])
        raw_poses, raw_scores = self._infer(points)
        limit = self.settings["topk_num_grasps"]
        if limit == -1:
            limit = 100 if self.settings["grasp_threshold"] == -1. else len(raw_poses)
        selected = grasp_diversity.select_diverse_grasps(
            raw_poses, raw_scores, limit,
            **{key: value for key, value in self.settings["candidate_selection"].items() if key != "method"})
        poses, scores = raw_poses[selected], raw_scores[selected]
        path = save_grasp_bundle(self.output_dir, identity, poses, scores, raw_poses=raw_poses, raw_scores=raw_scores,
                                 sample_points=points, selected_indices=selected)
        return GraspPrediction(poses, scores, path, self.gripper_metadata)

    def close(self):
        self._sampler = self._sampler_class = self._cfg = None
