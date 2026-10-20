"""Old GraspGenX inference contract with fake checkpoint/config/sampler modules."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.grasp_assets import load_grasp_bundle
from grasp_workbench.models.graspgenx import OfficialGraspGenXPredictor, _position_clip_compat, sample_object_points


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    source = tmp_path / "source"
    for relative in ("graspgenx/models/generator.py", "graspgenx/grasp_server.py", "graspgenx/x_grippers.py", "graspgenx/serving/types.py"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic-source")
    checkpoints = tmp_path / "checkpoints"
    for role in ("gen", "dis"):
        path = checkpoints / role
        path.mkdir(parents=True)
        (path / "model.pth").write_bytes(b"synthetic-weights")
        (path / "config.yaml").write_text("synthetic-config")
    descriptor = tmp_path / "assets" / "test-gripper"
    descriptor.mkdir(parents=True)
    config = {"fingertip": [0, 0, .1], "symmetric": True, "base_rotation": np.diag([-1., -1., 1., 1.]).tolist(),
              "type": "parallel_2f", "sweep_volume": {"extents": [1, 2, 3], "offset": [4, 5, 6],
                                                      "extents2": [7, 8, 9], "offset2": [10, 11, 12]}}
    (descriptor / "config.json").write_text(json.dumps(config))
    (descriptor / "vis_mesh.obj").write_text("synthetic-visual")
    (descriptor / "coll_mesh.obj").write_text("synthetic-collision")
    calls = {"cfg": [], "samplers": [], "infer": [], "seeds": []}

    def load_cfg(*args):
        calls["cfg"].append(args)
        return SimpleNamespace(diffusion=SimpleNamespace(gripper_backbone="sweep_volume_v2"),
                               discriminator=SimpleNamespace(gripper_backbone="sweep_volume_v2"),
                               data=SimpleNamespace(num_points=2000),
                               eval=SimpleNamespace(gen_checkpoint=str(checkpoints / "gen/model.pth"),
                                                    dis_checkpoint=str(checkpoints / "dis/model.pth")))

    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, 0, 3] = [0, .001, .07, .2, .201, -.1]
    scores = np.array([.99, .98, .4, .9, .5, .8])

    class Sampler:
        def __init__(self, cfg, gripper_id, **options):
            calls["samplers"].append((cfg, gripper_id, options))
            self.model = SimpleNamespace(grasp_generator=SimpleNamespace())

        @classmethod
        def from_sweep_volume(cls, cfg, descriptor):
            return cls(cfg, "virtual", descriptor=descriptor)

        @staticmethod
        def run_inference(points, sampler, **options):
            calls["infer"].append((points.copy(), sampler, options))
            return Tensor(poses), Tensor(scores)

    modules = {
        "graspgenx.utils.checkpoint_io": {"load_model_cfg": load_cfg},
        "graspgenx.x_grippers": {"resolve_gripper_asset_dir": lambda *args: str(descriptor)},
        "graspgenx.grasp_server": {"GraspGenXSampler": Sampler},
        "omegaconf": {"OmegaConf": SimpleNamespace(to_container=lambda value, **kw: vars(value).copy())},
        "torch": {"manual_seed": lambda value: calls["seeds"].append(value)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fake-torch")
    import grasp_workbench.models.graspgenx as model

    sampled = np.array([[.3, .4, .5], [.4, .5, .6]], np.float32)
    monkeypatch.setattr(model, "sample_object_points", lambda *args: calls.setdefault("sample", []).append(args) or sampled.copy())
    mesh = tmp_path / "mesh.ply"
    mesh.write_text("synthetic-mesh")
    request = SimpleNamespace(request_id="job-1", asset_id="object", object_id="A", mesh_path=str(mesh),
                              mesh_scale_to_m=.01, regenerate=False)

    def make(settings=None):
        return OfficialGraspGenXPredictor(source, checkpoint_dir=checkpoints, assets_dir=descriptor.parent,
                                          gripper_id="test-gripper", output_dir=tmp_path / "output", settings=settings)

    return make, request, calls, poses, scores, config


def test_one_model_lazily_loaded_and_previous_defaults_raw_pool_diversity_cache(backend):
    make, request, calls, poses, scores, config = backend
    predictor = make({"topk_num_grasps": 2})
    predictor.start()
    predictor.start()
    assert len(calls["cfg"]) == 1 and not calls["samplers"]
    result = predictor.predict(request)
    assert not result.cached and len(calls["samplers"]) == len(calls["infer"]) == 1
    assert calls["sample"][0][1:] == (.01, 2000, 42)
    assert calls["seeds"] == [42]
    assert calls["infer"][0][2] == {"num_grasps": 1000, "grasp_threshold": -1., "topk_num_grasps": 1000,
                                    "min_grasps": 40, "max_tries": 6, "remove_outliers": False}
    np.testing.assert_array_equal(result.grasps, poses[[0, 4]])
    np.testing.assert_array_equal(result.scores, scores[[0, 4]])
    assert result.gripper_metadata["config"] == config
    assert result.gripper_metadata["collision_mesh_status"] == "actual"
    assert any(item["relative_path"] == "vis_mesh.obj" for item in result.gripper_metadata["files"])
    with np.load(Path(result.bundle_path).with_name("candidate_pool.npz"), allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["grasps"], poses)
        np.testing.assert_array_equal(raw["selected_indices"], [0, 4])
    cached = predictor.predict(request)
    assert cached.cached and len(calls["infer"]) == 1
    np.testing.assert_array_equal(cached.grasps, result.grasps)
    request.asset_id = "other-object"
    predictor.predict(request)
    assert len(calls["infer"]) == 2 and len(calls["samplers"]) == 1
    predictor.close()
    assert predictor._cfg is None and predictor._sampler is None


def test_cold_cache_reuse_skips_sampler_and_regenerate_invokes_it(backend):
    make, request, calls, _poses, _scores, _config = backend
    first = make()
    first.start()
    generated = first.predict(request)
    first.close()
    second = make()
    second.start()
    cached = second.predict(request)
    assert cached.cached and cached.bundle_path == generated.bundle_path and len(calls["samplers"]) == 1
    request.regenerate = True
    regenerated = second.predict(request)
    assert not regenerated.cached and len(calls["samplers"]) == 2 and len(calls["infer"]) == 2
    second.close()


def test_virtual_scale_uses_sweep_descriptor_and_restores_original_local_translation(backend):
    make, request, calls, poses, _scores, config = backend
    predictor = make({"virtual_geometry_scale": 2., "topk_num_grasps": 2})
    predictor.start()
    result = predictor.predict(request)
    assert calls["samplers"][0][1] == "virtual"
    descriptor = calls["samplers"][0][2]["descriptor"]
    np.testing.assert_array_equal(descriptor["extents_open"], np.array(config["sweep_volume"]["extents"]) * 2)
    np.testing.assert_array_equal(calls["infer"][0][0], np.array([[.3, .4, .5], [.4, .5, .6]], np.float32) * 2)
    _poses, _confidence, metadata = load_grasp_bundle(result.bundle_path)
    assert metadata["model_metadata"]["virtual_sampler_route"] == "from_sweep_volume"
    with np.load(Path(result.bundle_path).with_name("candidate_pool.npz"), allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["grasps"][:, :3, 3], poses[:, :3, 3] / 2)
    predictor.close()


def test_real_mesh_sampling_applies_scale_once_without_recentering(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=[2, 4, 6])
    mesh.apply_translation([10, 20, 30])
    path = tmp_path / "offset.glb"
    mesh.export(path)
    points = sample_object_points(path, .01, 100, 7)
    np.testing.assert_array_equal(points, sample_object_points(path, .01, 100, 7))
    assert np.all(points >= np.array([.09, .18, .27], np.float32))
    assert np.all(points <= np.array([.11, .22, .33], np.float32))
    assert points.dtype == np.float32 and np.all(points.mean(axis=0) > [.09, .18, .27])


def test_nondefault_clip_compat_matches_old_clamped_x0_on_only_selected_scheduler():
    class Array(np.ndarray):
        @property
        def device(self):
            return "cpu"

        def to(self, device):
            assert device == "cpu"
            return self

        def sqrt(self):
            return np.sqrt(self)

        def clamp(self, low, high):
            return np.clip(self, low, high)

    def tensor(value):
        return np.asarray(value, dtype=float).view(Array)

    calls, changes = [], []
    scheduler = SimpleNamespace(alphas_cumprod={0: tensor(.25)},
                                register_to_config=lambda **values: changes.append(values),
                                step=lambda **values: calls.append(values) or values)
    original = scheduler.step
    generator = SimpleNamespace(noise_scheduler_pos=scheduler)
    _position_clip_compat(SimpleNamespace(model=SimpleNamespace(grasp_generator=generator)), .5)
    sample, epsilon = tensor([1, -2, .1]), tensor([0, 0, .1])
    scheduler.step(model_output=epsilon, timestep=0, sample=sample)
    expected = np.clip((sample - np.sqrt(.75) * epsilon) / .5, -.5, .5)
    np.testing.assert_array_equal(calls[0]["model_output"], expected)
    assert calls[0]["sample"] is sample and changes == [{"clip_sample": False, "prediction_type": "sample"}]
    native = SimpleNamespace(position_clip_sample_range=.5, noise_scheduler_pos=SimpleNamespace(step=original))
    _position_clip_compat(SimpleNamespace(model=SimpleNamespace(grasp_generator=native)), .5)
    assert native.noise_scheduler_pos.step is original


def test_import_and_constructor_do_not_load_torch_model_or_original_application():
    code = """
import sys
for name in ('torch', 'graspgenx', 'trimesh', 'omegaconf', 'object_pose_system'):
    sys.modules[name] = None
from grasp_workbench.models.graspgenx import OfficialGraspGenXPredictor
OfficialGraspGenXPredictor()
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=10)
    assert result.returncode == 0, result.stderr
