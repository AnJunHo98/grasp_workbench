"""Official pipeline boundary with fake weights; no CUDA/backend is imported."""
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.models.sam3d import (
    MOGE_MODEL_REVISION, OfficialSam3dPredictor, depth_to_pointmap, metric_vertices_and_pose,
)


INTRINSICS = {"fx": 2., "fy": 4., "cx": 1., "cy": 1.}


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    mesh = SimpleNamespace(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                           faces=np.array([[0, 1, 2]], dtype=np.int64),
                           visual=SimpleNamespace(vertex_colors=np.full((3, 4), 123, dtype=np.uint8)))
    calls = []

    class Pipeline:
        def run(self, rgba, **kwargs):
            calls.append((rgba.copy(), kwargs))
            return {"glb": mesh, "rotation": [1, 0, 0, 0], "translation": [.1, -.2, 1.5],
                    "scale": [.1, .2, .3]}

    config, created = SimpleNamespace(), []
    hydra = ModuleType("hydra.utils")
    hydra.instantiate = lambda loaded: created.append(loaded) or Pipeline()
    omega = ModuleType("omegaconf")
    omega.OmegaConf = SimpleNamespace(load=lambda path: config)
    monkeypatch.setitem(sys.modules, hydra.__name__, hydra)
    monkeypatch.setitem(sys.modules, omega.__name__, omega)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda data: data))
    monkeypatch.setitem(sys.modules, "sam3d_objects.pipeline.inference_pipeline_pointmap", ModuleType("fake-pipeline"))
    predictor = OfficialSam3dPredictor(tmp_path / "checkpoints/pipeline.yaml", source_dir=tmp_path / "vendor", seed=7)
    return predictor, config, created, calls, mesh


def test_lazy_one_load_exact_rgbd_mask_metric_asset_and_initial_camera_pose(pipeline):
    predictor, config, created, calls, asset = pipeline
    assert not created and predictor._pipeline is None
    predictor.start()
    predictor.start()
    assert len(created) == 1 and created[0] is config
    assert config.workspace_dir == str(predictor.config_path.parent)
    assert config.compile_model is False and config.rendering_engine == "pytorch3d"
    assert config.depth_model is None
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    mask = np.zeros((3, 4), dtype=bool)
    mask[1:, 1:3] = True
    depth = np.full((3, 4), .75, np.float32)
    depth[0, 0] = np.nan
    frame = CameraFrame(rgb.copy(), depth, {"sequence": 8, "intrinsics": INTRINSICS})
    result = predictor.predict(frame, mask)
    rgba, options = calls[0]
    np.testing.assert_array_equal(rgba[:, :, :3], rgb)
    np.testing.assert_array_equal(rgba[:, :, 3], mask.astype(np.uint8) * 255)
    np.testing.assert_array_equal(frame.rgb, rgb)
    np.testing.assert_array_equal(options.pop("pointmap"),
                                  depth_to_pointmap(depth, [[2, 0, 1], [0, 4, 1], [0, 0, 1]]))
    assert options == {"seed": 7, "stage1_only": False,
                       "with_mesh_postprocess": False, "with_texture_baking": False,
                       "with_layout_postprocess": False, "use_vertex_color": True}
    np.testing.assert_array_equal(result.mesh.vertices, np.array([[0, 0, 0], [.1, 0, 0], [0, 0, .3]], np.float32))
    np.testing.assert_array_equal(result.camera_from_object,
                                  [[-1, 0, 0, -.1], [0, -1, 0, .2], [0, 0, 1, 1.5], [0, 0, 0, 1]])
    assert result.mesh.vertices.dtype == np.float32 and result.mesh.faces.dtype == np.uint32
    assert result.mesh.colors.shape == (3, 3)
    asset.vertices[:] = 10
    asset.visual.vertex_colors[:] = 0
    assert result.mesh.vertices.max() == np.float32(.3) and result.mesh.colors.min() == 123
    predictor.close()
    assert predictor._pipeline is None


@pytest.mark.parametrize("mask", [np.zeros((3, 4), bool), np.ones((4, 3), bool), np.ones((3, 4), np.uint8)])
def test_invalid_selected_mask_does_not_invoke_pipeline(pipeline, mask):
    predictor, _config, _created, calls, _asset = pipeline
    predictor.start()
    frame = CameraFrame(np.zeros((3, 4, 3), np.uint8), np.zeros((3, 4), np.float32), {"intrinsics": INTRINSICS})
    with pytest.raises(ValueError, match="mask"):
        predictor.predict(frame, mask)
    assert not calls


def test_missing_mesh_is_an_error_and_no_colors_is_supported(pipeline):
    predictor, _config, _created, _calls, asset = pipeline
    frame = CameraFrame(np.zeros((3, 4, 3), np.uint8), np.zeros((3, 4), np.float32), {"intrinsics": INTRINSICS})
    mask = np.ones((3, 4), bool)
    with pytest.raises(RuntimeError, match="Start"):
        predictor.predict(frame, mask)
    predictor.start()
    asset.visual = None
    assert predictor.predict(frame, mask).mesh.colors is None
    predictor._pipeline.run = lambda *args, **kwargs: {"glb": None}
    with pytest.raises(ValueError, match="no mesh"):
        predictor.predict(frame, mask)


def test_adapter_import_does_not_load_heavy_backend():
    script = """
import sys
for name in ('torch', 'sam3d_objects', 'hydra', 'omegaconf', 'pyzed', 'pyrealsense2'):
    sys.modules[name] = None
from grasp_workbench.models.sam3d import OfficialSam3dPredictor
OfficialSam3dPredictor('outside-checkpoint/pipeline.yaml')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_depth_pointmap_uses_measured_metres_optical_intrinsics_and_invalid_nan():
    depth = np.array([[1, .5, 0], [2, np.inf, np.nan]], np.float32)
    points = depth_to_pointmap(depth, [[2, 0, 1], [0, 4, 1], [0, 0, 1]])
    np.testing.assert_array_equal(points[0, :2], [[.5, .25, 1], [0, .125, .5]])
    np.testing.assert_array_equal(points[1, 0], [1, 0, 2])
    assert np.isnan(points[0, 2]).all() and np.isnan(points[1, 1:]).all()
    assert points.dtype == np.float32


@pytest.mark.parametrize("scale", [[.1], [.1, .2, .3]])
def test_metric_conversion_preserves_canonical_axes_bakes_scale_once_and_rigid_pose(scale):
    canonical = np.array([[1, 2, 3], [4, 5, 6]], np.float64)
    axis_change = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    glb = canonical @ axis_change
    # A normalized 90-degree z quaternion; upstream pose rotation uses row vectors.
    quaternion = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    metric, pose = metric_vertices_and_pose(glb, quaternion, [.2, -.4, 1.2], scale)
    np.testing.assert_allclose(metric, canonical * scale)
    np.testing.assert_allclose(pose[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-14)
    np.testing.assert_array_equal(pose[:3, 3], [-.2, .4, 1.2])
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-14)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.)
    # Camera geometry matches the upstream scaled-row transform, then optical-axis conversion.
    row_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = ((canonical * scale) @ row_rotation + [.2, -.4, 1.2]) * [-1, -1, 1]
    np.testing.assert_allclose(metric @ pose[:3, :3].T + pose[:3, 3], expected, atol=1e-7)


def test_monocular_model_is_loaded_lazily_with_pinned_revision_and_no_measured_depth_claim(pipeline, monkeypatch):
    predictor, _config, _created, calls, _asset = pipeline
    predictor.start()
    predictor._pipeline.depth_model = None
    depth_config = SimpleNamespace(model=SimpleNamespace())
    config = SimpleNamespace(depth_model=depth_config)
    instantiated = []
    model = object()
    monkeypatch.setattr(sys.modules["omegaconf"].OmegaConf, "load", lambda path: config)
    monkeypatch.setattr(sys.modules["hydra.utils"], "instantiate", lambda loaded: instantiated.append(loaded) or model)
    frame = CameraFrame(np.zeros((3, 4, 3), np.uint8), np.zeros((3, 4), np.float32), {})
    for _ in range(2):
        prediction = predictor.predict(frame, np.ones((3, 4), bool), depth_source="monocular")
        assert prediction.length_unit == "monocular_predicted_scale"
    assert instantiated == [depth_config]
    assert depth_config.model.revision == MOGE_MODEL_REVISION
    assert predictor._pipeline.depth_model is model and all(kwargs["pointmap"] is None for _, kwargs in calls)


def test_baked_glb_preserves_texture_uv_and_converted_vertices_without_mutating_upstream(pipeline, monkeypatch):
    import io

    trimesh = pytest.importorskip("trimesh")
    image = pytest.importorskip("PIL.Image")
    module = sys.modules[OfficialSam3dPredictor.__module__]
    contexts = []
    monkeypatch.setattr(module, "_gsplat_texture_rendering", lambda: contexts.append("gsplat") or nullcontext())
    predictor, _config, _created, calls, _asset = pipeline
    predictor.start()
    asset = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    asset.visual = trimesh.visual.texture.TextureVisuals(uv=[[0, 0], [1, 0], [0, 1]],
                                                        image=image.new("RGB", (2, 2), (220, 30, 10)))
    original_vertices, original_uv = asset.vertices.copy(), asset.visual.uv.copy()

    def run(rgba, **kwargs):
        calls.append((rgba, kwargs))
        return {"glb": asset, "rotation": [1, 0, 0, 0], "translation": [0, 0, 1], "scale": [.1, .2, .3]}

    predictor._pipeline.run = run
    frame = CameraFrame(np.zeros((3, 4, 3), np.uint8), np.ones((3, 4), np.float32), {"intrinsics": INTRINSICS})
    prediction = predictor.predict(frame, np.ones((3, 4), bool), texture_baking=True)
    assert contexts == ["gsplat"] and calls[0][1]["with_texture_baking"] is True
    assert calls[0][1]["use_vertex_color"] is False and prediction.textured_glb[:4] == b"glTF"
    restored = trimesh.load(io.BytesIO(prediction.textured_glb), file_type="glb", force="mesh", process=False)
    np.testing.assert_allclose(restored.vertices, prediction.mesh.vertices)
    np.testing.assert_allclose(restored.visual.uv, original_uv)
    assert restored.visual.kind == "texture" and restored.visual.material is not None
    np.testing.assert_array_equal(asset.vertices, original_vertices)
    np.testing.assert_array_equal(asset.visual.uv, original_uv)


def test_gsplat_texture_context_preserves_other_options_and_restores_renderer_on_failure(monkeypatch):
    from grasp_workbench.models.sam3d import _gsplat_texture_rendering

    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "rendered"

    render_utils = SimpleNamespace(render_frames=original)
    monkeypatch.setitem(sys.modules, "sam3d_objects.model.backbone.tdfy_dit.utils",
                        SimpleNamespace(render_utils=render_utils))
    with pytest.raises(RuntimeError, match="actual rendering failure"):
        with _gsplat_texture_rendering():
            assert render_utils.render_frames("sample", "extrinsics", "intrinsics", {"size": 32}, other=True) == "rendered"
            assert calls[-1][0][3] == {"size": 32, "backend": "gsplat"}
            assert calls[-1][1] == {"other": True}
            raise RuntimeError("actual rendering failure")
    assert render_utils.render_frames is original
