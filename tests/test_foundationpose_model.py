from contextlib import nullcontext
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.models.foundationpose import (
    OfficialFoundationPosePredictor, _pose_from_output, visual_pose_mismatch_reason,
)


@pytest.fixture
def backend(monkeypatch, tmp_path):
    calls = []
    meshes = []

    class Mesh:
        def __init__(self):
            self.vertices = np.ones((3, 3))
            self.vertex_normals = np.ones((3, 3))

        def apply_scale(self, scale):
            self.vertices *= scale

    class Session:
        def __init__(self, **kwargs):
            calls.append(("session", kwargs))
            self.diameter = np.float64(.2)
            self.mesh_path = str(tmp_path / f"native-{len(calls)}.obj")
            Path(self.mesh_path).touch()

        def register(self, **kwargs):
            assert type(self.diameter) is float
            calls.append(("register", kwargs))
            return np.eye(4, dtype=np.float32)

        def track_one(self, **kwargs):
            calls.append(("track", kwargs))
            return np.eye(4, dtype=np.float32)

    def load(path, **kwargs):
        calls.append(("load", path, kwargs))
        mesh = Mesh()
        meshes.append(mesh)
        return mesh

    torch = SimpleNamespace(cuda=SimpleNamespace(set_device=lambda value: calls.append(("device", value))),
                            inference_mode=nullcontext)
    native = SimpleNamespace(ScorePredictor=lambda: object(), PoseRefinePredictor=lambda: object(),
                             dr=SimpleNamespace(RasterizeCudaContext=lambda device: ("renderer", device)),
                             FoundationPose=Session)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "estimater", native)
    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace(load=load))
    monkeypatch.setattr(sys, "path", list(sys.path))
    return calls, meshes


def frame():
    return CameraFrame(np.zeros((2, 3, 3), np.uint8), np.full((2, 3), .7, np.float32),
                       {"intrinsics": {"fx": 100., "fy": 110., "cx": 1., "cy": .5}})


def test_model_import_is_lazy():
    result = subprocess.run([sys.executable, "-c", "import sys; "
                             "import grasp_workbench.models.foundationpose; "
                             "assert not any(x in sys.modules for x in ('torch','estimater','trimesh'))"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_registers_two_sessions_with_shared_models_and_scaled_geometry(backend):
    calls, meshes = backend
    model = OfficialFoundationPosePredictor("example/backend", registration_iterations=3)
    model.start()
    source, mask = frame(), np.ones((2, 3), bool)
    model.register("A", source, mask, "a.glb", .001)
    model.register("B", source, mask, "b.glb", 1.)
    sessions = [call[1] for call in calls if call[0] == "session"]
    assert sessions[0]["scorer"] is sessions[1]["scorer"]
    assert sessions[0]["refiner"] is sessions[1]["refiner"]
    assert sessions[0]["glctx"] is sessions[1]["glctx"]
    np.testing.assert_allclose(meshes[0].vertices, .001)
    np.testing.assert_allclose(meshes[1].vertices, 1.)
    register = next(call[1] for call in calls if call[0] == "register")
    assert register["rgb"] is source.rgb and register["depth"] is source.depth_m
    assert register["ob_mask"] is mask and register["iteration"] == 3
    np.testing.assert_array_equal(register["K"], [[100, 0, 1], [0, 110, .5], [0, 0, 1]])
    model.close()


def test_track_uses_retained_session_and_configured_iterations(backend):
    calls, _ = backend
    model = OfficialFoundationPosePredictor("example/backend", tracking_iterations=2)
    model.start()
    source = frame()
    model.register("A", source, np.ones((2, 3), bool), "a.glb", 1.)
    for _ in range(3):
        pose = model.track("A", source)
        assert pose.shape == (4, 4) and pose.dtype == np.float64
    assert len([call for call in calls if call[0] == "session"]) == 1
    assert len([call for call in calls if call[0] == "register"]) == 1
    tracks = [call[1] for call in calls if call[0] == "track"]
    assert len(tracks) == 3 and all(call["iteration"] == 2 for call in tracks)
    assert all("ob_mask" not in call for call in tracks)
    model.close()


def test_remove_and_close_release_sessions_and_native_mesh_files(backend):
    model = OfficialFoundationPosePredictor("example/backend")
    model.start()
    for key in ("A", "B"):
        model.register(key, frame(), np.ones((2, 3), bool), f"{key}.glb", 1.)
    paths = {key: Path(session.mesh_path) for key, session in model._sessions.items()}
    model.remove("A")
    assert not paths["A"].exists() and paths["B"].exists()
    model.close()
    assert not paths["B"].exists() and not model._sessions
    assert model._scorer is model._refiner is model._glctx is None


def test_module_can_be_imported_without_backend_setup():
    assert importlib.import_module("grasp_workbench.models.foundationpose").OfficialFoundationPosePredictor


def test_reregistration_reuses_native_object_session_and_mesh(backend):
    calls, _ = backend
    model = OfficialFoundationPosePredictor("example/backend")
    model.start()
    for _ in range(2):
        model.register("A", frame(), np.ones((2, 3), bool), "a.glb", 1.)
    assert len([call for call in calls if call[0] == "session"]) == 1
    assert len([call for call in calls if call[0] == "load"]) == 1
    assert len([call for call in calls if call[0] == "register"]) == 2
    model.close()


@pytest.mark.parametrize("field,value", [("nonfinite", np.nan), ("bottom", .1), ("rotation", 1.01)])
def test_pose_output_keeps_previous_ui_se3_validation(field, value):
    pose = np.eye(4)
    if field == "nonfinite":
        pose[0, 3] = value
    elif field == "bottom":
        pose[3, 0] = value
    else:
        pose[0, 0] = value
    with pytest.raises(ValueError, match="SE\\(3\\)"):
        _pose_from_output(pose)
    pose = np.eye(4)
    pose[0, 0] += .0002
    assert _pose_from_output(pose).shape == (4, 4)  # Same TF32 tolerance as the old adapter.


def test_tracking_translation_jump_uses_previous_ui_quarter_metre_limit(backend):
    model = OfficialFoundationPosePredictor("example/backend")
    model.start()
    source = frame()
    model.register("A", source, np.ones((2, 3), bool), "a.glb", 1.)
    pose = np.eye(4)
    pose[0, 3] = .251
    model._sessions["A"].track_one = lambda **kwargs: pose
    with pytest.raises(ValueError, match="translation jump"):
        model.track("A", source)
    assert model._sessions["A"].pose_last is None
    model.register("A", source, np.ones((2, 3), bool), "a.glb", 1.)
    pose[0, 3] = .25
    np.testing.assert_array_equal(model.track("A", source), pose)
    model.close()


def test_visual_gate_uses_projected_origin_bbox_and_mask_depth_range_with_minimum_pixels():
    source = CameraFrame(np.zeros((6, 8, 3), np.uint8), np.full((6, 8), 1, np.float32),
                         {"intrinsics": {"fx": 100, "fy": 100, "cx": 4, "cy": 3}})
    pose, mask = np.eye(4), np.ones((6, 8), bool)
    pose[2, 3] = 1
    assert visual_pose_mismatch_reason(source, pose, mask, (0, 0, 8, 6)) is None
    pose[0, 3] = 1
    assert visual_pose_mismatch_reason(source, pose, mask, (0, 0, 8, 6)) == "visual_mismatch:bbox"
    pose[0, 3], pose[2, 3] = 0, 1.26
    assert visual_pose_mismatch_reason(source, pose, mask, (0, 0, 8, 6)) == "visual_mismatch:depth"
    mask[:] = False
    mask.flat[:15] = True
    assert visual_pose_mismatch_reason(source, pose, mask, (0, 0, 8, 6)) is None
    # This is min/max, not a median or centre-depth test.
    mask[:] = True
    source.depth_m[0, 0] = 1.4
    assert visual_pose_mismatch_reason(source, pose, mask, (0, 0, 8, 6)) is None
