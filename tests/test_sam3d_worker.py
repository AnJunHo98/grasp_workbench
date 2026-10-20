"""On-demand jobs/cache tested without model, camera or native IPC services."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.sam3d import MeshGeometry, MeshRequest, services
from grasp_workbench.models.sam3d import Sam3dPrediction
from grasp_workbench.workers.worker_sam3d import Sam3dWorker, build_parser


def request(request_id="job-1", asset_id="../A/unsafe-looking-asset"):
    frame = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
                        np.ones((3, 4), dtype=np.float32),
                        {"camera_id": "camera", "stream_id": "stream", "sequence": 7,
                         "capture_time_ns": 123, "timestamp_domain": "camera-clock"})
    return MeshRequest(request_id, asset_id, "A", "revision", "red cup", 0,
                       frame, np.arange(12).reshape(3, 4) % 2 == 0)


def geometry(colors=True):
    return MeshGeometry(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                        np.array([[0, 1, 2]], np.uint32),
                        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8) if colors else None)


class Mailbox:
    def __init__(self):
        self.value, self.closed = None, False

    def receive_latest(self):
        value, self.value = self.value, None
        return value

    def close(self):
        self.closed = True


class Publisher:
    def __init__(self):
        self.results, self.closed, self.refreshes = [], False, 0

    def update_connections(self):
        self.refreshes += 1

    def publish(self, result):
        self.results.append(result)
        return 1

    def close(self):
        self.closed = True


class Predictor:
    def __init__(self):
        self.starts, self.calls, self.closed = 0, [], False
        self.output, self.error = geometry(), None
        self.pose = np.eye(4)
        self.pose[:3, 3] = [.1, .2, .7]
        self.during_predict = lambda: None

    def start(self):
        self.starts += 1

    def predict(self, frame, mask):
        self.calls.append((frame, mask))
        self.during_predict()
        if self.error is not None:
            raise self.error
        return Sam3dPrediction(self.output, self.pose)

    def close(self):
        self.closed = True


def worker(tmp_path, frames=0, predictor=None):
    inbox, output, predictor = Mailbox(), Publisher(), predictor or Predictor()
    factories = []

    def make_predictor(*args, **kwargs):
        factories.append((args, kwargs))
        return predictor

    instance = Sam3dWorker(
        WorkerContext("sam3d-test"), "model.yaml", "vendor/source", tmp_path, seed=17,
        service="example/rgbd", frames=frames,
        subscriber_factory=lambda service: factories.append(service) or inbox,
        publisher_factory=lambda service: factories.append(service) or output,
        predictor_factory=make_predictor,
    )
    return instance, inbox, output, predictor, factories


def send(instance, inbox, value):
    inbox.value = value
    instance.step_once()


def test_lazy_single_model_statuses_exact_input_safe_cache_and_finite_cleanup(tmp_path):
    instance, inbox, output, predictor, factories = worker(tmp_path, frames=2)
    original = request()
    instance.on_start()
    try:
        instance.step_once()
        assert factories == list(services("example/rgbd"))
        assert predictor.starts == 0 and not list(tmp_path.iterdir())
        send(instance, inbox, original)
        assert [item.status for item in output.results] == ["running", "done"]
        assert predictor.starts == 1 and len(predictor.calls) == 1
        assert predictor.calls[0][0] is original.frame
        assert predictor.calls[0][1] is original.mask
        assert factories[-1] == (("model.yaml",), {"source_dir": "vendor/source", "seed": 17})
        done = output.results[-1]
        assert (done.request_id, done.asset_id, done.object_id, done.revision) == (
            original.request_id, original.asset_id, original.object_id, original.revision)
        path = Path(done.mesh_path)
        assert path.parent == tmp_path and path.suffix == ".ply" and len(path.stem) == 64
        header = path.read_bytes().split(b"end_header\n", 1)[0]
        assert b"format binary_little_endian 1.0" in header
        assert b"element vertex 3" in header and b"element face 1" in header
        assert b"property uchar red" in header and b"units metre" in header
        assert done.length_unit == "metre" and done.mesh_scale_to_m == 1.
        np.testing.assert_array_equal(done.camera_from_object, predictor.pose)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            assert data["fingerprint"].dtype.kind == "U"
            np.testing.assert_array_equal(data["vertices"], predictor.output.vertices)
            assert data["schema_version"].item() == 2
            assert data["mesh_scale_to_m"].item() == 1. and data["length_unit"].item() == "metre"
            np.testing.assert_array_equal(data["camera_from_object"], predictor.pose)
        send(instance, inbox, replace(original, request_id="job-2", asset_id="second asset"))
        assert predictor.starts == 1 and len(predictor.calls) == 2
        assert instance.processed_count == 2 and instance.ctx.stop_event.is_set()
    finally:
        instance.on_stop()
    assert inbox.closed and output.closed and predictor.closed
    assert not list(tmp_path.glob("*.tmp"))


def test_persisted_matching_cache_never_constructs_or_loads_model(tmp_path):
    first, inbox, output, predictor, _ = worker(tmp_path)
    first.on_start()
    try:
        send(first, inbox, request())
    finally:
        first.on_stop()
    second, inbox, output, predictor, factories = worker(tmp_path)
    second.on_start()
    try:
        send(second, inbox, request("new-request-id"))
        assert [item.status for item in output.results] == ["running", "done"]
        assert predictor.starts == 0 and not predictor.calls
        assert factories == list(services("example/rgbd"))
        np.testing.assert_array_equal(output.results[-1].mesh.colors, geometry().colors)
        assert output.results[-1].mesh.vertices.flags.owndata
        assert output.results[-1].length_unit == "metre" and output.results[-1].mesh_scale_to_m == 1.
        np.testing.assert_array_equal(output.results[-1].camera_from_object, predictor.pose)
        assert output.results[-1].camera_from_object.flags.owndata
    finally:
        second.on_stop()


class OptionPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.options = []
        self.texture = b"glTF-baked-texture-and-UV-bytes"

    def predict(self, frame, mask, **options):
        self.options.append(options)
        prediction = super().predict(frame, mask)
        if options.get("texture_baking"):
            prediction.textured_glb = self.texture
        return prediction


def test_request_source_texture_options_have_separate_caches_and_honest_units(tmp_path):
    predictor = OptionPredictor()
    instance, inbox, output, predictor, _ = worker(tmp_path, predictor=predictor)
    instance.on_start()
    paths = []
    try:
        for index, (source, texture) in enumerate((("depth", False), ("depth", True),
                                                  ("monocular", False), ("monocular", True))):
            selected = replace(request(str(index)), depth_source=source, texture_baking=texture)
            send(instance, inbox, selected)
            done = output.results[-1]
            assert done.status == "done", done.error
            path = Path(done.mesh_path)
            paths.append(path)
            assert path.suffix == (".glb" if texture else ".ply")
            unit = "metre" if source == "depth" else "monocular_predicted_scale"
            assert done.length_unit == unit and done.mesh_scale_to_m == 1
            if texture:
                assert path.read_bytes() == predictor.texture
            assert ("units " + unit).encode() in path.with_suffix(".ply").read_bytes()
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                assert data["depth_source"].item() == source and data["texture_baking"].item() == texture
                assert data["length_unit"].item() == unit
        assert len(set(paths)) == 4 and len(predictor.calls) == 4
        assert predictor.options == [{}, {"texture_baking": True}, {"depth_source": "monocular"},
                                      {"depth_source": "monocular", "texture_baking": True}]
        # Revisiting an option reuses its own cache rather than the last option.
        send(instance, inbox, request("depth-again"))
        assert len(predictor.calls) == 4 and output.results[-1].mesh_path == str(paths[0])
    finally:
        instance.on_stop()


def test_textured_cache_preserves_glb_bytes_and_restores_missing_asset_without_model_load(tmp_path):
    selected = replace(request(), depth_source="monocular", texture_baking=True)
    first, inbox, output, predictor, _ = worker(tmp_path, predictor=OptionPredictor())
    first.on_start()
    try:
        send(first, inbox, selected)
        path = Path(output.results[-1].mesh_path)
        assert path.read_bytes() == predictor.texture
        path.unlink()
    finally:
        first.on_stop()
    second, inbox, output, predictor, factories = worker(tmp_path)
    second.on_start()
    try:
        send(second, inbox, replace(selected, request_id="cached"))
        done = output.results[-1]
        assert done.status == "done" and done.length_unit == "monocular_predicted_scale"
        assert Path(done.mesh_path).read_bytes() == b"glTF-baked-texture-and-UV-bytes"
        assert predictor.starts == 0 and not predictor.calls and factories == list(services("example/rgbd"))
    finally:
        second.on_stop()


def test_texture_request_cannot_report_vertex_only_ply_as_a_baked_asset(tmp_path):
    predictor = OptionPredictor()
    predictor.texture = None
    instance, inbox, output, predictor, _ = worker(tmp_path, predictor=predictor)
    instance.on_start()
    try:
        send(instance, inbox, replace(request(), texture_baking=True))
        assert output.results[-1].status == "error" and "textured GLB" in output.results[-1].error
        assert not list(tmp_path.iterdir())
    finally:
        instance.on_stop()


def test_legacy_uncalibrated_cache_is_regenerated_in_metric_coordinates(tmp_path):
    instance, inbox, output, predictor, _ = worker(tmp_path)
    instance.on_start()
    try:
        original = request()
        send(instance, inbox, original)
        path, _ply = instance._paths(original.asset_id)
        with np.load(path, allow_pickle=False) as data:
            legacy = {key: data[key].copy() for key in ("fingerprint", "vertices", "faces", "has_colors", "colors")}
        np.savez(path, **legacy)
        send(instance, inbox, replace(original, request_id="metric-regeneration"))
        assert len(predictor.calls) == 2
        assert output.results[-1].status == "done" and output.results[-1].length_unit == "metre"
        with np.load(path, allow_pickle=False) as data:
            assert data["schema_version"].item() == 2
            np.testing.assert_array_equal(data["camera_from_object"], predictor.pose)
    finally:
        instance.on_stop()


@pytest.mark.parametrize("field", ["revision", "prompt", "candidate_index", "object_id", "sequence", "mask", "rgb", "depth"])
def test_changed_input_cannot_reuse_same_asset_cache(tmp_path, field):
    instance, inbox, output, predictor, _ = worker(tmp_path)
    instance.on_start()
    try:
        send(instance, inbox, request())
        changed = request("changed")
        if field in ("revision", "prompt", "object_id"):
            setattr(changed, field, "different")
        elif field == "candidate_index":
            changed.candidate_index = 1
        elif field == "sequence":
            changed.frame.metadata["sequence"] += 1
        elif field == "mask":
            changed.mask[0, 0] = False
        elif field == "rgb":
            changed.frame.rgb[0, 0, 0] += 1
        else:
            changed.frame.depth_m[0, 0] += 1
        send(instance, inbox, changed)
        assert len(predictor.calls) == 2
        assert output.results[-1].status == "done"
    finally:
        instance.on_stop()


def test_duplicate_request_does_not_infer_or_count_again(tmp_path):
    instance, inbox, output, predictor, _ = worker(tmp_path)
    instance.on_start()
    try:
        original = request()
        send(instance, inbox, original)
        terminal = output.results[-1]
        send(instance, inbox, original)
        assert output.results[-1] is terminal
        assert len(predictor.calls) == instance.processed_count == 1
        send(instance, inbox, replace(original, asset_id="conflicting-asset"))
        assert output.results[-1].status == "error"
        assert output.results[-1].asset_id == "conflicting-asset"
        assert output.results[-1].mesh is None
        send(instance, inbox, request("second", "other-asset"))
        send(instance, inbox, original)
        assert output.results[-1].status == "error"
        assert len(predictor.calls) == instance.processed_count == 2
    finally:
        instance.on_stop()


def test_empty_mask_and_object_failure_are_terminal_and_next_job_still_runs(tmp_path):
    instance, inbox, output, predictor, _ = worker(tmp_path)
    instance.on_start()
    try:
        empty = request()
        empty.mask[:] = False
        send(instance, inbox, empty)
        assert output.results[-1].status == "error" and "foreground" in output.results[-1].error
        assert predictor.starts == 0
        predictor.error = RuntimeError("one object failed")
        send(instance, inbox, request("failed"))
        assert output.results[-1].error == "RuntimeError: one object failed"
        assert not list(tmp_path.glob("*.npz"))
        predictor.error = None
        predictor.output = geometry(colors=False)
        send(instance, inbox, request("successful"))
        assert output.results[-1].status == "done"
        assert predictor.starts == 1 and len(predictor.calls) == 2
        assert output.results[-1].mesh.colors is None
    finally:
        instance.on_stop()


def test_invalid_geometry_is_an_error_not_a_saved_asset(tmp_path):
    instance, inbox, output, predictor, _ = worker(tmp_path)
    predictor.output.vertices[0, 0] = np.nan
    inbox.value = request()
    instance.frames = 1
    instance.run()
    assert [item.status for item in output.results] == ["running", "error"]
    assert "finite" in output.results[-1].error
    assert not list(tmp_path.iterdir())
    assert inbox.closed and output.closed and predictor.closed


def test_start_failure_is_closed_and_later_job_can_retry(tmp_path):
    instance, inbox, output, predictor, _ = worker(tmp_path)

    def fail_start():
        raise RuntimeError("load failed")

    predictor.start = fail_start
    instance.on_start()
    try:
        send(instance, inbox, request())
        assert output.results[-1].status == "error" and "load failed" in output.results[-1].error
        assert predictor.closed and instance.predictor is None
        replacement = Predictor()
        instance.predictor_factory = lambda *_args, **_kwargs: replacement
        send(instance, inbox, request("retry"))
        assert output.results[-1].status == "done" and replacement.starts == 1
    finally:
        instance.on_stop()
    assert replacement.closed


def test_corrupt_cache_reconstructs_and_missing_ply_restores_without_model(tmp_path):
    instance, inbox, output, predictor, _ = worker(tmp_path)
    instance.on_start()
    try:
        original = request()
        npz, ply = instance._paths(original.asset_id)
        npz.write_bytes(b"interrupted file")
        send(instance, inbox, original)
        assert output.results[-1].status == "done" and len(predictor.calls) == 1
        ply.unlink()
        send(instance, inbox, replace(original, request_id="restore-ply"))
        assert ply.is_file() and len(predictor.calls) == 1
    finally:
        instance.on_stop()


def test_stop_during_prediction_skips_terminal_save_and_closes_resources(tmp_path):
    instance, inbox, output, predictor, _ = worker(tmp_path)
    inbox.value = request()
    predictor.during_predict = instance.ctx.stop
    instance.run()
    assert [item.status for item in output.results] == ["running"]
    assert instance.processed_count == 0 and not list(tmp_path.iterdir())
    assert inbox.closed and output.closed and predictor.closed


def test_worker_import_is_model_sdk_and_native_binding_lazy():
    script = """
import sys
for name in ('torch', 'sam3d_objects', 'PIL', 'pyzed', 'pyrealsense2', 'iceoryx2'):
    sys.modules[name] = None
from grasp_workbench.workers.worker_sam3d import Sam3dWorker
from grasp_workbench.core.worker_base import WorkerContext
Sam3dWorker(WorkerContext('test'), 'model.yaml', 'source', 'output')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    args = build_parser().parse_args(["--config-path", "model.yaml", "--source-dir", "source",
                                     "--output-dir", "output"])
    assert args.seed == 42 and args.frames == 0
