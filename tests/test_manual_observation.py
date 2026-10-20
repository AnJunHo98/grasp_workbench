"""Frozen observations, real SAM3 prompt math and explicit FP seed initialization."""
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.ipc.foundationpose import (
    ObjectPose, PoseControl, PoseResult, decode_control, encode_control,
)
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result
from grasp_workbench.ipc.sam3_refine import (
    RefineRequest, RefineResult, decode_request, decode_result, encode_request, encode_result,
)
from grasp_workbench.models.foundationpose import OfficialFoundationPosePredictor
from grasp_workbench.models.sam3 import OfficialSam3Predictor, boxes_from_masks, normalized_box, seed_mask_logits
from grasp_workbench.snapshot import load_snapshot, save_snapshot
from test_foundationpose_model import backend as fp_backend_fixture
from test_foundationpose_worker import frame, target, worker as fp_worker
from test_sam3_worker import official_fakes as sam3_fakes_fixture

fp_backend, sam3_fakes = fp_backend_fixture, sam3_fakes_fixture


def sam3_result(source):
    mask = np.zeros((1, *source.rgb.shape[:2]), bool)
    mask[:, 1:, 1:3] = True
    return Sam3Result(source, "revision", (Sam3ObjectResult(
        "B", "yellow can", mask, np.array([[1, 1, 3, 3]], np.float32), np.array([.9], np.float32)),))


def test_snapshot_round_trip_preserves_each_exact_source_not_latest_display_frame(tmp_path):
    latest, segmented, tracked = frame(100), frame(37), frame(42)
    latest.depth_m[0, 0] = np.nan
    sam3 = sam3_result(segmented)
    poses = PoseResult(tracked, (ObjectPose("B", "session", "object.ply", "tracking", np.eye(4)),))
    objects = ({"object_id": "B", "mesh_path": "object.ply", "mesh_scale_to_m": 1.},)
    semantic = {"task_binding": {"source_observation_id": "source"}, "robot_command_published": False}
    path = save_snapshot(tmp_path / "snapshot.npz", latest, sam3=sam3, pose_result=poses, objects=objects, semantic=semantic)
    saved = load_snapshot(path)
    assert saved.frame.metadata["sequence"] == 100
    assert saved.sam3.frame.metadata["sequence"] == 37 and saved.pose_result.frame.metadata["sequence"] == 42
    np.testing.assert_array_equal(saved.frame.depth_m, latest.depth_m)
    np.testing.assert_array_equal(saved.sam3.objects[0].masks, sam3.objects[0].masks)
    assert saved.objects == objects and saved.semantic == semantic
    saved.sam3.objects[0].masks[:] = False
    assert sam3.objects[0].masks.any()
    path = save_snapshot(tmp_path / "only-rgbd.npz", latest)
    saved = load_snapshot(path)
    assert saved.sam3 is saved.pose_result is saved.semantic is None and saved.objects == ()


def test_fp_seed_wire_keeps_initial_frame_matrix_and_no_unrelated_mask():
    initial = np.eye(4)
    initial[:3, 3] = (.1, -.2, .8)
    seeded = replace(target(sequence=37), initial_pose=initial, initialization="seed", mask=None)
    decoded, = decode_control(encode_control(PoseControl((seeded,)))).targets
    assert decoded.frame.metadata["sequence"] == 37 and decoded.initialization == "seed" and decoded.mask is None
    np.testing.assert_array_equal(decoded.initial_pose, initial)
    with pytest.raises(ValueError, match="mask"):
        encode_control(PoseControl((replace(seeded, initialization="register"),)))
    with pytest.raises(ValueError, match="initial"):
        encode_control(PoseControl((replace(seeded, initial_pose=None),)))


def test_fp_seed_adopts_centered_mesh_inverse_before_tracking_without_registration(fp_backend):
    calls, _ = fp_backend
    centered = np.eye(4)
    centered[:3, 3] = (.2, -.3, .4)
    native = sys.modules["estimater"].FoundationPose
    native.get_tf_to_centered_mesh = lambda self: centered
    model = OfficialFoundationPosePredictor("example/backend")
    model.start()
    model._torch.float32 = np.float32
    model._torch.as_tensor = lambda value, **kwargs: np.asarray(value).copy()
    initial = np.eye(4)
    initial[:3, 3] = (.1, .2, .7)
    model.seed("A", initial, "a.glb", .001)
    np.testing.assert_allclose(model._sessions["A"].pose_last, initial @ np.linalg.inv(centered))
    assert not [value for value in calls if value[0] == "register"]
    assert len([value for value in calls if value[0] == "session"]) == 1
    track = model._sessions["A"].track_one

    def seeded_track(**kwargs):
        track(**kwargs)
        return initial.copy()

    model._sessions["A"].track_one = seeded_track
    model.track("A", frame())
    assert len([value for value in calls if value[0] == "track"]) == 1
    model.close()


def test_seeded_worker_tracks_exact_seed_frame_then_recovers_by_mask_registration():
    instance, camera, control, output, predictor, _ = fp_worker()
    adopted = []
    predictor.seed = lambda *args: adopted.append(args)
    seeded = replace(target(sequence=37), mask=None, initialization="seed", initial_pose=np.eye(4))
    try:
        control.value = PoseControl((seeded,))
        instance.step_once()
        assert len(adopted) == 1 and not predictor.registers
        assert predictor.tracks[0][1] is seeded.frame and output.results[0].frame is seeded.frame
        assert instance._associations["A"].accepted_box is None
        current = instance._register(seeded, frame(100), np.ones((3, 4), bool))
        assert current.status == "tracking" and len(adopted) == 1 and len(predictor.registers) == 1
    finally:
        instance.on_stop()


def refine_request():
    source = frame(37)
    return RefineRequest("refine", "B", "yellow can", source,
                         ({"xyxy": [1, 1, 3, 3], "label": True},),
                         ({"xy": [2, 2], "label": 1}, {"xy": [0, 0], "label": 0}),
                         sam3_result(source).objects[0].masks[0])


def test_refine_codec_preserves_frozen_source_seed_and_pixels():
    request = refine_request()
    decoded = decode_request(encode_request(request))
    assert decoded.request_id == request.request_id and decoded.frame.metadata == request.frame.metadata
    assert decoded.points == request.points and decoded.boxes == request.boxes
    np.testing.assert_array_equal(decoded.seed_mask, request.seed_mask)
    decoded.seed_mask[:] = False
    assert request.seed_mask.any()
    for result in (RefineResult("refine", "running"), RefineResult("refine", "done", sam3_result(frame())),
                   RefineResult("refine", "error", error="inference failed")):
        copied = decode_result(encode_result(result))
        assert copied.status == result.status and copied.error == result.error
        if copied.result is not None:
            np.testing.assert_array_equal(copied.result.objects[0].masks, result.result.objects[0].masks)


@pytest.mark.parametrize("change", [
    {"seed_mask": np.ones((2, 2), bool)}, {"points": ({"xy": [1, 2], "label": 2},)},
    {"boxes": ({"xyxy": [3, 1, 1, 2], "label": True},)},
])
def test_refine_contract_rejects_different_frame_masks_and_invalid_prompts(change):
    with pytest.raises(ValueError):
        encode_request(replace(refine_request(), **change))


def test_interactive_prompt_math_matches_old_pixel_geometry_and_seed_head_resolution():
    np.testing.assert_allclose(normalized_box((10, 20, 30, 60), 100, 200), (.2, .2, .2, .2))
    masks = np.zeros((2, 8, 10), bool)
    masks[0, 2:5, 3:7] = True
    np.testing.assert_array_equal(boxes_from_masks(masks), [[3, 2, 7, 5], [0, 0, 0, 0]])
    assert seed_mask_logits(masks[0], (2, 3)).shape == (1, 2, 3)
    assert seed_mask_logits(np.ones((2, 2), bool), (2, 2)).min() == 10


def test_point_refinement_uses_selected_seed_without_redetecting_text_instance(sam3_fakes):
    checkpoint, calls = sam3_fakes
    predictor = OfficialSam3Predictor(checkpoint, device="cpu")
    predictor.start()
    invoked = []
    predictor.model.inst_interactive_predictor = SimpleNamespace(model=SimpleNamespace(
        sam_prompt_encoder=SimpleNamespace(mask_input_size=(2, 2))))

    def predict_inst(state, **kwargs):
        invoked.append(kwargs)
        return np.ones((1, 3, 4), bool), np.array([.8], np.float32), None

    predictor.model.predict_inst = predict_inst
    source = refine_request()
    original = source.seed_mask.copy()
    result = predictor.refine(source)
    assert result.frame is source.frame and result.revision == source.request_id
    assert calls.processors[0].texts == [] and len(calls.processors[0].images) == 1
    assert len(invoked) == 1 and invoked[0]["multimask_output"] is False and invoked[0]["normalize_coords"] is True
    np.testing.assert_array_equal(invoked[0]["point_coords"], [[2, 2], [0, 0]])
    np.testing.assert_array_equal(invoked[0]["point_labels"], [1, 0])
    np.testing.assert_array_equal(invoked[0]["box"], [1, 1, 3, 3])
    assert invoked[0]["mask_input"].shape == (1, 2, 2)
    np.testing.assert_array_equal(source.seed_mask, original)
    predictor.close()


def test_box_refinement_keeps_normalized_geometric_prompt_and_label(sam3_fakes):
    checkpoint, calls = sam3_fakes
    predictor = OfficialSam3Predictor(checkpoint, device="cpu")
    predictor.start()
    invoked = []
    processor = calls.processors[0]

    def geometric(**kwargs):
        invoked.append(kwargs)
        return {"masks": np.ones((1, 3, 4), bool), "boxes": np.array([[0, 0, 4, 3]], np.float32),
                "scores": np.array([.8], np.float32)}

    processor.add_geometric_prompt = geometric
    predictor.refine(replace(refine_request(), points=(), boxes=({"xyxy": [0, 0, 4, 3], "label": False},)))
    assert invoked[0]["label"] is False
    np.testing.assert_allclose(invoked[0]["box"], [.5, .5, 1, 1])
    predictor.close()


def test_asset_store_preserves_generated_texture_glb_bytes_and_portable_manifest(tmp_path):
    from grasp_workbench.assets import AssetStore
    from grasp_workbench.ipc.sam3d import MeshResult

    source = tmp_path / "textured.glb"
    source.write_bytes(b"glTF fixture with UV/image payload")
    store = AssetStore(tmp_path / "assets", catalog_path=tmp_path / "no-catalog")
    saved = store.save(MeshResult("request", "asset", "A", "revision", "done", mesh_path=str(source),
                                  length_unit="metre"), "cup", "red cup")
    assert saved["mesh_path"].endswith("object.glb")
    assert source.read_bytes() == (tmp_path / "assets/asset/object.glb").read_bytes()
    assert json.loads((tmp_path / "assets/asset/manifest.json").read_text())["mesh_file"] == "object.glb"
    assert AssetStore(tmp_path / "assets", catalog_path=tmp_path / "no-catalog").get("asset") == saved
