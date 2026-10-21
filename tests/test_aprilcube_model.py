"""Official SDK adapter boundary, metre conversion, and raw camera overlay preview."""
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from grasp_workbench.models import aprilcube as aprilcube_model
from grasp_workbench.ipc.aprilcube import AprilCubeTarget
from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.models.aprilcube import OfficialAprilCubePredictor, camera_parameters
from grasp_workbench.models.aprilcube_overlay import MeshOverlayRenderer, _preview_geometry


def frame(sequence=1):
    return CameraFrame(np.broadcast_to(np.array([10, 20, 30], np.uint8), (120, 160, 3)).copy(),
                       np.full((120, 160), .7, np.float32),
                       {"camera_id": "camera", "stream_id": "epoch", "sequence": sequence,
                        "received_monotonic_ns": sequence * 1_000_000_000,
                        "intrinsics": {"fx": 200., "fy": 200., "cx": 80., "cy": 60.,
                                       "distortion_model": "zed_rectified",
                                       "coefficients": [.1, .2, .3, .4, .5]}})


@pytest.mark.parametrize("options, expected_fast", [({}, True), ({"fast": False}, False),
                                                   ({"fast": True}, True)])
def test_official_detectors_keep_native_state_rgb_bgr_and_convert_mm_once(
        monkeypatch, tmp_path, options, expected_fast):
    calls, detectors = [], []

    class Detector:
        def __init__(self, index):
            self.index, self.inputs = index, []
            self.config = SimpleNamespace(dict_id=1)
            self.detector = self.fallback_detector = SimpleNamespace(detectMarkers=lambda image: ())
            self.transform = np.eye(4)
            self.transform[:3, 3] = (100, -50, 700)

        def process_frame(self, bgr, timestamp=None):
            self.inputs.append((bgr.copy(), timestamp))
            return {"success": self.index == 0, "T": self.transform}

    def detector(path, matrix, **kwargs):
        calls.append((path, matrix.copy(), kwargs))
        value = Detector(len(detectors))
        detectors.append(value)
        return value

    monkeypatch.setitem(sys.modules, "aprilcube", SimpleNamespace(detector=detector))
    predictor = OfficialAprilCubePredictor(tmp_path, "a.json", "b.json", filter_enabled=False,
                                           **options)
    source = frame()
    rgb, depth = source.rgb.copy(), source.depth_m.copy()
    try:
        predictor.start()
        first, second = predictor.predict(source), predictor.predict(frame(2))
        assert len(calls) == 2
        assert all(call[2]["enable_filter"] is False and call[2]["fast"] is expected_fast
                   for call in calls)
        np.testing.assert_array_equal(calls[0][1], [[200, 0, 80], [0, 200, 60], [0, 0, 1]])
        assert all(np.count_nonzero(call[2]["dist_coeffs"]) == 0 for call in calls)
        np.testing.assert_allclose(first["A"][:3, 3], [.1, -.05, .7])
        np.testing.assert_array_equal(first["A"], second["A"])
        np.testing.assert_array_equal(detectors[0].transform[:3, 3], [100, -50, 700])
        assert first["B"] is None and second["B"] is None
        assert [item[1] for item in detectors[0].inputs] == [1., 2.]
        np.testing.assert_array_equal(detectors[0].inputs[0][0][0, 0], [30, 20, 10])
        np.testing.assert_array_equal(source.rgb, rgb)
        np.testing.assert_array_equal(source.depth_m, depth)
        source.metadata["stream_id"] = "new-epoch"
        predictor.predict(source)
        assert len(calls) == 4
    finally:
        predictor.close()
    assert predictor._detectors == {}


def test_sdk_diagnostics_keep_detection_and_prediction_details_json_serializable(monkeypatch, tmp_path):
    class Detector:
        config = SimpleNamespace(dict_id=1)

        def __init__(self):
            self.detector = self.fallback_detector = SimpleNamespace(detectMarkers=lambda image: ())

        def process_frame(self, *args, **kwargs):
            return {"success": False, "n_tags": np.int64(2), "n_inliers": np.int32(0),
                    "tag_ids": [np.int32(12), np.int32(13)], "visible_faces": {"top", "front"},
                    "reproj_error": float("inf"), "predicted": np.bool_(False)}

    monkeypatch.setitem(sys.modules, "aprilcube", SimpleNamespace(detector=lambda *a, **k: Detector()))
    predictor = OfficialAprilCubePredictor(tmp_path, "a.json", "b.json")
    try:
        predictor.start()
        assert predictor.predict(frame()) == {"A": None, "B": None}
        expected = {"success": False, "n_tags": 2, "n_inliers": 0, "tag_ids": [12, 13],
                    "visible_faces": ["front", "top"], "reproj_error": None, "predicted": False}
        assert predictor.diagnostics == {"A": expected, "B": expected}
        import json
        json.dumps(predictor.diagnostics, allow_nan=False)
    finally:
        predictor.close()


@pytest.mark.parametrize("log_timing", [False, True])
def test_model_timing_separates_preparation_and_each_detector(monkeypatch, tmp_path, log_timing):
    clock = SimpleNamespace(ns=0, reads=0)

    def tick():
        clock.reads += 1
        return clock.ns

    class Detector:
        def __init__(self, index):
            self.index = index
            self.config = SimpleNamespace(dict_id=1)
            self.detector = self.fallback_detector = SimpleNamespace(detectMarkers=lambda image: ())

        def process_frame(self, bgr, timestamp=None):
            clock.ns += (self.index + 2) * 1_000_000
            return {"success": False}

    detectors = []

    def detector(*args, **kwargs):
        clock.ns += 250_000
        result = Detector(len(detectors))
        detectors.append(result)
        return result

    monkeypatch.setattr(aprilcube_model, "time", SimpleNamespace(perf_counter_ns=tick))
    monkeypatch.setitem(sys.modules, "aprilcube", SimpleNamespace(detector=detector))
    predictor = OfficialAprilCubePredictor(tmp_path, "a.json", "b.json", log_timing=log_timing)
    try:
        predictor.start()
        assert predictor.timings_ms == {}
        assert predictor.predict(frame()) == {"A": None, "B": None}
        assert predictor.timings_ms == ({"prepare_ms": .5, "A_ms": 2., "B_ms": 3., "detect_ms": 0.}
                                       if log_timing else {})
        assert clock.reads == (6 if log_timing else 0)
        # Persistent detectors are not re-created on the next calibrated frame.
        predictor.predict(frame(2))
        assert len(detectors) == 2
        assert predictor.timings_ms == ({"prepare_ms": 0., "A_ms": 2., "B_ms": 3., "detect_ms": 0.}
                                       if log_timing else {})
        assert clock.reads == (12 if log_timing else 0)
    finally:
        predictor.close()


@pytest.mark.parametrize("dict_ids", [(1, 1), (1, 2)], ids=["shared-dictionary", "different-dictionaries"])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("use_fallback", [False, True], ids=["primary", "sdk-fallback"])
def test_marker_decode_once_per_dictionary_and_frame_with_independent_cube_state(
        monkeypatch, tmp_path, dict_ids, fast, use_fallback):
    decodes, estimators = [], []
    corners = tuple(np.full((1, 4, 2), value, np.float32) for value in (11, 22))
    ids = np.array([[11], [22]], np.int32)
    rejected = (np.full((1, 4, 2), -1, np.float32),)

    class MarkerDetector:
        def __init__(self, slot, kind):
            self.slot, self.kind = slot, kind

        def detectMarkers(self, image):
            decodes.append((self.slot, self.kind))
            if use_fallback and self.kind == "primary":
                return (), None, rejected
            return corners, ids, rejected

    class Estimator:
        def __init__(self, slot):
            self.slot, self.state = slot, []
            self.config = SimpleNamespace(dict_id=dict_ids[slot])
            self.detector = MarkerDetector(slot, "primary")
            self.fallback_detector = MarkerDetector(slot, "fallback")
            self.valid_ids = {11 if slot == 0 else 22}

        def process_frame(self, image, timestamp=None):
            found_corners, found_ids, found_rejected = self.detector.detectMarkers(image)
            if found_ids is None:
                found_corners, found_ids, found_rejected = self.fallback_detector.detectMarkers(image)
            selected = [int(tag_id) for tag_id in found_ids.flat if tag_id in self.valid_ids]
            self.state.append((timestamp, selected, found_rejected))
            assert found_corners is corners and found_rejected is rejected
            pose = np.eye(4)
            pose[0, 3], pose[2, 3] = selected[0], 700.
            return {"success": True, "T": pose}

    def detector(path, matrix, **kwargs):
        assert kwargs["fast"] is fast
        result = Estimator(len(estimators) % 2)
        estimators.append(result)
        return result

    monkeypatch.setitem(sys.modules, "aprilcube", SimpleNamespace(detector=detector))
    predictor = OfficialAprilCubePredictor(tmp_path, "a.json", "b.json", fast=fast)
    source = frame()
    rgb, depth = source.rgb.copy(), source.depth_m.copy()
    groups = len(set(dict_ids))
    passes = 2 if use_fallback else 1
    try:
        predictor.start()
        # Even the same frame object starts a fresh per-predict marker pass.
        for count, current in enumerate((source, frame(2), source), start=1):
            poses = predictor.predict(current)
            assert len(decodes) == count * groups * passes
            assert poses["A"][0, 3] == .011 and poses["B"][0, 3] == .022
        assert len(estimators) == 2
        assert estimators[0].state is not estimators[1].state
        assert [value[:2] for value in estimators[0].state] == [(1., [11]), (2., [11]), (1., [11])]
        assert [value[:2] for value in estimators[1].state] == [(1., [22]), (2., [22]), (1., [22])]
        assert (estimators[0].detector is estimators[1].detector) == (groups == 1)
        assert estimators[0].detector is not estimators[0].fallback_detector
        original_detectors = tuple(predictor._marker_detectors.values())
        source.metadata["stream_id"] = "new-calibration"
        predictor.predict(source)
        assert len(estimators) == 4 and len(decodes) == 4 * groups * passes
        assert not any(item in original_detectors for item in predictor._marker_detectors.values())
        np.testing.assert_array_equal(source.rgb, rgb)
        np.testing.assert_array_equal(source.depth_m, depth)
        np.testing.assert_array_equal(ids, [[11], [22]])
    finally:
        predictor.close()
    assert predictor._marker_detectors == {}


def test_shared_marker_timing_counts_decode_once_as_part_of_a_b(monkeypatch, tmp_path):
    clock = SimpleNamespace(ns=0)

    class MarkerDetector:
        def detectMarkers(self, image):
            clock.ns += 5_000_000
            return (), None, ()

    class Estimator:
        def __init__(self):
            self.config = SimpleNamespace(dict_id=1)
            self.detector = self.fallback_detector = MarkerDetector()

        def process_frame(self, image, timestamp=None):
            self.detector.detectMarkers(image)
            clock.ns += 1_000_000
            return {"success": False}

    monkeypatch.setattr(aprilcube_model, "time", SimpleNamespace(perf_counter_ns=lambda: clock.ns))
    monkeypatch.setitem(sys.modules, "aprilcube", SimpleNamespace(detector=lambda *args, **kwargs: Estimator()))
    predictor = OfficialAprilCubePredictor(tmp_path, "a.json", "b.json", log_timing=True)
    try:
        predictor.start()
        for current in (frame(), frame(2)):
            predictor.predict(current)
            assert predictor.timings_ms == {"prepare_ms": 0., "A_ms": 6., "B_ms": 1., "detect_ms": 5.}
    finally:
        predictor.close()


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("filter_enabled", [False, True])
def test_real_sdk_shares_marker_pass_and_preserves_a_b_poses(monkeypatch, fast, filter_enabled):
    from test_aprilcube_synthetic import _scene

    source, truth = _scene(monkeypatch)
    sdk = sys.modules["aprilcube"]
    original_factory = sdk.detector
    decodes = []

    class CountMarkers:
        def __init__(self, original, kind):
            self.original, self.kind = original, kind

        def detectMarkers(self, image):
            decodes.append(self.kind)
            return self.original.detectMarkers(image)

    def detector(*args, **kwargs):
        estimator = original_factory(*args, **kwargs)
        estimator.detector = CountMarkers(estimator.detector, "primary")
        estimator.fallback_detector = CountMarkers(estimator.fallback_detector, "fallback")
        return estimator

    monkeypatch.setattr(sdk, "detector", detector)
    root = Path(__file__).resolve().parents[1]
    predictor = OfficialAprilCubePredictor(
        root / "third_party/aprilcube", root / "examples/aprilcube/cube_a.json",
        root / "examples/aprilcube/cube_b.json", fast=fast, filter_enabled=filter_enabled)
    try:
        predictor.start()
        for count in range(1, 4):
            source.metadata["received_monotonic_ns"] += 33_333_333
            poses = predictor.predict(source)
            assert decodes == ["primary"] * count
            for slot, pose in poses.items():
                np.testing.assert_allclose(pose[:3, 3], truth[slot][:3, 3], atol=.01)
                np.testing.assert_allclose(pose[:3, :3], truth[slot][:3, :3], atol=.05)
                estimator = predictor._detectors[slot]
                ids = set(estimator._latest_result["tag_ids"])
                assert ids and ids <= estimator.valid_ids
                assert (estimator.pose_filter is not None) == filter_enabled
        assert predictor._detectors["A"].prev_rvec is not predictor._detectors["B"].prev_rvec
    finally:
        predictor.close()


def test_nonrectified_camera_keeps_intrinsic_distortion():
    source = frame()
    source.metadata["intrinsics"]["distortion_model"] = "plumb_bob"
    np.testing.assert_array_equal(camera_parameters(source)[1], [.1, .2, .3, .4, .5])


def test_overlay_scale_applies_to_local_vertices_once_and_preserves_mesh_bytes(tmp_path):
    pytest.importorskip("cv2")
    trimesh = pytest.importorskip("trimesh")
    path = tmp_path / "box.obj"
    trimesh.creation.box(extents=[100, 100, 100]).export(path)
    content = path.read_bytes()
    source = frame()
    original = source.rgb.copy()
    pose = np.eye(4)
    pose[2, 3] = .7
    target = AprilCubeTarget("A", "session", str(path), .001, np.eye(4))
    renderer = MeshOverlayRenderer()
    rgba = renderer.render(source, (target,), {"A": pose}, .65)
    assert rgba.shape == (120, 160, 4) and rgba.dtype == np.uint8
    assert np.count_nonzero(rgba[:, :, 3]) > 500
    assert set(np.unique(rgba[:, :, 3])) == {0, 166}
    # The 0.1m cube projects near the image centre, not millimetres-as-metres.
    assert np.all(rgba[:20, :, 3] == 0) and np.all(rgba[:, :40, 3] == 0)
    np.testing.assert_array_equal(source.rgb, original)
    assert path.read_bytes() == content
    assert _preview_geometry.cache_info().currsize >= 1
    assert not renderer.render(source, (target,), {"A": None}).any()


def test_preview_simplifies_only_display_geometry_to_at_most_2000_faces(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("fast_simplification")
    mesh = trimesh.creation.icosphere(subdivisions=4)
    path = tmp_path / "sphere.ply"
    mesh.export(path)
    content = path.read_bytes()
    stat = path.stat()
    vertices, faces = _preview_geometry(str(path), stat.st_mtime_ns, stat.st_size)
    assert len(faces) <= 2000 and len(vertices) > 0
    assert path.read_bytes() == content and len(mesh.faces) > 2000


def test_no_mesh_renders_cube_axes_and_absence_renders_transparent():
    pytest.importorskip("cv2")
    source = frame()
    pose = np.eye(4)
    pose[2, 3] = .7
    target = AprilCubeTarget("A", "session", "", 1., np.eye(4))
    renderer = MeshOverlayRenderer()
    assert renderer.render(source, (target,), {"A": pose})[:, :, 3].any()
    assert not renderer.render(source, (target,), {"A": None}).any()
