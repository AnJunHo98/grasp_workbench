"""On-demand semantic recognition routes existing/new meshes without provider calls."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtWidgets

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.sam3 import Sam3ObjectResult, Sam3Result
from grasp_workbench.ipc.sam3d import MeshResult
from grasp_workbench.ipc.vlm import VlmObject, VlmResult
from grasp_workbench.ui.camera_window import CameraWindow
from test_camera_ui import FakeReceiver, completed_mesh

LEFT, RIGHT = (5, 5, 25, 25), (45, 5, 65, 25)


class VlmFakeReceiver(FakeReceiver):
    def __init__(self, service, *, vlm=False, **kwargs):
        super().__init__(service, **kwargs)
        self.vlm, self.vlm_requests, self.vlm_latest = vlm, [], None

    def submit_vlm(self, request):
        self.vlm_requests.append(request)

    def take_vlm_latest(self):
        result, self.vlm_latest = self.vlm_latest, None
        return result


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def make_window(app, tmp_path):
    windows = []

    def make(**options):
        settings = {"vlm": True, "foundationpose": True, "sam3d": True,
                    "asset_dir": tmp_path / "assets", "receiver_factory": VlmFakeReceiver}
        settings.update(options)
        window = CameraWindow(**settings)
        window.timer.stop()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.receiver.alive = False
        window.close()
    app.processEvents()


def camera_frame(sequence=10):
    return CameraFrame(np.full((60, 80, 3), 30, dtype=np.uint8),
                       np.ones((60, 80), dtype=np.float32), {
                           "camera_id": "fake", "sequence": sequence,
                           "intrinsics": {"fx": 80., "fy": 80., "cx": 40., "cy": 30.},
                       })


def write_mesh(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
2 0 0
0 4 0
3 0 1 2
""", encoding="utf-8")
    return path


def add_known_asset(window, tmp_path, asset_id="known-id"):
    path = write_mesh(tmp_path / "source" / f"{asset_id}.ply")
    result = MeshResult("saved-request", asset_id, "A", "saved-revision", "done",
                        mesh_path=str(path), mesh_scale_to_m=0.01, length_unit="metre")
    return window.asset_store.save(result, "테이프", "brown packing tape roll")


def request_vlm(window):
    window.receiver.latest = camera_frame()
    window.poll_once()
    window.vlm_button.click()
    return window.receiver.vlm_requests[-1]


def finish_vlm(window, objects, *, request=None):
    request = request or window.receiver.vlm_requests[-1]
    window.receiver.vlm_latest = VlmResult(request.request_id, request.frame, "done", tuple(objects))
    window.poll_once()


def receive_segmentation(window, candidates):
    objects = []
    for key, choices in candidates.items():
        masks = np.zeros((len(choices), 60, 80), dtype=bool)
        for mask, (box, _score) in zip(masks, choices):
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
        objects.append(Sam3ObjectResult(key, window.prompt_inputs[key].text(), masks,
                                       np.asarray([box for box, _ in choices], dtype=np.float32),
                                       np.asarray([score for _, score in choices], dtype=np.float32)))
    result = Sam3Result(camera_frame(sequence=33), window._prompts.revision, tuple(objects))
    window.receiver.sam3_latest = result
    window.poll_once()
    return result


def test_vlm_button_freezes_current_frame_and_sends_public_catalog_once(make_window, tmp_path):
    window = make_window()
    known = add_known_asset(window, tmp_path)
    assert window.receiver.vlm and not window.vlm_button.isEnabled()
    source = camera_frame()
    window.receiver.latest = source
    window.poll_once()
    window.vlm_instruction.setText("  테이프와 공  ")
    window.vlm_button.click()
    request, = window.receiver.vlm_requests
    assert request.instruction == "테이프와 공"
    assert all(set(asset) == {"asset_id", "name", "prompt"} for asset in request.assets)
    assert any(asset["asset_id"] == known["asset_id"] for asset in request.assets)
    source.rgb[:] = 200
    source.depth_m[:] = 0.2
    source.metadata["sequence"] = 99
    assert (request.frame.rgb == 30).all() and (request.frame.depth_m == 1).all()
    assert request.frame.metadata["sequence"] == 10
    assert not window.vlm_button.isEnabled()
    window.poll_once()
    assert len(window.receiver.vlm_requests) == 1


def test_known_and_new_objects_route_independently_using_box_overlap_before_score(make_window, tmp_path):
    window = make_window()
    known = add_known_asset(window, tmp_path)
    request_vlm(window)
    finish_vlm(window, (VlmObject("테이프", "brown packing tape roll", LEFT, known["asset_id"]),
                        VlmObject("공", "orange ball", RIGHT, None)))
    assert window.prompt_inputs["A"].text() == "brown packing tape roll"
    assert window.prompt_inputs["B"].text() == "orange ball"
    assert not window.receiver.mesh_requests and not window._pose_targets
    source = receive_segmentation(window, {"A": [(RIGHT, 0.99), (LEFT, 0.1)], "B": [(RIGHT, 0.8)]})
    target = window._pose_targets["A"]
    assert target.mesh_path == known["mesh_path"] and target.mesh_scale_to_m == 0.01
    assert target.frame is source.frame
    np.testing.assert_array_equal(target.mask, source.objects[0].masks[1])
    mesh_request, = window.receiver.mesh_requests
    assert mesh_request.object_id == "B" and "B" not in window._pose_targets
    assert window._mesh_bindings["B"]["name"] == "공"
    np.testing.assert_array_equal(mesh_request.mask, source.objects[1].masks[0])


def test_two_slots_do_not_choose_the_exact_same_mask(make_window, tmp_path):
    window = make_window()
    known = add_known_asset(window, tmp_path)
    request_vlm(window)
    finish_vlm(window, (VlmObject("테이프", "brown tape", LEFT, known["asset_id"]),
                        VlmObject("캔", "yellow tuna can", LEFT, known["asset_id"])))
    source = receive_segmentation(window, {"A": [(LEFT, 0.9)], "B": [(LEFT, 0.99), (RIGHT, 0.2)]})
    assert set(window._pose_targets) == {"A", "B"} and not window.receiver.mesh_requests
    a, b = window._pose_targets["A"], window._pose_targets["B"]
    assert not np.array_equal(a.mask, b.mask)
    np.testing.assert_array_equal(b.mask, source.objects[1].masks[1])


def test_unknown_generated_mesh_is_saved_and_reused_by_id_after_reopening(make_window, tmp_path):
    window = make_window()
    request_vlm(window)
    finish_vlm(window, (VlmObject("새 공", "orange ball", LEFT, None),))
    receive_segmentation(window, {"A": [(LEFT, 0.9)]})
    request, = window.receiver.mesh_requests
    assert not window._pose_targets
    result = completed_mesh(request)
    result.mesh_path = str(write_mesh(tmp_path / "cache" / "new.ply"))
    window.receiver.mesh_events.append(result)
    window.poll_once()
    assert window._pose_targets["A"].frame is request.frame
    assert window._pose_targets["A"].mask is request.mask
    saved = window.asset_store.get(request.asset_id)
    assert saved["name"] == "새 공" and saved["prompt"] == "orange ball"
    assert (tmp_path / "assets" / request.asset_id / "manifest.json").is_file()
    (tmp_path / "cache" / "new.ply").unlink()
    reopened = make_window()
    request_vlm(reopened)
    finish_vlm(reopened, (VlmObject("다시 보이는 공", "orange ball", LEFT, request.asset_id),))
    receive_segmentation(reopened, {"A": [(LEFT, 0.9)]})
    assert not reopened.receiver.mesh_requests
    assert reopened._pose_targets["A"].mesh_path == saved["mesh_path"]


@pytest.mark.parametrize("action", ["off", "manual_apply"])
def test_off_or_manual_apply_ignores_pending_late_vlm_result(make_window, action):
    window = make_window()
    window.prompt_inputs["A"].setText("manual cup")
    window.apply_prompts()
    request = request_vlm(window)
    if action == "off":
        window.vlm_enabled.setChecked(False)
    else:
        window.prompt_inputs["A"].setText("manual replacement")
        window.apply_prompts()
    text = window.prompt_inputs["A"].text()
    revision = window._prompts.revision
    finish_vlm(window, (VlmObject("공", "orange ball", LEFT, None),), request=request)
    assert window.prompt_inputs["A"].text() == text
    assert window._prompts.revision == revision and not window._vlm_plan
    assert not window.receiver.mesh_requests and not window._pose_targets


def test_vlm_error_ends_request_and_ignores_its_later_completion(make_window):
    window = make_window()
    request = request_vlm(window)
    window.receiver.vlm_latest = VlmResult(request.request_id, request.frame, "error", error="provider failed")
    window.poll_once()
    assert "provider failed" in window.vlm_status.text()
    assert window.vlm_button.isEnabled()
    finish_vlm(window, (VlmObject("공", "orange ball", LEFT, None),), request=request)
    assert window.prompt_inputs["A"].text() == ""
    assert not window._vlm_plan and not window.receiver.mesh_requests


def test_stop_while_unknown_mesh_is_generating_cancels_automatic_pose_start(make_window, tmp_path):
    window = make_window()
    request_vlm(window)
    finish_vlm(window, (VlmObject("공", "orange ball", LEFT, None),))
    receive_segmentation(window, {"A": [(LEFT, 0.9)]})
    request, = window.receiver.mesh_requests
    window.stop_pose("A")
    control_count = len(window.receiver.pose_controls)
    result = completed_mesh(request)
    result.mesh_path = str(write_mesh(tmp_path / "cache" / "new.ply"))
    window.receiver.mesh_events.append(result)
    window.poll_once()
    assert not window._pose_targets and len(window.receiver.pose_controls) == control_count


def test_without_sam3d_known_mesh_starts_fp_and_unknown_mesh_requests_worker(make_window, tmp_path):
    window = make_window(sam3d=False)
    known = add_known_asset(window, tmp_path)
    request_vlm(window)
    finish_vlm(window, (VlmObject("테이프", "brown tape", LEFT, known["asset_id"]),
                        VlmObject("공", "orange ball", RIGHT, None)))
    receive_segmentation(window, {"A": [(LEFT, 0.9)], "B": [(RIGHT, 0.9)]})
    assert set(window._pose_targets) == {"A"}
    assert window._pose_targets["A"].mesh_path == known["mesh_path"]
    assert "SAM3D worker 필요" in window.vlm_status.text() and not window.receiver.mesh_requests


@pytest.mark.parametrize("sam3d", [False, True])
def test_without_foundationpose_vlm_only_applies_prompts(make_window, sam3d):
    window = make_window(foundationpose=False, sam3d=sam3d)
    request_vlm(window)
    finish_vlm(window, (VlmObject("공", "orange ball", LEFT, None),))
    assert window.prompt_inputs["A"].text() == "orange ball" and not window._vlm_plan
    receive_segmentation(window, {"A": [(LEFT, 0.9)]})
    assert not window.receiver.mesh_requests and not window._pose_targets


def test_vlm_disabled_keeps_manual_prompt_mesh_buttons_available(make_window):
    window = make_window()
    window.vlm_enabled.setChecked(False)
    window.prompt_inputs["A"].setText("manual cup")
    window.apply_prompts()
    receive_segmentation(window, {"A": [(LEFT, 0.9)]})
    window.mesh_buttons["A"].click()
    request, = window.receiver.mesh_requests
    assert request.prompt == "manual cup" and not window.receiver.vlm_requests
    assert not window._pose_targets
