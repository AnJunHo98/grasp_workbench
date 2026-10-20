"""Readonly semantic restoration alongside unchanged SAM3/mesh/FP routing."""
import json
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
from PyQt6 import QtCore, QtWidgets

from grasp_workbench.ipc.vlm import VlmResult
from test_policy_catalog_local import policy_path as policy_path_fixture
from test_semantic_session import coordination_fields, semantic_result
from test_vlm_ui import (
    LEFT, RIGHT, add_known_asset, app as app_fixture, make_window as make_window_fixture, receive_segmentation, request_vlm,
)

app, make_window, policy_path = app_fixture, make_window_fixture, policy_path_fixture


def deliver(window, result):
    window.receiver.vlm_latest = result
    window.poll_once()


def test_vlm_readonly_details_restore_all_semantics_and_policy_checkpoint_local_only(make_window, policy_path):
    window = make_window(foundationpose=False, sam3d=False, policy_catalog=policy_path)
    request = request_vlm(window)
    assert request.policies[0]["policy_id"] == "place-v1"
    assert "checkpoint" not in json.dumps(request.policies)
    deliver(window, semantic_result(request))
    record = window._semantic_session.record
    assert record["objects"][0]["slot"] == "A" and record["objects"][1]["slot"] == "B"
    assert record["task_binding"]["source_observation_id"] == "source-obs"
    assert record["coordination"] == coordination_fields()
    assert record["policy_reference"]["checkpoint"] == str(policy_path.parent / "weights/policy.pt")
    assert window.semantic_details.isReadOnly() and "reference_state" in window.semantic_details.toPlainText()
    details_button = next(button for button in window.findChildren(QtWidgets.QToolButton)
                          if button.text() == "Task / 팔 / 제약 / policy 상세")
    assert details_button.toolButtonStyle() == QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    assert "source: source-obs [B]" in window.semantic_summary.text() and "arm_mode: dual" in window.semantic_summary.text()
    assert not window._pose_targets and window.prompt_inputs["B"].text() == "tray"


def test_new_request_failure_and_cancel_keep_last_valid_semantic_and_arm_settings(make_window):
    window = make_window(foundationpose=False, sam3d=False)
    request = request_vlm(window)
    deliver(window, semantic_result(request))
    previous = window._semantic_session
    request = request_vlm(window)
    assert window._semantic_session is previous
    deliver(window, VlmResult(request.request_id, request.frame, "error", error="provider failed"))
    assert window._semantic_session is previous and previous.record["coordination"] == coordination_fields()
    request_vlm(window)
    window.vlm_enabled.setChecked(False)
    assert window._semantic_session is previous and previous.record["coordination"]["arm_mode"] == "dual"


def test_instruction_edit_marks_reanalysis_without_clearing_mode_or_blocking_manual_flow(make_window):
    window = make_window(foundationpose=False, sam3d=False)
    request = request_vlm(window)
    deliver(window, semantic_result(request))
    window.vlm_instruction.setText("pour instead")
    window.vlm_instruction.textEdited.emit("pour instead")
    record = window._semantic_session.record
    assert record["coordination_instruction_stale"] and record["coordination"] == coordination_fields()
    assert "재분석 필요" in window.semantic_summary.text() and window.vlm_button.isEnabled()
    window.apply_prompts()
    assert record["bindings_stale"] and all(row["slot"] is None for row in record["objects"])
    assert record["coordination"]["arm_mode"] == "dual" and window.receiver.prompts[-1].objects


def test_success_replaces_old_semantic_only_after_result_and_keeps_explicit_instance_mapping(make_window):
    window = make_window(foundationpose=False, sam3d=False)
    request = request_vlm(window)
    deliver(window, semantic_result(request))
    first = window._semantic_session
    first_source_id = first.record["objects"][1]["instance_id"]
    request = request_vlm(window)
    assert window._semantic_session is first
    deliver(window, semantic_result(request))
    assert window._semantic_session is not first
    assert window._semantic_session.record["objects"][1]["instance_id"] != first_source_id


def test_auto_fp_routes_both_observations_without_invalidating_but_manual_mesh_stop_detach(make_window, tmp_path):
    window = make_window()
    known = add_known_asset(window, tmp_path)
    request = request_vlm(window)
    result = semantic_result(request)
    result.objects = tuple(replace(obj, known_asset_id=known["asset_id"], bbox_xyxy=box)
                           for obj, box in zip(result.objects, (LEFT, RIGHT)))
    deliver(window, result)
    receive_segmentation(window, {"A": [(LEFT, 0.9)], "B": [(RIGHT, 0.8)]})
    record = window._semantic_session.record
    assert set(window._pose_targets) == {"A", "B"}
    assert not record["bindings_stale"] and all(row["binding_active"] for row in record["objects"])
    assert all(row["geometry"]["asset_id"] == known["asset_id"] for row in record["objects"])
    window.pose_mesh_inputs["A"].setText("other-manual.ply")
    assert not record["objects"][0]["binding_active"] and record["objects"][1]["binding_active"]
    window.stop_pose("B")
    assert all(not row["binding_active"] for row in record["objects"])
    assert record["coordination"] == coordination_fields()


def test_session_save_load_and_request_export_do_not_apply_saved_fp_pose_or_start_worker(make_window, tmp_path):
    window = make_window(foundationpose=False, sam3d=False)
    request = request_vlm(window)
    deliver(window, semantic_result(request))
    record = window._semantic_session.record
    row = record["objects"][1]
    row["geometry"] = {"asset_id": "same-asset", "mesh_path": "/assets/tray.ply", "mesh_scale_to_m": 1}
    row["foundationpose"] = {"session_id": "saved-pose", "camera_from_object": [[1, 0, 0, 2]]}
    row["grasp"] = {"bundle_path": "/assets/grasp.npz", "selected_index": 2}
    path = window.save_semantic_session(tmp_path / "task.json")
    before = len(window.receiver.pose_controls)
    assert path == tmp_path / "task.json"
    assert window.load_semantic_session(path) == path
    assert len(window.receiver.pose_controls) == before and not window._pose_targets and not window._pose_objects
    loaded = window._semantic_session.record
    assert loaded["objects"][1]["foundationpose"] == row["foundationpose"]
    assert "현재 관측 아님" in window.semantic_summary.text()
    output = tmp_path / "request.json"
    exported = window.export_semantic_request(output)
    assert exported == json.loads(output.read_text(encoding="utf-8"))
    assert exported["execution"] == "request_only" and exported["robot_command_published"] is False
    source_id = loaded["objects"][1]["instance_id"]
    assert exported["task_binding"]["source_instance_id"] == source_id
    assert {row["instance_id"] for row in exported["coordination"]["assignments"]} == {source_id}
    assert exported["objects"][1]["geometry"] == row["geometry"]


def test_legacy_result_keeps_existing_object_routing_and_explicitly_shows_no_arm_information(make_window):
    window = make_window(foundationpose=False, sam3d=False)
    request = request_vlm(window)
    result = semantic_result(request)
    result.task_binding = result.policy_selection = result.coordination = None
    deliver(window, result)
    assert window.prompt_inputs["A"].text() == "tray" and "팔 정보 없음" in window.semantic_summary.text()
    assert window._semantic_session.record["coordination"] is None


def test_open_details_refreshes_latest_references_without_per_frame_text_render(make_window, monkeypatch):
    window = make_window(foundationpose=False, sam3d=False)
    request = request_vlm(window)
    deliver(window, semantic_result(request))
    calls = []
    monkeypatch.setattr(window, "_sync_semantic_references", lambda: calls.append("sync"))
    monkeypatch.setattr(window, "_render_semantic_session", lambda: calls.append("render"))
    window._semantic_details_toggled(False)
    assert calls == []
    window._semantic_details_toggled(True)
    assert calls == ["sync", "render"]
