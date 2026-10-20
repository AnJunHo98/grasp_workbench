"""Standalone UI semantic catalog/session CLI routing, without starting workers."""
from grasp_workbench.main import main


def test_ui_semantic_paths_are_translated_to_ui_worker_options_only(monkeypatch):
    from grasp_workbench.workers import worker_camera_ui

    calls = []
    monkeypatch.setattr(worker_camera_ui, "main", lambda argv: calls.append(argv) or 0)
    assert main(["ui", "--vlm", "--vlm-policy-catalog", "/local/policies.json",
                 "--vlm-session-dir", "/local/tasks"]) == 0
    assert calls == [["--service", "grasp_workbench/camera/rgbd", "--frames", "0", "--timeout", "0", "--vlm",
                      "--policy-catalog", "/local/policies.json", "--session-dir", "/local/tasks"]]


def test_standalone_ui_passes_sam3d_request_options(monkeypatch):
    from grasp_workbench.workers import worker_camera_ui

    calls = []
    monkeypatch.setattr(worker_camera_ui, "main", lambda argv: calls.append(argv) or 0)
    assert main(["ui", "--sam3d", "--sam3d-depth-source", "monocular", "--sam3d-texture-baking"]) == 0
    assert calls == [["--service", "grasp_workbench/camera/rgbd", "--frames", "0", "--timeout", "0", "--sam3d",
                      "--sam3d-depth-source", "monocular", "--sam3d-texture-baking"]]
