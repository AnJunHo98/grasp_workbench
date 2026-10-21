"""Only the independent worker entry points used by the supervisor."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    module: str


WORKERS = {
    "camera": WorkerSpec("camera", "grasp_workbench.workers.worker_camera"),
    "camera_monitor": WorkerSpec("camera_monitor", "grasp_workbench.workers.worker_camera_monitor"),
    "camera_ui": WorkerSpec("camera_ui", "grasp_workbench.workers.worker_camera_ui"),
    "data_recorder": WorkerSpec("data_recorder", "grasp_workbench.workers.worker_data_recorder"),
    "aprilcube": WorkerSpec("aprilcube", "grasp_workbench.workers.worker_aprilcube"),
    "aprilcube_overlay": WorkerSpec("aprilcube_overlay", "grasp_workbench.workers.worker_aprilcube_overlay"),
    "sam3": WorkerSpec("sam3", "grasp_workbench.workers.worker_sam3"),
    "sam3d": WorkerSpec("sam3d", "grasp_workbench.workers.worker_sam3d"),
    "foundationpose": WorkerSpec("foundationpose", "grasp_workbench.workers.worker_foundationpose"),
    "vlm": WorkerSpec("vlm", "grasp_workbench.workers.worker_vlm"),
    "graspgenx": WorkerSpec("graspgenx", "grasp_workbench.workers.worker_graspgenx"),
    "grasp_selection": WorkerSpec("grasp_selection", "grasp_workbench.workers.worker_grasp_selection"),
}
