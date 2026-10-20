"""Real selection model/plugin/worker/iceoryx2 with only the robot solver mocked."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.grasp_selection import (
    SelectionRequest, SelectionRequestPublisher, SelectionResultSubscriber, services,
)
from test_grasp_selection_model import coordination, make_object as make_object_fixture

make_object = make_object_fixture
ROOT = Path(__file__).resolve().parents[1]
PLUGIN = """
import numpy as np
def matrix(x=0,y=0,z=0):
    pose=np.eye(4);pose[:3,3]=(x,y,z);return pose
class Robot:
    def start(self):pass
    def close(self):pass
    def snapshot(self):
        return {'base_from_camera':matrix(10,20,30),'joint_names':['j1','j2'],'joint_positions':[0.,0.],
                'current_hands':{'left':matrix(),'right':matrix()},'gripper_id':'test-gripper',
                'canonical_gripper_from_tcp':{'left':matrix(0,0,.2),'right':matrix(0,0,.3)}}
    def answer(self):
        return {'success':True,'q_pregrasp':[0.,0.],'q_grasp':[.1,.1],
                'position_errors_m':[0.001],'rotation_errors_rad':[0.01],
                'target_variant_index':0,'self_collision_checked':True}
    def solve_single(self,pregrasp,grasp,arm,*,seed):return self.answer()
    def solve_dual(self,left_pregrasp,right_pregrasp,left_grasp,right_grasp,*,seed):return self.answer()
def create(config,*,config_dir):return Robot()
"""


def test_existing_bundle_to_base_frame_single_and_same_object_dual_over_native_worker(tmp_path, make_object):
    pytest.importorskip("iceoryx2")
    pose = np.eye(4)
    pose[:3, 3] = (2, 3, 4)
    obj = make_object("B", camera=pose)
    bundle_data = Path(obj["bundle_path"]).with_name("grasps.npz").read_bytes()
    context = {"object_id": "A", "observation_id": "obs-a", "instance_id": "instance-A", "bundle_path": None,
               "mesh_path": None, "mesh_scale_to_m": 1., "camera_from_object": None}
    (tmp_path / "fixture_robot.py").write_text(PLUGIN)
    robot_config = tmp_path / "robot.yaml"
    robot_config.write_text("backend: fixture_robot:create\n")
    service = "selection-pipeline/" + uuid4().hex
    request_service, result_service = services(service)
    publisher, subscriber = SelectionRequestPublisher(request_service), SelectionResultSubscriber(result_service)
    log = tmp_path / "worker.log"
    process = None

    def wait_for_result(request):
        publisher.update_connections()
        publisher.publish(request)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = subscriber.receive_latest()
            if result is not None and result.request_id == request.request_id and result.status in {"done", "error"}:
                assert result.status == "done", result.error
                return result.selection
            assert process.poll() is None, log.read_text()
            time.sleep(.005)
        pytest.fail(log.read_text())

    try:
        with log.open("w") as stream:
            script = """
import sys
for name in ('torch','curobo','object_pose_system','ffw_system','pyzed','pyrealsense2'):
    sys.modules[name]=None
from grasp_workbench.workers.worker_grasp_selection import main
raise SystemExit(main())
"""
            process = subprocess.Popen([sys.executable, "-B", "-u", "-c", script, "--service", service,
                                        "--robot-config", str(robot_config)], cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, env={**os.environ,
                                       "PYTHONPATH": os.pathsep.join((str(ROOT), str(tmp_path)))})
            deadline = time.monotonic() + 10
            while '"event": "ready"' not in log.read_text():
                assert time.monotonic() < deadline and process.poll() is None, log.read_text()
                publisher.update_connections()
                time.sleep(.005)
            for mode, assignments in (("single", (("right", "obs-b"),)),
                                      ("dual", (("left", "obs-b"), ("right", "obs-b")))):
                request = SelectionRequest(mode, coordination(mode, assignments), (context, obj))
                result = wait_for_result(request)
                assert result["arm_mode"] == mode and result["robot_connected"] and result["robot_ik_checked"]
                assert result["self_collision_checked"] and not result["scene_collision_checked"]
                assert not result["trajectory_checked"] and not result["robot_command_published"]
                np.testing.assert_allclose(np.asarray(result["objects"]["obs-b"]["base_from_object"])[:3, 3], (12, 23, 34))
                candidate = result["candidates"][0]
                assert candidate["ik_status"] == "solved"
                targets = candidate["targets"]
                assert set(targets) == {arm for arm, _ in assignments}
                if mode == "dual":
                    assert targets["left"]["index"] != targets["right"]["index"]
                for arm, target in targets.items():
                    assert target["object_id"] == "B" and target["observation_id"] == "obs-b"
                    assert target["instance_id"] == "instance-B"
                    conversion = np.eye(4)
                    conversion[2, 3] = .2 if arm == "left" else .3
                    base_object = np.asarray(result["objects"]["obs-b"]["base_from_object"])
                    np.testing.assert_allclose(base_object @ target["object_from_gripper"] @ conversion,
                                               target["base_from_tcp"])
            process.terminate()
            assert process.wait(timeout=5) == 0, log.read_text()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        subscriber.close()
        publisher.close()
    assert Path(obj["bundle_path"]).with_name("grasps.npz").read_bytes() == bundle_data
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    assert sum(row.get("event") == "grasp-selection-result" for row in rows) == 2
