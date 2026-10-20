"""Optional FFW-SG2 state/FK and endpoint IK example. Never sends robot commands."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def _path(value, directory):
    path = Path(value).expanduser()
    return path if path.is_absolute() else directory / path


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


class _UrdfKinematics:
    def __init__(self, path):
        self.joints = {}
        for joint in ET.parse(path).getroot().findall("joint"):
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = (_rotation((0, 0, 1), rpy[2]) @ _rotation((0, 1, 0), rpy[1])
                                  @ _rotation((1, 0, 0), rpy[0]))
            axis = joint.find("axis")
            self.joints[joint.find("child").get("link")] = (
                joint.find("parent").get("link"), joint.get("name"), joint.get("type"), transform,
                np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" "),
            )

    def base_from_link(self, positions, link):
        transform = np.eye(4)
        while link != "base_link":
            parent, name, kind, origin, axis = self.joints[link]
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = _rotation(axis, positions[name])
            elif kind == "prismatic":
                motion[:3, 3] = axis * positions[name]
            transform = origin @ motion @ transform
            link = parent
        return transform


def _create_ik_backend(asset_dir, device):
    """Only initialize the existing solver's IK resources, not its worker/TrajOpt."""
    import math

    import curobo._src.runtime as curobo_runtime
    import torch
    from curobo._src.collision.attachment_manager import AttachmentManager
    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import DeviceCfg
    from motion_planning_system.robot_assets import load_curobo_robot_config
    from motion_planning_system.workers.curobo_worker import _CuroboBackend
    from whole_robot_system.robots.robotis_ai_worker.ffw_sg2.definition import (
        FFW_SG2_FREE_PRRTC_Q15_JOINT_NAMES,
    )

    curobo_runtime.kernel_backend = "pybind"
    backend = _CuroboBackend.__new__(_CuroboBackend)
    backend._torch = torch
    backend._device_cfg = DeviceCfg(device=torch.device(device), dtype=torch.float32)
    backend._ignore_world_collision_for_experiment = True
    backend._inactive_arm_tolerance = .01
    backend._mission_fallback_return_seeds = 8
    backend._mission_fallback_retry_count = 1
    robot = load_curobo_robot_config(
        asset_dir, profile_id="ffw_sg2.planning-q15.v1", allow_candidate=True,
        attached_object_parent_link="gripper_l_rh_p12_rn_base",
        position_limit_clip_by_joint={
            name: .001 if name == "lift_joint" else math.radians(.5)
            for name in FFW_SG2_FREE_PRRTC_Q15_JOINT_NAMES
        },
    )
    backend._mission_fallback_ik = InverseKinematics(InverseKinematicsCfg.create(
        robot=robot, scene_model=None, collision_cache=None, num_seeds=64,
        max_batch_size=16, self_collision_check=True, random_seed=84,
        device_cfg=backend._device_cfg,
    ))
    ik = backend._mission_fallback_ik
    backend._mission_fallback_attachment = AttachmentManager(
        ik.kinematics, ik.scene_collision_checker, backend._device_cfg,
    )
    backend._mission_fallback_self_collision = SelfCollisionCost(SelfCollisionCostCfg(
        weight=1., device_cfg=backend._device_cfg,
        self_collision_kin_config=ik.kinematics.get_self_collision_config(),
    ))
    limits = ik.kinematics.get_joint_limits().position.detach().cpu()
    backend._ik_soft_lower = tuple(limits[0].tolist())
    backend._ik_soft_upper = tuple(limits[1].tolist())
    backend._ik = ik
    backend._ik_attachment = backend._mission_fallback_attachment
    backend._ik_self_collision = backend._mission_fallback_self_collision
    backend._ik_return_seeds = backend._mission_fallback_return_seeds
    backend._ik_seed_retry_count = backend._mission_fallback_retry_count
    return backend


class FfwSg2RobotAdapter:
    def __init__(self, config, *, config_dir):
        directory = Path(config_dir)
        self.runtime_manifest = _path(config["runtime_manifest"], directory)
        self.asset_dir = _path(config["robot_asset_dir"], directory)
        self.device = config.get("device", "cuda:0")
        calibration = config["calibration"]
        record = yaml.safe_load(_path(calibration["path"], directory).read_text(encoding="utf-8"))
        self.parent_frame = record["parent_frame"]
        self.parent_from_camera = np.asarray(
            record["transforms"][calibration["sensor_frame"]]["parent_from_sensor"], dtype=float,
        )
        self._fk = _UrdfKinematics(self.asset_dir / "sources/ffw_sg2.urdf")
        self._subscriber = self._state = self._backend = None
        self._dual_backend = None

    def start(self):
        from robot_worker_runtime.transport.runtime import EndpointFactory

        self._subscriber = EndpointFactory(self.runtime_manifest).open_subscriber("ffw_sg2.state")

    def snapshot(self):
        from whole_robot_system.robots.robotis_ai_worker.ffw_sg2.codec import decode_state
        from whole_robot_system.robots.robotis_ai_worker.ffw_sg2.definition import (
            FFW_SG2_Q17_JOINT_NAMES, FFW_SG2_FREE_PRRTC_Q15_JOINT_NAMES,
        )

        message = self._subscriber.take_latest()
        if message is not None:
            self._state = decode_state(message.payload)
        if self._state is None:
            return None
        positions = dict(zip(FFW_SG2_Q17_JOINT_NAMES, self._state.position))
        canonical_from_tcp = np.array(((0, 1, 0, 0), (1, 0, 0, 0),
                                       (0, 0, -1, .137), (0, 0, 0, 1)), dtype=float)
        return {
            "gripper_id": "ffw_sg2_rh_p12_rn",
            "base_from_camera": self._fk.base_from_link(positions, self.parent_frame) @ self.parent_from_camera,
            "joint_names": tuple(FFW_SG2_FREE_PRRTC_Q15_JOINT_NAMES),
            "joint_positions": tuple(positions[name] for name in FFW_SG2_FREE_PRRTC_Q15_JOINT_NAMES),
            "state_sequence": self._state.sequence, "state_timestamp_ns": self._state.timestamp_ns,
            "current_hands": {arm: self._fk.base_from_link(positions, f"end_effector_{arm[0]}_link")
                              for arm in ("left", "right")},
            "canonical_gripper_from_tcp": {arm: canonical_from_tcp.copy() for arm in ("left", "right")},
        }

    def _solver(self, *, dual=False):
        # Single grasp IK uses batch16; bimanual IK uses batch4. The pinned
        # cuRobo goal buffer cannot switch those shapes on one solver instance.
        attribute = "_dual_backend" if dual else "_backend"
        backend = getattr(self, attribute)
        if backend is None:
            backend = _create_ik_backend(self.asset_dir, self.device)
            setattr(self, attribute, backend)
        return backend

    @staticmethod
    def _rigid(matrix):
        from whole_robot_system.common.control import RigidTransform

        return RigidTransform(tuple(np.asarray(matrix)[:3, :3].flat), tuple(np.asarray(matrix)[:3, 3]))

    @classmethod
    def _tool(cls, matrix, arm):
        from motion_planning_system.curobo.geometry import _target_pose_for_curobo_tool

        return _target_pose_for_curobo_tool(cls._rigid(matrix), target_frame=f"end_effector_{arm[0]}_link")

    def solve_single(self, pregrasp, grasp, arm, seed):
        from motion_planning_system.curobo.geometry import _mission_target_variants, _tool_frame_for_arm

        arm = arm.upper()
        backend = self._solver()
        backend._tool_frame = _tool_frame_for_arm(arm)
        backend._other_tool_frame = _tool_frame_for_arm("RIGHT" if arm == "LEFT" else "LEFT")
        exact = (self._rigid(pregrasp), self._rigid(grasp))
        variants = (exact, *_mission_target_variants(exact))
        converted = tuple(tuple(self._tool(_matrix(pose), arm.lower()) for pose in pair) for pair in variants)
        request = SimpleNamespace(arm=arm, start_q15=tuple(seed["joint_positions"]),
                                  hold_inactive_arm_and_lift=False)
        result = backend.solve_grasp_candidate_approaches(request, target_variants=(converted,), scene=None)[0]
        if result is None:
            return {"success": False, "self_collision_checked": True}
        index, before, after = result
        return {"success": True, "q_pregrasp": before.q15, "q_grasp": after.q15,
                "position_errors_m": (before.position_error_m, after.position_error_m),
                "rotation_errors_rad": (before.rotation_error_rad, after.rotation_error_rad),
                "target_variant_index": index, "self_collision_checked": True,
                "selected_pregrasp": _matrix(variants[index][0]), "selected_grasp": _matrix(variants[index][1])}

    def solve_dual(self, left_pregrasp, right_pregrasp, left_grasp, right_grasp, seed):
        from motion_planning_system.contracts import BimanualIkTarget

        backend = self._solver(dual=True)

        def solve(left, right, joints):
            target = BimanualIkTarget("selected-pair", self._tool(left, "left"), self._tool(right, "right"),
                                      translation_tolerance_m=.01, rotation_tolerance_rad=.1)
            request = SimpleNamespace(start_q15=tuple(joints), position_tolerance_m=.01,
                                      rotation_tolerance_rad=.1, targets=(target,))
            return backend.solve_bimanual_ik(request, scene=None)[0]

        before = solve(left_pregrasp, right_pregrasp, seed["joint_positions"])
        after = None if before is None else solve(left_grasp, right_grasp, before.q15)
        if after is None:
            return {"success": False, "self_collision_checked": True}
        return {"success": True, "q_pregrasp": before.q15, "q_grasp": after.q15,
                "position_errors_m": tuple(value for solution in (before, after) for value in (
                    solution.left_position_error_m, solution.right_position_error_m,
                    solution.relative_translation_error_m)),
                "rotation_errors_rad": tuple(value for solution in (before, after) for value in (
                    solution.left_rotation_error_rad, solution.right_rotation_error_rad,
                    solution.relative_rotation_error_rad)),
                "target_variant_index": None, "self_collision_checked": True,
                "selected_pregrasp": {"left": np.asarray(left_pregrasp), "right": np.asarray(right_pregrasp)},
                "selected_grasp": {"left": np.asarray(left_grasp), "right": np.asarray(right_grasp)}}

    def close(self):
        if self._subscriber is not None:
            self._subscriber.close()
        for backend in (self._backend, self._dual_backend):
            if backend is not None:
                backend._mission_fallback_ik.destroy()
        self._subscriber = self._backend = self._dual_backend = self._state = None


def _matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = np.asarray(pose.rotation).reshape(3, 3), pose.translation
    return matrix
