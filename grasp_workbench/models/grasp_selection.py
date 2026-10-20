"""Object-frame grasp ranking and optional read-only robot endpoint IK.

COM/contact geometry is a proxy, not force closure or a stability certificate.
Endpoint IK checks neither a path nor scene collision and publishes no commands.
"""
import itertools
from pathlib import Path

import numpy as np

from ..grasp_assets import load_grasp_bundle, sha256_file
from ..semantic import validate_coordination


# Values from the old table-leg example profile, not universal robot defaults.
DEFAULT_SETTINGS = {"top_k": 32, "pregrasp_distance_m": .15,
                    "grasp_retreat_distance_m": 0., "grasp_advance_distance_m": 0., "preference": "top_down"}


def _z(distance):
    matrix = np.eye(4)
    matrix[2, 3] = distance
    return matrix


def _list(matrix):
    return None if matrix is None else np.asarray(matrix).tolist()


def _point(value):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    return result if result.shape == (3,) and np.isfinite(result).all() else None


def estimate_mesh_com(mesh_path, mesh_scale_to_m=1.):
    """The old uniform-volume / bounding-box COM proxy, in the original object frame."""
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    mesh.apply_scale(mesh_scale_to_m)
    center = _point(mesh.center_mass) if mesh.is_volume else None
    method = "uniform_density_volume" if center is not None else "bounding_box_center"
    if center is None:
        center = _point(mesh.bounds.mean(axis=0))
    return {"position_m": _list(center), "method": method, "frame": "object",
            "mesh_scale_to_m": mesh_scale_to_m, "mesh_sha256": sha256_file(mesh_path),
            "physical_stability_verified": False}


def com_balance_metrics(left_point, right_point, com_point, gravity_object):
    """Old gravity-plane contact-segment midpoint/distance score, unchanged."""
    result = dict(score=None, projection_fraction=None, segment_distance_m=None,
                  midpoint_balance=None, projected_span_m=None, reason=None)
    left, right, com, gravity = map(_point, (left_point, right_point, com_point, gravity_object))
    if gravity is None or np.linalg.norm(gravity) <= 1e-12:
        return {**result, "reason": "gravity_unavailable"}
    if left is None or right is None or com is None:
        return {**result, "reason": "contact_or_com_unavailable"}
    unit = gravity / np.linalg.norm(gravity)
    segment, center = right - left, com - left
    segment = segment - np.dot(segment, unit) * unit
    center = center - np.dot(center, unit) * unit
    span = float(np.linalg.norm(segment))
    result["projected_span_m"] = span
    if span <= 1e-12:
        return {**result, "reason": "projected_contacts_coincide"}
    fraction = float(np.dot(center, segment) / span**2)
    distance = float(np.linalg.norm(center - np.clip(fraction, 0, 1) * segment))
    midpoint = 1. / (1. + (2. * fraction - 1.)**2)
    return {**result, "score": midpoint / (1. + (2. * distance / span)**2),
            "projection_fraction": fraction, "segment_distance_m": distance, "midpoint_balance": midpoint}


def _contact(target, metadata):
    config = metadata["gripper_metadata"].get("config", {})
    offset = _point(config.get("sweep_volume", {}).get("offset"))
    if offset is None:
        offset = _point(config.get("fingertip"))
    return None if offset is None else (np.asarray(target["object_from_gripper"]) @ np.r_[offset, 1.])[:3]


def _reachability_score(seed, answer):
    current, pre, grasp = map(np.asarray, (seed["joint_positions"], answer["q_pregrasp"], answer["q_grasp"]))
    distance = float(np.sqrt(np.sum((pre - current)**2) + np.sum((grasp - pre)**2)))
    variant = 1. if not answer.get("target_variant_index") else .85
    position = max(answer.get("position_errors_m", ()), default=0.)
    rotation = max(answer.get("rotation_errors_rad", ()), default=0.)
    return float(np.clip(.5 / (1. + distance) + .3 * variant + .2 / (1. + 100. * position + 10. * rotation), 0, 1))


def _topdown_metrics(pregrasp, grasp):
    """The old score of the actual base-frame pregrasp-to-grasp direction."""
    direction = np.asarray(grasp)[:3, 3] - np.asarray(pregrasp)[:3, 3]
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return {"topdown_score": 0., "approach_angle_deg": None}
    downward = float(np.clip(-direction[2] / length, -1., 1.))
    return {"topdown_score": max(0., downward), "approach_angle_deg": float(np.degrees(np.arccos(downward)))}


def _near_topdown(left, right, advance):
    # Old diversity compares the converted hand before the optional advance.
    left = np.asarray(next(iter(left["targets"].values()))["base_from_tcp"]) @ _z(advance)
    right = np.asarray(next(iter(right["targets"].values()))["base_from_tcp"]) @ _z(advance)
    distance = np.linalg.norm(left[:3, 3] - right[:3, 3])
    angle = np.arccos(np.clip((np.trace(left[:3, :3].T @ right[:3, :3]) - 1.) / 2., -1., 1.))
    return distance < .02 and angle < np.deg2rad(10.)


def _source_candidate_id(candidate):
    return next(iter(candidate["targets"].values()))["source_candidate_id"]


class GraspSelectionPredictor:
    def __init__(self, robot_config=None, settings=None, robot_factory=None):
        self.robot_config, self.robot_factory = robot_config, robot_factory
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        if type(self.settings["top_k"]) is not int or self.settings["top_k"] < 1:
            raise ValueError("top_k must be a positive integer")
        for name in ("pregrasp_distance_m", "grasp_retreat_distance_m", "grasp_advance_distance_m"):
            value = self.settings[name]
            if (isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value)
                    or value < 0 or (name == "pregrasp_distance_m" and value == 0)):
                raise ValueError(f"Invalid grasp profile distance: {name}")
        self._robot, self._started = None, False

    def start(self):
        if self._started:
            return
        if self.robot_factory is not None:
            self._robot = self.robot_factory()
        elif self.robot_config is not None:
            from .robot_reachability import load_robot_adapter

            self._robot = load_robot_adapter(self.robot_config)
        if self._robot is not None:
            self._robot.start()
        self._started = True

    def close(self):
        if self._robot is not None:
            self._robot.close()
        self._robot, self._started = None, False

    def robot_snapshot(self):
        """Use the same read-only adapter for explicit FK display and IK requests."""
        self.start()
        return None if self._robot is None else self._robot.snapshot()

    def _target(self, source, index, arm, *, flip=False):
        original = source["poses"][index]
        canonical = original @ np.diag([-1., -1., 1., 1.]) if flip else original
        conversion = source["conversions"].get(arm)
        pregrasp = tcp = None
        if conversion is not None:
            grasp = canonical @ _z(-self.settings["grasp_retreat_distance_m"])
            object_pregrasp = grasp @ _z(-self.settings["pregrasp_distance_m"]) @ conversion
            object_tcp = grasp @ conversion @ _z(-self.settings["grasp_advance_distance_m"])
            canonical = object_tcp @ np.linalg.inv(conversion)
            if source["base"] is not None:
                pregrasp, tcp = source["base"] @ object_pregrasp, source["base"] @ object_tcp
        record = source["record"]
        return {**{key: record[key] for key in ("object_id", "observation_id", "instance_id")},
                "index": index, "source_candidate_id": source["candidate_ids"][index],
                "object_from_gripper": canonical.tolist(),
                "camera_from_gripper": _list(None if source["camera"] is None else source["camera"] @ canonical),
                "base_from_gripper": _list(None if source["base"] is None else source["base"] @ canonical),
                "base_from_tcp": _list(tcp), "base_from_pregrasp": _list(pregrasp)}

    def _selected_target(self, target, source, arm, answer, *, dual):
        selected_grasp, selected_pregrasp = answer.get("selected_grasp"), answer.get("selected_pregrasp")
        if selected_grasp is None:
            return target
        grasp = np.asarray(selected_grasp[arm] if dual else selected_grasp)
        pregrasp = (target["base_from_pregrasp"] if selected_pregrasp is None else
                    selected_pregrasp[arm] if dual else selected_pregrasp)
        base_gripper = grasp @ np.linalg.inv(source["conversions"][arm])
        canonical = np.linalg.inv(source["base"]) @ base_gripper
        return {**target, "object_from_gripper": canonical.tolist(),
                "camera_from_gripper": _list(None if source["camera"] is None else source["camera"] @ canonical),
                "base_from_gripper": base_gripper.tolist(), "base_from_tcp": grasp.tolist(),
                "base_from_pregrasp": _list(pregrasp)}

    def _solve(self, candidate, sources, snapshot, *, dual, top_down=False):
        variants = [(False,)]
        arms = tuple(candidate["targets"])
        if dual:
            choices = [(False, True) if sources[target["observation_id"]]["metadata"]["gripper_metadata"]
                       .get("config", {}).get("symmetric", False) else (False,)
                       for target in candidate["targets"].values()]
            variants = list(itertools.product(*choices))
        best, checked = None, False
        for variant_index, flips in enumerate(variants):
            targets = {arm: self._target(sources[old["observation_id"]], old["index"], arm, flip=flip)
                       for (arm, old), flip in zip(candidate["targets"].items(), flips, strict=True)}
            if dual:
                left, right = targets["left"], targets["right"]
                answer = self._robot.solve_dual(left["base_from_pregrasp"], right["base_from_pregrasp"],
                                                left["base_from_tcp"], right["base_from_tcp"], seed=snapshot)
            else:
                arm = arms[0]
                answer = self._robot.solve_single(targets[arm]["base_from_pregrasp"], targets[arm]["base_from_tcp"],
                                                  arm, seed=snapshot)
            checked = checked or answer.get("self_collision_checked", False)
            if not answer["success"]:
                continue
            reachability = _reachability_score(snapshot, answer)
            if best is None or reachability > best["reachability_score"]:
                selected = {arm: self._selected_target(target, sources[target["observation_id"]], arm,
                                                       answer, dual=dual) for arm, target in targets.items()}
                best = {**candidate, "targets": selected, "ik_status": "solved",
                        "reachability_score": reachability,
                        "pair_score": .6 * candidate["geometry_score"] + .4 * reachability,
                        "target_variant_index": answer.get("target_variant_index"),
                        "symmetry_variant_index": variant_index,
                        "gripper_symmetry": dict(zip(arms, flips, strict=True))}
                if top_down:
                    target = selected[arms[0]]
                    metrics = _topdown_metrics(target["base_from_pregrasp"], target["base_from_tcp"])
                    best.update(**metrics, geometry_score=(.5 * metrics["topdown_score"] + .3 * candidate["model_score"]) / .8,
                                pair_score=.5 * metrics["topdown_score"]
                                + .3 * candidate["model_score"] + .2 * reachability)
        return best or {**candidate, "ik_status": "not_reachable"}, bool(checked)

    def predict(self, request):
        snapshot = self.robot_snapshot()
        records = {item["observation_id"]: item for item in request.objects}
        coordination = validate_coordination(request.coordination, records)
        mode = coordination["arm_mode"]
        if mode == "single" and self.settings["top_k"] > 32:
            raise ValueError("Single-arm top_k must be between 1 and 32")
        sources = {}
        for obs, record in records.items():
            camera = None if record["camera_from_object"] is None else np.asarray(record["camera_from_object"])
            base = None if snapshot is None or camera is None else np.asarray(snapshot["base_from_camera"]) @ camera
            fk, pose_source = record.get("grasp_fk"), "camera"
            if fk is not None and fk["state"] == "attached" and fk["instance_id"] == record["instance_id"]:
                base, pose_source = np.asarray(fk["base_from_object"]), "grasp_fk"
            sources[obs] = {"record": record, "camera": camera, "base": base, "pose_source": pose_source}
        result = {"arm_mode": mode, "robot_connected": snapshot is not None, "robot_ik_checked": False,
                  "self_collision_checked": False, "scene_collision_checked": False, "trajectory_checked": False,
                  "robot_command_published": False,
                  "objects": {obs: {"base_from_object": _list(source["base"]), "pose_source": source["pose_source"]}
                              for obs, source in sources.items()},
                  "candidates": [], "total_candidate_count": 0, "reason": None}
        assignments = {item["arm"]: item["observation_id"] for item in coordination["assignments"]}
        if (mode == "unknown" or None in assignments or None in assignments.values()
                or (mode == "single" and len(assignments) != 1)
                or (mode == "dual" and set(assignments) != {"left", "right"})):
            result["reason"] = "Arm/object assignments are unresolved"
            return result
        assignments = dict(sorted(assignments.items()))
        for obs in set(assignments.values()):
            source = sources[obs]
            record = source["record"]
            poses, scores, metadata = load_grasp_bundle(record["bundle_path"])
            geometry = metadata["asset_geometry"]
            if (geometry["mesh_sha256"] != sha256_file(record["mesh_path"])
                    or geometry["mesh_scale_to_m"] != record["mesh_scale_to_m"]):
                raise ValueError("Grasp bundle mesh/scale does not match the selected object")
            candidate_ids = tuple(f"graspgenx:grasp_{index}" for index in range(len(poses)))
            if Path(metadata["candidates_file"]).suffix.lower() in {".yaml", ".yml"}:
                import yaml

                directory = Path(record["bundle_path"])
                directory = directory if directory.is_dir() else directory.parent
                document = yaml.safe_load((directory / metadata["candidates_file"]).read_text(encoding="utf-8"))
                candidate_ids = tuple(f"graspgenx:{name}" for name in document["grasps"])
            source.update(poses=poses, scores=scores, metadata=metadata, conversions={}, candidate_ids=candidate_ids)
            if snapshot is not None and snapshot.get("gripper_id") == metadata["gripper_id"]:
                source["conversions"] = {arm: np.asarray(matrix) for arm, matrix in
                                         snapshot["canonical_gripper_from_tcp"].items()}
        arm_targets = {arm: [self._target(sources[obs], index, arm) for index in range(len(sources[obs]["poses"]))]
                       for arm, obs in assignments.items()}
        same_object = mode == "dual" and assignments["left"] == assignments["right"]
        com = gravity = None
        if same_object and sources[assignments["left"]]["base"] is not None:
            source = sources[assignments["left"]]
            record = source["record"]
            com = estimate_mesh_com(record["mesh_path"], record["mesh_scale_to_m"])
            gravity = source["base"][:3, :3].T @ np.array([0., 0., -1.])
        result["center_of_mass"] = com
        combinations = itertools.product(*arm_targets.values())
        candidates = []
        for targets in combinations:
            if same_object and targets[0]["index"] == targets[1]["index"]:
                continue
            values = [float(sources[target["observation_id"]]["scores"][target["index"]]) for target in targets]
            model = min(values) if same_object else float(np.mean(values))
            balance = None
            if same_object:
                contacts = [_contact(target, sources[target["observation_id"]]["metadata"]) for target in targets]
                balance = com_balance_metrics(*contacts, None if com is None else com["position_m"], gravity)
            geometry = model if balance is None or balance["score"] is None else .5 * model + .5 * balance["score"]
            target_map = dict(zip(assignments, targets, strict=True))
            distance = None
            if mode == "dual" and snapshot is not None and all(
                arm in snapshot.get("current_hands", {}) and target["base_from_tcp"] is not None
                for arm, target in target_map.items()
            ):
                distance = float(sum(np.linalg.norm(np.asarray(snapshot["current_hands"][arm])[:3, 3]
                    - np.asarray(target["base_from_tcp"])[:3, 3]) for arm, target in target_map.items()))
            candidates.append({"candidate_id": ":".join(f"{arm}:{target['observation_id']}:{target['index']}"
                                                         for arm, target in target_map.items()),
                               "targets": target_map, "model_score": model, "geometry_score": geometry,
                               "pair_score": geometry, "balance_metrics": balance, "ik_status": "not_checked",
                               "reachability_score": None, "assignment_distance_m": distance,
                               "search_priority_score": geometry / (1. + (distance or 0.))})
        candidates.sort(key=lambda item: (-item["search_priority_score"], -item["geometry_score"],
                                          -item["model_score"], item["candidate_id"]))
        can_solve = bool(candidates) and all(target["base_from_tcp"] is not None
                                            for target in candidates[0]["targets"].values())
        if can_solve:
            top_k = self.settings["top_k"]
            top_down = mode == "single" and self.settings["preference"] == "top_down"
            search = candidates
            if mode == "single":
                if top_down:
                    for candidate in candidates:
                        target = next(iter(candidate["targets"].values()))
                        metrics = _topdown_metrics(target["base_from_pregrasp"], target["base_from_tcp"])
                        score = (.5 * metrics["topdown_score"] + .3 * candidate["model_score"]) / .8
                        candidate.update(**metrics, preselection_score=score, geometry_score=score,
                                         pair_score=score, search_priority_score=score)
                    candidates.sort(key=lambda item: (-item["preselection_score"], -item["model_score"],
                                                      _source_candidate_id(item)))
                search = []
                for candidate in candidates:
                    if top_down and any(_near_topdown(candidate, other, self.settings["grasp_advance_distance_m"])
                                        for other in search):
                        continue
                    search.append(candidate)
                    if len(search) == top_k:
                        break
                for candidate in candidates:
                    candidate["ik_status"] = "not_in_mission_pool"
            for start in range(0, len(search), top_k):
                wave = search[start:start + top_k]
                for candidate in wave:
                    updated, checked = self._solve(candidate, sources, snapshot, dual=mode == "dual", top_down=top_down)
                    candidate.update(updated)
                    result["self_collision_checked"] |= checked
                    result["robot_ik_checked"] = True
                if mode == "single" or any(candidate["ik_status"] == "solved" for candidate in wave):
                    break
            candidates.sort(key=lambda item: ({"solved": 0, "not_reachable": 1, "not_checked": 2,
                                               "not_in_mission_pool": 2}[item["ik_status"]],
                                              -item["pair_score"], -item["model_score" if mode == "single" else
                                                                         "search_priority_score"],
                                              _source_candidate_id(item) if top_down else item["candidate_id"]))
        elif not candidates:
            result["reason"] = "No grasp candidates"
        elif snapshot is not None:
            result["reason"] = ("No matching gripper-to-TCP conversion" if any(
                not sources[obs]["conversions"] for obs in assignments.values()) else "Object pose is unavailable")
        else:
            result["reason"] = "Robot is not connected; canonical model-only candidates"
        result.update(candidates=candidates[:64] if mode == "dual" else candidates,
                      total_candidate_count=len(candidates))
        return result
