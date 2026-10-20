"""Display-only frozen grasp attachments propagated by measured hand FK."""
from copy import deepcopy

import numpy as np


def _quaternion(rotation):
    """Unit wxyz quaternion with the same positive-w convention as the old SE3."""
    trace = np.trace(rotation)
    if trace > 0:
        scale = np.sqrt(trace + 1.) * 2.
        value = np.array((scale / 4., (rotation[2, 1] - rotation[1, 2]) / scale,
                          (rotation[0, 2] - rotation[2, 0]) / scale,
                          (rotation[1, 0] - rotation[0, 1]) / scale))
    else:
        index = int(np.argmax(np.diag(rotation)))
        other, last = (index + 1) % 3, (index + 2) % 3
        scale = np.sqrt(1. + rotation[index, index] - rotation[other, other] - rotation[last, last]) * 2.
        value = np.zeros(4)
        value[0] = (rotation[last, other] - rotation[other, last]) / scale
        value[index + 1] = scale / 4.
        value[other + 1] = (rotation[other, index] + rotation[index, other]) / scale
        value[last + 1] = (rotation[last, index] + rotation[index, last]) / scale
    value /= np.linalg.norm(value)
    return -value if value[0] < 0 else value


def _average(poses):
    quaternions = [_quaternion(pose[:3, :3]) for pose in poses]
    total = sum((-value if np.dot(quaternions[0], value) < 0 else value) for value in quaternions)
    w, x, y, z = total / np.linalg.norm(total)
    matrix = np.eye(4)
    matrix[:3, :3] = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
    matrix[:3, 3] = np.mean([pose[:3, 3] for pose in poses], axis=0)
    return matrix


class GraspFkTracker:
    def __init__(self):
        self.objects = {}

    def attach(self, reference):
        """Freeze selected object↔control-TCP relations, never measured contact."""
        objects = {}
        for arm, target in reference["selected_candidate"]["targets"].items():
            observation = target["observation_id"]
            base_from_object = reference["object_poses"][observation]["base_from_object"]
            base_from_tcp = target["base_from_tcp"]
            if base_from_object is None or base_from_tcp is None:
                raise ValueError("Fixing a grasp relation requires base-frame object and TCP poses")
            relation = np.linalg.inv(np.asarray(base_from_object)) @ np.asarray(base_from_tcp)
            item = objects.setdefault(observation, {
                name: target[name] for name in ("object_id", "observation_id", "instance_id")})
            item.setdefault("object_from_tcp_by_arm", {})[arm] = relation.tolist()
        if not objects:
            raise ValueError("Select a grasp before fixing its relation")
        self.objects = objects

    def detach(self):
        self.objects = {}
        return {"kind": "grasp_fk", "state": "detached", "objects": {}, "robot_command_published": False}

    def estimate(self, snapshot):
        records = {}
        for observation, frozen in self.objects.items():
            item = {**deepcopy(frozen), "state": "waiting_for_robot_state", "reference_frame": "base_link",
                    "base_from_object": None, "base_from_object_by_arm": {},
                    "observed_sequence": None if snapshot is None else snapshot.get("state_sequence"),
                    "observed_timestamp_ns": None if snapshot is None else snapshot.get("state_timestamp_ns"),
                    "translation_disagreement_m": None, "rotation_disagreement_rad": None,
                    "reason": "waiting for measured robot hand poses"}
            hands = {} if snapshot is None else snapshot.get("current_hands", {})
            relations = frozen["object_from_tcp_by_arm"]
            if (item["observed_sequence"] is not None and item["observed_timestamp_ns"] is not None
                    and all(arm in hands for arm in relations)):
                estimates = {arm: np.asarray(hands[arm]) @ np.linalg.inv(np.asarray(relation))
                             for arm, relation in relations.items()}
                values = list(estimates.values())
                first, last = values[0], values[-1]
                cosine = abs(np.dot(_quaternion(first[:3, :3]), _quaternion(last[:3, :3])))
                item.update(state="attached", base_from_object=_average(values).tolist(),
                            base_from_object_by_arm={arm: matrix.tolist() for arm, matrix in estimates.items()},
                            translation_disagreement_m=float(np.linalg.norm(first[:3, 3] - last[:3, 3])),
                            rotation_disagreement_rad=float(2 * np.arccos(np.clip(cosine, 0., 1.))),
                            reason="frozen planned relation propagated from measured hand FK")
            records[observation] = item
        return {"kind": "grasp_fk", "state": "attached" if records and all(
            item["state"] == "attached" for item in records.values()) else "waiting_for_robot_state",
                "objects": records, "robot_command_published": False}
