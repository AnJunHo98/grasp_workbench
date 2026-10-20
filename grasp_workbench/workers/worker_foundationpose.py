"""Track A/B objects with the previous UI's SAM3 visual checks and recovery."""
import argparse

import numpy as np

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from ..ipc.foundationpose import ObjectPose, PoseResult, services
from ..models.foundationpose import visual_pose_mismatch_reason
from .worker_camera import DEFAULT_SERVICE


class FoundationPoseWorker(SingleRateWorker):
    def __init__(
        self, context, source_dir, device="cuda:0", registration_iterations=5,
        tracking_iterations=1, service=DEFAULT_SERVICE, frames=0, *,
        camera_subscriber_factory=None, control_subscriber_factory=None,
        result_publisher_factory=None, predictor_factory=None, sam3_subscriber_factory=None,
    ):
        super().__init__(context, hz=100)
        self.source_dir, self.device = source_dir, device
        self.registration_iterations, self.tracking_iterations = registration_iterations, tracking_iterations
        self.service, self.frames = service, frames
        self.camera_subscriber_factory = camera_subscriber_factory
        self.control_subscriber_factory = control_subscriber_factory
        self.result_publisher_factory, self.predictor_factory = result_publisher_factory, predictor_factory
        self.sam3_subscriber_factory = sam3_subscriber_factory
        self.camera = self.control_subscriber = self.publisher = self.predictor = None
        self.sam3_subscriber = None
        self.objects = {}
        self._targets, self._associations = {}, {}
        self.processed_count = 0

    def on_start(self):
        from ..ipc.camera import CameraSubscriber
        from ..ipc.foundationpose import PoseControlSubscriber, PoseResultPublisher
        from ..ipc.sam3 import Sam3Subscriber, services as sam3_services

        control_service, result_service = services(self.service)
        self.camera = (self.camera_subscriber_factory or CameraSubscriber)(self.service)
        self.control_subscriber = (self.control_subscriber_factory or PoseControlSubscriber)(control_service)
        self.publisher = (self.result_publisher_factory or PoseResultPublisher)(result_service)
        self.sam3_subscriber = (self.sam3_subscriber_factory or Sam3Subscriber)(sam3_services(self.service)[1])
        log_record("ready", worker=self.ctx.name, service=self.service)

    def _start_predictor(self):
        if self.predictor is not None:
            return
        if self.predictor_factory is None:
            from ..models.foundationpose import OfficialFoundationPosePredictor

            self.predictor_factory = OfficialFoundationPosePredictor
        predictor = self.predictor_factory(
            self.source_dir, device=self.device,
            registration_iterations=self.registration_iterations,
            tracking_iterations=self.tracking_iterations,
        )
        try:
            predictor.start()
        except Exception:
            predictor.close()
            raise
        self.predictor = predictor

    def _apply_control(self, control):
        from ..models.sam3_association import Sam3Association

        desired = {target.object_id: target for target in control.targets}
        for object_id, current in list(self.objects.items()):
            if object_id not in desired or desired[object_id].session_id != current.session_id:
                if self.predictor is not None:
                    self.predictor.remove(object_id)
                del self.objects[object_id]
                del self._targets[object_id]
                del self._associations[object_id]
        for target in control.targets:
            if target.object_id in self.objects:
                continue
            self._targets[target.object_id] = target
            self._associations[target.object_id] = Sam3Association(_mask_box(target.mask))
            current = self._register(target, target.frame, target.mask, initialize=True)
            self.objects[target.object_id] = current
            self.publisher.publish(PoseResult(target.frame, (current,)))

    def _register(self, target, frame, mask, box=None, *, initialize=False):
        current = ObjectPose(target.object_id, target.session_id, target.mesh_path, "tracking")
        try:
            self._start_predictor()
            if initialize and target.initialization == "seed":
                self.predictor.seed(target.object_id, target.initial_pose, target.mesh_path, target.mesh_scale_to_m)
                pose = self.predictor.track(target.object_id, frame)
            else:
                pose = self.predictor.register(target.object_id, frame, mask,
                                               target.mesh_path, target.mesh_scale_to_m)
            reason = visual_pose_mismatch_reason(frame, pose, mask, box) if box is not None else None
            if reason is not None:
                current.status, current.error = "recovering", reason
            else:
                current.camera_from_object = pose
        except Exception as error:
            current.status, current.error = "recovering", f"{type(error).__name__}: {error}"
        return current

    def _track_frame(self, frame):
        results = []
        for current in self.objects.values():
            if current.status == "tracking":
                try:
                    pose = self.predictor.track(current.object_id, frame)
                    current = ObjectPose(current.object_id, current.session_id, current.mesh_path,
                                         "tracking", pose,
                                         visual_mismatch_count=current.visual_mismatch_count)
                except Exception as error:
                    current = ObjectPose(current.object_id, current.session_id, current.mesh_path,
                                         "recovering", error=f"{type(error).__name__}: {error}",
                                         visual_mismatch_count=current.visual_mismatch_count)
                self.objects[current.object_id] = current
            results.append(current)
        self.publisher.publish(PoseResult(frame, tuple(results)))
        self.processed_count += 1
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()

    def _check_sam3(self, result):
        updates, used_masks = [], []
        for observation in result.objects:
            target = self._targets.get(observation.object_id)
            if target is None or target.prompt_revision != result.revision:
                continue
            current = self.objects[target.object_id]
            association = self._associations[target.object_id]
            indices, boxes = _candidate_boxes(observation, result.frame, used_masks)
            selected = association.select(boxes, observation.scores[indices])
            candidate = selected.candidate
            valid_depth_pixels = 0
            if candidate is None and selected.pending is not None:
                pending = selected.pending
                mask = observation.masks[indices[pending.index]]
                valid = mask & np.isfinite(result.frame.depth_m) & (result.frame.depth_m > 0)
                valid_depth_pixels = np.count_nonzero(valid)
                if valid_depth_pixels >= 16 and (
                    current.status == "recovering" or visual_pose_mismatch_reason(
                        result.frame, current.camera_from_object, mask, pending.box,
                    ) is None
                ):
                    candidate = association.accept_pending()
            evidence = candidate or selected.pending
            if evidence is None:
                continue  # Detector misses do not interrupt a working FP tracker.
            mask = observation.masks[indices[evidence.index]]
            used_masks.append(mask)
            if current.status == "tracking":
                reason = visual_pose_mismatch_reason(result.frame, current.camera_from_object,
                                                     mask, evidence.box)
                count = current.visual_mismatch_count + 1 if reason is not None else 0
                status = "recovering" if count >= 5 else "tracking"
                current = ObjectPose(current.object_id, current.session_id, current.mesh_path,
                                     status, current.camera_from_object if status == "tracking" else None,
                                     reason, count)
            if current.status == "recovering" and candidate is None and valid_depth_pixels >= 16:
                candidate = association.accept_pending()
            if current.status == "recovering" and candidate is not None:
                recovered = self._register(target, result.frame, observation.masks[indices[candidate.index]],
                                           candidate.box)
                if recovered.status == "recovering":
                    recovered.visual_mismatch_count = current.visual_mismatch_count
                current = recovered
            self.objects[target.object_id] = current
            updates.append(current)
        if updates:
            self.publisher.publish(PoseResult(result.frame, tuple(updates)))

    def step_once(self):
        self.publisher.update_connections()
        control = self.control_subscriber.receive_latest()
        if control is not None:
            self._apply_control(control)
        frame = self.camera.receive_latest()
        if frame is not None and self.objects:
            self._track_frame(frame)
        sam3 = self.sam3_subscriber.receive_latest()
        if sam3 is not None:
            self._check_sam3(sam3)

    def on_stop(self):
        try:
            if self.predictor is not None:
                self.predictor.close()
        finally:
            try:
                if self.publisher is not None:
                    self.publisher.close()
            finally:
                try:
                    if self.sam3_subscriber is not None:
                        self.sam3_subscriber.close()
                finally:
                    try:
                        if self.control_subscriber is not None:
                            self.control_subscriber.close()
                    finally:
                        if self.camera is not None:
                            self.camera.close()


def _mask_box(mask):
    if mask is None:
        return None
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return (float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1))


def _candidate_boxes(observation, frame, used_masks):
    """Keep the old source's nonempty masks and clipped, nondegenerate boxes."""
    height, width = frame.rgb.shape[:2]
    indices, boxes = [], []
    for index, (box, mask) in enumerate(zip(observation.boxes, observation.masks)):
        if (not mask.any() or not np.isfinite(box).all()
                or any(np.array_equal(mask, used) for used in used_masks)):
            continue
        clipped = np.clip(box, (0, 0, 0, 0), (width, height, width, height))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        indices.append(index)
        boxes.append(clipped)
    return indices, boxes


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--registration-iterations", type=int, default=5)
    parser.add_argument("--tracking-iterations", type=int, default=1)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--frames", type=int, default=0, help="Tracked frame results; 0 runs continuously")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    try:
        FoundationPoseWorker(
            WorkerContext("foundationpose"), args.source_dir, args.device,
            args.registration_iterations, args.tracking_iterations, args.service, args.frames,
        ).run()
    except Exception as error:
        log_record("error", worker="foundationpose", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
