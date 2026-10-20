"""Receive in one thread and retain one owned frame for the UI timer."""
from collections import deque
from threading import Event, Lock, Thread
import time


class LatestCameraReceiver(Thread):
    def __init__(self, service, *, sam3=False, sam3d=False, foundationpose=False, vlm=False, graspgenx=False,
                 grasp_selection=False, subscriber_factory=None,
                 prompt_publisher_factory=None, result_subscriber_factory=None,
                 mesh_request_publisher_factory=None, mesh_result_subscriber_factory=None,
                 pose_control_publisher_factory=None, pose_result_subscriber_factory=None,
                 vlm_request_publisher_factory=None, vlm_result_subscriber_factory=None,
                 grasp_request_publisher_factory=None, grasp_result_subscriber_factory=None,
                 selection_request_publisher_factory=None, selection_result_subscriber_factory=None,
                 refine_request_publisher_factory=None, refine_result_subscriber_factory=None):
        super().__init__(name="camera-ui-receiver")
        self.service = service
        self._subscriber_factory = subscriber_factory
        self.sam3, self.sam3d = sam3 or sam3d or foundationpose or vlm, sam3d
        self.foundationpose = foundationpose
        self.vlm = vlm
        self.graspgenx, self.grasp_selection = graspgenx or grasp_selection, grasp_selection
        self._prompt_publisher_factory = prompt_publisher_factory
        self._result_subscriber_factory = result_subscriber_factory
        self._mesh_request_publisher_factory = mesh_request_publisher_factory
        self._mesh_result_subscriber_factory = mesh_result_subscriber_factory
        self._pose_control_publisher_factory = pose_control_publisher_factory
        self._pose_result_subscriber_factory = pose_result_subscriber_factory
        self._vlm_request_publisher_factory = vlm_request_publisher_factory
        self._vlm_result_subscriber_factory = vlm_result_subscriber_factory
        self._grasp_request_publisher_factory = grasp_request_publisher_factory
        self._grasp_result_subscriber_factory = grasp_result_subscriber_factory
        self._selection_request_publisher_factory = selection_request_publisher_factory
        self._selection_result_subscriber_factory = selection_result_subscriber_factory
        self._refine_request_publisher_factory = refine_request_publisher_factory
        self._refine_result_subscriber_factory = refine_result_subscriber_factory
        self._stop_requested, self._lock = Event(), Lock()
        self._latest = None
        self._sam3_latest, self._prompts, self._pending_prompts = None, None, None
        self._mesh_queue, self._mesh_results = deque(), deque(maxlen=8)
        self._mesh_previews = {}
        self._mesh_inflight = None
        self._pending_pose_control, self._pose_latest = None, None
        self._pending_vlm_request, self._vlm_latest = None, None
        self._grasp_queue, self._grasp_results = deque(), deque(maxlen=8)
        self._grasp_inflight = None
        self._pending_selection, self._selection_latest = None, None
        self._fk_latest = None
        self._pending_refine, self._refine_latest = None, None
        self._ready, self._error, self._received_count = False, None, 0
        self._created_at = time.monotonic()
        self._last_received = None

    def stop(self):
        self._stop_requested.set()

    def run(self):
        endpoints = []
        try:
            factory = self._subscriber_factory
            if factory is None:
                from ..ipc.camera import CameraSubscriber

                factory = CameraSubscriber
            subscriber = factory(self.service)
            endpoints.append(subscriber)
            prompt_publisher = result_subscriber = None
            if self.sam3:
                from ..ipc.sam3 import PromptPublisher, Sam3Subscriber, services

                prompt_service, result_service = services(self.service)
                prompt_publisher = (self._prompt_publisher_factory or PromptPublisher)(prompt_service)
                endpoints.append(prompt_publisher)
                result_subscriber = (self._result_subscriber_factory or Sam3Subscriber)(result_service)
                endpoints.append(result_subscriber)
            if self.sam3d:
                from ..ipc.sam3d import MeshRequestPublisher, MeshResultSubscriber, services

                request_service, result_service = services(self.service)
                mesh_publisher = (self._mesh_request_publisher_factory or MeshRequestPublisher)(request_service)
                endpoints.append(mesh_publisher)
                mesh_subscriber = (self._mesh_result_subscriber_factory or MeshResultSubscriber)(result_service)
                endpoints.append(mesh_subscriber)
            if self.foundationpose:
                from ..ipc.foundationpose import PoseControlPublisher, PoseResultSubscriber, services

                control_service, result_service = services(self.service)
                pose_publisher = (self._pose_control_publisher_factory or PoseControlPublisher)(control_service)
                endpoints.append(pose_publisher)
                pose_subscriber = (self._pose_result_subscriber_factory or PoseResultSubscriber)(result_service)
                endpoints.append(pose_subscriber)
            if self.vlm:
                from ..ipc.vlm import VlmRequestPublisher, VlmResultSubscriber, services

                request_service, result_service = services(self.service)
                vlm_publisher = (self._vlm_request_publisher_factory or VlmRequestPublisher)(request_service)
                endpoints.append(vlm_publisher)
                vlm_subscriber = (self._vlm_result_subscriber_factory or VlmResultSubscriber)(result_service)
                endpoints.append(vlm_subscriber)
            if self.graspgenx:
                from ..ipc.graspgenx import GraspRequestPublisher, GraspResultSubscriber, services

                request_service, result_service = services(self.service)
                grasp_publisher = (self._grasp_request_publisher_factory or GraspRequestPublisher)(request_service)
                endpoints.append(grasp_publisher)
                grasp_subscriber = (self._grasp_result_subscriber_factory or GraspResultSubscriber)(result_service)
                endpoints.append(grasp_subscriber)
            if self.grasp_selection:
                from ..ipc.grasp_selection import SelectionRequestPublisher, SelectionResultSubscriber, services

                request_service, result_service = services(self.service)
                selection_publisher = (self._selection_request_publisher_factory or SelectionRequestPublisher)(request_service)
                endpoints.append(selection_publisher)
                selection_subscriber = (self._selection_result_subscriber_factory or SelectionResultSubscriber)(result_service)
                endpoints.append(selection_subscriber)
            with self._lock:
                self._ready = True
            refine_publisher = refine_subscriber = None
            fk_request_id = None
            while not self._stop_requested.is_set():
                if self.sam3:
                    with self._lock:
                        prompts, self._pending_prompts = self._pending_prompts, None
                    if prompts is not None:
                        prompt_publisher.publish(prompts)
                    prompt_publisher.update_connections()
                    result = result_subscriber.receive_latest()
                    if result is not None:
                        with self._lock:
                            if (self._prompts is not None and self._prompts.objects
                                    and result.revision == self._prompts.revision):
                                self._sam3_latest = result
                if self.sam3d:
                    with self._lock:
                        request = None
                        if self._mesh_inflight is None and self._mesh_queue:
                            request = self._mesh_inflight = self._mesh_queue.popleft()
                    if request is not None:
                        mesh_publisher.publish(request)
                    mesh_publisher.update_connections()
                    result = mesh_subscriber.receive_latest()
                    if result is not None:
                        with self._lock:
                            matched = self._mesh_inflight is not None and result.request_id == self._mesh_inflight.request_id
                            current = self._prompts is None or result.revision == self._prompts.revision
                        preview = None
                        if matched and current and result.status == "done" and result.mesh is not None:
                            from .mesh_preview import preview_mesh

                            preview = preview_mesh(result.mesh)
                        with self._lock:
                            if matched:
                                current = self._prompts is None or result.revision == self._prompts.revision
                                if preview is not None and current and result.object_id in {"A", "B"}:
                                    self._mesh_previews[result.object_id] = (result.asset_id, preview)
                                self._mesh_results.append(result)
                                if result.status in {"done", "error"}:
                                    self._mesh_inflight = None
                if self.foundationpose:
                    with self._lock:
                        control, self._pending_pose_control = self._pending_pose_control, None
                    if control is not None:
                        pose_publisher.publish(control)
                    pose_publisher.update_connections()
                    result = pose_subscriber.receive_latest()
                    if result is not None:
                        with self._lock:
                            self._pose_latest = result
                if self.vlm:
                    with self._lock:
                        request, self._pending_vlm_request = self._pending_vlm_request, None
                    if request is not None:
                        vlm_publisher.publish(request)
                    vlm_publisher.update_connections()
                    result = vlm_subscriber.receive_latest()
                    if result is not None:
                        with self._lock:
                            self._vlm_latest = result
                if self.graspgenx:
                    with self._lock:
                        request = None
                        if self._grasp_inflight is None and self._grasp_queue:
                            request = self._grasp_inflight = self._grasp_queue.popleft()
                    if request is not None:
                        grasp_publisher.publish(request)
                    grasp_publisher.update_connections()
                    result = grasp_subscriber.receive_latest()
                    if result is not None:
                        with self._lock:
                            if self._grasp_inflight is not None and result.request_id == self._grasp_inflight.request_id:
                                self._grasp_results.append(result)
                                if result.status in {"done", "error"}:
                                    self._grasp_inflight = None
                if self.grasp_selection:
                    with self._lock:
                        request, self._pending_selection = self._pending_selection, None
                    if request is not None:
                        selection_publisher.publish(request)
                        if request.action != "select":
                            fk_request_id = request.request_id
                    selection_publisher.update_connections()
                    result = selection_subscriber.receive_latest()
                    if result is not None:
                        with self._lock:
                            if result.request_id == fk_request_id or result.selection is not None and result.selection.get("kind") == "grasp_fk":
                                self._fk_latest = result
                            else:
                                self._selection_latest = result
                if self.sam3:
                    with self._lock:
                        request, self._pending_refine = self._pending_refine, None
                    if request is not None:
                        if refine_publisher is None:
                            from ..ipc.sam3_refine import RefineRequestPublisher, RefineResultSubscriber, services

                            request_service, result_service = services(self.service)
                            refine_publisher = (self._refine_request_publisher_factory or RefineRequestPublisher)(request_service)
                            endpoints.append(refine_publisher)
                            refine_subscriber = (self._refine_result_subscriber_factory or RefineResultSubscriber)(result_service)
                            endpoints.append(refine_subscriber)
                        refine_publisher.publish(request)
                    if refine_publisher is not None:
                        refine_publisher.update_connections()
                        result = refine_subscriber.receive_latest()
                        if result is not None:
                            with self._lock:
                                self._refine_latest = result
                frame = subscriber.receive_latest()
                if frame is None:
                    self._stop_requested.wait(0.005)
                else:
                    self._accept(frame, time.monotonic())
        except Exception as error:
            with self._lock:
                self._error = f"{type(error).__name__}: {error}"
        finally:
            for endpoint in reversed(endpoints):
                try:
                    endpoint.close()
                except Exception as error:
                    with self._lock:
                        if self._error is None:
                            self._error = f"{type(error).__name__}: {error}"
            with self._lock:
                self._ready = False

    def set_prompts(self, prompts):
        with self._lock:
            self._prompts = self._pending_prompts = prompts
            self._sam3_latest = None
            self._mesh_queue = deque(item for item in self._mesh_queue if item.revision == prompts.revision)
            self._mesh_results.clear()
            self._mesh_previews.clear()

    def set_pose_control(self, control):
        with self._lock:
            self._pending_pose_control = control

    def submit_vlm(self, request):
        with self._lock:
            self._pending_vlm_request = request

    def submit_grasp(self, request):
        with self._lock:
            self._grasp_queue.append(request)

    def submit_selection(self, request):
        with self._lock:
            self._pending_selection = request

    def take_selection_latest(self):
        with self._lock:
            latest, self._selection_latest = self._selection_latest, None
            return latest

    def take_fk_latest(self):
        with self._lock:
            latest, self._fk_latest = self._fk_latest, None
            return latest

    def submit_refine(self, request):
        with self._lock:
            self._pending_refine = request

    def take_refine_latest(self):
        with self._lock:
            latest, self._refine_latest = self._refine_latest, None
            return latest

    def take_grasp_results(self):
        with self._lock:
            results = list(self._grasp_results)
            self._grasp_results.clear()
            return results

    def take_vlm_latest(self):
        with self._lock:
            latest, self._vlm_latest = self._vlm_latest, None
            return latest

    def take_pose_latest(self):
        with self._lock:
            latest, self._pose_latest = self._pose_latest, None
            return latest

    def mesh_preview(self, object_id, asset_id):
        with self._lock:
            cached = self._mesh_previews.get(object_id)
            return cached[1] if cached is not None and cached[0] == asset_id else None

    def submit_mesh(self, request):
        with self._lock:
            self._mesh_queue.append(request)

    def take_mesh_results(self):
        with self._lock:
            results = list(self._mesh_results)
            self._mesh_results.clear()
            return results

    def take_sam3_latest(self):
        with self._lock:
            latest, self._sam3_latest = self._sam3_latest, None
            return latest

    def _accept(self, frame, now):
        with self._lock:
            self._last_received = now
            self._received_count += 1
            self._latest = frame

    def take_latest(self):
        with self._lock:
            latest, self._latest = self._latest, None
            return latest

    def snapshot(self):
        with self._lock:
            return {"ready": self._ready, "error": self._error,
                    "received_count": self._received_count,
                    "elapsed_s": time.monotonic() - self._created_at,
                    "last_received_monotonic_s": self._last_received}
