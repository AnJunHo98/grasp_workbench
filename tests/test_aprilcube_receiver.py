"""AprilCube shares pose results, not FoundationPose/SAM3 control endpoints."""
from collections import deque
from threading import Event, get_ident
import time

import numpy as np

from grasp_workbench.ipc.aprilcube import AprilCubeControl, AprilCubeOverlay, AprilCubeTarget
from grasp_workbench.ipc.foundationpose import PoseResult
from grasp_workbench.ui.receiver import LatestCameraReceiver


def test_aprilcube_overrides_all_model_endpoints_and_owns_ipc_on_receiver_thread():
    calls, published = [], []
    sent = Event()
    pose = PoseResult(None, ())
    overlay = AprilCubeOverlay({"sequence": 12}, np.zeros((2, 3, 4), dtype=np.uint8))
    incoming = {"test/foundationpose/results": deque([pose]), "test/aprilcube/overlay": deque([overlay])}

    class Endpoint:
        def __init__(self, service):
            self.service = service
            calls.append(("create", service, get_ident()))

        def publish(self, value):
            published.append(value)
            calls.append(("publish", self.service, get_ident()))
            sent.set()

        def update_connections(self):
            calls.append(("refresh", self.service, get_ident()))

        def receive_latest(self):
            calls.append(("read", self.service, get_ident()))
            values = incoming.get(self.service)
            return values.popleft() if values else None

        def close(self):
            calls.append(("close", self.service, get_ident()))

    def forbidden(_service):
        raise AssertionError("Disabled model endpoint opened")

    receiver = LatestCameraReceiver(
        "test", aprilcube=True, sam3=True, sam3d=True, foundationpose=True, vlm=True,
        graspgenx=True, grasp_selection=True, subscriber_factory=Endpoint,
        pose_result_subscriber_factory=Endpoint, pose_control_publisher_factory=forbidden,
        prompt_publisher_factory=forbidden, mesh_request_publisher_factory=forbidden,
        vlm_request_publisher_factory=forbidden, grasp_request_publisher_factory=forbidden,
        selection_request_publisher_factory=forbidden,
        aprilcube_control_publisher_factory=Endpoint, aprilcube_overlay_subscriber_factory=Endpoint,
    )
    target = AprilCubeTarget("A", "new", "", 1., np.eye(4))
    receiver.send_aprilcube_control(AprilCubeControl(()))
    latest_control = AprilCubeControl((target,))
    receiver.send_aprilcube_control(latest_control)
    receiver.start()
    try:
        assert sent.wait(timeout=2)
        deadline = time.monotonic() + 2
        while receiver._aprilcube_overlay_latest is None and time.monotonic() < deadline:
            time.sleep(.001)
        assert receiver.take_pose_latest() is pose
        assert receiver.take_pose_latest() is None
        assert receiver.take_aprilcube_overlay_latest() is overlay
        assert receiver.take_aprilcube_overlay_latest() is None
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert receiver.aprilcube and not any((receiver.sam3, receiver.sam3d, receiver.foundationpose,
                                           receiver.vlm, receiver.graspgenx, receiver.grasp_selection))
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert published == [latest_control]
    created = [service for operation, service, _thread in calls if operation == "create"]
    assert created == ["test", "test/foundationpose/results", "test/aprilcube/control", "test/aprilcube/overlay"]
    assert [service for operation, service, _thread in calls if operation == "close"] == created[::-1]
    assert {thread for _operation, _service, thread in calls} == {receiver.ident}
    assert ("refresh", "test/aprilcube/control", receiver.ident) in calls


def test_overlay_connection_failure_closes_already_created_aprilcube_endpoints():
    calls = []

    class Endpoint:
        def __init__(self, service):
            self.service = service

        def close(self):
            calls.append(self.service)

    def failure(_service):
        raise RuntimeError("injected overlay connection failure")

    receiver = LatestCameraReceiver("test", aprilcube=True, subscriber_factory=Endpoint,
                                    pose_result_subscriber_factory=Endpoint,
                                    aprilcube_control_publisher_factory=Endpoint,
                                    aprilcube_overlay_subscriber_factory=failure)
    receiver.start()
    receiver.join(timeout=2)
    assert not receiver.is_alive()
    assert calls == ["test/aprilcube/control", "test/foundationpose/results", "test"]
    assert "injected overlay connection failure" in receiver.snapshot()["error"]
