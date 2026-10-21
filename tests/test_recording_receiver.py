"""Recording commands/status stay on the UI receiver's native endpoint thread."""
from collections import deque
from threading import Event, get_ident
import time

from grasp_workbench.ipc.recording import RecordingControl, RecordingStatus
from grasp_workbench.ui.receiver import LatestCameraReceiver


def test_recording_endpoints_are_owned_by_receiver_thread_and_work_with_aprilcube():
    calls, published = [], []
    received, sent = Event(), Event()
    status = RecordingStatus(state="recording", episode_frames=8)
    incoming = {"test/recording/status": deque([status])}

    class Endpoint:
        def __init__(self, service):
            self.service = service
            calls.append(("create", service, get_ident()))

        def publish(self, value):
            calls.append(("publish", self.service, get_ident()))
            published.append(value)
            sent.set()

        def update_connections(self):
            calls.append(("refresh", self.service, get_ident()))

        def receive_latest(self):
            calls.append(("read", self.service, get_ident()))
            values = incoming.get(self.service)
            if values:
                received.set()
                return values.popleft()
            return None

        def close(self):
            calls.append(("close", self.service, get_ident()))

    receiver = LatestCameraReceiver(
        "test", recording=True, aprilcube=True, subscriber_factory=Endpoint,
        pose_result_subscriber_factory=Endpoint, aprilcube_control_publisher_factory=Endpoint,
        aprilcube_overlay_subscriber_factory=Endpoint, recording_control_publisher_factory=Endpoint,
        recording_status_subscriber_factory=Endpoint,
    )
    command = RecordingControl("record-1", "start")
    receiver.send_recording_control(command)
    receiver.start()
    try:
        assert sent.wait(timeout=2) and received.wait(timeout=2)
        deadline = time.monotonic() + 2
        latest = None
        while latest is None and time.monotonic() < deadline:
            latest = receiver.take_recording_latest()
            time.sleep(.001)
        assert latest is status
        assert receiver.take_recording_latest() is None
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert published == [command]
    assert receiver.recording and receiver.aprilcube
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    created = [service for action, service, _thread in calls if action == "create"]
    assert created == ["test", "test/foundationpose/results", "test/aprilcube/control", "test/aprilcube/overlay",
                       "test/recording/control", "test/recording/status"]
    assert [service for action, service, _thread in calls if action == "close"] == created[::-1]
    assert {thread for _action, _service, thread in calls} == {receiver.ident}
    assert ("refresh", "test/recording/control", receiver.ident) in calls


def test_disabled_recording_does_not_construct_recording_endpoints():
    read = Event()

    class Camera:
        def __init__(self, _service):
            pass

        def receive_latest(self):
            read.set()
            return None

        def close(self):
            pass

    def forbidden(_service):
        raise AssertionError("Disabled recording endpoint opened")

    receiver = LatestCameraReceiver("test", subscriber_factory=Camera,
                                    recording_control_publisher_factory=forbidden,
                                    recording_status_subscriber_factory=forbidden)
    receiver.start()
    try:
        assert read.wait(timeout=2)
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert receiver.snapshot()["error"] is None and not receiver.is_alive()


def test_status_connection_failure_closes_already_created_recording_endpoints():
    closed = []

    class Endpoint:
        def __init__(self, service):
            self.service = service

        def close(self):
            closed.append(self.service)

    def failure(_service):
        raise RuntimeError("injected recording status connection failure")

    receiver = LatestCameraReceiver("test", recording=True, subscriber_factory=Endpoint,
                                    recording_control_publisher_factory=Endpoint,
                                    recording_status_subscriber_factory=failure)
    receiver.start()
    receiver.join(timeout=2)
    assert not receiver.is_alive()
    assert closed == ["test/recording/control", "test"]
    assert "injected recording status connection failure" in receiver.snapshot()["error"]
