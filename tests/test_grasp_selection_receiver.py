"""Selection IPC belongs to the receiver thread and retains only its latest values."""
from collections import deque
from threading import Event, get_ident
import time

from grasp_workbench.ipc.grasp_selection import SelectionRequest, SelectionResult
from grasp_workbench.ui.receiver import LatestCameraReceiver


def test_selection_latest_pending_and_result_endpoints_owned_by_receiver_thread():
    calls, published, incoming = [], [], deque()
    connected, delivered = Event(), Event()

    class Endpoint:
        def __init__(self, service):
            self.service, self.history = service, None
            calls.append(("create", service, get_ident()))

        def publish(self, value):
            calls.append(("publish", self.service, get_ident()))
            if self.service.endswith("grasp_selection/requests"):
                published.append(value)
                self.history = value

        def update_connections(self):
            calls.append(("refresh", self.service, get_ident()))
            if self.service.endswith("grasp_selection/requests") and self.history is not None and connected.is_set():
                delivered.set()

        def receive_latest(self):
            calls.append(("receive", self.service, get_ident()))
            return incoming.popleft() if self.service.endswith("grasp_selection/results") and incoming else None

        def close(self):
            calls.append(("close", self.service, get_ident()))

    receiver = LatestCameraReceiver("test", grasp_selection=True, subscriber_factory=Endpoint,
                                    grasp_request_publisher_factory=Endpoint, grasp_result_subscriber_factory=Endpoint,
                                    selection_request_publisher_factory=Endpoint, selection_result_subscriber_factory=Endpoint)
    first, last = (SelectionRequest(value, {}, ()) for value in ("first", "last"))
    receiver.submit_selection(first)
    receiver.submit_selection(last)
    receiver.start()

    def wait_for(predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert predicate()

    try:
        wait_for(lambda: bool(published))
        assert published == [last] and not delivered.is_set()
        connected.set()
        assert delivered.wait(timeout=2)
        incoming.append(SelectionResult("old", "running"))
        incoming.append(SelectionResult("last", "done", {"candidates": []}))
        wait_for(lambda: receiver._selection_latest is not None and receiver._selection_latest.status == "done")
        assert receiver.take_selection_latest().request_id == "last"
        assert receiver.take_selection_latest() is None
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert receiver.graspgenx and receiver.grasp_selection and not receiver.sam3
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert {row[2] for row in calls} == {receiver.ident}
    assert sum(row[0] == "create" for row in calls) == sum(row[0] == "close" for row in calls) == 5
    assert ("create", "test/grasp_selection/requests", receiver.ident) in calls
    assert ("create", "test/grasp_selection/results", receiver.ident) in calls


def test_fk_stream_and_attach_errors_do_not_replace_normal_selection():
    incoming, published = deque(), []

    class Endpoint:
        def __init__(self, service):
            self.service = service

        def publish(self, value):
            published.append(value)

        def update_connections(self):
            pass

        def receive_latest(self):
            return incoming.popleft() if self.service.endswith("grasp_selection/results") and incoming else None

        def close(self):
            pass

    receiver = LatestCameraReceiver("test", grasp_selection=True, subscriber_factory=Endpoint,
        grasp_request_publisher_factory=Endpoint, grasp_result_subscriber_factory=Endpoint,
        selection_request_publisher_factory=Endpoint, selection_result_subscriber_factory=Endpoint)
    receiver.submit_selection(SelectionRequest("attach", {}, (), action="attach", attachment={}))
    receiver.start()

    def wait_for(predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.001)
        assert predicate()

    try:
        wait_for(lambda: bool(published))
        incoming.append(SelectionResult("attach", "error", error="robot state absent"))
        wait_for(lambda: receiver._fk_latest is not None)
        assert receiver.take_fk_latest().error == "robot state absent"
        assert receiver.take_selection_latest() is None
        receiver.submit_selection(SelectionRequest("select", {}, ()))
        wait_for(lambda: len(published) == 2)
        incoming.append(SelectionResult("select", "done", {"candidates": []}))
        incoming.append(SelectionResult("attach", "done", {"kind": "grasp_fk", "state": "attached", "objects": {}}))
        wait_for(lambda: receiver._selection_latest is not None and receiver._fk_latest is not None)
        assert receiver.take_selection_latest().request_id == "select"
        assert receiver.take_fk_latest().request_id == "attach"
        assert receiver.take_fk_latest() is None
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert receiver.snapshot()["error"] is None
