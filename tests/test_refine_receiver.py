"""SAM3 frozen refinement endpoints are independently owned by the receiver thread."""
from collections import deque
from threading import get_ident
import time

from grasp_workbench.ipc.sam3_refine import RefineRequest, RefineResult
from grasp_workbench.ui.receiver import LatestCameraReceiver


def test_refine_keeps_latest_request_and_result_without_replacing_continuous_sam3():
    calls, incoming, published = [], deque(), []

    class Endpoint:
        def __init__(self, service):
            self.service = service
            calls.append(("create", service, get_ident()))

        def publish(self, request):
            calls.append(("publish", self.service, get_ident()))
            if self.service.endswith("refine/requests"):
                published.append(request)

        def update_connections(self):
            calls.append(("update", self.service, get_ident()))

        def receive_latest(self):
            calls.append(("receive", self.service, get_ident()))
            return incoming.popleft() if self.service.endswith("refine/results") and incoming else None

        def close(self):
            calls.append(("close", self.service, get_ident()))

    receiver = LatestCameraReceiver("test", sam3=True, subscriber_factory=Endpoint,
        prompt_publisher_factory=Endpoint, result_subscriber_factory=Endpoint,
        refine_request_publisher_factory=Endpoint, refine_result_subscriber_factory=Endpoint)
    first, last = (RefineRequest(value, "B", "ball", None) for value in ("first", "last"))
    receiver.submit_refine(first)
    receiver.submit_refine(last)
    receiver.start()

    def wait_for(predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.001)
        assert predicate()

    try:
        wait_for(lambda: bool(published))
        assert published == [last]
        incoming.append(RefineResult("last", "running"))
        incoming.append(RefineResult("last", "error", error="fixture error"))
        wait_for(lambda: receiver._refine_latest is not None and receiver._refine_latest.status == "error")
        assert receiver.take_refine_latest().error == "fixture error"
        assert receiver.take_refine_latest() is None and receiver.take_sam3_latest() is None
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert receiver.snapshot()["error"] is None and not receiver.is_alive()
    assert {row[2] for row in calls} == {receiver.ident}
    assert sum(row[0] == "create" for row in calls) == sum(row[0] == "close" for row in calls) == 5
    assert ("create", "test/sam3/refine/requests", receiver.ident) in calls
    assert ("create", "test/sam3/refine/results", receiver.ident) in calls
