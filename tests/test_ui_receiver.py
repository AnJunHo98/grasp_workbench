from collections import deque
from threading import Event, get_ident
import time
from types import SimpleNamespace

from grasp_workbench.ui.receiver import LatestCameraReceiver


def frame(sequence=0, stream="one", camera="camera", captured=123):
    return SimpleNamespace(metadata={"camera_id": camera, "stream_id": stream,
                                     "sequence": sequence, "capture_time_ns": captured})


def test_receive_count_does_not_change_when_ui_repeats_poll():
    receiver = LatestCameraReceiver("test")
    receiver._accept(frame(captured=10**20), 10)
    receiver._accept(frame(sequence=3, captured=1), 10.05)
    state = receiver.snapshot()
    assert state["received_count"] == 2
    assert state["last_received_monotonic_s"] == 10.05
    assert receiver.take_latest().metadata["sequence"] == 3
    assert receiver.take_latest() is None
    assert receiver.snapshot()["received_count"] == 2


def test_new_stream_preserves_original_metadata():
    receiver = LatestCameraReceiver("test")
    receiver._accept(frame(), 1)
    receiver._accept(frame(1), 2)
    receiver._accept(frame(stream="new", camera="another", captured=100), 100)
    latest = receiver.take_latest()
    assert latest.metadata["stream_id"] == "new"
    assert latest.metadata["camera_id"] == "another"
    assert latest.metadata["capture_time_ns"] == 100


def test_one_latest_frame_not_a_growing_frame_queue():
    receiver = LatestCameraReceiver("test")
    for index in range(1000):
        receiver._accept(frame(index), 1 + index / 30)
    state = receiver.snapshot()
    assert state["received_count"] == 1000
    assert receiver.take_latest().metadata["sequence"] == 999
    assert receiver.take_latest() is None


def test_subscriber_construct_receive_and_close_owned_by_receiver_thread():
    identities = []
    read = Event()

    class Subscriber:
        def __init__(self, _service):
            identities.append(get_ident())

        def receive_latest(self):
            identities.append(get_ident())
            read.set()
            return None

        def close(self):
            identities.append(get_ident())

    receiver = LatestCameraReceiver("test", subscriber_factory=Subscriber)
    receiver.start()
    try:
        assert read.wait(timeout=2)
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert not receiver.is_alive()
    assert set(identities) == {receiver.ident}
    assert get_ident() not in identities


def test_receiver_failure_records_error_and_closes():
    closed = Event()

    class Subscriber:
        def __init__(self, _service):
            pass

        def receive_latest(self):
            raise RuntimeError("injected receive failure")

        def close(self):
            closed.set()

    receiver = LatestCameraReceiver("test", subscriber_factory=Subscriber)
    receiver.start()
    receiver.join(timeout=2)
    assert closed.is_set() and not receiver.is_alive()
    assert "injected receive failure" in receiver.snapshot()["error"]


def test_snapshot_does_not_expose_mutable_receiver_state():
    receiver = LatestCameraReceiver("test")
    receiver._accept(frame(), 10)
    receiver._accept(frame(1), 11)
    state = receiver.snapshot()
    state["received_count"] = 0
    assert receiver.snapshot()["received_count"] == 2


def test_sam3_endpoints_publish_receive_close_on_receiver_thread_and_latest_revision_only():
    from grasp_workbench.ipc.sam3 import ObjectPrompt, PromptSet, Sam3Result

    calls, published = [], []
    accepted, allow_result, sent = Event(), Event(), Event()
    prompt = PromptSet("current", (ObjectPrompt("A", "cup"),))

    class CameraSubscriber:
        def __init__(self, service):
            calls.append(("camera-create", service, get_ident()))

        def receive_latest(self):
            calls.append(("camera-read", "", get_ident()))
            return None

        def close(self):
            calls.append(("camera-close", "", get_ident()))

    class Publisher:
        def __init__(self, service):
            calls.append(("prompt-create", service, get_ident()))

        def publish(self, value):
            calls.append(("prompt-publish", "", get_ident()))
            published.append(value)

        def update_connections(self):
            calls.append(("prompt-refresh", "", get_ident()))

        def close(self):
            calls.append(("prompt-close", "", get_ident()))

    class ResultSubscriber:
        def __init__(self, service):
            calls.append(("result-create", service, get_ident()))
            self.old_sent = False

        def receive_latest(self):
            calls.append(("result-read", "", get_ident()))
            if not self.old_sent:
                self.old_sent = True
                accepted.set()
                return Sam3Result(frame(1), "obsolete", ())
            if allow_result.is_set() and not sent.is_set():
                sent.set()
                return Sam3Result(frame(99), prompt.revision, ())
            return None

        def close(self):
            calls.append(("result-close", "", get_ident()))

    receiver = LatestCameraReceiver("camera-test", sam3=True, subscriber_factory=CameraSubscriber,
                                    prompt_publisher_factory=Publisher, result_subscriber_factory=ResultSubscriber)
    receiver.set_prompts(PromptSet("superseded-before-start", (ObjectPrompt("B", "ignored"),)))
    receiver.set_prompts(prompt)
    receiver.start()
    try:
        assert accepted.wait(timeout=2)
        assert receiver.take_sam3_latest() is None
        allow_result.set()
        assert sent.wait(timeout=2)
        deadline = time.monotonic() + 2
        result = None
        while result is None and time.monotonic() < deadline:
            result = receiver.take_sam3_latest()
            time.sleep(0.001)
        assert result is not None and result.frame.metadata["sequence"] == 99
        assert receiver.take_sam3_latest() is None
        receiver.set_prompts(PromptSet("stop", ()))
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert published[0] is prompt
    assert set(item[2] for item in calls) == {receiver.ident}
    assert {item[0] for item in calls} >= {"camera-close", "prompt-close", "result-close", "prompt-publish", "prompt-refresh"}
    assert ("prompt-create", "camera-test/sam3/prompts", receiver.ident) in calls
    assert ("result-create", "camera-test/sam3/results", receiver.ident) in calls


def test_sam3_partial_start_failure_closes_created_camera_and_prompt_endpoints():
    calls = []

    class Endpoint:
        def __init__(self, service):
            self.service = service
            calls.append(("create", service, get_ident()))

        def close(self):
            calls.append(("close", self.service, get_ident()))

    def failure(service):
        raise RuntimeError("result endpoint failed")

    receiver = LatestCameraReceiver("camera-test", sam3=True, subscriber_factory=Endpoint,
                                    prompt_publisher_factory=Endpoint, result_subscriber_factory=failure)
    receiver.start()
    receiver.join(timeout=2)
    assert not receiver.is_alive() and "result endpoint failed" in receiver.snapshot()["error"]
    assert {call[1] for call in calls if call[0] == "close"} == {"camera-test", "camera-test/sam3/prompts"}
    assert set(call[2] for call in calls) == {receiver.ident}


def test_prompt_change_discards_pending_result_immediately():
    from grasp_workbench.ipc.sam3 import ObjectPrompt, PromptSet, Sam3Result

    receiver = LatestCameraReceiver("test", sam3=True)
    receiver.set_prompts(PromptSet("old", (ObjectPrompt("A", "cup"),)))
    receiver._sam3_latest = Sam3Result(frame(), "old", ())
    receiver.set_prompts(PromptSet("new", ()))
    assert receiver.take_sam3_latest() is None


def test_prompt_connection_refresh_delivers_history_to_late_joiner_without_republishing():
    from grasp_workbench.ipc.sam3 import ObjectPrompt, PromptSet

    published, connected, delivered = Event(), Event(), Event()
    identities, published_values = [], []

    class Subscriber:
        def __init__(self, _service):
            pass

        def receive_latest(self):
            return None

        def close(self):
            pass

    class Publisher:
        def __init__(self, _service):
            self.history = None

        def publish(self, prompts):
            identities.append(get_ident())
            published_values.append(prompts)
            self.history = prompts
            published.set()

        def update_connections(self):
            identities.append(get_ident())
            if connected.is_set() and self.history is not None:
                delivered.set()

        def close(self):
            identities.append(get_ident())

    prompts = PromptSet("same-revision", (ObjectPrompt("A", "cup"),))
    receiver = LatestCameraReceiver("test", sam3=True, subscriber_factory=Subscriber,
                                    prompt_publisher_factory=Publisher, result_subscriber_factory=Subscriber)
    receiver.set_prompts(prompts)
    receiver.start()
    try:
        assert published.wait(timeout=2)
        assert not delivered.is_set()
        connected.set()
        assert delivered.wait(timeout=2)
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert published_values == [prompts]
    assert len(identities) >= 3 and set(identities) == {receiver.ident}


def test_mesh_queue_waits_for_matching_terminal_and_releases_old_revision_on_owner_thread():
    from grasp_workbench.ipc.sam3 import PromptSet
    from grasp_workbench.ipc.sam3d import MeshResult

    calls, published, incoming = [], [], deque()

    class Endpoint:
        def __init__(self, service):
            self.service = service
            calls.append(("create", service, get_ident()))

        def receive_latest(self):
            calls.append(("receive", self.service, get_ident()))
            return incoming.popleft() if self.service.endswith("sam3d/results") and incoming else None

        def publish(self, request):
            calls.append(("publish", self.service, get_ident()))
            if self.service.endswith("sam3d/requests"):
                published.append(request)

        def update_connections(self):
            calls.append(("refresh", self.service, get_ident()))

        def close(self):
            calls.append(("close", self.service, get_ident()))

    def wait_for(predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert predicate()

    a = SimpleNamespace(request_id="request-a", asset_id="asset-a", object_id="A", revision="old")
    old_b = SimpleNamespace(request_id="old-b", asset_id="old-asset-b", object_id="B", revision="old")
    b = SimpleNamespace(request_id="request-b", asset_id="asset-b", object_id="B", revision="new")
    receiver = LatestCameraReceiver("test", sam3d=True, subscriber_factory=Endpoint,
                                    prompt_publisher_factory=Endpoint, result_subscriber_factory=Endpoint,
                                    mesh_request_publisher_factory=Endpoint, mesh_result_subscriber_factory=Endpoint)
    receiver.set_prompts(PromptSet("old", ()))
    receiver.submit_mesh(a)
    receiver.submit_mesh(old_b)
    receiver.start()
    try:
        wait_for(lambda: len(published) == 1)
        incoming.append(MeshResult(a.request_id, a.asset_id, "A", "old", "running"))
        time.sleep(0.025)
        assert published == [a]
        receiver.set_prompts(PromptSet("new", ()))
        receiver.submit_mesh(b)
        incoming.append(MeshResult("unmatched", "other", "B", "new", "done"))
        time.sleep(0.025)
        assert published == [a]
        incoming.append(MeshResult(a.request_id, a.asset_id, "A", "old", "error", error="old job done"))
        wait_for(lambda: len(published) == 2)
        assert published == [a, b]
        incoming.append(MeshResult(b.request_id, b.asset_id, "B", "new", "done"))
        wait_for(lambda: receiver._mesh_inflight is None)
        events = receiver.take_mesh_results()
        assert [(item.request_id, item.status) for item in events] == [(a.request_id, "error"), (b.request_id, "done")]
        assert receiver.take_mesh_results() == []
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert {item[2] for item in calls} == {receiver.ident}
    assert sum(item[0] == "create" for item in calls) == sum(item[0] == "close" for item in calls) == 5
    assert any(item[:2] == ("refresh", "test/sam3d/requests") for item in calls)


def test_mesh_endpoint_partial_start_failure_closes_other_endpoints():
    calls = []

    class Endpoint:
        def __init__(self, service):
            self.service = service

        def close(self):
            calls.append((self.service, get_ident()))

    def failure(service):
        raise RuntimeError("mesh result endpoint failed")

    receiver = LatestCameraReceiver("test", sam3d=True, subscriber_factory=Endpoint,
                                    prompt_publisher_factory=Endpoint, result_subscriber_factory=Endpoint,
                                    mesh_request_publisher_factory=Endpoint, mesh_result_subscriber_factory=failure)
    receiver.start()
    receiver.join(timeout=2)
    assert not receiver.is_alive() and "mesh result endpoint failed" in receiver.snapshot()["error"]
    assert len(calls) == 4 and {item[1] for item in calls} == {receiver.ident}


def test_pose_control_and_latest_results_use_receiver_thread_without_model_imports():
    from grasp_workbench.ipc.foundationpose import PoseControl, PoseResult

    calls, published = [], []
    incoming, received = deque(), Event()

    class Endpoint:
        def __init__(self, service):
            self.service = service
            calls.append(("create", service, get_ident()))

        def publish(self, value):
            calls.append(("publish", self.service, get_ident()))
            if self.service.endswith("foundationpose/control"):
                published.append(value)

        def update_connections(self):
            calls.append(("refresh", self.service, get_ident()))

        def receive_latest(self):
            calls.append(("receive", self.service, get_ident()))
            if self.service.endswith("foundationpose/results") and incoming:
                result = incoming.popleft()
                received.set()
                return result
            return None

        def close(self):
            calls.append(("close", self.service, get_ident()))

    receiver = LatestCameraReceiver("test", foundationpose=True, subscriber_factory=Endpoint,
                                    prompt_publisher_factory=Endpoint, result_subscriber_factory=Endpoint,
                                    pose_control_publisher_factory=Endpoint, pose_result_subscriber_factory=Endpoint)
    control = PoseControl(())
    receiver.set_pose_control(PoseControl(()))
    receiver.set_pose_control(control)
    result = PoseResult(frame(99), ())
    incoming.append(result)
    receiver.start()
    try:
        assert received.wait(timeout=2)
        deadline = time.monotonic() + 2
        observed = None
        while observed is None and time.monotonic() < deadline:
            observed = receiver.take_pose_latest()
            time.sleep(0.001)
        assert observed is result and receiver.take_pose_latest() is None
        assert published == [control]
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert receiver.sam3 and not receiver.sam3d
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert {item[2] for item in calls} == {receiver.ident}
    assert sum(item[0] == "create" for item in calls) == sum(item[0] == "close" for item in calls) == 5
    assert ("create", "test/foundationpose/control", receiver.ident) in calls
    assert ("create", "test/foundationpose/results", receiver.ident) in calls
    assert ("refresh", "test/foundationpose/control", receiver.ident) in calls


def test_vlm_single_pending_request_and_latest_results_stay_on_receiver_thread():
    from grasp_workbench.ipc.vlm import VlmRequest, VlmResult

    calls, published, incoming = [], [], deque()
    received = Event()

    class Endpoint:
        def __init__(self, service):
            self.service = service
            calls.append(("create", service, get_ident()))

        def publish(self, value):
            calls.append(("publish", self.service, get_ident()))
            if self.service.endswith("vlm/requests"):
                published.append(value)

        def update_connections(self):
            calls.append(("refresh", self.service, get_ident()))

        def receive_latest(self):
            calls.append(("receive", self.service, get_ident()))
            if self.service.endswith("vlm/results") and incoming:
                received.set()
                return incoming.popleft()
            return None

        def close(self):
            calls.append(("close", self.service, get_ident()))

    request = VlmRequest("current", frame(17), "find two objects", ())
    running = VlmResult(request.request_id, request.frame, "running")
    done = VlmResult(request.request_id, request.frame, "done")
    receiver = LatestCameraReceiver("test", vlm=True, subscriber_factory=Endpoint,
                                    prompt_publisher_factory=Endpoint, result_subscriber_factory=Endpoint,
                                    vlm_request_publisher_factory=Endpoint, vlm_result_subscriber_factory=Endpoint)
    receiver.submit_vlm(VlmRequest("replaced-before-send", frame(), "old", ()))
    receiver.submit_vlm(request)
    incoming.append(running)
    receiver.start()

    def wait_result():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = receiver.take_vlm_latest()
            if result is not None:
                return result
            time.sleep(.001)
        raise AssertionError("VLM result was not received")

    try:
        assert received.wait(timeout=2)
        assert wait_result() is running and receiver.take_vlm_latest() is None
        incoming.append(done)
        assert wait_result() is done and receiver.take_vlm_latest() is None
        assert published == [request]
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert receiver.sam3 and receiver.vlm and not receiver.sam3d and not receiver.foundationpose
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert {item[2] for item in calls} == {receiver.ident}
    assert sum(item[0] == "create" for item in calls) == sum(item[0] == "close" for item in calls) == 5
    assert ("create", "test/vlm/requests", receiver.ident) in calls
    assert ("create", "test/vlm/results", receiver.ident) in calls
    assert ("refresh", "test/vlm/requests", receiver.ident) in calls


def test_vlm_partial_endpoint_start_failure_closes_created_endpoints():
    closed = []

    class Endpoint:
        def __init__(self, service):
            self.service = service

        def close(self):
            closed.append((self.service, get_ident()))

    def fail(_service):
        raise RuntimeError("VLM result subscriber failed")

    receiver = LatestCameraReceiver("test", vlm=True, subscriber_factory=Endpoint,
                                    prompt_publisher_factory=Endpoint, result_subscriber_factory=Endpoint,
                                    vlm_request_publisher_factory=Endpoint, vlm_result_subscriber_factory=fail)
    receiver.start()
    receiver.join(timeout=2)
    assert not receiver.is_alive() and "VLM result subscriber failed" in receiver.snapshot()["error"]
    assert {identity for _, identity in closed} == {receiver.ident}
    assert [service for service, _ in closed] == [
        "test/vlm/requests", "test/sam3/results", "test/sam3/prompts", "test"]


def test_grasp_a_b_queue_matching_terminals_and_endpoints_owned_by_receiver_thread():
    from grasp_workbench.ipc.graspgenx import GraspRequest, GraspResult

    calls, published, incoming = [], [], deque()

    class Endpoint:
        def __init__(self, service):
            self.service = service
            calls.append(("create", service, get_ident()))

        def receive_latest(self):
            calls.append(("receive", self.service, get_ident()))
            return incoming.popleft() if self.service.endswith("graspgenx/results") and incoming else None

        def publish(self, value):
            calls.append(("publish", self.service, get_ident()))
            published.append(value)

        def update_connections(self):
            calls.append(("refresh", self.service, get_ident()))

        def close(self):
            calls.append(("close", self.service, get_ident()))

    def wait_for(predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.001)
        assert predicate()

    a, b = GraspRequest("a", "asset-a", "A", "a.ply"), GraspRequest("b", "asset-b", "B", "b.ply")
    receiver = LatestCameraReceiver("test", graspgenx=True, subscriber_factory=Endpoint,
                                    grasp_request_publisher_factory=Endpoint, grasp_result_subscriber_factory=Endpoint)
    receiver.submit_grasp(a)
    receiver.submit_grasp(b)
    receiver.start()
    try:
        wait_for(lambda: len(published) == 1)
        incoming.append(GraspResult("unmatched", "other", "B", "done"))
        incoming.append(GraspResult(a.request_id, a.asset_id, "A", "running"))
        time.sleep(.025)
        assert published == [a]
        incoming.append(GraspResult(a.request_id, a.asset_id, "A", "error", error="failed"))
        wait_for(lambda: len(published) == 2)
        assert published == [a, b]
        incoming.append(GraspResult(b.request_id, b.asset_id, "B", "done"))
        wait_for(lambda: receiver._grasp_inflight is None)
        assert [(value.request_id, value.status) for value in receiver.take_grasp_results()] == [
            ("a", "running"), ("a", "error"), ("b", "done")]
        assert receiver.take_grasp_results() == []
    finally:
        receiver.stop()
        receiver.join(timeout=2)
    assert not receiver.sam3 and not receiver.foundationpose and not receiver.sam3d and not receiver.vlm
    assert not receiver.is_alive() and receiver.snapshot()["error"] is None
    assert {item[2] for item in calls} == {receiver.ident}
    assert sum(item[0] == "create" for item in calls) == sum(item[0] == "close" for item in calls) == 3
    assert ("refresh", "test/graspgenx/requests", receiver.ident) in calls


def test_grasp_partial_endpoint_start_failure_closes_created_endpoints():
    closed = []

    class Endpoint:
        def __init__(self, service):
            self.service = service

        def close(self):
            closed.append((self.service, get_ident()))

    def fail(_service):
        raise RuntimeError("grasp result subscriber failed")

    receiver = LatestCameraReceiver("test", graspgenx=True, subscriber_factory=Endpoint,
                                    grasp_request_publisher_factory=Endpoint, grasp_result_subscriber_factory=fail)
    receiver.start()
    receiver.join(timeout=2)
    assert not receiver.is_alive() and "grasp result subscriber failed" in receiver.snapshot()["error"]
    assert {identity for _, identity in closed} == {receiver.ident}
    assert [service for service, _ in closed] == ["test/graspgenx/requests", "test"]
