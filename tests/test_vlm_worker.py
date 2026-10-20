"""Worker lifecycle and native subprocess transport with synthetic predictors only."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.vlm import VlmObject, VlmRequest, VlmRequestPublisher, VlmResultSubscriber, services
from grasp_workbench.models.vlm import VlmError
from grasp_workbench.workers.worker_vlm import VlmWorker, build_parser


def request(request_id="job-1"):
    return VlmRequest(request_id, CameraFrame(np.zeros((3, 4, 3), np.uint8),
                                             np.ones((3, 4), np.float32), {"sequence": 9}),
                      "tape and tuna", ({"asset_id": "can", "name": "tuna", "prompt": "yellow tuna can"},))


class Inbox:
    def __init__(self):
        self.value, self.closed = None, False

    def receive_latest(self):
        value, self.value = self.value, None
        return value

    def close(self):
        self.closed = True


class Output:
    def __init__(self):
        self.results, self.closed, self.refreshes = [], False, 0

    def update_connections(self):
        self.refreshes += 1

    def publish(self, value):
        self.results.append(value)

    def close(self):
        self.closed = True


class Predictor:
    def __init__(self):
        self.starts, self.requests, self.closed = 0, [], False
        self.error = None

    def start(self):
        self.starts += 1

    def predict(self, value):
        self.requests.append(value)
        if self.error is not None:
            raise self.error
        return (VlmObject("tuna", "yellow tuna can", None, "can"),)

    def close(self):
        self.closed = True


def worker(frames=0):
    inbox, output, predictor, calls = Inbox(), Output(), Predictor(), []

    def make(**options):
        calls.append(options)
        return predictor

    instance = VlmWorker(WorkerContext("vlm-test"), model="example-model", api_key_file="not-read-in-test",
                         service="test/rgbd", frames=frames,
                         subscriber_factory=lambda service: calls.append(service) or inbox,
                         publisher_factory=lambda service: calls.append(service) or output,
                         predictor_factory=make)
    return instance, inbox, output, predictor, calls


def send(instance, inbox, value):
    inbox.value = value
    instance.step_once()


def test_lazy_one_model_call_per_request_exact_frame_statuses_and_cleanup(capsys):
    instance, inbox, output, predictor, calls = worker(frames=2)
    instance.on_start()
    try:
        instance.step_once()
        assert predictor.starts == 0 and calls == list(services("test/rgbd"))
        original = request()
        send(instance, inbox, original)
        assert [item.status for item in output.results] == ["running", "done"]
        assert output.results[-1].frame is original.frame
        assert output.results[-1].task_binding is output.results[-1].policy_selection is output.results[-1].coordination is None
        assert predictor.requests == [original]
        assert calls[-1] == {"model": "example-model", "api_key_file": "not-read-in-test"}
        cached = output.results[-1]
        send(instance, inbox, original)
        assert output.results[-1] is cached and len(predictor.requests) == 1
        send(instance, inbox, replace(original, request_id="job-2"))
        assert predictor.starts == 1 and len(predictor.requests) == 2
        assert instance.ctx.stop_event.is_set() and instance.processed_count == 2
    finally:
        instance.on_stop()
    assert predictor.closed and inbox.closed and output.closed and output.refreshes == 4
    assert "not-read-in-test" not in capsys.readouterr().out


@pytest.mark.parametrize("error", [RuntimeError("synthetic-sensitive-message"), VlmError("AuthenticationError")])
def test_errors_only_publish_exception_type_and_next_request_runs(error, capsys):
    instance, inbox, output, predictor, _calls = worker()
    instance.on_start()
    try:
        predictor.error = error
        send(instance, inbox, request())
        expected = "AuthenticationError" if isinstance(error, VlmError) else "RuntimeError"
        assert output.results[-1].status == "error" and output.results[-1].error == expected
        predictor.error = None
        send(instance, inbox, request("next"))
        assert output.results[-1].status == "done" and len(predictor.requests) == 2
    finally:
        instance.on_stop()
    captured = capsys.readouterr()
    assert "synthetic-sensitive-message" not in captured.out + captured.err


def test_start_failure_closes_predictor_and_later_request_can_retry():
    instance, inbox, output, predictor, _calls = worker()
    good_start = predictor.start

    def fail_start():
        raise RuntimeError("synthetic-sensitive-message")

    predictor.start = fail_start
    instance.on_start()
    try:
        send(instance, inbox, request())
        assert predictor.closed and instance.predictor is None
        assert output.results[-1].error == "RuntimeError"
        predictor.start = good_start
        send(instance, inbox, request("second"))
        assert output.results[-1].status == "done"
    finally:
        instance.on_stop()


def test_cli_defaults():
    args = build_parser().parse_args([])
    assert args.model == "gpt-6.1-sol" and args.api_key_file is None and args.frames == 0
    assert args.backend == "openai_api" and args.codex_executable == "codex"


def test_codex_factory_receives_model_and_executable_without_key_path():
    inbox, output, predictor, calls = Inbox(), Output(), Predictor(), []

    def make(**options):
        calls.append(options)
        return predictor

    instance = VlmWorker(WorkerContext("codex-test"), "same-model", "/unused/no-key-read", "test/rgbd", 1,
                         backend="codex_cli", codex_executable="/local/codex",
                         subscriber_factory=lambda _service: inbox, publisher_factory=lambda _service: output,
                         predictor_factory=make)
    instance.on_start()
    try:
        send(instance, inbox, request())
        assert calls == [{"model": "same-model", "executable": "/local/codex"}]
        assert output.results[-1].status == "done"
        assert instance.ctx.stop_event.is_set()
    finally:
        instance.on_stop()
    assert predictor.closed and inbox.closed and output.closed


def test_codex_default_predictor_is_loaded_only_on_first_request(monkeypatch):
    inbox, output, predictor, options = Inbox(), Output(), Predictor(), []
    module = ModuleType("grasp_workbench.models.vlm_codex")
    module.CodexCliPredictor = lambda **kwargs: options.append(kwargs) or predictor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    instance = VlmWorker(WorkerContext("codex-test"), model="configured-model", backend="codex_cli",
                         codex_executable="codex-custom", api_key_file="/unused/no-key-read",
                         subscriber_factory=lambda _service: inbox, publisher_factory=lambda _service: output)
    instance.on_start()
    try:
        instance.step_once()
        assert options == []
        send(instance, inbox, request())
        assert options == [{"model": "configured-model", "executable": "codex-custom"}]
        assert predictor.starts == 1 and output.results[-1].status == "done"
    finally:
        instance.on_stop()


@pytest.mark.parametrize("backend", ["codex_cli", "anthropic_api"])
def test_worker_cli_passes_backend_options_as_keyword_arguments(monkeypatch, backend):
    from grasp_workbench.workers import worker_vlm

    calls = []

    class Worker:
        def __init__(self, *args, **kwargs):
            calls.append((args[1:], kwargs))

        def run(self):
            calls.append("run")

    monkeypatch.setattr(worker_vlm, "VlmWorker", Worker)
    assert worker_vlm.main(["--backend", backend, "--codex-executable", "codex-custom",
                            "--model", "configured-model", "--service", "example/rgbd"]) == 0
    assert calls == [(("configured-model", None, "example/rgbd", 0),
                      {"backend": backend, "codex_executable": "codex-custom"}), "run"]


def test_worker_cli_rejects_unknown_backend():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--backend", "unknown"])
    assert error.value.code == 2


def test_anthropic_dispatch_is_lazy_and_preserves_semantic_result_without_other_sdks(monkeypatch):
    inbox, output, predictor, options = Inbox(), Output(), Predictor(), []
    module = ModuleType("grasp_workbench.models.vlm_anthropic")
    module.AnthropicVlmPredictor = lambda **kwargs: options.append(kwargs) or predictor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in ("openai", "anthropic", "grasp_workbench.models.vlm_codex"):
        monkeypatch.setitem(sys.modules, name, None)
    prediction = SimpleNamespace(
        objects=(VlmObject("cup", "red cup", None, None, "cup-current"),),
        task_binding={"source_observation_id": "cup-current", "target_observation_id": None, "intent": "grasp"},
        policy_selection={"task_type": "grasp", "intent_origin": "explicit", "policy_id": None, "reason": "none"},
        coordination={"arm_mode": "unknown", "assignments": [], "constraints": [], "reason": "unknown"})
    predictor.predict = lambda _value: prediction
    instance = VlmWorker(WorkerContext("anthropic-test"), model="configured-claude", backend="anthropic_api",
                         api_key_file="not-read-in-test", codex_executable="unused-codex",
                         subscriber_factory=lambda _service: inbox, publisher_factory=lambda _service: output)
    instance.on_start()
    try:
        instance.step_once()
        assert options == [] and predictor.starts == 0
        original = request()
        send(instance, inbox, original)
        assert options == [{"model": "configured-claude", "api_key_file": "not-read-in-test"}]
        assert predictor.starts == 1
        value = output.results[-1]
        assert value.status == "done" and value.model == "configured-claude"
        assert value.request_id == original.request_id and value.frame is original.frame
        assert value.objects == prediction.objects and value.task_binding == prediction.task_binding
        assert value.policy_selection == prediction.policy_selection and value.coordination == prediction.coordination
    finally:
        instance.on_stop()


def test_unknown_programmatic_backend_does_not_use_an_api_factory():
    inbox, output, predictor, calls = Inbox(), Output(), Predictor(), []
    instance = VlmWorker(WorkerContext("invalid-backend"), backend="unknown",
                         subscriber_factory=lambda _service: inbox, publisher_factory=lambda _service: output,
                         predictor_factory=lambda **options: calls.append(options) or predictor)
    instance.on_start()
    try:
        send(instance, inbox, request())
        assert calls == [] and predictor.starts == 0
        assert output.results[-1].status == "error" and output.results[-1].error == "ValueError"
    finally:
        instance.on_stop()


def test_prediction_preserves_task_policy_coordination_and_exact_request_identity():
    instance, inbox, output, predictor, _calls = worker()
    objects = (VlmObject("cup", "red cup", None, None, "current-cup"),)
    prediction = SimpleNamespace(
        objects=objects,
        task_binding={"source_observation_id": "current-cup", "target_observation_id": None, "intent": "lift"},
        policy_selection={"task_type": "lift", "intent_origin": "explicit", "policy_id": None, "reason": "no catalog"},
        coordination={"arm_mode": "unknown", "assignments": [], "constraints": [], "reason": "unspecified"})
    predictor.predict = lambda _request: prediction
    original = request()
    instance.on_start()
    try:
        send(instance, inbox, original)
        value = output.results[-1]
        assert value.request_id == original.request_id and value.frame is original.frame
        assert value.objects == objects and value.objects[0].observation_id == "current-cup"
        assert value.task_binding == prediction.task_binding
        assert value.policy_selection == prediction.policy_selection
        assert value.coordination == prediction.coordination
        from grasp_workbench.ipc.vlm import decode_result, encode_result

        decoded = decode_result(encode_result(value))
        assert decoded.coordination == prediction.coordination and decoded.task_binding == prediction.task_binding
        send(instance, inbox, original)
        assert output.results[-1] is value and instance.processed_count == 1
    finally:
        instance.on_stop()


_NATIVE_WORKER = """
import json, sys
for name in ('openai', 'torch', 'PIL'):
    sys.modules[name] = None
from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.vlm import VlmObject
from grasp_workbench.workers.worker_vlm import VlmWorker
class Predictor:
    def __init__(self):
        self.calls = self.starts = self.closes = 0
    def start(self):
        self.starts += 1
    def predict(self, request):
        self.calls += 1
        assert request.frame.metadata == {'sequence':9}
        assert request.assets == ({'asset_id':'can','name':'tuna','prompt':'yellow tuna can'},)
        if request.instruction == 'fail':
            raise RuntimeError('synthetic-sensitive-message')
        return (VlmObject('tuna','yellow tuna can',(0,0,3,2),'can'),)
    def close(self):
        self.closes += 1
predictor = Predictor()
worker = VlmWorker(WorkerContext('fake-vlm'), service=sys.argv[1],
    predictor_factory=lambda **options: predictor)
try:
    worker.run()
finally:
    print(json.dumps({'event':'fake-summary','calls':predictor.calls,'starts':predictor.starts,
        'closes':predictor.closes,'processed':worker.processed_count}),flush=True)
"""


def test_native_worker_subprocess_late_request_error_and_clean_shutdown():
    pytest.importorskip("iceoryx2")
    service = "vlm-process/" + uuid4().hex
    request_service, result_service = services(service)
    publisher, subscriber = VlmRequestPublisher(request_service), VlmResultSubscriber(result_service)
    process, output, error = None, "", ""
    try:
        original = request()
        assert publisher.publish(original) == 0
        root = Path(__file__).resolve().parents[1]
        process = subprocess.Popen([sys.executable, "-B", "-u", "-c", _NATIVE_WORKER, service],
                                   cwd=root, env={**os.environ, "PYTHONPATH": str(root)},
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

        def terminal(request_id):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                assert process.poll() is None
                publisher.update_connections()
                value = subscriber.receive_latest()
                if value is not None and value.request_id == request_id and value.status != "running":
                    return value
                time.sleep(.005)
            pytest.fail("No VLM terminal result over native IPC")

        done = terminal(original.request_id)
        assert done.status == "done" and done.objects[0].known_asset_id == "can"
        np.testing.assert_array_equal(done.frame.rgb, original.frame.rgb)
        publisher.publish(replace(original, request_id="failed", instruction="fail"))
        failed = terminal("failed")
        assert failed.status == "error" and failed.error == "RuntimeError"
        publisher.publish(replace(original, request_id="next"))
        assert terminal("next").status == "done"
    finally:
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                output, error = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                output, error = process.communicate()
                pytest.fail("VLM fake worker failed to shut down")
        subscriber.close()
        publisher.close()
    assert process.returncode == 0, error
    assert "synthetic-sensitive-message" not in output + error
    records = [json.loads(line) for line in output.splitlines()]
    assert records[-1] == {"event": "fake-summary", "calls": 3, "starts": 1, "closes": 1, "processed": 3}
