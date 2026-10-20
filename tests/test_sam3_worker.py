"""Independent SAM3 adapter/worker contracts with fake models and local endpoints."""

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from grasp_workbench.core.worker_base import WorkerContext
from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.sam3 import ObjectPrompt, PromptSet, Sam3Result, services
from grasp_workbench.models.sam3 import OfficialSam3Predictor
from grasp_workbench.workers.worker_sam3 import Sam3Worker, build_parser


def frame(sequence=0):
    return CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
                       np.full((3, 4), .75, dtype=np.float32),
                       {"camera_id": "fake-camera", "stream_id": "session", "sequence": sequence})


def prompts(revision="first", texts=("red cup", "blue bowl")):
    return PromptSet(revision, tuple(ObjectPrompt(chr(65 + i), text) for i, text in enumerate(texts)))


@pytest.fixture
def official_fakes(tmp_path, monkeypatch):
    calls = SimpleNamespace(build=[], processors=[], to=[], autocast=[])

    class Processor:
        def __init__(self, model, **options):
            self.model, self.options = model, options
            self.images, self.resets, self.texts, self.outputs = [], [], [], []
            self.fail_image = False
            calls.processors.append(self)

        def set_image(self, image):
            self.images.append(np.asarray(image).copy())
            assert image.size == (4, 3)
            if self.fail_image:
                raise RuntimeError("image encoding failed")
            return {"backbone_out": {"image_features": object()}}

        def reset_all_prompts(self, state):
            self.resets.append(state["backbone_out"]["image_features"])
            state.pop("previous_text", None)

        def set_text_prompt(self, prompt, state):
            assert "previous_text" not in state, "Object prompts must not leak into each other"
            state["previous_text"] = prompt
            self.texts.append(prompt)
            if prompt == "fail":
                raise RuntimeError("one object failed")
            masks = np.zeros((3, 1, 3, 4), dtype=np.bool_)
            masks[:, 0, 0, 0] = True
            masks[1, 0, 1, 1] = True
            boxes = np.arange(12, dtype=np.float32).reshape(3, 4)
            scores = np.array([.2, .9, .9], dtype=np.float32)
            if prompt == "malformed":
                masks = masks[:, :, :2]
            output = {"masks": masks, "boxes": boxes, "scores": scores}
            self.outputs.append(output)
            return output

    def build(**options):
        calls.build.append(options)
        return SimpleNamespace(to=lambda device: calls.to.append(device))

    def autocast(**options):
        calls.autocast.append(options)
        return nullcontext()

    for name in ("sam3", "sam3.model"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    builder = ModuleType("sam3.model_builder")
    builder.build_sam3_image_model = build
    processor = ModuleType("sam3.model.sam3_image_processor")
    processor.Sam3Processor = Processor
    monkeypatch.setitem(sys.modules, builder.__name__, builder)
    monkeypatch.setitem(sys.modules, processor.__name__, processor)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext, autocast=autocast, bfloat16="bfloat16"))
    checkpoint = tmp_path / "external-model.pt"
    checkpoint.write_bytes(b"fake checkpoint; no weights are loaded")
    return checkpoint, calls


def test_one_model_one_image_encoding_independent_prompts_and_all_sorted_candidates(official_fakes):
    checkpoint, calls = official_fakes
    predictor = OfficialSam3Predictor(checkpoint, device="cuda:2", confidence_threshold=.4)
    predictor.start()
    predictor.start()
    original = frame()
    result = predictor.predict(original, prompts())
    processor = calls.processors[0]
    assert len(calls.build) == len(calls.processors) == 1
    assert calls.build[0] == {"checkpoint_path": str(checkpoint), "load_from_HF": False,
                              "device": "cuda:2", "enable_inst_interactivity": True}
    assert calls.to == ["cuda:2"]
    assert processor.options == {"device": "cuda:2", "confidence_threshold": .4}
    assert len(processor.images) == 1 and len(processor.resets) == 2
    assert processor.resets[0] is processor.resets[1]
    assert processor.texts == ["red cup", "blue bowl"]
    np.testing.assert_array_equal(processor.images[0], original.rgb)
    assert result.frame is original and result.revision == "first"
    for item in result.objects:
        assert item.error is None and item.masks.shape == (3, 3, 4)
        assert item.masks.dtype == np.bool_
        np.testing.assert_array_equal(item.scores, np.array([.9, .9, .2], np.float32))
        np.testing.assert_array_equal(item.boxes, np.arange(12, dtype=np.float32).reshape(3, 4)[[1, 2, 0]])
    processor.outputs[0]["masks"][:] = False
    processor.outputs[0]["boxes"][:] = 0
    assert result.objects[0].masks.any() and result.objects[0].boxes.any()
    predictor.predict(frame(1), prompts())
    assert len(processor.images) == 2 and len(calls.build) == 1
    predictor.close()
    assert predictor.model is predictor.processor is None


@pytest.mark.parametrize("failure", ["fail", "malformed"])
def test_one_object_error_does_not_disable_the_other_prompt(official_fakes, failure):
    checkpoint, calls = official_fakes
    predictor = OfficialSam3Predictor(checkpoint, device="cpu")
    predictor.start()
    result = predictor.predict(frame(), prompts(texts=(failure, "blue bowl")))
    bad, good = result.objects
    assert bad.error and bad.masks.shape == (0, 3, 4)
    assert bad.boxes.shape == (0, 4) and bad.scores.shape == (0,)
    assert good.error is None and len(good.scores) == 3
    assert len(calls.processors[0].images) == 1 and len(calls.processors[0].resets) == 2
    assert calls.autocast[-1]["enabled"] is False


def test_shared_encoder_failure_reports_errors_without_reencoding(official_fakes):
    checkpoint, calls = official_fakes
    predictor = OfficialSam3Predictor(checkpoint)
    predictor.start()
    calls.processors[0].fail_image = True
    result = predictor.predict(frame(), prompts())
    assert len(calls.processors[0].images) == 1 and not calls.processors[0].texts
    assert all("image encoding failed" in item.error for item in result.objects)


def test_empty_prompt_has_no_model_dependency_and_missing_checkpoint_is_explicit(tmp_path):
    predictor = OfficialSam3Predictor(tmp_path / "missing.pt")
    assert predictor.predict(frame(), prompts(texts=())).objects == ()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        predictor.start()
    with pytest.raises(RuntimeError, match="Start SAM3"):
        predictor.predict(frame(), prompts())


class Mailbox:
    def __init__(self):
        self.value, self.closed = None, False

    def receive_latest(self):
        value, self.value = self.value, None
        return value

    def close(self):
        self.closed = True


class Publisher:
    def __init__(self):
        self.results, self.closed = [], False

    def publish(self, result):
        self.results.append(result)
        return 1

    def close(self):
        self.closed = True


class Predictor:
    def __init__(self):
        self.starts, self.calls, self.closed = 0, [], False
        self.during_predict = lambda: None

    def start(self):
        self.starts += 1

    def predict(self, value, command):
        self.calls.append((value, command))
        self.during_predict()
        return Sam3Result(value, command.revision, ())

    def close(self):
        self.closed = True


def worker(frames=0):
    camera, commands, output, predictor = Mailbox(), Mailbox(), Publisher(), Predictor()
    factories = []
    instance = Sam3Worker(
        WorkerContext("sam3-test"), "unused.pt", "camera-service", frames=frames,
        camera_subscriber_factory=lambda service: factories.append(service) or camera,
        prompt_subscriber_factory=lambda service: factories.append(service) or commands,
        publisher_factory=lambda service: factories.append(service) or output,
        predictor_factory=lambda *args, **kwargs: predictor,
        refine_subscriber_factory=lambda service: Mailbox(),
        refine_publisher_factory=lambda service: Publisher(),
    )
    return instance, camera, commands, output, predictor, factories


def test_worker_lazy_load_dedup_latest_frame_and_prompt_revision():
    instance, camera, commands, output, predictor, factories = worker()
    instance.on_start()
    try:
        assert factories == ["camera-service", *services("camera-service")]
        camera.value = frame(0)
        instance.step_once()
        assert predictor.starts == 0 and not predictor.calls
        commands.value = prompts()
        instance.step_once()
        assert predictor.starts == 1 and not predictor.calls
        camera.value = frame(1)
        instance.step_once()
        assert output.results[-1].frame.metadata["sequence"] == 1
        instance.step_once()
        assert len(predictor.calls) == 1
        camera.value = frame(1)
        instance.step_once()
        assert len(predictor.calls) == 1
        commands.value = prompts("changed", ("green bottle",))
        instance.step_once()
        assert len(predictor.calls) == 2 and output.results[-1].revision == "changed"
        commands.value = prompts("stopped", ())
        camera.value = frame(2)
        instance.step_once()
        assert len(predictor.calls) == 2 and predictor.starts == 1
        commands.value = prompts("resumed")
        instance.step_once()
        assert output.results[-1].frame.metadata["sequence"] == 2 and predictor.starts == 1
    finally:
        instance.on_stop()
    assert camera.closed and commands.closed and output.closed and predictor.closed


@pytest.mark.parametrize("cancel", [False, True])
def test_changed_revision_or_cancellation_during_inference_discards_old_result(cancel):
    instance, camera, commands, output, predictor, _ = worker()
    instance.on_start()
    try:
        camera.value, commands.value = frame(), prompts()
        instance.step_once()
        predictor.during_predict = (instance.ctx.stop if cancel else
                                    lambda: setattr(commands, "value", prompts("stopped", ())))
        instance.step_once()
        assert not output.results and instance.processed_count == 0
        if not cancel:
            instance.step_once()
            assert len(predictor.calls) == 1
    finally:
        instance.on_stop()


def test_finite_results_and_start_failure_cleanup():
    instance, camera, commands, output, predictor, _ = worker(frames=1)
    camera.value, commands.value = frame(), prompts()
    instance.run()
    assert instance.processed_count == 1 and instance.ctx.stop_event.is_set()
    assert all(value.closed for value in (camera, commands, output, predictor))
    instance, camera, commands, output, predictor, _ = worker(frames=1)
    camera.value, commands.value = frame(), prompts()

    def fail():
        raise RuntimeError("load failed")

    predictor.start = fail
    with pytest.raises(RuntimeError, match="load failed"):
        instance.run()
    assert all(value.closed for value in (camera, commands, output, predictor))


def test_imports_are_heavyweight_lazy_and_checkpoint_is_required():
    script = """
import sys
for name in ('torch', 'sam3', 'PIL', 'pyzed', 'pyrealsense2'):
    sys.modules[name] = None
from grasp_workbench.models.sam3 import OfficialSam3Predictor
from grasp_workbench.workers.worker_sam3 import Sam3Worker
OfficialSam3Predictor('external.pt')
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", script], cwd=root,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    assert build_parser().parse_args(["--checkpoint", "external.pt"]).device == "cuda"
