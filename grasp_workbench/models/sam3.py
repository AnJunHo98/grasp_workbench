"""Official SAM3 image segmentation, not temporal tracking or object identity."""

from pathlib import Path

import numpy as np

from ..ipc.sam3 import Sam3ObjectResult, Sam3Result


def _numpy(value, dtype):
    if hasattr(value, "detach"):
        value = value.detach()
        if dtype == np.float32:
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _object_result(prompt, raw, shape):
    masks = _numpy(raw["masks"], np.bool_)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    boxes = _numpy(raw["boxes"], np.float32).reshape(-1, 4)
    scores = _numpy(raw["scores"], np.float32).reshape(-1)
    if masks.ndim != 3 or masks.shape[1:] != shape:
        raise ValueError("SAM3 masks do not match the original image dimensions")
    if len(masks) != len(boxes) or len(boxes) != len(scores):
        raise ValueError("SAM3 mask, box and score counts differ")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("SAM3 returned nonfinite boxes or scores")
    order = np.argsort(-scores, kind="stable")
    return Sam3ObjectResult(prompt.object_id, prompt.text, masks[order], boxes[order], scores[order])


def _failure(prompt, shape, error):
    return Sam3ObjectResult(
        prompt.object_id, prompt.text, np.empty((0, *shape), dtype=np.bool_),
        np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32),
        error=f"{type(error).__name__}: {error}",
    )


def normalized_box(xyxy, width, height):
    x1, y1, x2, y2 = map(float, xyxy)
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]


def boxes_from_masks(masks):
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for index, mask in enumerate(masks):
        rows, columns = np.nonzero(mask)
        if len(rows):
            boxes[index] = columns.min(), rows.min(), columns.max() + 1, rows.max() + 1
    return boxes


def seed_mask_logits(mask, size):
    from PIL import Image

    image = Image.fromarray(np.asarray(mask, np.uint8) * 255)
    small = np.asarray(image.resize((size[1], size[0]), Image.Resampling.BILINEAR), dtype=np.float32)
    return ((small / 255 - .5) * 20)[None]


class OfficialSam3Predictor:
    def __init__(self, checkpoint, device="cuda", confidence_threshold=0.5):
        self.checkpoint = Path(checkpoint).expanduser()
        self.device, self.confidence_threshold = device, confidence_threshold
        self.model = self.processor = None

    def start(self):
        if self.processor is not None:
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint}")
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint), load_from_HF=False,
            device=self.device, enable_inst_interactivity=True,
        )
        # The pinned builder moves weights only for the literal 'cuda' value.
        model.to(self.device)
        processor = Sam3Processor(model, device=self.device,
                                  confidence_threshold=self.confidence_threshold)
        self.model, self.processor = model, processor

    def predict(self, frame, prompts):
        if not prompts.objects:
            return Sam3Result(frame, prompts.revision, ())
        if self.processor is None:
            raise RuntimeError("Start SAM3 before prediction")
        import torch
        from PIL import Image

        results, shape = [], frame.rgb.shape[:2]
        device_type = self.device.split(":")[0]
        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda",
        ):
            try:
                # PIL preserves H/W; the pinned processor's ndarray path expects CHW.
                state = self.processor.set_image(Image.fromarray(frame.rgb))
            except Exception as error:
                return Sam3Result(frame, prompts.revision, tuple(
                    _failure(prompt, shape, error) for prompt in prompts.objects
                ))
            for prompt in prompts.objects:
                try:
                    self.processor.reset_all_prompts(state)
                    raw = self.processor.set_text_prompt(prompt=prompt.text, state=state)
                    results.append(_object_result(prompt, raw, shape))
                except Exception as error:
                    results.append(_failure(prompt, shape, error))
        return Sam3Result(frame, prompts.revision, tuple(results))

    def refine(self, request):
        """Use the old detector/interactive head on the explicitly frozen selection."""
        import torch
        from PIL import Image

        from ..ipc.sam3 import ObjectPrompt

        if self.processor is None:
            raise RuntimeError("Start SAM3 before refinement")
        frame, mask = request.frame, request.seed_mask
        if not request.prompt.strip() and not request.boxes and not request.points:
            raise ValueError("Provide text, boxes or point prompts")
        if mask is not None and (mask.dtype != np.bool_ or mask.shape != frame.rgb.shape[:2]):
            raise ValueError("Refinement mask must belong to the exact input resolution")
        height, width = frame.rgb.shape[:2]
        prompt = ObjectPrompt(request.object_id, request.prompt)
        device_type = self.device.split(":")[0]
        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda",
        ):
            state = self.processor.set_image(Image.fromarray(frame.rgb))
            grounding = None
            if not request.points or mask is None:
                if request.prompt.strip():
                    grounding = self.processor.set_text_prompt(prompt=request.prompt, state=state)
                for box in request.boxes:
                    grounding = self.processor.add_geometric_prompt(
                        box=normalized_box(box["xyxy"], width, height), label=box.get("label", True), state=state,
                    )
            if not request.points:
                result = _object_result(prompt, grounding, (height, width))
                if not len(result.masks) and len(request.boxes) == 1 and request.boxes[0].get("label", True):
                    masks, scores, _ = self.model.predict_inst(
                        state, box=np.asarray(request.boxes[0]["xyxy"], np.float32),
                        multimask_output=False, normalize_coords=True,
                    )
                    masks, scores = np.asarray(masks), np.asarray(scores)
                    nonempty = masks.any(axis=(-2, -1))
                    masks, scores = masks[nonempty], scores[nonempty]
                    result = _object_result(prompt, {"masks": masks, "boxes": boxes_from_masks(masks), "scores": scores},
                                            (height, width))
            else:
                seed_box = None
                if mask is not None and mask.any():
                    seed_box = boxes_from_masks(mask[None])[0]
                if seed_box is None:
                    seed_box = next((box["xyxy"] for box in request.boxes if box.get("label", True)), None)
                size = self.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size
                masks, scores, _ = self.model.predict_inst(
                    state, point_coords=np.asarray([point["xy"] for point in request.points], np.float32),
                    point_labels=np.asarray([point["label"] for point in request.points], np.int32),
                    box=None if seed_box is None else np.asarray(seed_box, np.float32),
                    mask_input=None if mask is None else seed_mask_logits(mask, size),
                    multimask_output=mask is None, normalize_coords=True,
                )
                masks, scores = np.asarray(masks), np.asarray(scores)
                result = _object_result(prompt, {"masks": masks, "boxes": boxes_from_masks(masks), "scores": scores},
                                        (height, width))
        return Sam3Result(frame, request.request_id, (result,))

    def close(self):
        self.processor = self.model = None
