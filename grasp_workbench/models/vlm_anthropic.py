"""Optional Anthropic Messages boundary sharing the workbench's VLM semantics."""
import base64
import io
import json
from pathlib import Path

from ..ipc.vlm import public_assets
from ..semantic import public_policy_catalog
from .vlm import VlmError, _INSTRUCTIONS, _parse_prediction, _schema


def _anthropic_schema(asset_ids, policies):
    schema = _schema(asset_ids, policies)
    # Claude does not support maxItems or minItems > 1. The shared parser still
    # enforces both limits after generation; nullable references stay unchanged.
    objects = schema["properties"]["objects"]
    objects.pop("maxItems")
    objects["description"] = "At most two distinct visible objects."
    bbox = objects["items"]["properties"]["bbox_xyxy"]["anyOf"][0]
    bbox.pop("minItems")
    bbox.pop("maxItems")
    bbox["description"] = "Exactly four original-image pixel coordinates [left, top, right, bottom]."
    return schema


class AnthropicVlmPredictor:
    def __init__(self, model, api_key_file=None, *, client_factory=None):
        self.model, self.api_key_file = model, api_key_file
        self._client_factory, self._client = client_factory, None

    def start(self):
        if self._client is not None:
            return
        try:
            factory = self._client_factory
            if factory is None:
                from anthropic import Anthropic

                factory = Anthropic
            options = {"max_retries": 0}
            if self.api_key_file is not None:
                options["api_key"] = Path(self.api_key_file).read_text(encoding="utf-8").strip()
            self._client = factory(**options)
        except Exception as error:
            raise VlmError(type(error).__name__) from None

    def predict(self, request):
        if self._client is None:
            raise RuntimeError("Start VLM before prediction")
        try:
            from PIL import Image

            assets = public_assets(request.assets)
            policies = public_policy_catalog(request.policies)
            asset_ids = list(dict.fromkeys(asset["asset_id"] for asset in assets))
            height, width = request.frame.rgb.shape[:2]
            buffer = io.BytesIO()
            Image.fromarray(request.frame.rgb).save(buffer, format="PNG")
            response = self._client.messages.create(
                model=self.model, max_tokens=4096, system=_INSTRUCTIONS,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                               "data": base64.b64encode(buffer.getvalue()).decode("ascii")}},
                    {"type": "text", "text": json.dumps({
                        "instruction": request.instruction, "image_size": {"width": width, "height": height},
                        "assets": assets, "policies": policies,
                    }, ensure_ascii=False)},
                ]}],
                output_config={"format": {"type": "json_schema", "schema": _anthropic_schema(asset_ids, policies)}},
            )
            if response.stop_reason != "end_turn":
                raise ValueError("VLM response did not complete")
            text = "".join(block.text for block in response.content if block.type == "text")
            return _parse_prediction(text, request.frame, asset_ids, policies)
        except Exception as error:
            raise VlmError(type(error).__name__) from None

    def close(self):
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as error:
                raise VlmError(type(error).__name__) from None
