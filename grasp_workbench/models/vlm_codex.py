"""One-shot Codex CLI semantic adapter using the CLI's existing login."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from ..ipc.vlm import public_assets
from ..semantic import public_policy_catalog
from .vlm import DEFAULT_MODEL, VlmError, _INSTRUCTIONS, _parse_prediction, _schema


class CodexCliPredictor:
    def __init__(self, model=DEFAULT_MODEL, executable="codex", *, runner=subprocess.run):
        if "/" in executable or "\\" in executable or executable.startswith((".", "~")):
            executable = str(Path(executable).expanduser().absolute())
        self.model, self.executable, self._runner = model, executable, runner

    def start(self):
        pass

    def predict(self, request):
        try:
            from PIL import Image

            assets = public_assets(request.assets)
            policies = public_policy_catalog(request.policies)
            asset_ids = list(dict.fromkeys(asset["asset_id"] for asset in assets))
            height, width = request.frame.rgb.shape[:2]
            prompt = _INSTRUCTIONS + " Use only the attached image and supplied catalog. Do not use tools.\n" + json.dumps({
                "instruction": request.instruction, "image_size": {"width": width, "height": height},
                "assets": assets, "policies": policies,
            }, ensure_ascii=False)
            with TemporaryDirectory(prefix="grasp-workbench-vlm-") as directory:
                directory = Path(directory)
                image_path, schema_path, result_path = (directory / name for name in ("image.png", "schema.json", "result.json"))
                Image.fromarray(request.frame.rgb).save(image_path, format="PNG")
                schema_path.write_text(json.dumps(_schema(asset_ids, policies)), encoding="utf-8")
                command = [self.executable, "exec", "--image", str(image_path), "--output-schema", str(schema_path),
                           "--output-last-message", str(result_path), "--sandbox", "read-only", "--ephemeral",
                           "--skip-git-repo-check", "--ignore-user-config", "--disable", "shell_tool",
                           "-c", 'web_search="disabled"', "-m", self.model, "-"]
                completed = self._runner(command, input=prompt, cwd=str(directory), text=True, capture_output=True)
                if completed.returncode != 0:
                    raise VlmError("CodexProcessError")
                return _parse_prediction(result_path.read_text(encoding="utf-8"), request.frame, asset_ids, policies)
        except VlmError:
            raise
        except Exception as error:
            raise VlmError(type(error).__name__) from None

    def close(self):
        pass
