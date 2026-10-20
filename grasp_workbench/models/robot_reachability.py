"""Optional read-only robot plugins; ordinary perception imports no robot stack."""
import importlib
from pathlib import Path

import yaml


def load_robot_adapter(path):
    """Load a builtin example or ``module:factory(config, config_dir=...)``."""
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    backend = config["backend"]
    if backend == "ffw_sg2":
        from .robot_ffw import FfwSg2RobotAdapter

        factory = FfwSg2RobotAdapter
    else:
        module, name = backend.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
    return factory(config, config_dir=path.parent)
