"""YAML → real VLM CLI/iceoryx2 → Claude adapter; only the SDK is a stub."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest
import yaml

from grasp_workbench.config import load_run_config
from grasp_workbench.ipc.vlm import VlmRequestPublisher, VlmResultSubscriber, services
from grasp_workbench.main import build_commands, build_parser
from test_semantic import coordination, semantic_payload
from test_vlm_model import object_record, request

ROOT = Path(__file__).resolve().parents[1]
_SDK = """
import base64, json, os, struct, sys
from pathlib import Path
from types import SimpleNamespace
for name in ('openai','torch','pyzed','pyrealsense2'):
    sys.modules[name] = None

class Anthropic:
    def __init__(self, **options):
        assert options['max_retries'] == 0
        key = options.get('api_key', os.environ.get('ANTHROPIC_API_KEY'))
        assert key == 'synthetic-credential-not-for-network'
        self.messages = self
    def create(self, **options):
        assert options['model'] == 'configured-claude-model'
        assert options['output_config']['format']['type'] == 'json_schema'
        content = options['messages'][0]['content']
        image = next(item['source'] for item in content if item['type'] == 'image')
        assert image['type'] == 'base64' and image['media_type'] == 'image/png'
        png = base64.b64decode(image['data'])
        assert png[:8] == b'\\x89PNG\\r\\n\\x1a\\n' and struct.unpack('>II', png[16:24]) == (4,3)
        text = json.loads(next(item['text'] for item in content if item['type'] == 'text'))
        assert text['instruction'] == '테이프와 참치캔'
        assert text['assets'] == [{'asset_id':'saved-tuna','name':'참치캔','prompt':'yellow tuna can'}]
        assert 'private-camera-context' not in json.dumps(options) and '/local/mesh' not in json.dumps(options)
        print(json.dumps({'event':'sdk-request','model':options['model']}),flush=True)
        payload = Path(os.environ['FAKE_CLAUDE_PAYLOAD']).read_text()
        return SimpleNamespace(stop_reason='end_turn', content=[SimpleNamespace(type='text',text=payload)])
    def close(self):
        print(json.dumps({'event':'sdk-closed'}),flush=True)
"""


@pytest.mark.parametrize("key_file", [True, False])
def test_yaml_backend_and_auth_round_trip_preserve_full_semantics_and_source(tmp_path, key_file):
    pytest.importorskip("iceoryx2")
    pytest.importorskip("PIL")
    (tmp_path / "anthropic.py").write_text(_SDK)
    key = tmp_path / "test-key.txt"
    key.write_text("synthetic-credential-not-for-network\n")
    service = "vlm-anthropic-native/" + uuid4().hex
    settings = tmp_path / "config.yaml"
    settings.write_text(yaml.safe_dump({"ipc": {"service": service}, "vlm": {
        "enabled": True, "python": sys.executable, "backend": "anthropic_api",
        "model": "configured-claude-model", "api_key_file": key.name if key_file else None}}))
    args = build_parser(load_run_config(settings)).parse_args(["run"])
    commands = build_commands(args)
    command = next(item for item in commands if "grasp_workbench.workers.worker_vlm" in item)
    assert len(commands) == 3  # UI, chosen VLM, camera; no other model is enabled.
    value = request()
    value.frame.depth_m[0, 0] = np.nan
    payload = semantic_payload([
        dict(object_record(), label="받침", segmentation_prompt="yellow pad", bbox_xyxy=[0, 0, 1, 2]),
        dict(object_record(), bbox_xyxy=[2, 0, 4, 3]),
    ])
    payload["task_binding"].update(source_observation_id="obs_2", target_observation_id="obs_1")
    payload["coordination"] = coordination()
    for assignment in payload["coordination"]["assignments"]:
        assignment["observation_id"] = "obs_2"
    output = tmp_path / "response.json"
    output.write_text(json.dumps(payload, ensure_ascii=False))
    environment = {name: value for name, value in os.environ.items()
                   if name not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}}
    environment.update(PYTHONPATH=os.pathsep.join((str(tmp_path), str(ROOT))),
                       FAKE_CLAUDE_PAYLOAD=str(output))
    if not key_file:
        environment["ANTHROPIC_API_KEY"] = "synthetic-credential-not-for-network"
    publisher = VlmRequestPublisher(services(service)[0])
    subscriber = VlmResultSubscriber(services(service)[1])
    log = tmp_path / "worker.log"
    worker, result = None, None
    try:
        with log.open("w") as stream:
            worker = subprocess.Popen(command, cwd=ROOT, env=environment,
                                      stdout=stream, stderr=subprocess.STDOUT)
            deadline, sent = time.monotonic() + 10, False
            while time.monotonic() < deadline:
                publisher.update_connections()
                if not sent and '"event": "ready"' in log.read_text():
                    publisher.publish(value)
                    sent = True
                incoming = subscriber.receive_latest()
                if incoming is not None and incoming.status in {"done", "error"}:
                    result = incoming
                    break
                assert worker.poll() is None, log.read_text()
                time.sleep(.01)
            assert result is not None, log.read_text()
            assert result.status == "done", result.error
            assert result.model == "configured-claude-model" and result.request_id == value.request_id
            assert result.task_binding == payload["task_binding"] and result.coordination == payload["coordination"]
            assert result.policy_selection == payload["policy_selection"]
            assert [item.observation_id for item in result.objects] == ["obs_1", "obs_2"]
            assert result.frame.metadata == value.frame.metadata
            np.testing.assert_array_equal(result.frame.rgb, value.frame.rgb)
            np.testing.assert_array_equal(result.frame.depth_m, value.frame.depth_m)
    finally:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        subscriber.close()
        publisher.close()
    assert worker.returncode == 0, log.read_text()
    records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    assert sum(row.get("event") == "sdk-request" for row in records) == 1
    assert sum(row.get("event") == "sdk-closed" for row in records) == 1
    assert "synthetic-credential-not-for-network" not in log.read_text()
