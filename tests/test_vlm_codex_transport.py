"""Native VLM worker CLI with an executable Codex stub; no login or API calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.vlm import VlmRequest, VlmRequestPublisher, VlmResultSubscriber, services

ROOT = Path(__file__).resolve().parents[1]
_WORKER = """
import sys
for name in ('openai', 'torch', 'pyzed', 'pyrealsense2'):
    sys.modules[name] = None
from grasp_workbench.workers.worker_vlm import main
raise SystemExit(main())
"""
_CODEX = r"""
import json, struct, sys
from pathlib import Path

args = sys.argv[1:]
assert args[0] == 'exec'
prompt = sys.stdin.read()
assert '테이프와 공' in prompt and 'known-tape' in prompt
image = Path(args[args.index('--image') + 1]).read_bytes()
assert image[:8] == b'\x89PNG\r\n\x1a\n'
assert struct.unpack('>II',image[16:24]) == (12,8)
schema = json.loads(Path(args[args.index('--output-schema') + 1]).read_text())
assert schema['type'] == 'object' and 'objects' in schema['properties']
output = Path(args[args.index('--output-last-message') + 1])
output.write_text(json.dumps({'objects':[
    {'observation_id':'obs-tape','label':'테이프','segmentation_prompt':'brown tape roll','bbox_xyxy':[0,0,4,4],
     'known_asset_id':'known-tape'},
    {'observation_id':'obs-ball','label':'공','segmentation_prompt':'orange ball','bbox_xyxy':[6,2,12,8],
     'known_asset_id':None}
], 'task_binding':{'source_observation_id':'obs-tape','target_observation_id':None,'intent':'테이프 잡기'},
   'policy_selection':{'task_type':'grasp','intent_origin':'explicit','policy_id':None,'reason':'정책 없음'},
   'coordination':{'arm_mode':'single','assignments':[
       {'arm':'left','observation_id':'obs-tape','role':'잡기','grasp_region':'테두리'}],
       'constraints':[],'reason':'한팔 지시'}
},ensure_ascii=False),encoding='utf-8')
"""


def test_worker_codex_cli_round_trip_with_executable_stub(tmp_path):
    pytest.importorskip("iceoryx2")
    pytest.importorskip("PIL")
    executable = tmp_path / "fakecodex"
    executable.write_text(f"#!{sys.executable}\n" + _CODEX, encoding="utf-8")
    executable.chmod(0o755)
    service = f"vlm-codex-cli/{uuid4().hex}"
    request_service, result_service = services(service)
    publisher = VlmRequestPublisher(request_service)
    subscriber = VlmResultSubscriber(result_service)
    frame = CameraFrame(np.arange(288, dtype=np.uint8).reshape(8, 12, 3),
                        np.full((8, 12), .75, np.float32), {
                            "camera_id": "frozen-example", "sequence": 37, "capture_time_ns": 12345,
                            "intrinsics": {"fx": 20., "fy": 20., "cx": 6., "cy": 4.},
                        })
    frame.depth_m[0, 0] = np.nan
    request = VlmRequest("codex-cli-request", frame, "테이프와 공", (
        {"asset_id": "known-tape", "name": "테이프", "prompt": "brown tape roll"},))
    worker = None
    log = tmp_path / "worker.log"
    result = None
    try:
        with log.open("w") as stream:
            worker = subprocess.Popen(
                [sys.executable, "-B", "-u", "-c", _WORKER, "--backend", "codex_cli",
                 "--codex-executable", str(executable), "--model", "local-cli-test-model",
                 "--service", service, "--frames", "1"], cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=stream, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 10
            sent = False
            while time.monotonic() < deadline:
                publisher.update_connections()
                if not sent and '"event": "ready"' in log.read_text():
                    publisher.publish(request)
                    sent = True
                incoming = subscriber.receive_latest()
                if incoming is not None and incoming.status in {"done", "error"}:
                    result = incoming
                    break
                if worker.poll() not in (None, 0):
                    pytest.fail(f"Worker exited before its terminal result:\n{log.read_text()}")
                time.sleep(.01)
            assert result is not None, log.read_text()
            assert worker.wait(timeout=5) == 0, log.read_text()
        assert result.status == "done", result.error
        assert result.task_binding["source_observation_id"] == "obs-tape"
        assert result.coordination["arm_mode"] == "single"
        assert result.coordination["assignments"][0]["arm"] == "left"
        assert result.policy_selection["policy_id"] is None
        assert result.request_id == request.request_id and result.model == "local-cli-test-model"
        assert result.frame.metadata == frame.metadata
        np.testing.assert_array_equal(result.frame.rgb, frame.rgb)
        np.testing.assert_array_equal(result.frame.depth_m, frame.depth_m)
        assert [(item.label, item.segmentation_prompt, item.bbox_xyxy, item.known_asset_id)
                for item in result.objects] == [
            ("테이프", "brown tape roll", (0., 0., 4., 4.), "known-tape"),
            ("공", "orange ball", (6., 2., 12., 8.), None),
        ]
        records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
        assert sum(item.get("event") == "vlm-result" for item in records) == 1
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        subscriber.close()
        publisher.close()
