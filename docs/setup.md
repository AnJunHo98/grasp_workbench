# 설치와 모델별 설정

**카메라·모델·VLM·gripper 환경은 공유받는 사람이 선택하고 준비한다.** 이 문서는 저장소 루트에서
실행하는 설치 명령과 `config.local.yaml` 예시를 모았다. 버튼 사용 순서는 [사용 가이드](usage.md)를 참고한다.
모든 모델을 한 번에 설치하거나 특정 로봇 환경을 필수로 연결하는 설치 프로그램은 아니다.

## 목차

- [기본 환경](#requirements)
- [YAML과 실행 환경](#config)
- [ZED / RealSense](#camera)
- [SAM3](#sam3)
- [VLM: OpenAI / Codex / Anthropic](#vlm)
- [SAM3D](#sam3d)
- [FoundationPose](#foundationpose)
- [GraspGenX](#graspgenx)
- [Grasp 선택과 선택적인 robot adapter](#grasp-selection)
- [다른 VLM adapter 추가](#custom-vlm)

<a id="requirements"></a>
## 1. 기본 환경

Git checkout과 editable 설치(`-e`)를 기준으로 한다. 단독 wheel에는 저장소 루트의 YAML·mesh·submodule이
포함되지 않으므로 독립 wheel 배포는 지원하지 않는다.

```bash
# 저장소 루트에서 실행
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,ui]'
```

`.[ui]`는 PyQt6·Matplotlib 등의 UI 의존성을 설치한다. SDK·모델 checkpoint·전체 CUDA/native 환경은 별도다.
모든 모델 기능을 끄고 `ui.enabled: false`로 카메라 콘솔 monitor만 사용할 때는 `.[test]`만 설치해도 된다.
카메라 SDK는 캡처 프로세스에서 사용하며 UI 프로세스에서는 사용하지 않는다.

| 준비 대상 | 이 repo에서 추가하는 것 | 사용자가 별도로 준비할 것 |
|---|---|---|
| ZED / RealSense | 카메라 연결 예제·전송 | SDK/binding·장치·드라이버·권한 |
| SAM3 / SAM3D / FP / GraspGenX | 얇은 adapter용 extra | 공식 모델 설치·GPU/native 의존성·가중치 |
| VLM | 선택한 SDK 또는 CLI adapter | 모델 접근 권한·API 키 또는 자신의 CLI 로그인 |
| Robot adapter | 선택적 상태/FK·endpoint IK 연결 | state publisher·모델 asset·calibration·solver 환경 |

새 PC에서 clone부터 모든 모델·하드웨어 실행까지 검증한 것은 아니다.
실제 모델·fake/native 테스트와 미검증 항목은 [README의 검증 상태](../README.md)를 확인한다.

<a id="config"></a>
## 2. YAML과 실행 환경

공유 기본값은 [config.yaml](../config.yaml)에 있다. 기기별 설정은 Git에서 제외되는 **저장소 루트의
`config.local.yaml`**에 필요한 항목만 적는다. [공유용 예시](../examples/config.local.example.yaml)는
개인 설정이 아니므로, 경로·모델 ID를 자신의 값으로 바꾼다.

- `config.yaml`을 먼저 읽고 `config.local.yaml`에 적힌 항목만 덮어쓴다.
- 상대 경로는 **그 값을 정의한 YAML 파일의 폴더 기준**이다. 명령을 실행한 현재 폴더 기준이 아니다.
- 각 기능의 `python: null`은 실행 중인 Python을 사용한다. 모델별 별도 환경을 쓰려면 해당 Python 경로를 지정한다.
- `enabled: true`는 worker 시작이다. mesh·pose·grasp·VLM 요청을 즉시 실행한다는 뜻은 아니다.
- 모델 worker는 각각 켠다. 다른 기능을 켰다는 이유로 SAM3·SAM3D·FP 등의 backend를 자동 시작하지 않는다.
  필요한 단계 조합과 수동/Task 흐름은 [사용 가이드](usage.md)에 있다.
- `run`은 카메라를 항상 시작하며 `camera.enabled`는 지원하지 않는다. 모델 기능을 켜면 UI도 사용한다.

설정 후 일반 실행 명령은 하나다.

```bash
grasp-workbench run
```

환경을 activate하지 않았다면 `.venv/bin/grasp-workbench run`을 사용한다.
의존성이 설치된 Python의 `python -m grasp_workbench.main run`도 동일하다.
다른 설정 파일 하나만 쓰려면 다음과 같이 지정한다. 이 경우 기본 공유·개인 파일을 추가로 읽지 않는다.

```bash
grasp-workbench run --config /path/to/settings.yaml
```

<a id="camera"></a>
## 3. ZED / RealSense

### ZED

기기에 맞는 **ZED SDK와 pyzed**를 별도로 설치한다.
[공식 Python API 안내](https://github.com/stereolabs/zed-python-api)를 따른다.
SDK 설치 후 해당 환경에서 실행하는 기존 예시는 다음과 같다.

```bash
python /usr/local/zed/get_python_api.py
```

`third_party/zed-python-api` 소스를 받는 것만으로 SDK가 설치되지는 않는다.

```yaml
# config.local.yaml
camera:
  type: zed
  serial: null            # 특정 장치를 쓰면 자신의 serial 문자열
  resolution: HD720
  fps: 30
```

### RealSense

카메라 캡처에 사용할 Python 환경에서 binding을 추가한다. 장치·드라이버·권한은 별도로 준비한다.

```bash
python -m pip install -e '.[realsense]'
```

```yaml
# config.local.yaml
camera:
  type: realsense
  serial: null
```

장치 목록만 조회하려면 다음 명령을 사용한다. 목록 조회 성공과 실제 RGB-D 캡처 성공은 구분한다.

```bash
grasp-workbench cameras --camera zed
grasp-workbench cameras --camera realsense
```

<a id="sam3"></a>
## 4. SAM3

카메라/UI와 다른 Python 환경에 설치할 수 있다. 공식 `sam3.pt` checkpoint는 사용자가 준비한다.
가중치·접근 토큰을 Git에 포함하지 않으며 앱에서 인증·가중치 다운로드를 자동 수행하지 않는다.
[공식 설치·checkpoint 안내](https://github.com/facebookresearch/sam3#installation)를 따른다.
고정 submodule revision은 `2345a4ad109ac29c569da749c91d84f10dc08c40`이다.

다음 PyTorch 버전은 **기존 기기의 검증 예시**이지 모든 GPU에 대한 설치 권장값이 아니다.
자신의 GPU/CUDA에 맞는 PyTorch·torchvision을 먼저 준비한다.

```bash
python3.12 -m venv .venv-sam3
.venv-sam3/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu130
.venv-sam3/bin/python -m pip install -e '.[sam3]' -e third_party/sam3
```

```yaml
# config.local.yaml
sam3:
  enabled: true
  python: .venv-sam3/bin/python
  checkpoint: /path/to/sam3.pt
  device: cuda
```

UI의 A/B에는 `red cup`, `blue bowl`처럼 서로 다른 영어 prompt를 입력한다. 자동 번역은 하지 않는다.
실제 분할·후보 선택·snapshot mask 보정은 [사용 가이드](usage.md)를 참고한다.

<a id="vlm"></a>
## 5. VLM: OpenAI / Codex / Anthropic

공유 기본값은 `vlm.enabled: false`, `backend: openai_api`, `api_key_file: null`이다.
기존 flat `vlm` section을 사용하며 `providers:` 같은 추가 계층은 없다.

| `vlm.backend` | 연결 방식 | Python 설치 |
|---|---|---|
| `openai_api` | OpenAI Responses API | `.[vlm]` |
| `codex_cli` | 별도 설치한 Codex CLI | 이 repo와 Pillow 또는 기존 `.[vlm]` 환경 |
| `anthropic_api` | Claude Messages API | `.[vlm-anthropic]` |

backend를 바꿔도 `model`은 자동 변경하지 않는다. 각 provider에서 이미지·구조화 출력을 지원하고 자신의
계정에서 사용할 수 있는 실제 모델 ID를 지정한다. 공유 기본 `gpt-6.1-sol`을 Claude 모델로 바꿔주지는 않는다.
모델·인증·네트워크 환경은 사용자가 준비하며 provider 사이 자동 fallback은 없다.

<a id="vlm-openai"></a>
### GPT / OpenAI API

[공식 이미지 입력](https://developers.openai.com/api/docs/guides/images-vision)과
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) 안내를 참고한다.

```bash
python3.12 -m venv .venv-vlm
.venv-vlm/bin/python -m pip install -e '.[vlm]'
```

이미 `.venv-vlm`이 있으면 새로 만들 필요 없이 같은 환경을 사용한다. 이 extra는 OpenAI SDK·Pillow를 설치한다.

```yaml
# config.local.yaml
vlm:
  enabled: true
  python: .venv-vlm/bin/python
  backend: openai_api
  model: gpt-6.1-sol
  api_key_file: .secrets/openai_api_key.txt
  asset_dir: runs/assets
```

`api_key_file: null`이면 worker 환경의 `OPENAI_API_KEY`를 사용한다.

<a id="vlm-codex"></a>
### Codex CLI

Codex CLI는 Python SDK와 별도로 설치한다. [공식 CLI 설치](https://developers.openai.com/codex/cli)와
[인증 안내](https://developers.openai.com/codex/auth)를 따른다. npm이 준비된 경우의 기존 예시는 다음과 같다.

```bash
npm install -g @openai/codex
codex login
```

각 사용자가 자신의 환경에서 로그인한다. Python에는 이미지 저장용 Pillow가 필요하다.
기존 `.[vlm]` 환경을 재사용하거나 OpenAI SDK 없이 다음과 같이 준비할 수 있다.

```bash
python3.12 -m venv .venv-vlm
.venv-vlm/bin/python -m pip install -e . pillow
```

```yaml
# config.local.yaml
vlm:
  enabled: true
  python: .venv-vlm/bin/python
  backend: codex_cli
  codex_executable: codex  # 별도 경로 예: ./tools/codex
  model: gpt-6.1-sol
  asset_dir: runs/assets
```

`codex`는 PATH의 명령이며 경로를 적으면 YAML 파일의 폴더 기준으로 해석한다.
`api_key_file`은 설정되어 있어도 전달·읽지 않는다. `auth.json`·`.codex` 인증자료를 복사·배포하거나 Git에 넣지 않는다.
adapter는 이미지와 공개 catalog를 단발 `codex exec`로 전달한다. 임시 폴더에서 개인 작업 설정·shell·web 도구를
끄고 호출하지만 read-only 옵션을 모든 파일 읽기 차단이라고 해석하지 않는다.
[비대화형 실행](https://developers.openai.com/codex/noninteractive)·[CLI 옵션](https://developers.openai.com/codex/cli/reference)을 참고한다.

<a id="vlm-anthropic"></a>
### Claude / Anthropic API

**Claude API이며 Claude Code CLI의 설치·로그인을 사용하는 backend가 아니다.**
[공식 vision](https://platform.claude.com/docs/en/build-with-claude/vision)과
[Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) 안내를 참고한다.

```bash
python3.12 -m venv .venv-vlm-anthropic
.venv-vlm-anthropic/bin/python -m pip install -e '.[vlm-anthropic]'
```

```yaml
# config.local.yaml — model은 자신의 계정에서 지원되는 실제 ID로 교체
vlm:
  enabled: true
  python: .venv-vlm-anthropic/bin/python
  backend: anthropic_api
  model: YOUR_CLAUDE_VISION_STRUCTURED_OUTPUT_MODEL_ID
  api_key_file: .secrets/anthropic_api_key.txt
  asset_dir: runs/assets
```

`api_key_file: null`이면 worker 환경의 `ANTHROPIC_API_KEY`를 사용한다.
OpenAI 키나 Codex/Claude Code 로그인 자료를 Anthropic API 인증으로 재사용하지 않는다.
fake SDK·native IPC와 실제 SDK의 로컬 mock HTTP 직렬화는 검사했지만 **실제 Claude API 호출은 검증하지 않았다**.

### API 키와 선택적인 로컬 catalog

키 값은 YAML·코드·문서·Git에 적지 않는다. 키 파일은 해당 backend의 VLM worker만 읽으며
UI·iceoryx2 요청·로그에 키 값이나 인증 파일 경로를 넣지 않는다. `.secrets/`와 `config.local.yaml`은 Git 제외 대상이다.

```yaml
# config.local.yaml — 둘 다 선택 사항
vlm:
  policy_catalog: /path/to/policies.json
  session_dir: runs/tasks
```

기본 `policy_catalog: null`은 policy 목록·checkpoint가 없다는 뜻이다. catalog의 checkpoint는 JSON 폴더 기준
로컬 표시·export용 참조일 뿐 로드·실행하지 않는다. provider에는 공개 metadata만 전달한다.
catalog 형식·Task·Session 사용법은 [사용 가이드](usage.md)를 참고한다.

<a id="sam3d"></a>
## 6. SAM3D

[공식 설치 안내](../third_party/sam-3d-objects/doc/setup.md)를 따라 GPU 환경·가중치를 준비한다.
submodule revision은 `f91db411c50efee93d8db7aeb323885650f6f722`다.
공식 환경을 준비한 뒤 **저장소 루트에서 해당 환경의 Python**으로 얇은 extra를 추가한다.

```bash
python -m pip install -e '.[sam3d]'
```

```yaml
# config.local.yaml — SAM3도 별도로 설정·활성화
sam3:
  enabled: true
  python: .venv-sam3/bin/python
  checkpoint: /path/to/sam3.pt
sam3d:
  enabled: true
  python: .venv-sam3d/bin/python
  source_dir: third_party/sam-3d-objects
  config_path: /path/to/checkpoints/pipeline.yaml
  output_dir: runs/meshes
  seed: 42
  depth_source: depth      # depth 또는 monocular
  texture_baking: false
```

- `depth_source: depth`는 실측 depth·intrinsics pointmap을 사용한다. 기본 저장은 vertex-color PLY다.
- `depth_source: monocular`는 MoGe 예측 경로다. `length_unit: monocular_predicted_scale`이며 실측 metre를 보장하지 않는다.
- `texture_baking: true`는 UV/material GLB를 보존한다. 필요한 MoGe·texture backend/가중치는 별도로 준비한다.
- 설정은 수동·VLM·Task 요청에 동일하게 적용하고 UI에서는 읽기 전용으로 표시한다.

이 extra는 공식 CUDA/native 환경을 대신 설치하지 않는다. **실제 MoGe 추론·texture baking·관련 다운로드는
검증하지 않았다.** fake backend·codec·GLB 직렬화/cache 보존 검사와 실제 모델 실행은 구분한다.

<a id="foundationpose"></a>
## 7. FoundationPose

[공식 설치 안내](https://github.com/NVlabs/FoundationPose/tree/a1b694b83e633c2cb6115b9063d940a687759392)를 따라
독립 Python 환경·native extension·`third_party/FoundationPose/weights/`를 준비한다.
[upstream LICENSE](../third_party/FoundationPose/LICENSE)를 유지한다.
공식 환경에서 저장소 루트를 기준으로 adapter 의존성을 추가한다.

```bash
python -m pip install -e '.[foundationpose]'
```

```yaml
# config.local.yaml
sam3:
  enabled: true
  python: .venv-sam3/bin/python
  checkpoint: /path/to/sam3.pt
foundationpose:
  enabled: true
  python: .venv-foundationpose/bin/python
  source_dir: third_party/FoundationPose
  device: cuda:0
  registration_iterations: 5
  tracking_iterations: 1
```

FP는 SAM3 worker를 자동으로 시작하지 않는다. 기존 mesh를 쓰면 SAM3D는 필요 없고,
새 mesh 생성 후 시작하려면 SAM3D도 별도로 설정한다. mesh 파일의 단위·scale은 사용자가 확인한다.
mask 등록·현재/SAM3D seed refinement·tracking 사용법은 [사용 가이드](usage.md)를 참고한다.

<a id="graspgenx"></a>
## 8. GraspGenX

[공식 GraspGenX 설치 안내](https://github.com/NVlabs/GraspGenX)를 따라 GPU 환경을 준비한다.
고정 revision은 `b9429097728cb1c430dd78b92edf17ba318aad03`이다.
upstream이 안내하는 Python **3.10/3.11** 환경을 준비한 뒤 저장소 루트에서 얇은 extra를 추가하는 기존 방식을 사용한다.

```bash
/path/to/graspgenx-env/bin/python -m pip install -e '.[graspgenx]'
```

```yaml
# config.local.yaml
graspgenx:
  enabled: true
  python: /path/to/graspgenx-env/bin/python
  source_dir: third_party/GraspGenX
  checkpoint_dir: /path/to/graspgenx_checkpoints/release
  assets_dir: /path/to/graspgenx_assets
  gripper_id: franka_panda
  output_dir: runs/grasps
  settings: {}
```

이 extra는 PyTorch/CUDA·native 모델 환경 전체를 대신 설치하지 않는다. 공식 패키지는 첫 import 때
checkpoint·gripper description을 다운로드할 수 있다. 가중치·descriptor·경로·네트워크 설정은 사용자가 준비한다.
공식 SDK는 submodule에서 사용하고 upstream 소스는 수정하지 않으며 FFW·ROS 환경을 요구하지 않는다.

mesh-only 후보 생성에는 SAM3·SAM3D·FP·VLM이 필요 없다. 다만 `run`은 카메라를 시작한다.
카메라 없이 저장 mesh만 다루려면 별도 `worker_graspgenx`와 같은 service의 `grasp-workbench ui --graspgenx`를 연결한다.
자세한 요청·bundle/cache 사용은 [사용 가이드](usage.md)에 있다.
기존 재사용 GraspGenX 환경에는 upstream의 torch/torchvision 선언과 불일치가 남아 있었다.
실제 모델 실행 확인과 전체 환경의 의존성 검사 통과는 구분한다([검증 상태](../README.md)).

<a id="grasp-selection"></a>
<a id="robot"></a>
## 9. Grasp 선택과 선택적인 robot adapter

로봇 없이 이미 생성한 bundle에서 canonical 후보를 선택하려면 다음 설정으로 시작한다.
후보 신규 생성이 필요하면 GraspGenX worker도 켠다.

```yaml
# config.local.yaml
grasp_selection:
  enabled: true
  robot_config: null
  settings: {}
```

`robot_config: null`에서는 base 변환·joint seed·TCP 보정·IK 성공을 임의로 만들지 않는다.
기존 예제의 `settings: {}` 값은 `top_k: 32`, `pregrasp_distance_m: 0.15`,
`grasp_retreat_distance_m: 0.0`, `grasp_advance_distance_m: 0.0`, `preference: top_down`이다.
모든 로봇/gripper에 맞는 보편 기본값이 아니므로 자신의 모델·TCP에 맞춰 설정한다.
이전 점수 경로는 `settings: {preference: legacy}`로 선택할 수 있다.

### 일반 robot adapter

```yaml
# config.local.yaml
grasp_selection:
  enabled: true
  python: /path/to/robot-env/bin/python
  robot_config: /path/to/robot.yaml
  settings: {}
```

로컬 robot YAML의 `backend: module:factory`를 지정하면 `factory(config, config_dir=...)`로 호출한다.
사용자가 자신의 상태 수신·현재 FK·calibration·endpoint IK 연결을 제공한다.
이 기능은 **로봇 기준 좌표 변환과 pregrasp/grasp endpoint 도달 가능성 확인**이며 로봇 이동·경로 계획이 아니다.

### 선택적인 FFW 예제

[`examples/robot_ffw.yaml`](../examples/robot_ffw.yaml)은 `backend: ffw_sg2`의 선택적 예제다.
FFW 앱 저장소를 import하거나 필수로 설치하지 않는다. 이 예제를 쓸 때만 별도 환경에 독립 패키지
`robot-worker-runtime[iceoryx2]`, `ffw-robot-common`, `motion-planning-system[curobo]`와 해당 GPU 환경을 준비한다.
runtime manifest·robot model assets·calibration 파일은 사용자가 제공한다. 예제의 `/path/to/`를 자신의 값으로 바꾸며
YAML 안의 상대 경로는 그 YAML의 폴더 기준이다.

calibration YAML은 `parent_frame`과 `transforms.<sensor_frame>.parent_from_sensor`를 사용한다.
`base_link`는 고정 base calibration이고 `head_link2`는 실측 head·lift 관절의 현재 FK와 합성한다.
**실제 로봇 state 연결은 검증하지 않았다.** offline GPU endpoint IK 확인을 실측 state/FK 검증으로 해석하지 않는다.
driver·TrajOpt·world scene·trajectory·actuator/이동 명령을 시작하지 않는다.

<a id="custom-vlm"></a>
## 10. 다른 VLM adapter 추가

현재 backend는 `openai_api`, `codex_cli`, `anthropic_api`다. 다른 provider는 YAML에 이름만 적어서 사용할 수 없다.

1. `models/`에 `start()`·`predict(request)`·`close()` adapter를 구현한다.
   `predict`는 공통 `VlmPrediction(objects, task_binding, policy_selection, coordination)`으로 정규화한다.
2. `config.py`, `main.py`의 `--vlm-backend`, VLM worker CLI 선택지와 predictor 생성 분기에 등록한다.
   필요 SDK는 선택 extra로 추가하고 adapter에서만 import한다. 기존 flat `model`·`api_key_file` 설정과 공통 검증을 재사용한다.
3. 기존 VLM worker·`VlmRequest`/`VlmResult`·iceoryx2·UI를 유지한다. 키는 worker 내부에서만 읽으며
   키 값과 SDK 원문 오류를 IPC·로그에 넣지 않는다.

별도 실행 환경의 외부 프로세스가 필요한 경우에만 기존 `/vlm/requests`·`/vlm/results` 계약의 worker를 연결한다.
provider를 바꾼다는 이유만으로 새 framework·`providers:` 설정 계층·기능별 새 worker를 추가하지 않는다.
