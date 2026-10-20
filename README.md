# Grasp Workbench

**RGB-D 카메라에서 물체의 mesh·pose·grasp 후보까지 연결하는 선택형 worker 예제입니다.**

카메라·VLM·모델·gripper 환경은 사용하는 사람이 준비합니다. 이 저장소는 각 기능을 독립 프로세스와
**iceoryx2**로 연결하고, 필요한 단계만 YAML로 켜서 사용하는 최소 통합 예제를 제공합니다.
FFW·ROS는 필수가 아니며, **실제 로봇 이동·관절 명령은 수행하지 않습니다.**

![입력 장면과 물체 mesh·grasp 시각화 예시](image/asset.png)

## 목차

- [빠른 시작](#quick-start)
- [어디까지 사용할지 선택](#stages)
- [YAML 설정](#configuration)
- [VLM 선택](#vlm)
- [UI 사용 흐름](#workflow)
- [문서와 저장소 구조](#documentation)
- [검증 범위](#verification)
- [공유 시 주의사항](#sharing)

## 주요 기능

| 기능 | 역할 |
|---|---|
| Camera | ZED·RealSense RGB-D 연결 예제 |
| VLM · 선택 | GPT API / Codex CLI / Claude API로 물체·영어 prompt·작업 역할 인식 |
| SAM3 | A/B 두 prompt의 연속 분할, 후보 선택, 박스·포인트 보정 |
| SAM3D · 선택 | 고정한 RGB-D·mask에서 mesh 생성. 기존 mesh가 있으면 생략 가능 |
| FoundationPose | mesh 등록·연속 pose 추적, 영상 overlay·3D mesh 표시 |
| GraspGenX | mesh 기준 grasp 후보 생성·저장·재사용 |
| Grasp selection · 선택 | 한팔/양팔 후보 선택, 로봇 연결 시 base 좌표 변환·endpoint IK 확인 |

**A/B는 물체 슬롯이지 좌/우 팔이 아닙니다.** VLM은 의미 정보를 만들며 mask·수치 pose·실제 접촉 상태를 추정하지 않습니다.

<a id="quick-start"></a>
## 1. 빠른 시작

### 저장소와 기본 UI 환경 설치

현재는 **Git checkout + editable 설치(`-e`)**를 기준으로 합니다. Private 저장소 접근 권한이 필요합니다.

```bash
git clone --recurse-submodules https://github.com/AnJunHo98/grasp_workbench.git
cd grasp_workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ui,test]'
```

이미 clone했다면 `git submodule update --init --recursive`로 submodule을 받습니다.
루트 YAML·mesh·submodule을 포함하는 독립 wheel 배포는 지원하지 않습니다.

### 카메라 준비

| 카메라 | 추가 준비 |
|---|---|
| ZED · 기본값 | 해당 기기의 ZED SDK와 현재 Python 환경의 `pyzed` |
| RealSense | `python -m pip install -e '.[realsense]'`, 장치·드라이버·접근 권한 |

submodule을 받는 것만으로 카메라 SDK나 모델 환경이 설치되지는 않습니다.
카메라별 방법은 [설치 안내](docs/setup.md#camera)를 참고합니다.

### 개인 설정 후 실행

```bash
# 기존 config.local.yaml이 있으면 덮어쓰지 않습니다.
cp -n examples/config.local.example.yaml config.local.yaml
```

`config.local.yaml`에서 카메라 종류를 정합니다. 예를 들어 RealSense를 사용한다면:

```yaml
camera:
  type: realsense          # zed / realsense
  serial: null             # null: SDK 기본 장치
```

공유 기본값과 예시 파일은 **카메라·UI 켜짐, 모든 선택 모델 꺼짐**입니다.
모델을 사용하려면 아래 단계표대로 켜고 [모델별 설치·경로](docs/setup.md)를 준비합니다.

```bash
grasp-workbench run
```

Ctrl+C 또는 UI 종료로 해당 실행의 worker를 정리합니다. 일반 실행에 긴 옵션을 붙일 필요는 없습니다.

<a id="stages"></a>
## 2. 어디까지 사용할지 선택

다음 표의 값은 각 YAML section의 **`enabled`**입니다. VLM을 끈 수동 흐름 기준이며 필요한 조합을 합칠 수 있습니다.

| 사용할 범위 | `sam3` | `sam3d` | `foundationpose` | `graspgenx` | `grasp_selection` |
|---|---|---|---|---|---|
| 카메라 영상만 | false | false | false | false | false |
| mask까지만 | true | false | false | false | false |
| mesh 생성까지만 | true | true | false | false | false |
| 기존 mesh로 pose | true | false | true | false | false |
| 새 mesh 생성 후 pose | true | true | true | false | false |
| 파일 mesh에서 grasp 후보만 | false | false | false | true | false |
| 저장 grasp 선택·양팔 조합만 | false | false | false | false | true |
| 새 mesh → pose → grasp → 선택 | true | true | true | true | true |

- **VLM 인식을 사용하려면** 위 조합에 `vlm.enabled: true`와 backend 설정을 추가합니다.
- **`enabled: true`는 worker 시작**입니다. 버튼을 누르기 전부터 모든 추론을 실행한다는 뜻은 아닙니다.
  SAM3는 Apply 이후 연속 처리하고, 나머지는 버튼 또는 명시적인 VLM/Task 진행으로 요청합니다.
- 각 모델 worker는 따로 켭니다. VLM·FP·SAM3D를 켠다고 SAM3 backend까지 자동 시작하지 않습니다.
- `run`은 항상 카메라를 시작합니다. **`camera.enabled`는 없습니다.**
- 모델 기능을 켜면 `ui.enabled: false`여도 UI를 사용합니다. 카메라만 콘솔로 확인할 때는 모든 모델을 끄고
  `ui.enabled: false`로 설정합니다. Headless 모델 실행 설정으로 해석하지 않습니다.

<a id="configuration"></a>
## 3. YAML 설정

| 파일 | 용도 |
|---|---|
| [`config.yaml`](config.yaml) | 공유 기본값·전체 설정 구조 |
| `config.local.yaml` | 사용자별 on/off·장치·모델 경로·인증 파일 경로. Git 제외 |
| [`examples/config.local.example.yaml`](examples/config.local.example.yaml) | 개인 설정을 작성할 때 참고할 전체 예시 |
| [모델별 설치·설정](docs/setup.md) | checkpoint·독립 Python 환경·VLM 인증 준비 |

일반 `run`은 저장소의 `config.yaml`을 읽고, `config.local.yaml`에 **적힌 항목만 덮어씁니다.**
실행한 현재 폴더가 아니라 저장소 위치 기준입니다. 상대 경로는 해당 값을 정의한 YAML 파일의 폴더 기준이며,
각 모델의 `python: null`은 현재 실행 중인 Python을 사용합니다.

주요 선택 옵션은 다음과 같습니다. 활성화 여부와 모델 필수 경로는 별도로 설정합니다.

```yaml
sam3d:
  depth_source: depth       # depth: 실측 RGB-D / monocular: RGB에서 깊이 추정
  texture_baking: false     # true: UV·material을 보존한 texture GLB 생성

graspgenx:
  gripper_id: franka_panda  # 사용자가 준비한 gripper ID
  assets_dir: null          # null: GraspGenX 기본 assets. 별도 descriptor 경로 지정 가능

grasp_selection:
  robot_config: null        # null: 로봇 없이 object-local 후보 선택·표시
```

`depth` 결과는 실측 depth 기반 metre 좌표를 사용합니다. `monocular`는 **예측 scale**이며 실제 크기·거리가 보장되지 않습니다.
texture는 표면 표현 옵션이고 pose·grasp 생성의 필수 조건이 아닙니다. [크기·좌표·cache 설명](docs/usage.md#sam3d)

<a id="vlm"></a>
## 4. VLM 선택

기존 flat `vlm` section의 **`backend`와 `model`**을 함께 설정합니다. VLM을 끄면 UI에서 A/B prompt를 직접 입력합니다.

| `backend` | 연결 방식 | 준비할 것 |
|---|---|---|
| `openai_api` | GPT / OpenAI Responses API | `.[vlm]`, 자신의 모델 ID·API 키 |
| `codex_cli` | 설치된 Codex CLI의 단발 인식 | CLI 설치·자신의 로그인, Python Pillow |
| `anthropic_api` | Claude Messages API | `.[vlm-anthropic]`, 자신의 모델 ID·Anthropic API 키 |

아래 **셋 중 하나의 `vlm` 블록**을 개인 설정에 사용합니다. 환경 생성·SDK 설치·로그인 방법은
[설치 안내](docs/setup.md)에 있습니다.

### GPT API

```yaml
vlm:
  enabled: true
  python: .venv-vlm/bin/python
  backend: openai_api
  model: gpt-6.1-sol
  api_key_file: .secrets/openai_api_key.txt
```

[GPT API 설치·인증](docs/setup.md#vlm-openai)

### Codex CLI

```yaml
vlm:
  enabled: true
  python: .venv-vlm/bin/python
  backend: codex_cli
  model: gpt-6.1-sol
  codex_executable: codex
  api_key_file: null
```

사용자가 설치한 CLI의 인증 상태를 사용하며 지정 API 키 파일은 읽지 않습니다.
[CLI 설치·로그인](docs/setup.md#vlm-codex)

### Claude API

```yaml
vlm:
  enabled: true
  python: .venv-vlm-anthropic/bin/python
  backend: anthropic_api
  model: YOUR_CLAUDE_VISION_STRUCTURED_OUTPUT_MODEL_ID
  api_key_file: .secrets/anthropic_api_key.txt
```

`model`은 자신의 계정에서 **이미지 입력·구조화 출력을 지원하는 실제 모델 ID**로 바꿉니다.
Claude Code CLI를 사용하는 backend는 아닙니다. [Claude API 설치·인증](docs/setup.md#vlm-anthropic)

- backend를 바꿔도 모델 이름이나 인증은 자동 변경하지 않습니다. 실패 시 다른 provider로 자동 fallback하지 않습니다.
- API backend의 `api_key_file: null`은 각각 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 환경변수를 사용합니다.
- 키 값은 YAML에 쓰지 않습니다. 각 사용자가 준비한 Git 제외 파일이나 환경변수를 사용합니다.
- 새 VLM은 공통 결과를 반환하는 adapter를 기존 VLM worker에 등록합니다. [다른 VLM 추가 방법](docs/setup.md#custom-vlm)

<a id="workflow"></a>
## 5. UI 사용 흐름

### 수동 prompt로 진행

1. **분할:** A/B에 서로 다른 영어 prompt를 넣고 **Apply**합니다. 예: `cell phone`, `yellow tuna can`.
   자동 번역은 없으며, SAM3 후보가 여러 개면 사용할 후보를 선택합니다.
2. **mesh·pose 분기:** 기존 mesh는 파일·scale을 선택해 **기존 mesh로 Start**, 새 mesh는
   **mesh 없음 · 생성 후 Start**로 생성 후 FP를 시작합니다. **Generate mesh**는 mesh만 만듭니다.
3. **표시:** 최신 카메라 RGB 위에 SAM3 mask·FP pose 축을 표시하고, 오른쪽 3D plot에서 mesh·pose를 확인합니다.
4. **grasp:** GraspGenX로 후보를 생성하거나 기존 bundle을 불러온 뒤, 한팔/양팔 선택을 요청합니다.
   FP 없이 파일 mesh의 object-local grasp 후보만 확인할 수도 있습니다.

### VLM으로 진행

**VLM 사용**을 켜고 지시문을 입력한 뒤 **VLM 인식 → Task 진행** 버튼으로 현재 RGB 한 장을 인식합니다.

- 기존 asset ID가 선택되면 저장 mesh로 FP를 시작하고, 새 물체는 SAM3D 생성 후 FP에 연결합니다.
- 활성화한 기능 범위에서 FP → **Source grasp cache 조회·생성 → 후보 선택**까지 연결합니다.
- Source/Target은 작업상의 물체 역할이며 팔 배정과 다릅니다. 역할을 수동 변경하면 기존 의미 기반 팔 배정을 해제합니다.
- VLM 인식은 매 프레임 실행하지 않습니다. FP를 끄면 prompt·semantic 결과까지만 반영합니다.

고정 snapshot·mask 보정·FP seed·Session 저장·표시용 hand FK 등은 [상세 사용법](docs/usage.md)을 참고합니다.

<a id="documentation"></a>
## 6. 문서와 저장소 구조

| 문서 | 내용 |
|---|---|
| [설치·설정](docs/setup.md) | 카메라·모델·VLM별 환경, YAML 예시, robot adapter 준비 |
| [상세 사용법](docs/usage.md) | UI 조작, Source/Target, 단위·좌표·cache, IPC 계약 |
| [검증 이력](docs/verification.md) | 실제 모델·저장 입력·모의 backend 검증과 미검증 범위 |
| [보관 mesh](assets/meshes/README.md) | mesh 출처·원본/최적화 파일·catalog |

```text
grasp_workbench/    # 실행 진입점, worker, camera, model adapter, IPC, Qt UI
config.yaml        # 공유 기본값
examples/          # 개인 설정·선택적 robot adapter 예시
third_party/       # 원본을 수정하지 않는 공식 submodule 6개
assets/meshes/     # 기존 물체 mesh·catalog
image/             # README 예시 이미지
docs/              # 설치·사용·검증 상세 문서
runs/              # 생성 asset·grasp cache·snapshot·로그: Git 제외
```

각 기능은 독립 worker 프로세스로 실행합니다. worker 사이의 이미지·mask·mesh·pose·요청/결과는 iceoryx2로 전달합니다.
원본 SDK와 upstream 저장소를 수정하거나 특정 FFW·ROS 환경을 필수로 연결하지 않습니다.

<a id="verification"></a>
## 7. 검증 범위

로컬 검증 환경은 **Linux x86_64·Python 3.12.3**이며, 최근 전체 **937개 테스트·Ruff 검사**를 통과했습니다.
새 PC의 전체 설치·하드웨어 실행이 검증되었다는 뜻은 아닙니다.

```bash
python -m pytest tests -q
# 선택 사항: Ruff는 별도로 설치합니다.
python -m pip install ruff
ruff check grasp_workbench tests
```

- 자동 테스트는 모의 SDK·모델과 실제 독립 프로세스·iceoryx2·offscreen UI를 함께 사용합니다.
- 실제 모델을 사용한 저장 RGB-D 재생 검증과 실제 카메라 검증은 [이력](docs/verification.md)에 구분했습니다.
- **Claude 실서비스 호출, monocular·texture 실추론, 실측 로봇 hand FK는 미검증**입니다.
  Claude는 실제 SDK의 오프라인 모의 HTTP 직렬화·파싱까지 확인했습니다.
- IK는 pregrasp/grasp **endpoint 도달 가능성 확인**입니다. 장면 충돌·경로 계획·실물 grasp 성공을 보장하지 않습니다.

<a id="sharing"></a>
## 8. 공유 시 주의사항

- `.secrets/`, `.env*`, `config.local.yaml`, CLI 인증자료, `.venv*/`, 가중치와 `runs/`는 공유하지 않습니다.
  `.gitignore`는 폴더 전체 ZIP에 적용되지 않으며, 이미 추적된 파일·과거 커밋도 별도로 확인합니다.
- 이 저장소 자체의 라이선스는 아직 미정이며 루트 `LICENSE`가 없습니다.
  보관한 45개 mesh의 재배포 권한과 각 upstream 소스·가중치 조건은 별도로 확인해야 합니다.
- 이 예제는 **로봇 제어·이동, 데이터 수집·학습, policy 실행, trajectory·장면 충돌 검사**를 포함하지 않습니다.
