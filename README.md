# Grasp Workbench — 카메라·분할·mesh·pose·grasp worker 예제

**1원칙: 카메라·VLM 설치와 선택은 공유받는 사람이 설정한다. 우리는 최소 연결 예제만 제공한다.**

현재 단계는 **camera → iceoryx2 → UI**, 선택적인 **VLM 단발 인식 → SAM3 A/B 연속 분할 → SAM3D mesh 생성 / FoundationPose 등록·추적 / GraspGenX 후보 생성 / 한팔·양팔 grasp 선택**이다.
UI는 RGB·생성 mesh를 표시하며, FP를 켜면 영상의 SAM3 mask·box 위에 A/B pose 축을 겹쳐 표시한다.
오른쪽 3D plot에는 카메라 기준 위치·회전축과 추정 pose로 변환한 mesh를 함께 표시한다.
고정 RGB-D snapshot 검토·SAM3 박스/포인트 보정·FP seed refinement와 명시적인 Task 후처리도 제공한다.
이전 단계표 UI는 제거했다. FFW·ROS를 필수로 요구하지 않으며 자동 번역·로봇 이동 명령은 연결하지 않는다.
선택적인 robot adapter가 있으면 **로봇 기준 좌표 변환·grasp 도달 가능성 확인**까지 수행한다.
이는 endpoint IK이며 actuator 명령·이동·TrajOpt·장면 충돌 검사·trajectory 생성은 포함하지 않는다.
`igris_teleop_v4`의 worker 역할 분리만 참고하며, 원본 프로젝트 코드는 변경하지 않는다.

## 공유 전 확인

개발·통합 예제로 검토할 수 있지만, **공개 배포 준비가 모두 끝난 저장소는 아니다**.

- 이 repo 자체의 라이선스는 아직 정하지 않았으며 루트 `LICENSE`가 없다.
  복사된 `assets/meshes/`의 재배포 권한도 확인이 필요하다. 각 upstream 소스·가중치의 라이선스는 별도로 확인한다.
- 카메라 SDK·GPU/모델 환경·checkpoint·VLM 인증·gripper 설정은 공유받는 사람이 준비한다.
  아래 설치 명령은 기본 UI/전송 환경용이며 모든 모델을 자동 설치하지 않는다.
- 기존 로컬 환경의 **전체 테스트는 937개 통과**했으나, 새 PC에서 clone부터 전체 설치·하드웨어 실행까지는 검증하지 않았다.
  기능별 실제 모델·fake/native 검증과 미검증 범위는 아래 검증 이력에 구분했다.
- 커밋 대상에 개인 설정·인증자료·가중치·`runs/`가 포함되지 않았는지 확인한다.

## 구조

```text
grasp_workbench/
├── main.py                         # worker 프로세스 시작·종료
├── config.py                       # YAML 설정 읽기
├── core/worker_base.py              # on_start → step_once → on_stop
├── workers/registry.py              # 기능별 worker 등록
├── workers/worker_camera.py         # 선택한 SDK 캡처·발행 프로세스
├── workers/worker_camera_monitor.py # 독립 수신·정보 출력 프로세스
├── workers/worker_camera_ui.py      # 선택적인 독립 Qt UI 프로세스
├── workers/worker_sam3.py           # 선택적인 독립 SAM3 프로세스 · 모델 하나
├── workers/worker_sam3d.py          # 명시적 mesh 요청 · 입력별 asset cache
├── workers/worker_foundationpose.py # A/B tracking · SAM3 검증·재등록
├── workers/worker_vlm.py            # 버튼 요청당 한 번의 인식 · API/CLI backend 선택
├── workers/worker_graspgenx.py      # mesh-only 후보 요청 · 독립 GraspGenX worker
├── workers/worker_grasp_selection.py # 한팔/양팔 선택 · 선택적인 상태/FK/endpoint IK
├── models/sam3.py                   # 공식 SAM3 image API 최소 adapter
├── models/sam3_association.py       # FP 검증용 후보 bbox 연결
├── models/sam3d.py                  # 실측 pointmap · metric mesh/좌표 변환
├── models/foundationpose.py         # 공식 FoundationPose 최소 adapter
├── models/vlm.py                    # OpenAI API adapter · 공통 구조화 결과 계약
├── models/vlm_codex.py              # 기존 CLI 로그인으로 단발 codex exec
├── models/vlm_anthropic.py          # Claude API adapter · 동일 semantic 결과
├── models/graspgenx.py              # 공식 GraspGenX adapter · 이전 추론 설정 유지
├── models/grasp_diversity.py        # object-local SE(3) 다양성 선택
├── models/grasp_selection.py        # 후보 조합·COM proxy·도달 가능성 순위
├── models/grasp_fk.py               # 명시적 파지 관계 + 실측 hand FK · 표시용
├── models/robot_reachability.py     # 선택적인 module:factory robot adapter
├── models/robot_ffw.py              # 선택적인 FFW 상태/FK·cuRobo endpoint IK 예제
├── assets.py                        # 기존 catalog + 새 mesh의 영속 asset 저장소
├── grasp_assets.py                  # geometry·gripper·model·설정별 grasp bundle/cache
├── semantic.py                      # 관측 ID·Task/팔/상대 제약·공개 policy 계약
├── policy_catalog.py                # 선택적인 policy metadata · checkpoint는 로컬 참조
├── semantic_session.py              # 관측→instance 연결·snapshot 저장·request_only export
├── snapshot.py                      # RGB-D·mask·pose 고정 검토용 NPZ 저장/불러오기
├── ui/receiver.py                   # UI 내부 수신 thread · 최신 frame 하나 유지
├── ui/camera_window.py              # RGB·후보 선택 + mesh/pose/grasp 3D 표시
├── ui/mesh_preview.py               # 표시용 간소화 · FP와 동일한 static mesh 로드
├── ui/grasp_preview.py              # 선택 grasp 축·gripper preview와 좌표 합성
├── ui/mask_editor.py                # 원본 pixel 좌표의 포함/제외 박스·포인트 편집
├── cameras/zed.py                   # 간단한 ZED 연결 예제
├── cameras/realsense.py             # 간단한 RealSense 연결 예제
├── ipc/camera.py                    # iceoryx2 raw RGB-D 전송
├── ipc/sam3.py                      # prompt 설정·동일 프레임 mask 결과
├── ipc/sam3_refine.py               # 기존 SAM3 worker의 단발 고정 frame 보정
├── ipc/sam3d.py                     # 고정 frame/mask 요청·raw mesh 결과
├── ipc/foundationpose.py            # A/B session 제어·RGB-D와 pose 결과
├── ipc/vlm.py                       # 인식 요청·원본 frame·물체 결과
├── ipc/graspgenx.py                 # mesh 요청·raw object-local grasp/score 결과
└── ipc/grasp_selection.py           # coordination·geometry snapshot 요청·선택 결과
config.yaml                         # 공유 가능한 기본 설정
config.local.yaml                   # 선택적인 개인 설정 · Git 제외
examples/config.local.example.yaml  # 복사한 뒤 필요한 기능만 켜는 전체 설정 예시
examples/robot_ffw.yaml              # 사용자 경로를 채우는 선택적 robot adapter 예제
assets/meshes/                      # 기존 FFW 물체 21개 원본/최적화 + Table/Leg 보관본
runs/assets/                        # 새 asset별 object.ply/GLB·manifest.json · Git 제외
runs/grasps/                        # grasp 후보·원본 pool·metadata cache · Git 제외
runs/tasks/                         # semantic session·실행 요청 JSON · Git 제외
third_party/
├── iceoryx2/                        # 실제 submodule, v0.9.3
├── zed-python-api/                  # 실제 submodule, v5.4.0
├── sam3/                           # 실제 submodule, 2345a4ad (기존 검증 revision)
├── sam-3d-objects/                  # 실제 submodule, f91db411
├── FoundationPose/                 # 실제 submodule, a1b694b8
└── GraspGenX/                      # 실제 submodule, b9429097
```

## 설치

```bash
git submodule update --init
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,ui]'
```

현재는 **Git checkout + editable 설치(`-e`)**를 기준으로 한다. 단독 wheel에는 저장소 루트의 YAML·mesh·submodule이 포함되지 않으므로 독립 wheel 배포는 지원하지 않는다.

ZED를 쓰는 사람은 기기에 맞는 **ZED SDK와 pyzed**를 별도로 설치한다.
예: SDK 설치 후 해당 환경에서 `python /usr/local/zed/get_python_api.py`.
`third_party/zed-python-api` 소스를 받는 것만으로 SDK가 설치되지는 않는다.
자세한 설치는 [ZED 공식 Python API](https://github.com/stereolabs/zed-python-api)를 따른다.

RealSense 예제는 `python -m pip install -e '.[realsense]'`로 Python binding을 추가한다.
장치·드라이버·권한은 사용하는 사람이 준비한다. 다른 카메라 예제를 사용할 때 이 SDK는 필요 없다.

기본 설정은 UI를 사용하므로 위 설치 예제에 PyQt6·Matplotlib(`.[ui]`)을 포함했다.
콘솔만 쓸 때는 SAM3·SAM3D·FoundationPose·VLM·GraspGenX·grasp_selection을 끄고 `ui.enabled: false`로 설정한 뒤 `.[test]`만 설치해도 된다.
카메라 SDK는 UI 프로세스에서 사용하지 않는다.

## 설정하고 `run`으로 실행

평소에는 **YAML을 수정하고 아래 명령 하나만 실행**한다. 공유 기본값은 ZED·UI 켜짐·모델 worker 꺼짐이다.

```bash
grasp-workbench run
```

저장소 루트의 `config.yaml`을 읽고, `config.local.yaml`이 있으면 **적힌 항목만 덮어쓴다**.
실행한 현재 폴더가 아니라 저장소 위치를 기준으로 두 파일을 찾는다.
공유 `config.yaml`의 설정 구조는 다음과 같다.

```yaml
camera:
  type: zed               # zed 또는 realsense
  serial: null            # null이면 SDK 기본 장치
  resolution: HD720       # ZED 설정
  fps: 30
ipc:
  service: grasp_workbench/camera/rgbd
ui:
  enabled: true           # false이면 콘솔 monitor
sam3:
  enabled: false
  python: null            # null이면 실행 중인 Python
  checkpoint: null
  device: cuda
sam3d:
  enabled: false
  python: null
  source_dir: null        # 기본값: third_party/sam-3d-objects
  config_path: null       # 설치한 모델의 pipeline.yaml
  output_dir: runs/meshes
  seed: 42
  depth_source: depth      # depth 또는 monocular
  texture_baking: false
foundationpose:
  enabled: false
  python: null
  source_dir: null        # 기본값: third_party/FoundationPose
  device: cuda:0
  registration_iterations: 5
  tracking_iterations: 1
vlm:
  enabled: false
  python: null
  backend: openai_api      # openai_api / codex_cli / anthropic_api
  codex_executable: codex  # CLI backend의 명령 또는 실행 파일 경로
  model: gpt-6.1-sol
  api_key_file: null      # API: 로컬 키 파일 또는 null(해당 provider 환경변수) · CLI는 무시
  asset_dir: runs/assets
  policy_catalog: null   # 선택적인 로컬 JSON · 없으면 policy 목록 없음
  session_dir: runs/tasks
graspgenx:
  enabled: false
  python: null
  source_dir: null        # 기본값: third_party/GraspGenX
  checkpoint_dir: null
  assets_dir: null
  gripper_id: franka_panda
  output_dir: runs/grasps
  settings: {}
grasp_selection:
  enabled: false
  python: null
  robot_config: null      # null이면 canonical 후보 선택·표시만 수행
  settings: {}
run:
  frames: 0               # 수신 N프레임 뒤 종료 · 0이면 계속
  timeout: 0              # 제한 시간(초) · 0이면 제한 없음
```

환경을 activate하지 않았다면 `.venv/bin/grasp-workbench`로 실행한다.
설치 없이 이미 의존성이 있는 Python에서는 `python -m grasp_workbench.main run`도 동일하다.
Ctrl+C나 UI 종료 시 해당 실행에서 시작한 worker를 정리한다.

### 어디까지 사용할지 선택

아래는 **한 가지 기능만 확인하는 기본 조합**이다. `config.local.yaml`에 필요한 `enabled`만 덮어쓴다.
전체 항목을 보고 수정하려면 `examples/config.local.example.yaml`을 저장소 루트의 `config.local.yaml`로 복사한다.
이미 개인 설정 파일이 있으면 덮어쓰지 말고 필요한 항목만 참고한다.
모델을 켜기 전 각 환경·checkpoint·필수 경로는 아래 기능별 설치 안내대로 채운다.

| 목적 | `sam3` | `sam3d` | `foundationpose` | `graspgenx` | `grasp_selection` | `vlm` | UI에서 할 일 |
|---|---|---|---|---|---|---|---|
| 카메라 영상만 | false | false | false | false | false | false | 영상 수신 |
| mask까지만 | true | false | false | false | false | false | A/B prompt 입력·Apply |
| mesh까지만 | true | true | false | false | false | false | SAM3 후보 선택 → Generate mesh |
| 기존 mesh로 pose | true | false | true | false | false | false | mesh·scale 선택 → 기존 mesh로 Start |
| 새 mesh 생성 후 pose | true | true | true | false | false | false | 후보 선택 → mesh 없음 · 생성 후 Start |
| grasp 후보까지만 | false | false | false | true | false | false | mesh·scale 선택 → Generate grasp |
| 저장 grasp 선택·조합 | false | false | false | false | true | false | 기존 bundle 불러오기 → 선택 요청 |

- 표의 이름은 YAML section이며 값은 각 section의 `enabled`다. 필요한 조합은 합칠 수 있다.
  예를 들어 기존 mesh pose와 grasp 선택을 함께 쓰려면 해당 기능을 모두 `true`로 정한다.
- `run`은 항상 카메라 worker를 시작한다. **`camera.enabled`는 지원하지 않는다.**
  기본 `ui.enabled: true`는 영상 UI이며, 모델 기능을 하나라도 켜면 `ui.enabled: false`여도 UI를 사용한다.
  모든 모델 기능을 끈 카메라 전송만 확인할 때에만 `ui.enabled: false`로 콘솔 monitor를 선택한다.
- **`enabled: true`는 worker를 실행한다는 뜻이지 모든 추론을 즉시 실행한다는 뜻이 아니다.**
  SAM3는 Apply한 prompt를 연속 처리하며 mesh·pose·grasp·선택은 해당 버튼이나 명시적 VLM/Task 흐름으로 요청한다.
- VLM으로 prompt·기존/신규 mesh 분기를 맡기려면 `vlm.enabled: true`를 추가하고 아래 backend 중 하나를 설정한다.
  기존 mesh pose에는 SAM3·FP, 새 mesh pose에는 SAM3D도 직접 켠다. VLM이 다른 worker를 자동 활성화하지 않는다.
- `grasp_selection.robot_config: null`이면 로봇 없이 canonical 후보를 고른다.
  base/TCP·IK 확인까지 하려면 자신의 robot adapter 설정과 현재 물체 pose가 필요하다.
  실시간 pose가 필요하면 위 pose 조합도 함께 켜며, 로봇 명령·이동은 수행하지 않는다.

### 개인 설정·짧은 확인

개인 장치 serial·모델 경로 등은 Git에서 제외되는 `config.local.yaml`에 필요한 항목만 적는다.
예를 들어 콘솔에서 30프레임만 확인하려면 다음 설정 후 같은 `grasp-workbench run`을 실행한다.

```yaml
# config.local.yaml
ui:
  enabled: false
sam3:
  enabled: false
sam3d:
  enabled: false
foundationpose:
  enabled: false
vlm:
  enabled: false
graspgenx:
  enabled: false
grasp_selection:
  enabled: false
run:
  frames: 30
  timeout: 20
```

특정 장치를 고르려면 `camera.serial`을 자신의 serial 문자열로 설정한다.
RealSense는 binding 설치 후 `camera.type: realsense`로 바꾼다. 장치 목록만 조회하는 명령은 다음과 같다.

```bash
grasp-workbench cameras --camera zed
grasp-workbench cameras --camera realsense
```

다른 설정 파일 하나만 사용하려면 `grasp-workbench run --config /path/to/settings.yaml`을 지정한다.
이 경우 기본 공유·개인 파일을 추가로 읽지 않는다. 상대 경로는 **그 값을 정의한 YAML 파일의 폴더 기준**이다.
기존 `run`의 개별 옵션도 일회성 YAML override로 사용할 수 있지만, 일반 실행에는 필요 없다.

각 worker는 단독 실행도 가능하다. 터미널 두 개에서 같은 service 이름을 사용한다.

```bash
python -m grasp_workbench.workers.worker_camera_monitor --service example/camera/rgbd
python -m grasp_workbench.workers.worker_camera --camera zed --service example/camera/rgbd
```

## 카메라 UI

```bash
# ui.enabled: true 설정으로 camera + 별도 UI worker 실행
grasp-workbench run

# 이미 발행 중인 service만 구독 — 카메라를 새로 열지 않음
grasp-workbench ui --service example/camera/rgbd
```

```text
main (프로세스 관리)
├── camera worker ── iceoryx2 RGB-D ──┐
└── UI worker ◀──────────────────────┘
    ├── receiver thread: subscriber 생성·수신·해제, 최신 frame 하나 유지
    └── Qt main thread: RGB 영상 + 선택적인 생성 mesh 표시
```

- 왼쪽 RGB는 비율을 유지한다. 오른쪽 3D 축은 입력 전 비어 있으며, SAM3D asset 하나 또는
  FP mesh·pose 축·선택 grasp를 표시한다. 임의의 결과나 FPS 그래프는 만들지 않는다. 별도 plot 프로세스도 없다.
- UI를 닫거나 Ctrl+C를 누르면 해당 실행에서 시작한 worker를 정리한다.
  단독 `ui` 명령은 외부에서 실행한 camera publisher를 종료하지 않는다.
- UI는 수신 count·마지막 수신 경과·camera/sequence를 표시한다. 영상이 없으면 대기 상태를 유지한다.
- `run.frames: N`, `run.timeout: S`는 콘솔과 UI에 동일하게 적용한다. N개를 못 받은 채 시간이 끝나면 실패한다.
  `run.frames: 0`에서는 시간 종료 자체가 오류는 아니며, 수신 성공 여부는 실제 수신 count로 구분한다.

## SAM3: UI에서 두 물체 prompt 입력

VLM 없이 **A/B text prompt**를 직접 입력한다. A/B는 서로 다른 prompt를 위한 고정 슬롯이며,
SAM3 모델을 두 벌 띄우지 않는다. 한 슬롯을 비우면 해당 슬롯은 처리하지 않는다.

### 선택적인 모델 환경

카메라/UI 환경과 분리할 수 있다. 먼저 장치에 맞는 PyTorch·torchvision을 설치하고,
이 repo의 SAM3 submodule과 최소 adapter 의존성을 설치한다.

```bash
python3.12 -m venv .venv-sam3
# 아래는 이번 기기의 검증 예시. 각자 GPU/CUDA에 맞는 PyTorch를 선택한다.
.venv-sam3/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu130
.venv-sam3/bin/python -m pip install -e '.[sam3]' -e third_party/sam3
```

공식 `sam3.pt` checkpoint는 사용자가 준비한다. 가중치·접근 토큰은 Git에 포함하지 않으며,
인증이나 가중치 다운로드를 앱에서 자동 수행하지 않는다.
[공식 SAM3 설치·checkpoint 안내](https://github.com/facebookresearch/sam3#installation)를 참고한다.
이 예제는 submodule의 고정 revision `2345a4ad109ac29c569da749c91d84f10dc08c40`을 사용한다.

`config.local.yaml`에 모델 환경과 checkpoint를 지정한다. `sam3.enabled: true`이면 UI도 함께 켠다.
경로는 예시이므로 자신의 파일 위치로 바꾼다. 상대 Python 경로는 이 YAML이 있는 폴더 기준이다.

```yaml
# config.local.yaml
sam3:
  enabled: true
  python: .venv-sam3/bin/python
  checkpoint: /path/to/sam3.pt
  device: cuda
```

```bash
grasp-workbench run
```

UI에서 예를 들어 **A: `red cup` / B: `blue bowl` → Apply**를 누른다.
입력 문구는 SAM3에 그대로 전달하며 **자동 번역하지 않는다**. 공식 [모델 카드](https://huggingface.co/facebook/sam3)는
English로 표시되어 있다. 한글 입력을 막지는 않지만, 우선 `cell phone`, `orange ball`처럼 영어 이름·외형으로 확인한다.
같은 물체라도 prompt에 따라 검출 결과가 달라질 수 있다. 오류 없이 **0개 후보**인 경우는 추론 실패와 구분한다.
**Stop**은 SAM3 추론을 중지하고 원본 카메라 영상 표시로 돌아간다. 카메라 캡처는 계속된다.
진행 중인 GPU 호출은 완료될 수 있으며, 이전 revision의 결과는 표시하지 않는다.
prompt를 수정한 뒤에는 Apply를 다시 누른다. 기본값은 비어 있으며 자동 추론하지 않는다.

```text
UI ── prompt revision + A/B text ──▶ SAM3 worker
camera ── RGB-D ──────────────────▶ SAM3 worker
                                     │ 모델 1회 load · frame image encode 1회
                                     │ A/B prompt 각각 reset·분할
UI ◀── revision + 원본 RGB-D + A/B masks/boxes/scores
```

- 위 프로세스 간 데이터와 prompt 설정은 모두 iceoryx2로 전달한다. 늦게 시작한 worker에도
  마지막 prompt 설정이 전달되도록 UI 수신 thread가 publisher connection을 갱신한다.
- 연속으로 들어오는 **최신 프레임의 image segmentation**이다. 모든 카메라 프레임을 추론하거나
  SAM3 video tracking·영구적인 물체 identity를 구현했다고 주장하지 않는다.
- prompt마다 후보를 모두 보존하고, A는 초록색/B는 주황색으로 `A[1]`, `B[1]` 등의 mask·box를
  표시한다. 후보가 여러 개면 임의로 한 물체를 확정하지 않는다. 0개·슬롯별 오류도 구분한다.
- 화면은 **최신 카메라 RGB + 최신 SAM3 mask·box**로 계속 갱신한다. 카메라와 추론 결과의 sequence를
  각각 표시하며, 동기화된 측정이라고 해석하지 않는다. 표시용 overlay와 별개로 mesh 생성·FP 등록에는
  SAM3 결과에 포함된 정확한 원본 RGB-D/mask를 사용한다. 카메라 수신 전에는 결과 원본 영상을 대신 표시한다.
- Apply/Stop마다 revision이 바뀌고 이전 revision 결과는 폐기한다. A/B ID는 prompt 슬롯이지,
  SAM3D mesh cache의 영구 물체 key나 6D pose가 아니다. FP와 VLM은 별도 선택 기능이다.

이미 발행 중인 카메라에 각각 독립적으로 연결할 수도 있다.

```bash
.venv-sam3/bin/python -m grasp_workbench.workers.worker_sam3 \
    --service example/camera/rgbd --checkpoint /path/to/sam3.pt
.venv/bin/grasp-workbench ui --sam3 --service example/camera/rgbd
```

### 고정 snapshot과 mask 보정

- **현재 프레임 캡처**로 RGB-D와 수신한 SAM3 mask·pose·mesh 참조를 고정해 검토한다.
  **Snapshot 저장/불러오기**는 이를 NPZ에 보존하며 **실시간 보기**로 돌아갈 수 있다.
  각 결과는 자기 원본 frame을 보존하므로 RGB·SAM3·FP를 같은 시각으로 임의 동기화하지 않는다.
  불러온 snapshot은 과거 입력이며 load만으로 추론·tracking·로봇 명령을 시작하지 않는다.
  이 repo의 snapshot 형식이며 이전 앱의 NPZ snapshot 호환 기능은 추가하지 않았다.
- A/B **mask 보정**에서 포함/제외 포인트 또는 박스를 지정하고 **SAM3 보정 요청**을 누른다.
  좌클릭은 포함, 우클릭은 제외이며 박스는 드래그한다. UI의 표시 좌표를 원본 pixel 좌표로 바꿔
  정확히 고정한 RGB-D·선택 seed mask와 함께 기존 SAM3 worker로 보낸다.
- `/sam3/refine/requests`, `/sam3/refine/results`는 별도 IPC 채널이지만 **새 모델 worker를 띄우지 않는다**.
  보정 결과는 고정 입력으로 표시되며 이후 mesh 생성·FP 입력에도 그 원본을 사용한다.
  mask를 UI에서 임의의 사각형/픽셀 배열로 대체하는 기능이 아니라 실제 SAM3 재추론이다.

## VLM: 버튼 한 번으로 인식·기존 mesh 재사용

VLM은 선택 기능이며 공유 기본값은 **꺼짐·`backend: openai_api`·키 파일 경로 null**이다.
기존 flat `vlm` section에서 `backend`·`model`·`python`·`api_key_file`을 바꾼다.
`providers:` 같은 추가 설정 계층은 없다. 모델·인증·네트워크 환경은 사용자가 준비한다.

| `vlm.backend` | 연결 방식 | 준비할 것 |
|---|---|---|
| `openai_api` | GPT/OpenAI Responses API | `.[vlm]`, 자신의 모델 ID·API 키 |
| `codex_cli` | 로컬 Codex CLI의 단발 이미지 인식 | Codex 설치·자신의 CLI 로그인, Python Pillow |
| `anthropic_api` | Claude Messages API | `.[vlm-anthropic]`, 자신의 모델 ID·Anthropic API 키 |

backend를 바꿔도 `model`은 자동 변경하지 않는다. 특히 공유 기본 `gpt-6.1-sol`을 Claude 모델로 바꿔주지는 않는다.
각 provider에서 **이미지 입력과 구조화 출력을 지원하며 자신의 계정에서 사용 가능한 모델 ID**를 직접 지정한다.

### GPT / OpenAI API backend

OpenAI Responses API에 현재 RGB 한 장을 전달하고 JSON Schema로 최대 두 물체를 받는다.
[공식 이미지 입력 안내](https://developers.openai.com/api/docs/guides/images-vision)와
[Structured Outputs 안내](https://developers.openai.com/api/docs/guides/structured-outputs)를 따른다.

```bash
python3.12 -m venv .venv-vlm
.venv-vlm/bin/python -m pip install -e '.[vlm]'
```

이미 `.venv-vlm`이 있으면 새로 만들 필요 없이 같은 환경에 설치한다. `.[vlm]`은 OpenAI SDK·Pillow를 설치한다.

```yaml
# config.local.yaml — 자신의 로컬 인증 파일 경로만 설정
vlm:
  enabled: true
  python: .venv-vlm/bin/python
  backend: openai_api
  model: gpt-6.1-sol
  api_key_file: .secrets/openai_api_key.txt
  asset_dir: runs/assets
```

키 값은 YAML·코드·Git에 적지 않는다. `.secrets/`와 `config.local.yaml`은 Git에서 제외하며,
API 키 파일은 해당 backend의 VLM worker만 읽는다. UI 요청·iceoryx2 메시지·로그에 키 값이나 인증 파일 경로를 넣지 않는다.
`api_key_file: null`이면 worker 환경의 `OPENAI_API_KEY`를 사용한다. 키 값은 예시 YAML에 넣지 않는다.

### Codex CLI backend

Codex CLI는 Python SDK와 별도로 설치한다. [공식 CLI 설치 안내](https://developers.openai.com/codex/cli)의
설치 방법 중 하나를 선택한 뒤 **각 사용자가 자신의 환경에서** [로그인](https://developers.openai.com/codex/auth)한다.
예를 들어 npm이 준비되어 있다면 다음과 같다.

```bash
npm install -g @openai/codex
codex login
```

`vlm.python` 환경에는 이미지 저장용 Pillow가 필요하다. 기존 `.[vlm]` 환경을 재사용하거나,
OpenAI SDK 없이 쓰려면 해당 Python에 이 repo와 Pillow만 설치한다.

```bash
python3.12 -m venv .venv-vlm
.venv-vlm/bin/python -m pip install -e . pillow
```

```yaml
# config.local.yaml — API 키 파일 없이 자신의 CLI 로그인 사용
vlm:
  enabled: true
  python: .venv-vlm/bin/python
  backend: codex_cli
  codex_executable: codex  # 별도 설치 경로 예: ./tools/codex
  model: gpt-6.1-sol
  asset_dir: runs/assets
```

`codex`는 PATH의 명령을 사용하며, `./tools/codex` 같은 경로는 해당 YAML 파일의 폴더 기준이다.
CLI의 기존 로그인 상태를 사용하고 `api_key_file`은 설정되어 있어도 무시한다.
`auth.json`·`.codex`의 인증자료를 repo로 복사하거나 다른 사용자에게 배포·Git에 포함하지 않는다.
CLI가 실패하면 해당 오류를 표시하며 **API backend로 자동 fallback하지 않는다**.

adapter는 이미지와 공개 catalog만 인식하는 **단발 `codex exec`**를 임시 폴더에서 실행한다.
개인 작업 설정을 적용하지 않고 shell·web 도구를 끈다. read-only sandbox 옵션을 사용하지만
이것이 모든 파일의 읽기를 차단한다는 의미는 아니다.
[공식 비대화형 실행 안내](https://developers.openai.com/codex/noninteractive)와
[CLI 옵션 안내](https://developers.openai.com/codex/cli/reference)를 참고한다.

### Claude / Anthropic API backend

이 backend는 **Claude API**이며 Claude Code CLI의 설치·로그인을 사용하는 기능이 아니다.
이미지와 JSON Schema 구조화 출력을 받는 Messages API를 사용한다.
[공식 vision 안내](https://platform.claude.com/docs/en/build-with-claude/vision)와
[Structured Outputs 안내](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)를 확인한다.

```bash
python3.12 -m venv .venv-vlm-anthropic
.venv-vlm-anthropic/bin/python -m pip install -e '.[vlm-anthropic]'
```

```yaml
# config.local.yaml — model은 자신의 계정에서 지원되는 실제 Claude 모델 ID로 교체
vlm:
  enabled: true
  python: .venv-vlm-anthropic/bin/python
  backend: anthropic_api
  model: YOUR_CLAUDE_VISION_STRUCTURED_OUTPUT_MODEL_ID
  api_key_file: .secrets/anthropic_api_key.txt
  asset_dir: runs/assets
```

파일을 쓸 때는 키 값을 Git 제외 로컬 파일에 둔다. `api_key_file: null`이면 worker 환경의 `ANTHROPIC_API_KEY`를 사용한다.
기존 OpenAI 키나 Codex/Claude Code 로그인 자료를 Anthropic API 인증으로 재사용하지 않는다.
실패 시 오류를 표시하며 다른 provider로 자동 fallback하거나 요청을 자동 재시도하지 않는다.
**fake SDK·native IPC 테스트**로 worker·인증 전달 계약을 검사했다. 또한 실제 Anthropic SDK에 로컬 mock HTTP를
연결해 이미지·구조화 요청 직렬화, 응답 파싱과 출력 중단 처리를 검사했다. 둘 다 외부 provider 호출은 없으며
실제 Claude API·계정·과금 호출의 성공을 검증한 것은 아니다.

### 세 backend의 공통 UI 흐름

모델에 전달하는 입력은 **현재 RGB·지시문·공개 asset/policy 텍스트 catalog**이며 depth·mesh·과거 참조 이미지·과거 pose는 보내지 않는다.

실행은 그대로 **`grasp-workbench run`**이다. 필요한 `sam3.enabled`, `sam3d.enabled`,
`foundationpose.enabled`는 각각 설정한다. VLM을 켜도 다른 모델 worker를 자동으로 설치·활성화하지 않는다.
기존 mesh의 pose만 구하려면 SAM3·FP가 필요하고, 새 mesh 생성까지 하려면 SAM3D도 필요하다.
Source grasp 후처리까지 쓰려면 `graspgenx.enabled`, 조합 선택까지 쓰려면 `grasp_selection.enabled`도 각각 설정한다.
각 worker를 외부에서 따로 실행해 연결하는 방식도 유지한다.

1. **VLM 사용**을 체크하고 물체 설명을 입력한 뒤 **VLM 인식 → Task 진행**을 누른다.
   설명이 비어 있으면 현재 보이는 물체를 최대 두 개 찾는다. 버튼 요청당 RGB 한 장만 인식하며
   **카메라 매 프레임마다 API·CLI 인식을 호출하지 않는다**.
2. VLM 결과의 A/B prompt는 `brown packing tape roll`, `yellow tuna can` 같은 **짧은 영어 명사구**다.
   긴 문장·상대 위치 설명 대신 물체 종류와 단순한 색상·재질로 SAM3를 실행한다.
3. **기존 asset ID가 선택되면** 저장 mesh를 재사용하되, 현재 SAM3 후보의 새 RGB-D/mask로 FP를 등록한다.
   이 경로는 SAM3D를 건너뛰며 과거 pose를 가져오지 않는다.
4. **새 물체이면** 현재 SAM3 후보로 YAML에서 설정한 SAM3D mesh를 만들고 `runs/assets/`에 저장한 뒤 FP를 시작한다.
   기본값은 실측 depth이며 `monocular`를 명시하면 VLM/Task 경로에도 그 설정을 사용한다.
   생성 입력·최초 FP 등록에는 같은 원본 RGB-D/mask를 사용하고, 이후 수신 카메라 프레임으로 tracking한다.
5. FP가 완료되면 활성화된 기능 범위에서 **Task Source의 grasp cache 조회/생성 → 조합 선택**으로 이어진다.
   Target은 pose 참조 대상이며 Source grasp를 자동 생성한다. 배정된 물체·팔이 불명확하면 A/B 순서로 채우지 않는다.
6. FP를 끄면 버튼은 **VLM 인식 → prompt**로 표시되며 prompt·semantic 결과를 반영한다. mesh 생성·FP 자동 경로는 실행하지 않는다.
   **VLM 사용** 체크를 끄면 기존 수동 prompt·후보·mesh·Start/Stop 버튼으로 사용한다.

VLM 요청이나 아래의 명시적인 Task 진행에서만 후속 단계를 연결한다. 수동 Generate·Start/Stop 버튼의 동작은 유지한다.
VLM bbox는 현재 SAM3 후보 선택을 돕는 정보이지 mask·6D pose 측정이 아니다.
catalog 이름·prompt를 보고 제안한 asset ID도 **같은 실물 물체라는 보장이나 형상 정확도 검증이 아니다**.

### Task / 팔 / 제약 semantic 결과

OpenAI API·Codex CLI·Anthropic API는 같은 구조로 최대 두 물체의 `observation_id`와 다음 의미 정보를 반환한다.

- **Task Source/Target**: 움직이거나 잡을 물체와 참조 대상의 현재 관측 ID 및 `intent`.
  관측 ID는 이번 요청 안의 참조이며 저장 mesh의 asset ID·물리적 tracking ID와 다르다.
- **Task type / policy 선택**: `task_type`, 의도 출처(`explicit`/`inferred`/`unclear`), 선택적인 `policy_id`와 이유.
- **Coordination**: `arm_mode`(`single`/`dual`/`unknown`), 팔별 관측 ID·역할·`grasp_region`의 `assignments`,
  참조/이동 frame·관측 ID·상대 동작·`phases`·`reference_state`·설명을 담은 `constraints`.

**A/B는 물체 슬롯이지 좌/우 팔이 아니다.** 팔 배정은 명시적인 observation ID로 표현하며 두 팔이 같은 물체를
참조할 수도 있다. 불명확한 팔·대상·제약은 unknown/null/빈 목록으로 남기고 A/B 순서로 임의 배정하지 않는다.
Task intent·잡을 부위·상대 제약은 **언어적 의미 정보**이지 수치 pose·관절 제약·접촉/부착 상태 측정이나 실행 허가가 아니다.

UI의 **Task / 팔 / 제약 / policy 상세**는 읽기 전용 JSON이며 별도 **Source/Target 편집기**로 현재 관측 ID를 바꿀 수 있다.
Source와 Target은 서로 다른 관측이며 Target 없음도 가능하다. 역할 수정은 이전 semantic coordination을 해제하며
관측 ID를 서로 바꾸거나 팔 배정을 재사용하지 않는다. 이후 **수동 left/right/dual 선택** 또는 **VLM 재분석**으로 팔 정보를 정한다.
**Pose · Grasp 진행**은 선택한 역할의 pose를 준비하고 Source grasp cache 조회/생성·선택을 명시적으로 요청한다.
worker가 꺼져 있으면 해당 기능을 자동 활성화하지 않는다. 관측 ID를 로컬 `instance_id`에 연결하고,
가능하면 해당 물체의 geometry·FP session/pose·선택 grasp 참조를 함께 기록한다.
지시문을 편집해도 마지막 semantic 판단을 지우거나 새 지시문에 맞게 임의 수정하지 않으며 **재분석 필요**로 표시한다.
수동 prompt·후보·mesh 변경 등으로 기존 관측 연결이 해제된 경우에도 이전 의미 결과와 현재 연결 상태를 구분한다.

### 선택적인 policy catalog

공유 기본값 `vlm.policy_catalog: null`에는 policy·모델 checkpoint를 요구하지 않는다.
사용하려면 로컬 JSON 경로만 설정한다. `session_dir`는 저장 대화상자의 기본 폴더다.

```yaml
# config.local.yaml — 선택 기능이며 실행 모델을 연결하는 설정이 아님
vlm:
  policy_catalog: /path/to/policies.json
  session_dir: runs/tasks
```

JSON은 다음과 같은 항목의 목록이다. 예시의 training asset은 정책 설명용 참조이며 현재 관측 ID가 아니다.

```json
[
  {
    "policy_id": "example-pour",
    "display_name": "Pour example",
    "task_type": "pour",
    "description": "Move the source toward the target to pour",
    "training_source": {"asset_id": "example_source", "name": "cup"},
    "training_target": {"asset_id": "example_target", "name": "bowl"},
    "pose_convention": "target_from_source",
    "transfer_note": "Unverified semantic transfer reference",
    "checkpoint": "checkpoints/example.pt"
  }
]
```

`checkpoint`는 **JSON 파일의 폴더 기준**으로 해석하는 로컬 표시·export용 참조다. 앱은 checkpoint를 로드하거나
policy를 실행하지 않는다. provider·IPC 요청에는 `policy_id`, `display_name`, `task_type`, `description`,
`training_source`, `training_target`, `pose_convention`, `transfer_note`만 보내며 training 참조도 `asset_id`·`name`만 남긴다.
checkpoint·개인 경로·추가 sidecar 정보는 보내지 않는다. 목록이 없거나 맞는 정책이 없으면 선택은 null로 유지한다.

### Session 저장·불러오기·요청 export

- **Session 저장**: 지시문·마지막 semantic 판단·관측→instance 연결·geometry/FP/grasp 참조와 provenance를 JSON으로 저장한다.
- **Session 불러오기**: 저장된 snapshot을 표시한다. **현재 관측·현재 pose가 아니며**, 이전 A/B 활성 연결을 해제하고
  재분석 필요로 표시한다. 불러오기만으로 SAM3·FP·grasp 추론이나 로봇 명령을 시작하지 않는다.
- **실행 요청 export**: observation ID를 로컬 instance ID에 연결한 참조를 JSON으로 내보낸다.
  `execution: "request_only"`, `robot_command_published: false`이며 그 자체가 실행·허가·IK 성공을 의미하지 않는다.

별도 `grasp_ik_worker` 대신 아래의 선택적인 `grasp_selection` worker에서 후보 조합·base 좌표 변환·endpoint IK를 처리한다.
semantic 저장·export와 실제 실행은 별개이며, snapshot이나 언어적 constraint를 수치 운동 제약으로 자동 변환하지 않는다.

### 새 mesh를 다음 실행에도 기억하기

`assets.py`의 `AssetStore`는 기존 `assets/meshes/catalog.json`과 새 저장 asset을 합쳐 목록을 만든다.
VLM에 전달하는 공개 catalog에는 **`asset_id`·`name`·`prompt`만** 있으며 로컬 mesh 경로·참조 이미지·pose는 없다.
새 mesh는 `runs/assets/<asset_id>/object.ply`(texture 사용 시 `object.glb`)와 `manifest.json`에 저장한다. manifest에는 이름·prompt·단위·scale을
보존하고, 다음 실행에서도 저장 mesh로 **새 SAM3 관측과 FP 등록**을 수행한다.
이 저장소는 `runs/meshes`의 입력별 추론 cache와 별개이므로 추론 cache를 지워도 보존된다.
`vlm.asset_dir`를 삭제하면 새로 저장한 물체의 기억도 삭제된다. repo에 보관한 기존 mesh catalog는 그대로다.

### 다른 VLM을 붙이는 최소 확장

현재 선택지는 위 세 backend다. 다른 provider를 쓰려면 새 framework나 기능별 worker를 먼저 만들 필요는 없다.

1. `models/`에 `start()`·`predict(request)`·`close()` adapter를 추가한다. `predict`는 공통
   `VlmPrediction(objects, task_binding, policy_selection, coordination)`으로 정규화하고 기존 schema·검증을 재사용한다.
2. `config.py`, `main.py`의 `--vlm-backend`, VLM worker CLI의 backend 선택지와
   `workers/worker_vlm.py`의 predictor 생성 분기에 등록한다.
   필요 SDK는 선택 extra로 추가하고 해당 adapter에서만 import한다. `model`·`api_key_file` 같은 기존 flat 설정을 사용한다.
3. 기존 VLM worker·`VlmRequest`/`VlmResult`·iceoryx2 채널과 UI·후속 분기를 그대로 사용한다.
   키는 worker 내부에서만 읽고 키/SDK 원문 오류를 IPC·로그에 넣지 않는다.

별도 실행 환경의 외부 프로세스가 정말 필요한 경우에만 같은 `/vlm/requests`·`/vlm/results` 계약에 맞춘
worker를 연결한다. HTTP provider를 바꾼다는 이유만으로 새 worker나 `providers:` 설정 체계를 추가하지 않는다.

## SAM3D: 선택한 후보로 mesh 생성

먼저 사용하는 GPU에 맞춰 [공식 설치 안내](third_party/sam-3d-objects/doc/setup.md)에 따라
SAM3D 환경·가중치를 준비한다. submodule은 `f91db411c50efee93d8db7aeb323885650f6f722`로 고정했다.
그 환경에서 이 repo의 `python -m pip install -e '.[sam3d]'`를 추가한다. 이는 **얇은 adapter 의존성만**
설치하며 공식 모델의 전체 CUDA/native/backend 환경을 대신 설치하지 않는다.

일반적인 A/B 흐름은 `config.local.yaml`에서 SAM3와 SAM3D를 **각각 켠 뒤** 실행한다.
`sam3d.enabled`는 SAM3 모델 worker까지 자동으로 켜지 않으므로, 외부 SAM3 worker를 쓸 때도 구분할 수 있다.
아래 경로는 각자 설치한 환경·파일로 바꾼다.

```yaml
# config.local.yaml
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
  depth_source: depth      # 단일 RGB 예측은 monocular
  texture_baking: false    # UV/material GLB가 필요하면 true
```

```bash
.venv/bin/grasp-workbench run
```

1. A/B prompt를 입력하고 **Apply**한다. 후보가 하나면 자동 선택하고, 여러 개면 dropdown에서 직접 고른다.
2. 선택한 **원본 RGB-D frame과 mask 복사본**이 고정되고 해당 sequence가 표시된다. 이후 영상·분할이
   갱신되어도 생성 입력을 몰래 바꾸지 않는다. A/B의 **Generate mesh** 버튼으로 각각 요청한다.
3. UI는 한 번에 한 요청만 전송하고 완료·오류를 받은 뒤 다음 요청을 보낸다. A/B를 빠르게 눌러도
   최신 요청 하나로 덮어쓰지 않는다. 상태는 Generating/Done/Error이며 임의의 진행률은 만들지 않는다.
4. 생성 모드는 **YAML의 `sam3d.depth_source`·`sam3d.texture_baking`**으로 정한다. UI에는 현재 설정만 표시한다.
   기본값은 실측 depth·texture 꺼짐으로 기존 vertex-color PLY 경로를 유지한다.
   `depth_source: monocular`, `texture_baking: true`로 바꾸면 수동·VLM·Task 요청 모두 같은 설정을 사용한다.
   실행 명령은 계속 **`grasp-workbench run`**이며 VLM 경로에서 실측 depth로 강제 변경하지 않는다.
5. 오른쪽 A/B mesh 선택기로 **완료된 asset 하나씩** 본다. 기본 실측 depth 경로의 좌표는 **asset 로컬 metre 단위**다.
   A/B를 같은 world 좌표로 정렬했다고 해석하지 않는다. 큰 mesh는 수신 thread에서 표면을
   유지하는 표시용 간소화를 수행한다. preview는 평균 RGB와 shading으로 표시하며 정확한 texture가 아니다.
   IPC·cache·PLY의 전체 geometry와 vertex color는 그대로 보존한다. Texture baking을 고르면
   실제 UV/material이 있는 GLB를 cache·영속 asset에 보존하며 단순 vertex color를 texture라고 부르지 않는다.
   Matplotlib preview의 평균색/shading 표시는 원본 GLB texture 렌더링과 구분한다.

- asset UUID는 선택한 고정 입력에 연결되며 같은 선택의 재요청은 같은 UUID를 쓴다. worker는
  **같은 asset ID + 정확히 같은 입력 fingerprint**일 때만 저장 cache를 재사용한다. Apply·prompt·후보
  변경은 연결을 해제한다. 다음 실행에서 같은 실물 물체를 자동 식별·매칭하는 기능은 아니다.
- 요청·raw vertices/faces/colors 결과는 **iceoryx2**로 전달한다. 디스크의 NPZ는 cache, binary PLY 또는 texture GLB는
  저장 mesh이며 경로는 결과 metadata다. `depth_source`·`texture_baking`도 입력 fingerprint·cache에 반영한다.
- 기본 생성 adapter는 RGB+mask와 **같은 frame의 실측 depth·intrinsics로 만든 pointmap**을 공식 pipeline에 전달한다.
  선택적인 `depth_source: monocular`는 MoGe 예측 pointmap 경로이며 **실측 metric scale을 주장하지 않는다**.
  이 결과의 `length_unit`는 `monocular_predicted_scale`로 보존한다. 필요한 MoGe·texture backend/가중치는 사용자가 준비한다.
  이전 repo와 같은 방식으로 GLB 표시 축 변환을 해제하고 출력 scale을 정점에 한 번 적용한다.
  mesh와 별도의 OpenCV `camera_from_object`를 IPC·cache에 보존하며 실측 depth 경로만 `length_unit: metre`다.
  초기 pose는 생성 원본 frame 기준이며 실시간 tracking 결과가 아니다. 신규 mesh는 **scale → m = 1.0**으로
  FP에 전달하되 monocular의 scale=1은 예측 scale이 이미 적용됐다는 뜻이지 실측 metre 보장이 아니다.
  기존 비보정 cache는 구분해 재생성한다. 실측 입력을 사용해도 형상·치수 정확도를 보장하지 않는다.

## FoundationPose: mesh 등록 후 연속 tracking

사용자는 [공식 설치 안내](https://github.com/NVlabs/FoundationPose/tree/a1b694b83e633c2cb6115b9063d940a687759392)를 따라
독립 Python 환경·native extension·`third_party/FoundationPose/weights/`를 준비한다.
그 환경에 `python -m pip install -e '.[foundationpose]'`를 추가한다. 이 extra는 얇은 adapter 의존성만 설치하며
전체 GPU backend 설치를 대신하지 않는다. upstream의 [LICENSE](third_party/FoundationPose/LICENSE)를 유지한다.

```yaml
# config.local.yaml — 경로는 각자 설치한 환경으로 설정
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

실행은 그대로 **`grasp-workbench run`**이다. FP를 켜도 SAM3 모델 worker가 자동으로 켜지지는 않으므로
위처럼 각각 켜거나 외부 SAM3 worker를 연결한다. SAM3D를 실행할 필요 없이 기존 mesh를 사용할 수 있다.

1. A/B prompt를 Apply하고 SAM3 후보를 고른다.
2. 각 슬롯의 **이름으로 mesh 선택** 목록에서 저장 물체를 고른다. 이름을 선택하면 기존 최적화 mesh 경로와
   scale이 입력된다. **Browse**나 직접 경로 입력도 유지한다. 기존 보관 mesh는
   원본 단위 선언에 따라 `1.0`을 사용한다. 기본 실측 depth SAM3D mesh는 metre mesh이므로 완료 경로와
   `scale → m = 1.0`을 빈 mesh 입력칸에 자동 입력한다. monocular 예측 scale은 앞의 단위 설명을 따른다.
   외부 mesh는 사용자가 해당 파일의 단위를 설정한다.
3. VLM을 사용하지 않는 수동 흐름에서는 사용자가 버튼으로 분기한다. 파일 존재 여부나 물체 identity로 자동 판정하지 않는다.
   - **기존 mesh로 Start**: 선택한 저장 mesh와 현재 SAM3 후보의 RGB-D/mask로 FP를 시작한다.
   - **mesh 없음 · 생성 후 Start**: 선택한 후보로 SAM3D mesh를 생성한 뒤, 완료 경로를 FP 입력에 넣고
     자동 시작한다. 이 버튼은 SAM3D와 FP가 모두 켜진 경우 표시한다. 생성 입력으로 고정한 RGB-D/mask를
     FP 최초 등록에도 사용하며, 이후 카메라 worker에서 수신한 RGB-D로 연속 tracking한다.
   - 기존 **Generate mesh**는 mesh만 생성하며 FP를 자동 시작하지 않는다.
   생성 후 Start는 생성 경로와 **scale → m = 1.0**으로 교체해 scale 중복 적용을 막는다. 기본 FP 초기화는
   mask register이며 아래의 seed refinement는 별도로 선택한다. 생성 대기 중 **Stop**을 누르면 완료 후 FP 자동
   시작만 취소하며 생성 mesh는 보존한다.
   A/B session은 별도이며 scorer/refiner는 FP worker 내부에서 공유한다.
4. UI 영상은 **최신 카메라 RGB + 최신 SAM3 mask·box + 슬롯별 최신 A/B pose 축**을 합성한다.
   pose 축 길이는 5cm이며 **X=빨강 / Y=초록 / Z=파랑**, 물체 ID를 함께 표시한다.
   RGB는 FP 추론 완료를 기다리지 않으며, FP가 처리한 과거 RGB로 화면을 되돌리지 않는다.
   한 poll에서는 새 결과를 모두 반영한 뒤 한 번만 합성한다. 하단에 카메라·FP pose·SAM3 overlay sequence를
   각각 표시하며 별도 시간 매칭·보간은 하지 않는다. overlay 갱신 속도는 각 모델의 처리 속도 그대로다.
   오른쪽 3D plot에는 **카메라 기준 mesh·XYZ(m)·회전축**을 함께 표시한다. 원본 object 좌표의 정점에
   입력 scale을 한 번 적용한 뒤 `v_camera = R @ v_scaled + t`로 변환한다. UI는 FP와 같은 GLB/OBJ/PLY
   로딩·scene 변환을 사용한다. 새 SAM3D mesh는 기존 preview를 재사용한다.
   물체당 최대 약 **2,000개 삼각형**의 표시용 mesh와 plot artist를 재사용하며 원본 FP 입력은 변경하지 않는다.
   현재 카메라 pose가 없는 registering/recovering 및 **Stop**에는 해당 mesh·축을 제거한다.
   영상에는 축만 겹치며 mesh 전체의 2D 투영이나 base 좌표 변환은 하지 않는다.

### 현재/SAM3D pose seed로 refinement

기존 mask register 외에 **SAM3D pose seed** 또는 **현재 / 저장 FP seed → Seed → FP refinement**를 선택할 수 있다.
같은 mesh·scale의 pose와 그 pose를 얻은 정확한 원본 RGB-D를 사용하며 최신 카메라 frame으로 seed 입력을 바꾸지 않는다.
seed 모드는 `initialization: seed`와 `initial_pose`를 전송한다. SAM3D seed에는 해당 생성 원본 mask를 함께 보내고,
현재/저장 FP seed는 `mask: null`도 허용한다. 기본 `register` 모드는 계속 mask를 요구한다.
FoundationPose 내부 centered-mesh frame과 원본 object frame은 다음 식으로 구분한다.

```text
pose_last = camera_from_object @ inverse(get_tf_to_centered_mesh())
```

이후 같은 원본 frame에서 `track_one` refinement를 수행하고 정상 연속 tracking으로 이어간다.
과거 seed 목록·시간 gate·자동 seed 탐색을 추가한 것은 아니다.

연속 추적에는 이전 UI의 **SAM3 후보 연결·pose 검증·재등록** 경로를 사용한다.
- Start 때 선택한 mask bbox를 기준으로 이후 후보를 bbox IoU 우선(동률 score)으로 연결한다.
  기존 IoU·면적비·pending 후보 확인 방식을 사용하며 UI의 전체 후보 순서는 변경하지 않는다.
- 새로운 SAM3 결과마다 pose origin의 투영 위치가 bbox ±40px 안인지, 유효 depth가 16픽셀 이상이면
  pose z가 mask depth min/max ±0.25m 안인지 검사한다. 연속 **검증 불일치 5회**면 recovering으로 전환한다.
  이는 카메라 5프레임이 아니라 SAM3 증거 5회이며, 미검출만으로 정상 FP를 중지하지 않는다.
- 기존 SE(3) 유효성·translation jump 0.25m 검사를 유지한다. 추적 예외도 recovering으로 표시하며
  이전 pose를 계속 표시하지 않는다. 새 SAM3 원본 RGB-D/mask로 재등록한 결과를 다시 검증해 tracking으로 복귀한다.
  A/B 상태는 독립이고 UI에 불일치 횟수·recovering 사유가 표시된다.
- Start의 prompt revision에 대응하는 SAM3 결과만 사용한다. prompt를 새로 Apply한 뒤 검증 대상을 바꾸려면
  해당 슬롯의 Start로 새 session을 시작한다. ROI box fallback은 아직 연결하지 않았다.

pose history·시간차 gate·cooldown·별도 registration 프로세스·EKF는 추가하지 않았다.
한 프로세스에서 순차 실행하므로 A를 등록하는 동안 B tracking도 기다린다.
제어·등록 RGB-D/mask·pose 결과는 iceoryx2로 전달하며 mesh 파일은 사용자가 선택한 저장 asset이다.

### 이전에 생성한 mesh

기존 FFW의 저장 물체 21개(원본·최적화 GLB 42개)와 Table/Leg OBJ 3개를
[`assets/meshes/`](assets/meshes/README.md)에 복사했다. 이름·상대 경로·단위·SHA-256은
[`catalog.json`](assets/meshes/catalog.json)에 있다. 원본은 보존했으며 총 약 289.8 MiB다.
FP의 A/B 이름 목록에서 물체 21개와 Table/Leg 3개를 선택할 수 있다. UUID 경로·mesh 파일 바이트는 유지하며
표시 이름은 catalog에 있다. 선택한 mesh로 새 pose를 구하며 과거 pose를 가져오지는 않는다.

## GraspGenX: mesh에서 grasp 후보 생성·선택

독립 GraspGenX worker에 mesh 경로·단위를 보내 **object-local grasp 후보와 score**를 받는다.
현재 pose 없이도 사용할 수 있으며 SAM3·SAM3D·FP·VLM을 강제로 켜지 않는다.
A/B는 **물체 슬롯**이지 왼팔·오른팔이 아니다. 한팔·양팔 조합 선택과 선택적인 로봇 기준 도달 가능성 확인은
별도의 `grasp_selection` worker가 담당하며, 후보 생성만 사용할 때 이 worker는 필요 없다.

### 선택적인 GPU 환경

[공식 GraspGenX 설치 안내](https://github.com/NVlabs/GraspGenX)를 따라 GPU 환경을 준비한다.
이 repo의 원본 submodule revision은 `b9429097728cb1c430dd78b92edf17ba318aad03`이다.
upstream은 Python **3.10/3.11** 환경의 설치를 안내하므로, 준비된 해당 Python 환경에서
repo의 기본 전송 의존성과 얇은 adapter extra를 추가하는 방식을 권장한다.

```bash
# 공식 GraspGenX 설치를 완료한 환경에서, 이 repo 루트 기준
/path/to/graspgenx-env/bin/python -m pip install -e '.[graspgenx]'
```

이 extra는 전체 PyTorch/CUDA·native 모델 환경을 대신 설치하지 않는다.
공식 패키지는 첫 import 때 checkpoint·gripper description을 다운로드할 수 있다.
가중치·gripper descriptor·경로·네트워크 설정은 사용자가 준비하며 공유 repo에 포함하지 않는다.
공식 submodule의 SDK를 사용하되 upstream 소스는 수정하지 않는다. 이전 앱 adapter를 import하지 않으며 FFW·ROS 환경을 요구하지 않는다.

```yaml
# config.local.yaml — 각자 준비한 환경·파일로 설정
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

일반 실행은 그대로 **`grasp-workbench run`**이다. mesh만 다루는 독립 사용도 가능하다.
별도 `worker_graspgenx`를 같은 service로 실행한 뒤 **`grasp-workbench ui --graspgenx`**로 연결하면
카메라·SAM3·FP 결과 없이 파일에서 후보를 생성·표시할 수 있다.

### UI와 좌표

1. A/B의 이름 목록·Browse·직접 경로로 mesh를 고르고 **scale → m**를 설정한다.
   활성 FP session이 있으면 그 session의 실제 mesh·scale을 사용한다.
2. **Generate grasp**는 해당 geometry의 저장 후보를 재사용하거나 없으면 생성한다.
   **Regenerate**는 cache를 건너뛰고 같은 설정으로 다시 추론한다. seed가 고정되어 있으므로 결과가 같을 수 있다.
   수동 생성은 이 버튼으로 요청한다. 명시적인 VLM/Task 진행에서는 FP 완료 뒤 Source grasp cache 조회/생성을 연결하며,
   단순히 FP pose가 움직였다고 모델을 다시 호출하지 않는다.
3. A/B 후보 dropdown에서 index·score를 보고 선택한다. FP가 없으면 오른쪽 plot에 선택 물체의
   **object-local mesh·gripper·grasp 축**을 표시한다. gripper visual descriptor가 없으면 축만 표시한다.
4. FP가 있으면 `camera_from_gripper = camera_from_object @ object_from_canonical_gripper`로 합성한다.
   최신 카메라 RGB에는 선택 grasp 축을, 오른쪽 plot에는 mesh·pose 축·gripper를 함께 표시한다.
   일반 object-local 후보는 물체 pose에 맞춰 표시 변환만 바꾼다. 아래의 single 선택 결과는 별도의 effective-pose lock을 사용하며
   원본 FP 추적과 최신 RGB 갱신 경로는 유지한다.

모델 score는 **IK 성공·장면 충돌 없음·실행 허가를 뜻하지 않는다**. 다른 물체·테이블·로봇과의 장면 충돌이나
양팔 동시 실행 가능성을 검사한 후보라고 해석하지 않는다.

### 추론 설정과 저장

이전 worker의 sampling·seed·다양성 선택 기본값을 유지한다. `settings: {}`는 seed **42**,
`num_grasps: 1000`, `topk_num_grasps: 100`, `min_grasps: 40`, `max_tries: 6` 등의 기존 기본값을 사용한다.
score를 시작점으로 SE(3) 거리의 다양성을 선택하며 기존 기준은 translation **0.01m**·rotation **15°**다.
여기서 `max_tries`는 기존 모델 추론의 옵션이지 새 worker의 자동 retry 정책이 아니다.
기본값이 아닌 `position_clip_sample_range`의 호환 처리는 adapter의 sampler instance에서만 적용하며 upstream 소스는 수정하지 않는다.

cache identity는 asset ID·**mesh SHA-256·scale·gripper/descriptor·model·추론 설정**으로 구성한다.
현재 카메라 pose는 포함하지 않는다. `runs/grasps/<cache_key>/`에 선택 후보 `grasps.npz`, 원본 후보·sample points의
`candidate_pool.npz`, provenance·좌표 convention의 `metadata.json`을 저장하고 같은 입력의 다음 요청에서 재사용한다.
요청·raw float64 `N×4×4` grasp와 `N` score는 iceoryx2로 전달한다. UI receiver는 A/B 요청을 한 번에 하나씩 전송하고
완료·오류 후 다음 요청을 처리한다. pose history·시간 gate·새 자동 retry/fallback은 추가하지 않는다.

### 기존 Isaac YAML bundle 읽기

A/B의 저장 bundle 불러오기는 새 NPZ뿐 아니라 **원본 `metadata.json`이 함께 있는 Isaac grasp YAML**도 읽는다.
metadata의 mesh hash·scale·gripper/model·설정 identity와 candidate hash를 유지하며 confidence·원래 후보 ID를 사용한다.
YAML 파일만 보고 빠진 provenance·raw candidate pool을 꾸며 만들지 않는다. 기존 파일을 덮어쓰거나
옛 YAML 저장 형식으로 새 결과를 쓰지 않으며, 새 추론 저장 형식은 계속 이 repo의 NPZ·JSON이다.

## Grasp 선택: 한팔·양팔 조합과 선택적인 도달 가능성 확인

`grasp_selection`은 **이미 생성한 grasp bundle**에서 후보를 고르는 독립 worker다.
공유 기본값은 `enabled: false`, `python: null`, `robot_config: null`, `settings: {}`이며 일반 실행 명령은 바뀌지 않는다.
카메라·VLM·SAM3·FP·GraspGenX 모델을 강제로 켜지 않는다. 직접 선택 요청은 기존 bundle을 사용하며,
명시적인 VLM/Task 진행의 Source cache 조회/생성은 앞선 GraspGenX 단계에서 처리한다.

```yaml
# config.local.yaml — 로봇 없이 후보 선택부터 확인
grasp_selection:
  enabled: true
  robot_config: null
  settings: {}
```

### UI와 후보 조합

- A/B에서 먼저 grasp 후보를 생성하고 **수동 single · left/right** 또는 **수동 dual · 같은 물체**와 대상 A/B를 고른다.
  VLM이 켜져 있으면 현재 observation ID의 `assignments`를 사용할 수도 있다. A/B 순서를 팔 배정으로 바꾸지 않는다.
- **grasp 선택 · 로봇 기준 확인**을 누르면 해당 geometry·bundle·현재 물체 pose의 snapshot을 보낸다.
  결과 dropdown에서 조합·score·IK 상태를 확인하고 선택 조합을 영상/plot에 표시한다.
  single 결과는 선택된 effective object/gripper pose를 고정하고 **새로고침 · 다시 평가** 또는 명시적인 후보 재선택 때 갱신한다.
  원본 FP pose를 정지시키거나 과거 pose history로 필터링하는 것이 아니다.
  pose 이동이나 semantic 갱신만으로 선택 요청·후보 생성을 반복하지 않는다.
- 같은 물체를 양팔로 잡을 때는 이전의 **서로 다른 후보의 ordered pair·gripper 대칭 변형**과 COM 균형 점수를 유지한다.
  COM은 균일 밀도 volume 또는 bounding-box 중심의 proxy이며, 로봇 기준 물체 pose로 중력 방향을 알 수 있을 때 사용한다.
  접촉선과 COM의 투영 균형은 물리적 안정성·force closure 보장이 아니다.
- 서로 다른 물체의 dual assignments도 표현한다. 이 경로는 두 모델 score의 **평균**을 쓰고,
  서로 다른 물체에 하나의 COM 균형 점수를 적용하지 않는다. 이는 기존 same-object 흐름에 더한 요청 표현 범위다.
  `grasp_region`·상대 제약은 의미 정보로 남으며 접촉·운동 제약으로 자동 변환하지 않는다.
- 로봇 연결이 없으면 **canonical 후보 선택·표시**를 유지한다. base transform·joint seed·TCP 보정을 임의로 만들지 않으며
  base pose·관절 해·IK 성공을 꾸며 표시하지 않는다.

`settings: {}`의 `top_k: 32`, `pregrasp_distance_m: 0.15`, `grasp_retreat_distance_m: 0.0`,
`grasp_advance_distance_m: 0.0`은 이전 `ffw_system/configs/table_leg_grasp.json` 예제에서 가져온 값이다.
모든 로봇·gripper에 맞는 보편 기본값이 아니므로 자신의 모델·TCP에 맞게 설정한다.
로봇이 연결되면 현재 관절/FK를 기준으로 후보를 확인하고 endpoint IK 결과를 순위에 반영한다.
single 기본 `preference: top_down`은 실제 base-frame pregrasp→grasp 접근 방향의 top-down **50%**·모델 score **30%**·
도달 가능성 **20%**를 사용한다. IK 전에는 top-down/model을 정규화해 정렬하고 기존 **2cm·10°** 사전 다양성 선택을 적용한다.
`settings: {preference: legacy}`로 이전 geometry/도달 가능성 점수 경로를 선택할 수 있다.
base/TCP가 없는 no-robot 경로에서 top-down 방향·IK score를 임의 생성하지 않는다.
pose history·시간 gate·새 retry/fallback 정책은 추가하지 않는다.

### 선택적인 robot adapter와 좌표

상태 수신·현재 FK·camera calibration·endpoint IK는 **같은 grasp_selection worker 안**의 adapter가 담당한다.
일반 adapter는 로컬 robot YAML의 `backend: module:factory`로 지정하며 factory는
`factory(config, config_dir=...)`로 호출된다. 각 사용자가 자신의 상태/모델/solver 연결을 제공한다.

```yaml
# config.local.yaml — 선택적인 로봇 연결
grasp_selection:
  enabled: true
  python: /path/to/robot-env/bin/python
  robot_config: /path/to/robot.yaml
  settings: {}
```

좌표 합성은 다음 순서다. 여기서 canonical은 GraspGenX의 canonical gripper frame이며 TCP·solver tool frame과 구분한다.

```text
base_from_object = base_from_camera @ camera_from_object
base_from_tcp = base_from_camera @ camera_from_object @ object_from_canonical @ canonical_from_tcp
```

`object_from_canonical`은 bundle의 `object_from_gripper`이며, pregrasp/retreat/advance 설정은 각각의 frame에 적용한다.
TCP 목표를 solver에 넘길 때는 해당 solver tool frame으로 별도 변환한다. 두 frame이 같다고 가정하지 않는다.
확인 범위는 **pregrasp/grasp endpoint의 도달 가능성**이다. adapter가 검사한 self-collision 여부는 결과에 표시하지만,
장면 충돌·경로·trajectory·실제 이동 가능성을 확인한 것으로 해석하지 않는다. actuator/이동 명령은 발행하지 않는다.

#### 선택적인 FFW 예제

[`examples/robot_ffw.yaml`](examples/robot_ffw.yaml)은 `backend: ffw_sg2`의 선택적 예제다.
FFW 앱 저장소를 import하거나 필수로 설치하지 않는다. 이 예제를 쓸 때만 별도 Python 환경에 독립 패키지
`robot-worker-runtime[iceoryx2]`, `ffw-robot-common`, `motion-planning-system[curobo]`와 해당 GPU 환경을 준비한다.
runtime manifest·robot model assets·calibration 파일은 사용자가 제공한다. 예제의 `/path/to/`를 자신의 파일로 바꾸며
YAML 안의 상대 경로는 그 YAML의 폴더 기준이다.

calibration YAML은 `parent_frame`과 `transforms.<sensor_frame>.parent_from_sensor`를 사용한다.
`parent_frame: base_link`는 고정 base calibration이고, `head_link2`는 실측 head·lift 관절의 현재 FK와 합성한다.
실제 state publisher가 없으면 실측 상태가 연결된 것으로 간주하지 않는다. 공유 예제에 특정 calibration·관절값을 넣지 않는다.
이 adapter는 기존 solver의 IK 자원만 사용하며 robot driver·TrajOpt·world scene·trajectory 실행을 시작하지 않는다.

### 표시용 파지 관계 고정과 hand FK

base-frame 선택 결과에서 **파지 관계 고정(표시용)**을 누르면 선택 당시의 `object_from_tcp` 관계를 저장한다.
같은 선택 worker가 실측 hand FK와 state sequence를 읽고 다음 식으로 base-frame 물체 pose를 갱신한다.

```text
base_from_object_fk = measured_base_from_tcp @ inverse(frozen_object_from_tcp)
```

이는 **계획한 파지 관계를 유지한다고 가정한 read-only 표시**이지 접촉·gripper 폐쇄·미끄러짐을 측정한 결과가 아니다.
실측 state가 없으면 waiting 상태이며 명령 관절값으로 대신 채우지 않는다. **파지 관계 해제**로 명시적으로 끝낸다.
같은 worker의 결과 stream을 사용하므로 별도 FK worker는 추가하지 않는다.

원본 camera/FP 결과는 덮어쓰지 않고 base_link 3D 표시 모드와 구분한다. 다음 **명시적인 다시 평가** 요청에서만
동일 `instance_id`의 FK pose를 base-frame 입력으로 사용할 수 있다. 자동 재평가·카메라 pose 위조·로봇 명령은 하지 않는다.

### 이번 이식의 범위

수동/Task 처리·snapshot/refine·기존 cache 읽기·후보 선택·선택적인 FK/endpoint IK를 기존 worker에 연결했다.
원본 `object_robot_ui`의 6개 탭을 동일하게 복제한 앱이라는 뜻은 아니다. 데이터 수집·학습·새 제약 연구·
학습 policy 추론/실행·actuator 명령·로봇 이동·TrajOpt·world scene·trajectory는 이식 범위에서 제외한다.

## 전송 계약

- uint8 **RGB**, float32 **metre depth**를 원본 해상도로 함께 보낸다.
- camera ID, stream ID, sequence, SDK image timestamp/clock domain, host 수신 monotonic 시간,
  intrinsics와 depth 원본 시간·단위를 함께 보존한다.
- ZED `IMAGE` timestamp는 SDK가 제공한 UNIX ns(image가 PC memory에 준비된 시점)이며,
  sensor exposure 시각이나 host monotonic 시각으로 바꾸어 해석하지 않는다.
- ZED는 left image에 정렬된 SDK depth, RealSense는 color에 align한 depth를 사용한다.
- NaN/inf/0 등 invalid depth를 임의의 측정값으로 바꾸지 않는다.
- 메시지는 길이 prefix + 작은 JSON metadata + raw RGB/depth이다. JPEG/PNG·base64·pickle을 사용하지 않는다.
- 프로세스 간 프레임 전달은 **iceoryx2만** 사용한다. 별도 수동 SHM·Queue·Pipe는 없다.
  iceoryx2 자체는 내부적으로 공유 메모리를 사용한다.
- 위 raw 전송 계약은 worker 간 IPC 기준이다. VLM의 API backend는 현재 RGB를 PNG/base64 입력으로,
  CLI backend는 임시 PNG 파일로 변환한다. 이는 외부 모델 호출 adapter의 입력이며
  UI·카메라·SAM3·FP·VLM worker 사이의 raw iceoryx2 전송은 바뀌지 않는다.
- SDK→발행 loan과 수신 loan→소유 배열의 복사는 있으며, end-to-end 무복사를 주장하지 않는다.
- 최신 프레임 확인용 작은 버퍼이며 느린 수신자의 모든 프레임 보존을 약속하지 않는다.
  여러 카메라 예제를 동시에 실행할 때는 각 설정의 `ipc.service`로 채널을 구분한다.
- grasp 선택은 같은 service 아래 `/grasp_selection/requests`, `/grasp_selection/results`에서
  coordination·geometry/bundle 참조·pose snapshot과 선택 결과를 JSON으로 주고받는다. 로봇 명령 채널은 아니다.

공식 예제: [iceoryx2 v0.9.3 dynamic data](https://github.com/eclipse-iceoryx/iceoryx2/tree/v0.9.3/examples/python/publish_subscribe_dynamic_data),
[ZED depth API](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/using-the-api),
[RealSense alignment](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py).

## 검증

```bash
python -m pytest tests -q
ruff check grasp_workbench tests
```

테스트는 fake SDK의 색상·단위·timestamp·해제와 실제 두 프로세스 iceoryx2 raw RGB-D 전달을 검사한다.
UI 의존성이 설치되어 있으면 offscreen RGB 색상·빈 3D 영역·종료 처리 및 별도 UI 프로세스의 실제
iceoryx2 구독도 검사한다. UI 테스트에서는 카메라 SDK를 열지 않는다.
SAM3D는 fake predictor와 실제 IPC로 요청 순서·고정 입력·cache·오류 후 계속 실행·mesh 표시를 검사한다.
SAM3D pointmap·scale·좌표 변환과 초기 pose, FP의 SAM3 후보 연결·검증·재등록도 단위 테스트한다.
별도 FP 프로세스와 실제 iceoryx2를 사용하는 fake backend 테스트로 A/B 독립 회복과 등록 원본을 확인한다.
VLM·asset 테스트는 단발 요청·공개 catalog·기존/새 mesh 분기와 다음 실행의 저장 mesh 재사용을 검사한다.
fake 결과를 실제 GPU 모델의 생성 품질 검증으로 해석하지 않는다.
실제 카메라 결과와 fake 테스트 결과는 구분해 기록한다. 카메라 목록 조회는 SDK 연결·캡처 성공을 의미하지 않는다.

### 2026-10-19 실제 확인

- 새 독립 `.venv`: Python 3.12.3, iceoryx2 0.9.3, ZED SDK 5.4.1 / pyzed 5.4.
  기존 FFW/IGRIS Python 환경이나 소스를 연결하지 않았다.
- UI 확장 전 테스트 **58 passed**, Ruff 통과. native iceoryx2 두 프로세스 raw 전송 포함.
- 실제 **ZED-M 17751856 / HD720@30**: 발행 32프레임, 별도 프로세스 수신 **30프레임**,
  RGB `720×1280×3` / metre depth `720×1280`; 종료 후 두 worker 정리.
- 장치가 낸 **self-calibration skipped (low texture/occlusion)** 경고와 depth 유효 픽셀 **0개**를
  그대로 기록했다. 전송 성공이지 depth 품질·정확도 검증이 아니다.
- SDK는 예제의 `PERFORMANCE` depth mode에 deprecation 경고도 냈다. 자동으로 다른 모델을
  다운로드하거나 모드를 바꾸지 않았으며, 사용자가 장치에 맞춰 예제를 설정한다.
- 로그: `runs/zed_iceoryx2_smoke.log` (Git 제외, 이미지 자체는 저장하지 않음).
- RealSense 예제는 fake SDK와 설치된 binding의 장치 목록까지만 확인했다. 연결된 RealSense가
  없어 실제 캡처는 검증하지 않았다.

### UI 추가 확인

- `.[test,ui]` 환경에서 **82 passed**, Ruff 통과. 실제 iceoryx2 별도 발행 프로세스 → UI worker
  구독, RGB 색상·이미지 소유권·빈 3D 축·수신 thread 종료·예외 시 정리를 검사했다.
- 실제 ZED → iceoryx2 → 별도 Qt UI worker를 `QT_QPA_PLATFORM=offscreen`으로 실행해
  **30프레임 수신**, 정상 종료와 카메라·worker 잔류 없음 확인.
  로그: `runs/zed_camera_ui_smoke.log` (Git 제외).
- 합성 RGB로 레이아웃을 별도 캡처해 영상 비율 유지와 빈 3D 영역을 시각 확인했다.
  `runs/camera_ui_layout.png` (Git 제외); 실제 카메라 이미지나 mesh/pose 추론 결과가 아니다.

### SAM3 추가 확인

- 전체 **153 passed**, Ruff 통과. 실제 별도 프로세스의 camera/prompt/result IPC, 늦게 연결되는
  subscriber의 prompt 수신, 두 prompt 격리, revision 변경·Stop, 최신 프레임 처리와 종료를 검사했다.
- 새 독립 `.venv-sam3`: Python 3.12.13, PyTorch 2.9.1+cu130, torchvision 0.24.1+cu130,
  NumPy 1.26.4. 원본 FFW의 Python 환경·adapter를 import하지 않는다.
- 기존 로컬 공식 `sam3.pt`(checkpoint revision `3c879f39826c281e95690f02c7821c4de09afae7`)를
  외부 가중치 파일로만 사용했다. GPU 모델 worker + 샘플 영상 발행 프로세스 + offscreen UI에서
  **A=`truck` / B=`wheel`**을 연속 처리하고, 같은 프레임의 mask·box 표시를 확인했다.
- 공식 `assets/images/truck.jpg`를 반복 발행한 테스트이며 실제 카메라 장면·물리적 tracking이나
  검출 정확도 검증은 아니다. A는 1개, B는 4개 후보가 나왔고 모든 후보를 그대로 표시했다.
  depth는 명시적인 NaN 테스트 입력이며 3D 결과를 만들지 않았다.
- 로그·시각 확인: `runs/sam3_official_worker.log`, `runs/sam3_official_ui_smoke.log`,
  `runs/sam3_official_ui.png` (모두 Git 제외). 종료 후 모델·UI·발행 프로세스 잔류 없음 확인.
- 실제 ZED의 SAM3 활성 실행도 별도 확인했다. camera/UI/SAM3 세 프로세스를 시작해 UI에서
  30프레임을 수신하고 모두 정상 정리했다. prompt가 비어 있는 초기 상태라 모델 추론은 하지 않았다.
  `runs/zed_sam3_idle_smoke.log` (Git 제외). 실제 ZED 장면의 A/B 검출 정확도는 아직 미검증이다.

### YAML 실행 추가 확인

- 전체 **208 passed**, Ruff 통과. 공유/개인 YAML의 부분 덮어쓰기, 파일 기준 상대 경로,
  venv Python symlink 보존, 기존 옵션 override와 잘못된 설정의 시작 전 오류를 검사했다.
- 저장소 밖 `/tmp`에서 옵션 없이 `grasp-workbench run`을 실행해, 로컬 YAML의 ZED·UI·SAM3
  세 worker 시작과 실제 영상 **83프레임 수신**, 종료 후 프로세스·카메라 정리를 확인했다.
  prompt는 빈 초기 상태라 모델 추론은 하지 않았다. 로그: `runs/yaml_run_smoke.log` (Git 제외).

### 2026-10-20 SAM3D 추가 확인

- 전체 **317 passed**, Ruff 통과. raw mesh codec, 실제 native IPC, A/B 요청 순서,
  원본 frame/mask 고정, cache 재사용, 오류 뒤 다음 요청 처리, stale 결과 무시를 검사했다.
  큰 mesh의 표시용 간소화는 표면 유지와 원본 배열 보존, receiver thread 실행을 검사했다.
- 새 독립 `.venv-sam3d`에서 공식 submodule의 pipeline과 외부 가중치로 실제 GPU 생성을 확인했다.
  기존 FFW application·adapter나 기존 Python 환경은 import하지 않았다. 모델 가중치 파일과
  범용 native library는 로컬 테스트에 사용했으며 Git에 포함하지 않는다.
- 실제 ZED에서 저장한 RGB-D frame(sequence 4)에 공식 SAM3를 실행한 결과를 고정해 재발행했다.
  A=`cell phone` / B=`orange ball`의 **동일 원본 frame·mask → 별도 SAM3D worker → native iceoryx2
  → offscreen Qt plot**을 확인했다. 이 생성 검증은 저장 입력 재생이며 실시간 카메라·SAM3 동시 추론은 아니다.
- 휴대폰 mesh **160,064 vertices / 320,120 faces**, 공 mesh **530,014 vertices / 1,060,116 faces**를
  생성하고 저장 PLY를 다시 읽어 검사했다. A/B plot 전환과 같은 고정 asset의 cache 재사용도 확인했다.
  생성 성공·표시 확인이며 실제 형상 정확도, metric scale, 현재 카메라 pose의 검증은 아니다.
- 별도로 저장소 밖 `/tmp`에서 옵션 없는 `grasp-workbench run`으로 camera·UI·SAM3·SAM3D
  네 worker 시작과 실제 ZED **103프레임 수신**, 정상 종료·장치 해제를 확인했다.
  이 실행은 prompt가 비어 있어 모델 추론하지 않았다.
- 로그·화면: `runs/sam3d_official_worker.log`, `runs/sam3d_official_ui_smoke.log`,
  `runs/sam3d_cached_preview_smoke.log`, `runs/sam3d_official_A.png`, `runs/sam3d_official_B.png`,
  `runs/sam3d_yaml_run_smoke.log` (모두 Git 제외).

### FoundationPose 추가 확인

- 전체 **367 passed**, Ruff 통과. A/B 제어·원본 RGB-D/mask·pose codec, 실제 iceoryx2 별도
  worker 프로세스, 세션당 1회 등록 후 연속 추적 호출, 개별 Stop, UI XYZ·회전축 표시를 검사했다.
- 새 독립 `.venv-foundationpose`에서 공식 submodule `a1b694b83e633c2cb6115b9063d940a687759392`와
  외부 공식 scorer/refiner 가중치를 사용했다. 기존 FFW application·adapter·Python 환경을 import하지 않았다.
  native extension은 이 Python에 맞게 설치·빌드했고 가중치·환경은 Git에서 제외했다.
- 실제 GPU 등록 중 upstream crop 계산에서 `float32/float64` 불일치가 발생했다. adapter에서
  `session.diameter`를 Python float로 변환하는 작은 수정 후 등록·tracking을 다시 실행해 통과했다.
  upstream 소스는 수정하지 않았으며 추가 예방·복구 로직은 넣지 않았다.
- 기존 저장 ZED RGB-D와 mask(sequence 600)의 **검은 병 / 검은 쟁반**을 반복 재생하고,
  repo 안에 복사한 해당 optimized mesh로 실제 A/B 등록과 연속 `track_one()`을 확인했다.
  별도 FP worker → native iceoryx2 → offscreen UI에서 A/B 동시 pose 결과 8회와 위치·회전축 plot,
  A Stop 이후 B 단독 결과, B Stop을 확인했다. 정지 장면 재생이며 움직임·가림 복구·pose 정확도 검증은 아니다.
- 별도로 `/tmp`에서 옵션 없는 `grasp-workbench run`을 실행했다. FP·SAM3·SAM3D·UI worker는
  시작했지만 현재 ZED가 **CAMERA NOT DETECTED**를 반환해 카메라 캡처·실시간 FP는 확인하지 못했다.
  카메라 설정·입력을 자동 변경하지 않았고 종료 후 잔류 worker는 없었다.
- 로그·화면: `runs/foundationpose_official_worker.log`, `runs/foundationpose_official_ui_smoke.log`,
  `runs/foundationpose_official_results.json`, `runs/foundationpose_official_ui.png`,
  `runs/foundationpose_yaml_run_smoke.log` (모두 Git 제외).

### 이름 선택·현재 테이프/참치캔 확인

- FP A/B 입력에 **이름으로 mesh 선택**을 추가했다. 물체 21개 + Table/Leg 3개의 이름을 표시하고,
  선택 시 해당 경로·scale을 입력한다. Browse·직접 경로 입력은 유지한다. 원본·최적화 mesh의 파일명·
  바이트와 UUID 경로는 바꾸지 않았다. 전체 **370 passed**, Ruff 통과.
- 이번에는 실제 ZED-M 17751856이 연결되어 HD720@30 RGB-D 캡처를 확인했다. 영상에는
  왼쪽 노란 참치캔과 큐브 위의 테이프가 있었다. 각각의 저장 mesh도 이름 목록에서 선택했다.
- 같은 캡처 영상에서 `tuna can`은 후보 0개, `tape roll`은 참치캔까지 포함한 후보 2개였다.
  임계값·분할 로직을 바꾸지 않고 prompt를 비교했다. 현재 장면의 `brown packing tape roll` /
  `yellow tuna can`으로 실제 연속 SAM3 worker에서 각각 1개 후보를 확인했다.
- 저장 영상 재생이 아닌 **실제 camera worker → SAM3 worker → FP worker → iceoryx2 → offscreen UI**로
  두 mesh의 등록·연속 tracking과 XYZ·회전축 표시를 확인했다. A/B 공통 pose 결과 11회, A Stop 뒤
  B 단독 결과와 B Stop을 확인했다. 정지 장면의 실행 확인이며 형상·pose 정확도나 움직임·가림 복구 검증은 아니다.
- 로그·화면: `runs/tape_tuna_camera.log`, `runs/tape_tuna_prompt_check.json`,
  `runs/tape_tuna_live_{camera,sam3,foundationpose}.log`, `runs/tape_tuna_live_results.json`,
  `runs/tape_tuna_live_ui.png`, `runs/tape_tuna_live_pose_overlay.png` (모두 Git 제외).

### 실시간 영상의 SAM3 + pose overlay 확인

- UI만 변경해 FP 결과 RGB 위에 최신 SAM3 mask·box를 그리고, 그 위에 A/B pose의 5cm XYZ 축과
  물체 ID를 표시한다. 오른쪽 3D plot은 유지한다. 추적 worker·모델·IPC 계약은 변경하지 않았다.
- 전체 **379 passed**, Ruff 통과. 카메라 intrinsics와 pose로 계산한 축 위치·RGB 색상, mask·box와
  동시 표시, 원본 배열 보존, SAM3/FP 각각의 화면 갱신, Start/Stop·Apply 후 표시 변경을 검사했다.
- 실제 ZED-M의 테이프·참치캔을 SAM3·FP worker로 처리하고 같은 Qt 영상에 mask·box·pose 축이
  함께 표시되는 것을 확인했다. 영상 하단의 FP sequence와 SAM3 overlay sequence는 각각 표시하며,
  별도 시간 매칭·history·추적 실패 판단·자동 재등록은 추가하지 않았다.
- 확인 화면·로그: `runs/tape_tuna_overlay_ui.png`, `runs/tape_tuna_overlay_ui.log` (Git 제외).

### 기존 mesh / 생성 후 FP 버튼 분기 확인

- **기존 mesh로 Start** / **mesh 없음 · 생성 후 Start**를 A/B 각각 제공한다. mesh 존재 여부를
  자동 검색하지 않으며 사용자가 분기한다. 기존 **Generate mesh**는 mesh만 생성하는 동작을 유지한다.
- 전체 **391 passed**, Ruff 통과. 생성 완료 뒤 FP 제어 발행, 고정 RGB-D/mask와 사용자 scale 전달,
  기존 선택 경로 대신 생성 경로 사용, A/B 독립 연결, Stop·입력 변경에 따른 대기 중 자동 시작 취소를 검사했다.
  실제 native iceoryx2·별도 SAM3D/FP worker·Qt UI를 잇는 fake predictor 통합 테스트도 포함한다.
- 실제 ZED 장면에서 **참치캔 B는 기존 최적화 mesh**, **테이프 A는 새 SAM3D 생성 mesh**로 분기했다.
  테이프는 새 asset으로 **557,466 vertices / 1,114,912 faces**를 생성했고, 완료 후 추가 Start 클릭 없이
  새 PLY 경로로 FP 등록·연속 추적이 시작됐다. A/B 공통 pose 결과 11회와 영상 overlay, 개별 Stop을 확인했다.
- 신규 테이프 mesh의 scale은 테스트 입력으로 **0.11**을 수동 지정했다. 자동 scale 추정·보정이 아니며,
  이 검증은 연결·실행 확인이다. 치수·pose 정확도나 움직임·가림 복구를 검증한 것은 아니다.
- 확인 자료: `runs/generate_to_pose_live_ui.png`, `runs/generate_to_pose_live_ui.log`,
  `runs/generate_to_pose_live_results.json`, `runs/generate_to_pose_live_{camera,sam3,sam3d,foundationpose}.log`
  (모두 Git 제외).

### 2026-10-20 실측 SAM3D·SAM3 추적 검증 이식

- 이후 요청에 따라 신규 SAM3D 입력을 실측 depth pointmap으로 바꾸고, 이전 repo의 scale·좌표 변환을
  이식했다. pointmap·metric 정점·초기 pose를 이전 수식과 20개 수치 입력으로 비교했다.
- 전체 **431 passed**, Ruff 통과. 실제 native iceoryx2·별도 FP 프로세스와 fake backend로 후보 순서/score 변화,
  SAM3 미검출, 불일치 5회 재등록, 추적 예외 후 회복, revision 분리, 재등록 RGB-D/mask 원본을 검사했다.
- 실제 ZED 장면에서 **테이프 A는 새 실측 SAM3D mesh**, **참치캔 B는 기존 mesh**로 각각 FP를 시작했다.
  새 mesh는 556,188 vertices / 1,112,352 faces이며 출력 extents는 약 0.107 × 0.106 × 0.056m다.
  UI의 기존 수동 scale 0.11을 **1.0으로 교체**해 추가 수동 보정 없이 등록·추적했다. 초기 SAM3D pose도 보존했다.
- A/B 공통 결과 38회와 SAM3 mask·pose overlay, 개별 Stop·종료를 확인했다. 해당 정지 장면의
  SAM3 검증 불일치는 0회였다. 재등록 분기는 위 fake backend/native IPC 테스트로 확인했으며,
  실물 이동·가림 후 복구와 치수·pose의 정답 대비 정확도는 아직 검증하지 않았다.
- 기록: `runs/metric_visual_live_ui.png`, `runs/metric_visual_live_results.json`,
  `runs/metric_visual_live_{camera,sam3,sam3d,foundationpose,ui}.log` (Git 제외).
  실행 명령은 계속 **`grasp-workbench run`**이며 worker 분리·iceoryx2 전송을 유지한다.

### 2026-10-20 FP 사용 시 영상 끊김 수정

- 카메라 sequence=100을 받아도 화면은 FP 결과 sequence=42에 머무르는 문제를 재현했다.
  영상 갱신을 최신 카메라 기준으로 바꾸고 SAM3/FP는 overlay 상태만 갱신하도록 분리했다.
  같은 poll의 중복 합성과 부분 FP 결과 수신 시 다른 슬롯의 축이 사라지는 문제도 제거했다.
  backend·IPC·고정 생성/등록 입력은 변경하지 않았다. history·시간 gate·보간·새 thread도 추가하지 않았다.
- 실제 Qt event loop에 **모의 카메라 약 30 FPS·모의 FP 약 5 FPS**를 공급한 비교에서
  서로 다른 RGB 표시 속도가 **4.71 → 30.63 FPS**로 개선됐다. 모델 추론 자체를 가속한 결과는 아니다.
- 전체 **434 passed**, Ruff 통과. 느린 FP 사이의 30개 카메라 갱신, 과거 결과 수신 시 RGB 유지,
  A/B 부분 결과·recovering·Stop 및 poll당 합성 1회를 검사했다.
- 실행 중인 사용자 앱에 명령을 발행하거나 카메라를 다시 열지 않고, 기존 ZED 스트림을 읽는 별도
  화면에서도 RGB 약 **29.5 FPS**를 확인했다. 이때 SAM3/FP 결과가 발행되지 않아 실장치의 동시 추론
  overlay 성능 검증은 하지 못했다. 기존 실행 중인 UI에는 코드가 자동 반영되지 않으므로 재실행해야 한다.
- 기록: `runs/benchmark_pose_ui_cadence{,_after}.{py,json,log}`,
  `runs/live_display_readonly.{png,json,log}` (Git 제외).

### 2026-10-20 오른쪽 pose plot에 mesh 추가

- A/B static mesh를 한 번 로드·간소화하고, FP의 원본 object 좌표·입력 scale·camera_from_object로 표시한다.
  SAM3D 생성 경로는 이미 수신한 preview를 재사용한다. mesh·axis·label artist도 재사용해 매번 axes를
  초기화하지 않으며, 표시 실패는 UI에 알리고 독립 FP session을 중단하지 않는다.
- 실제 PLY/OBJ/GLB의 scale·scene 변환·textured GLB, 원본 파일/정점 불변, pose 변환·artist 캐시,
  A/B 부분 결과·recovering·Stop 및 mesh-only 기존 표시를 검사했다. 전체 **446 passed**, Ruff 통과.
- 실제 테이프·참치 mesh를 사용한 모의 카메라 약 30 FPS·FP 약 5 FPS 입력에서, 물체당 약 2,000 faces의
  production UI는 **서로 다른 RGB 표시 약 30.7 FPS**를 유지했다. 20,000 faces 두 개의 대조군은 약 22.5 FPS였다.
  이 측정은 렌더링 테스트이며 GPU 추론 성능 측정은 아니다.
- 저장된 실제 ZED 장면·FP pose와 실제 mesh를 재생해 오른쪽 mesh 두 개·pose 축을 확인했다.
  기록: `runs/pose_mesh_ui.png`, `runs/pose_mesh_ui.log`, `runs/benchmark_pose_mesh_production_ui.{py,json,log}`
  (Git 제외). 이번 표시 기능 검증에서는 실시간 SDK·GPU 추론을 새로 실행하지 않았다.

### 2026-10-20 VLM 단발 인식·asset 재사용 추가 확인

- 전체 **541 passed**, Ruff·Git diff 공백 검사 통과. VLM 환경의 `pip check`도 통과했다.
  실제 native iceoryx2·독립 fake model worker·Qt UI 테스트에서 기존 asset 경로와 새 물체의
  SAM3D 생성·영속 저장·FP 연결 경로를 각각 검사했다. API 키 값은 테스트 결과나 공개 catalog에 넣지 않는다.
- 저장한 실제 ZED RGB-D를 재생하고 실제 **gpt-6.1-sol → SAM3 → FoundationPose**를 연결했다.
  UI 버튼 한 번으로 테이프/`brown packing tape roll`, 참치캔/`yellow tuna can`을 인식하고
  각각 SAM3 후보 1개와 기존 mesh의 FP 등록·tracking을 확인했다.
- **A/B가 함께 tracking인 결과 16회**, SAM3D 요청 **0회**, 오른쪽 plot의 A **2,000 faces**·B **1,998 faces**를
  확인했다. 시작·모델 로딩을 포함한 전체 실행은 약 54초였다. 저장 영상 재생 확인이며 실물 이동·가림 복구나
  형상·pose 정확도 검증은 아니다. 새 물체 자동 생성 경로는 위 fake backend/native IPC 테스트로 구분해 검증했다.
- 기록: `runs/vlm_pipeline_ui.png`, `runs/vlm_pipeline_results.json`,
  `runs/vlm_pipeline_{vlm,sam3,sam3d,foundationpose,ui}.log` (Git 제외).

### Codex CLI backend 추가 확인

- 전체 **572 passed**, Ruff·Git diff 공백 검사·VLM 환경 `pip check` 통과.
  기본 API 경로와 CLI 선택·경로 해석·API 키 미전달, 실제 native IPC와 모의 CLI 실행을 검사했다.
- 기존 **ChatGPT 로그인 + 실제 Codex CLI(`gpt-6.1-sol`)**로 저장 ZED RGB 한 장을 인식했다.
  단독 인식은 약 **9.20초**였으며 API 키 파일을 사용하거나 CLI 인증자료를 repo로 복사하지 않았다.
- 저장 ZED RGB-D 재생에 실제 **Codex CLI → SAM3 → SAM3D/FP → iceoryx2 → Qt UI**를 연결했다.
  버튼 한 번만 눌렀고 수동 Apply·후보 선택·FP Start는 하지 않았다. A/B의 SAM3 후보는 각각 1개였다.
  테이프 A는 `known_asset_id=null`로 반환되어 **실측 metric mesh 한 개 생성 → 영속 저장 → FP 등록**을 수행했고,
  참치캔 B는 기존 asset ID `e76130be39c44e3282d47be1056760f5`의 mesh로 SAM3D 없이 FP를 시작했다.
- A의 저장소를 다시 열어 이름·`scale=1`·`object.ply`와 `manifest.json` 파일을 확인했다.
  **A/B가 함께 tracking인 결과 16회**, plot A **2,000 faces**·B **1,998 faces**를 확인했다.
  모델 로딩·mesh 생성을 포함한 전체 실행은 약 **96.79초**였다. 이로써 새 물체 분기도 실제 모델로 확인했다.
- VLM은 catalog에 있던 테이프의 기존 ID를 선택하지 않았다. 이는 관측된 semantic matching 한계이며,
  임의 이름 matcher나 API fallback으로 바꾸지 않았다. asset ID를 실제 물체 identity의 정답으로 해석하지 않는다.
  카메라 SDK를 새로 열지 않은 저장 입력 재생이며 실물 움직임·가림 복구·metric 치수·pose 정답 정확도는 미검증이다.
- 기록: `runs/vlm_codex_verification.json`, `runs/vlm_codex_pipeline_ui.png`,
  `runs/vlm_codex_pipeline_results.json`,
  `runs/vlm_codex_pipeline_{vlm,sam3,sam3d,foundationpose,ui}.log` (Git 제외).

### GraspGenX 후보 생성·표시 추가 확인

- 전체 **668 passed**, Ruff·Git diff 공백 검사 통과. raw grasp codec·lazy worker·mesh-only 설정과
  실제 native iceoryx2의 A/B queue·cache 결과·오류 후 다음 요청·종료를 검사했다.
  실제 Qt UI와 별도 fake 추론 worker를 연결해 Generate/Regenerate·FP 결과 합성·최신 카메라 RGB 유지도 검사했다.
- 저장 ZED RGB-D를 재생하고 실제 **SAM3·FoundationPose·GraspGenX 독립 worker → iceoryx2 → Qt**를 연결했다.
  테이프 A와 참치캔 B 각각 실제 모델의 **raw 후보 1,000개 → 기존 SE(3) 다양성 선택 100개**를 저장했다.
  A는 먼저 실생성해 저장했으며 최종 통합 실행에서는 **A cache 재사용·B 신규 생성**이었다.
  A의 Generate 재요청도 `cached=true`를 확인했다. 최종 실행은 모델 로딩을 포함해 약 **30.20초**였다.
- 새 bundle 두 개를 `load_grasp_bundle`로 다시 읽어 각각 후보 100개·원본 pool 1,000개·공식 source revision
  `b9429097728cb1c430dd78b92edf17ba318aad03`와 저장 metadata를 확인했다.
  저장 형식은 새 repo의 `runs/grasps/<cache_key>/` NPZ·JSON이며 기존 Isaac용 YAML 저장 형식과는 다르다.
- A/B가 함께 FP tracking인 표시를 4회 검사하고 선택 grasp 축의 영상 overlay와 오른쪽 plot의
  canonical gripper 실제 visual mesh를 확인했다. 로컬에서 설정한 gripper descriptor를 사용했으며
  공유 기본값은 `enabled=false`·`franka_panda`를 유지했다. FP sequence **886**과 최신 RGB sequence **891**도
  구분해 기록했다. 원본 SDK·공식 submodule은 변경하지 않았다.
- metadata의 `scene_collision_checked=false`, `robot_ik_checked=false`를 확인했다.
  실제 후보 생성·저장·표시 검증이며 물리 grasp 성공·장면 충돌·IK·실물 움직임은 검증하지 않았다.
- 재사용한 GraspGenX 환경의 PyTorch **2.7.1+cu128**·torchvision **0.22.1+cu128**로 실제 모델은 실행됐지만,
  upstream 선언 `torch<2.7`·`torchvision<0.22`와의 불일치 2건은 `uv pip check`에 남아 있다.
  이번에는 두 패키지를 변경·다운그레이드하지 않았다. 실제 실행 확인과 전체 환경의 의존성 검증 통과는 구분한다.
- 기록: `runs/graspgenx_pipeline_ui.png`, `runs/graspgenx_pipeline_results.json`,
  `runs/graspgenx_pipeline_{sam3,graspgenx,foundationpose,ui}.log` (Git 제외).

### VLM Task·팔·제약 semantic 복원 확인

- 전체 테스트 **734개 통과**(9.97초), Ruff·`git diff --check` 통과.
- 저장 ZED RGB-D 한 장을 재생하고 실제 **Codex CLI(`gpt-6.1-sol`) → 독립 VLM worker → native iceoryx2 → Qt**를 확인했다.
  버튼 요청 두 번에서 **왼팔 참치캔 grasp → single**(약 **11.18초**), **양팔 테이프 grasp → dual**(약 **18.10초**)을 받았다.
- 양팔 요청은 같은 Source 관측 **`obs_tape`**를 left/right assignments가 함께 참조했다.
  각 팔의 역할·`grasp_region`과, 양팔이 잡은 뒤 들어올리기·운반 단계에서 `maintain_relative_pose`를 유지한다는
  의미 제약이 반환됐다. A/B 슬롯을 좌/우 팔로 치환하거나 수치 pose·운동 제약을 생성한 결과가 아니다.
- Session 저장·불러오기·`request_only` export를 확인했으며 로봇 명령은 **0회**였다.
  이번 실제 인식에는 policy catalog를 제공하지 않아 선택은 null이었다. catalog 선택·checkpoint 공개 제외는
  fixture/native IPC 테스트로 구분해 확인했다.
- 이번 검증에서는 SAM3·FoundationPose·GraspGenX GPU worker를 실행하지 않았다.
  실제 VLM의 의미 출력·IPC·UI·저장 JSON 확인이며 실물 grasp·양팔 IK·이동·가림 복구 검증이 아니다.
  기존 모델 연결 경로의 fake/native 회귀 테스트와 실제 모델 검증 이력은 위에 별도로 남겼다.
- 기록: `runs/semantic_restore/results.json`, `single.json`·`dual.json`,
  `single.session.json`·`dual.session.json`, `single.request.json`·`dual.request.json`,
  `single.png`·`dual.png`, `worker.log`(같은 폴더), `runs/semantic_restore_smoke.py` (Git 제외).

### grasp 선택·선택적인 endpoint IK 추가 확인

- 전체 테스트 **798개 통과**(10.92초), Ruff·`git diff --check` 통과. 공식 submodule 6개는 변경하지 않았다.
- `tests/test_grasp_selection_pipeline.py`는 production predictor·generic plugin loader·독립 worker CLI·native iceoryx2를
  그대로 사용하고 **robot state/solver plugin만 fixture**로 대체했다. 잡지 않는 Target A와 Source B에서
  right single → 같은 B의 dual 요청, `base_from_object` 위치 `(12, 23, 34)`의 합성·팔별 canonical→TCP offset·
  동일 source instance 연결·IK checked 결과·bundle 원본 바이트 보존을 확인했다. 실측 로봇 검증은 아니다.
- 저장 ZED RGB-D와 **기존 실제 GraspGenX 100개 후보 bundle**을 native IPC로 Qt에 전달하고,
  실제 no-robot 선택 worker에서 **9,900 ordered pairs → 64개 compact 결과**를 확인했다.
  좌/우 label·두 canonical gripper visual·결과 combo 선택까지 표시했으며 통합 표시 예시는 약 **0.13초**였다.
  GraspGenX 추론을 재실행한 시간이 아니며 IK는 unchecked·로봇 명령은 **0회**였다.
- 실제 GPU cuRobo를 사용하되 **실측 로봇 state 없이** retract 관절값에서 URDF FK로 목표를 만든 offline smoke를 실행했다.
  endpoint 확인을 위해 pregrasp와 grasp를 같은 목표로 두었으며 좌/우 single·dual 모두 IK 성공과
  self-collision 검사를 확인했다. 로봇 명령은 **0회**이며 실제 상태 수신·이동·장면 충돌·trajectory 검증이 아니다.
- single batch 16 뒤 dual batch 4를 같은 solver에서 처리할 때의 shape 문제를 실제 재현한 뒤,
  single/dual solver cache를 분리해 같은 smoke가 통과하는 것을 확인했다. 새 예방 gate·retry 정책은 추가하지 않았다.
- 로컬에서는 geometry 선택을 켜고 `robot_config: null`로 사용한다. 실제 로봇 상태 연결은 아직 확인하지 않았으며
  공유 기본값은 선택 worker 꺼짐·robot config null을 유지한다.
- 기록: `runs/robot_ffw_ik_gpu_smoke.log`, `runs/robot_ffw_ik_gpu_smoke_before_batch_fix.log` (Git 제외).
  UI 기록은 `runs/grasp_selection/results.json`·`ui.png`·`worker.log`,
  driver는 `runs/grasp_selection_ui_smoke.py`다 (Git 제외).

### 고정 입력 보정·비실행 후처리 이식 확인

- 전체 테스트 **869개 통과**(13.73초), Ruff·`git diff --check` 통과. 공식 submodule 6개는 변경하지 않았다.
- 저장 ZED RGB-D를 실제 SAM3 독립 worker·native iceoryx2로 보내 **text → point refine → box refine**를 확인했다.
  각각 약 **6.08초·0.67초·0.10초**였고 세 결과 모두 비어 있지 않은 mask를 반환했다.
  실시간 실물 움직임·가림에 대한 추적 정확도 검증은 아니다.
- 같은 저장 RGB-D **sequence 29**·실제 SAM3 point-refined 참치캔 mask·기존 can mesh로 실제 FP register 후
  같은 predictor의 별도 FP session에서 seed/refinement를 확인했다. 모델 로딩 포함 약 **19.01초**, 두 추정값의 차이는 translation 약
  **0.001865m**·rotation 약 **7.04°**였다. 이는 두 추정 결과의 차이이며 **정답 대비 정확도가 아니다**.
  이 GPU 확인 자체는 worker/native IPC 실행이 아니다.
- 별도 SAM3/FP native 회귀 테스트는 3회 반복 통과했다. fake 모델로 live sequence 100/101과 고정 입력 37을 구분하고
  mask register·mask 없는 seed가 정확한 원본 frame을 사용하는 것을 검사했다. 실제 GPU 모델 검증과 구분한다.
- 실제 Qt에서 `seeded.npz`를 새로 불러와 geometry를 복원하고 FP가 자동 시작되지 않는 것을 확인했다.
  기존 100개 grasp bundle을 읽고 실제 no-robot 선택 worker·native IPC로 left single 선택·mask/mesh/gripper 표시를 확인했다.
  이 Qt 검증에서는 API·GPU 추론·카메라 SDK·로봇을 호출하지 않았다.
- native Task 회귀는 **명시적 Pose · Grasp 진행 버튼**과 **VLM 완료 뒤 추가 클릭 없는 자동 경로**를 각각 3회 반복 확인했다.
  fake 모델과 독립 worker·실제 IPC/Qt를 사용해 Source B만 grasp를 요청하고 FP pose가 계속 갱신되어도
  cache 조회/생성·선택은 한 번씩만 요청하는 것을 검사했다. 실제 VLM/GPU 모델 실행 검증은 아니다.
- SAM3D monocular/texture 옵션은 fake backend·IPC codec·실제 GLB 직렬화·cache/AssetStore 보존 테스트로 확인했다.
  **실제 MoGe 추론·texture baking·관련 모델 다운로드는 실행하지 않았다.** 저장 GLB의 UV/material 보존과
  실제 생성 모델의 texture 품질·monocular scale 정확도는 구분한다.
- hand FK 연속 stream은 native fake robot state로 검증하며 실제 로봇 state 연결은 확인하지 않았다.
  위의 실제 offline GPU IK smoke는 endpoint solver 확인 이력으로 유지하며 실측 hand FK 검증으로 바꾸어 해석하지 않는다.
- SAM3 기록: `runs/manual_vision/sam3_results.json`·`sam3.log`·`refined.bin`·`boxed.bin`·`refined.npz`,
  driver는 `runs/sam3_refine_smoke.py`다 (Git 제외).
  FP 기록은 `runs/manual_vision/fp.log`·`fp_seed_results.json`·`seeded.npz`다 (Git 제외).
  Qt 기록은 `runs/manual_vision/ui_review_results.json`·`ui_review.png`·`ui_selection.log`다 (Git 제외).

### VLM backend 설정·Anthropic 연결 확인

- 전체 **937개 테스트 통과**(13.89초), Ruff·`git diff --check` 통과. 공식 submodule 6개는 변경하지 않았다.
- README의 YAML 예시와 `examples/config.local.example.yaml`을 실제 설정 parser로 확인했다.
  공유 기본 OpenAI 설정은 유지하며 backend 변경 시 model ID를 임의로 바꾸지 않는다.
- Anthropic fake SDK 단위 테스트와 독립 worker·실제 iceoryx2·SDK stub의 native 테스트로
  키 파일/환경변수 경로, 원본 RGB-D·semantic 결과·오류 전달을 검사했다. native 두 인증 경로는 각각 3회 반복 통과했다.
- 실제 Anthropic SDK는 외부 서버 대신 로컬 mock HTTP에 연결해 요청 직렬화와 응답 파싱을 확인했다.
  `end_turn` 응답은 받고 `max_tokens`로 중단된 출력은 완료 결과로 사용하지 않는 것을 검사했다.
  기록: `runs/anthropic_sdk_check/result.json`·`verify.py` (Git 제외).
- 위 확인은 실제 Claude 모델 호출·실제 인증·과금·서비스 접근 검증이 아니다.
