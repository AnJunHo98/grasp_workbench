# UI 사용법과 데이터 계약

[빠른 시작](../README.md) · [설치·YAML·VLM backend 설정](setup.md) · [검증 이력과 미검증 범위](verification.md)

이 문서는 기능을 켠 뒤의 UI 조작과 결과 해석을 설명한다. 장치·모델 환경·인증은 각 사용자가 준비한다.
평소 실행은 `grasp-workbench run`이며 기능별 `enabled` 조합은 [README](../README.md)를 따른다.
**worker 활성화와 추론 요청은 다르다.** VLM·Task는 다른 모델 worker를 자동 설치·활성화하지 않는다.

**A/B는 물체 슬롯이지 좌/우 팔이 아니다.** 물체·팔 배정과 Source/Target 역할은 별도로 다룬다.
영상의 표시용 overlay와 mesh 생성·FP 입력에 쓰는 정확한 원본 관측도 구분한다.

- [화면 배치와 worker 탭](#worker-tabs)
- [SAM3 A/B prompt와 후보](#sam3)
- [고정 원본에서 mask 보정](#snapshot)
- [SAM3D mesh·단위·cache](#sam3d)
- [FoundationPose 등록·seed·추적·overlay](#foundationpose)
- [VLM Task·Source/Target·policy](#vlm)
- [GraspGenX 후보·좌표·cache·기존 bundle](#graspgenx)
- [한팔·양팔 선택과 endpoint IK](#grasp-selection)
- [표시용 Grasp FK](#grasp-fk)
- [LeRobot data recording](recording.md)
- [프로세스 구조와 IPC](#ipc)
- [구현 범위](#scope)

<a id="worker-tabs"></a>
## 화면 배치와 worker 탭

상단의 **VLM 지시문 입력·인식·진행·상태**는 공통 영역이다. VLM 전용 탭은 따로 만들지 않는다.
Source/Target 편집기·Task/팔/제약/policy 상세·선택한 역할로 진행·VLM 요청 선택은 접이식 작업 상세 안에 있다.
provider·모델·인증·worker 실행 여부는 YAML에서 정하며, UI의 VLM 요청 선택은 프로세스 on/off가 아니다.

왼쪽 세로 탭은 아래 순서로 해당 worker의 조작만 표시한다. 오른쪽의 최신 RGB와 Mesh / Pose 3D는
공통으로 유지한다. **탭 전환만으로 요청을 보내거나 worker를 시작·중지하지 않으며 추적도 계속된다.**

| 탭 | 입력·상태 | 기본 버튼 |
|---|---|---|
| Camera | 장치·해상도·수신 상태 | 없음. 실행 시 자동 연결 |
| SAM3 | A/B prompt·mask 후보 | 적용·분할 시작, 분할 중지, A/B Mask 보정 |
| SAM3D | 선택 mask·생성 방식·상태·표시할 A/B mesh | A/B Mesh 생성 |
| FoundationPose | A/B mesh·scale·상태 | 추적 시작·재등록, 추적 중지, Mesh 생성 후 추적 |
| GraspGenX | A/B mesh·후보·score | 후보 생성, 다시 생성, 기존 후보 불러오기 · bundle |
| Grasp Selection | 대상·한팔/양팔 mode·결과 | 후보 선택·다시 평가 |
| AprilCube | A/B mesh·scale·cube→mesh XYZ/RPY | A/B 적용 |
| Recording | Dataset path, status, frame/episode count | Start Recording / Save Episode / Discard Episode |

꺼진 worker 탭은 **OFF 안내만 표시**하고 작업 조작은 제공하지 않는다. AprilCube 모드에서는
Camera·AprilCube와 별도로 켠 Recording만 활성화하고 상단 VLM도 비활성화한다. YAML의 일반 worker 값 자체를 고치지는 않는다.
FP seed refinement는 FoundationPose 탭의 **고급**, 로봇 기준 pose·IK 상세와 표시용 hand FK는
Grasp Selection 탭의 **고급**에 둔다. SAM3/FP 중지 버튼은 해당 작업의 중지이지 프로세스 종료가 아니다.

Snapshot 캡처·저장·불러오기, Session 저장·불러오기, 실행 요청 export 버튼은 기본 화면에서 제거했다.
관련 backend helper API와 기존 저장 파일은 삭제하지 않는다. mask 보정에 필요한 원본 고정은 그대로 유지하고,
화면이 고정된 동안만 상단 VLM 영역 아래에 **실시간으로 돌아가기**를 표시한다.

<a id="sam3"></a>
## SAM3: A/B prompt와 후보 선택

1. SAM3 탭에서 A/B에 서로 다른 prompt를 입력하고 **적용·분할 시작**한다. 예: A `red cup`, B `blue bowl`.
   한 슬롯을 비우면 해당 슬롯은 처리하지 않는다. prompt 수정 후에는 다시 적용한다.
2. 슬롯별 mask·box·score와 후보 수를 확인한다. A는 초록색, B는 주황색이며 `A[1]`, `B[1]`처럼 표시한다.
   후보가 하나면 자동 선택하고 여러 개면 dropdown에서 직접 선택한다. **0개 후보와 추론 오류는 다르다.**
3. **분할 중지**는 SAM3 추론을 중지한다. 카메라 캡처·원본 영상 표시는 계속되며 진행 중 GPU 호출은 완료될 수 있다.
   적용/중지로 revision이 바뀌면 이전 revision의 결과는 표시하지 않는다.

입력 문구는 그대로 전달하며 **자동 번역하지 않는다**. 한글을 막지는 않지만 우선 `cell phone`,
`orange ball`처럼 영어 이름·외형으로 확인한다. 같은 물체도 prompt에 따라 검출 결과가 달라질 수 있다.
기본 prompt는 비어 있으며 적용 전에는 자동 추론하지 않는다.

SAM3 worker는 모델을 한 번 로드하고 같은 frame의 image feature를 A/B가 공유한다.
각 prompt의 후보는 보존한다. 이는 최신 프레임의 **image segmentation**이지 모든 카메라 프레임의 처리,
SAM3 video tracking, 영구적인 물체 identity 또는 6D pose 추정이라는 뜻은 아니다.

화면은 최신 카메라 RGB에 최근 SAM3 mask·box를 합성한다. 카메라와 SAM3의 sequence를 각각 표시하며
같은 시각의 측정으로 임의 해석하지 않는다. mesh 생성·FP 등록에는 overlay가 아니라 SAM3 결과에 포함된
**정확한 원본 RGB-D와 선택 mask**를 사용한다. A/B 슬롯 ID는 mesh asset의 영구 key가 아니다.

관련 코드: [SAM3 모델](../grasp_workbench/models/sam3.py), [worker](../grasp_workbench/workers/worker_sam3.py).

<a id="snapshot"></a>
## 고정 원본에서 mask 보정

- SAM3 탭의 A/B **Mask 보정**은 선택한 후보의 정확한 원본 RGB-D·mask를 고정해 보정창을 연다.
  카메라 worker와 연속 추론 worker는 탭 전환이나 표시 고정만으로 중지되지 않는다.
- RGB·SAM3·FP 결과는 각자의 원본 frame을 보존한다. 한 화면에 있다고 같은 시각으로 동기화한 것은 아니다.
- 보정창에서 포함/제외 포인트 또는 박스를 지정하고 **SAM3 보정 요청**을 누른다.
  좌클릭은 포함, 우클릭은 제외이며 박스는 드래그한다. 표시 좌표를 원본 pixel 좌표로 변환해
  고정 RGB-D·선택 seed mask와 함께 기존 SAM3 worker로 보낸다.
- 보정은 실제 SAM3 재추론이다. UI가 사각형/픽셀 배열로 mask를 임의 대체하지 않는다.
  보정 결과는 고정 입력으로 표시되며 이후 mesh 생성·FP 입력에도 그 원본을 사용한다.
- 고정 표시 중에는 **실시간으로 돌아가기**로 최신 카메라 화면에 복귀한다. 이 버튼은 평상시에는 숨긴다.

파일용 [Snapshot helper](../grasp_workbench/snapshot.py)의 저장·복원 API와 기존 NPZ 파일은 유지하지만
기본 UI에는 캡처·저장·불러오기 버튼을 두지 않는다. helper로 불러온 snapshot도 과거 입력이며 불러오기만으로
추론·tracking·로봇 명령을 시작하지 않는다. 이전 앱 NPZ와의 호환 기능은 추가하지 않았다.

refine IPC는 `/sam3/refine/requests`·`/sam3/refine/results`이며 **별도 모델 worker는 추가하지 않는다**.

<a id="sam3d"></a>
## SAM3D: mesh 생성·단위·cache

### UI 조작

1. SAM3 후보를 선택한다. 해당 원본 RGB-D frame과 mask 복사본을 고정하고 sequence를 표시한다.
2. SAM3D 탭 A/B의 **Mesh 생성**으로 요청한다. 이후 영상·분할이 갱신되어도 생성 입력을 바꾸지 않는다.
3. 상태는 Generating/Done/Error다. A/B 요청은 한 번에 하나씩 전송하고 완료·오류 후 다음 요청을 처리한다.
   임의의 진행률은 표시하지 않는다.
4. 완료된 asset은 SAM3D 탭의 A/B mesh 선택기로 하나씩 본다. FP까지 이어가려면 FoundationPose 탭의
   [Mesh 생성 후 추적](#foundationpose)을 사용한다. Mesh 생성 자체는 FP를 시작하지 않는다.

생성 방식은 YAML의 `sam3d.depth_source`·`sam3d.texture_baking`으로 정하고 UI는 현재 설정만 표시한다.
수동·VLM·Task 요청에 같은 설정을 사용하며 VLM을 이유로 실측 depth로 강제 변경하지 않는다.

### 단위와 표시

| 설정/결과 | 의미 |
|---|---|
| `depth_source: depth` | 같은 frame의 실측 depth·intrinsics로 만든 pointmap 사용. `length_unit: metre` |
| `depth_source: monocular` | MoGe 예측 pointmap 사용. `length_unit: monocular_predicted_scale`; 실측 metre 보장 없음 |
| `texture_baking: false` | 기본 vertex-color PLY 경로 |
| `texture_baking: true` | 실제 UV/material을 가진 GLB를 cache·영속 asset에 보존 |

GLB 표시 축 변환을 해제하고 출력 scale을 정점에 **한 번만** 적용한다. mesh와 별도의 OpenCV
`camera_from_object`를 보존한다. 초기 pose는 생성 원본 frame 기준이지 실시간 tracking 결과가 아니다.
새 mesh의 FP 입력 scale은 `1.0`이다. monocular에서의 `1.0`은 예측 scale이 이미 적용됐다는 뜻이지
실측 metre라는 뜻이 아니다. 실측 depth를 사용해도 형상·치수 정확도를 보장하지 않는다.

기본 실측 경로의 preview는 asset-local metre 좌표이며 A/B를 공통 world 좌표로 정렬한 것이 아니다.
큰 mesh는 표시용으로 간소화하지만 전체 geometry·vertex color는 IPC·cache·저장 mesh에 보존한다.
Matplotlib의 평균색·shading preview는 원본 GLB texture 렌더링과 다르다.

### 입력 cache와 영속 asset의 차이

- mesh asset UUID는 선택한 고정 입력에 연결된다. 같은 선택의 재요청은 같은 UUID를 사용한다.
- 추론 cache는 **같은 asset ID + 정확히 같은 입력 fingerprint**일 때만 재사용한다.
  `depth_source`·`texture_baking`도 fingerprint에 포함한다. prompt 적용·후보 변경은 입력 연결을 해제한다.
- `runs/meshes/`의 NPZ는 입력별 cache이고 PLY/GLB는 저장 mesh다. 파일은 IPC를 대체하는 전송 수단이 아니다.
- [AssetStore](../grasp_workbench/assets.py)는 기존 catalog와 새 저장 asset을 합친다.
  VLM/Task의 영속 저장은 `runs/assets/<asset_id>/object.ply` 또는 `object.glb`와 `manifest.json`에 이름·prompt·단위·scale을 보존한다.
  입력 cache를 지워도 이 영속 asset은 남는다. `vlm.asset_dir`를 삭제하면 새 저장 물체의 기억도 사라진다.
- 저장 asset을 다시 쓸 때도 새 SAM3 관측·FP 등록을 수행한다. 과거 mask·pose나 실물 identity를 재사용하지 않는다.

관련 코드: [SAM3D adapter](../grasp_workbench/models/sam3d.py), [mesh IPC](../grasp_workbench/ipc/sam3d.py).

<a id="foundationpose"></a>
## FoundationPose: 수동 분기·seed·연속 추적

### 기존 mesh 또는 새 mesh 선택

SAM3 후보를 고른 뒤 FoundationPose 탭에서 A/B별로 **이름으로 mesh 선택**, **파일 찾기**, 직접 경로 입력 중 하나를 사용한다.
외부 mesh의 단위는 사용자가 `scale → m`에 지정한다. 이름 목록의 기존 mesh는 catalog의 단위 선언을 사용한다.
보관한 21개 물체의 원본·최적화 GLB와 Table/Leg OBJ는 [mesh 목록](../assets/meshes/README.md)과
[catalog](../assets/meshes/catalog.json)에 있다. 저장 mesh를 선택해도 과거 pose를 가져오지는 않는다.

| 버튼 | 동작 |
|---|---|
| **추적 시작·재등록** | 선택 mesh와 현재 SAM3 후보의 원본 RGB-D/mask로 FP 등록 |
| **Mesh 생성 후 추적** | 선택 후보로 SAM3D 생성 → 완료 mesh·scale `1.0` 입력 → 같은 원본 RGB-D/mask로 FP 최초 등록 |
| SAM3D 탭의 **Mesh 생성** | mesh만 생성. FP 자동 시작 없음 |
| **추적 중지** | 해당 FP session 중지. 생성 대기 중이면 완료 후 FP 자동 시작만 취소하고 생성 mesh는 보존 |

수동 분기는 사용자가 버튼으로 정한다. 파일 존재 여부나 물체 identity로 자동 판정하지 않는다.
Mesh 생성 후 추적은 SAM3D·FP가 모두 켜진 경우 사용할 수 있다. 이후에는 수신 카메라 RGB-D로 tracking한다.
A/B session은 별도이며 scorer/refiner는 FP worker 안에서 공유한다.

### 현재/SAM3D pose로 seed refinement

FoundationPose 탭의 **고급**에서 기본 mask register 외에 **SAM3D pose seed** 또는 **현재 / 저장 FP seed**를
선택한 뒤 **Seed로 정밀화**할 수 있다.
같은 mesh·scale의 pose와 그 pose를 얻은 **정확한 원본 RGB-D**를 사용한다. 최신 카메라 frame으로 seed 입력을 바꾸지 않는다.
SAM3D seed는 생성 원본 mask를 함께 보내고 현재/저장 FP seed는 `mask: null`도 허용한다.
기본 `register` 모드는 계속 mask를 요구한다.

원본 object frame과 FP 내부 centered-mesh frame은 다음 식으로 구분한다.

```text
pose_last = camera_from_object @ inverse(get_tf_to_centered_mesh())
```

같은 원본 frame에서 `track_one` refinement 후 정상 tracking으로 이어진다.
과거 seed 목록·자동 seed 탐색·시간 gate를 추가한 기능은 아니다.

### 추적 검증과 회복

- 추적 시작 때 선택한 mask bbox를 기준으로 이후 SAM3 후보를 bbox IoU 우선, 동률이면 score로 연결한다.
  기존 IoU·면적비·pending 후보 확인 방식을 사용하며 UI의 전체 후보 순서는 바꾸지 않는다.
- SAM3 증거마다 pose origin의 투영 위치가 bbox ±40px 안인지 검사한다. 유효 depth가 16픽셀 이상이면
  pose z가 mask depth min/max ±0.25m 안인지도 검사한다. **검증 불일치 5회**면 recovering으로 전환한다.
  이는 카메라 5프레임이 아니라 SAM3 증거 5회다. SAM3 미검출만으로 정상 FP를 중지하지 않는다.
- SE(3) 유효성·translation jump 0.25m 검사와 추적 예외도 recovering으로 표시한다.
  새 SAM3 원본 RGB-D/mask로 재등록한 결과를 다시 검증해 tracking으로 복귀한다.
- A/B 상태는 독립이며 불일치 횟수·회복 사유를 표시한다. 추적 시작의 prompt revision에 대응하는 SAM3 결과만 사용한다.
  prompt를 새로 적용해 검증 대상을 바꾸려면 해당 슬롯의 추적 시작·재등록을 다시 요청한다.
- ROI box fallback·별도 registration 프로세스·pose history·cooldown·EKF는 연결하지 않았다.
  같은 프로세스의 순차 처리이므로 A 등록 중에는 B tracking도 기다린다.

### 영상과 3D plot

영상은 **최신 카메라 RGB + 최신 SAM3 mask·box + 슬롯별 최신 pose 축**을 합성한다.
FP가 처리한 과거 RGB로 화면을 되돌리지 않는다. sequence를 각각 표시하며 시간 매칭·보간은 하지 않는다.
pose 축 길이는 5cm, X=빨강/Y=초록/Z=파랑이며 물체 ID를 표시한다.

오른쪽 plot은 카메라 기준 mesh·XYZ·회전축을 함께 표시한다. 입력 scale을 정점에 한 번 적용한 뒤
`v_camera = R @ v_scaled + t`로 변환하며 FP와 같은 GLB/OBJ/PLY scene 변환을 사용한다.
물체당 약 2,000개 삼각형의 표시용 mesh·artist를 재사용하고 원본 FP mesh는 바꾸지 않는다.
pose가 없는 registering/recovering 및 추적 중지에는 해당 mesh·축을 제거한다. 영상에는 축을 겹치며
mesh 전체의 2D 투영이나 base 좌표 변환 결과라고 해석하지 않는다.

관련 코드: [FP adapter](../grasp_workbench/models/foundationpose.py), [FP worker](../grasp_workbench/workers/worker_foundationpose.py).

<a id="vlm"></a>
## VLM: Task·Source/Target·policy

backend 설치·인증·모델 선택은 [설정 안내](setup.md)를 따른다. 세 backend 모두 현재 RGB·지시문·공개
asset/policy 텍스트 catalog를 입력받는다. depth·mesh·과거 참조 이미지·과거 pose는 provider로 보내지 않는다.

### 자동 Task 흐름

1. YAML에서 VLM을 켜고 상단에 설명을 입력한 뒤 **인식·진행**을 누른다.
   비워두면 보이는 물체를 최대 두 개 찾는다. 버튼 요청당 한 장이며 카메라 매 프레임 호출하지 않는다.
2. 짧은 영어 SAM3 prompt를 반영한다. 예: `brown packing tape roll`, `yellow tuna can`.
3. 기존 asset ID가 선택되면 저장 mesh를 재사용하고 현재 SAM3 원본 RGB-D/mask로 FP를 등록한다. SAM3D는 생략한다.
4. 새 물체이면 현재 SAM3 후보로 설정된 방식의 SAM3D mesh를 생성·영속 저장한 뒤 FP를 시작한다.
   최초 생성·등록은 같은 원본 RGB-D/mask를 사용한다.
5. FP 완료 후 활성화된 기능 범위에서 **Task Source의 grasp cache 조회/생성 → 조합 선택**으로 이어진다.
   Target은 pose 참조 대상이다. 자동 grasp 생성 대상은 Source이며, 다른 물체에도 팔을 배정했다면 그 물체의 bundle도 필요하다.
6. FP가 꺼져 있으면 prompt·semantic만 반영하고 mesh/FP 자동 경로는 실행하지 않는다.
   수동으로 진행할 때는 각 worker 탭의 prompt·후보·mesh·추적 버튼을 사용한다.

필요한 SAM3·SAM3D·FP·GraspGenX·선택 worker는 각각 켜야 한다. bbox는 SAM3 후보 선택을 돕는 정보이지
mask·6D pose 측정이 아니며 catalog 이름·prompt에 따른 asset ID도 동일 실물·형상 정확도의 보장이 아니다.

### 의미 정보와 수동 역할 편집

| 결과 | 내용 |
|---|---|
| `objects` | 최대 두 물체의 label·영어 SAM3 prompt·bbox·선택적 asset hint·현재 `observation_id` |
| `task_binding` | 움직이거나 잡을 Source와 참조 Target의 관측 ID·`intent` |
| `policy_selection` | `task_type`, 의도 출처 explicit/inferred/unclear, 선택적 `policy_id`·이유 |
| `coordination` | single/dual/unknown, 팔별 관측·역할·`grasp_region`, 상대 동작·phases·reference_state·설명 |

관측 ID는 이번 이미지 안의 참조이며 asset ID·영구 tracking ID가 아니다. **Source/Target은 좌/우 팔이 아니다.**
두 팔이 같은 물체를 참조할 수도 있다. 불명확한 팔·대상·제약은 unknown/null/빈 목록으로 남기고 A/B 순서로 채우지 않는다.
intent·잡을 부위·상대 제약은 언어적 의미 정보이지 수치 pose·운동 제약·접촉 측정·실행 허가가 아니다.

상단 접이식 **작업 상세** 안의 Task / 팔 / 제약 / policy 결과는 읽기 전용 JSON이다.
같은 영역의 **Source/Target 편집기**에서 현재 관측 ID의 역할을
바꿀 수 있으며 Source와 Target은 달라야 한다. Target 없음도 가능하다. 역할 수정은 이전 coordination을 해제하고
관측 ID 교환이나 팔 재배정으로 추정하지 않는다. 이후 수동 left/right/dual 선택 또는 VLM 재분석으로 팔 정보를 정한다.

**선택한 역할로 진행**은 VLM을 재호출하지 않고 선택한 역할의 pose를 준비해 Source grasp cache 조회/생성·선택을 명시적으로 요청한다.
지시문 편집은 마지막 semantic을 보존하고 **재분석 필요**로 표시한다. 수동 prompt·후보·mesh 수정으로 관측 연결이
해제돼도 이전 의미 결과와 현재 연결 상태를 구분한다. 활성 관측은 로컬 `instance_id`에 연결한다.

### 선택적 policy catalog

`vlm.policy_catalog: null`이면 목록·checkpoint가 필요 없다. 사용할 때는 로컬 JSON 목록 경로를 지정한다.

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

training asset은 설명용 참조이지 현재 관측 ID가 아니다. `checkpoint`는 JSON 폴더 기준의 로컬 표시/export용 참조이며
앱이 로드·실행하지 않는다. provider·IPC에는 위 필드 중 checkpoint를 제외한 공개 의미 필드만 보내고 training 참조도
asset_id·name만 남긴다. 개인 경로·추가 sidecar는 보내지 않으며 맞는 정책이 없으면 null을 유지한다.

### 유지하는 Session·export helper API

기본 UI에는 Session 저장·불러오기나 실행 요청 export 버튼을 두지 않는다. 기존 helper API와
저장 기록은 유지하며 아래 계약도 바꾸지 않는다. 여기서 Session은 로그인이나 worker 실행 세션이 아닌 의미 기록이다.

- 저장 API: 지시문·마지막 판단·관측→instance 연결·geometry/FP/grasp 참조·provenance를 JSON으로 저장한다.
- 불러오기 API: 저장된 기록을 표시하고 이전 A/B 활성 연결을 해제해 재분석 필요로 표시한다.
  현재 관측·pose가 아니며 불러오기만으로 모델·tracking·로봇 실행을 시작하지 않는다.
- 실행 요청 export API: 로컬 instance 참조를 포함한 JSON이다. `execution: "request_only"`,
  `robot_command_published: false`이며 실행·허가·IK 성공을 뜻하지 않는다.

이는 [RGB-D 원본 고정·Snapshot helper](#snapshot)와 별개다. 기본 Session 폴더는 `runs/tasks/`이며 `vlm.session_dir`로 바꾼다.
관련 코드: [semantic 계약](../grasp_workbench/semantic.py), [Session](../grasp_workbench/semantic_session.py),
[policy catalog](../grasp_workbench/policy_catalog.py).

<a id="graspgenx"></a>
## GraspGenX: 후보 생성·geometry·cache

GraspGenX는 mesh-only object-local 후보를 만든다. FP·VLM 없이도 파일에서 후보를 생성·표시할 수 있고,
양팔 조합·선택적 도달성 확인은 별도의 [grasp_selection](#grasp-selection)이 담당한다.

1. GraspGenX 탭 A/B의 이름 목록·파일 찾기·직접 경로에서 mesh를 고르고 **scale → m**를 설정한다.
   활성 FP session이 있으면 그 session의 실제 mesh·scale을 사용한다.
2. **후보 생성**은 같은 geometry의 저장 후보를 재사용하거나 없으면 생성한다.
   **다시 생성**은 cache를 건너뛴다. seed가 고정되어 있으면 결과가 같을 수도 있다.
3. dropdown에서 index·score를 보고 고른다. FP가 없으면 object-local mesh·gripper·축을 표시하고 visual이 없으면 축만 표시한다.
4. FP가 있으면 `camera_from_gripper = camera_from_object @ object_from_canonical_gripper`로 합성한다.
   영상에는 선택 grasp 축, 3D plot에는 mesh·pose 축·gripper를 표시한다. FP pose 이동만으로 후보를 재생성하지 않는다.

**모델 score는 IK 성공·장면 충돌 없음·양팔 동시 실행 가능성·실행 허가가 아니다.**

### 추론 설정과 저장

`settings: {}`는 이전 sampling·seed·다양성 선택 기본값을 유지한다: seed 42, `num_grasps: 1000`,
`topk_num_grasps: 100`, `min_grasps: 40`, `max_tries: 6`. 다양성 기준은 translation 0.01m·rotation 15°다.
`max_tries`는 기존 모델 추론 옵션이지 새 worker 자동 retry 정책이 아니다. non-default position clip 호환도
adapter의 sampler instance에만 적용하며 upstream 소스는 바꾸지 않는다.

cache identity는 asset ID·mesh SHA-256·scale·gripper/descriptor·model·추론 설정이며 **카메라 pose는 포함하지 않는다**.
`runs/grasps/<cache_key>/`에는 선택 후보 `grasps.npz`, 원본 후보·sample points의 `candidate_pool.npz`,
좌표 convention·provenance의 `metadata.json`을 저장한다. 같은 입력의 다음 요청에서 재사용한다.
원본 object 원점을 바꾸지 않으며 scale을 geometry에 한 번 적용한다. 저장된 metre grasp translation에 scale을 다시 곱하지 않는다.

### 이전 Isaac YAML bundle

A/B **기존 후보 불러오기 · bundle**은 새 NPZ metadata와 원본 `metadata.json`이 함께 있는 Isaac grasp YAML을 읽는다.
mesh hash·scale·gripper/model·설정 identity와 candidate hash를 검증하고 confidence·원래 후보 ID를 유지한다.
YAML 단독 파일에서 빠진 provenance·raw pool을 만들어내지 않는다. 기존 파일을 덮어쓰지 않으며
새 추론 결과는 계속 이 repo의 NPZ·JSON으로 저장한다.

관련 코드: [GraspGenX adapter](../grasp_workbench/models/graspgenx.py), [bundle 저장소](../grasp_workbench/grasp_assets.py).

<a id="grasp-selection"></a>
## 한팔·양팔 선택과 선택적 endpoint IK

`grasp_selection`은 이미 생성하거나 불러온 bundle에서 후보를 고른다. 선택 버튼 자체가 후보를 자동 생성하지 않는다.
VLM/Task 흐름의 Source cache 조회/생성은 앞선 GraspGenX 단계에서 수행한다.

1. Grasp Selection 탭에서 **수동 single · left/right** 또는 **수동 dual · 같은 물체**와 대상 A/B를 고른다.
   VLM의 현재 `assignments`를 사용할 수도 있다. 배정된 물체마다 기존 bundle이 필요하다.
2. **후보 선택·다시 평가**로 geometry·bundle·현재 물체 pose snapshot을 보낸다.
3. 결과 dropdown에서 조합·score·IK 상태를 확인한다. single은 선택한 **effective object-local grasp**를 고정하고
   같은 **후보 선택·다시 평가** 버튼 또는 명시적 재선택 때 갱신한다. 중복 평가 버튼은 두지 않는다.
   원본 FP tracking과 최신 RGB는 계속 갱신된다.
   pose 이동·semantic 갱신만으로 선택 요청이나 후보 생성을 반복하지 않는다.

| 경우 | 선택 기준 |
|---|---|
| single, 기본 `preference: top_down` | 실제 base-frame 접근 방향 top-down 50% + 모델 30% + 도달성 20% |
| single, IK 이전 | top-down/model 가용 가중치를 정규화하고 2cm·10° 다양성 선택 후 `top_k` 적용 |
| `preference: legacy` | 기존 geometry/도달성 점수 경로 |
| 같은 물체 dual | 서로 다른 후보의 ordered pair·gripper 대칭 변형·가능할 때 COM 균형 proxy |
| 서로 다른 물체 dual | 두 모델 score 평균. 서로 다른 물체에 하나의 COM 점수를 적용하지 않음 |

COM은 균일 밀도 volume 또는 bounding-box 중심 proxy다. base 기준 중력 방향이 있을 때 투영 균형을 계산하며
물리적 안정성·force closure의 보장이 아니다. `grasp_region`·언어적 상대 제약은 수치 접촉/운동 제약으로 변환하지 않는다.

기본 `top_k: 32`, pregrasp 0.15m, retreat/advance 0은 이전 table-leg 예제의 값이지 모든 로봇의 보편 기본값이 아니다.
자신의 모델·gripper·TCP에 맞춰 설정한다. 로봇이 없으면 canonical 후보 선택·표시를 유지하고
base 변환·joint seed·TCP 보정·top-down 방향·IK 성공 수치를 임의 생성하지 않는다.

### 로봇 좌표와 확인 범위

Grasp Selection 탭의 **고급**에서 로봇 기준 pose·IK 결과 상세를 확인한다.
선택적 adapter가 같은 grasp_selection worker 안에서 상태 수신·현재 FK·camera calibration·endpoint IK를 담당한다.
일반 adapter는 robot YAML의 `backend: module:factory`이며 `factory(config, config_dir=...)`로 생성한다.
[FFW 예제](../examples/robot_ffw.yaml)는 선택 사항이다. 필요한 별도 환경·runtime/model/calibration은 [설정 안내](setup.md)를 따른다.

```text
base_from_object = base_from_camera @ camera_from_object
base_from_tcp = base_from_object @ object_from_canonical_gripper @ canonical_gripper_from_tcp
```

canonical gripper·TCP·solver tool frame은 구분하며 solver에 넘길 때 tool frame으로 별도 변환한다.
pregrasp/retreat/advance는 각 frame의 정의대로 적용한다. 현재 관절 seed로 단팔 또는 동시 양팔의
**pregrasp/grasp endpoint**를 확인하고 선택된 IK 변형은 effective pose에 반영한다.
검사한 self-collision 여부는 표시하지만 장면 충돌·경로·trajectory·실제 이동 가능성의 검증은 아니다.
카메라·base의 최적 위치를 탐색하거나 actuator 명령을 발행하지 않는다.

관련 코드: [선택 모델](../grasp_workbench/models/grasp_selection.py),
[robot adapter 경계](../grasp_workbench/models/robot_reachability.py).

<a id="grasp-fk"></a>
## 표시용 파지 관계 고정과 hand FK

Grasp Selection 탭의 **고급**에서 base-frame 선택 결과의 **파지 관계 고정(표시용)**을 누르면
선택 당시의 `object_from_tcp` 관계를 고정한다.
같은 선택 worker가 실측 hand FK와 state sequence로 물체 pose를 갱신한다.

```text
base_from_object_fk = measured_base_from_tcp @ inverse(frozen_object_from_tcp)
```

이는 **계획한 파지 관계를 유지한다고 가정한 read-only 표시**다. 접촉·gripper 폐쇄·미끄러짐 측정 결과가 아니다.
실측 state가 없으면 waiting이며 명령 관절값으로 대신 채우지 않는다. **파지 관계 해제**로 명시적으로 끝낸다.
별도 FK worker는 추가하지 않는다.

원본 camera/FP 결과는 보존하고 base_link 3D 표시 모드와 구분한다. 다음 **명시적 다시 평가**에서만 동일
`instance_id`의 FK pose를 base 입력으로 사용할 수 있다. 자동 재평가·카메라 pose 위조·로봇 명령은 하지 않는다.

## Optional LeRobot data recording

Enable `recording.enabled` in YAML and prepare the optional recorder Python environment described in
[recording.md](recording.md). The **Recording** tab provides explicit Start Recording, Save Episode and Discard Episode
requests; enabling the worker alone does not start recording. The worker receives data over iceoryx2 and never
opens another camera or runs another pose model. It remains independently selectable in AprilCube mode.

With `recording.source: pose`, RGB-D and A/B poses come from the same `PoseResult`, including unavailable results.
No frames are recorded until a pose worker publishes results. `source: camera` records raw camera RGB-D without
pose labels. Original frame IDs and timestamps are preserved alongside LeRobot's nominal dataset timeline;
latest-only subscribers do not guarantee every published camera frame is saved. There are no fabricated robot actions,
training jobs or automatic Hub uploads. See [stored features and episode behavior](recording.md).

<a id="ipc"></a>
## 프로세스 구조와 IPC 계약

```text
main: 프로세스 시작·종료
├── camera worker ── RGB-D ───────────────────────┐
├── SAM3 worker ── A/B mask·refine ────────────────┤
├── SAM3D worker ── 요청별 mesh ───────────────────┤
├── FoundationPose worker ── A/B session ──────────┤  iceoryx2
├── VLM worker ── 단발 provider adapter ───────────┤
├── GraspGenX worker ── object-local 후보 ─────────┤
├── grasp_selection worker ── 선택·선택적 FK/IK ───┤
├── data_recorder worker ◀── camera / pose ───────┤ → local LeRobotDataset
└── UI worker ◀──────────────────────────────────┘
    ├── receiver thread: 구독·발행·배열 소유권·최신 결과 수신
    └── Qt main thread: 영상·3D plot·조작
```

설정한 worker만 시작한다. refine은 SAM3, seed는 FP, 표시 FK는 grasp_selection 내부 기능이며
기능마다 작은 worker를 추가하지 않는다. plot도 별도 프로세스가 아니다.

`<service>`는 YAML의 `ipc.service`다. 관련 계약은 [ipc 디렉터리](../grasp_workbench/ipc/)에 있다.

| 기능 | `<service>` 뒤 채널 |
|---|---|
| camera | 기본 service 자체 |
| SAM3 | `/sam3/prompts`, `/sam3/results` |
| SAM3 보정 | `/sam3/refine/requests`, `/sam3/refine/results` |
| SAM3D | `/sam3d/requests`, `/sam3d/results` |
| FP | `/foundationpose/control`, `/foundationpose/results` |
| VLM | `/vlm/requests`, `/vlm/results` |
| GraspGenX | `/graspgenx/requests`, `/graspgenx/results` |
| 선택·표시 FK | `/grasp_selection/requests`, `/grasp_selection/results` |

- 카메라는 uint8 RGB·float32 metre depth를 원본 해상도로 보낸다. camera/stream ID·sequence·SDK timestamp와
  clock domain·host 수신 monotonic 시간·intrinsics·depth 원본 시간/단위도 보존한다.
- ZED는 left image 정렬 depth, RealSense는 color에 align한 depth를 사용한다. invalid NaN/inf/0을 임의 측정값으로 바꾸지 않는다.
  ZED IMAGE의 SDK UNIX ns를 sensor exposure나 host monotonic 시각으로 해석하지 않는다.
- frame은 길이 prefix·작은 JSON metadata·raw RGB/depth다. worker 간 전송에 JPEG/PNG·base64·pickle을 쓰지 않는다.
  mesh는 raw vertices/faces/colors, grasp는 float64 `N×4×4`·`N` score를 전송한다.
- 프로세스 간 전송은 **iceoryx2만** 사용한다. 별도 수동 SHM·Queue·Pipe는 없다. iceoryx2 자체는 공유 메모리를 사용한다.
- 외부 VLM 호출 경계에서만 API용 PNG/base64 또는 CLI용 임시 PNG로 변환한다. 내부 raw IPC를 바꾸는 것이 아니다.
- SDK→발행 loan과 수신 loan→소유 배열 복사는 있다. end-to-end 무복사를 주장하지 않는다.
- 최신 frame 확인용 작은 버퍼이므로 느린 수신자의 모든 프레임 보존을 약속하지 않는다.
  여러 카메라 예제를 실행할 때는 `ipc.service`를 구분한다.

<a id="scope"></a>
## 구현 범위와 해석 주의

이 앱은 수동/Task 관측 처리·원본 고정/refine·mesh/pose·grasp cache·후보 선택·선택적인 FK/endpoint IK 예제다.
원본 `object_robot_ui`의 6개 탭을 동일하게 복제한 앱은 아니다.

**제외:** 학습·새 제약 연구·학습 policy 추론/실행·actuator 명령·로봇 이동·TrajOpt·world scene·trajectory.
언어적 semantic·후보 score·endpoint IK 중 어느 것도 로봇 실행 허가나 작업 성공 보장이 아니다.
실제 장치/모델 확인과 fake/native/offline 검증의 차이는 [검증 이력](verification.md)을 참고한다.
