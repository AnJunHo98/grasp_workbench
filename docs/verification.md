# 검증 기록

[← README](../README.md)

## 기록 읽는 방법

- 아래 이력은 기존 README의 검증 기록을 원문 그대로 옮긴 것이다. 테스트 수·환경·실행 시간은
  각 확인 당시의 기록이며, 최신 테스트 수나 필수 설치 환경을 뜻하지 않는다.
- `runs/...`를 비롯한 이력 속 로컬 산출물 경로는 **저장소 루트 기준**이다. 모두 Git 제외 기록이며,
  이 문서로 옮겼다고 파일이 커밋되거나 공유되는 것은 아니다.
- fake backend·native IPC·실제 모델·실제 장치 확인은 서로 다른 검증 범위다. 아래의 한계와 미검증 항목도
  함께 읽어야 한다. 원문에는 상대 Markdown 링크가 없어 경로 변경 없이 보존했다.

## 최근 기록 요약

- 최근 기록은 **937개 테스트 통과**, Ruff·`git diff --check` 통과다.
- Anthropic은 fake SDK·독립 worker/native iceoryx2와 실제 SDK의 **로컬 mock HTTP** 요청 직렬화·응답 파싱을
  확인했다. 실제 Claude 호출·실제 인증·과금·서비스 접근은 확인하지 않았다.
- 새 PC의 전체 설치나 현재 하드웨어 연결·실물 grasp 성공을 검증했다는 뜻은 아니다.
  과거 실제 카메라/GPU 확인과 각 미검증 범위는 아래 이력에 구분해 남겼다.

## 상세 이력

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
