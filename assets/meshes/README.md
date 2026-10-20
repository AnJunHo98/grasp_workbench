# 기존 FFW mesh 보관본

`ffw_object_pose_system/assets`의 정식 물체 라이브러리에서 **복사**했다. 원본 파일은 변경하지 않았다.

- `ffw_objects/<asset_id>/object.glb`: 저장된 물체 21개의 원본 mesh.
- `ffw_objects/<asset_id>/object.optimized.glb`: 같은 물체의 기존 최적화 mesh.
- `table_leg/`: `Table001.obj`, `Leg001.obj`, `Leg001_colored.obj`.
- `catalog.json`: 물체 이름·prompt·파일 상대 경로·기존 단위 정보·크기·SHA-256 목록.

총 **45개 mesh / 303,867,548 bytes (약 289.8 MiB)**다. GLB의 texture·buffer는 내부 포함이며,
OBJ에는 외부 MTL·texture 참조가 없다. 복사 후 모든 mesh의 SHA-256이 원본과 일치함을 확인했다.

좌표·색상·스케일은 변환하지 않았다. 기존 manifest의 `metre` / `mesh_scale_to_m: 1.0` 선언을
목록에 보존했으며, 형상 정확도나 현재 카메라 기준 pose를 새로 검증했다는 뜻은 아니다.
촬영 snapshot·과거 pose·로봇 설정·grasp cache·가중치·로그 중복본은 가져오지 않았다.

이 디렉터리는 현재 SAM3D cache와 별개다. FP UI의 A/B **이름으로 mesh 선택**에서 물체 21개와
Table/Leg 3개를 선택할 수 있다. 선택하면 해당 저장 mesh 경로와 scale이 입력되며, Browse도 유지한다.
표시 이름만 catalog에서 읽고 UUID 경로·mesh 파일 바이트는 바꾸지 않는다.
원래 FFW application이나 개인 경로 없이 이 repo 안의 파일을 사용한다.

- **테이프**: `ffw_objects/70b913cc9042458585b173a9a07bb111/object.optimized.glb`
- **중앙의 작은 노란 동원참치 캔**: `ffw_objects/e76130be39c44e3282d47be1056760f5/object.optimized.glb`
