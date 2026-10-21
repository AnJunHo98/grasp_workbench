"""Camera RGB, optional generated meshes and pose displays without model SDKs."""
from copy import deepcopy
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..core.worker_base import log_record
from ..workers.worker_camera import DEFAULT_SERVICE


def _box_iou(first, second):
    low, high = np.maximum(first[:2], second[:2]), np.minimum(first[2:], second[2:])
    intersection = float(np.maximum(high - low, 0).prod())
    union = float(np.maximum(np.subtract(first[2:], first[:2]), 0).prod()
                  + np.maximum(np.subtract(second[2:], second[:2]), 0).prod()) - intersection
    return intersection / union if union else 0.


class RgbImageView(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__("카메라 영상 대기 중 · iceoryx2 발행 worker를 실행하세요", parent)
        self.image = None
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #15171a; color: #d8dee9")

    def set_rgb(self, rgb, *, redraw=True):
        pixels = np.ascontiguousarray(rgb)
        self.image = QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                                 QtGui.QImage.Format.Format_RGB888).copy()
        if redraw:
            self._draw_image()

    def _draw_image(self):
        if self.image is not None:
            self.setPixmap(QtGui.QPixmap.fromImage(self.image).scaled(
                self.contentsRect().size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._draw_image()


class CameraWindow(QtWidgets.QMainWindow):
    def __init__(self, service=DEFAULT_SERVICE, frames=0, timeout=0, *, sam3=False, sam3d=False,
                 foundationpose=False, aprilcube=False, vlm=False, graspgenx=False, grasp_selection=False, recording=False, asset_dir=None, policy_catalog=None,
                 session_dir=None, receiver_factory=None, sam3d_depth_source="depth", sam3d_texture_baking=False):
        super().__init__()
        if receiver_factory is None:
            from .receiver import LatestCameraReceiver

            receiver_factory = LatestCameraReceiver
        self.worker_enabled = {
            "camera": True, "sam3": bool(sam3), "sam3d": bool(sam3d),
            "foundationpose": bool(foundationpose), "graspgenx": bool(graspgenx),
            "grasp_selection": bool(grasp_selection), "aprilcube": bool(aprilcube),
            "recording": bool(recording),
        }
        if aprilcube:
            sam3 = sam3d = foundationpose = vlm = graspgenx = grasp_selection = False
            self.worker_enabled = {key: key in {"camera", "aprilcube"} or key == "recording" and recording
                                   for key in self.worker_enabled}
        sam3 = sam3 or sam3d or foundationpose or vlm
        graspgenx = graspgenx or grasp_selection
        options = {"sam3": True, "sam3d": True} if sam3d else {"sam3": True} if sam3 else {}
        if foundationpose:
            options["foundationpose"] = True
        if aprilcube:
            options["aprilcube"] = True
        if vlm:
            options["vlm"] = True
        if graspgenx:
            options["graspgenx"] = True
        if grasp_selection:
            options["grasp_selection"] = True
        if recording:
            options["recording"] = True
        self.receiver = receiver_factory(service, **options)
        self.sam3, self.sam3d = sam3, sam3d
        self.sam3d_depth_source, self.sam3d_texture_baking = sam3d_depth_source, sam3d_texture_baking
        self.foundationpose = foundationpose
        self.aprilcube = aprilcube
        self.recording = recording
        self._recording_request_id, self._recording_state = None, None
        self._aprilcube_overlay = None
        self.vlm = vlm
        self.graspgenx = graspgenx
        self.grasp_selection = grasp_selection
        self._selection_request, self._selection_result = None, None
        self._selection_sources, self._selection_semantic_id = {}, None
        self._selection_lock = None
        self._task_flow, self._task_role_updating = None, False
        self._manual_task_objects = {}
        self._grasp_fk, self._fk_request_id = None, None
        self._fk_attachment, self._fk_meshes, self._fk_previews, self._fk_artists = None, {}, {}, {}
        self._snapshot, self._snapshot_active = None, False
        self._refine_request, self._refine_key, self._refine_source = None, None, None
        self._grasp_requests, self._grasp_results, self._grasp_states = {}, {}, {}
        self._grasp_previews, self._grasp_object_meshes, self._grasp_artists = {}, {}, {}
        self._grasp_catalog = ()
        self._vlm_request_id = None
        self._vlm_instruction_at_request = ""
        self._vlm_plan, self._vlm_mesh_jobs, self._vlm_states = {}, {}, {}
        self._semantic_session, self._semantic_updating = None, False
        if vlm:
            from ..assets import AssetStore
            from ..policy_catalog import load_policy_catalog

            self.asset_store = AssetStore(asset_dir or Path(__file__).resolve().parents[2] / "runs/assets")
            self.policy_catalog = load_policy_catalog(policy_catalog)
            self.session_dir = Path(session_dir or Path(__file__).resolve().parents[2] / "runs/tasks")
        self._prompts, self._sam3_result, self._latest_frame = None, None, None
        self._candidate_sources, self._mesh_bindings, self._mesh_results = {}, {}, {}
        self._mesh_jobs, self._mesh_states, self._mesh_dirty = {}, {}, set()
        self._mesh_pose_jobs, self.mesh_pose_buttons, self.pose_start_buttons = {}, {}, {}
        self._pose_targets, self._pose_objects = {}, {}
        self._pose_meshes, self._pose_mesh_errors = {}, {}
        self._pose_mesh_artists, self._pose_axis_artists, self._pose_label_artists = {}, {}, {}
        self._plot_mode = None
        self._pose_result = None
        self.frames, self.timeout = frames, timeout
        self.exit_code, self.received_count = 0, 0
        self._closing, self._ready_logged = False, False
        self._started = time.monotonic()
        self.setWindowTitle("Grasp Workbench · iceoryx2")
        self.resize(1440, 850)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer_layout = QtWidgets.QVBoxLayout(central)

        self.vlm_header = QtWidgets.QWidget()
        self.vlm_header.setObjectName("vlm-header")
        header_layout = QtWidgets.QVBoxLayout(self.vlm_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("VLM"))
        if vlm:
            self.vlm_instruction = QtWidgets.QLineEdit()
            self.vlm_instruction.setPlaceholderText("작업 지시문 · 비우면 보이는 물체 최대 2개")
            self.vlm_instruction.textEdited.connect(self._semantic_instruction_edited)
            row.addWidget(self.vlm_instruction, 1)
            self.vlm_button = QtWidgets.QPushButton("인식·진행")
            self.vlm_button.setToolTip("현재 RGB 한 장을 인식하고 활성 worker 범위에서 Task를 진행합니다")
            self.vlm_button.clicked.connect(self.start_vlm)
            self.vlm_button.setEnabled(False)
            row.addWidget(self.vlm_button)
        else:
            instruction = QtWidgets.QLineEdit()
            instruction.setPlaceholderText("AprilCube 모드 · VLM 사용 안 함" if aprilcube else "vlm.enabled: false · YAML에서 설정")
            instruction.setEnabled(False)
            row.addWidget(instruction, 1)
            button = QtWidgets.QPushButton("인식·진행")
            button.setEnabled(False)
            row.addWidget(button)
        header_layout.addLayout(row)
        if vlm:
            self.vlm_status = self._wrapped_label("VLM · 버튼으로 현재 RGB 한 장 인식 · 카메라 대기")
            header_layout.addWidget(self.vlm_status)
        if foundationpose or vlm:
            self.task_details_button, self.task_details = self._fold_section(
                header_layout, "Task / 팔 / 제약 / policy 상세")
            task_layout = QtWidgets.QVBoxLayout(self.task_details)
            task_layout.setContentsMargins(8, 0, 8, 0)
            if vlm:
                self.vlm_enabled = QtWidgets.QCheckBox("VLM 요청 사용 · worker 실행 설정은 YAML")
                self.vlm_enabled.setChecked(True)
                self.vlm_enabled.toggled.connect(self._vlm_mode_changed)
                task_layout.addWidget(self.vlm_enabled)
                self.semantic_summary = self._wrapped_label("Task / coordination · 결과 대기 · A/B=물체 slot, 팔 배정 아님")
                header_layout.insertWidget(2, self.semantic_summary)
                self.semantic_details = QtWidgets.QPlainTextEdit()
                self.semantic_details.setReadOnly(True)
                self.semantic_details.setMaximumHeight(130)
                task_layout.addWidget(self.semantic_details)
                self.task_details_button.toggled.connect(self._semantic_details_toggled)
            row = QtWidgets.QHBoxLayout()
            self.source_role_input, self.target_role_input = QtWidgets.QComboBox(), QtWidgets.QComboBox()
            for title, combo in (("Source", self.source_role_input), ("Target", self.target_role_input)):
                row.addWidget(QtWidgets.QLabel(title))
                combo.addItem("미선택", None)
                combo.currentIndexChanged.connect(self._task_roles_changed)
                combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(12)
                row.addWidget(combo, 1)
            self.task_button = QtWidgets.QPushButton("선택한 역할로 진행")
            self.task_button.clicked.connect(self.start_task_pipeline)
            row.addWidget(self.task_button)
            task_layout.addLayout(row)
            self.task_status = self._wrapped_label("수동 역할 선택 또는 VLM 분석 · 역할과 팔 배정은 별개")
            task_layout.addWidget(self.task_status)
        outer_layout.addWidget(self.vlm_header)

        self.worker_tabs = QtWidgets.QListWidget()
        self.worker_tabs.setObjectName("worker-tabs")
        self.worker_tabs.setFixedWidth(128)
        self.worker_tabs.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.worker_stack = QtWidgets.QStackedWidget()
        self.worker_stack.setObjectName("worker-stack")
        self.worker_pages, self.worker_page_contents, worker_layouts = {}, {}, {}
        worker_titles = {"camera": "Camera", "sam3": "SAM3", "sam3d": "SAM3D",
                         "foundationpose": "FoundationPose", "graspgenx": "GraspGenX",
                         "grasp_selection": "Grasp Selection", "aprilcube": "AprilCube",
                         "recording": "Recording"}
        for key, title in worker_titles.items():
            enabled = self.worker_enabled[key]
            item = QtWidgets.QListWidgetItem(f"{title}\n{'ON' if enabled else 'OFF'}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setSizeHint(QtCore.QSize(120, 48))
            self.worker_tabs.addItem(item)
            scroll = QtWidgets.QScrollArea()
            scroll.setObjectName(f"worker-page-{key}")
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(8, 8, 8, 8)
            page_layout.addWidget(self._wrapped_label(f"{title} · {'ON' if enabled else 'OFF'}"))
            if not enabled:
                reason = ("AprilCube 모드에서는 이 worker를 실행하지 않습니다." if aprilcube and key != "recording" else
                          f"{key}.enabled: false · YAML에서 worker를 선택하세요.")
                page_layout.addWidget(self._wrapped_label(reason))
            content = QtWidgets.QWidget()
            content.setEnabled(enabled)
            content_layout = QtWidgets.QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.addWidget(content)
            page_layout.addStretch()
            scroll.setWidget(page)
            self.worker_stack.addWidget(scroll)
            self.worker_pages[key], self.worker_page_contents[key] = scroll, content
            worker_layouts[key] = content_layout
        self.worker_tabs.currentRowChanged.connect(self.worker_stack.setCurrentIndex)
        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("worker-sidebar")
        sidebar.setMinimumWidth(370)
        sidebar.setMaximumWidth(460)
        sidebar_layout = QtWidgets.QHBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(4)
        sidebar_layout.addWidget(self.worker_tabs)
        sidebar_layout.addWidget(self.worker_stack, 1)
        body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        body.addWidget(sidebar)
        outer_layout.addWidget(body, 1)

        camera_layout = worker_layouts["camera"]
        camera_layout.addWidget(self._wrapped_label(f"iceoryx2: {service}"))
        self.camera_info = self._wrapped_label("카메라 영상 대기 · 수신 metadata에서 장치·해상도 정보를 표시합니다")
        camera_layout.addWidget(self.camera_info)
        camera_layout.addWidget(self._wrapped_label("카메라는 별도 worker에서 자동 연결합니다. 장치와 해상도는 YAML에서 설정합니다."))
        if recording:
            recording_layout = worker_layouts["recording"]
            recording_layout.addWidget(self._wrapped_label("LeRobot dataset · 원본 RGB-D / frame metadata / A/B pose"))
            self.recording_buttons = {}
            for action, title in (("start", "Start Recording"), ("save", "Save Episode"), ("discard", "Discard Episode")):
                button = QtWidgets.QPushButton(title)
                button.setEnabled(False)
                button.clicked.connect(lambda _checked=False, action=action: self.send_recording_control(action))
                recording_layout.addWidget(button)
                self.recording_buttons[action] = button
            self.recording_status = self._wrapped_label("Recording · worker 연결 대기")
            self.recording_counts = self._wrapped_label("Episode 0 · 0 frames · Total 0 frames")
            self.recording_output = self._wrapped_label("Output · worker 설정 확인 중")
            self.recording_output.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            for label in (self.recording_status, self.recording_counts, self.recording_output):
                recording_layout.addWidget(label)
        self.live_camera_button = QtWidgets.QPushButton("실시간으로 돌아가기")
        self.live_camera_button.setObjectName("live-camera-button")
        self.live_camera_button.setVisible(False)
        self.live_camera_button.clicked.connect(self.show_live_camera)
        outer_layout.insertWidget(1, self.live_camera_button)

        if sam3:
            layout = worker_layouts["sam3"]
            self.prompt_inputs = {}
            sam3_slots = {}
            for key, example in (("A", "cell phone"), ("B", "orange ball")):
                slot = self._slot_panel(layout, key)
                sam3_slots[key] = slot
                field = QtWidgets.QLineEdit()
                field.setPlaceholderText(f"{example} · 비우면 제외")
                self.prompt_inputs[key] = field
                if vlm:
                    field.textEdited.connect(lambda _text, key=key: self._mark_semantic_bindings_stale("prompt 편집", (key,)))
                slot.addWidget(QtWidgets.QLabel("Prompt"))
                slot.addWidget(field)
            self.sam3_apply_button = QtWidgets.QPushButton("적용·분할 시작")
            self.sam3_apply_button.clicked.connect(self.apply_prompts)
            layout.addWidget(self.sam3_apply_button)
            self.sam3_stop_button = QtWidgets.QPushButton("분할 중지")
            self.sam3_stop_button.clicked.connect(lambda: self._set_prompts(()))
            layout.addWidget(self.sam3_stop_button)
            self.prompt_hint = self._wrapped_label("입력 문구를 SAM3에 그대로 전달 · 자동 번역 없음 · 영어 예: cell phone / orange ball")
            layout.addWidget(self.prompt_hint)
            self.sam3_status = self._wrapped_label("SAM3 중지 · A/B는 prompt slot이며 지속 추적 ID가 아닙니다")
            layout.addWidget(self.sam3_status)
            self._set_prompts(())
            self.candidate_inputs, self.mesh_buttons = {}, {}
            for key, slot in sam3_slots.items():
                combo = QtWidgets.QComboBox()
                combo.addItem(f"{key}: SAM3 후보 대기", None)
                combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(12)
                if sam3d:
                    combo.currentIndexChanged.connect(lambda _index, key=key: self._select_mesh_candidate(key))
                if vlm:
                    combo.activated.connect(lambda _index, key=key: self._mark_semantic_bindings_stale("SAM3 후보 변경", (key,)))
                self.candidate_inputs[key] = combo
                slot.addWidget(combo)
                refine = QtWidgets.QPushButton("Mask 보정")
                refine.clicked.connect(lambda _checked=False, key=key: self.edit_mask(key))
                slot.addWidget(refine)
        if sam3d:
            layout = worker_layouts["sam3d"]
            for key in ("A", "B"):
                slot = self._slot_panel(layout, key)
                slot.addWidget(self._wrapped_label("SAM3 탭에서 mask 후보를 선택하세요."))
                button = QtWidgets.QPushButton("Mesh 생성")
                button.setEnabled(False)
                button.clicked.connect(lambda _checked=False, key=key: self.generate_mesh(key))
                self.mesh_buttons[key] = button
                slot.addWidget(button)
                self.prompt_inputs[key].textChanged.connect(lambda _text, key=key: self._mesh_prompt_edited(key))
            self.mesh_object_input = QtWidgets.QComboBox()
            self.mesh_object_input.addItems(("A mesh", "B mesh"))
            self.mesh_object_input.currentIndexChanged.connect(self._plot_selected_mesh)
            layout.addWidget(QtWidgets.QLabel("3D에 표시할 mesh"))
            layout.addWidget(self.mesh_object_input)
            self.mesh_options_label = self._wrapped_label(
                f"SAM3D 설정 · YAML · depth_source={sam3d_depth_source} · texture_baking={str(sam3d_texture_baking).lower()}")
            layout.addWidget(self.mesh_options_label)
            unit = "monocular 예측 scale · 실측 단위 아님" if sam3d_depth_source == "monocular" else "실측 depth 기반 mesh, asset 로컬 좌표(m)"
            self.mesh_status = self._wrapped_label(f"SAM3D · 후보 선택 대기 · {unit}")
            layout.addWidget(self.mesh_status)

        mesh_slots = {}
        if foundationpose or graspgenx or aprilcube:
            catalog_path = Path(__file__).resolve().parents[2] / "assets" / "meshes" / "catalog.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.is_file() else {}
            meshes = [(item["display_name"], item["preferred_mesh"], item["mesh_scale_to_m"])
                      for item in catalog.get("objects", [])]
            if vlm:
                meshes = [(item["name"], item["mesh_path"], item["mesh_scale_to_m"])
                          for item in self.asset_store.list_assets()]
            if graspgenx:
                from ..assets import AssetStore

                self._grasp_catalog = AssetStore(asset_dir or Path(__file__).resolve().parents[2] / "runs/assets").list_assets()
            meshes += [(item["display_name"], item["path"], item.get("mesh_scale_to_m", 1.0))
                       for item in catalog.get("table_leg", [])]
            self.pose_mesh_inputs, self.pose_scale_inputs, self.pose_mesh_choices, self.pose_seed_inputs = {}, {}, {}, {}
            self.aprilcube_xyz_inputs, self.aprilcube_rpy_inputs = {}, {}
            self.pose_advanced, self.pose_advanced_buttons = {}, {}
            mesh_owner = ("foundationpose" if foundationpose else "aprilcube" if aprilcube else
                          "graspgenx" if self.worker_enabled["graspgenx"] else "grasp_selection")
            for key in ("A", "B"):
                slot = self._slot_panel(worker_layouts[mesh_owner], key)
                mesh_slots[key] = slot
                slot.addWidget(QtWidgets.QLabel("Mesh"))
                choice = QtWidgets.QComboBox()
                choice.addItem("이름으로 mesh 선택", None)
                for name, mesh_path, scale_to_m in meshes:
                    choice.addItem(name, {"mesh_path": str(catalog_path.parent / mesh_path), "scale_to_m": scale_to_m})
                choice.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                choice.setMinimumContentsLength(12)
                choice.currentIndexChanged.connect(lambda _index, key=key: self._select_pose_mesh(key))
                self.pose_mesh_choices[key] = choice
                slot.addWidget(choice)
                path = QtWidgets.QLineEdit()
                path.setPlaceholderText("GLB / PLY / OBJ mesh 파일")
                path.textEdited.connect(lambda _text, key=key: self.pose_mesh_choices[key].setCurrentIndex(0))
                self.pose_mesh_inputs[key] = path
                slot.addWidget(path)
                browse = QtWidgets.QPushButton("파일 찾기")
                browse.clicked.connect(lambda _checked=False, key=key: self._browse_pose_mesh(key))
                slot.addWidget(browse)
                scale = QtWidgets.QDoubleSpinBox()
                scale.setDecimals(6)
                scale.setRange(0.000001, 1000000)
                scale.setValue(1)
                self.pose_scale_inputs[key] = scale
                form = QtWidgets.QFormLayout()
                form.addRow("scale → m", scale)
                slot.addLayout(form)
                if foundationpose:
                    for text, callback in (("추적 시작·재등록", self.start_pose), ("추적 중지", self.stop_pose)):
                        button = QtWidgets.QPushButton(text)
                        button.clicked.connect(lambda _checked=False, key=key, callback=callback: callback(key))
                        if callback == self.start_pose:
                            button.setToolTip("현재 mesh와 SAM3 후보로 추적 시작·재등록")
                            self.pose_start_buttons[key] = button
                        slot.addWidget(button)
                    if sam3d:
                        button = QtWidgets.QPushButton("Mesh 생성 후 추적")
                        button.setToolTip("SAM3D 설정으로 mesh를 생성하고 FoundationPose를 시작합니다")
                        button.setEnabled(False)
                        button.clicked.connect(lambda _checked=False, key=key: self.generate_mesh_and_start_pose(key))
                        self.mesh_pose_buttons[key] = button
                        slot.addWidget(button)
                    advanced_button, advanced = self._fold_section(slot, "고급 · pose seed")
                    self.pose_advanced_buttons[key], self.pose_advanced[key] = advanced_button, advanced
                    advanced_layout = QtWidgets.QVBoxLayout(advanced)
                    advanced_layout.setContentsMargins(0, 0, 0, 0)
                    seed = QtWidgets.QComboBox()
                    seed.addItem("SAM3D pose seed", "sam3d")
                    seed.addItem("현재 / 저장 FP seed", "foundationpose")
                    self.pose_seed_inputs[key] = seed
                    advanced_layout.addWidget(seed)
                    refine = QtWidgets.QPushButton("Seed로 정밀화")
                    refine.clicked.connect(lambda _checked=False, key=key: self.refine_pose(key))
                    advanced_layout.addWidget(refine)
                if aprilcube:
                    slot.addWidget(self._wrapped_label("cube_from_object · 큐브 기준 mesh 변환"))
                    for title, fields, names, limit, decimals in (
                        ("XYZ (m)", self.aprilcube_xyz_inputs, "XYZ", 1000, 4),
                        ("RPY (deg)", self.aprilcube_rpy_inputs, "RPY", 360, 2),
                    ):
                        slot.addWidget(QtWidgets.QLabel(title))
                        fields[key] = []
                        row = QtWidgets.QHBoxLayout()
                        for name in names:
                            field = QtWidgets.QDoubleSpinBox()
                            field.setRange(-limit, limit)
                            field.setDecimals(decimals)
                            field.setSingleStep(0.01 if names == "XYZ" else 1)
                            field.setPrefix(f"{name}: ")
                            field.setMinimumWidth(0)
                            field.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
                            fields[key].append(field)
                            row.addWidget(field)
                        slot.addLayout(row)
                    button = QtWidgets.QPushButton("적용")
                    button.clicked.connect(lambda _checked=False, key=key: self.apply_aprilcube_target(key))
                    slot.addWidget(button)
        if foundationpose or aprilcube:
            self.pose_status = self._wrapped_label("AprilCube · continuous pose · empty mesh shows cube axes" if aprilcube else
                                                   "FoundationPose · mesh와 현재 SAM3 후보를 선택하고 Start")
            worker_layouts["aprilcube" if aprilcube else "foundationpose"].addWidget(self.pose_status)
        if graspgenx:
            layout = worker_layouts["graspgenx" if self.worker_enabled["graspgenx"] else "grasp_selection"]
            self.grasp_inputs, self.grasp_scores, self.grasp_buttons = {}, {}, {}
            self.grasp_mesh_labels = {}
            for key in ("A", "B"):
                slot = mesh_slots[key] if mesh_owner in {"graspgenx", "grasp_selection"} else self._slot_panel(layout, key)
                if foundationpose:
                    label = self._wrapped_label("Mesh · FoundationPose 탭의 선택을 사용합니다")
                    self.grasp_mesh_labels[key] = label
                    self.pose_mesh_inputs[key].textChanged.connect(
                        lambda text, label=label: label.setText(f"Mesh · {text or 'FoundationPose 탭에서 선택'}"))
                    slot.addWidget(label)
                if self.worker_enabled["graspgenx"]:
                    for text, regenerate in (("후보 생성", False), ("다시 생성", True)):
                        button = QtWidgets.QPushButton(text)
                        button.clicked.connect(lambda _checked=False, key=key, regenerate=regenerate:
                                               self.generate_grasp(key, regenerate=regenerate))
                        self.grasp_buttons[(key, regenerate)] = button
                        slot.addWidget(button)
                load = QtWidgets.QPushButton("기존 후보 불러오기 · bundle")
                load.clicked.connect(lambda _checked=False, key=key: self.load_grasp_bundle_path(key))
                slot.addWidget(load)
                combo = QtWidgets.QComboBox()
                combo.addItem("후보 대기", None)
                combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(12)
                combo.currentIndexChanged.connect(lambda _index, key=key: self._select_grasp(key))
                self.grasp_inputs[key] = combo
                slot.addWidget(combo)
                score = QtWidgets.QLabel("score: —")
                self.grasp_scores[key] = score
                slot.addWidget(score)
            self.grasp_object_input = QtWidgets.QComboBox()
            self.grasp_object_input.addItems(("A grasp · object local", "B grasp · object local"))
            self.grasp_object_input.currentIndexChanged.connect(self._plot_grasp_local)
            layout.addWidget(QtWidgets.QLabel("3D에 표시할 후보"))
            layout.addWidget(self.grasp_object_input)
            self.grasp_status = self._wrapped_label("GraspGenX · mesh 선택 → 생성 · A/B는 물체 slot이며 좌/우 팔이 아닙니다" if self.worker_enabled["graspgenx"] else
                                                   "저장된 grasp bundle 불러오기 · GraspGenX worker는 실행하지 않습니다")
            layout.addWidget(self.grasp_status)
        if grasp_selection:
            layout = worker_layouts["grasp_selection"]
            self.selection_mode_input = QtWidgets.QComboBox()
            if vlm:
                self.selection_mode_input.addItem("VLM coordination · unknown 그대로", "semantic")
            for text, mode in (("수동 single · left", "left"), ("수동 single · right", "right"),
                               ("수동 dual · 같은 물체", "dual")):
                self.selection_mode_input.addItem(text, mode)
            layout.addWidget(QtWidgets.QLabel("Grasp mode"))
            layout.addWidget(self.selection_mode_input)
            self.selection_object_input = QtWidgets.QComboBox()
            self.selection_object_input.addItems(("수동 대상 A", "수동 대상 B"))
            layout.addWidget(self.selection_object_input)
            self.selection_button = QtWidgets.QPushButton("후보 선택·다시 평가")
            self.selection_button.clicked.connect(self.request_grasp_selection)
            layout.addWidget(self.selection_button)
            self.selection_candidate_input = QtWidgets.QComboBox()
            self.selection_candidate_input.addItem("선택 결과 대기", None)
            self.selection_candidate_input.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.selection_candidate_input.setMinimumContentsLength(12)
            self.selection_candidate_input.currentIndexChanged.connect(self._selection_candidate_changed)
            layout.addWidget(self.selection_candidate_input)
            self.selection_status = self._wrapped_label("기존 grasp bundle 후처리 · 로봇 미연결이면 geometry-only · 자동 생성/실행 없음")
            layout.addWidget(self.selection_status)
            self.selection_advanced_button, self.selection_advanced = self._fold_section(layout, "고급 · base pose / IK / 표시용 FK")
            advanced_layout = QtWidgets.QVBoxLayout(self.selection_advanced)
            advanced_layout.setContentsMargins(0, 0, 0, 0)
            self.selection_details = QtWidgets.QPlainTextEdit()
            self.selection_details.setReadOnly(True)
            self.selection_details.setMaximumHeight(160)
            advanced_layout.addWidget(self.selection_details)
            for text, callback in (("파지 관계 고정(표시용)", self.attach_grasp_fk), ("파지 관계 해제", self.detach_grasp_fk)):
                button = QtWidgets.QPushButton(text)
                button.clicked.connect(lambda _checked=False, callback=callback: callback())
                advanced_layout.addWidget(button)
            self.fk_status = self._wrapped_label("Grasp FK · 표시용 고정 전 · 로봇 명령 없음")
            advanced_layout.addWidget(self.fk_status)

        active_key = "aprilcube" if aprilcube else next((key for key in worker_titles if key != "camera" and self.worker_enabled[key]), "camera")
        self.worker_tabs.setCurrentRow(tuple(worker_titles).index(active_key))
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.image_view = RgbImageView()
        split.addWidget(self.image_view)
        self.mesh_view = FigureCanvasQTAgg(Figure(figsize=(5, 4)))
        self.mesh_view.figure.subplots_adjust(left=0.02, right=0.86, bottom=0.14 if foundationpose or aprilcube else 0.06, top=0.9)
        self.axes = self.mesh_view.figure.add_subplot(111, projection="3d")
        self.axes.set_title("Mesh / Pose · Waiting for data")
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_zlabel("Z")
        self.mesh_view.setToolTip("AprilCube · camera-frame mesh / pose · display preview" if aprilcube else
                                 "FoundationPose · 카메라 좌표계 mesh·XYZ(m)·회전축 · 표시용 간소화" if foundationpose else
                                 "Mesh / Pose 표시 영역 · 입력 전까지 비어 있습니다")
        split.addWidget(self.mesh_view)
        split.setSizes([550, 450])
        body.addWidget(split)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([440, 1000])
        self.metadata_label = self._wrapped_label("수신 영상 없음 · 3D 영역은 Mesh / Pose 입력 전까지 비어 있습니다")
        self.metadata_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        outer_layout.addWidget(self.metadata_label)
        self.status = self._wrapped_label("구독 연결 중 · 카메라 영상 대기")
        outer_layout.addWidget(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.poll_once)
        if grasp_selection:
            self.selection_mode_input.currentIndexChanged.connect(lambda _index: self._invalidate_selection("선택 모드 변경"))
            self.selection_object_input.currentIndexChanged.connect(lambda _index: self._invalidate_selection("수동 대상 변경"))
        if graspgenx:
            for key in ("A", "B"):
                self.pose_mesh_inputs[key].textChanged.connect(lambda _text, key=key: self._grasp_geometry_changed(key))
                self.pose_scale_inputs[key].valueChanged.connect(lambda _scale, key=key: self._grasp_geometry_changed(key))
        if vlm and hasattr(self, "pose_mesh_inputs"):
            for key in ("A", "B"):
                self.pose_mesh_inputs[key].textChanged.connect(lambda _text, key=key: self._semantic_mesh_changed(key))
                self.pose_scale_inputs[key].valueChanged.connect(lambda _scale, key=key: self._semantic_mesh_changed(key))
        if hasattr(self, "pose_mesh_inputs"):
            for field in self.pose_mesh_inputs.values():
                field.textChanged.connect(self._refresh_task_roles)
        if foundationpose:
            self._publish_pose_control()
        if aprilcube:
            for key in ("A", "B"):
                self.apply_aprilcube_target(key, redraw=False)
        self._refresh_task_roles()
        self.receiver.start()
        self.timer.start()

    @staticmethod
    def _wrapped_label(text):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        return label

    @staticmethod
    def _slot_panel(layout, key):
        group = QtWidgets.QGroupBox(f"물체 {key}")
        group.setMinimumWidth(0)
        layout.addWidget(group)
        return QtWidgets.QVBoxLayout(group)

    @staticmethod
    def _fold_section(layout, title):
        button = QtWidgets.QToolButton()
        button.setText(title)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setCheckable(True)
        button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        layout.addWidget(button)
        content = QtWidgets.QWidget()
        content.hide()
        layout.addWidget(content)
        button.toggled.connect(content.setVisible)
        button.toggled.connect(lambda checked: button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow))
        return button, content

    def poll_once(self):
        if self._closing:
            if not self.receiver.is_alive():
                self.close()
            return
        frame = self.receiver.take_latest()
        stats = self.receiver.snapshot()
        self.received_count = stats["received_count"]
        if stats.get("error"):
            self._finish(1, f"수신 오류: {stats['error']}")
            return
        if stats.get("ready") and not self._ready_logged:
            log_record("ready", worker="camera_ui")
            self._ready_logged = True
        if self.recording:
            recording_status = self.receiver.take_recording_latest()
            if recording_status is not None:
                self._accept_recording_status(recording_status)
        if frame is not None:
            self._latest_frame = frame
            elapsed = stats.get("elapsed_s", 0)
            fps = f" · 평균 {self.received_count / elapsed:.1f} FPS" if elapsed else ""
            self.camera_info.setText(f"{frame.metadata.get('camera_id', 'camera')} · "
                                     f"{frame.rgb.shape[1]} × {frame.rgb.shape[0]} · {self.received_count} frames{fps}")
        display_changed = frame is not None
        if self.vlm:
            result = self.receiver.take_vlm_latest()
            if result is not None:
                display_changed = self._accept_vlm_result(result) or display_changed
            self.vlm_button.setEnabled(self.vlm_enabled.isChecked() and self._latest_frame is not None
                                       and self._vlm_request_id is None)
        result = self.receiver.take_sam3_latest() if self.sam3 else None
        if (result is not None and self._prompts.objects
                and result.revision == self._prompts.revision):
            self._sam3_result = result
            display_changed = True
            if not self._snapshot_active:
                self._update_mesh_candidates(result)
            if self.vlm and self._vlm_plan:
                self._semantic_updating = True
                try:
                    self._route_vlm_objects(result)
                finally:
                    self._semantic_updating = False
        if self.sam3d:
            for result in self.receiver.take_mesh_results():
                self._semantic_updating = True
                try:
                    self._accept_mesh_result(result)
                finally:
                    self._semantic_updating = False
                display_changed = True
        if self.foundationpose or self.aprilcube:
            result = self.receiver.take_pose_latest()
            if result is not None:
                display_changed = self._accept_pose_result(result) or display_changed
        if self.aprilcube:
            overlay = self.receiver.take_aprilcube_overlay_latest()
            if overlay is not None:
                self._aprilcube_overlay = overlay
                display_changed = True
        if self.graspgenx:
            for result in self.receiver.take_grasp_results():
                display_changed = self._accept_grasp_result(result) or display_changed
        if self.grasp_selection:
            result = self.receiver.take_selection_latest()
            if result is not None:
                display_changed = self._accept_selection_result(result) or display_changed
            if self._fk_request_id is not None:
                fk = self.receiver.take_fk_latest()
                if fk is not None:
                    self._accept_fk_result(fk)
        if self.sam3 and self._refine_request is not None:
            refined = self.receiver.take_refine_latest()
            if refined is not None:
                display_changed = self._accept_refine_result(refined) or display_changed
        self._advance_task_pipeline()
        if display_changed:
            if self.vlm:
                self._sync_semantic_references()
            self._display_pose_frame()
        last = stats.get("last_received_monotonic_s")
        if last is None:
            self.status.setText("iceoryx2 구독 준비 · 카메라 영상 대기 중" if stats.get("ready") else "구독 연결 중")
        else:
            self.status.setText(f"수신 {self.received_count} frames · 마지막 수신 {max(0, time.monotonic() - last):.2f} s 전")
        if self.frames and self.received_count >= self.frames:
            self._finish(0, f"수신 완료: {self.received_count} frames")
        elif self.timeout and time.monotonic() - self._started >= self.timeout:
            failed = self.frames and self.received_count < self.frames
            self._finish(1 if failed else 0,
                         f"시간 종료: 수신 {self.received_count}" + (f"/{self.frames}" if self.frames else "") + " frames")

    def send_recording_control(self, action):
        from ..ipc.recording import RecordingControl

        if not self.recording or self._recording_request_id is not None:
            return
        request = RecordingControl(uuid4().hex, action)
        self._recording_request_id = request.request_id
        self.receiver.send_recording_control(request)
        for button in self.recording_buttons.values():
            button.setEnabled(False)
        self.recording_status.setText(f"Recording · {action} 요청 중")

    def _accept_recording_status(self, status):
        if self._recording_request_id is not None and status.request_id != self._recording_request_id:
            return
        self._recording_request_id = None
        self._recording_state = status.state
        self.recording_status.setText(f"Recording · {status.state}" + (f" · {status.error}" if status.error else ""))
        self.recording_counts.setText(f"Episode {status.episode_index} · {status.episode_frames} frames · "
                                      f"Total {status.total_frames} frames")
        self.recording_output.setText(f"Output · {status.output_dir}")
        self.recording_buttons["start"].setEnabled(status.state in {"ready", "saved", "discarded"})
        self.recording_buttons["save"].setEnabled(status.state == "recording" and status.episode_frames > 0)
        self.recording_buttons["discard"].setEnabled(status.state == "recording")

    def apply_prompts(self, *, redraw=True):
        from ..ipc.sam3 import ObjectPrompt

        self._set_prompts(tuple(ObjectPrompt(key, field.text().strip())
                                for key, field in self.prompt_inputs.items() if field.text().strip()), redraw=redraw)

    def _task_rows(self):
        if self._semantic_session is not None:
            rows = [row for row in self._semantic_session.record["objects"] if row["binding_active"]]
            if rows:
                return rows
        for key in ("A", "B"):
            prompt = self.prompt_inputs[key].text().strip() if hasattr(self, "prompt_inputs") else ""
            path = self.pose_mesh_inputs[key].text().strip() if hasattr(self, "pose_mesh_inputs") else ""
            if prompt or path:
                self._manual_task_objects.setdefault(key, {"observation_id": f"manual-{uuid4().hex}", "instance_id": uuid4().hex,
                                                          "slot": key, "binding_active": True})
                self._manual_task_objects[key]["label"] = prompt or Path(path).stem
            else:
                self._manual_task_objects.pop(key, None)
        return list(self._manual_task_objects.values())

    def _refresh_task_roles(self, *_args):
        if not hasattr(self, "source_role_input"):
            return
        binding = self._semantic_session.record["task_binding"] if self._semantic_session is not None else None
        self._task_role_updating = True
        for role, combo in (("source", self.source_role_input), ("target", self.target_role_input)):
            selected = binding.get(f"{role}_observation_id") if binding is not None else combo.currentData()
            combo.clear()
            combo.addItem("미선택" if role == "source" else "없음 · 단일 Grasp", None)
            for row in self._task_rows():
                combo.addItem(f"{row['slot']} · {row.get('label', row['observation_id'])}", row["observation_id"])
            combo.setCurrentIndex(max(0, combo.findData(selected)))
        self._task_role_updating = False

    def _task_roles_changed(self, *_args):
        if self._task_role_updating:
            return
        self._task_flow = None
        if self.grasp_selection:
            self._invalidate_selection("Source / Target 역할 변경")
        source, target = self.source_role_input.currentData(), self.target_role_input.currentData()
        if source is not None and source == target:
            if self._semantic_session is not None:
                self._semantic_session.record["coordination"] = None
                self._render_semantic_session()
            self.task_status.setText("Source와 Target은 서로 다른 관측을 선택하세요 · 수동 기능은 계속 사용 가능")
            return
        if self._semantic_session is not None:
            self._semantic_session.record["task_binding"] = {"source_observation_id": source, "target_observation_id": target,
                "intent": (self._semantic_session.record["task_binding"] or {}).get("intent", "")}
            self._semantic_session.record["coordination"] = None
            self._render_semantic_session()
        self.task_status.setText("물체 역할 변경 · 팔 역할·상대 pose VLM 재분석 또는 명시적 수동 팔 mode 선택 필요")

    def start_task_pipeline(self):
        source, target = self.source_role_input.currentData(), self.target_role_input.currentData()
        if source is None or source == target:
            self.task_status.setText("서로 다른 Source / Target 역할을 선택하세요 · Target 없음은 가능")
            return
        self._task_flow = {"request_id": uuid4().hex, "source": source, "target": target, "started": set(),
                           "grasp_jobs": {}, "selection_id": None, "phase": "pose"}
        self._advance_task_pipeline()

    def _advance_task_pipeline(self):
        flow = self._task_flow
        if flow is None or flow["phase"] in {"done", "error"}:
            return
        rows = {row["observation_id"]: row for row in self._task_rows()}
        if flow["source"] not in rows or flow["target"] is not None and flow["target"] not in rows:
            flow["phase"] = "error"
            self.task_status.setText("Task · 선택한 관측 연결이 변경되었습니다 · 역할을 다시 선택하세요")
            return
        selected = [rows[key] for key in (flow["source"], flow["target"]) if key in rows]
        if not self.foundationpose:
            self.task_status.setText("Task · FoundationPose UI/worker 설정 필요 · 자동으로 켜지 않습니다")
            flow["phase"] = "error"
            return
        if flow["phase"] == "pose":
            for row in selected:
                key = row["slot"]
                if key not in self._pose_targets and key not in flow["started"]:
                    source = self._input_sam3_result()
                    item = next((item for item in source.objects if item.object_id == key), None) if source else None
                    if item is None or self.candidate_inputs[key].currentData() is None:
                        self.task_status.setText(f"Task · {key} SAM3 후보 선택 대기")
                        return
                    asset = self.asset_store.get(row.get("known_asset_id")) if self.vlm else None
                    if not self.pose_mesh_inputs[key].text().strip() and asset is not None:
                        self._semantic_updating = True
                        self.pose_mesh_inputs[key].setText(asset["mesh_path"])
                        self.pose_scale_inputs[key].setValue(asset["mesh_scale_to_m"])
                        self._semantic_updating = False
                    if self.pose_mesh_inputs[key].text().strip():
                        self.start_pose(key, redraw=False)
                    elif self.sam3d:
                        self.generate_mesh(key, start_pose=True)
                    else:
                        self.task_status.setText(f"Task · {key} mesh 또는 SAM3D 설정 필요")
                        return
                    flow["started"].add(key)
            for row in selected:
                key = row["slot"]
                pose = self._pose_objects.get(key)
                if pose is not None and pose.status == "error":
                    flow["phase"] = "error"
                    self.task_status.setText(f"Task · {key} FP 오류: {pose.error}")
                    return
                if pose is None or pose.camera_from_object is None:
                    self.task_status.setText(f"Task · {key} FP 결과 대기")
                    return
            flow["phase"] = "grasp"
        if flow["phase"] == "grasp":
            source_row = rows.get(flow["source"])
            if source_row is None:
                return
            grasp_rows = {source_row["slot"]: source_row}
            if not self.graspgenx:
                self.task_status.setText("Task · FP 완료 · Grasp worker 설정 없음")
                flow["phase"] = "done"
                return
            for key in grasp_rows:
                result = self._grasp_results.get(key)
                if result is None or not result.bundle_path:
                    if key not in flow["grasp_jobs"]:
                        self.generate_grasp(key)
                        request = self._grasp_requests.get(key)
                        if request is None:
                            flow["phase"] = "error"
                            self.task_status.setText(f"Task · {key} grasp 입력 mesh가 없습니다")
                            return
                        flow["grasp_jobs"][key] = request.request_id
                    self.task_status.setText(f"Task · {key} grasp cache 조회 / 생성 결과 대기")
                    return
            if self.grasp_selection:
                if self.selection_mode_input.currentData() != "semantic":
                    blocker = QtCore.QSignalBlocker(self.selection_object_input)
                    self.selection_object_input.setCurrentIndex(("A", "B").index(source_row["slot"]))
                    del blocker
                self.request_grasp_selection()
                if self._selection_request is not None:
                    flow["selection_id"], flow["phase"] = self._selection_request.request_id, "selection"
                else:
                    flow["phase"] = "error"
            else:
                flow["phase"] = "done"
                self.task_status.setText("Task · Source grasp 준비 완료 · 로봇 명령 없음")

    def _set_prompts(self, objects, *, redraw=True):
        from ..ipc.sam3 import PromptSet

        if not self._semantic_updating:
            self._task_flow = None
            self._manual_task_objects.clear()
        if self.vlm:
            self._mark_semantic_bindings_stale("수동 Apply / SAM3 Stop")
            self._cancel_vlm()
        self._prompts = PromptSet(uuid4().hex, objects)
        self.receiver.set_prompts(self._prompts)
        if hasattr(self, "candidate_inputs"):
            self._candidate_sources.clear()
            self._mesh_bindings.clear()
            self._mesh_results.clear()
            self._mesh_jobs.clear()
            self._mesh_pose_jobs.clear()
            self._mesh_states.clear()
            self._mesh_dirty.clear()
            for key, combo in self.candidate_inputs.items():
                combo.clear()
                combo.addItem(f"{key}: SAM3 후보 대기", None)
                if self.sam3d:
                    self._enable_mesh_buttons(key, False)
            if self.sam3d:
                self._update_mesh_status()
                self._plot_selected_mesh()
        raw = self._latest_frame or (self._sam3_result.frame if self._sam3_result is not None else None)
        self._sam3_result = None
        if redraw and raw is not None:
            self._display_frame(raw, poses=tuple(self._pose_objects.values()),
                                live_frame=self._latest_frame is not None)
        elif redraw:
            self._display_pose_frame()
        text = " / ".join(f"{item.object_id}: {item.text}" for item in objects)
        self.sam3_status.setText((f"SAM3 결과 대기 · {text}" if objects else "SAM3 중지")
                                 + " · A/B는 prompt slot이며 지속 추적 ID가 아닙니다")
        self._refresh_task_roles()

    def _cancel_vlm(self):
        if not self._semantic_updating:
            self._task_flow = None
        self._vlm_request_id = None
        self._vlm_plan.clear()
        for key, request_id in self._vlm_mesh_jobs.items():
            if self._mesh_pose_jobs.get(key) == request_id:
                self._mesh_pose_jobs.pop(key, None)
        self._vlm_mesh_jobs.clear()

    def _vlm_mode_changed(self, enabled):
        if not enabled:
            self._cancel_vlm()
            self.vlm_status.setText("VLM 사용 안 함 · 수동 prompt·mesh 버튼 사용")
        self.vlm_button.setEnabled(enabled and self._latest_frame is not None and self._vlm_request_id is None)

    def start_vlm(self):
        from ..ipc.camera import CameraFrame
        from ..ipc.vlm import VlmRequest
        from ..policy_catalog import public_policy_catalog

        if not self.vlm_enabled.isChecked() or self._latest_frame is None:
            return
        self._cancel_vlm()
        frame = self._latest_frame
        frozen = CameraFrame(frame.rgb.copy(), frame.depth_m.copy(), deepcopy(frame.metadata))
        self._vlm_request_id = uuid4().hex
        self._vlm_instruction_at_request = self.vlm_instruction.text().strip()
        self.receiver.submit_vlm(VlmRequest(self._vlm_request_id, frozen, self._vlm_instruction_at_request,
                                           self.asset_store.public_catalog(), tuple(public_policy_catalog(self.policy_catalog))))
        self.vlm_button.setEnabled(False)
        self.vlm_status.setText(f"VLM 인식 중 · frame {frame.metadata.get('sequence')}")

    def _accept_vlm_result(self, result):
        if result.request_id != self._vlm_request_id or not self.vlm_enabled.isChecked():
            return False
        if result.status == "running":
            return False
        self._vlm_request_id = None
        if result.status == "error":
            self.vlm_status.setText(f"VLM 오류 · {result.error}")
            return False
        if self.grasp_selection:
            self._invalidate_selection("새 VLM semantic", redraw=False)
        self._semantic_updating = True
        if self.foundationpose:
            for key in list(self._pose_targets):
                self.stop_pose(key, redraw=False)
        objects = dict(zip(("A", "B"), result.objects))
        for key, field in self.prompt_inputs.items():
            field.setText(objects[key].segmentation_prompt if key in objects else "")
        self.apply_prompts(redraw=False)
        self._semantic_updating = False
        from ..semantic_session import SemanticSession
        from ..policy_catalog import resolve_policy

        self._semantic_session = SemanticSession.from_result(result, self._vlm_instruction_at_request)
        selection = result.policy_selection
        if selection is not None:
            try:
                policy = resolve_policy(self.policy_catalog, selection["policy_id"], selection["task_type"])
                if policy is not None:
                    self._semantic_session.record["policy_reference"] = {key: policy[key] for key in
                                                                         ("policy_id", "task_type", "display_name", "checkpoint")}
            except ValueError as error:
                self._semantic_session.record["policy_reference"] = {"resolution_error": str(error)}
        if self.vlm_instruction.text().strip() != self._vlm_instruction_at_request:
            self._semantic_session.instruction_edited(self.vlm_instruction.text())
        self._sync_semantic_references()
        self._render_semantic_session()
        self._vlm_plan = objects if self.foundationpose else {}
        self._vlm_states = {key: f"{item.label} · SAM3 후보 대기" for key, item in objects.items()}
        self.vlm_status.setText(" / ".join(f"{key}: {text}" for key, text in self._vlm_states.items())
                                or "VLM · 대상 물체 없음")
        self._refresh_task_roles()
        if self.foundationpose and result.task_binding is not None and result.task_binding["source_observation_id"] is not None:
            self.start_task_pipeline()
        return True

    def _route_vlm_objects(self, result):
        used_masks = []
        for key, target in list(self._vlm_plan.items()):
            observation = next((item for item in result.objects if item.object_id == key), None)
            if observation is None or observation.error:
                continue
            candidates = [index for index, mask in enumerate(observation.masks)
                          if mask.any() and not any(np.array_equal(mask, used) for used in used_masks)]
            if not candidates:
                continue
            index = max(candidates, key=lambda i: (
                _box_iou(observation.boxes[i], target.bbox_xyxy) if target.bbox_xyxy is not None else 0.,
                float(observation.scores[i])))
            used_masks.append(observation.masks[index])
            self.candidate_inputs[key].setCurrentIndex(index + 1)
            asset = self.asset_store.get(target.known_asset_id)
            if asset is not None and Path(asset["mesh_path"]).is_file():
                self.pose_mesh_choices[key].setCurrentIndex(0)
                self.pose_mesh_inputs[key].setText(asset["mesh_path"])
                self.pose_scale_inputs[key].setValue(asset["mesh_scale_to_m"])
                self.start_pose(key, redraw=False)
                self._vlm_states[key] = f"{target.label} · 기존 mesh → FP"
            elif self.sam3d:
                self._mesh_bindings[key]["name"] = target.label
                self.generate_mesh_and_start_pose(key)
                self._vlm_mesh_jobs[key] = self._mesh_jobs[key]
                self._vlm_states[key] = f"{target.label} · 새 mesh 생성 → FP"
            else:
                self._vlm_states[key] = f"{target.label} · mesh 없음 · SAM3D worker 필요"
            self._vlm_plan.pop(key, None)
        self.vlm_status.setText(" / ".join(f"{key}: {text}" for key, text in self._vlm_states.items()))

    def _semantic_instruction_edited(self, text):
        if self._semantic_session is not None:
            self._semantic_session.instruction_edited(text)
            self._render_semantic_session()

    def _mark_semantic_bindings_stale(self, reason, slots=("A", "B")):
        if self._semantic_session is not None and not self._semantic_updating:
            self._sync_semantic_references()
            self._semantic_session.invalidate_bindings(reason, slots)
            if self.grasp_selection:
                self._invalidate_selection(reason, redraw=False)
            self._render_semantic_session()

    def _semantic_mesh_changed(self, key):
        self._mark_semantic_bindings_stale("수동 mesh / scale 변경", (key,))

    def _sync_semantic_references(self):
        from ..semantic_session import frame_reference

        if self._semantic_session is None:
            return
        for row in self._semantic_session.record["objects"]:
            if not row["binding_active"]:
                continue
            key = row["slot"]
            asset = self.asset_store.get(row["known_asset_id"])
            if row["geometry"] is None and asset is not None:
                row["geometry"] = {name: asset[name] for name in ("asset_id", "mesh_path", "mesh_scale_to_m")}
            target = self._pose_targets.get(key)
            if target is not None:
                result = self._mesh_results.get(key)
                row["geometry"] = {"asset_id": (result.asset_id if result is not None
                                                 and result.mesh_path == target.mesh_path else
                                                 asset["asset_id"] if asset is not None else None),
                                   "mesh_path": target.mesh_path, "mesh_scale_to_m": target.mesh_scale_to_m}
                item = self._pose_objects.get(key)
                row["foundationpose"] = {"session_id": target.session_id,
                                         "status": item.status if item is not None else "registering",
                                         "camera_from_object": (item.camera_from_object.tolist() if item is not None
                                                                 and item.camera_from_object is not None else None),
                                         "frame": (frame_reference(self._pose_result.frame) if self._pose_result is not None
                                                   else frame_reference(target.frame))}
            if self.graspgenx:
                result = self._grasp_results.get(key)
                index = self.grasp_inputs[key].currentData()
                if result is not None and index is not None and self._selected_grasp(key) is not None:
                    row["geometry"] = {"asset_id": result.asset_id, "mesh_path": result.mesh_path,
                                       "mesh_scale_to_m": result.mesh_scale_to_m}
                    row["grasp"] = {"request_id": result.request_id, "bundle_path": result.bundle_path,
                                    "gripper_id": result.gripper_id, "selected_index": index,
                                    "score": float(result.scores[index]) if result.scores is not None else None,
                                    "object_from_canonical_gripper": result.grasps[index].tolist()}
        if self.grasp_selection and any(row["binding_active"] for row in self._semantic_session.record["objects"]):
            self._semantic_session.record["grasp_selection"] = self.selection_reference()

    def _render_semantic_session(self):
        if self._semantic_session is None:
            return
        record = self._semantic_session.record
        slots = {row["observation_id"]: row["slot"] or "연결 해제" for row in record["objects"]}
        binding = record["task_binding"] or {}
        roles = " → ".join(f"{role}: {binding.get(role + '_observation_id') or '미정'}"
                            f" [{slots.get(binding.get(role + '_observation_id'), '—')}]" for role in ("source", "target"))
        coordination = record["coordination"]
        arms = f"arm_mode: {coordination['arm_mode']}" if coordination is not None else "팔 정보 없음 · legacy result"
        origin = "저장된 semantic · 현재 관측 아님" if record["provenance"]["source"] == "loaded_session" else "VLM semantic"
        stale = (" · 재분석 필요" if record.get("bindings_stale") or record.get("coordination_instruction_stale") else "")
        self.semantic_summary.setText(f"{origin} · {roles} · {arms}{stale}")
        self.semantic_details.setPlainText(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))

    def _semantic_details_toggled(self, visible):
        if visible:
            self._sync_semantic_references()
            self._render_semantic_session()

    def save_semantic_session(self, path=None):
        if self._semantic_session is None:
            self.semantic_summary.setText("저장할 VLM semantic 결과가 없습니다")
            return None
        self._sync_semantic_references()
        if path is None:
            path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Semantic session 저장", str(self.session_dir / f"{self._semantic_session.record['session_id']}.json"),
                "JSON (*.json)")
        if not path:
            return None
        try:
            saved = self._semantic_session.save(path)
            self.semantic_summary.setToolTip(f"Session 저장: {saved}")
            return saved
        except (OSError, ValueError) as error:
            self.semantic_summary.setToolTip(f"Session 저장 오류: {error}")
            return None

    def load_semantic_session(self, path=None):
        from ..semantic_session import SemanticSession

        if path is None:
            path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Semantic session 불러오기", str(self.session_dir),
                                                               "JSON (*.json)")
        if not path:
            return None
        try:
            session = SemanticSession.load(path)
            if self.grasp_selection:
                self._invalidate_selection("Session 불러옴 · 현재 관측 아님", redraw=False)
            self._semantic_session = session
            self._task_flow = None
            self._refresh_task_roles()
            self._render_semantic_session()
            if self.grasp_selection:
                self._redraw_grasp_selection()
            return Path(path).resolve()
        except (OSError, ValueError) as error:
            self.semantic_summary.setToolTip(f"Session 불러오기 오류: {error}")
            return None

    def export_semantic_request(self, path=None):
        from ..semantic_session import build_execution_request, write_record

        if self._semantic_session is None:
            self.semantic_summary.setText("Export할 VLM semantic 결과가 없습니다")
            return None
        self._sync_semantic_references()
        record = build_execution_request(self._semantic_session.record)
        if path is None:
            path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "실행 요청 export · 로봇 명령 전송 없음",
                str(self.session_dir / f"{record['session_id']}.execution.json"), "JSON (*.json)")
        if not path:
            return None
        try:
            saved = write_record(path, record)
            self.semantic_summary.setToolTip(f"request_only export: {saved} · robot_command_published=false")
            return record
        except (OSError, ValueError) as error:
            self.semantic_summary.setToolTip(f"Export 오류: {error}")
            return None

    def _display_frame(self, frame, result=None, poses=(), *, live_frame=False):
        rgb, annotations, counts = frame.rgb, [], []
        if result is not None:
            rgb = frame.rgb.copy()
            by_id = {item.object_id: item for item in result.objects}
            for prompt in (self._prompts.objects if self._prompts is not None and self._prompts.objects else result.objects):
                item = by_id.get(prompt.object_id)
                if item is None:
                    counts.append(f"{prompt.object_id}: 결과 없음")
                    continue
                color = (57, 217, 138) if item.object_id == "A" else (255, 180, 84)
                counts.append(f"{item.object_id}: {len(item.masks)}개 후보" + (f" · 오류 {item.error}" if item.error else ""))
                for index, (mask, box) in enumerate(zip(item.masks, item.boxes), 1):
                    rgb[mask] = (rgb[mask] * 0.6 + np.asarray(color) * 0.4).astype(np.uint8)
                    annotations.append((box, color, f"{item.object_id}[{index}]"))
            self.sam3_status.setText(" / ".join(counts) + " · 최신 SAM3 overlay · A/B는 지속 추적 ID가 아닙니다")
        direct_aprilcube_axes = (self.aprilcube and not self._snapshot_active
                                 and not any(target.mesh_path for target in self._pose_targets.values()))
        overlay = (self._aprilcube_overlay
                   if self.aprilcube and not self._snapshot_active and not direct_aprilcube_axes else None)
        if overlay is not None and overlay.rgba.shape[:2] == rgb.shape[:2]:
            alpha = overlay.rgba[..., 3:4].astype(np.float32) / 255
            rgb = (rgb * (1 - alpha) + overlay.rgba[..., :3] * alpha).astype(np.uint8)
        if self.aprilcube and not direct_aprilcube_axes:
            # Selected meshes and their axes remain on the RGBA worker path.
            poses = ()
        if direct_aprilcube_axes and poses:
            # Draw axes from the same latest pose as the plot, then scale once.
            self.image_view.set_rgb(rgb, redraw=False)
        else:
            self.image_view.set_rgb(rgb)
        if annotations or poses:
            painter = QtGui.QPainter(self.image_view.image)
            try:
                for box, color, label in annotations:
                    painter.setPen(QtGui.QPen(QtGui.QColor(*color), 2))
                    x1, y1, x2, y2 = map(float, box)
                    painter.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
                    painter.drawText(QtCore.QPointF(x1, max(12, y1 - 3)), label)
                for item in poses:
                    if item.camera_from_object is None:
                        continue
                    pose = item.camera_from_object
                    origin = pose[:3, 3]
                    points = np.vstack((origin, origin + pose[:3, :3].T * 0.05))
                    if direct_aprilcube_axes and not np.all(points[:, 2] > 1e-6):
                        continue
                    intrinsics = frame.metadata["intrinsics"]
                    pixels = points[:, :2] / points[:, 2:3]
                    pixels = pixels * (intrinsics["fx"], intrinsics["fy"]) + (intrinsics["cx"], intrinsics["cy"])
                    if (direct_aprilcube_axes and intrinsics.get("distortion_model") != "zed_rectified"
                            and np.any(intrinsics.get("coefficients", []))):
                        from ..models.aprilcube import camera_parameters
                        from ..models.aprilcube_overlay import _project

                        pixels = _project(points, *camera_parameters(frame))
                    if direct_aprilcube_axes and not np.isfinite(pixels).all():
                        continue
                    start = QtCore.QPointF(*pixels[0])
                    for endpoint, axis, color in zip(pixels[1:], "XYZ", ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                        painter.setPen(QtGui.QPen(QtGui.QColor(*color), 3))
                        end = QtCore.QPointF(*endpoint)
                        painter.drawLine(start, end)
                        painter.drawText(end + QtCore.QPointF(3, -3), axis)
                    painter.setPen(QtGui.QColor("white"))
                    painter.drawText(start + QtCore.QPointF(4, -4), f"{item.object_id} pose")
                    if self.graspgenx:
                        for arm, selected in self._display_grasps(item.object_id):
                            from .grasp_preview import grasp_axis_points

                            points = grasp_axis_points(pose @ selected)
                            pixels = points[:, :2] / points[:, 2:3]
                            pixels = pixels * (intrinsics["fx"], intrinsics["fy"]) + (intrinsics["cx"], intrinsics["cy"])
                            start = QtCore.QPointF(*pixels[0])
                            for endpoint, color in zip(pixels[1:], ((255, 60, 60), (60, 255, 60), (60, 100, 255))):
                                painter.setPen(QtGui.QPen(QtGui.QColor(*color), 3))
                                painter.drawLine(start, QtCore.QPointF(*endpoint))
                            painter.setPen(QtGui.QColor("white"))
                            label = f"{item.object_id} grasp" if arm == "gripper" else f"{item.object_id} {arm} grasp"
                            painter.drawText(start + QtCore.QPointF(4, -4), label)
            finally:
                painter.end()
            self.image_view._draw_image()
        metadata = frame.metadata
        self.metadata_label.setText(
            f"camera={metadata.get('camera_id')} · sequence={metadata.get('sequence')} · "
            f"RGB {frame.rgb.shape[1]}×{frame.rgb.shape[0]} · "
            f"capture={metadata.get('capture_time_ns')} ({metadata.get('timestamp_domain')})"
            + (" · 실시간 카메라" if live_frame else
               " · SAM3 결과의 원본 프레임" if result is not None else
               f" · {'AprilCube' if self.aprilcube else 'FP'} 결과의 원본 프레임" if self._pose_result is not None else " · 원본 카메라")
            + (f" · {'AprilCube' if self.aprilcube else 'FP'} pose sequence={self._pose_result.frame.metadata.get('sequence')}"
               if self._pose_result is not None and self._pose_targets else "")
            + (f" · AprilCube overlay sequence={overlay.metadata.get('sequence')}" if overlay is not None else "")
            + (f" · SAM3 overlay sequence={result.frame.metadata.get('sequence')}" if result is not None else ""))

    def _mesh_prompt_edited(self, key):
        if self.vlm:
            self._mark_semantic_bindings_stale("prompt 편집", (key,))
        self._vlm_plan.pop(key, None)
        self._mesh_dirty.add(key)
        self._mesh_bindings.pop(key, None)
        self._mesh_results.pop(key, None)
        self._mesh_pose_jobs.pop(key, None)
        self._enable_mesh_buttons(key, False)
        self._mesh_states[key] = "prompt 편집됨 · Apply 필요"
        self._update_mesh_status()
        self._plot_selected_mesh()

    def _update_mesh_candidates(self, result):
        for key, combo in self.candidate_inputs.items():
            if key in self._mesh_bindings or key in self._mesh_dirty:
                continue
            previous = combo.currentData()
            self._candidate_sources[key] = result
            item = next((item for item in result.objects if item.object_id == key), None)
            blocker = QtCore.QSignalBlocker(combo)
            combo.clear()
            combo.addItem(f"{key}: 후보 선택" if item is not None and len(item.masks) else f"{key}: 후보 없음", None)
            if item is not None:
                for index in range(len(item.masks)):
                    combo.addItem(f"{key}[{index + 1}] · frame {result.frame.metadata.get('sequence')}", index)
                if len(item.masks) == 1:
                    combo.setCurrentIndex(1)
                elif previous is not None and previous < len(item.masks):
                    combo.setCurrentIndex(previous + 1)
            del blocker
            if self.sam3d:
                self._select_mesh_candidate(key)

    def _select_mesh_candidate(self, key):
        from ..ipc.camera import CameraFrame

        index = self.candidate_inputs[key].currentData()
        self._mesh_bindings.pop(key, None)
        self._mesh_results.pop(key, None)
        self._mesh_pose_jobs.pop(key, None)
        if index is not None and key not in self._mesh_dirty:
            source = self._candidate_sources[key]
            item = next(item for item in source.objects if item.object_id == key)
            frame = source.frame
            self._mesh_bindings[key] = {
                "asset_id": uuid4().hex, "revision": self._prompts.revision, "prompt": item.text,
                "candidate_index": index, "mask": item.masks[index].copy(),
                "frame": CameraFrame(frame.rgb.copy(), frame.depth_m.copy(), deepcopy(frame.metadata)),
            }
        self._enable_mesh_buttons(key, key in self._mesh_bindings and key not in self._mesh_jobs)
        self._mesh_states[key] = "준비" if key in self._mesh_bindings else "후보 선택 대기"
        self._update_mesh_status()
        self._plot_selected_mesh()

    def _enable_mesh_buttons(self, key, enabled):
        self.mesh_buttons[key].setEnabled(enabled)
        if key in self.mesh_pose_buttons:
            self.mesh_pose_buttons[key].setEnabled(enabled)

    def generate_mesh_and_start_pose(self, key):
        self.generate_mesh(key, start_pose=True)

    def generate_mesh(self, key, *, start_pose=False):
        from ..ipc.sam3d import MeshRequest

        binding = self._mesh_bindings.get(key)
        if binding is None or key in self._mesh_jobs:
            return
        request = MeshRequest(uuid4().hex, binding["asset_id"], key, binding["revision"], binding["prompt"],
                              binding["candidate_index"], binding["frame"], binding["mask"],
                              depth_source=self.sam3d_depth_source, texture_baking=self.sam3d_texture_baking)
        self._mesh_jobs[key] = request.request_id
        if start_pose:
            self._mesh_pose_jobs[key] = request.request_id
            self.pose_status.setText(f"{key}: SAM3D 생성 후 FoundationPose 시작 대기")
        self._mesh_states[key] = "대기 / Generating"
        self._enable_mesh_buttons(key, False)
        self.receiver.submit_mesh(request)
        self._update_mesh_status()

    def _accept_mesh_result(self, result):
        key = result.object_id
        if result.revision != self._prompts.revision or self._mesh_jobs.get(key) != result.request_id:
            return
        binding = self._mesh_bindings.get(key)
        start_pose = self._mesh_pose_jobs.get(key) == result.request_id
        if result.status in {"done", "error"}:
            self._mesh_jobs.pop(key, None)
            self._mesh_pose_jobs.pop(key, None)
            self._enable_mesh_buttons(key, binding is not None and key not in self._mesh_dirty)
        if binding is None or result.asset_id != binding["asset_id"]:
            return
        self._mesh_states[key] = ("Done" if result.status == "done" else
                                  f"Error · {result.error}" if result.status == "error" else "Generating")
        if start_pose and result.status == "error":
            self.pose_status.setText(f"{key}: SAM3D Error · {result.error}")
            if self._task_flow is not None and key in self._task_flow["started"]:
                self._task_flow["phase"] = "error"
                self.task_status.setText(f"Task · {key} SAM3D 오류: {result.error}")
        if result.status == "done":
            self._mesh_results[key] = result
            if self.vlm:
                try:
                    asset = self.asset_store.save(result, binding.get("name", binding["prompt"]), binding["prompt"])
                    if self.foundationpose:
                        for combo in self.pose_mesh_choices.values():
                            if not any((combo.itemData(i) or {}).get("mesh_path") == asset["mesh_path"]
                                       for i in range(combo.count())):
                                combo.addItem(asset["name"], {"mesh_path": asset["mesh_path"],
                                                           "scale_to_m": asset["mesh_scale_to_m"]})
                except (OSError, ValueError) as error:
                    self._mesh_states[key] = f"Done · catalog 저장 오류: {type(error).__name__}"
            if (self.foundationpose or self.graspgenx) and result.mesh_path and not self.pose_mesh_inputs[key].text():
                self.pose_mesh_inputs[key].setText(result.mesh_path)
                if result.length_unit == "metre":
                    self.pose_scale_inputs[key].setValue(result.mesh_scale_to_m)
            self.mesh_object_input.setCurrentIndex(("A", "B").index(key))
            self._plot_selected_mesh()
            if start_pose:
                self.pose_mesh_choices[key].setCurrentIndex(0)
                self.pose_mesh_inputs[key].setText(result.mesh_path)
                if result.length_unit == "metre":
                    self.pose_scale_inputs[key].setValue(result.mesh_scale_to_m)
                self.start_pose(key, frame=binding["frame"], mask=binding["mask"],
                                prompt_revision=binding["revision"], redraw=False)
        self._update_mesh_status()
        if self.vlm and self._vlm_mesh_jobs.get(key) == result.request_id and result.status in {"done", "error"}:
            self._vlm_mesh_jobs.pop(key)
            self._vlm_states[key] = "생성 mesh 저장 → FP" if result.status == "done" else f"mesh 생성 오류 · {result.error}"
            self.vlm_status.setText(" / ".join(f"{slot}: {text}" for slot, text in self._vlm_states.items()))

    def _update_mesh_status(self):
        labels = []
        for key in ("A", "B"):
            binding = self._mesh_bindings.get(key)
            source = (f" · frozen frame {binding['frame'].metadata.get('sequence')}"
                      f" / candidate {binding['candidate_index'] + 1}" if binding else "")
            labels.append(f"{key}: {self._mesh_states.get(key, '후보 대기')}{source}")
        units = " / ".join(f"{key}: {result.length_unit}" for key, result in self._mesh_results.items())
        self.mesh_status.setText(" / ".join(labels) + " · asset 로컬 좌표 · " + (units or "실측 depth 기본 · monocular는 예측 scale"))
        self.mesh_status.setToolTip("\n".join(f"{key}: {result.mesh_path}" for key, result in self._mesh_results.items()
                                            if result.mesh_path))

    def _plot_selected_mesh(self, *_args):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if self._pose_targets:
            self._plot_pose()
            return
        if self.graspgenx and self._grasp_results:
            self._plot_grasp_local()
            return
        result = self._mesh_results.get(("A", "B")[self.mesh_object_input.currentIndex()])
        self._plot_mode = "mesh"
        self._grasp_artists = {}
        self.axes.clear()
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_zlabel("Z")
        self.axes.set_title("Mesh · Waiting for data")
        if result is not None and result.mesh is not None:
            if result.length_unit == "metre":
                self.axes.set_xlabel("X (m)")
                self.axes.set_ylabel("Y (m)")
                self.axes.set_zlabel("Z (m)")
            mesh = self.receiver.mesh_preview(result.object_id, result.asset_id) or result.mesh
            faces = mesh.faces
            colors = mesh.colors[faces].mean(axis=1) / 255 if mesh.colors is not None else "#39d98a"
            self.axes.add_collection3d(Poly3DCollection(mesh.vertices[faces], facecolors=colors, linewidths=0, shade=True))
            center = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
            half = max(float(np.ptp(mesh.vertices, axis=0).max()) / 2, 1e-6)
            self.axes.set_xlim(center[0] - half, center[0] + half)
            self.axes.set_ylim(center[1] - half, center[1] + half)
            self.axes.set_zlim(center[2] - half, center[2] + half)
            self.axes.set_box_aspect((1, 1, 1))
            self.axes.set_title(f"{result.object_id} mesh · local preview")
        self.mesh_view.draw_idle()

    def _select_pose_mesh(self, key):
        mesh = self.pose_mesh_choices[key].currentData()
        if mesh is not None:
            self.pose_mesh_inputs[key].setText(mesh["mesh_path"])
            self.pose_scale_inputs[key].setValue(mesh["scale_to_m"])

    def _browse_pose_mesh(self, key):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, f"{key} pose mesh", str(Path(__file__).resolve().parents[2] / "assets" / "meshes"),
            "Mesh (*.glb *.gltf *.ply *.obj *.stl);;All files (*)")
        if path:
            self.pose_mesh_choices[key].setCurrentIndex(0)
            self.pose_mesh_inputs[key].setText(path)

    def _publish_pose_control(self):
        from ..ipc.foundationpose import PoseControl

        self.receiver.set_pose_control(PoseControl(tuple(self._pose_targets.values())))

    def apply_aprilcube_target(self, key, *, redraw=True):
        from ..ipc.aprilcube import AprilCubeControl, AprilCubeTarget

        roll, pitch, yaw = np.deg2rad([field.value() for field in self.aprilcube_rpy_inputs[key]])
        cr, cp, cy = np.cos((roll, pitch, yaw))
        sr, sp, sy = np.sin((roll, pitch, yaw))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                             (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                             (-sp, cp * sr, cp * cr))
        transform[:3, 3] = [field.value() for field in self.aprilcube_xyz_inputs[key]]
        path = self.pose_mesh_inputs[key].text().strip()
        target = AprilCubeTarget(key, uuid4().hex, path, self.pose_scale_inputs[key].value(), transform)
        self._pose_targets[key] = target
        self._pose_objects.pop(key, None)
        self._pose_meshes.pop(key, None)
        self._pose_mesh_errors.pop(key, None)
        self._aprilcube_overlay = None
        self.receiver.send_aprilcube_control(AprilCubeControl(tuple(self._pose_targets.values())))
        if path:
            self._prepare_pose_mesh(key)
        self.pose_status.setText(f"AprilCube · {key}: applied · " + (Path(path).name if path else "cube axes only"))
        if redraw:
            self._plot_pose()
            self._display_pose_frame()
        return target

    def start_pose(self, key, *, frame=None, mask=None, prompt_revision=None, redraw=True, initial_pose=None, initialization="register"):
        from ..ipc.foundationpose import PoseTarget

        self._mesh_pose_jobs.pop(key, None)
        self._vlm_plan.pop(key, None)
        path = self.pose_mesh_inputs[key].text().strip()
        if frame is None:
            result = self._input_sam3_result()
            index = self.candidate_inputs[key].currentData()
            item = next((item for item in result.objects if item.object_id == key), None) if result else None
            if not path or item is None or index is None or index >= len(item.masks):
                self.pose_status.setText(f"{key}: mesh 파일과 현재 SAM3 후보를 선택하세요")
                return
            frame, mask = result.frame, item.masks[index]
            prompt_revision = self._prompts.revision
        self._pose_targets[key] = PoseTarget(key, uuid4().hex, path, self.pose_scale_inputs[key].value(),
                                             frame, mask, prompt_revision=prompt_revision or self._prompts.revision,
                                             initial_pose=initial_pose, initialization=initialization)
        if self.graspgenx:
            request = self._grasp_requests.get(key)
            if request is not None and self._grasp_binding(request) != self._grasp_geometry(key):
                self._grasp_geometry_changed(key, redraw=False)
        self._pose_objects.pop(key, None)
        self._publish_pose_control()
        self.pose_status.setText(f"{key}: registering · frame {frame.metadata.get('sequence')}")
        self._prepare_pose_mesh(key)
        self._plot_pose()
        if redraw:
            self._display_pose_frame()

    def stop_pose(self, key, *, redraw=True):
        if not self._semantic_updating:
            self._task_flow = None
        if self.vlm:
            self._mark_semantic_bindings_stale("수동 FP Stop", (key,))
        self._vlm_plan.pop(key, None)
        self._vlm_mesh_jobs.pop(key, None)
        self._mesh_pose_jobs.pop(key, None)
        self._pose_targets.pop(key, None)
        self._pose_objects.pop(key, None)
        self._pose_meshes.pop(key, None)
        self._pose_mesh_errors.pop(key, None)
        if self.graspgenx:
            request = self._grasp_requests.get(key)
            if request is not None and self._grasp_binding(request) != self._grasp_geometry(key):
                self._grasp_geometry_changed(key, redraw=False)
        self._publish_pose_control()
        self.pose_status.setText(f"{key}: stopped")
        self._plot_pose()
        if not self._pose_targets:
            self._pose_result = None
            if self.graspgenx:
                self._plot_grasp_local()
        if redraw:
            self._display_pose_frame()

    def _display_pose_frame(self):
        if self._snapshot_active and self._snapshot is not None:
            source = self._snapshot.sam3
            frame = source.frame if source is not None else self._snapshot.frame
            self._display_frame(frame, source)
            self.metadata_label.setText(self.metadata_label.text() + " · 고정 snapshot · 명시적 작업만 실행")
            return
        frame = self._latest_frame
        if frame is None and self._sam3_result is not None:
            frame = self._sam3_result.frame
        if frame is None and self._pose_result is not None and not self.aprilcube:
            frame = self._pose_result.frame
        if frame is not None:
            self._display_frame(frame, self._sam3_result, tuple(self._pose_objects.values()),
                                live_frame=self._latest_frame is not None)

    def _input_sam3_result(self):
        return self._snapshot.sam3 if self._snapshot_active and self._snapshot is not None else self._sam3_result

    def capture_snapshot(self):
        from ..ipc.camera import CameraFrame
        from ..snapshot import WorkbenchSnapshot

        frame = self._latest_frame or (self._sam3_result.frame if self._sam3_result is not None else None)
        if frame is None:
            self.status.setText("캡처할 RGB-D frame이 없습니다")
            return None
        objects = tuple({"object_id": key,
                         "mesh_path": (self._pose_targets[key].mesh_path if key in self._pose_targets
                                       else self.pose_mesh_inputs[key].text().strip()),
                         "mesh_scale_to_m": (self._pose_targets[key].mesh_scale_to_m if key in self._pose_targets
                                             else self.pose_scale_inputs[key].value())}
                        for key in ("A", "B")) if hasattr(self, "pose_mesh_inputs") else ()
        self._snapshot = WorkbenchSnapshot(CameraFrame(frame.rgb.copy(), frame.depth_m.copy(), deepcopy(frame.metadata)),
                                           deepcopy(self._sam3_result), deepcopy(self._pose_result), objects,
                                           deepcopy(self._semantic_session.record) if self._semantic_session else None)
        self._show_snapshot()
        return self._snapshot

    def _show_snapshot(self, *, redraw=True):
        self._snapshot_active = True
        self.live_camera_button.setVisible(True)
        self._task_flow = None
        if hasattr(self, "pose_mesh_inputs"):
            previous, self._semantic_updating = self._semantic_updating, True
            try:
                for obj in self._snapshot.objects:
                    key = obj["object_id"]
                    if obj.get("mesh_path"):
                        self.pose_mesh_inputs[key].setText(obj["mesh_path"])
                        self.pose_scale_inputs[key].setValue(obj["mesh_scale_to_m"])
            finally:
                self._semantic_updating = previous
        if self._snapshot.sam3 is not None and hasattr(self, "candidate_inputs"):
            self._mesh_bindings.clear()
            self._mesh_dirty.clear()
            self._update_mesh_candidates(self._snapshot.sam3)
        if redraw:
            self._display_pose_frame()

    def save_snapshot(self, path=None):
        from ..snapshot import save_snapshot

        if self._snapshot is None and self.capture_snapshot() is None:
            return None
        if path is None:
            path, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "RGB-D snapshot 저장", "snapshot.npz", "Snapshot (*.npz)")
        if not path:
            return None
        snapshot = self._snapshot
        return save_snapshot(path, snapshot.frame, sam3=snapshot.sam3, pose_result=snapshot.pose_result,
                             objects=snapshot.objects, semantic=snapshot.semantic)

    def load_snapshot(self, path=None):
        from ..snapshot import load_snapshot

        if path is None:
            path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "RGB-D snapshot 불러오기", "", "Snapshot (*.npz)")
        if not path:
            return None
        self._snapshot = load_snapshot(path)
        self._show_snapshot()
        return self._snapshot

    def show_live_camera(self):
        self._snapshot_active = False
        self.live_camera_button.setVisible(False)
        self._mesh_bindings.clear()
        if self._sam3_result is not None and hasattr(self, "candidate_inputs"):
            self._update_mesh_candidates(self._sam3_result)
        self._display_pose_frame()

    def edit_mask(self, key):
        from ..ipc.camera import CameraFrame
        from ..ipc.sam3_refine import RefineRequest
        from .mask_editor import MaskEditor

        source = self._input_sam3_result()
        item = next((item for item in source.objects if item.object_id == key), None) if source else None
        index = self.candidate_inputs[key].currentData()
        mask = item.masks[index].copy() if item is not None and index is not None else None
        frame = source.frame if source is not None else self._snapshot.frame if self._snapshot_active and self._snapshot else self._latest_frame
        if frame is None:
            self.status.setText("mask 보정할 RGB-D frame이 없습니다")
            return None
        frozen = CameraFrame(frame.rgb.copy(), frame.depth_m.copy(), deepcopy(frame.metadata))
        dialog = MaskEditor(frozen.rgb, mask, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        request = RefineRequest(uuid4().hex, key, self.prompt_inputs[key].text().strip(), frozen,
                                tuple(deepcopy(dialog.canvas.boxes)), tuple(deepcopy(dialog.canvas.points)), mask)
        self._refine_request, self._refine_key = request.request_id, key
        self._refine_source = deepcopy(source)
        self.receiver.submit_refine(request)
        self.sam3_status.setText(f"{key}: 고정 frame {frame.metadata.get('sequence')} SAM3 보정 응답 대기")
        return request

    def _accept_refine_result(self, result):
        from ..snapshot import WorkbenchSnapshot

        if result.request_id != self._refine_request:
            return False
        if result.status == "running":
            return False
        self._refine_request = None
        if result.status == "error":
            self.sam3_status.setText(f"SAM3 보정 오류: {result.error}")
            return False
        source = result.result
        if self._refine_source is not None:
            from ..ipc.sam3 import Sam3Result

            source = Sam3Result(source.frame, source.revision, source.objects + tuple(
                item for item in self._refine_source.objects if item.object_id != self._refine_key))
        self._snapshot = WorkbenchSnapshot(source.frame, source,
                                           self._snapshot.pose_result if self._snapshot else None,
                                           self._snapshot.objects if self._snapshot else (),
                                           self._snapshot.semantic if self._snapshot else None)
        self._show_snapshot(redraw=False)
        self.sam3_status.setText(f"{self._refine_key}: SAM3 보정 완료 · 고정 원본 frame · 수동 mesh / FP 버튼 사용")
        return True

    def _accept_pose_result(self, result):
        accepted = False
        for item in result.objects:
            target = self._pose_targets.get(item.object_id)
            if (target is not None and target.session_id == item.session_id
                    and (not self.aprilcube or item.pose_source == "aprilcube")):
                self._pose_objects[item.object_id] = item
                accepted = True
        if not accepted:
            return False
        self._pose_result = result
        labels = []
        for key, item in self._pose_objects.items():
            position = item.camera_from_object[:3, 3] if item.camera_from_object is not None else None
            location = " · XYZ(m)=" + ", ".join(f"{value:.3f}" for value in position) if position is not None else ""
            mismatch = f" · SAM3 불일치 {item.visual_mismatch_count}/5" if item.visual_mismatch_count else ""
            mesh_error = self._pose_mesh_errors.get(key)
            labels.append(f"{key}: {item.status}{location}{mismatch}" + (f" · {item.error}" if item.error else "")
                          + (f" · mesh preview: {mesh_error}" if mesh_error else ""))
        self.pose_status.setText(("AprilCube · " if self.aprilcube else "") + " / ".join(labels))
        self._plot_pose()
        return True

    def _prepare_pose_mesh(self, key):
        from .mesh_preview import load_pose_mesh, preview_mesh
        from ..ipc.sam3d import MeshGeometry

        target = self._pose_targets[key]
        self._pose_meshes.pop(key, None)
        self._pose_mesh_errors.pop(key, None)
        try:
            result = self._mesh_results.get(key)
            if result is not None and result.mesh_path == target.mesh_path:
                source = self.receiver.mesh_preview(key, result.asset_id) or result.mesh
                preview = preview_mesh(source, max_faces=2000)
                mesh = MeshGeometry(preview.vertices * target.mesh_scale_to_m, preview.faces, preview.colors)
            else:
                mesh = load_pose_mesh(target.mesh_path, target.mesh_scale_to_m, max_faces=2000)
            self._pose_meshes[key] = (target.session_id, mesh)
        except Exception as error:
            # A display asset error must not stop the independent pose session.
            self._pose_mesh_errors[key] = f"{type(error).__name__}: {error}"
            state = "AprilCube · applied" if self.aprilcube else "registering"
            self.pose_status.setText(f"{key}: {state} · mesh preview: {self._pose_mesh_errors[key]}")

    def refine_pose(self, key):
        from .grasp_preview import mesh_binding

        path, scale = mesh_binding(self.pose_mesh_inputs[key].text().strip(), self.pose_scale_inputs[key].value())
        if self.pose_seed_inputs[key].currentData() == "sam3d":
            result, binding = self._mesh_results.get(key), self._mesh_bindings.get(key)
            if (result is None or binding is None or result.camera_from_object is None
                    or str(Path(result.mesh_path).resolve()) != path):
                self.pose_status.setText(f"{key}: 현재 mesh와 같은 SAM3D seed가 필요합니다")
                return
            initial = result.camera_from_object.copy()
            initial[:3, 3] *= scale / result.mesh_scale_to_m
            self.start_pose(key, frame=binding["frame"], mask=binding["mask"], initial_pose=initial, initialization="seed")
            if result.length_unit == "monocular_predicted_scale":
                self.pose_status.setText(self.pose_status.text() + " · monocular 예측 scale × 수동 scale · 실측 단위 아님")
            return
        result = self._snapshot.pose_result if self._snapshot_active and self._snapshot is not None else self._pose_result
        item = next((item for item in result.objects if item.object_id == key), None) if result is not None else None
        record = (next((row for row in self._snapshot.objects if row["object_id"] == key), None)
                  if self._snapshot_active and self._snapshot is not None else None)
        target = self._pose_targets.get(key)
        source_scale = record["mesh_scale_to_m"] if record else target.mesh_scale_to_m if target else None
        if item is None or item.camera_from_object is None or str(Path(item.mesh_path).resolve()) != path or source_scale != scale:
            self.pose_status.setText(f"{key}: 현재 mesh/scale과 같은 FP seed가 필요합니다")
            return
        self.start_pose(key, frame=result.frame, mask=None, initial_pose=item.camera_from_object.copy(), initialization="seed")

    def load_grasp_bundle_path(self, key, path=None):
        from ..grasp_assets import load_grasp_bundle, sha256_file
        from ..ipc.graspgenx import GraspRequest, GraspResult

        if path is None:
            path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, f"{key} 저장 grasp bundle", "", "Grasp bundle (*.json *.yaml *.yml)")
        if not path:
            return None
        try:
            poses, scores, metadata = load_grasp_bundle(path)
            mesh_path, scale = self._grasp_geometry(key)
            geometry = metadata["asset_geometry"]
            if sha256_file(mesh_path) != geometry["mesh_sha256"] or scale != geometry["mesh_scale_to_m"]:
                raise ValueError("선택 mesh/scale과 bundle geometry가 다릅니다")
            source = Path(path).resolve()
            source = source / "metadata.json" if source.is_dir() else source.parent / "metadata.json" if source.suffix in {".yaml", ".yml"} else source
            self._grasp_geometry_changed(key, redraw=False)
            request = GraspRequest(uuid4().hex, metadata["asset_id"], key, mesh_path, scale)
            self._grasp_requests[key] = request
            result = GraspResult(request.request_id, request.asset_id, key, "done", mesh_path, scale, metadata["gripper_id"],
                                 poses, scores, str(source), metadata["gripper_metadata"], cached=True)
            self._accept_grasp_result(result)
            return result
        except (OSError, ValueError, KeyError) as error:
            self.grasp_status.setText(f"{key} bundle 불러오기 오류: {error}")
            return None

    @staticmethod
    def _grasp_binding(value):
        from .grasp_preview import mesh_binding

        return mesh_binding(value.mesh_path, value.mesh_scale_to_m)

    def _grasp_geometry(self, key):
        from .grasp_preview import mesh_binding

        target = self._pose_targets.get(key)
        if target is not None:
            return self._grasp_binding(target)
        return mesh_binding(self.pose_mesh_inputs[key].text().strip(), self.pose_scale_inputs[key].value())

    def _grasp_geometry_changed(self, key, *, redraw=True):
        from .grasp_preview import remove_artists

        if self.grasp_selection and key in self._selection_sources:
            self._invalidate_selection("grasp / mesh 입력 변경", redraw=False)
        self._grasp_requests.pop(key, None)
        self._grasp_results.pop(key, None)
        self._grasp_previews.pop(key, None)
        self._grasp_object_meshes.pop(key, None)
        for artist_key in list(self._grasp_artists):
            if artist_key == key or isinstance(artist_key, tuple) and artist_key[0] == key:
                remove_artists(self._grasp_artists.pop(artist_key))
        combo = self.grasp_inputs[key]
        blocker = QtCore.QSignalBlocker(combo)
        combo.clear()
        combo.addItem("후보 대기", None)
        del blocker
        self.grasp_scores[key].setText("score: —")
        self._grasp_states[key] = "mesh 변경 · 생성 대기"
        self._update_grasp_status()
        if redraw:
            if self._pose_targets:
                self._plot_pose()
            else:
                self._plot_mode = None
                self._plot_grasp_local()
            self._display_pose_frame()

    def generate_grasp(self, key, *, regenerate=False):
        from ..ipc.graspgenx import GraspRequest
        from .grasp_preview import asset_identity

        if key not in self._pose_targets and not self.pose_mesh_inputs[key].text().strip():
            self._grasp_states[key] = "mesh 파일을 선택하세요"
            self._update_grasp_status()
            return
        path, scale = self._grasp_geometry(key)
        catalog = self.asset_store.list_assets() if self.vlm else self._grasp_catalog
        request = GraspRequest(uuid4().hex, asset_identity(path, scale, catalog), key, path, scale, regenerate)
        self._grasp_geometry_changed(key, redraw=False)
        self._grasp_requests[key] = request
        self._grasp_states[key] = "대기 · active FP mesh" if key in self._pose_targets else "대기 · object local mesh"
        self.receiver.submit_grasp(request)
        self._update_grasp_status()
        if self._pose_targets:
            self._plot_pose()
        else:
            self._plot_grasp_local()

    def _accept_grasp_result(self, result):
        from .grasp_preview import load_gripper_preview
        from .mesh_preview import load_pose_mesh

        key = result.object_id
        request = self._grasp_requests.get(key)
        if request is None or result.request_id != request.request_id or result.asset_id != request.asset_id:
            return False
        if self._grasp_binding(result) != self._grasp_binding(request) or self._grasp_geometry(key) != self._grasp_binding(request):
            return False
        self._grasp_states[key] = (f"Error · {result.error}" if result.status == "error" else
                                   "Generating" if result.status == "running" else
                                   f"Done · {len(result.grasps)} 후보" + (" · cache" if result.cached else ""))
        if (result.status == "error" and self._task_flow is not None
                and self._task_flow["grasp_jobs"].get(key) == result.request_id):
            self._task_flow["phase"] = "error"
            self.task_status.setText(f"Task · {key} grasp 오류: {result.error}")
        if result.status == "done":
            self._grasp_results[key] = result
            self._grasp_previews[key] = load_gripper_preview(result.gripper_metadata)
            try:
                cached = self._pose_meshes.get(key)
                target = self._pose_targets.get(key)
                self._grasp_object_meshes[key] = (cached[1] if cached is not None and target is not None
                                                 and cached[0] == target.session_id else
                                                 load_pose_mesh(result.mesh_path, result.mesh_scale_to_m, max_faces=2000))
            except Exception as error:
                self._grasp_states[key] += f" · object preview: {type(error).__name__}"
            combo = self.grasp_inputs[key]
            blocker = QtCore.QSignalBlocker(combo)
            combo.clear()
            for index in range(len(result.grasps)):
                score = f" · {result.scores[index]:.4f}" if result.scores is not None else ""
                combo.addItem(f"{index + 1}{score}", index)
            if not len(result.grasps):
                combo.addItem("후보 없음", None)
            del blocker
            self._select_grasp(key, redraw=False)
            if not self._pose_targets:
                blocker = QtCore.QSignalBlocker(self.grasp_object_input)
                self.grasp_object_input.setCurrentIndex(("A", "B").index(key))
                del blocker
                self._plot_grasp_local()
            else:
                self._plot_pose()
        self._update_grasp_status()
        return result.status == "done"

    def _selected_grasp(self, key):
        result = self._grasp_results.get(key)
        index = self.grasp_inputs[key].currentData()
        if result is None or index is None or self._grasp_binding(result) != self._grasp_geometry(key):
            return None
        return result.grasps[index]

    def _select_grasp(self, key, *, redraw=True):
        result = self._grasp_results.get(key)
        index = self.grasp_inputs[key].currentData()
        self.grasp_scores[key].setText(f"score: {result.scores[index]:.4f}" if result is not None
                                       and index is not None and result.scores is not None else "score: —")
        if redraw:
            if self._pose_targets:
                self._plot_pose()
            else:
                self._plot_grasp_local()
            self._display_pose_frame()

    def _update_grasp_status(self):
        self.grasp_status.setText(" / ".join(f"{key}: {self._grasp_states.get(key, 'mesh 선택 대기')}"
                                            for key in ("A", "B")) + " · A/B=물체 slot · canonical gripper (m)")
        self.grasp_status.setToolTip("\n".join(f"{key}: {value[1]}" for key, value in self._grasp_previews.items()))

    def _selection_source_identity(self, key):
        result = self._grasp_results.get(key)
        return (result.request_id, result.bundle_path, self._grasp_binding(result)) if result is not None else None

    def _selection_object_record(self, key, row=None):
        result, target, pose = self._grasp_results.get(key), self._pose_targets.get(key), self._pose_objects.get(key)
        geometry = row.get("geometry") or {} if row is not None else {}
        path = target.mesh_path if target is not None else result.mesh_path if result is not None else geometry.get("mesh_path")
        scale = (target.mesh_scale_to_m if target is not None else result.mesh_scale_to_m if result is not None else
                 geometry.get("mesh_scale_to_m", 1.0))
        record = {"object_id": key, "observation_id": row["observation_id"] if row is not None else f"manual-{key}",
                "instance_id": row["instance_id"] if row is not None else f"manual-{result.request_id}",
                "bundle_path": result.bundle_path if result is not None else None,
                "mesh_path": path, "mesh_scale_to_m": scale,
                "camera_from_object": (pose.camera_from_object.tolist() if pose is not None
                                       and pose.camera_from_object is not None else None)}
        fk = (self._grasp_fk or {}).get("objects", {}).get(record["observation_id"])
        if fk is not None and fk["instance_id"] == record["instance_id"]:
            record["grasp_fk"] = deepcopy(fk)
        return record

    def attach_grasp_fk(self):
        from ..ipc.grasp_selection import SelectionRequest
        from .mesh_preview import load_pose_mesh

        attachment = self.selection_reference()
        if attachment is None:
            self.fk_status.setText("표시용으로 고정할 grasp 선택 결과가 없습니다")
            return None
        self._fk_attachment, self._fk_meshes = deepcopy(attachment), {}
        self._fk_previews = {key: value[0] for key, value in self._grasp_previews.items()}
        for obj in attachment["objects"]:
            key = obj["object_id"]
            mesh = self._grasp_object_meshes.get(key)
            if mesh is None and obj["mesh_path"]:
                try:
                    mesh = load_pose_mesh(obj["mesh_path"], obj["mesh_scale_to_m"], max_faces=2000)
                except (OSError, ValueError):
                    pass
            if mesh is not None:
                self._fk_meshes[key] = mesh
        request = SelectionRequest(uuid4().hex, {}, (), action="attach", attachment=attachment)
        self._fk_request_id = request.request_id
        self.receiver.submit_selection(request)
        self.fk_status.setText("계획 파지 관계 고정 요청 · 실제 파지 확인/로봇 명령 아님")
        return request

    def detach_grasp_fk(self):
        from ..ipc.grasp_selection import SelectionRequest

        request = SelectionRequest(uuid4().hex, {}, (), action="detach")
        self._fk_request_id = request.request_id
        self.receiver.submit_selection(request)
        return request

    def _accept_fk_result(self, result):
        if result.request_id != self._fk_request_id:
            return
        if result.status == "error":
            self.fk_status.setText(f"Grasp FK 오류: {result.error}")
            return
        if result.status == "running" or result.selection is None:
            return
        self._grasp_fk = result.selection if result.selection["state"] != "detached" else None
        self.fk_status.setText("Grasp FK · " + result.selection["state"] + " · 계획 관계 + 실측 FK · RGB/FP 원본 유지 · 로봇 명령 없음")
        self.fk_status.setToolTip(json.dumps(result.selection, ensure_ascii=False, indent=2, allow_nan=False))
        if self._grasp_fk is None:
            self._plot_mode = None
        self._redraw_grasp_selection()

    def _plot_grasp_fk(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from .grasp_preview import update_grasp_artists

        if self._plot_mode != "grasp_fk":
            self.axes.clear()
            self._grasp_artists, self._fk_artists = {}, {}
            self._plot_mode = "grasp_fk"
            self.axes.set_title("base_link · Grasp FK / recorded vision · 표시 전용")
            self.axes.set_xlabel("X (m)")
            self.axes.set_ylabel("Y (m)")
            self.axes.set_zlabel("Z (m)")
        source = {row["observation_id"]: row for row in self._fk_attachment["objects"]}
        poses = {key: (value.get("base_from_object"), "vision · requested pose")
                 for key, value in (self._fk_attachment.get("object_poses") or {}).items()}
        poses.update({key: (row["base_from_object"], "Grasp FK · measured")
                      for key, row in self._grasp_fk["objects"].items()})
        visible = {source[key]["object_id"] for key, (matrix, _label) in poses.items()
                   if matrix is not None and key in source}
        from .grasp_preview import remove_artists

        for artist_key in list(self._fk_artists):
            key = artist_key[0] if isinstance(artist_key, tuple) else artist_key
            if key not in visible:
                remove_artists(self._fk_artists.pop(artist_key))
        points = [np.zeros(3)]
        for observation, (matrix, label) in poses.items():
            if matrix is None or observation not in source:
                continue
            key, pose = source[observation]["object_id"], np.asarray(matrix)
            artists = self._fk_artists.setdefault(key, {})
            if "label" not in artists:
                artists["label"] = self.axes.text(*pose[:3, 3], f"{key} {label}")
            artists["label"].set_text(f"{key} {label}")
            artists["label"].set_position_3d(pose[:3, 3])
            mesh = self._fk_meshes.get(key)
            if mesh is not None:
                vertices = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
                if "mesh" not in artists:
                    artists["mesh"] = Poly3DCollection(vertices[mesh.faces], facecolors="#39d98a", alpha=.6, linewidths=0)
                    self.axes.add_collection3d(artists["mesh"])
                else:
                    artists["mesh"].set_verts(vertices[mesh.faces])
                points.extend(vertices)
            points.append(pose[:3, 3])
            for arm, target in self._fk_attachment["selected_candidate"]["targets"].items():
                if target["observation_id"] == observation:
                    gripper = self._fk_artists.setdefault((key, arm), {})
                    canonical = np.asarray(target["object_from_gripper"])
                    preview = self._fk_previews.get(key)
                    points.extend(update_grasp_artists(self.axes, pose @ canonical, preview, gripper))
        points = np.asarray(points)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        half = max(float(np.ptp(points, axis=0).max()) / 2, .1)
        self.axes.set_xlim(center[0] - half, center[0] + half)
        self.axes.set_ylim(center[1] - half, center[1] + half)
        self.axes.set_zlim(center[2] - half, center[2] + half)
        self.axes.set_box_aspect((1, 1, 1))
        self.mesh_view.draw_idle()

    def request_grasp_selection(self):
        from ..ipc.grasp_selection import SelectionRequest

        mode = self.selection_mode_input.currentData()
        if mode == "semantic":
            if self._semantic_session is None or self._semantic_session.record["coordination"] is None:
                self.selection_status.setText("VLM 팔 정보 없음 · 수동 mode를 선택하거나 VLM으로 분석하세요")
                return
            coordination = deepcopy(self._semantic_session.record["coordination"])
            objects = tuple(self._selection_object_record(row["slot"], row)
                            for row in self._semantic_session.record["objects"] if row["binding_active"])
        else:
            key = ("A", "B")[self.selection_object_input.currentIndex()]
            if self._grasp_results.get(key) is None:
                self.selection_status.setText(f"{key}: 기존 grasp bundle 필요 · 후보를 자동 생성하지 않습니다")
                return
            row = self._semantic_session.object_for_slot(key) if self._semantic_session is not None else None
            task_rows = [item for item in self._task_rows() if self._task_flow is not None
                         and item["observation_id"] in (self._task_flow["source"], self._task_flow["target"])]
            if task_rows:
                row = next(item for item in task_rows if item["slot"] == key)
            objects = (tuple(self._selection_object_record(item["slot"], item) for item in task_rows)
                       if task_rows else (self._selection_object_record(key, row),))
            grasped = next(item for item in objects if item["object_id"] == key)
            arms = ("left", "right") if mode == "dual" else (mode,)
            coordination = {"arm_mode": "dual" if mode == "dual" else "single", "constraints": [],
                            "assignments": [{"arm": arm, "observation_id": grasped["observation_id"],
                                             "role": "manual selection", "grasp_region": ""} for arm in arms],
                            "reason": "Explicit manual UI selection"}
        assigned = {item["observation_id"] for item in coordination["assignments"] if item["observation_id"] is not None}
        by_id = {item["observation_id"]: item for item in objects}
        if any(observation not in by_id or not by_id[observation]["bundle_path"] for observation in assigned):
            self.selection_status.setText("배정된 관측의 기존 grasp bundle 필요 · 현재 binding과 후보를 확인하세요")
            return
        self._invalidate_selection("후처리 요청 준비", redraw=False)
        request = SelectionRequest(uuid4().hex, coordination, objects)
        self._selection_request = request
        self._selection_sources = {item["object_id"]: self._selection_source_identity(item["object_id"]) for item in objects}
        self._selection_semantic_id = (self._semantic_session.record["session_id"] if mode == "semantic" else None)
        self.receiver.submit_selection(request)
        self.selection_status.setText("grasp 선택 worker 응답 대기 · 별도 worker 설정 필요 · 후보 자동 생성 없음")
        self._redraw_grasp_selection()
        return request

    def _invalidate_selection(self, reason, *, redraw=True):
        if not self.grasp_selection or not hasattr(self, "selection_candidate_input"):
            return
        self._selection_request, self._selection_result = None, None
        self._selection_lock = None
        self._selection_sources, self._selection_semantic_id = {}, None
        blocker = QtCore.QSignalBlocker(self.selection_candidate_input)
        self.selection_candidate_input.clear()
        self.selection_candidate_input.addItem("선택 결과 대기", None)
        del blocker
        self.selection_status.setText(reason)
        self.selection_details.clear()
        if self.vlm:
            self._sync_semantic_references()
        if redraw:
            self._redraw_grasp_selection()
            self._display_pose_frame()

    def _accept_selection_result(self, result):
        if self._selection_request is None or result.request_id != self._selection_request.request_id:
            return False
        if any(self._selection_source_identity(key) != source for key, source in self._selection_sources.items()):
            self._invalidate_selection("grasp 입력 연결 변경", redraw=False)
            return False
        if self._selection_semantic_id is not None and self._semantic_session.record["session_id"] != self._selection_semantic_id:
            self._invalidate_selection("semantic instance 변경", redraw=False)
            return False
        if result.status == "error":
            self.selection_status.setText(f"grasp 선택 오류: {result.error}")
            if self._task_flow is not None and self._task_flow["selection_id"] == result.request_id:
                self._task_flow["phase"] = "error"
                self.task_status.setText(f"Task · grasp 선택 오류: {result.error}")
            return False
        if result.status == "running":
            self.selection_status.setText("grasp 후처리 중 · 기존 후보에 대한 요청 snapshot 평가")
            return False
        if self._selection_lock is not None:
            return False
        self._selection_result = deepcopy(result.selection)
        blocker = QtCore.QSignalBlocker(self.selection_candidate_input)
        self.selection_candidate_input.clear()
        for index, candidate in enumerate(result.selection.get("candidates", [])):
            score = candidate.get("pair_score", candidate.get("geometry_score", candidate.get("model_score")))
            label = f"{candidate['candidate_id']}" + (f" · {score:.3f}" if score is not None else "")
            self.selection_candidate_input.addItem(label, index)
        if not self.selection_candidate_input.count():
            self.selection_candidate_input.addItem("후보 없음", None)
        del blocker
        candidate = self._selected_selection_candidate()
        if not self._pose_targets and candidate is not None and candidate["targets"]:
            key = next(iter(candidate["targets"].values()))["object_id"]
            blocker = QtCore.QSignalBlocker(self.grasp_object_input)
            self.grasp_object_input.setCurrentIndex(("A", "B").index(key))
            del blocker
        self._selection_candidate_changed(redraw=False)
        if self._task_flow is not None and self._task_flow["selection_id"] == result.request_id:
            self._task_flow["phase"] = "done"
            self.task_status.setText("Task · Source grasp 선택 완료 · 로봇 명령 없음")
        self._redraw_grasp_selection()
        return True

    def _selected_selection_candidate(self):
        if self._selection_result is None:
            return None
        index = self.selection_candidate_input.currentData()
        if self._selection_lock is not None and self._selection_lock["index"] == index:
            return self._selection_lock["candidate"]
        return self._selection_result["candidates"][index] if index is not None else None

    def selection_reference(self):
        candidate = self._selected_selection_candidate() if self.grasp_selection else None
        if candidate is None:
            return None
        return {"request_id": self._selection_request.request_id, "coordination": deepcopy(self._selection_request.coordination),
                "objects": deepcopy(self._selection_request.objects), "selected_candidate": deepcopy(candidate),
                "object_poses": deepcopy(self._selection_result.get("objects")),
                **{key: deepcopy(value) for key, value in self._selection_result.items() if key not in {"candidates", "objects"}}}

    def _selection_candidate_changed(self, *_args, redraw=True):
        if self._selection_result is None:
            return
        index = self.selection_candidate_input.currentData()
        self._selection_lock = ({"index": index, "candidate": deepcopy(self._selection_result["candidates"][index])}
                                if index is not None and self._selection_result.get("arm_mode") == "single" else None)
        candidate = self._selected_selection_candidate()
        context = ("로봇 미연결 · geometry-only · IK 미검증" if not self._selection_result.get("robot_connected") else
                   "로봇 기준 · IK 요청 snapshot 평가" if self._selection_result.get("robot_ik_checked") else
                   "로봇 기준 pose · IK 미검증")
        self.selection_status.setText(f"{self._selection_result.get('arm_mode', 'unknown')} · " + context
                                      + (f" · {candidate.get('ik_status', 'not_checked')}" if candidate else " · 후보 없음")
                                      + (" · single 선택 고정 · FP는 표시만 갱신" if self._selection_lock is not None else "")
                                      + " · 로봇 명령 전송 없음")
        self.selection_details.setPlainText(json.dumps(self.selection_reference() or self._selection_result,
                                                       ensure_ascii=False, indent=2, allow_nan=False))
        if self.vlm:
            self._sync_semantic_references()
        if redraw:
            self._redraw_grasp_selection()
            self._display_pose_frame()

    def _redraw_grasp_selection(self):
        self._plot_pose() if self._pose_targets else self._plot_grasp_local()

    def _display_grasps(self, key):
        candidate = self._selected_selection_candidate() if self.grasp_selection else None
        if candidate is not None:
            return [(arm, np.asarray(target["object_from_gripper"], dtype=float))
                    for arm, target in candidate["targets"].items() if target["object_id"] == key]
        selected = self._selected_grasp(key)
        return [("gripper", selected)] if selected is not None else []

    @staticmethod
    def _grasp_artist_key(key, arm):
        return key if arm == "gripper" else (key, arm)

    def _label_grasp_artist(self, artists, key, arm, transform):
        if arm != "gripper":
            if "label" not in artists:
                artists["label"] = self.axes.text(*transform[:3, 3], f"{key} {arm}", zorder=13)
            else:
                artists["label"].set_position_3d(transform[:3, 3])

    def _plot_grasp_local(self, *_args):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from .grasp_preview import remove_artists, update_grasp_artists

        if self._grasp_fk is not None:
            self._plot_grasp_fk()
            return
        if self._pose_targets:
            self._plot_pose()
            return
        key = ("A", "B")[self.grasp_object_input.currentIndex()]
        if self._plot_mode != ("grasp", key):
            self.axes.clear()
            self._grasp_artists = {}
            self._grasp_local_object_artist = None
            self._plot_mode = ("grasp", key)
            self.axes.set_xlabel("X (m)")
            self.axes.set_ylabel("Y (m)")
            self.axes.set_zlabel("Z (m)")
            self.axes.set_title(f"{key} · mesh / grasp · object coordinates")
        points = [np.zeros(3)]
        mesh = self._grasp_object_meshes.get(key)
        if mesh is not None:
            triangles = mesh.vertices[mesh.faces]
            if self._grasp_local_object_artist is None:
                self._grasp_local_object_artist = Poly3DCollection(triangles, facecolors="#39d98a", alpha=0.55,
                                                                  linewidths=0, shade=True)
                self.axes.add_collection3d(self._grasp_local_object_artist)
            else:
                self._grasp_local_object_artist.set_verts(triangles)
            points.extend(mesh.vertices)
        display = self._display_grasps(key)
        desired = {self._grasp_artist_key(key, arm) for arm, _pose in display}
        for artist_key in list(self._grasp_artists):
            if artist_key not in desired:
                remove_artists(self._grasp_artists.pop(artist_key))
        for arm, selected in display:
            artists = self._grasp_artists.setdefault(self._grasp_artist_key(key, arm), {})
            preview = self._grasp_previews.get(key, (None, ""))[0]
            points.extend(update_grasp_artists(self.axes, selected, preview, artists))
            self._label_grasp_artist(artists, key, arm, selected)
        points = np.asarray(points)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        half = max(float(np.ptp(points, axis=0).max()) / 2, 0.1)
        self.axes.set_xlim(center[0] - half, center[0] + half)
        self.axes.set_ylim(center[1] - half, center[1] + half)
        self.axes.set_zlim(center[2] - half, center[2] + half)
        self.axes.set_box_aspect((1, 1, 1))
        self.mesh_view.draw_idle()

    def _plot_pose(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from .grasp_preview import remove_artists, update_grasp_artists

        if self._grasp_fk is not None:
            self._plot_grasp_fk()
            return
        if self._plot_mode != "pose":
            self.axes.clear()
            self._pose_mesh_artists, self._pose_axis_artists, self._pose_label_artists = {}, {}, {}
            self._grasp_artists = {}
            self.axes.set_xlabel("X (m)")
            self.axes.set_ylabel("Y (m)")
            self.axes.set_zlabel("Z (m)")
            self.axes.set_title(f"{'AprilCube' if self.aprilcube else 'FoundationPose'}\nMesh / camera coordinates")
            self._plot_mode = "pose"
        visible = {key for key, item in self._pose_objects.items() if item.camera_from_object is not None}
        desired = ({self._grasp_artist_key(key, arm) for key in visible for arm, _pose in self._display_grasps(key)}
                   if self.graspgenx else set())
        for artist_key in list(self._grasp_artists):
            if artist_key not in desired:
                remove_artists(self._grasp_artists.pop(artist_key))
        for artists in (self._pose_mesh_artists, self._pose_axis_artists, self._pose_label_artists):
            for key in list(artists):
                if key not in visible:
                    value = artists.pop(key)
                    for artist in value if isinstance(value, list) else [value]:
                        artist.remove()
        points = [np.zeros(3)]
        for key, item in self._pose_objects.items():
            pose = item.camera_from_object
            if pose is None:
                continue
            origin = pose[:3, 3]
            if key not in self._pose_axis_artists:
                self._pose_axis_artists[key] = [self.axes.plot([], [], [], color=color, linewidth=1.5, zorder=10)[0]
                                               for color in ("r", "g", "b")]
                self._pose_label_artists[key] = self.axes.text(*origin, key, zorder=11)
            self._pose_label_artists[key].set_position_3d(origin)
            points.append(origin)
            for axis, artist in zip(pose[:3, :3].T, self._pose_axis_artists[key]):
                end = origin + axis * 0.05
                points.append(end)
                artist.set_data_3d(np.stack((origin, end)).T)
            cached = self._pose_meshes.get(key)
            if cached is not None and cached[0] == item.session_id:
                mesh = cached[1]
                vertices = mesh.vertices @ pose[:3, :3].T + origin
                triangles = vertices[mesh.faces]
                if key not in self._pose_mesh_artists:
                    colors = mesh.colors[mesh.faces].mean(axis=1) / 255 if mesh.colors is not None else (
                        "#39d98a" if key == "A" else "#ffb454")
                    artist = Poly3DCollection(triangles, facecolors=colors, linewidths=0, shade=True, alpha=0.65)
                    self._pose_mesh_artists[key] = artist
                    self.axes.add_collection3d(artist)
                else:
                    self._pose_mesh_artists[key].set_verts(triangles)
                points.extend(vertices)
            if self.graspgenx:
                for arm, selected in self._display_grasps(key):
                    artists = self._grasp_artists.setdefault(self._grasp_artist_key(key, arm), {})
                    preview = self._grasp_previews.get(key, (None, ""))[0]
                    points.extend(update_grasp_artists(self.axes, pose @ selected, preview, artists))
                    self._label_grasp_artist(artists, key, arm, pose @ selected)
        points = np.asarray(points)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        half = max(float(np.ptp(points, axis=0).max()) / 2, 0.1)
        self.axes.set_xlim(center[0] - half, center[0] + half)
        self.axes.set_ylim(center[1] - half, center[1] + half)
        self.axes.set_zlim(center[2] - half, center[2] + half)
        self.axes.set_box_aspect((1, 1, 1))
        self.mesh_view.draw_idle()

    def _finish(self, code, message):
        self.exit_code = code
        self.status.setText(message)
        log_record("error" if code else "ui-completed", message=message, received_count=self.received_count)
        self.close()

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self.receiver.stop()
        if self.receiver.is_alive():
            event.ignore()
            return
        error = self.receiver.snapshot().get("error")
        if error and self.exit_code == 0:
            self.exit_code = 1
            self.status.setText(f"수신 종료 오류: {error}")
            log_record("error", worker="camera_ui", error=error)
        self.timer.stop()
        log_record("ui-stopped", received_count=self.received_count, exit_code=self.exit_code)
        event.accept()
