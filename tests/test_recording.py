"""LeRobot storage preserves source pixels, metric depth, and failed detections."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_workbench.ipc.camera import CameraFrame
from grasp_workbench.ipc.foundationpose import ObjectPose, PoseResult
from grasp_workbench.recording import LeRobotRecorder, REPO_ID


def result(sequence=3):
    rgb = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    depth = np.arange(24, dtype=np.float32).reshape(4, 6) / np.float32(10)
    depth[0, :3] = (np.nan, np.inf, -np.inf)
    frame = CameraFrame(rgb, depth, {
        "camera_id": "camera", "stream_id": "stream", "sequence": sequence,
        "capture_time_ns": 1792558000123456789, "received_monotonic_ns": 9876543210123456,
        "timestamp_domain": "device", "camera_frame": "camera_optical",
        "intrinsics": {"fx": 12., "fy": 13., "cx": 3., "cy": 2.,
                       "width": 6, "height": 4, "distortion_model": "none", "coefficients": []},
    })
    pose = np.eye(4)
    pose[:3, 3] = (.12, -.3, .9)
    return PoseResult(frame, (
        ObjectPose("A", "a", "cube.ply", "tracking", pose, pose_source="aprilcube",
                   diagnostics={"n_tags": 2, "tag_ids": [3, 4], "predicted": False,
                                "reproj_error": .125}),
        ObjectPose("B", "b", "", "unavailable", pose_source="aprilcube"),
    ))


class Dataset:
    def __init__(self):
        self.frames, self.episodes = [], []
        self.discards = self.finalizes = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes.append(self.frames)
        self.frames = []

    def clear_episode_buffer(self):
        self.frames = []
        self.discards += 1

    def finalize(self):
        self.finalizes += 1


def recorder(tmp_path, **kwargs):
    data, calls = Dataset(), []
    instance = LeRobotRecorder(tmp_path / "dataset", dataset_factory=(
        lambda **options: calls.append(options) or data), **kwargs)
    return instance, data, calls


def test_lazy_lossless_image_features_no_fake_robot_or_actions(tmp_path):
    instance, data, calls = recorder(tmp_path, fps=30)
    source = result()
    assert not instance.output_dir.exists()
    instance.start()
    assert not calls
    instance.add_frame(source)
    options, = calls
    assert options["root"] == instance.output_dir and options["repo_id"] == REPO_ID
    assert options["fps"] == 30 and not options["use_videos"]
    assert options["image_writer_processes"] == options["image_writer_threads"] == 0
    assert options["video_backend"] == "pyav"
    assert "robot_type" not in options
    assert "action" not in options["features"]
    rgb = options["features"]["observation.images.rgb"]
    depth = options["features"]["observation.depth_m"]
    assert rgb["dtype"] == "image" and rgb["shape"] == (3, 4, 6)
    assert depth["dtype"] == "image" and depth["shape"] == (1, 4, 6)
    assert depth["info"] == {"is_depth_map": True, "depth_unit": "m"}
    stored, = data.frames
    np.testing.assert_array_equal(stored["observation.images.rgb"], source.frame.rgb.transpose(2, 0, 1))
    np.testing.assert_array_equal(stored["observation.depth_m"].view(np.uint32),
                                  source.frame.depth_m[None, ...].view(np.uint32))
    assert json.loads(stored["observation.camera_metadata"]) == source.frame.metadata
    assert json.loads(stored["observation.camera_metadata"])["capture_time_ns"] == 1792558000123456789
    np.testing.assert_array_equal(stored["observation.pose.A"], source.objects[0].camera_from_object.ravel())
    assert stored["observation.pose.A_valid"].item()
    assert not stored["observation.pose.B_valid"].item()
    np.testing.assert_array_equal(stored["observation.pose.B"], np.zeros(16))
    assert stored["observation.pose.B_valid"].dtype == np.bool_
    objects = json.loads(stored["observation.objects"])
    assert objects[1]["status"] == "unavailable" and objects[1]["camera_from_object"] is None
    assert objects[0]["pose_source"] == "aprilcube"
    assert objects[0]["diagnostics"] == source.objects[0].diagnostics
    assert instance.episode_frames == instance.total_frames == 1
    instance.close()
    assert instance.closed and data.finalizes == 1 and len(data.episodes) == 1


def test_save_discard_and_orderly_close_save_only_active_nonempty_episode(tmp_path):
    instance, data, _ = recorder(tmp_path)
    instance.start()
    instance.add_frame(result())
    instance.add_frame(result(4))
    instance.save()
    assert not instance.recording and instance.episode_index == 1
    assert instance.episode_frames == 0 and instance.total_frames == 2
    instance.start()
    instance.add_frame(result(5))
    instance.discard()
    assert data.discards == 1 and instance.episode_index == 1 and instance.total_frames == 2
    instance.start()
    instance.add_frame(result(6))
    instance.close()
    instance.close()
    assert instance.closed and instance.episode_index == 2 and instance.total_frames == 3
    assert len(data.episodes) == 2 and data.finalizes == 1
    assert [len(episode) for episode in data.episodes] == [2, 1]


def test_empty_episode_has_no_dataset_and_close_does_not_invent_episode(tmp_path):
    instance, data, calls = recorder(tmp_path)
    instance.start()
    with pytest.raises(RuntimeError, match="empty"):
        instance.save()
    instance.close()
    assert not calls and data.finalizes == 0
    assert instance.episode_index == instance.episode_frames == instance.total_frames == 0
    assert not instance.output_dir.exists()


def test_camera_only_recording_has_explicit_invalid_a_b(tmp_path):
    instance, data, _ = recorder(tmp_path)
    instance.start()
    source = result()
    instance.add_frame(PoseResult(source.frame, ()))
    stored, = data.frames
    assert json.loads(stored["observation.objects"]) == []
    assert not stored["observation.pose.A_valid"].item()
    assert not stored["observation.pose.B_valid"].item()
    instance.close()


def test_only_selected_run_metadata_written_not_entire_config(tmp_path):
    instance, _, _ = recorder(tmp_path, run_metadata={
        "source": "pose", "aprilcube": {"fast": True, "filter_enabled": True},
        "vlm": {"api_key": "should-not-be-written"}, "api_key": "should-not-be-written",
    })
    instance.start()
    instance.add_frame(result())
    text = (instance.output_dir / "recording.json").read_text()
    metadata = json.loads(text)
    assert metadata["run"] == {"source": "pose", "aprilcube": {"fast": True, "filter_enabled": True}}
    assert "should-not-be-written" not in text
    assert metadata["depth_unit"] == "m" and metadata["pose_layout"] == "row_major_4x4"
    instance.close()


@pytest.mark.parametrize("fps", [0, -1, 2.5, True])
def test_invalid_fps(fps, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        LeRobotRecorder(tmp_path, fps)


def test_lifecycle_errors_and_dimension_change_do_not_count_frame(tmp_path):
    instance, data, _ = recorder(tmp_path)
    with pytest.raises(RuntimeError, match="Start"):
        instance.add_frame(result())
    for method in (instance.save, instance.discard):
        with pytest.raises(RuntimeError, match="active"):
            method()
    instance.start()
    with pytest.raises(RuntimeError, match="already"):
        instance.start()
    instance.add_frame(result())
    changed = result()
    changed.frame.rgb = changed.frame.rgb[:2]
    changed.frame.depth_m = changed.frame.depth_m[:2]
    with pytest.raises(ValueError, match="dimensions"):
        instance.add_frame(changed)
    assert instance.total_frames == 1 and len(data.frames) == 1
    instance.close()
    with pytest.raises(RuntimeError, match="closed"):
        instance.start()


def test_close_still_finalizes_if_saving_raises(tmp_path):
    instance, data, _ = recorder(tmp_path)
    instance.start()
    instance.add_frame(result())

    def fail():
        raise OSError("disk full")

    data.save_episode = fail
    with pytest.raises(OSError, match="disk full"):
        instance.close()
    assert instance.closed and data.finalizes == 1


def test_real_lerobot_image_dataset_roundtrip(tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    instance = LeRobotRecorder(tmp_path / "dataset")
    sources = [result(sequence) for sequence in range(3)]
    instance.start()
    for source in sources:
        instance.add_frame(source)
    instance.save()
    instance.start()
    instance.add_frame(result(4))
    instance.discard()
    instance.close()
    dataset = LeRobotDataset(REPO_ID, root=instance.output_dir,
                            depth_output_unit="m", video_backend="pyav", return_uint8=True)
    assert len(dataset) == 3 and dataset.num_episodes == 1
    for index, source in enumerate(sources):
        row = dataset[index]
        rgb = np.rint(row["observation.images.rgb"].numpy() * 255).astype(np.uint8)
        np.testing.assert_array_equal(rgb, source.frame.rgb.transpose(2, 0, 1))
        raw = dataset.hf_dataset.with_format(None)[index]
        np.testing.assert_array_equal(np.asarray(raw["observation.images.rgb"]), source.frame.rgb)
        np.testing.assert_array_equal(row["observation.depth_m"].numpy().view(np.uint32),
                                      source.frame.depth_m[None, ...].view(np.uint32))
        assert json.loads(row["observation.camera_metadata"]) == source.frame.metadata
        objects = json.loads(row["observation.objects"])
        assert objects[1]["status"] == "unavailable"
        assert objects[0]["diagnostics"] == source.objects[0].diagnostics
        np.testing.assert_array_equal(np.asarray(raw["observation.pose.A"], dtype=np.float64),
                                      source.objects[0].camera_from_object.ravel())
        assert not row["observation.pose.B_valid"].item()


def test_real_dataset_factory_is_bound_before_ready_without_creating_dataset(tmp_path, monkeypatch):
    calls = []

    def create(**options):
        calls.append(options)
        return Dataset()

    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset",
                        SimpleNamespace(LeRobotDataset=SimpleNamespace(create=create)))
    instance = LeRobotRecorder(tmp_path / "dataset")
    assert instance.dataset_factory is create
    assert instance.dataset is None and not calls and not instance.output_dir.exists()
    instance.start()
    assert not calls
    instance.add_frame(result())
    assert len(calls) == 1
    instance.close()
