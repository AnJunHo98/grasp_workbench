# LeRobot data recording

The optional `data_recorder` worker stores received RGB-D and A/B pose results using **LeRobotDataset 0.6.1 APIs**.
It does not run `lerobot-record`, reopen a camera, run inference, control a robot, train a policy, or upload to the Hub.
Camera and model workers keep their existing iceoryx2 interfaces. See the
[official dataset overview](https://huggingface.co/docs/lerobot/v0.6.1/lerobot-dataset-v3) and
[version-pinned dataset API](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/lerobot_dataset.py).

## Optional environment

LeRobot 0.6.1 requires Python 3.12+ and has its own NumPy/PyTorch dependencies. Keep it separate from camera and
model environments rather than upgrading those environments for recording. The recorder only needs the workbench
base dependencies and its `recording` extra; it does not need a ZED/RealSense SDK or Qt.

From the repository checkout:

```bash
python3.12 -m venv .venv-recorder
.venv-recorder/bin/python -m pip install -e '.[recording]'
```

An existing compatible environment can be used instead: install `.[recording]` into it and set `recording.python`
to that environment's Python. `python: null` uses the Python that launched the workbench and therefore requires
the optional dependencies in that environment. The main UI environment still needs its normal `.[ui]` dependencies.
The `.venv*/` and `runs/` paths are excluded from Git.

## YAML configuration

Add only the settings you want to override to your ignored `config.local.yaml`:

```yaml
recording:
  enabled: true
  python: .venv-recorder/bin/python
  output_dir: runs/datasets
  fps: 30
  source: pose             # pose / camera
```

Then use the normal command:

```bash
grasp-workbench run
```

| Setting | Meaning |
|---|---|
| `enabled` | Starts an idle independent recorder and enables its UI controls. It does not start an episode. |
| `python` | Optional recorder-only Python environment; `null` uses the launcher Python. |
| `output_dir` | Parent directory for a new timestamp/UUID dataset run. Relative paths resolve against the defining YAML file. |
| `fps` | Nominal LeRobot dataset rate, not a guarantee of the real acquisition rate. Original source timestamps are also stored. |
| `source: pose` | Subscribes to `/foundationpose/results`: exact pose-input RGB-D together with A/B results, including unavailable poses. |
| `source: camera` | Subscribes to the raw camera service: RGB-D and camera metadata, without pose labels. |

`pose` needs an independently enabled AprilCube or FoundationPose worker to publish results. Enabling recording does
not enable any model worker. It can wait with zero frames when no pose results are available. Use `camera` for
camera-only recording. `aprilcube.enabled: true` still disables the six model workers but **does not disable recording**.

Advanced CLI overrides use the existing naming pattern: `--recording`, `--recording-python`,
`--recording-output-dir`, `--recording-fps`, and `--recording-source`. Ordinary use needs only YAML and `run`.

## Episode controls

In the left **Recording** tab:

1. Select **Start Recording** to begin receiving frames into a new episode.
2. Move the observed objects through the conditions you want to inspect.
3. Select **Save Episode** to finish and persist the current episode, or **Discard Episode** to remove its unsaved frames.
4. Start another episode if needed. Close the UI or stop the worker normally when finished: shutdown saves a nonempty
   active episode and finalizes the dataset. **Finalize before loading the dataset**; Save Episode alone can leave
   LeRobot's Parquet writers open until the worker exits.

The tab shows recorder status, output directory, episode index, episode frame count and total frame count.
Changing tabs does not stop recording. No recording starts automatically when the UI opens.

## Stored observations

Each dataset run contains LeRobot's `meta/` and `data/` with image-backed observations. It uses `use_videos=False` so no lossy RGB
video encoding is introduced:

| Observation | Representation |
|---|---|
| RGB | Original uint8 RGB in lossless PNG image storage, feature `observation.images.rgb` |
| Depth | Original float32 metre depth in TIFF image storage, feature `observation.depth_m`, `is_depth_map: true`, `depth_unit: m` |
| Camera metadata | JSON string preserving camera/stream identity, source sequence, original timestamps, clock-domain and intrinsics metadata |
| A/B pose | Flattened 4x4 `camera_from_object` values plus a validity flag for each slot |
| Object states | JSON string preserving status, pose source, session/mesh reference and available diagnostics |

The validity flag must be checked before interpreting a pose. Missing/unavailable results are not converted into valid
identity transforms. Camera-only collection has no valid A/B pose labels. A/B are object slots, not robot arms.
There is no fabricated robot action feature. RGB is raw camera RGB, not the UI's axes, masks or mesh overlay.

In pose mode, the pose and RGB-D originate from **the same `PoseResult`**; the recorder does not join a latest raw
camera image to an unrelated latest pose. Existing AprilCube diagnostics include detected tag IDs/count, inliers,
visible faces, reprojection error and prediction/success flags when supplied by the pose worker. These are SDK outputs,
not new acceptance rules or inferred rejection reasons. Camera mode does not generate these diagnostics.

Do not dump private YAML or provider credentials into a dataset. Treat recorded RGB and local mesh references as
potentially private data and inspect the dataset before sharing it. Recording does not automatically upload anything.
The run's `recording.json` stores only the recording source plus a small launcher settings whitelist: camera type,
resolution and FPS, and AprilCube enabled/fast/filter values. The launcher does not include device serials, private
configuration paths or provider settings in this run metadata. Per-frame camera metadata still preserves its source identity.

## Read a saved run

After normal worker shutdown/finalization, use the recorder environment to read the saved path. Read depth explicitly in metres: LeRobot's default depth output
unit is not metres.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/grasp_workbench",
    root="runs/datasets/YOUR_SAVED_RUN",
    depth_output_unit="m",
    video_backend="pyav",
)
sample = dataset[0]
print(sample.keys())
print(sample["observation.images.rgb"].shape)
print(sample["observation.depth_m"].shape)
```

Use the saved camera metadata's original sequence and timestamps for timing analysis, not the nominal `frame_index / fps`
timeline. The current camera/pose IPC subscribers are latest-only: a slow recorder may skip published frames. Saving a
dataset does not imply lossless collection of every camera frame or a measured 30 Hz rate. Original sequences make
such gaps inspectable. Lossless images and float32 depth also consume more space than compressed videos.

This worker provides collection only. An automated inference replay/comparison UI is not added by this feature.
