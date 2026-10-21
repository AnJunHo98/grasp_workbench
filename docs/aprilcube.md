# AprilCube pose and mesh overlay

[Back to the README](../README.md)

AprilCube is an alternative pose backend. An independent pose worker reads camera
RGB/intrinsics and estimates the two cube poses. A separate overlay worker reads
those results and renders the selected meshes and axes. It requires neither SAM3
nor FoundationPose, model weights, or a GPU.
The existing camera worker still publishes RGB-D; AprilCube uses RGB for pose.

## Install

From the repository checkout and your chosen Python environment:

```bash
git submodule update --init --recursive
python -m pip install -e '.[ui,aprilcube]'
```

The `aprilcube` extra installs headless OpenCV with ArUco support and the mesh
preview dependencies. The SDK is loaded from the pinned source submodule; its
unrelated optional visualization tools are not required. Camera SDK setup is
unchanged; see [camera installation](setup.md#camera).

## Select the worker in YAML

Add this to the repository-root `config.local.yaml`:

```yaml
aprilcube:
  enabled: true
  python: null
  source_dir: null
  config_a: examples/aprilcube/cube_a.json
  config_b: examples/aprilcube/cube_b.json
  enable_filter: true
  fast: true
  log_timing: false
  overlay_alpha: 0.65
```

Run as usual:

```bash
grasp-workbench run
```

`python: null` uses the current Python environment. `source_dir: null` uses
`third_party/aprilcube`. Paths are resolved relative to the YAML file defining
them. The shared default leaves `aprilcube.enabled: false`.

**When AprilCube is enabled, it takes priority over all six normal stages:**
VLM, SAM3, SAM3D, FoundationPose, GraspGenX, and grasp selection are not started,
even if their YAML or CLI flags are true. Their UI controls are not enabled.
This leaves four processes: camera, AprilCube pose, AprilCube overlay, and UI.
The YAML files are not
rewritten; setting AprilCube back to false restores the configured normal mode.
Missing SAM3 checkpoints or SAM3D model paths are not required in AprilCube mode.

`enable_filter` selects the upstream SDK's own filter, with the previous UI's
default of true. Upstream tracking, filtering, and prediction remain SDK
behavior, not a claim that every returned pose is a fresh tag measurement.
No extra workbench history, cooldown, retry, or timing gates are added.

## Compare detection speed and latency

`fast` selects the official SDK's fast detector parameters. Its default is true.
The SDK uses fewer adaptive
threshold passes and disables subpixel corner refinement. This may reduce
processing time, but can change small or oblique marker detection and pose
stability. Set false to compare the standard detector settings; it does not change
the filter. The run CLI also accepts `--aprilcube-fast` and
`--aprilcube-no-fast` to explicitly override YAML in either direction.

The pose worker shares one marker decoding pass for A/B when their dictionaries
match, as in the bundled definitions. Different dictionaries require a pass for
each dictionary. The upstream source remains unmodified: a workbench adapter
reuses each frame's primary and existing fallback marker-pass results through
the detectors' marker interface. Per-detector grayscale/CLAHE preparation and
SDK debug drawing may still run twice, and A/B pose estimation and filter states
remain independent.

Overlay rendering is no longer on the pose worker's serial path:

```text
camera RGB-D -> shared markers -> A/B pose -> /foundationpose/results
                                             |-> UI 3D plot
                                             `-> overlay worker -> /aprilcube/overlay -> UI RGB
```

The overlay worker receives the latest common pose result and AprilCube controls.
It matches each slot's session and AprilCube pose source before using that
control's selected mesh and scale;
it does not apply the cube-to-object transform a second time. Slow rendering
therefore does not block the pose worker's next frame. This separation does not
guarantee a particular output FPS or remove the age difference between the live
RGB and the rendered layer.

For a controlled comparison, add this to `config.local.yaml`:

```yaml
aprilcube:
  enabled: true
  fast: true
  log_timing: true
```

Restart the workers with the usual command to load the new settings:

```bash
grasp-workbench run
```

Compare `fast: false` and `fast: true` with the same camera settings, scene, mesh
selections, and cube motions. Check detection loss and pose jitter as well as
time. A synthetic image benchmark is not a real-camera latency measurement.

`log_timing` defaults to false. When enabled, each worker writes its own JSON
`timing` event for each processed frame, with the source `sequence`. Use `worker`
to distinguish the pose and overlay measurements rather than summing both
workers' totals. Pose logs also include the `fast` setting.

Pose worker `timings_ms`:

| Field | Measurement |
|---|---|
| `receive_ms` | Receive/control portion before pose prediction |
| `predict_ms` | Complete predictor call, including preparation and A/B processing |
| `prepare_ms` | Predictor input and detector preparation |
| `A_ms`, `B_ms` | Each cube's SDK processing call |
| `detect_ms` | Actual shared primary and existing fallback marker decoding time, already included in `A_ms`/`B_ms` |
| `compose_ms` | Cube-to-object transforms and pose result construction |
| `pose_publish_ms` | Common pose result publication |
| `total_ms` | Pose worker processing through pose publication, excluding JSON logging |
| `source_age_at_pose_ms` | Source-frame age after pose publication |

Overlay worker `timings_ms`:

| Field | Measurement |
|---|---|
| `receive_ms` | Receive/control portion before rendering |
| `render_ms` | Display RGBA mesh/axis rendering |
| `overlay_publish_ms` | Display overlay publication |
| `total_ms` | Overlay worker processing through overlay publication, excluding JSON logging |
| `source_age_at_overlay_ms` | Source-frame age after overlay publication |

`predict_ms` already contains `prepare_ms`, `A_ms`, and `B_ms`; `detect_ms` is
already included in `A_ms`/`B_ms`. Do not sum every field as if they were
independent stages. Initial detector/mesh setup may affect
the first processed frames. Per-frame JSON logging adds overhead, so enable it
only for comparisons and set `log_timing: false` for normal use.

Source ages use the camera frame's host `received_monotonic_ns` timestamp and are
null when it is missing. They are not ZED device-clock measurements, camera
exposure-to-result latency, or UI display latency. The raw RGB-D, existing IPC
payloads, filter, and UI update behavior are unchanged by these options.

### Shared decoding comparison (predictor only)

A same-process comparison rotated three predictor configurations over the same
captured 1280x720 physical-camera image, with SDK filtering enabled. Each case
used five warm-up frames and 20 measured frames. The independent-detector cases
recreated the previous decoding path with the official SDK API; they did not
import the previous application.

| Predictor configuration | Median processing time |
|---|---|
| Independent A/B decoding, `fast: false` | 118.0 ms |
| Independent A/B decoding, `fast: true` | 24.0 ms |
| Shared A/B decoding, `fast: true` | 15.4 ms |

These are predictor-only static-image timings, not native IPC throughput, mesh
rendering time, live camera/UI latency, or moving-cube accuracy. Separately, the
two-worker native IPC test holds the renderer until a second pose arrives to
verify that rendering does not block pose publication. A captured-image replay
through both workers and the actual offscreen UI also checks mesh/axis overlay,
the two plot meshes, and preservation of the raw RGB-D.

### Historical captured-frame replay baseline (before shared detection and worker split)

A pre-split controlled check used the actual SDK and one combined pose/overlay
worker, with separate A/B marker detection and a repeated
1280x720 image captured from a physical camera, published over native iceoryx2 at
approximately 30 Hz. No camera hardware or UI ran during this replay. Mesh
selections were empty, and the SDK filter and timing logs were enabled. The trial
order was standard, fast, fast, standard, with 20 measured frames per trial after
three warm-up frames.

| Per-trial median | `fast: false` | `fast: true` |
|---|---|---|
| `predict_ms` | 117.6-118.1 ms | 25.0-25.5 ms |
| `total_ms` | 121.8-122.1 ms | 28.6-29.1 ms |

These are historical static captured-frame replay results, not measurements of
the current shared-detection, split-worker implementation or live camera/UI latency.
They do not establish moving-cube detection reliability, pose jitter, or accuracy
against ground truth; check those separately on your device before selecting fast
mode.

## Cube definitions and mesh placement

The bundled [example definitions](../examples/aprilcube/README.md) use two
100 mm cubes with 40 mm markers: A has IDs 48-71, B has IDs 0-23, dictionary
`4x4_100`. They must match your printed cubes. Choose other definition files in
YAML when needed; these example IDs do not identify robot arms or task roles.

Open the **AprilCube** tab in the left worker panel. Camera and AprilCube are
active in this mode; the other worker tabs show OFF, and the global VLM input
is disabled. Selecting a tab never starts/stops workers or requests inference;
worker selection remains a YAML setting. RGB and the 3D plot stay visible to
the right, regardless of the selected tab.

Select a library mesh or browse a GLB/PLY/OBJ for each A/B slot.
Set its metres-per-mesh-unit scale and cube-to-mesh XYZ translation in metres
and roll/pitch/yaw in degrees, then press **적용 (Apply)** for that slot.
With an empty selection and identity
placement, the cube center axes are displayed without a mesh. When both mesh
selections are empty, RGB axes are drawn directly from the same latest
`PoseResult` used by the 3D plot, without waiting for or compositing RGBA. The
image is scaled once after drawing the axes. The 3D plot and pose backend are
unchanged. If either mesh is selected, the existing worker-rendered RGBA path
continues to display the meshes and axes. An explicit
placement still transforms the reported frame even if no mesh is selected.

Snapshot capture/save/load, semantic Session save/load, and request export are
not part of the main UI. Their existing helper APIs and saved files are kept;
none is needed for continuous AprilCube pose estimation or mesh overlay.

```text
camera_from_object = camera_from_cube @ cube_from_object
rotation = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

Mesh scale applies once to vertices, not to pose translation. The original mesh
origin/axes are preserved; there is no automatic recentering. Identity placement
puts the mesh origin at the cube center. This is a virtual mesh overlay, not
independent recovery of the selected object's physical pose. A fixed cube-to-object
transform must be calibrated if the cube is attached to a real object.

The overlay is a shaded, translucent triangle preview, not a texture/UV renderer
or a scene-depth occlusion solver. Missing cube poses have no visible mesh layer.
The right plot uses the same object-frame pose and selected mesh scale.

## iceoryx2 outputs

| Service suffix | Payload |
|---|---|
| `/foundationpose/results` | Existing `PoseResult`: original RGB-D source frame and A/B object poses |
| `/aprilcube/control` | A/B mesh selections, scale, session IDs, and cube-to-object transforms |
| `/aprilcube/overlay` | Display-only RGBA layer and original frame metadata |

These suffixes are appended to `ipc.service`. There is no manually managed SHM
or extra queue/pipe transport. AprilCube and FoundationPose are alternative
publishers, never launched together by `run`. `ObjectPose.pose_source` identifies
`aprilcube` versus the backward-compatible `foundationpose` default. The same
pose subscriber can read either backend without assuming the estimator.

The pose worker only publishes the common pose result. The independent overlay
worker subscribes to that result and `/aprilcube/control`, then publishes the
existing RGBA payload. These service names and payload contracts are unchanged.

Raw camera RGB/depth and the pose result's source frame are not overwritten with
rendered pixels. With a mesh selected, the UI composites the latest display layer
over the latest camera RGB once per poll. With no meshes, it draws the latest
pose axes directly on an owned display image instead. Neither path replaces the
live view with an older detector frame or claims synchronized measurements.
ZED RGB is already rectified, so its distortion coefficients are not applied a
second time; nonrectified axes retain the existing distortion projection.

The feature does not start grasp generation/selection, robot control, data
collection, or learning. Synthetic/offline checks do not establish real-camera
accuracy; test your cube dimensions, calibration, and placement on your device.
