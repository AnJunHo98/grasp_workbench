# AprilCube configurations

These two example definitions preserve the previous perception UI's printed
cube layouts, without camera, robot, or personal settings:

| Slot | File | Dictionary | Marker IDs | Cube edge | Marker edge |
|---|---|---|---|---|---|
| A | `cube_a.json` | `4x4_100` | 48-71 | 100 mm | 40 mm |
| B | `cube_b.json` | `4x4_100` | 0-23 | 100 mm | 40 mm |

Each face has a 2x2 marker grid. These are ArUco dictionary definitions, not
`apriltag_36h11`. Use configurations matching the actual printed cubes.
Dimensions in the JSON files are millimetres; the worker publishes pose
translation in metres. Slots A/B do not assign task roles or robot arms.

Source definitions:

- `ffw_object_pose_system/configs/aprilcube/aprilcube_3_2x2x2_40_100.json`
  -> `cube_a.json`, SHA-256
  `f3390e4425ac0dd6e9f5bd178b176e4b49d055c2210d3d6981758fa07e4d5620`
- `ffw_object_pose_system/configs/aprilcube/aprilcube_2x2x2_40_100.json`
  -> `cube_b.json`, SHA-256
  `fb9bc7c22ce6445a85b3e64a4debdf410d804d3590a6019f7baceac6412c4422`

The detector is the unmodified [AprilCube upstream](https://github.com/younghyopark/aprilcube),
pinned at `9d679b56cf15be6126c846593fd37471cf8f0ad0` in
`third_party/aprilcube`. The previous application's vendored local SDK patches
are not included. The pinned upstream declares MIT in its README, but has no
standalone LICENSE file; check its terms before redistributing.

See [AprilCube mode](../../docs/aprilcube.md) for installation and UI usage.
