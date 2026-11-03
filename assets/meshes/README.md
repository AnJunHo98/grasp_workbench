# Bundled Mesh Library

Copied from the author's object library. The source files and all mesh bytes
are unchanged.

- `ffw_objects/<asset_name>/object.glb`: original meshes for 21 object assets.
- `ffw_objects/<asset_name>/object.optimized.glb`: existing optimized meshes.
- `table_leg/`: `Table001.obj`, `Leg001.obj`, and `Leg001_colored.obj`.
- [catalog.json](catalog.json): English names and prompts, relative file paths,
  source-declared units, file sizes, and SHA-256 hashes.

## Names and descriptions

Folder names use the English `display_name` in lowercase, with spaces replaced
by underscores. Names and prompts describe the object type and appearance, not
its position, orientation, or cropping in the original camera image.

The two black-bottle assets have the same available appearance description.
`Black Bottle 1` and `Black Bottle 2` distinguish library entries only; their
numbers do not encode scene position or assert physical object identity.

Examples:

- **Tape Roll**: `ffw_objects/tape_roll/object.optimized.glb`
- **Dongwon Tuna Can**: `ffw_objects/dongwon_tuna_can/object.optimized.glb`
- **Cooking Pack Box**: `ffw_objects/cooking_pack_box/object.optimized.glb`

## Geometry and provenance

The library contains **45 meshes / 303,867,548 bytes (about 289.8 MiB)**.
GLB textures and buffers are embedded; OBJ files have no external MTL or texture
references. All mesh SHA-256 hashes match the source files.

Coordinates, colors, scale, `asset_id`, and `object_frame` are preserved. The
catalog retains the source declarations of `metre` and `mesh_scale_to_m: 1.0`;
these do not establish measured shape accuracy or a current camera-frame pose.
No snapshots, past poses, robot settings, grasp caches, model weights, or logs
are included.

## Usage

This library is separate from the SAM3D inference cache. The A/B mesh selectors
in the FoundationPose UI list the 21 object assets and 3 table/leg meshes.
Selecting a name fills in the mesh path and scale; manual file browsing remains
available. Assets resolve within this repository without the original FFW
application or personal filesystem paths.

Restart a running UI or reselect an asset after renaming folders. Previously
saved local records with old file paths are not rewritten.
