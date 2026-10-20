"""Display-only surface simplification; preserve full inference/IPC geometry."""
import numpy as np

from ..ipc.sam3d import MeshGeometry


def preview_mesh(mesh, max_faces=20000):
    if len(mesh.faces) <= max_faces:
        return mesh
    import fast_simplification

    vertices, faces = fast_simplification.simplify(mesh.vertices, mesh.faces, target_count=max_faces)
    colors = None
    if mesh.colors is not None:
        average = mesh.colors.mean(axis=0).astype(np.uint8)
        colors = np.broadcast_to(average, (len(vertices), 3)).copy()
    return MeshGeometry(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32), colors)


def load_pose_mesh(path, scale_to_m, *, max_faces=1000):
    """Load the same scaled local geometry as FoundationPose, then simplify for display."""
    import trimesh

    asset = trimesh.load(path, force="mesh")
    asset.apply_scale(scale_to_m)
    vertex_colors = getattr(asset.visual, "vertex_colors", None)
    colors = None if vertex_colors is None else np.asarray(vertex_colors, dtype=np.uint8)[:, :3].copy()
    mesh = MeshGeometry(np.asarray(asset.vertices, dtype=np.float32).copy(),
                        np.asarray(asset.faces, dtype=np.uint32).copy(), colors)
    return preview_mesh(mesh, max_faces=max_faces)
