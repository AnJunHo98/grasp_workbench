"""Display canonical, metric grasp frames without grasp inference or robot logic."""
import hashlib
import json
from pathlib import Path

import numpy as np


def mesh_binding(mesh_path, mesh_scale_to_m):
    return str(Path(mesh_path).expanduser().resolve()), float(mesh_scale_to_m)


def asset_identity(mesh_path, mesh_scale_to_m, catalog=()):
    binding = mesh_binding(mesh_path, mesh_scale_to_m)
    for asset in catalog:
        if binding == mesh_binding(asset["mesh_path"], asset["mesh_scale_to_m"]):
            return asset["asset_id"]
    payload = json.dumps(binding, separators=(",", ":")).encode("utf-8")
    return "mesh-" + hashlib.sha256(payload).hexdigest()


def transform_points(points, transform):
    """Canonical gripper geometry is already in metres; do not rotate or recenter it again."""
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def grasp_axis_points(transform, length_m=0.03):
    return transform_points(np.vstack((np.zeros(3), np.eye(3) * length_m)), transform)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gripper_preview(metadata):
    """Use recorded canonical visual geometry; missing descriptors retain axes-only display."""
    from .mesh_preview import load_pose_mesh

    metadata = metadata or {}
    config_path = metadata.get("config_path")
    files = metadata.get("files", [])
    if not config_path or not any(item.get("relative_path") == "vis_mesh.obj" for item in files):
        return None, "descriptor unavailable · axes only"
    try:
        directory = Path(config_path).parent
        if _sha256(config_path) != metadata.get("config_sha256"):
            return None, "descriptor changed · axes only"
        for item in files:
            if _sha256(directory / item["relative_path"]) != item["sha256"]:
                return None, "descriptor geometry changed · axes only"
        mesh = load_pose_mesh(directory / "vis_mesh.obj", 1.0, max_faces=1000)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            return None, "visual mesh unavailable · axes only"
        return mesh, "recorded canonical visual mesh"
    except (OSError, ValueError, KeyError, TypeError):
        return None, "descriptor files unavailable · axes only"


def remove_artists(artists):
    for value in artists.values():
        for artist in value if isinstance(value, list) else [value]:
            artist.remove()


def update_grasp_artists(axes, transform, mesh, artists):
    """Update one selected candidate's existing plot artists and return its bounds points."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points = grasp_axis_points(transform)
    if "axes" not in artists:
        artists["axes"] = [axes.plot([], [], [], color=color, linewidth=2.5, zorder=12)[0]
                           for color in ("r", "g", "b")]
    for endpoint, artist in zip(points[1:], artists["axes"]):
        artist.set_data_3d(np.stack((points[0], endpoint)).T)
    if mesh is not None:
        vertices = transform_points(mesh.vertices, transform)
        if "mesh" not in artists:
            artists["mesh"] = Poly3DCollection(vertices[mesh.faces], facecolors="#67b8ff", alpha=0.8,
                                                linewidths=0, shade=True)
            axes.add_collection3d(artists["mesh"])
        else:
            artists["mesh"].set_verts(vertices[mesh.faces])
        points = np.vstack((points, vertices))
    return points
