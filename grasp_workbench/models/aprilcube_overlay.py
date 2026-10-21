"""Cached CPU mesh preview layers; never mutate camera RGB or inference meshes."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from .aprilcube import camera_parameters


@lru_cache(maxsize=8)
def _preview_geometry(path, modified_ns, size):
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if len(faces) > 2000:
        import fast_simplification

        vertices, faces = fast_simplification.simplify(vertices, faces, target_count=2000)
    return np.asarray(vertices, dtype=float).copy(), np.asarray(faces, dtype=int).copy()


def _project(vertices, matrix, distortion):
    import cv2

    pixels, _ = cv2.projectPoints(vertices, np.zeros(3), np.zeros(3), matrix, distortion)
    return pixels.reshape(-1, 2)


class MeshOverlayRenderer:
    def render(self, frame, targets, poses, alpha=.65):
        """Straight-alpha RGBA; painter ordering is not scene-depth occlusion."""
        import cv2

        height, width = frame.rgb.shape[:2]
        layer = np.zeros((height, width, 4), dtype=np.uint8)
        matrix, distortion = camera_parameters(frame)
        triangles, axes = [], []
        for target in targets:
            pose = poses.get(target.object_id)
            if pose is None:
                continue
            axes.append((np.vstack((np.zeros(3), np.eye(3) * .05)) @ pose[:3, :3].T
                         + pose[:3, 3]))
            if not target.mesh_path:
                continue
            path = Path(target.mesh_path).expanduser().resolve()
            stat = path.stat()
            local, faces = _preview_geometry(str(path), stat.st_mtime_ns, stat.st_size)
            vertices = local * target.mesh_scale_to_m @ pose[:3, :3].T + pose[:3, 3]
            faces = faces[np.all(vertices[faces, 2] > 1e-6, axis=1)]
            if not len(faces):
                continue
            projected = _project(vertices, matrix, distortion)
            points = vertices[faces]
            normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            brightness = .65 + .35 * np.divide(
                np.abs(normals[:, 2]), lengths, out=np.zeros_like(lengths), where=lengths > 0,
            )
            color = np.asarray((60, 180, 235) if target.object_id == "A" else (235, 165, 65))
            for face, depth, shade in zip(faces, points[:, :, 2].mean(axis=1), brightness):
                triangles.append((depth, projected[face], color * shade))
        opacity = int(round(alpha * 255))
        for _, pixels, color in sorted(triangles, key=lambda item: item[0], reverse=True):
            if (not np.isfinite(pixels).all() or np.any(pixels.max(axis=0) < 0)
                    or pixels[:, 0].min() >= width or pixels[:, 1].min() >= height):
                continue
            pixels = np.rint(np.clip(pixels, -max(width, height) * 4,
                                     max(width, height) * 4)).astype(np.int32)
            cv2.fillConvexPoly(layer, pixels, (*map(int, np.clip(color, 0, 255)), opacity))
        for points in axes:
            if not np.all(points[:, 2] > 1e-6):
                continue
            pixels = _project(points, matrix, distortion)
            if not np.isfinite(pixels).all():
                continue
            pixels = np.rint(np.clip(pixels, -max(width, height) * 4,
                                     max(width, height) * 4)).astype(int)
            for endpoint, color in zip(pixels[1:], ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                cv2.arrowedLine(layer, tuple(pixels[0]), tuple(endpoint), (*color, opacity),
                                2, tipLength=.15)
        return layer
