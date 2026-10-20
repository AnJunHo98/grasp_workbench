"""Selected RGB-D/mask requests and object-frame meshes over raw iceoryx2."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from .camera import CameraFrame, decode_frame, encode_frame
from .sam3 import _BytePublisher, _ByteSubscriber

_LENGTH = struct.Struct("<I")


@dataclass
class MeshGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None = None


@dataclass
class MeshRequest:
    request_id: str
    asset_id: str
    object_id: str
    revision: str
    prompt: str
    candidate_index: int
    frame: CameraFrame
    mask: np.ndarray
    depth_source: str = "depth"
    texture_baking: bool = False


@dataclass
class MeshResult:
    request_id: str
    asset_id: str
    object_id: str
    revision: str
    status: str
    mesh: MeshGeometry | None = None
    mesh_path: str | None = None
    error: str | None = None
    mesh_scale_to_m: float = 1.0
    camera_from_object: np.ndarray | None = None
    length_unit: str | None = None


def services(base):
    return base + "/sam3d/requests", base + "/sam3d/results"


def _identity(value):
    result = {name: getattr(value, name) for name in ("request_id", "asset_id", "object_id", "revision")}
    if not all(isinstance(item, str) for item in result.values()):
        raise ValueError("Mesh request/asset/object/revision IDs must be strings")
    return result


def _header(kind, **fields):
    record = {"schema_version": 1, "kind": kind, **fields}
    data = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return _LENGTH.pack(len(data)) + data


def _read(payload, kind):
    view = memoryview(payload).cast("B")
    if len(view) < 4:
        raise ValueError("Truncated mesh header")
    offset = 4 + _LENGTH.unpack_from(view)[0]
    if offset > len(view):
        raise ValueError("Truncated mesh metadata")
    record = json.loads(bytes(view[4:offset]))
    if not isinstance(record, dict) or (
        type(record.get("schema_version")) is not int or record["schema_version"] != 1
        or record.get("kind") != kind
        or not all(isinstance(record.get(name), str)
                   for name in ("request_id", "asset_id", "object_id", "revision"))
    ):
        raise ValueError("Unsupported mesh schema or identity")
    return record, view, offset


def validate_mesh(mesh):
    """Reject malformed geometry; its coordinate contract belongs to the result."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (vertices.dtype.kind != "f" or vertices.dtype.itemsize != 4
            or vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValueError("Vertices must be a nonempty finite float32 Nx3 array")
    if (faces.dtype.kind != "u" or faces.dtype.itemsize != 4
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or np.any(faces >= len(vertices))):
        raise ValueError("Faces must be a nonempty uint32 Mx3 array with valid vertex indices")
    if mesh.colors is not None:
        colors = np.asarray(mesh.colors)
        if colors.dtype != np.uint8 or colors.shape != vertices.shape:
            raise ValueError("Colors must be uint8 Nx3 matching vertices")


def _pose(value):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Mesh camera_from_object must be a finite 4x4 matrix")
    return matrix.copy()


def encode_request(request: MeshRequest) -> bytes:
    identity = _identity(request)
    if not isinstance(request.prompt, str) or type(request.candidate_index) is not int or request.candidate_index < 0:
        raise ValueError("Prompt must be a string and candidate_index a nonnegative integer")
    frame = encode_frame(request.frame)
    mask = np.asarray(request.mask)
    if mask.dtype != np.bool_ or mask.shape != request.frame.rgb.shape[:2]:
        raise ValueError("Selection mask must be bool HxW matching the source frame")
    if request.depth_source not in ("depth", "monocular") or type(request.texture_baking) is not bool:
        raise ValueError("SAM3D depth_source must be depth/monocular and texture_baking a bool")
    return b"".join((_header("mesh_request", **identity, prompt=request.prompt,
                            candidate_index=request.candidate_index, frame_size=len(frame),
                            mask_encoding="bool8", depth_source=request.depth_source,
                            texture_baking=request.texture_baking), frame, mask.tobytes(order="C")))


def decode_request(payload) -> MeshRequest:
    record, view, offset = _read(payload, "mesh_request")
    size = record.get("frame_size")
    candidate = record.get("candidate_index")
    if (type(size) is not int or size <= 0 or offset + size > len(view)
            or record.get("mask_encoding") != "bool8" or not isinstance(record.get("prompt"), str)
            or type(candidate) is not int or candidate < 0
            or record.get("depth_source", "depth") not in ("depth", "monocular")
            or type(record.get("texture_baking", False)) is not bool):
        raise ValueError("Invalid mesh request frame or selection")
    frame = decode_frame(view[offset:offset + size])
    offset += size
    shape = frame.rgb.shape[:2]
    if len(view) != offset + shape[0] * shape[1]:
        raise ValueError("Selection mask byte length does not match the source frame")
    mask = np.frombuffer(view, np.bool_, shape[0] * shape[1], offset).reshape(shape).copy()
    return MeshRequest(*(record[name] for name in ("request_id", "asset_id", "object_id", "revision")),
                       record["prompt"], candidate, frame, mask,
                       record.get("depth_source", "depth"), record.get("texture_baking", False))


def encode_result(result: MeshResult) -> bytes:
    identity = _identity(result)
    if result.status not in ("running", "done", "error"):
        raise ValueError("Mesh status must be running, done or error")
    if any(value is not None and not isinstance(value, str) for value in (result.mesh_path, result.error)):
        raise ValueError("Mesh path/error must be a string or None")
    if (result.status == "done") != (result.mesh is not None):
        raise ValueError("Only a done result must contain geometry")
    pose = _pose(result.camera_from_object)
    if result.length_unit is not None and not isinstance(result.length_unit, str):
        raise ValueError("Mesh length_unit must be a string or None")
    vertices = faces = 0
    arrays = []
    colors = False
    if result.mesh is not None:
        validate_mesh(result.mesh)
        vertices, faces = len(result.mesh.vertices), len(result.mesh.faces)
        colors = result.mesh.colors is not None
        arrays = [np.asarray(result.mesh.vertices, dtype="<f4").tobytes(order="C"),
                  np.asarray(result.mesh.faces, dtype="<u4").tobytes(order="C")]
        if colors:
            arrays.append(np.asarray(result.mesh.colors).tobytes(order="C"))
    return b"".join((_header("mesh_result", **identity, status=result.status,
                            mesh_path=result.mesh_path, error=result.error,
                            mesh_scale_to_m=float(result.mesh_scale_to_m),
                            camera_from_object=None if pose is None else pose.tolist(),
                            length_unit=result.length_unit,
                            vertex_count=vertices, face_count=faces, has_colors=colors,
                            vertex_encoding="float32_le", face_encoding="uint32_le",
                            color_encoding="rgb8"), *arrays))


def decode_result(payload) -> MeshResult:
    record, view, offset = _read(payload, "mesh_result")
    vertices, faces, colors = record.get("vertex_count"), record.get("face_count"), record.get("has_colors")
    if (any(type(count) is not int or count < 0 for count in (vertices, faces))
            or type(colors) is not bool or record.get("vertex_encoding") != "float32_le"
            or record.get("face_encoding") != "uint32_le" or record.get("color_encoding") != "rgb8"
            or record.get("status") not in ("running", "done", "error")
            or any(record.get(name) is not None and not isinstance(record[name], str)
                   for name in ("mesh_path", "error", "length_unit"))):
        raise ValueError("Invalid mesh result metadata")
    done = record["status"] == "done"
    if (done and (not vertices or not faces)) or (not done and (vertices or faces or colors)):
        raise ValueError("Only a done result must contain geometry")
    if len(view) != offset + vertices * (15 if colors else 12) + faces * 12:
        raise ValueError("Mesh byte length does not match geometry counts")
    mesh = None
    if done:
        points = np.frombuffer(view, "<f4", vertices * 3, offset).reshape(vertices, 3).astype(np.float32, copy=True)
        offset += vertices * 12
        triangles = np.frombuffer(view, "<u4", faces * 3, offset).reshape(faces, 3).astype(np.uint32, copy=True)
        offset += faces * 12
        rgb = np.frombuffer(view, np.uint8, vertices * 3, offset).reshape(vertices, 3).copy() if colors else None
        mesh = MeshGeometry(points, triangles, rgb)
        validate_mesh(mesh)
    return MeshResult(*(record[name] for name in ("request_id", "asset_id", "object_id", "revision")),
                      record["status"], mesh, record.get("mesh_path"), record.get("error"),
                      float(record.get("mesh_scale_to_m", 1.0)),
                      _pose(record.get("camera_from_object")), record.get("length_unit"))


class _MeshPublisher(_BytePublisher):
    def update_connections(self):
        if self._publisher is None:
            raise RuntimeError("Mesh publisher is closed")
        return self._publisher.update_connections()


class MeshRequestPublisher(_MeshPublisher):
    def publish(self, request: MeshRequest):
        return self._publish(encode_request(request))


class MeshRequestSubscriber(_ByteSubscriber):
    def receive_latest(self) -> MeshRequest | None:
        return self._receive(decode_request)


class MeshResultPublisher(_MeshPublisher):
    def publish(self, result: MeshResult):
        return self._publish(encode_result(result))


class MeshResultSubscriber(_ByteSubscriber):
    def receive_latest(self) -> MeshResult | None:
        return self._receive(decode_result)
