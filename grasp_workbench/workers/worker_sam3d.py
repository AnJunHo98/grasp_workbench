"""On-demand SAM 3D mesh generation and exact-input asset caching."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..core.worker_base import SingleRateWorker, WorkerContext, log_record
from ..ipc.camera import encode_frame
from ..ipc.sam3d import MeshGeometry, MeshResult, services, validate_mesh
from ..models.sam3d import Sam3dPrediction
from .worker_camera import DEFAULT_SERVICE


def _fingerprint(request):
    selection = {name: getattr(request, name)
                 for name in ("object_id", "revision", "prompt", "candidate_index", "depth_source", "texture_baking")}
    digest = hashlib.sha256(json.dumps(selection, sort_keys=True, ensure_ascii=False).encode())
    digest.update(encode_frame(request.frame))
    digest.update(np.asarray(request.mask).tobytes(order="C"))
    return digest.hexdigest()


def _write_ply(path, mesh, length_unit="metre"):
    """Binary PLY in the canonical object frame, separate from its rigid pose."""
    fields = [(name, "<f4") for name in ("x", "y", "z")]
    if mesh.colors is not None:
        fields += [(name, "u1") for name in ("red", "green", "blue")]
    points = np.empty(len(mesh.vertices), dtype=fields)
    for index, name in enumerate(("x", "y", "z")):
        points[name] = mesh.vertices[:, index]
    if mesh.colors is not None:
        for index, name in enumerate(("red", "green", "blue")):
            points[name] = mesh.colors[:, index]
    triangles = np.empty(len(mesh.faces), dtype=[("count", "u1"), ("indices", "<u4", (3,))])
    triangles["count"], triangles["indices"] = 3, mesh.faces
    header = ["ply", "format binary_little_endian 1.0",
              f"comment canonical object frame; units {length_unit}; output scale applied once",
              f"element vertex {len(points)}",
              "property float x", "property float y", "property float z"]
    if mesh.colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [f"element face {len(triangles)}", "property list uchar uint vertex_indices", "end_header"]
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        stream.write(points.tobytes())
        stream.write(triangles.tobytes())


class Sam3dWorker(SingleRateWorker):
    def __init__(
        self, context, config_path, source_dir, output_dir, seed=42,
        service=DEFAULT_SERVICE, frames=0, *, subscriber_factory=None,
        publisher_factory=None, predictor_factory=None,
    ):
        super().__init__(context, hz=100)
        self.config_path, self.source_dir, self.seed = config_path, source_dir, seed
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.service, self.frames = service, frames
        self.subscriber_factory, self.publisher_factory = subscriber_factory, publisher_factory
        self.predictor_factory = predictor_factory
        self.subscriber = self.publisher = self.predictor = None
        self.processed_count, self._seen, self._last_result = 0, set(), None
        self._last_input = None

    def on_start(self):
        from ..ipc.sam3d import MeshRequestSubscriber, MeshResultPublisher

        request_service, result_service = services(self.service)
        self.subscriber = (self.subscriber_factory or MeshRequestSubscriber)(request_service)
        self.publisher = (self.publisher_factory or MeshResultPublisher)(result_service)
        log_record("ready", worker=self.ctx.name, service=self.service)

    def _paths(self, asset_id, depth_source="depth", texture_baking=False):
        identity = asset_id if depth_source == "depth" and not texture_baking else json.dumps(
            [asset_id, depth_source, texture_baking], ensure_ascii=False)
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.output_dir / (key + ".npz"), self.output_dir / (key + ".ply")

    def _load(self, path, fingerprint):
        try:
            with np.load(path, allow_pickle=False) as data:
                if data["schema_version"].item() != 2 or str(data["fingerprint"].item()) != fingerprint:
                    return None
                mesh = MeshGeometry(data["vertices"].copy(), data["faces"].copy(),
                                    data["colors"].copy() if data["has_colors"].item() else None)
                validate_mesh(mesh)
                texture = data["textured_glb"].tobytes() if "textured_glb" in data and data["textured_glb"].size else None
                return Sam3dPrediction(mesh, data["camera_from_object"].copy(), texture,
                                       str(data["length_unit"].item()))
        except (OSError, ValueError, KeyError):
            # An interrupted/corrupt cache is not a valid reconstruction.
            return None

    def _save(self, path, ply_path, fingerprint, prediction):
        mesh = prediction.mesh
        self.output_dir.mkdir(parents=True, exist_ok=True)
        glb_path = ply_path.with_suffix(".glb")
        npz_temp, ply_temp = path.with_suffix(".npz.tmp"), ply_path.with_suffix(".ply.tmp")
        glb_temp = glb_path.with_suffix(".glb.tmp")
        try:
            _write_ply(ply_temp, mesh, prediction.length_unit)
            if prediction.textured_glb is not None:
                glb_temp.write_bytes(prediction.textured_glb)
            with npz_temp.open("wb") as stream:
                np.savez(stream, schema_version=np.asarray(2), fingerprint=np.asarray(fingerprint),
                         camera_from_object=prediction.camera_from_object,
                         mesh_scale_to_m=np.asarray(1.0), length_unit=np.asarray(prediction.length_unit),
                         depth_source=np.asarray("depth" if prediction.length_unit == "metre" else "monocular"),
                         texture_baking=np.asarray(prediction.textured_glb is not None),
                         textured_glb=np.frombuffer(prediction.textured_glb or b"", dtype=np.uint8), vertices=mesh.vertices,
                         faces=mesh.faces, has_colors=np.asarray(mesh.colors is not None),
                         colors=mesh.colors if mesh.colors is not None else np.empty((0, 3), np.uint8))
            os.replace(ply_temp, ply_path)
            if prediction.textured_glb is not None:
                os.replace(glb_temp, glb_path)
            os.replace(npz_temp, path)
        finally:
            npz_temp.unlink(missing_ok=True)
            ply_temp.unlink(missing_ok=True)
            glb_temp.unlink(missing_ok=True)

    def _result(self, request, status, **fields):
        return MeshResult(request.request_id, request.asset_id, request.object_id,
                          request.revision, status, **fields)

    def step_once(self):
        self.publisher.update_connections()
        request = self.subscriber.receive_latest()
        if request is None:
            return
        if request.request_id in self._seen:
            previous = self._last_result
            try:
                identical = (request.asset_id, _fingerprint(request)) == self._last_input
            except Exception:
                identical = False
            self.publisher.publish(previous if identical and previous is not None
                                   and previous.request_id == request.request_id
                                   else self._result(request, "error", error="request_id already processed"))
            return
        self._seen.add(request.request_id)
        self._last_input = None
        self.publisher.publish(self._result(request, "running"))
        try:
            mask = np.asarray(request.mask)
            if mask.dtype != np.bool_ or mask.shape != request.frame.rgb.shape[:2] or not mask.any():
                raise ValueError("Selection mask must contain foreground and match the RGB frame")
            fingerprint = _fingerprint(request)
            self._last_input = (request.asset_id, fingerprint)
            path, ply_path = self._paths(request.asset_id, request.depth_source, request.texture_baking)
            mesh_path = ply_path.with_suffix(".glb") if request.texture_baking else ply_path
            prediction = self._load(path, fingerprint)
            cached = prediction is not None
            if prediction is None:
                if self.predictor is None:
                    if self.predictor_factory is None:
                        from ..models.sam3d import OfficialSam3dPredictor

                        self.predictor_factory = OfficialSam3dPredictor
                    self.predictor = self.predictor_factory(
                        self.config_path, source_dir=self.source_dir, seed=self.seed,
                    )
                    try:
                        self.predictor.start()
                    except Exception:
                        try:
                            self.predictor.close()
                        finally:
                            self.predictor = None
                        raise
                options = {}
                if request.depth_source != "depth":
                    options["depth_source"] = request.depth_source
                if request.texture_baking:
                    options["texture_baking"] = True
                prediction = self.predictor.predict(request.frame, mask, **options)
                prediction.length_unit = "metre" if request.depth_source == "depth" else "monocular_predicted_scale"
                validate_mesh(prediction.mesh)
            if request.texture_baking and prediction.textured_glb is None:
                raise ValueError("SAM3D texture baking returned no textured GLB")
            if self.ctx.stop_event.is_set():
                return
            if not cached or not ply_path.is_file() or not mesh_path.is_file():
                self._save(path, ply_path, fingerprint, prediction)
            result = self._result(request, "done", mesh=prediction.mesh, mesh_path=str(mesh_path),
                                  mesh_scale_to_m=1.0, length_unit=prediction.length_unit,
                                  camera_from_object=prediction.camera_from_object)
        except Exception as error:
            result = self._result(request, "error", error=f"{type(error).__name__}: {error}")
        self._last_result = result
        self.publisher.publish(result)
        self.processed_count += 1
        log_record("sam3d-result", request_id=result.request_id, asset_id=result.asset_id,
                   object_id=result.object_id, revision=result.revision, status=result.status,
                   mesh_path=result.mesh_path, error=result.error)
        if self.frames and self.processed_count >= self.frames:
            self.ctx.stop()

    def on_stop(self):
        try:
            if self.predictor is not None:
                self.predictor.close()
        finally:
            try:
                if self.publisher is not None:
                    self.publisher.close()
            finally:
                if self.subscriber is not None:
                    self.subscriber.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--frames", type=int, default=0, help="Terminal jobs; 0 waits continuously")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    try:
        Sam3dWorker(WorkerContext("sam3d"), args.config_path, args.source_dir, args.output_dir,
                    args.seed, args.service, args.frames).run()
    except Exception as error:
        log_record("error", worker="sam3d", error=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
