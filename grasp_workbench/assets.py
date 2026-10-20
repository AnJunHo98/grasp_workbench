"""Persistent mesh identities, separate from per-frame reconstruction caches."""
import json
import math
from pathlib import Path
import shutil


def _entry(asset_id, name, prompt, mesh_path, scale):
    if (not isinstance(asset_id, str) or not asset_id or asset_id in {".", ".."}
            or Path(asset_id).name != asset_id):
        raise ValueError("Asset ID must be a single nonempty path component")
    if not isinstance(name, str) or not isinstance(prompt, str):
        raise ValueError("Asset name and prompt must be strings")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("Asset mesh_scale_to_m must be a finite number")
    return {"asset_id": asset_id, "name": name, "prompt": prompt,
            "mesh_path": str(Path(mesh_path).expanduser().resolve()), "mesh_scale_to_m": float(scale)}


class AssetStore:
    def __init__(self, directory="runs/assets", *, catalog_path=None):
        self.directory = Path(directory).expanduser().resolve()
        catalog_path = (Path(catalog_path).expanduser().resolve() if catalog_path is not None else
                        Path(__file__).resolve().parents[1] / "assets" / "meshes" / "catalog.json")
        self._assets = {}
        if catalog_path.is_file():
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            for item in catalog.get("objects", []):
                entry = _entry(item["asset_id"], item["display_name"], item["prompt"],
                               catalog_path.parent / item["preferred_mesh"], item["mesh_scale_to_m"])
                self._assets[entry["asset_id"]] = entry
        for path in sorted(self.directory.glob("*/manifest.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema_version") != 1 or record.get("mesh_file") not in {"object.ply", "object.glb"}:
                raise ValueError(f"Unsupported asset manifest: {path}")
            entry = _entry(record["asset_id"], record["name"], record["prompt"],
                           path.parent / record["mesh_file"], record["mesh_scale_to_m"])
            self._assets[entry["asset_id"]] = entry

    def list_assets(self):
        return tuple(entry.copy() for entry in self._assets.values())

    def get(self, asset_id):
        entry = self._assets.get(asset_id)
        return entry.copy() if entry is not None else None

    def public_catalog(self):
        return tuple({key: entry[key] for key in ("asset_id", "name", "prompt")}
                     for entry in self._assets.values())

    def save(self, result, name, prompt):
        if result.status != "done" or not isinstance(result.mesh_path, str) or not result.mesh_path:
            raise ValueError("Saving an asset requires a completed mesh file")
        if result.length_unit is not None and not isinstance(result.length_unit, str):
            raise ValueError("Asset length_unit must be a string or None")
        source = Path(result.mesh_path).expanduser().resolve()
        # Validate the identity before using it to construct a destination directory.
        entry = _entry(result.asset_id, name, prompt, source, result.mesh_scale_to_m)
        destination = self.directory / result.asset_id / ("object.glb" if source.suffix.lower() == ".glb" else "object.ply")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            shutil.copyfile(source, destination)
        manifest = {"schema_version": 1, "asset_id": entry["asset_id"], "name": entry["name"],
                    "prompt": entry["prompt"], "mesh_file": destination.name,
                    "mesh_scale_to_m": entry["mesh_scale_to_m"], "length_unit": result.length_unit}
        destination.with_name("manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        entry["mesh_path"] = str(destination)
        self._assets[entry["asset_id"]] = entry
        return entry.copy()
