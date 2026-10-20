"""Persistent mesh identity/catalog storage without camera state or model imports."""
import json

import numpy as np
import pytest

from grasp_workbench.assets import AssetStore
from grasp_workbench.ipc.sam3d import MeshResult


@pytest.fixture
def catalog(tmp_path):
    directory = tmp_path / "shared"
    directory.mkdir()
    mesh = directory / "original.glb"
    mesh.write_bytes(b"original mesh bytes")
    path = directory / "catalog.json"
    path.write_text(json.dumps({"schema_version": 1, "objects": [{
        "asset_id": "existing-id", "display_name": "참치캔", "prompt": "yellow tuna can",
        "preferred_mesh": "original.glb", "mesh_scale_to_m": 0.001,
        "length_unit": "millimetre",
    }], "table_leg": [{"display_name": "table", "path": "table.obj"}]}), encoding="utf-8")
    return path


@pytest.fixture
def result(tmp_path):
    path = tmp_path / "cache" / "generated.ply"
    path.parent.mkdir()
    path.write_bytes(b"ply\nformat ascii 1.0\ncomment source bytes\n")
    return MeshResult("request-id", "new-id", "A", "revision", "done", mesh_path=str(path),
                      mesh_scale_to_m=0.25, length_unit="metre", camera_from_object=np.eye(4))


def test_static_objects_map_names_and_paths_without_importing_scene_context(tmp_path, catalog):
    store = AssetStore(tmp_path / "runtime", catalog_path=catalog)
    asset, = store.list_assets()
    assert asset == {
        "asset_id": "existing-id", "name": "참치캔", "prompt": "yellow tuna can",
        "mesh_path": str(catalog.parent / "original.glb"), "mesh_scale_to_m": 0.001,
    }
    assert store.get("existing-id") == asset
    assert store.get("참치캔") is None and store.get("yellow tuna can") is None
    assert not (tmp_path / "runtime").exists()


def test_public_catalog_contains_only_identity_name_prompt(tmp_path, catalog):
    store = AssetStore(tmp_path / "runtime", catalog_path=catalog)
    assert store.public_catalog() == ({
        "asset_id": "existing-id", "name": "참치캔", "prompt": "yellow tuna can",
    },)


def test_save_copies_mesh_and_writes_only_portable_asset_metadata(tmp_path, catalog, result):
    store = AssetStore(tmp_path / "runtime", catalog_path=catalog)
    source = result.mesh_path
    before = (tmp_path / "cache" / "generated.ply").read_bytes()
    saved = store.save(result, "새 물체", "orange ball")
    destination = tmp_path / "runtime" / result.asset_id / "object.ply"
    assert saved == {
        "asset_id": "new-id", "name": "새 물체", "prompt": "orange ball",
        "mesh_path": str(destination), "mesh_scale_to_m": 0.25,
    }
    assert destination.read_bytes() == before
    assert result.mesh_path == source
    assert (tmp_path / "cache" / "generated.ply").read_bytes() == before
    manifest = json.loads(destination.with_name("manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1, "asset_id": "new-id", "name": "새 물체", "prompt": "orange ball",
        "mesh_file": "object.ply", "mesh_scale_to_m": 0.25, "length_unit": "metre",
    }
    assert set(destination.parent.iterdir()) == {destination, destination.with_name("manifest.json")}
    assert store.get("new-id") == saved


def test_reopen_finds_saved_identity_without_mesh_generation_or_label_matching(tmp_path, catalog, result):
    directory = tmp_path / "runtime"
    first = AssetStore(directory, catalog_path=catalog)
    saved = first.save(result, "공", "orange ball")
    (tmp_path / "cache" / "generated.ply").unlink()
    second = AssetStore(directory, catalog_path=catalog)
    assert second.get("new-id") == saved
    assert (directory / "new-id" / "object.ply").read_bytes().startswith(b"ply\n")
    assert len(second.list_assets()) == 2
    assert second.get("공") is None and second.get("orange ball") is None
    assert second.public_catalog()[-1] == {"asset_id": "new-id", "name": "공", "prompt": "orange ball"}


def test_runtime_same_id_overrides_static_and_resave_updates_that_identity(tmp_path, catalog, result):
    result.asset_id = "existing-id"
    store = AssetStore(tmp_path / "runtime", catalog_path=catalog)
    first = store.save(result, "새 참치캔", "blue tuna can")
    assert len(store.list_assets()) == 1 and store.get("existing-id") == first
    (tmp_path / "cache" / "generated.ply").write_bytes(b"newer mesh bytes")
    second = store.save(result, "새 이름", "tuna can")
    assert second["name"] == "새 이름"
    assert (tmp_path / "runtime" / "existing-id" / "object.ply").read_bytes() == b"newer mesh bytes"
    reopened = AssetStore(tmp_path / "runtime", catalog_path=catalog)
    assert reopened.list_assets() == (second,)
    assert (catalog.parent / "original.glb").read_bytes() == b"original mesh bytes"


def test_empty_catalog_and_runtime_are_allowed(tmp_path):
    store = AssetStore(tmp_path / "runtime", catalog_path=tmp_path / "missing.json")
    assert store.list_assets() == store.public_catalog() == ()
    assert store.get("unknown") is None


def test_catalog_results_do_not_expose_mutable_store_state(tmp_path, catalog):
    store = AssetStore(tmp_path / "runtime", catalog_path=catalog)
    store.get("existing-id")["name"] = "changed"
    store.list_assets()[0]["prompt"] = "changed"
    store.public_catalog()[0]["asset_id"] = "changed"
    assert store.get("existing-id")["name"] == "참치캔"
    assert store.public_catalog()[0]["asset_id"] == "existing-id"


@pytest.mark.parametrize("field,value", [("name", 12), ("prompt", None), ("mesh_scale_to_m", "large")])
def test_saved_manifest_requires_basic_field_types(tmp_path, catalog, result, field, value):
    directory = tmp_path / "runtime"
    AssetStore(directory, catalog_path=catalog).save(result, "ball", "orange ball")
    path = directory / "new-id" / "manifest.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        AssetStore(directory, catalog_path=catalog)


@pytest.mark.parametrize("status,path", [("running", "not-yet.ply"), ("error", None), ("done", None)])
def test_save_requires_a_completed_mesh_file(tmp_path, catalog, result, status, path):
    result.status, result.mesh_path = status, path
    directory = tmp_path / "runtime"
    with pytest.raises(ValueError):
        AssetStore(directory, catalog_path=catalog).save(result, "ball", "orange ball")
    assert not directory.exists()
