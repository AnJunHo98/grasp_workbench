"""Local checkpoint resolution is separate from public VLM policy metadata."""
import json

import pytest

from grasp_workbench.policy_catalog import load_policy_catalog, public_policy_catalog, resolve_policy


def policy_entry():
    return {"policy_id": "place-v1", "display_name": "Place on tray", "task_type": "place_on_surface",
            "description": "Move source onto target", "training_source": {"asset_id": "bottle", "name": "bottle"},
            "training_target": {"asset_id": "tray", "name": "tray"}, "pose_convention": "target_from_source",
            "transfer_note": "New source transfer unverified", "checkpoint": "weights/policy.pt"}


@pytest.fixture
def policy_path(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([policy_entry()]), encoding="utf-8")
    return path


def test_optional_catalog_resolves_only_local_checkpoint_relative_to_json(policy_path):
    assert load_policy_catalog() == []
    entries = load_policy_catalog(policy_path)
    assert entries[0]["checkpoint"] == str(policy_path.parent / "weights/policy.pt")
    assert not (policy_path.parent / "weights/policy.pt").exists(), "Catalog lookup must not require/load a model"
    assert resolve_policy(entries, "place-v1", "place_on_surface") == entries[0]
    assert resolve_policy(entries, None, "anything") is None


def test_public_policy_whitelist_removes_checkpoint_and_nested_private_sidecars(policy_path):
    entry = policy_entry()
    entry["private_path"], entry["api_key_file"] = "/private/path", "/never/read"
    entry["training_source"]["checkpoint"] = "/private/source.pt"
    policy_path.write_text(json.dumps([entry]), encoding="utf-8")
    public, = public_policy_catalog(load_policy_catalog(policy_path))
    assert set(public) == {"policy_id", "display_name", "task_type", "description", "training_source",
                           "training_target", "pose_convention", "transfer_note"}
    assert set(public["training_source"]) == {"asset_id", "name"}
    assert "/private" not in json.dumps(public) and "api_key" not in json.dumps(public)


def test_policy_selection_requires_matching_stable_id_and_task_type(policy_path):
    entries = load_policy_catalog(policy_path)
    with pytest.raises(ValueError, match="does not match"):
        resolve_policy(entries, "place-v1", "pour_contents")
    with pytest.raises(ValueError, match="Unknown policy"):
        resolve_policy(entries, "other-id", "place_on_surface")


def test_duplicate_ids_follow_original_catalog_contract(policy_path):
    policy_path.write_text(json.dumps([policy_entry(), policy_entry()]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_policy_catalog(policy_path)
