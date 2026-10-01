"""Multi-resolution training blueprint helpers (library/preprocess/multires.py).

Pure, headless: weight parsing / repeat normalization / blueprint shape. The
subprocess orchestration (scripts/tasks/multires.py) and the GUI section sit
on top of these.
"""

from pathlib import Path

import pytest
import toml

from library.preprocess.multires import (
    clear_variant_blueprint,
    multires_dataset_config,
    parse_tiers,
    parse_weights,
    share_summary,
    tier_dir,
    weights_to_repeats,
    write_variant_blueprint,
)


def test_parse_tiers_normalizes_and_sorts():
    assert parse_tiers("1024, 896") == [896, 1024]
    assert parse_tiers([1536, 512]) == [512, 1536]
    assert parse_tiers("1024，896，1024") == [896, 1024]  # fullwidth commas, dedup
    assert parse_tiers(None) == []
    with pytest.raises(ValueError, match="3-4 digit"):
        parse_tiers("12")


def test_parse_weights_defaults_and_drops_stale_tiers():
    assert parse_weights(None, [896, 1024]) == {896: 1.0, 1024: 1.0}
    assert parse_weights("1024:2", [896, 1024]) == {896: 1.0, 1024: 2.0}
    # 512 no longer selected → its stale entry is dropped
    assert parse_weights("1024:2, 512:5", [896, 1024]) == {896: 1.0, 1024: 2.0}
    assert parse_weights("2", [896, 1024]) == {896: 2.0, 1024: 2.0}
    with pytest.raises(ValueError, match="tier:weight"):
        parse_weights("banana", [896])
    with pytest.raises(ValueError, match="> 0"):
        parse_weights("896:0", [896])


def test_weights_to_repeats_normalizes_to_positive_ints():
    assert weights_to_repeats({1024: 2.0, 896: 1.0}) == {1024: 2, 896: 1}
    assert weights_to_repeats({1024: 0.6, 896: 0.4}) == {1024: 2, 896: 1}
    assert weights_to_repeats({1024: 0.1}) == {1024: 1}
    assert weights_to_repeats({}) == {}


def test_multires_dataset_config_shape():
    cfg = multires_dataset_config(
        [896, 1024],
        {896: 1, 1024: 2},
        resized_root="post_image_dataset/resized",
        lora_root="post_image_dataset/lora",
    )
    subsets = cfg["datasets"][0]["subsets"]
    assert [s["image_dir"] for s in subsets] == [
        "post_image_dataset/resized_t896",
        "post_image_dataset/resized_t1024",
    ]
    assert [s["cache_dir"] for s in subsets] == [
        "post_image_dataset/lora_t896",
        "post_image_dataset/lora_t1024",
    ]
    assert [s["num_repeats"] for s in subsets] == [1, 2]
    assert share_summary({896: 1, 1024: 2}) == "1024: 66% · 896: 33%"


def test_tier_dir_is_sibling():
    assert tier_dir("post_image_dataset/resized", 896) == (
        "post_image_dataset/resized_t896"
    )
    assert tier_dir(Path("a/b/resized"), 1024) == "a/b/resized_t1024"
    assert tier_dir("resized", 512) == "resized_t512"


def test_variant_blueprint_round_trip(tmp_path):
    variant = tmp_path / "lora.toml"
    variant.write_text(
        toml.dumps({"network_dim": 32, "output_name": "anima"}),
        encoding="utf-8",
    )
    blueprint = multires_dataset_config([896, 1024], {896: 1, 1024: 3})
    write_variant_blueprint(variant, blueprint)

    data = toml.loads(variant.read_text(encoding="utf-8"))
    # machine keys survive; blueprint landed inline
    assert data["network_dim"] == 32
    assert data["datasets"][0]["subsets"][1]["num_repeats"] == 3

    assert clear_variant_blueprint(variant) is True
    data = toml.loads(variant.read_text(encoding="utf-8"))
    assert "datasets" not in data and "general" not in data
    assert data["network_dim"] == 32
    # second clear is a no-op
    assert clear_variant_blueprint(variant) is False
