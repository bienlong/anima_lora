"""``make mask``'s rules → ``SamMaskRequest`` translation (scripts/tasks/masking.py).

The rules are ``sam_mask.yaml``'s (the package's CLI stopped reading it at
anime_tools 0.4.0; the trainer normalizes the flat / ``rules:`` schemas itself
and builds one request per rule). The argv those requests produce is round-tripped through the
package parser in ``test_anime_tools_cli_contract.py``; this file pins the
normalization, including the pre-0.6.4 ``prompts`` / ``focus_prompts`` pair
(``library.config.sam_masks``). MIT (the text masker) was removed in v2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from anime_tools.downloads import DEFAULT_SUBJECT_PROMPT_EMBED

from scripts.tasks import masking

SOFT_GIRL = f"keep:soft:{DEFAULT_SUBJECT_PROMPT_EMBED}"


def _specs(req) -> list[str]:
    return [m.spec() for m in req.masks]


def test_flat_config_is_one_rule_with_top_level_thresholds():
    rules = masking._sam_rules(
        {"masks": ["ignore:text:speech bubble"], "threshold": 0.7, "dilate": 3}
    )
    assert rules == [
        {
            "masks": ("ignore:text:speech bubble",),
            "path_pattern": None,
            "threshold": 0.7,
            "dilate": 3,
        }
    ]


def test_rules_fall_back_to_top_level_then_package_defaults():
    rules = masking._sam_rules(
        {
            "threshold": 0.6,
            "rules": [
                {"masks": ["ignore:text:bubble"]},
                {"path_pattern": "a/*", "masks": ["keep:text:girl"], "dilate": 8},
                {"path_pattern": "*", "masks": ["ignore:text:text"], "threshold": 0.9},
            ],
        }
    )
    assert [r["threshold"] for r in rules] == [0.6, 0.6, 0.9]
    assert "dilate" not in rules[0] and rules[1]["dilate"] == 8
    assert [r["path_pattern"] for r in rules] == [None, "a/*", None]
    req = masking._sam_request(Path("r"), Path("o"), rules[0], None)
    assert req.dilate == 5  # the package default, not a trainer literal


def test_rule_pattern_wins_over_the_global_scope():
    rules = masking._sam_rules(
        {
            "rules": [
                {"path_pattern": "a/*", "masks": ["ignore:text:x"]},
                {"masks": ["ignore:text:y"]},
            ]
        }
    )
    own = masking._sam_request(Path("r"), Path("o"), rules[0], "manga/*")
    scoped = masking._sam_request(Path("r"), Path("o"), rules[1], "manga/*")
    assert own.path_pattern == "a/*"
    assert scoped.path_pattern == "manga/*"


def test_a_pre_064_yaml_pair_reads_as_masks():
    """``prompts`` → ignore:text, ``focus_prompts`` → keep:text, and ``girl`` —
    which the old stage served through ``--prompt_embed`` — the soft entry. An
    empty focus list is no keep region (the argv spells the ignore list alone,
    so the child does not add the default subject on top)."""
    rule = masking._sam_rules({"prompts": ["bubble"], "focus_prompts": []})[0]
    req = masking._sam_request(Path("r"), Path("o"), rule, None)
    assert _specs(req) == ["ignore:text:bubble"]
    argv = req.to_argv()
    assert argv[argv.index("--masks") + 1 :] == ["ignore:text:bubble"]
    rule = masking._sam_rules(
        {"rules": [{"prompts": ["text"], "focus_prompts": ["girl", "face"]}]}
    )[0]
    assert rule["masks"] == (SOFT_GIRL, "keep:text:face", "ignore:text:text")


def test_yaml_run_sam_off_means_no_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(masking, "_load_mask_config", lambda: {"run_sam": False})
    assert masking._sam_requests(Path("resized"), tmp_path) == []


def test_run_switches_accept_bools_and_env_style_strings():
    assert masking._config_flag({}, "run_sam") is True
    assert masking._config_flag({"run_sam": False}, "run_sam") is False
    assert masking._config_flag({"run_sam": "0"}, "run_sam") is False
    assert masking._config_flag({"run_sam": "no"}, "run_sam") is False
    assert masking._config_flag({"run_sam": "1"}, "run_sam") is True


def test_make_mask_refuses_stray_args():
    with pytest.raises(SystemExit, match="takes no ARGS"):
        masking.cmd_mask(["--force"])


def test_rule_without_masks_fails_before_the_sam3_load():
    rule = masking._sam_rules({"rules": [{"path_pattern": "a/*"}]})[0]
    with pytest.raises(SystemExit, match="nothing to mask"):
        masking._sam_request(Path("r"), Path("o"), rule, None)
