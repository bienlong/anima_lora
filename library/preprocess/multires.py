"""Multi-resolution training (多重分辨率): per-tier preprocess + weighted subsets.

The user-facing feature: the SAME image is preprocessed and trained at EVERY
listed resolution tier, and the user allocates each tier a share of the
training time ("每档占用时长"). Implementation rides entirely on existing
machinery — no dataset/discovery code changes:

* Preprocess per tier: one ``preprocess-config`` pass per tier into tier
  suffixed dirs (``post_image_dataset/resized_t{tier}`` / ``lora_t{tier}``).
  Each pass resizes every source image with ``--target_res {tier}`` so the
  same image lands in that tier's band.
* Training blueprint: one ``[[datasets.subsets]]`` per tier with its own
  ``image_dir`` / ``cache_dir`` and ``num_repeats`` = the tier's relative
  weight. Repeats are the time share: with equal image counts per tier,
  step share ≈ repeats share (per-step token counts differ slightly across
  tiers, so wall-clock share is approximate — that is inherent, not waste).

This module is the pure, headless part: weight parsing / repeat
normalization / blueprint construction. The task runner
(``scripts/tasks/multires.py``) shells out to ``tasks.py preprocess-config``
per tier; the GUI Preprocess tab submits that task through the daemon.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_RESIZED_ROOT = "post_image_dataset/resized"
DEFAULT_LORA_ROOT = "post_image_dataset/lora"

_TIER_WEIGHT_RE = re.compile(r"^\s*(\d{3,4})\s*:\s*(\d+(?:\.\d+)?)\s*$")


def parse_tiers(raw: str | list | tuple | None) -> list[int]:
    """Parse a tier list ("1024,896" / [1024, 896]) → sorted unique ints.

    Values must be known free-fit tiers is NOT enforced here (the resize
    script validates); this only normalizes the shape.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace("，", ",").split(",") if p.strip()]
        tiers = [int(p) for p in parts]
    else:
        tiers = [int(t) for t in raw]
    if not tiers:
        return []
    bad = [t for t in tiers if t < 100 or t > 9999]
    if bad:
        raise ValueError(f"multi-res tiers must be 3-4 digit edges, got {bad}")
    return sorted(set(tiers))


def parse_weights(raw: str | None, tiers: list[int]) -> dict[int, float]:
    """Parse "1024:2, 896:1" (tier:weight) → ``{tier: float}``.

    Tiers without an entry default to weight 1.0; entries for tiers not in
    ``tiers`` are dropped with no error (they may come from a stale saved
    string after the tier list changed). A bare number ("2") applies to all
    tiers. Weights must be > 0.
    """
    out: dict[int, float] = {t: 1.0 for t in tiers}
    if raw is None or not str(raw).strip():
        return out
    for part in str(raw).replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        m = _TIER_WEIGHT_RE.match(part)
        if m:
            tier, weight = int(m.group(1)), float(m.group(2))
            if tier in out:
                if weight <= 0:
                    raise ValueError(f"multi-res weight must be > 0: {part!r}")
                out[tier] = weight
            continue
        # Bare number → uniform weight.
        try:
            uniform = float(part)
        except ValueError:
            raise ValueError(
                f"multi-res weights: expected 'tier:weight' pairs like "
                f"'1024:2, 896:1', got {part!r}"
            ) from None
        if uniform <= 0:
            raise ValueError(f"multi-res weight must be > 0: {part!r}")
        out = {t: uniform for t in tiers}
    return out


def weights_to_repeats(weights: dict[int, float]) -> dict[int, int]:
    """Normalize weights to integer repeats ≥ 1, preserving ratios.

    The smallest weight maps to 1 and the rest round proportionally, so
    ``{1024: 2.0, 896: 1.0}`` → ``{1024: 2, 896: 1}`` and
    ``{1024: 0.6, 896: 0.4}`` → ``{1024: 2, 896: 1}`` (3:2 exactly is not
    always reachable with ints; nearest-int is fine for time shares). A
    tiny epsilon guards the float division (0.6 / 0.4 = 1.4999… must round
    to 2, not banker's-round down to 1).
    """
    if not weights:
        return {}
    floor = min(weights.values())
    return {tier: max(1, round(w / floor + 1e-9)) for tier, w in weights.items()}


def tier_dir(base_dir: str | Path, tier: int) -> str:
    """Tier-suffixed sibling of a base dataset dir:
    ``post_image_dataset/resized`` + 896 → ``post_image_dataset/resized_t896``.

    Always POSIX-separated — blueprint TOMLs use forward slashes like
    base.toml, regardless of the host OS.
    """
    p = Path(base_dir)
    parent = p.parent.as_posix()
    name = p.name + f"_t{tier}"
    return f"{parent}/{name}" if parent and parent != "." else name


def subset_for_tier(
    tier: int,
    repeats: int,
    *,
    resized_root: str = DEFAULT_RESIZED_ROOT,
    lora_root: str = DEFAULT_LORA_ROOT,
) -> dict:
    """One ``[[datasets.subsets]]`` entry for a tier."""
    return {
        "image_dir": tier_dir(resized_root, tier),
        "cache_dir": tier_dir(lora_root, tier),
        "num_repeats": int(repeats),
    }


def multires_dataset_config(
    tiers: list[int],
    repeats: dict[int, int],
    *,
    resized_root: str = DEFAULT_RESIZED_ROOT,
    lora_root: str = DEFAULT_LORA_ROOT,
    batch_size: int = 1,
) -> dict:
    """A full self-contained dataset blueprint: one subset per tier.

    Shape matches the trainer's ``[[datasets]]`` + ``[[datasets.subsets]]``
    schema (the self-contained layout a variant TOML or ``--dataset_config``
    file carries).
    """
    subsets = [
        subset_for_tier(
            t, repeats.get(t, 1), resized_root=resized_root, lora_root=lora_root
        )
        for t in tiers
    ]
    return {
        "general": {"batch_size": int(batch_size)},
        "datasets": [{"batch_size": int(batch_size), "subsets": subsets}],
    }


def share_summary(repeats: dict[int, int]) -> str:
    """Human-readable step-share estimate, e.g. "1024: 67% · 896: 33%"."""
    total = sum(repeats.values()) or 1
    return " · ".join(
        f"{tier}: {r * 100 // total}%"
        for tier, r in sorted(repeats.items(), reverse=True)
    )


def sanitize_variant_stem(name: str) -> str:
    """File-safe component for variant/dataset file names."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_") or "multires"


def write_variant_blueprint(
    variant_toml_path: Path,
    blueprint: dict,
) -> None:
    """Replace the ``[[datasets]]`` blueprint inside a variant TOML.

    Variants are machine-written (the GUI save round-trips through
    ``toml.dumps`` and strips comments anyway), so a plain load→update→dump
    is the established write path. The blueprint is written in the
    self-contained per-method layout: ``[[datasets]]`` with inline
    ``subsets``, which the config loader treats as a full replacement of
    base.toml's blueprint.
    """
    import toml

    path = Path(variant_toml_path)
    data = toml.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.pop("datasets", None)
    data.pop("general", None)
    data.update(blueprint)
    path.write_text(toml.dumps(data), encoding="utf-8")


def clear_variant_blueprint(variant_toml_path: Path) -> bool:
    """Remove an inline blueprint from a variant TOML (back to base's).

    Returns True when a blueprint was actually removed.
    """
    import toml

    path = Path(variant_toml_path)
    if not path.exists():
        return False
    data = toml.loads(path.read_text(encoding="utf-8"))
    if "datasets" not in data and "general" not in data:
        return False
    data.pop("datasets", None)
    data.pop("general", None)
    path.write_text(toml.dumps(data), encoding="utf-8")
    return True
