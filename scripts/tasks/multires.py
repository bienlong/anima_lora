"""Multi-resolution training task (多重分辨率训练).

For each tier the user lists, runs one ``preprocess-config`` pass that
resizes EVERY source image with ``--target_res {tier}`` and caches VAE + TE
outputs into tier-suffixed dirs (``post_image_dataset/resized_t{tier}`` /
``lora_t{tier}``). Then writes a self-contained dataset blueprint with one
``[[datasets.subsets]]`` per tier and ``num_repeats`` = the tier's relative
time share (the "每档占用时长" allocation).

Usage (from the anima_lora root):

    python tasks.py multires --tiers 1024,896 --weights "1024:2,896:1"
    python tasks.py multires --tiers 1024,896 --skip-preprocess   # blueprint only

The written blueprint (``configs/datasets/multires.toml`` by default) is a
plain ``--dataset_config`` file; the GUI's "写入变体" action copies the same
blueprint into the active gui-methods variant instead, after which the normal
Train button picks it up (self-contained per-method layout).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from library.preprocess.multires import (
    DEFAULT_LORA_ROOT,
    DEFAULT_RESIZED_ROOT,
    multires_dataset_config,
    parse_tiers,
    parse_weights,
    share_summary,
    tier_dir,
    weights_to_repeats,
)

PY = sys.executable


def cmd_multires(args: list[str]) -> None:
    """`multires` task entry: per-tier preprocess + weighted blueprint."""
    import toml

    tiers: list[int] = []
    weights_raw: str | None = None
    src: str | None = None
    dataset_out = str(Path("configs") / "datasets" / "multires.toml")
    skip_preprocess = False
    vae = qwen3 = dit = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tiers" and i + 1 < len(args):
            tiers = parse_tiers(args[i + 1])
            i += 2
        elif a == "--weights" and i + 1 < len(args):
            weights_raw = args[i + 1]
            i += 2
        elif a == "--src" and i + 1 < len(args):
            src = args[i + 1]
            i += 2
        elif a == "--dataset-out" and i + 1 < len(args):
            dataset_out = args[i + 1]
            i += 2
        elif a == "--skip-preprocess":
            skip_preprocess = True
            i += 1
        elif a == "--vae" and i + 1 < len(args):
            vae = args[i + 1]
            i += 2
        elif a == "--qwen3" and i + 1 < len(args):
            qwen3 = args[i + 1]
            i += 2
        elif a == "--dit" and i + 1 < len(args):
            dit = args[i + 1]
            i += 2
        else:
            print(f"[multires] unknown argument {a!r} ignored")
            i += 1

    if not tiers:
        print(
            "[multires] --tiers is required, e.g. --tiers 1024,896 "
            "(every source image is preprocessed and trained at each tier)"
        )
        raise SystemExit(1)

    src = src or "image_dataset"
    weights = parse_weights(weights_raw, tiers)
    repeats = weights_to_repeats(weights)

    root = Path.cwd()
    # Absolute dirs for the subprocess passes; the final blueprint keeps
    # repo-relative paths (matching base.toml's convention) so it stays
    # portable across checkouts.
    abs_resized_root = str(root / DEFAULT_RESIZED_ROOT)
    abs_lora_root = str(root / DEFAULT_LORA_ROOT)

    if not skip_preprocess:
        for tier in tiers:
            image_dir = tier_dir(abs_resized_root, tier)
            cache_dir = tier_dir(abs_lora_root, tier)
            one_tier = {
                "datasets": [
                    {"subsets": [{"image_dir": image_dir, "cache_dir": cache_dir}]}
                ]
            }
            fd, tmp_name = tempfile.mkstemp(suffix="_multires.toml")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                toml.dump(one_tier, fh)
            try:
                cmd = [
                    PY,
                    "tasks.py",
                    "preprocess-config",
                    "--dataset_config",
                    tmp_name,
                    "--src",
                    str(Path(src).resolve()),
                    "--target_res",
                    str(tier),
                ]
                # Model paths: only forward when explicitly given —
                # preprocess-config falls back to its config-chain defaults.
                if vae:
                    cmd += ["--vae", vae]
                if qwen3:
                    cmd += ["--qwen3", qwen3]
                if dit:
                    cmd += ["--dit", dit]
                print(f"[multires] tier {tier}: preprocess → {image_dir}")
                result = subprocess.run(cmd, cwd=str(root))
                if result.returncode != 0:
                    raise SystemExit(
                        f"[multires] preprocess failed for tier {tier} "
                        f"(exit {result.returncode})"
                    )
            finally:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    else:
        print("[multires] --skip-preprocess: writing blueprint only")

    config = multires_dataset_config(
        tiers, repeats, resized_root=DEFAULT_RESIZED_ROOT, lora_root=DEFAULT_LORA_ROOT
    )
    out_path = Path(dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(toml.dumps(config), encoding="utf-8")
    print(
        f"[multires] blueprint written: {out_path}\n"
        f"[multires] tiers {tiers}, repeats {repeats} "
        f"(step share ≈ {share_summary(repeats)})\n"
        f"[multires] train with --dataset_config {out_path}, or use the GUI "
        "'写入变体' action to make the normal Train button use it."
    )
