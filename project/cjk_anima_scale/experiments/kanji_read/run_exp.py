#!/usr/bin/env python
"""kanji_read — a kanji batch's singles against the floor caches it already has (2026-09-28)

A small read of a ``retrain_kanji_b*`` run: the batch's kanji that the seed
floor already rendered as lone singles (``en`` clause, the ``native`` stage's
8 prompts × 2 seeds) in ``native_single/``, ``native_densea0/`` and
``native_stagei/``. No floor render — a lone glyph encodes the same routed
or not, so the unrouted floor caches stand. The run renders the first
``N_PROMPTS`` prompts × ``SEEDS`` seeds per kanji (user: ≈ 90 renders, not
the full 16 per key) into ``<run>/native_r2_en/``, paired with the floor's
renders of the same prompt × seed (Stage B's scoring, McNemar).

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label kanji-read-b1 \\
      project/cjk_anima_scale/experiments/kanji_read/run_exp.py \\
      --label b1 --run retrain_kanji_b1"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the run's own encoding (singles: same either way)
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, floor_dir  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, LINE / "experiments" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SB = _load("stage_b_exp", "stage_b/run_exp.py")

N_PROMPTS = 2
SEEDS = 2
CLAUSE = "en"
FLOOR_CACHES = ("native_single", "native_densea0", "native_stagei")  # 8 × 2, en
SUB = f"r{N_PROMPTS}_{CLAUSE}"


def floor_reads(chars: set) -> tuple[list, dict]:
    """The floor's lone-glyph reads of ``chars`` on this grid, first cache
    wins; each key's source cache."""
    recs, src = [], {}
    for c in FLOOR_CACHES:
        for m in json.loads((floor_dir() / c / "native_reads.json").read_text("utf-8")):
            t = m["text"]
            if t not in chars or m["clause"] != CLAUSE or src.get(t, c) != c:
                continue
            if int(m["pi"]) < N_PROMPTS and int(m["seed"]) < SEEDS:
                src[t] = c
                recs.append(m)
    return recs, src


def grid(h: dict) -> dict:
    return {k: v for k, v in h.items() if int(k[2]) < N_PROMPTS and int(k[3]) < SEEDS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    from cjk_scale.config import load_run
    from cjk_scale.eval import TRAINED_ARM, _fold, _load_reads, probe_args
    from stages import run as run_stage

    rc = load_run(args.run)
    arm = OUT / rc.name
    vocabs = set(json.loads((arm / "data" / "vocabs.json").read_text("utf-8")))
    frecs, src = floor_reads(vocabs)
    chars = sorted(src)
    print(
        f"{rc.name}: {len(chars)} kanji with a floor ({''.join(chars)}), "
        f"{N_PROMPTS} prompts × {SEEDS} seeds → {len(chars) * N_PROMPTS * SEEDS} renders",
        flush=True,
    )
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "kanji_read", label=args.label, root=LINE / "experiments" / "kanji_read" / "results"
    )
    dst = arm / f"native_{SUB}" / "native_reads.json"
    held = {m["text"] for m in _load_reads(dst)}
    miss = [c for c in chars if c not in held]
    if miss:
        a = probe_args(
            rc,
            TRAINED_ARM,
            ["native"],
            [
                "--eval_tag", f"add_{SUB}",
                "--native_chars", ",".join(miss),
                "--native_clauses", CLAUSE,
                "--native_limit", str(N_PROMPTS),
                "--seeds", str(SEEDS),
            ],
        )
        a.arm_path = str(arm)
        run_stage("native", a)
        scratch = arm / f"native_add_{SUB}"
        _fold("native", _load_reads(scratch / "native_reads.json"), dst, move=True)
        shutil.rmtree(scratch)

    # the pairs must be the same prompt × seed
    fprompt = {(m["text"], int(m["pi"]), int(m["seed"])): m["prompt"] for m in frecs}
    for m in _load_reads(dst):
        k = (m["text"], int(m["pi"]), int(m["seed"]))
        if k in fprompt:
            assert fprompt[k] == m["prompt"], (k, fprompt[k], m["prompt"])

    ffile = run_dir / "floor_reads.json"
    ffile.write_text(json.dumps(frecs, ensure_ascii=False), encoding="utf-8")
    mine = grid(SB.hits(dst, chars, CLAUSE))
    ref = grid(SB.hits(ffile, chars, CLAUSE))
    print(f"{rc.name} · singles ({CLAUSE}, {N_PROMPTS}×{SEEDS}):", flush=True)
    t_mine = SB.tally(mine)
    print("floor:", flush=True)
    t_ref = SB.tally(ref)
    by_cache = {
        c: SB.paired(
            {k: v for k, v in mine.items() if src[k[0]] == c},
            {k: v for k, v in ref.items() if src[k[0]] == c},
        )
        for c in FLOOR_CACHES
    }
    pr = SB.paired(mine, ref)
    print(f"  paired {rc.name} vs floor {pr}", flush=True)
    for c, v in by_cache.items():
        print(f"    {c}: {v}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={
            "chars": "".join(chars),
            "source": src,
            "grid": [N_PROMPTS, SEEDS],
            "run": t_mine,
            "floor": t_ref,
            "paired": pr,
            "paired_by_cache": by_cache,
        },
        artifacts=[str(dst)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
