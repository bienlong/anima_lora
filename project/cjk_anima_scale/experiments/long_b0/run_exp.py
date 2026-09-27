#!/usr/bin/env python
"""long_b0 — plan_2900 § 2's micro: what budget do 3–5-glyph pieces need?
(2026-09-27)

Run B retrains the seed's 521 pieces of 3–5 glyphs warm. 90 steps / row on
the piece table is the measured failure for 4–5 glyphs: ちょっと 0 → 2,
こんにちは 0 → 0 of 16 alone in ``run0925_300f`` (``reports/piece_2026_09_25.md``),
with exposure per item no shorter than a 2-glyph piece's. Budget is the
measured lever for thin identity (stage_i × 3 doubled cold kanji,
``reports/stage_i_2026_09_26.md`` § 5); it was never run on pieces.

- **Vocabs:** 12 pieces, 4 per glyph count (``MICRO``). All have seed rows
  (the cold 1 900), so all start warm.
- **Arms:** the line's piece table (``builder.TABLE``: ``b0507`` + ``b0305``)
  as is, at ``--budget 1`` (90 steps / row) and ``--budget 3`` (270, items
  × 3). One job per budget.
- **Read:** the 12 alone, ``en``, 16 each (the piece ruler's ``native``
  stage), against the floor (the seed rows): official / contained (the
  piece report's *lenient*) / exact / in-word doubling, per piece and per
  glyph count. The floor keys go into the seed dir's ``native_piece/`` —
  the piece ruler's own cache, so B's read reuses them (four are cached:
  すごい ちょっと ありがとう こんにちは).

Sets B's budget per glyph count (and C-p's for its 3+-glyph pieces): 90
where × 1 lifts off the floor as far as × 3, 270 where only × 3 lifts.

Legs:
  data   (CPU) the arm's data dir (``run0927_b0[_x3]``)
  train  (GPU) the arm's rows
  floor  (GPU) the 12 pieces' floor keys into the seed dir's ``native_piece/``
  read   (GPU) the arm on the 12
  score  (CPU) the tables from the reads on disk (both budgets when present)

``--dry_run`` checks the encodings and prints the table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, floor_dir  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, LINE / "experiments" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SI = _load("stage_i_exp", "stage_i/run_exp.py")  # runs bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

MICRO = (
    "すごい", "一緒に", "気持ち", "わかる",
    "ちょっと", "まったく", "しっかり", "もちろん",
    "ありがとう", "こんにちは", "ありません", "わからない",
)  # fmt: skip
TAG = "piece"  # the piece ruler's read dir: native_piece/
SI.TAG = TAG  # SI.native_read writes into native_<TAG>/
BUDGETS = (1, 3)
METRICS = ("official", "contained", "exact", "wdup")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["score"],
        choices=["data", "train", "floor", "read", "score"],
    )
    p.add_argument("--budget", type=int, default=1, choices=BUDGETS)
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _sfx(b: int) -> str:
    return f"_x{b}" if b > 1 else ""


def run_name(b: int) -> str:
    return f"run0927_b0{_sfx(b)}"


def rc(b: int):
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=run_name(b),
        path=Path(__file__),
        vocabs=("list:" + ",".join(MICRO),),
        read=(),
    )


def check_encodings(ext) -> dict:
    """Every micro piece is one ext id with a seed row, and has 3–5 glyphs."""
    import torch

    seed = {
        int(e)
        for e in torch.load(SEED_ROWS, map_location="cpu", weights_only=False)["delta"][
            "ext_ids"
        ]
    }
    ids = {}
    for v in MICRO:
        e = ext(v)
        assert len(e) == 1 and e[0] in seed, (v, e)
        assert 3 <= len(v) <= 5, v
        ids[v] = e[0]
    return ids


def score() -> dict:
    keys = list(MICRO)
    src = {"floor": floor_dir() / f"native_{TAG}" / "native_reads.json"}
    for b in BUDGETS:
        src[f"b0{_sfx(b)}"] = OUT / run_name(b) / f"native_{TAG}" / "native_reads.json"
    H: dict = {}
    out: dict = {}
    for arm, f in src.items():
        if not f.exists():
            continue
        h = SI.hits(f, keys)
        if not h:
            continue
        H[arm] = h
        out[arm] = SI.tally(arm, h)
        by_n: dict = {}
        for (text, _pi, _s), v in h.items():
            c = by_n.setdefault(len(text), {"n": 0, **dict.fromkeys(METRICS, 0)})
            c["n"] += 1
            for m in METRICS:
                c[m] += v[m]
        out[arm]["by_glyphs"] = by_n
        for n, c in sorted(by_n.items()):
            print(
                f"  {arm:<8} {n} glyphs  "
                + "  ".join(f"{m} {c[m]:>3}" for m in METRICS)
                + f" / {c['n']}",
                flush=True,
            )
    for arm in [a for a in H if a.startswith("b0")]:
        for ref in ("floor", "b0"):
            if ref == arm or ref not in H:
                continue
            pr = SI.paired(H[arm], H[ref], METRICS)
            out[arm][f"paired_vs_{ref}"] = pr
            print(f"  {arm} vs {ref}: {pr}", flush=True)
    for arm in H:
        per = out[arm]["per_key"]
        print(
            f"  per piece {arm:<8} official/contained: "
            + " ".join(
                f"{t} {per[t]['official']}/{per[t]['contained']}"
                for t in MICRO
                if t in per
            ),
            flush=True,
        )
    return out


def main():
    args = parse_args()
    SI.set_budget(args.budget)
    ids = check_encodings(SI.SB.encoder())
    print(
        f"micro {' '.join(MICRO)}: ids {json.dumps(ids, ensure_ascii=False)}",
        flush=True,
    )
    from cjk_scale.builder import TABLE

    for g in (g for g in TABLE if g.kind == "piece"):
        print(
            f"piece {g.name} σ {g.band} share {g.share}: "
            f"{[(t.recipe, t.weight, t.params) for t in g.tiers]}",
            flush=True,
        )
    metrics: dict = {"micro": list(MICRO), "ids": ids, "budget": args.budget}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "long_b0", label=args.label, root=LINE / "experiments" / "long_b0" / "results"
    )
    b = args.budget
    if "data" in args.legs:
        from cjk_scale.builder import build

        build(rc(b), workers=args.workers)
    if "train" in args.legs:
        from cjk_scale.train import train

        train(rc(b))
    if "floor" in args.legs:
        from cjk_scale.eval import ensure_native_floor

        n = ensure_native_floor(rc(b), f"native_{TAG}", list(MICRO), "en")
        print(f"floor: {n} keys rendered into native_{TAG}/", flush=True)
    if "read" in args.legs:
        SI.native_read(OUT / run_name(b), OUT / run_name(b) / "data", list(MICRO))
    metrics["score"] = score()
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / run_name(b))],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
