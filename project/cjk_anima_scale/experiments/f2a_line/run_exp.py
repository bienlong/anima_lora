#!/usr/bin/env python
"""f2a_line — proposal.md § 2.0 F2a: does an ungated ``v_line`` exist? (2026-09-27)

F1's ``v_line`` was trained gated, and read ungated (F1 at 0.5, the
``ungated`` leg of ``f1_line``) it keeps the words (≤ 1 edit 78 vs gated 80
/ 160) but costs the lone glyph: singles official 124 vs 152, repeat 43 vs
26, alone as a line 91 vs 74 / 320. Nothing in its training asked it to be
harmless alone. Here ``v_line`` trains **ungated** (``line_gate`` ``"all"``:
on every pack row, alone or in a word) with **every row frozen at the seed**
(``rows_frozen``: the rows it will be summed onto; ``v_line`` is the only
trainable), on data where a lone glyph must still render one glyph:

- Stage B's donor words (``scene_spelled``, b0507 + b0305, the 36 donors),
  **without** the count tier: its 28–40 px lone items sit off the seed's
  training distribution, so their gradient would pull ``v_line`` toward
  "render small lone glyphs better" rather than hold it harmless.
- ``builder.TABLE``'s b0709 single group (``scene_single`` + ``grid_single``)
  at share ``B0709_SHARE``, drawing the donors only: the band the seed rows
  were fit in, so at ``v_line`` = 0 its gradient is ≈ 0 and it acts as the
  "alone, stay as you are" constraint.

Same seed rows, trainer schedule (90 steps / row × 36) and held-out ten as
F1. The trained ``trained.pt`` is the seed rows + an ungated ``v_line``, so
the run dir is the dose-1 arm; other doses are built beside it.

Legs:
  data   (CPU) the data dir (Stage B's words + b0709 singles)
  train  (GPU) ``v_line`` only, ungated, rows frozen → ``run0927_f2a_line``
  read   (GPU) the held-out keys (en + swap) at ``--doses`` (1 = the run
         dir; else ``tf2_<label>_d<d>``) vs the floor, F1 gated 0.5
         (``tf_l1_line0.5``) and F1 ungated 0.5 (``tf_l1_ug0.5``), plus
         singles alone as a line

Pass (proposal § 2.0 acceptance 1–2 on kana): singles at the floor
(official ~149, repeat ~25, line ~78 / 320) with words ≤ 1 edit near gated
0.5's 80 / 160. The comparison that decides existence is against F1
ungated's trade-off: a point better on both axes means the lone items found
a direction; one that only shrinks ``v_line`` (singles back, words gone)
is the dose axis again.

``--dry_run`` prints the table and the encodings.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "f1_line_exp", LINE / "experiments" / "f1_line" / "run_exp.py"
)
F1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F1)
SB = F1.SB

EXP = OUT / "experiments"
NAME = "run0927_f2a_line"
B0709_SHARE = 0.5  # of the kind's items: 1 200 lone items beside 2 400 words
REFS = {"gated0.5": EXP / "tf_l1_line0.5", "f1_ug0.5": EXP / "tf_l1_ug0.5"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["data"], choices=["data", "train", "read"]
    )
    p.add_argument("--doses", type=float, nargs="+", default=[1.0])
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def rc():
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=NAME, path=Path(__file__), vocabs=(f"chars:{SB.DONOR}",), read=()
    )


def table() -> tuple:
    """Stage B's word tiers without the count tier, plus the b0709 singles."""
    from cjk_scale.builder import TABLE, Tier

    b0507, b0305 = SB.table()
    spelled = next(t for t in b0507.tiers if t.recipe == "scene_spelled")
    b0709 = next(g for g in TABLE if g.name == "b0709")
    return (
        dataclasses.replace(b0709, share=B0709_SHARE),
        dataclasses.replace(b0507, tiers=(Tier("scene_spelled", 1.0, spelled.params),)),
        b0305,
    )


def dose_arm(label: str, dose: float) -> Path:
    """The run's rows with ``dose`` · ``v_line`` (still ungated)."""
    import torch

    if dose == 1.0:
        return OUT / NAME
    sd = torch.load(OUT / NAME / "trained.pt", map_location="cpu", weights_only=False)
    d = sd["delta"]
    assert d.get("line_gate") == "all", "the F2a rows carry no ungated v_line"
    dst = EXP / f"tf2_{label}_d{dose:g}"
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(
        {**sd, "delta": {**d, "line": d["line"].float() * dose}, "dose": dose},
        dst / "trained.pt",
    )
    return dst


def read(metrics: dict, label: str, doses) -> list:
    chars = SB.held_keys()
    cl = "en,swap"
    floor = SB.check_floor(chars, cl)
    ref_h = {"floor": SB.hits(floor, chars, cl)}
    ref_l = {"floor": F1.alone_line(floor, chars, cl)}
    for k, p in REFS.items():
        reads = p / f"native_{SB.TAG}" / "native_reads.json"
        assert reads.exists(), f"no reads at {reads}"
        ref_h[k] = SB.hits(reads, chars, cl)
        ref_l[k] = F1.alone_line(reads, chars, cl)
    for k in ref_h:
        print(f"{k}, held-out keys:", flush=True)
        metrics[k] = SB.tally(ref_h[k])
        metrics[k]["singles_line"] = sum(ref_l[k].values())
        print(f"  SINGLES alone as a line {metrics[k]['singles_line']}", flush=True)
    arms = []
    for dose in doses:
        path = dose_arm(label, dose)
        arms.append(path)
        key = f"f2a_d{dose:g}"
        print(f"{path.name} (dose {dose:g}), held-out keys:", flush=True)
        reads = SB.native_read(path, path / "data", chars, cl)
        h = SB.hits(reads, chars, cl)
        ln = F1.alone_line(reads, chars, cl)
        metrics[key] = SB.tally(h)
        metrics[key]["singles_line"] = sum(ln.values())
        print(f"  SINGLES alone as a line {sum(ln.values())} / {len(ln)}", flush=True)
        for k in ref_h:
            metrics[key][f"paired_vs_{k}"] = {
                **SB.paired(h, ref_h[k]),
                "line": F1.paired_line(ln, ref_l[k]),
            }
            print(f"  vs {k} {metrics[key][f'paired_vs_{k}']}", flush=True)
    return arms


def main():
    args = parse_args()
    from cjk_scale import recipes

    recipes.RECIPES["scene_spelled"] = SB.scene_spelled
    words = json.loads((OUT / SB.NAME / "donor_words.json").read_text("utf-8"))
    cov = SB.set_words(words)
    ids = SB.check_spelling(SB.encoder(), words)
    tb = table()
    for g in tb:
        print(
            f"{g.name} σ {g.band} share {g.share}: "
            f"{[(t.recipe, t.weight, t.params) for t in g.tiers]}",
            flush=True,
        )
    print(
        f"{NAME}: {len(words)} donor words (per-glyph min {min(cov.values())}), "
        f"held ids {json.dumps({c: ids[c] for c in SB.HELD}, ensure_ascii=False)}",
        flush=True,
    )
    metrics: dict = {"name": NAME, "b0709_share": B0709_SHARE, "doses": args.doses}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "f2a_line", label=args.label, root=LINE / "experiments" / "f2a_line" / "results"
    )
    arms: list = []
    if "data" in args.legs:
        from cjk_scale.builder import build

        build(rc(), workers=args.workers, table=tb)
    if "train" in args.legs:
        from cjk_scale.train import train

        train(rc(), line_mode="all", rows_frozen=True)
    if "read" in args.legs:
        arms = read(metrics, args.label, args.doses)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / NAME), *(str(a) for a in arms)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
