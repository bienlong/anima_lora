#!/usr/bin/env python
"""dense_a0 — plan_2900 § 1's micro: does warm retraining lift dense kanji?
(2026-09-27)

Run A retrains 305 dense kanji (ink ≥ 11 cells² per glyph) warm from the
seed at 90 steps / row. That was never measured: stage_i measured cold kanji
(90 vs 270, ``reports/stage_i_2026_09_26.md`` § 5), and the seed's
well-trained singles lost native to displacement once (``conflict_joint``
§ 4-3). This micro reads it before the 3.3 h run.

- **Vocabs:** the 12 most frequent of A's 58 ``kanji:200`` dense kanji
  (``MICRO``). All have seed rows (``step1_0921`` base), so all start warm.
  最 輩 間 are the dense glyphs of the kanji_mode words 最高 先輩 何時間.
- **Arms:** stage_i's I0 (``b0709`` as is), at ``--budget 1`` (90 steps /
  row, the line's rule) and ``--budget 3`` (270, items × 3). One job per
  budget.
- **Read:**
  1. alone — the 12, ``en``, 16 each: official / contained / repeat / line
     per glyph, against the floor (the seed rows). These floor keys are new
     (``native_densea0/``) and are the first of A's own read sample.
  2. under the mode — the arm's rows + 0.5 · ``v_line`` on the six
     kanji_mode words, against their cached floor and mode floor
     (``native_kmode/``). Three words carry a retrained glyph; the other
     three are unchanged rows, a determinism check.

Picks A's budget: 90 if it lifts alone without losing the floor, 270 if
only × 3 lifts, and A is re-scoped if neither does.

Legs:
  data   (CPU) the arm's data dir (``run0927_a0[_x3]``)
  train  (GPU) the arm's rows
  mode   (CPU) ``experiments/a0[_x3]`` = the arm's rows + 0.5 · ``v_line``
  floor  (GPU) the 12 singles' floor keys into the seed dir's ``native_densea0/``
  read   (GPU) the arm on the singles and the words
  score  (CPU) the tables from the reads on disk (both budgets when present)

``--dry_run`` checks the encodings and exits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, floor_dir  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, LINE / "experiments" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SI = _load("stage_i_exp", "stage_i/run_exp.py")  # runs bootstrap()
KM = _load("kanji_mode_exp", "kanji_mode/run_exp.py")
from bench._common import make_run_dir, write_result  # noqa: E402

MICRO = "精俺感間奥愛様無最輩帰飲"
TAG = "densea0"
SI.TAG = TAG  # SI.native_read writes into native_<TAG>/
BUDGETS = (1, 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["score"],
        choices=["data", "train", "mode", "floor", "read", "score"],
    )
    p.add_argument("--budget", type=int, default=1, choices=BUDGETS)
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _sfx(b: int) -> str:
    return f"_x{b}" if b > 1 else ""


def run_name(b: int) -> str:
    return f"run0927_a0{_sfx(b)}"


def arm_dir(b: int) -> Path:
    return OUT / "experiments" / f"a0{_sfx(b)}"


def rc(b: int):
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=run_name(b), path=Path(__file__), vocabs=(f"chars:{MICRO}",), read=()
    )


def check_encodings(ext) -> dict:
    """Every micro glyph is one ext id with a ``step1_0921`` base row."""
    import torch

    base = {
        int(e)
        for e in torch.load(KM.BASE_ROWS, map_location="cpu", weights_only=False)[
            "delta"
        ]["ext_ids"]
    }
    ids = {}
    for c in MICRO:
        v = ext(c)
        assert len(v) == 1 and v[0] in base, (c, v)
        ids[c] = v[0]
    return ids


def mode_arm(b: int) -> Path:
    """The arm's merged rows + ``DOSE`` · ``v_line`` in the arm's row units."""
    import torch

    line = torch.load(
        SI.LINE_SRC / "trained.pt", map_location="cpu", weights_only=False
    )["delta"]
    sd = torch.load(
        OUT / run_name(b) / "trained.pt", map_location="cpu", weights_only=False
    )
    d = sd["delta"]
    k = float(line["row_scale"]) / float(d["row_scale"])
    dst = arm_dir(b)
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **sd,
            "delta": {**d, "line": line["line"].float() * k * SI.DOSE},
            "transplant": {"line_from": str(SI.LINE_SRC), "rescale": k, "dose": SI.DOSE},
        },
        dst / "trained.pt",
    )
    return dst


def score() -> dict:
    singles = list(MICRO)
    words = [SI.spell(w) for w in KM.WORDS]
    touched = [SI.spell(w) for w in KM.WORDS if set(w) & set(MICRO)]
    s_src = {"floor": floor_dir() / f"native_{TAG}" / "native_reads.json"}
    w_src = {
        "floor": floor_dir() / f"native_{KM.TAG}" / "native_reads.json",
        "mode_floor": SI.MODE_FLOOR / f"native_{KM.TAG}" / "native_reads.json",
    }
    for b in BUDGETS:
        f = arm_dir(b) / f"native_{TAG}" / "native_reads.json"
        s_src[f"a0{_sfx(b)}"] = w_src[f"a0{_sfx(b)}"] = f
    out: dict = {"singles": {}, "words": {}, "touched": {}}
    H: dict = {}
    for grp, keys, srcs in (
        ("singles", singles, s_src),
        ("words", words, w_src),
        ("touched", touched, w_src),
    ):
        for arm, f in srcs.items():
            if not f.exists():
                continue
            h = SI.hits(f, keys)
            if not h:
                continue
            H[(grp, arm)] = h
            out[grp][arm] = SI.tally(f"{grp}:{arm}", h)
            if grp != "singles":
                lo = KM.loose(f, {k.replace(" ", "") for k in keys})
                out[grp][arm]["loose"] = lo
                print(f"  {grp}:{arm:<10} loose {lo}", flush=True)
        ms = SI.SINGLE_M if grp == "singles" else SI.WORD_M
        for arm in [a for (g, a) in H if g == grp and a.startswith("a0")]:
            for ref in ("floor", "mode_floor", "a0"):
                if ref == arm or (grp, ref) not in H:
                    continue
                pr = SI.paired(H[(grp, arm)], H[(grp, ref)], ms)
                out[grp][f"{arm}_vs_{ref}"] = pr
                print(f"  {grp} {arm} vs {ref}: {pr}", flush=True)
    for arm in [a for (g, a) in H if g == "singles"]:
        per = out["singles"][arm]["per_key"]
        print(
            f"  per glyph {arm:<8} official/contained: "
            + " ".join(f"{t}{per[t]['official']}/{per[t]['contained']}" for t in MICRO if t in per),
            flush=True,
        )
    return out


def main():
    args = parse_args()
    SI.set_budget(args.budget)
    ids = check_encodings(SI.SB.encoder())
    print(f"micro {MICRO}: ids {json.dumps(ids, ensure_ascii=False)}", flush=True)
    for g in SI.table("I0"):
        print(f"I0 {g.name} σ {g.band} share {g.share}: {[(t.recipe, t.weight, t.params) for t in g.tiers]}", flush=True)
    metrics: dict = {"micro": MICRO, "ids": ids, "dose": SI.DOSE, "budget": args.budget}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "dense_a0", label=args.label, root=LINE / "experiments" / "dense_a0" / "results"
    )
    b = args.budget
    singles, words = list(MICRO), [SI.spell(w) for w in KM.WORDS]
    if "data" in args.legs:
        from cjk_scale.builder import build

        build(rc(b), workers=args.workers, table=SI.table("I0"))
    if "train" in args.legs:
        from cjk_scale.train import train

        train(rc(b))
    if "mode" in args.legs:
        print(f"mode arm {mode_arm(b)}", flush=True)
    if "floor" in args.legs:
        from cjk_scale.eval import ensure_native_floor

        n = ensure_native_floor(rc(b), f"native_{TAG}", singles, "en")
        print(f"floor: {n} keys rendered into native_{TAG}/", flush=True)
    if "read" in args.legs:
        SI.native_read(arm_dir(b), OUT / run_name(b) / "data", singles + words)
    metrics["score"] = score()
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / run_name(b)), str(arm_dir(b))],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
