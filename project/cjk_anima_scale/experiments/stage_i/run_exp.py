#!/usr/bin/env python
"""stage_i — proposal.md § 2.3: which scene makes a good identity (2026-09-26)

With the line carried by ``v_line`` (seed rows + 0.5 · ``v_line`` + the run
gate), the question for a row ``r_i`` is which training context gives the
identity with the least layout baked in. Three candidates, each ``r_i``
data only (one glyph per item or per grid cell, gate off in training), on a
micro set of **cold rows**: 12 kanji the seed rows lack, frequent in the
manga109s dialogue pool and forming real words among themselves.

- **I0**, the incumbent: ``builder.TABLE``'s b0709 as is — ``scene_single``
  (fill 0.7, ≈ 50 px) + ``grid_single`` (1×1–3×3), σ 0.7–0.9.
- **I1**, scene only with a px spread: ``scene_single`` at a target px in a
  bubble it fills 0.2–1.0 of; half the items at font 45–66 px (σ 0.7–0.9),
  half at 27–44 px (σ 0.5–0.7 — the band law's 24–40 px single row).
- **I2**, grid only at small cells (3×3 / 2×3): the same two px halves on
  ``grid_single`` — the negative control for "native needs scene".

Every arm has the same items per row (``ITEMS_PER_VOCAB``), trainer and
steps / row, warm from the seed (these 12 start cold). The read arm is the
trained rows + 0.5 · ``v_line`` (F1's, rescaled to the arm's row_scale):

1. **alone** — the 12 kanji, ``en``: official, repeat, alone-as-a-line
   (either reader reads ≥ 3 kana / kanji); a lone glyph has no neighbour,
   so the gate is off and this is the rows alone.
2. **under the mode** — six spelled words of the set (山田太郎 小山田 太郎
   道場 星空 天地), ``en``: exact, contained, ≤ 1 edit (≥ 3 glyphs), in-word
   doubling ``wdup``.

Reference arms: the floor (the seed rows: pack rows for these 12) on both,
and ``tf_l1_line0.5`` (the seed rows + 0.5 · ``v_line``: the mode with no
trained identity) on the words. Pick the recipe that wins 2 without losing
1; it closes if every candidate ties on 2.

Legs:
  data   (CPU) one data dir per arm (``run0926_si_<arm>``)
  train  (GPU) the arms' rows
  mode   (CPU) ``experiments/si_<arm>`` = the arm's rows + 0.5 · ``v_line``
  floor  (GPU) the floor keys into the seed dir's ``native_stagei/``, and
         the words on ``tf_l1_line0.5``
  read   (GPU) the arms on every key
  score  (CPU) the tables from the reads on disk

``--budget N`` scales items / row and steps / row by N for this experiment
(names ``_xN``; score adds the x1 reads on disk as references). The x1 run
had 9 / 12 glyphs never official alone (reports/stage_i_2026_09_26.md), so
criterion 2 was unreadable.

``--dry_run`` prints the tables and checks the encodings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, floor_dir  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage_b_exp", LINE / "experiments" / "stage_b" / "run_exp.py"
)
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)

EXP = OUT / "experiments"
MICRO = "山田野郎太道場星空天地小"
WORDS = ("山田太郎", "小山田", "太郎", "道場", "星空", "天地")
ARMS = ("I0", "I1", "I2")
DOSE = 0.5
LINE_SRC = OUT / "run0926_f1_line"  # v_line
MODE_FLOOR = EXP / "tf_l1_line0.5"  # the seed rows + 0.5 · v_line
TAG = "stagei"
GLYPH = re.compile(r"[ぁ-ゟァ-ヿ一-鿿]")
DUP = re.compile(r"(.)\1")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "train", "mode", "floor", "read", "score"],
    )
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument(
        "--budget",
        type=int,
        default=1,
        help="× items / row and × steps / row (the line's rules scaled for "
        "this experiment only); > 1 suffixes every name with _x<budget>",
    )
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


BUDGET = 1  # set from --budget in main


def _sfx(budget: int | None = None) -> str:
    b = BUDGET if budget is None else budget
    return f"_x{b}" if b > 1 else ""


def run_name(arm: str, budget: int | None = None) -> str:
    return f"run0926_si_{arm.lower()}{_sfx(budget)}"


def arm_dir(arm: str, budget: int | None = None) -> Path:
    return EXP / f"si_{arm.lower()}{_sfx(budget)}"


def set_budget(b: int) -> None:
    """× the builder's items / row and the trainer's steps / row."""
    global BUDGET
    from cjk_scale import builder
    from cjk_scale import train as tr

    BUDGET = b
    builder.ITEMS_PER_VOCAB = builder.ITEMS_PER_VOCAB * b
    tr.STEPS_PER_VOCAB = tr.STEPS_PER_VOCAB * b


def rc(arm: str):
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=run_name(arm), path=Path(__file__), vocabs=(f"chars:{MICRO}",), read=()
    )


def table(arm: str) -> tuple:
    from cjk_scale.builder import TABLE, Group, Tier

    b0709 = next(g for g in TABLE if g.name == "b0709")
    if arm == "I0":
        return (b0709,)
    if arm == "I1":
        recipe, big, small = (
            "scene_single",
            {"glyph_px": [45, 66], "fill": 1.0, "fill_min": 0.2, "min_glyph": 12},
            {"glyph_px": [27, 44], "fill": 1.0, "fill_min": 0.2, "min_glyph": 12},
        )
    else:
        grid = {"grids": "3x3:1,2x3:1", "bubble_frac": 0.5, "mark_horizontal": True}
        recipe, big, small = (
            "grid_single",
            {**grid, "glyph_px": [45, 85]},
            {**grid, "glyph_px": [27, 44]},
        )
    return (
        Group("b0709", "single", (0.7, 0.9), 0.5, (Tier(recipe, 1.0, big),)),
        Group("b0507", "single", (0.5, 0.7), 0.5, (Tier(recipe, 1.0, small),)),
    )


def spell(s: str) -> str:
    return " ".join(s)


def keys() -> tuple[list[str], list[str]]:
    return list(MICRO), [spell(w) for w in WORDS]


def check_encodings(ext) -> dict:
    """Every micro glyph is one ext id the seed rows lack; every spelled word
    encodes to its glyphs' ids."""
    import torch

    have = {
        int(e)
        for e in torch.load(SEED_ROWS, map_location="cpu", weights_only=False)[
            "delta"
        ]["ext_ids"]
    }
    ids = {}
    for c in MICRO:
        v = ext(c)
        assert len(v) == 1, (c, v)
        assert v[0] not in have, f"{c} has a seed row"
        ids[c] = v[0]
    for w in WORDS:
        assert set(w) <= set(MICRO) and len(set(w)) == len(w), w
        assert ext(spell(w)) == [ids[c] for c in w], (w, ext(spell(w)))
    return ids


# ----------------------------------------------------------------------------
# arms


def mode_arm(arm: str) -> Path:
    """The arm's merged rows + ``DOSE`` · ``v_line`` in the arm's row units."""
    import torch

    line = torch.load(
        LINE_SRC / "trained.pt", map_location="cpu", weights_only=False
    )["delta"]
    src = OUT / run_name(arm) / "trained.pt"
    sd = torch.load(src, map_location="cpu", weights_only=False)
    d = sd["delta"]
    k = float(line["row_scale"]) / float(d["row_scale"])
    dst = arm_dir(arm)
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **sd,
            "delta": {**d, "line": line["line"].float() * k * DOSE},
            "transplant": {"line_from": str(LINE_SRC), "rescale": k, "dose": DOSE},
        },
        dst / "trained.pt",
    )
    return dst


def native_read(arm_path: Path, data_path: Path, chars, clauses: str = "en") -> Path:
    from cjk_scale.config import RunConfig
    from cjk_scale.eval import TRAINED_ARM, probe_args
    from stages import run as run_stage

    reads = arm_path / f"native_{TAG}" / "native_reads.json"
    if reads.exists():
        held = {(m["text"], m["clause"]) for m in json.loads(reads.read_text("utf-8"))}
        if all((k, clauses) in held for k in chars):
            print(f"  (read from disk: {reads})", flush=True)
            return reads
    a = probe_args(
        RunConfig(name="stage_i", path=Path(__file__), vocabs=(), read=()),
        TRAINED_ARM,
        ["native"],
        ["--eval_tag", TAG, "--native_chars", ",".join(chars), "--native_clauses", clauses],
    )
    a.arm_path, a.data_path = str(arm_path), str(data_path)
    run_stage("native", a)
    return reads


# ----------------------------------------------------------------------------
# scoring


def hits(path: Path, chars) -> dict:
    from common.readers import norm

    out = {}
    for m in json.loads(path.read_text("utf-8")):
        if m["text"] not in chars or m["clause"] != "en":
            continue
        t = norm(m["text"])
        reads = [
            norm(r.get(x) or "") for r in m.get("reads", []) for x in ("sfx", "vl")
        ]
        best = min((SB._lev(r, t) for r in reads if r), default=len(t))
        out[(m["text"], m["pi"], m["seed"])] = {
            "official": bool(m.get("hit_sfx")) and bool(m.get("hit_vl")),
            "contained": any(t in r for r in reads),
            "exact": any(r == t for r in reads),
            "le1": len(t) > 2 and best <= 1,
            "repeat": len(t) == 1 and any(r.count(t) >= 2 for r in reads),
            "line": len(t) == 1 and any(len(GLYPH.findall(r)) >= 3 for r in reads),
            "wdup": len(t) > 1
            and any(mt.group(1) in t for r in reads for mt in DUP.finditer(r)),
        }
    return out


SINGLE_M = ("official", "contained", "repeat", "line")
WORD_M = ("exact", "contained", "le1", "wdup")


def tally(name: str, h: dict) -> dict:
    per: dict = {}
    for (text, _pi, _s), v in h.items():
        c = per.setdefault(text, {"n": 0, **dict.fromkeys(v, 0)})
        c["n"] += 1
        for k, x in v.items():
            c[k] += x
    tot = {}
    for grp, ms in (("singles", SINGLE_M), ("words", WORD_M)):
        sub = {t: c for t, c in per.items() if (len(t) == 1) == (grp == "singles")}
        if not sub:
            continue
        tot[grp] = {k: sum(c[k] for c in sub.values()) for k in ("n", *ms)}
        print(
            f"  {name:<12} {grp.upper():<8} "
            + "  ".join(f"{k} {tot[grp][k]:>3}" for k in ms)
            + f" / {tot[grp]['n']}",
            flush=True,
        )
    return {"per_key": per, "total": tot}


def paired(a: dict, b: dict, ms) -> dict:
    from math import comb

    ks = sorted(set(a) & set(b))
    out = {"n": len(ks)}
    for m in ms:
        g = sum(a[x][m] and not b[x][m] for x in ks)
        lo = sum(b[x][m] and not a[x][m] for x in ks)
        n = g + lo
        p = (
            min(1.0, 2 * sum(comb(n, i) for i in range(min(g, lo) + 1)) / 2**n)
            if n
            else 1.0
        )
        out[m] = [g, lo, float(f"{p:.2g}")]
    return out


def score(arms) -> dict:
    singles, words = keys()
    sub = Path(f"native_{TAG}") / "native_reads.json"
    src = {"floor": floor_dir() / sub, "mode_floor": MODE_FLOOR / sub}
    src.update({a: arm_dir(a) / sub for a in arms})
    if BUDGET > 1:  # the same recipes at the line's budget, read on disk
        src.update({f"{a}x1": arm_dir(a, 1) / sub for a in arms})
    H = {}
    for name, f in src.items():
        if f.exists():
            H[name] = hits(f, singles + words)
    out = {n: tally(n, h) for n, h in H.items()}
    for n in arms:
        if n not in H:
            continue
        refs = ("floor", "mode_floor", f"{n}x1", *(a for a in arms if a < n))
        for ref in refs:
            if ref not in H:
                continue
            s = {k: v for k, v in H[n].items() if len(k[0]) == 1}
            w = {k: v for k, v in H[n].items() if len(k[0]) > 1}
            rs = {k: v for k, v in H[ref].items() if len(k[0]) == 1}
            rw = {k: v for k, v in H[ref].items() if len(k[0]) > 1}
            pr = {}
            if rs:
                pr["singles"] = paired(s, rs, SINGLE_M)
            if rw:
                pr["words"] = paired(w, rw, WORD_M)
            out[n][f"paired_vs_{ref}"] = pr
            print(f"  {n} vs {ref}: {pr}", flush=True)
    return out


def main():
    args = parse_args()
    set_budget(args.budget)
    ids = check_encodings(SB.encoder())
    print(f"micro {MICRO}: ids {json.dumps(ids, ensure_ascii=False)}", flush=True)
    for a in args.arms:
        for g in table(a):
            print(
                f"{a} {g.name} σ {g.band} share {g.share}: "
                f"{[(t.recipe, t.weight, t.params) for t in g.tiers]}",
                flush=True,
            )
    metrics: dict = {
        "micro": MICRO,
        "words": list(WORDS),
        "ids": ids,
        "dose": DOSE,
        "budget": args.budget,
    }
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "stage_i", label=args.label, root=LINE / "experiments" / "stage_i" / "results"
    )
    singles, words = keys()
    if "data" in args.legs:
        from cjk_scale.builder import build

        for a in args.arms:
            build(rc(a), workers=args.workers, table=table(a))
    if "train" in args.legs:
        from cjk_scale.train import train

        for a in args.arms:
            train(rc(a))
    if "mode" in args.legs:
        for a in args.arms:
            print(f"mode arm {mode_arm(a)}", flush=True)
    if "floor" in args.legs:
        from cjk_scale.eval import ensure_native_floor

        n = ensure_native_floor(rc(args.arms[0]), f"native_{TAG}", singles + words, "en")
        print(f"floor: {n} keys rendered into native_{TAG}/", flush=True)
        native_read(MODE_FLOOR, MODE_FLOOR / "data", words)
    if "read" in args.legs:
        for a in args.arms:
            p = arm_dir(a)
            native_read(p, OUT / run_name(a) / "data", singles + words)
    if {"floor", "read", "score"} & set(args.legs):
        metrics["score"] = score(args.arms)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / run_name(a)) for a in args.arms]
        + [str(arm_dir(a)) for a in args.arms],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
