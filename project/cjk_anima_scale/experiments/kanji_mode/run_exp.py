#!/usr/bin/env python
"""kanji_mode — proposal.md § 2.3's new open item: does ``v_line`` compose
warm kanji rows? (2026-09-27)

Stage I's cold kanji (90 / 270 steps / row) never composed under
0.5 · ``v_line`` (≥ 2 glyphs in order ≤ 3 / 96,
``reports/stage_i_2026_09_26.md`` § 5), while the seed's kana did (≤ 1 edit
11 → 80 / 160). Two readings: kanji vs kana, or row maturity.

This read takes kanji whose rows are the seed's ``step1_0921`` base rows —
the same run, recipe (singles + grids, no spelled line) and steps as the
seed's kana that compose — and spells six words of them. Training-free:
the floor (the seed rows) and ``tf_l1_line0.5`` (the seed rows +
0.5 · ``v_line``), ``en`` only.

- composes (≤ 1 edit / contained well above the floor) → Stage I's gap was
  row maturity; cold kanji need the seed's budget, and the kana-trained
  mode carries over to kanji.
- does not → kanji vs kana; the next arm is a kanji-trained line mode.

The kana held-out words (Stage B ``HELD_WORDS``, already on disk in
``native_spell/`` for both arms) are scored beside them as the reference.

Legs:
  floor  (GPU) the words' floor keys into the seed dir's ``native_kmode/``
  read   (GPU) the words on ``tf_l1_line0.5``
  score  (CPU) the tables from the reads on disk

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

_spec = importlib.util.spec_from_file_location(
    "stage_i_exp", LINE / "experiments" / "stage_i" / "run_exp.py"
)
SI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SI)  # runs bootstrap(); SI.SB = stage_b
from bench._common import make_run_dir, write_result  # noqa: E402

BASE_ROWS = OUT.parents[0] / "wake_probe" / "rows_step1_0921_s30k" / "trained.pt"
WORDS = ("大丈夫", "日本人", "美少女", "何時間", "先輩", "最高")
TAG = "kmode"
SI.TAG = TAG  # native_read writes into native_<TAG>/


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["score"], choices=["floor", "read", "score"]
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def check_encodings(ext) -> dict:
    """Every glyph is one ext id with a ``step1_0921`` base row; every spelled
    word encodes to its glyphs' ids."""
    import torch

    base = {
        int(e)
        for e in torch.load(BASE_ROWS, map_location="cpu", weights_only=False)[
            "delta"
        ]["ext_ids"]
    }
    ids = {}
    for w in WORDS:
        assert len(set(w)) == len(w), w
        for c in w:
            v = ext(c)
            assert len(v) == 1 and v[0] in base, (c, v)
            ids[c] = v[0]
        assert ext(SI.spell(w)) == [ids[c] for c in w], (w, ext(SI.spell(w)))
    return ids


def loose(path: Path, words) -> dict:
    """Stage I § 5's loose word reads: the first glyph in a read, ≥ 2 glyphs
    in order, the mean share of the word's glyphs read."""
    from common.readers import norm

    out = {"n": 0, "first": 0, "ge2_order": 0, "share": 0.0}
    for m in json.loads(path.read_text("utf-8")):
        if m["clause"] != "en" or m["text"].replace(" ", "") not in words:
            continue
        t = norm(m["text"].replace(" ", ""))
        reads = [
            norm(r.get(x) or "") for r in m.get("reads", []) for x in ("sfx", "vl")
        ]
        reads = [r for r in reads if r]
        out["n"] += 1
        out["first"] += any(t[0] in r for r in reads)

        def in_order(r: str) -> int:  # longest common subsequence
            row = [0] * (len(r) + 1)
            for a in t:
                prev = 0
                for j, b in enumerate(r):
                    cur = row[j + 1]
                    row[j + 1] = prev + 1 if a == b else max(row[j + 1], row[j])
                    prev = cur
            return row[-1]

        out["ge2_order"] += any(in_order(r) >= 2 for r in reads)
        out["share"] += max((len(set(t) & set(r)) / len(t) for r in reads), default=0)
    if out["n"]:
        out["share"] = round(out["share"] / out["n"], 3)
    return out


def score() -> dict:
    words = [SI.spell(w) for w in WORDS]
    kana = [SI.spell(w) for w in SI.SB.HELD_WORDS]
    src = {
        "floor": floor_dir() / f"native_{TAG}" / "native_reads.json",
        "mode": SI.MODE_FLOOR / f"native_{TAG}" / "native_reads.json",
    }
    kana_src = {
        "floor": floor_dir() / "native_spell" / "native_reads.json",
        "mode": SI.MODE_FLOOR / "native_spell" / "native_reads.json",
    }
    out: dict = {}
    H = {}
    for grp, keys, srcs in (("kanji", words, src), ("kana", kana, kana_src)):
        out[grp] = {}
        for arm, f in srcs.items():
            if not f.exists():
                print(f"  {grp} {arm}: no reads at {f}", flush=True)
                continue
            h = SI.hits(f, keys)
            H[(grp, arm)] = h
            out[grp][arm] = SI.tally(f"{grp}:{arm}", h)
            lo = loose(f, {k.replace(" ", "") for k in keys})
            out[grp][arm]["loose"] = lo
            print(f"  {grp}:{arm:<6} loose {lo}", flush=True)
        if (grp, "mode") in H and (grp, "floor") in H:
            pr = SI.paired(H[(grp, "mode")], H[(grp, "floor")], SI.WORD_M)
            out[grp]["paired_mode_vs_floor"] = pr
            print(f"  {grp} mode vs floor: {pr}", flush=True)
    return out


def main():
    args = parse_args()
    ids = check_encodings(SI.SB.encoder())
    print(f"words {WORDS}: ids {json.dumps(ids, ensure_ascii=False)}", flush=True)
    metrics: dict = {"words": list(WORDS), "ids": ids, "dose": SI.DOSE}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "kanji_mode", label=args.label, root=LINE / "experiments" / "kanji_mode" / "results"
    )
    words = [SI.spell(w) for w in WORDS]
    if "floor" in args.legs:
        from cjk_scale.config import RunConfig
        from cjk_scale.eval import ensure_native_floor

        rc = RunConfig(name="kanji_mode", path=Path(__file__), vocabs=(), read=())
        n = ensure_native_floor(rc, f"native_{TAG}", words, "en")
        print(f"floor: {n} keys rendered into native_{TAG}/", flush=True)
    if "read" in args.legs:
        SI.native_read(SI.MODE_FLOOR, SI.MODE_FLOOR / "data", words)
    metrics["score"] = score()
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(floor_dir() / f"native_{TAG}"), str(SI.MODE_FLOOR / f"native_{TAG}")],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
