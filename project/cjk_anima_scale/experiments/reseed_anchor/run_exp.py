#!/usr/bin/env python
"""reseed_anchor — reseed_recap with the columns lettered as Japanese, and the base's own text beside the rows

`reseed_recap` (the 81 hiragana + 85 katakana rows cold, 135 / row, plain
grid captions, the gradient read's bands) drew its windows with the renderer
every data dir of record used: of its 3 970 column windows, 489 carry a
turned ー 〜 up to 0.2 em off the column axis and 1 356 a small kana drawn as
the horizontal glyph (at the bottom of its cell, where a column has it at
the top right), and 507 of its 5 644 windows open on ー or a small kana
(``ーシングラか``, ``ゅぱ``, ``ォこれ``). `seed_synth`'s redraw fixed the
first for its canvases only (user, 10-01). This arm (user, 10-03) is
`reseed_recap` — rows, budget, table, shares, plain captions, bands — with:

**The lettering** (both variants):
- ``tategaki`` + ``vert_forms`` on the two `bubbleN` tiers
  (`render_into_scene`): a column's ー 〜 and small kana are the font's
  vertical alternates (OpenType ``vert`` through libraqm) — the bar on the
  column axis as the font draws it, the small kana at the top right. A font
  with none (TanukiMagic, 1 of 15) keeps the turned bar, centred, its small
  kana nudged.
- no window opens on a small kana or ー (``scene.NO_HEAD``, `seed_synth`'s
  swap rule), dropped from the pool before the draw.

**The base's text** (``--variant anchor``, the default; ``fix`` = the
lettering alone, the control against `reseed_recap`):
- ``MARK_FRAC`` of the window draws take a window that a ！ / ？ closes in
  the corpus (the run's last 2–6 glyphs before the mark, the pool's rules),
  with the mark: drawn in the bubble, written in the caption. The pack's
  encode fold sends ！ ？ to the base's ``!`` ``?``, so the slot after the
  word is the base's to fill (`findings.md` § 2: こんにちは！ 5 / 8 official
  against 2 / 8, the repeats gone). A glyph with no such window keeps a
  plain one.
- ``EN_FRAC`` of the multi-cell grids give one cell, at random, to an EN
  word (`sigma_split`'s ``EN_GRID_POOL``, 36 words the base writes in a 3 × 3
  at recall 0.84 — `findings.md` § 6), a line at the cells' font px; the
  deck deals one glyph fewer. Its clause is the cell's plain one
  (``On the middle, text reads as "SNOW".``); the cell is outside the
  item's kind and px (the band is the kana cells').

Both are text the base writes from its own rows, frozen here, in the items
the cold rows train on — does the base's own text beside them raise what
the rows learn (user)? One arm carries both, so a difference from ``fix``
is theirs together.

Legs, as `reseed_recap`'s:
- ``data`` (CPU) → ``OUT/run1003_reseed_anchor[_fix]/data``;
- ``recap`` (CPU) → ``…/data_recap`` (plain captions, the law's bands) and
  ``…/data_recap_hp`` (`reseed_recap`'s ``hp`` bands);
- ``train`` (GPU) → ``OUT/experiments/reseed_anchor_cold_kana_<variant>``;
- ``read`` (GPU): `reseed_recap`'s (`en` / `swap` against ``retrain_kana``);
- ``read_plain`` (GPU): the plain clause on the kana run's 13 words + 14
  singles, this arm against `reseed_recap`'s ``hp`` arm and ``retrain_kana``
  (both cached).

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      .venv/bin/python project/cjk_anima_scale/experiments/reseed_anchor/run_exp.py \\
      --label anchor_data --legs data recap
    … make daemon-run ARGS="…/reseed_anchor/run_exp.py --label anchor \\
      --legs train read read_plain"
    # the lettering alone
    … --label fix --variant fix --legs data recap train read read_plain
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale import paths  # noqa: E402
from cjk_scale.paths import OUT, SEED_ROWS_0921, bootstrap, load_experiment  # noqa: E402

bootstrap()
paths.pin_old_seed()  # the kana run's seed: the build's context and the floor
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "reseed_anchor"
ARM = f"{NAME}_cold_kana"
MARK_FRAC = 0.3  # of the window draws: a window a ！ / ？ closes, with the mark
EN_FRAC = 0.5  # of the multi-cell grids: one cell an EN word
MARKS = {"！": "！", "!": "！", "？": "？", "?": "？"}  # corpus mark → the drawn one
LETTERING = {"tategaki": True, "vert_forms": True}


def table(base: tuple, lone: str, variant: str, en_words: list) -> tuple:
    """`reseed_recap`'s table with the lettering on its `bubbleN` tiers and,
    for ``anchor``, the marks on them and the EN cell on its multi-cell grids."""
    anchor = variant == "anchor"
    out = []
    for g in base:
        ts = []
        for t in g.tiers:
            extra: dict = {}
            if t.recipe == "bubbleN":
                extra = LETTERING | ({"mark_frac": MARK_FRAC} if anchor else {})
            elif t.recipe == "grid" and t.params["grids"] != lone and anchor:
                extra = {"en_frac": EN_FRAC, "en_words": list(en_words)}
            ts.append(dataclasses.replace(t, params=t.params | extra))
        out.append(dataclasses.replace(g, tiers=tuple(ts)))
    return tuple(out)


def marked_windows(glyphs: set, lines, held, length: tuple, heads: set) -> list:
    """Every run of ``glyphs`` in ``lines`` that a mark closes: its last
    ``length`` glyphs under `recipes.window_pool`'s rules (no repeated
    glyph, no trigram of a ``held`` string), not opening on ``heads``, with
    the mark (one, full width)."""
    grams = set()
    for h in held:
        n = min(3, len(h))
        grams |= {h[i : i + n] for i in range(len(h) - n + 1)}
    lo, hi = length
    out = set()
    for ln in lines:
        run = ""
        for c in ln + "\n":
            if c in glyphs:
                run += c
                continue
            for n in range(lo, min(hi, len(run)) + 1) if c in MARKS else ():
                w = run[-n:]
                if (
                    len(set(w)) == n
                    and w[0] not in heads
                    and not any(g in w for g in grams)
                ):
                    out.add(w + MARKS[c])
            run = ""
    return sorted(out)


def prepare(variant: str):
    """`builder.build`'s ``prepare``: the head rule on the window pool, and
    (``anchor``) the marked windows on ``pools.marked``."""

    def fn(pools, rc) -> dict:
        from cjk_scale.config import dataset_ja_lines, phrase_file
        from cjk_scale.recipes import WINDOW_LEN, ext_encoder, window_glyphs
        from common.render.scene import NO_HEAD

        n0 = len({w for ws in pools.windows.values() for w in ws})
        keys0 = set(pools.windows)
        pools.windows = {
            g: kept
            for g, ws in pools.windows.items()
            if (kept := [w for w in ws if w[0] not in NO_HEAD])
        }
        pools.window_keys = [k for k in pools.window_keys if k in pools.windows]
        ok = {w for ws in pools.windows.values() for w in ws}
        n = sorted(len(v) for v in pools.windows.values())
        stats: dict = {
            "variant": variant,
            "head_rule": {
                "windows": len(ok),
                "dropped": n0 - len(ok),
                "glyphs_lost": sorted(keys0 - set(pools.windows)),
                "per_glyph_min": n[0],
                "per_glyph_median": n[len(n) // 2],
            },
        }
        if variant == "anchor":
            lines = [
                ln.split("\t")[0]
                for ln in Path(phrase_file()).read_text(encoding="utf-8").splitlines()
            ]
            ws = marked_windows(
                window_glyphs(pools.singles),
                lines + dataset_ja_lines(),
                rc.read,
                WINDOW_LEN,
                NO_HEAD,
            )
            ext = ext_encoder()
            # the bare window passed the routing check; the mark adds no row
            kept = [w for w in ws if w[:-1] in ok and ext(True, w) == ext(True, w[:-1])]
            pools.marked = {
                g: v for g in pools.windows if (v := [w for w in kept if g in w])
            }
            m = sorted(len(v) for v in pools.marked.values())
            stats["marked"] = {
                "mark_frac": MARK_FRAC,
                "windows": len(kept),
                "dropped": len(ws) - len(kept),
                "by_mark": dict(Counter(w[-1] for w in kept)),
                "by_length": dict(sorted(Counter(len(w) - 1 for w in kept).items())),
                "glyphs": len(pools.marked),
                "glyphs_without": sorted(set(pools.windows) - set(pools.marked)),
                "per_glyph_min": m[0],
                "per_glyph_median": m[len(m) // 2],
            }
        print(f"prepare: {json.dumps(stats, ensure_ascii=False)}", flush=True)
        return stats

    return fn


def composition(data: Path) -> dict:
    """What the build drew of the lettering and the base's text, per tier."""
    from common.render.scene import NO_HEAD, V_ROTATE, V_SMALL

    recs = [
        json.loads(ln) for ln in (data / "train.jsonl").read_text("utf-8").splitlines()
    ]
    out: dict = {}
    for r in recs:
        t = out.setdefault(r["tier"], Counter())
        t["items"] += 1
        if r["recipe"] == "bubbleN":
            w = r["text"]
            t["column"] += not r["horizontal"]
            t["marked"] += w[-1] in MARKS
            t["head"] += w[0] in NO_HEAD
            t["column_turned"] += not r["horizontal"] and bool(set(w) & V_ROTATE)
            t["column_small"] += not r["horizontal"] and bool(set(w) & V_SMALL)
        elif r.get("base_cells"):
            t["en_grids"] += 1
            t[f"en_{r['grid']}"] += 1
        if r["layout"] == "grid":
            t["cells"] += len(r["units"])
            t["kana_cells"] += len(r["units"]) - len(r.get("base_cells", ()))
    return {k: dict(v) for k, v in sorted(out.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "recap", "train", "read", "read_plain"],
    )
    p.add_argument("--variant", default="anchor", choices=["anchor", "fix"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    GS = load_experiment("grid_small")
    GL = load_experiment("grid_lone")
    RR = load_experiment("reseed_recap")
    en_words = list(load_experiment("sigma_split").EN_GRID_POOL)
    data_run = f"run1003_{NAME}" + ("" if args.variant == "anchor" else "_fix")
    rc, rows = RR.run_config()
    rc = dataclasses.replace(rc, name=data_run)
    tbl = table(RR.table(GS, GL), GL.LONE, args.variant, en_words)
    arm = f"{ARM}_{args.variant}"
    root = OUT / data_run
    data_dir = root / "data_recap_hp"
    steps = RR.STEPS_PER_ROW * len(rows)
    shares = {g.label: g.share for g in tbl}
    KR = load_experiment("kana_reband")
    budget = KR.check_trainer(RR.STEPS_PER_ROW)
    print(
        f"{arm}: {len(rows)} rows (hiragana 81 + katakana 85) cold × "
        f"{RR.STEPS_PER_ROW} = {steps} steps on {SEED_ROWS_0921}; groups {shares} "
        f"(Σ {sum(shares.values()):g}); lettering {LETTERING} + the head rule"
        + (
            f"; marks {MARK_FRAC} of the windows, an EN word in {EN_FRAC} of the "
            f"multi-cell grids ({len(en_words)} words)"
            if args.variant == "anchor"
            else ""
        )
        + f"; data {data_dir}; {budget}",
        flush=True,
    )
    if args.dry_run:
        return
    metrics: dict = {
        "rows": len(rows),
        "steps": steps,
        "shares": shares,
        "arm": arm,
        "variant": args.variant,
        **budget,
    }
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        metrics["data"] = GS.data(
            rc,
            rows,
            args.workers,
            tbl,
            max_px=None,
            prepare=prepare(args.variant),
            base_chars="".join(sorted(set(MARKS.values()) | set("".join(en_words)))),
        )
        metrics["composition"] = composition(root / "data")
        print(json.dumps(metrics["data"], ensure_ascii=False, indent=1), flush=True)
        print(
            json.dumps(metrics["composition"], ensure_ascii=False, indent=1),
            flush=True,
        )
    if "recap" in args.legs:
        metrics["derive"] = GS.derive(root / "data", root / "data_recap", None, True)
        metrics["reband"] = GL.reband(
            root / "data_recap", data_dir, GL.BANDS["recap_hp"] | RR.BANDS_44
        )
        print(json.dumps(metrics["reband"], ensure_ascii=False, indent=1), flush=True)
    if "train" in args.legs:
        from cjk_scale import train as T

        T.train(
            dataclasses.replace(rc, name=arm),
            data=data_dir,
            out=OUT / "experiments" / arm,
            cold=True,
            steps_per_row=RR.STEPS_PER_ROW,
            context=SEED_ROWS_0921,
        )
    if "read" in args.legs:
        metrics["read"] = RR.read(arm)
    if "read_plain" in args.legs:
        arms = {
            arm: OUT / "experiments" / arm,
            f"{RR.ARM}_hp": OUT / "experiments" / f"{RR.ARM}_hp",
            RR.SRC_RUN: OUT / RR.SRC_RUN,
        }
        other = "fix" if args.variant == "anchor" else "anchor"
        if (OUT / "experiments" / f"{ARM}_{other}" / "trained.pt").exists():
            arms[f"{ARM}_{other}"] = OUT / "experiments" / f"{ARM}_{other}"
        metrics["read_plain"] = GL.read_plain(arms, "all")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(data_dir), str(OUT / "experiments" / arm)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
