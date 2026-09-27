#!/usr/bin/env python
"""doubling_box — proposal.md § 2.1 (a): what in-word doubling (B) is (2026-09-26)

CPU only, reads on disk. Every spelled-word render of the arms below
(``native_spell/native_reads.json``), per render:

- the **line box**: the non-whole reader box whose read (sfx or vl) is
  closest to the word by edit distance (the whole-image read only when no
  box was found), its glyphs read (kana / kanji), px = √(box area / glyphs),
  orientation (h > w: vertical), and slots = long side / px;
- ``dup`` exactly as ``stage_b.hits`` counts it (any read holds a doubled
  glyph), and for a doubled render the **dup read** (the line box's read if
  it doubles, else the first read that does): which glyph doubled, its
  position in the word (first / inner / last / foreign), the glyph before it,
  and whether collapsing the run gives the word back (pure padding).

The two hypotheses (proposal § 2.1):
- **H1, a fill prior** — the line fills its box at the trained px, and a word
  shorter than the box pads. Predicts: doubled renders sit at the trained
  line px (Stage B's items, ``px`` in ``train.jsonl``), in longer boxes than
  the clean renders of the same word (slots − length > 0), and ``dup``
  follows the scene prompt (which sets the box), not the glyph.
- **H2, a sequence failure** — the mode is on, the order is weak. Predicts:
  box and px of doubled renders ≈ clean renders; the doubling concentrates on
  particular positions or bigrams; it is flat over prompts.

``--dry_run`` lists the arms and their render counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, floor_dir  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

EXP = OUT / "experiments"
# arm → its dir; the held-out words (ひまわり さくら みどり くもり まくら) on
# the first seven, the donors' own words on the rest
ARMS = {
    "floor": floor_dir(),
    "u0.5": EXP / "tb_t1_u0.5",
    "u1": EXP / "tb_t1_u1",
    "u2": EXP / "tb_t1_u2",
    "rand1": EXP / "tb_t1_rand1",
    "v_line1": EXP / "tf_l1_line",
    "v_line0.5": EXP / "tf_l1_line0.5",
    "spell_b": OUT / "run0926_spell_b",
    "stage_b_donor": OUT / "run0926_stage_b",
    "f1": OUT / "run0926_f1_line",
}
TRAIN_ITEMS = OUT / "run0926_stage_b" / "data" / "train.jsonl"
GLYPH = re.compile(r"[ぁ-ゟァ-ヿ一-鿿]")
DUP = re.compile(r"(.)\1")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def records(path: Path) -> list[dict]:
    """The word renders (≥ 2 glyphs) of one arm's ``native_spell`` reads."""
    f = path / "native_spell" / "native_reads.json"
    if not f.exists():
        return []
    ms = json.loads(f.read_text("utf-8"))
    return [
        m
        for m in ms
        if len(m["text"].replace(" ", "")) > 1 and m.get("cond", "trained") != "floor"
    ]


def geom(box) -> dict:
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    return {"w": w, "h": h, "area": w * h, "vertical": h > w, "long": max(w, h)}


def per_render(m: dict) -> dict:
    from common.readers import norm

    t = norm(m["text"])
    boxes = [r for r in m.get("reads", []) if not r.get("whole")]
    whole_only = not boxes
    boxes = boxes or list(m.get("reads", []))

    def reads_of(r):
        return [norm(r.get(x) or "") for x in ("sfx", "vl")]

    def best(r):
        rs = [x for x in reads_of(r) if x]
        return min(((_lev(x, t), x) for x in rs), default=(len(t), ""))

    out = {"text": t, "len": len(t), "clause": m["clause"], "pi": m["pi"]}
    out["seed"] = m["seed"]
    all_reads = [x for r in m.get("reads", []) for x in reads_of(r) if x]
    out["dup"] = any(DUP.search(x) for x in all_reads)
    # the class of the double: a glyph of the word (in-word B), another
    # kana / kanji, or reader noise (latin, digits, punctuation)
    dups = [mt.group(1) for x in all_reads for mt in DUP.finditer(x)]
    out["dup_class"] = (
        "word"
        if any(g in t for g in dups)
        else "kana"
        if any(GLYPH.match(g) for g in dups)
        else "noise"
        if dups
        else None
    )
    if not boxes:
        out.update(box=None)
        return out
    line = min(boxes, key=lambda r: best(r)[0])
    dist, read = best(line)
    n = len(GLYPH.findall(read))
    g = geom(line["box"])
    out.update(
        box=line["box"],
        whole_only=whole_only,
        read=read,
        dist=dist,
        n=n,
        px=(g["area"] / n) ** 0.5 if n else None,
        vertical=g["vertical"],
        long=g["long"],
    )
    out["slots"] = g["long"] / out["px"] if out["px"] else None
    if out["dup"]:
        # the doubled read: the line box's if it doubles, else the first one
        def word_dup(x):
            return next((mt for mt in DUP.finditer(x) if mt.group(1) in t), None)

        pick = word_dup if out["dup_class"] == "word" else DUP.search
        cand = [x for x in reads_of(line) if x and pick(x)] or [
            x for x in all_reads if pick(x)
        ]
        dr = cand[0]
        mt = pick(dr)
        gl = mt.group(1)
        i = mt.start()
        if gl in t:
            k = t.index(gl)
            pos = "first" if k == 0 else "last" if k == len(t) - 1 else "inner"
        else:
            pos = "foreign"
        out.update(
            dup_read=dr,
            dup_in_line=bool([x for x in reads_of(line) if x and pick(x)]),
            dup_glyph=gl,
            dup_pos=pos,
            dup_prev=dr[i - 1] if i else "^",
            dup_run=len(re.match(r"(.)\1*", dr[i:]).group(0)),
            padding=re.sub(r"(.)\1+", r"\1", dr) == t,
        )
    return out


def train_px() -> dict:
    """Stage B's word items: px and the fill, per band group."""
    px = defaultdict(list)
    for ln in TRAIN_ITEMS.read_text("utf-8").splitlines():
        it = json.loads(ln)
        if it["recipe"] == "scene_spelled" and it.get("px"):
            px[it["group"]].append(it["px"])
    return {g: quant(v) for g, v in px.items()}


def quant(v: list) -> dict:
    v = sorted(x for x in v if x is not None)
    if not v:
        return {"n": 0}
    q = lambda f: round(v[min(int(f * len(v)), len(v) - 1)], 1)  # noqa: E731
    return {"n": len(v), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9)}


def summarize(rs: list[dict]) -> dict:
    has = [r for r in rs if r.get("px")]
    dup = [r for r in has if r["dup_class"] == "word"]
    clean = [r for r in has if not r["dup"]]
    # paired within word × clause × prompt: dup vs clean renders of the same
    # key, box geometry only (the seeds differ, the scene does not)
    out = {
        "n": len(rs),
        "dup": sum(r["dup"] for r in rs),
        "no_box": sum(1 for r in rs if not r.get("box")),
        "class": dict(Counter(r["dup_class"] for r in rs if r["dup"])),
        "px": {"dup": quant([r["px"] for r in dup]), "clean": quant([r["px"] for r in clean])},
        "slots_minus_len": {
            "dup": quant([r["slots"] - r["len"] for r in dup]),
            "clean": quant([r["slots"] - r["len"] for r in clean]),
        },
        "long": {"dup": quant([r["long"] for r in dup]), "clean": quant([r["long"] for r in clean])},
        "n_minus_len": {
            "dup": quant([r["n"] - r["len"] for r in dup]),
            "clean": quant([r["n"] - r["len"] for r in clean]),
        },
        "vertical": {
            "dup": sum(r["vertical"] for r in dup),
            "clean": sum(r["vertical"] for r in clean),
        },
    }
    d = [r for r in rs if r["dup_class"] == "word"]
    out["pos"] = dict(Counter(r.get("dup_pos") for r in d))
    out["dup_in_line"] = sum(bool(r.get("dup_in_line")) for r in d)
    out["padding"] = sum(bool(r.get("padding")) for r in d)
    out["run_len"] = dict(Counter(r.get("dup_run") for r in d))
    out["glyph"] = dict(Counter(r.get("dup_glyph") for r in d).most_common(8))
    out["bigram"] = dict(
        Counter(f"{r.get('dup_prev')}{r.get('dup_glyph')}" for r in d).most_common(8)
    )
    def by(f):
        return {
            v: [
                sum(r["dup_class"] == "word" for r in rs if r[f] == v),
                sum(1 for r in rs if r[f] == v),
            ]
            for v in sorted({r[f] for r in rs})
        }

    out["by_prompt"], out["by_word"], out["by_clause"] = (
        by("pi"),
        by("text"),
        by("clause"),
    )
    # the within-key contrast: same word × clause × prompt, one seed doubled
    # and one clean — Δ long side and Δ px (dup − clean)
    by_key = defaultdict(list)
    for r in has:
        by_key[(r["text"], r["clause"], r["pi"])].append(r)
    dl, dp = [], []
    for grp in by_key.values():
        a = [r for r in grp if r["dup_class"] == "word"]
        b = [r for r in grp if not r["dup"]]
        if a and b:
            dl.append(median(r["long"] for r in a) - median(r["long"] for r in b))
            dp.append(median(r["px"] for r in a) - median(r["px"] for r in b))
    out["within_key"] = {
        "keys": len(dl),
        "d_long": quant(dl),
        "d_px": quant(dp),
        "longer": sum(x > 0 for x in dl),
        "shorter": sum(x < 0 for x in dl),
    }
    return out


def show(name: str, s: dict) -> None:
    print(
        f"== {name}: {s['dup']} / {s['n']} doubled {s['class']} (no box {s['no_box']})",
        flush=True,
    )
    for k in ("px", "slots_minus_len", "long", "n_minus_len"):
        print(f"  {k:<16} dup {s[k]['dup']}\n  {'':<16} clean {s[k]['clean']}", flush=True)
    print(
        f"  vertical dup {s['vertical']['dup']} clean {s['vertical']['clean']}; "
        f"dup in the line box {s['dup_in_line']}; pure padding {s['padding']}",
        flush=True,
    )
    print(f"  pos {s['pos']}  run {s['run_len']}", flush=True)
    print(f"  glyph {s['glyph']}\n  bigram {s['bigram']}", flush=True)
    print(f"  by word {s['by_word']}\n  by prompt {s['by_prompt']}", flush=True)
    print(f"  by clause {s['by_clause']}\n  within key {s['within_key']}", flush=True)


def main():
    args = parse_args()
    arms = {k: records(v) for k, v in ARMS.items()}
    for k, v in arms.items():
        print(f"{k:<14} {len(v):>4} word renders  ({ARMS[k]})", flush=True)
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "doubling_box",
        label=args.label,
        root=LINE / "experiments" / "doubling_box" / "results",
    )
    tp = train_px()
    print(f"Stage B word items, px by group: {tp}", flush=True)
    metrics: dict = {"train_px": tp, "arms": {}}
    rows = {}
    for k, ms in arms.items():
        rows[k] = [per_render(m) for m in ms]
        metrics["arms"][k] = summarize(rows[k])
        show(k, metrics["arms"][k])
    (run_dir / "renders.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(v) for v in ARMS.values()],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
