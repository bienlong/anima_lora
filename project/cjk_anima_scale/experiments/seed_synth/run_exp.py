#!/usr/bin/env python
"""seed_synth — the seed's own renders as canvases, a first look (proposal_seed_synthesis step 1, small)

`proposal_seed_synthesis.md` § Proposal at ``--n`` (300) renders instead of
3 000: what the seed rows draw for a held-out word on a scene prompt outside
the ``sent`` grid, how many of those renders the selection keeps, and what a
kept render looks like with its banner redrawn. No training.

- **Canvas.** The seed rows (routed), 28 steps, cfg 4, on the scene pools'
  prompt stream (``scenes.stage.scene_items``: rating, count, character,
  @artist, generals) **with its text frame kept** — ``s1w``'s four frames
  (``bubble_reads`` / ``reads_as`` / ``saying`` / ``sign``) and six canvases
  (448 × 640 … 640 × 384) — and the caption the seed trained on
  (``data.synth.scene_caption``: ``english text`` → ``japanese text``, the
  frame's clause with ``X`` in its quote, ``HORIZONTAL`` of them marked
  horizontal). The text has a bubble or a sign to sit in, so a redraw erases
  a flat fill and not the scene (user, 2026-10-01: the first look, the
  ruler's bare ``{p}, japanese text. Japanese text reads as "X".`` at 512²,
  drew the word over the background — ``seed_synth_nobubble_partial/``).
  ``X`` = a ``scene_window`` text of ``retrain_kana``'s data
  (the seed's windowed word pool, kana), ``--lens`` glyphs in equal shares,
  no punctuation, no doubled glyph, no ``sent`` / ``target`` / run ``read``
  string nor a trigram of one (the hold rule).
- **Selection.** Every render is read (sfx + VL). The main box is the CJK box
  whose read, doubled glyphs collapsed, is nearest ``X`` (the larger on a
  tie). ``exact`` = both readers read ``X``; ``near`` = not exact, the main
  box within ``min(2, n − 1)`` edits (a 2-glyph word within 2 edits is every
  2-glyph read) and no other CJK box > 1.5 × its area; ``second_box`` = near
  but for that rule; ``far`` / ``no_text`` the rest. ``near`` are the items.
- **Redraw.** The line's scene renderer (``common.render.scene.render_into_scene``,
  what the seed's ``scene_window`` items were drawn with) on each item: the
  main box's bubble / sign is found (``scenes.judge.bubble_region``; none →
  dropped, the text sits on the scene), the box is erased **whole**, inside
  the bubble interior, and ``X`` is drawn as **one line, or one column — two
  columns, top-aligned and unequal (3 + 2, 4 + 2: the first the longer), from
  ``TWO_COLS`` glyphs when one column would be under ``SPLIT`` × the model's
  glyph** (user, 2026-10-01: `かっ / てる / なら` is not how a word is
  written; a column break keeps the first glyphs level; two equal columns
  read as a grid, so `えガン / グロだ` is one column). The model's layout is
  the rows × columns grid that gives its read the squarest cells, a single
  row / column whenever its cells are within 2 : 1. One row / one column: ``X`` in the same orientation, fitted
  into the model's own text extent — the box, its cross axis allowed ``GROW``
  × (one leftover slot of six is × 1.2). A grid (the model split the word):
  a line and a column are both tried, about the model's text centre, along
  the bubble's usable rectangle, the glyph at most the model's cell; the
  larger wins, the caption's orientation on a near tie. Every draw is shrunk
  (``SHRINK``) until its ink keeps ``MARGIN`` of a glyph clear of the bubble
  outline, or the item is dropped. Whatever else the bubble held is wiped:
  its whole interior outside the drawn box, and — the renderer's erase stops
  at the interior, and a glyph touching the outline is not interior (`り` of
  `りで / ああそそこま` stayed) — the model's text box inside the interior
  closed over a glyph's stroke.
  Read again; ``kept`` = the canvas reads within 1 edit. (``inject_count.redraw``,
  the first look's, erases only the new line's band: a two-line bubble kept
  its old text, 95 of 175 items.)

Outputs under ``OUT/experiments/seed_synth/``: ``render/`` and ``canvas/``
(``img/`` + ``native_reads.json``), ``items.jsonl``, ``sheet_all_<k>.png``
(every render: X | class | the main read), ``sheet_items.png`` (``--sheet_n``
items, render with its main box | canvas). Renders and reads are reused when
present.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/seed_synth/run_exp.py \\
      --label look0"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics as st
import sys
import time
import types
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "seed_synth"
ROOT = OUT / "experiments" / NAME
WORD_RUN = "retrain_kana"  # its scene_window texts = the seed's kana word pool
CHAIN = (
    "retrain_kana",
    "retrain_kanji_b1",
    "retrain_kanji_b2",
    "retrain_kanji_b3",
    "retrain_kanji_b4",
)
FRAMES = "bubble_reads,reads_as,saying,sign"  # scenes_s1w's
SHAPES = "448x640,384x640,640x448,576x448,448x576,640x384"  # scenes_s1w's (W x H)
HORIZONTAL = 0.3  # the builder's horizontal_frac: captions marked horizontal
PUNCT = set("、。「」〜！？～・…")
STREAM_SEED = 1001  # the prompt stream's and the word draw's
AREA_RULE = 1.5  # no other CJK box larger than this × the main box
GROW = 1.5  # the fit region's cross axis / the model's text box
MIN_GLYPH = 12
TWO_COLS = 5  # glyphs from which a column may break into two (3 + 2, 4 + 2)
SPLIT = 0.75  # … when one column's glyph is under this × the model's cell
SHRINK = (1.0, 0.88, 0.76, 0.65)  # the fit region's fill, until the ink is inside
MARGIN = 0.12  # of a glyph, kept clear between the ink and the bubble outline
CJK = re.compile(r"[ぁ-ヿ一-鿿]")


def held() -> set:
    """Every string a ruler reads: the ``sent`` floor's, the chain runs'
    ``read``, the ``target`` captions' quoted spans."""
    from cjk_scale.config import load_run
    from common.prompts import TARGET_PROMPTS

    SS = load_experiment("sigma_split")
    out = {it["text"] for it in SS.floor_items()}
    for r in CHAIN:
        out |= set(load_run(r).read)
    for ln in TARGET_PROMPTS.read_text("utf-8").splitlines():
        if not ln.startswith("#"):
            out |= set(re.findall(r'"([^"]+)"', ln))
    return out


def word_pool(lens: list) -> tuple[dict, dict]:
    """``{n: [texts]}`` over the seed's kana windows, and why the rest dropped."""
    grams = set()
    for h in held():
        n = min(3, len(h))
        grams |= {h[i : i + n] for i in range(len(h) - n + 1)}
    texts = set()
    for ln in (OUT / WORD_RUN / "data" / "train.jsonl").read_text("utf-8").splitlines():
        if ln:
            r = json.loads(ln)
            if r["recipe"] == "scene_window":
                texts.add(r["text"])
    why, pool = Counter(), {n: [] for n in lens}
    for t in sorted(texts):
        if len(t) not in pool:
            why["length"] += 1
        elif any(c in PUNCT for c in t):
            why["punctuation"] += 1
        elif any(a == b for a, b in zip(t, t[1:])):
            why["doubled glyph"] += 1
        elif any(g in t for g in grams):
            why["held trigram"] += 1
        else:
            pool[len(t)].append(t)
    return pool, dict(why)


def scene_stream(n: int, seed: int) -> list[dict]:
    """``n`` scenes of the pools' prompt stream (frame, canvas and tags)."""
    from scenes.stage import scene_items

    a = types.SimpleNamespace(
        seed=seed,
        scene_n=n,
        scene_shapes=SHAPES,
        scene_anchors="",
        scene_frames=FRAMES,
        scene_ja_anchors="",
        scene_extra_tags="",
        scene_bubble_tag="speech bubble",
        scene_char_frac=0.3,
        scene_artist_frac=0.8,
        scene_artists="sincos,hews",
    )
    return scene_items(a)


def plan(n: int, lens: list, seed: int) -> tuple[list, dict]:
    pool, why = word_pool(lens)
    rng = random.Random(seed)
    words = []
    for k in lens:
        words += rng.sample(pool[k], -(-n // len(lens)))
    rng.shuffle(words)
    words = words[:n]
    from data.synth import scene_caption

    items = []
    for i, (sc, w) in enumerate(zip(scene_stream(n, seed), words)):
        hz = rng.random() < HORIZONTAL
        items.append(
            {
                "pi": i,
                "seed": seed * 100_000 + i,
                "prompt": sc["tags"],
                "text": w,
                "clause": "en",
                "caption": scene_caption(sc, w, horizontal=hz),
                "frame": sc["frame"],
                "shape": sc["shape"],  # (W, H)
                "horizontal": hz,
                "file": str(ROOT / "render" / "img" / f"r_{i:04d}_{w}.png"),
                "cond": "render",
            }
        )
    return items, {
        "pool": {k: len(v) for k, v in pool.items()},
        "dropped": why,
        "glyphs": dict(sorted(Counter(len(it["text"]) for it in items).items())),
        "frames": dict(Counter(it["frame"] for it in items)),
        "shapes": dict(Counter("x".join(map(str, it["shape"])) for it in items)),
        "horizontal": sum(it["horizontal"] for it in items),
    }


def render(items: list) -> str:
    if all(Path(it["file"]).exists() for it in items):
        print("renders reused", flush=True)
        return "cuda"
    SS = load_experiment("sigma_split")
    sp = SS.Splitter()
    t0 = time.time()
    for n, it in enumerate(items):
        sp.render(Path(it["file"]), it, "seed", "seed", 0.5)  # seed rows throughout
        if n % 20 == 0:
            print(
                f"  render {n}/{len(items)} · {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
    device = sp.device
    sp.free()
    return device


def read(manifest: list, out: Path, device: str, reuse: bool = True) -> list:
    """``manifest`` with ``reads`` / ``hit_*``; the stored reads are reused
    when they cover the same files (``reuse``: never for the canvases, whose
    pixels change with the redraw)."""
    from common.readers import Readers, hit, read_scored

    f = out / "native_reads.json"
    if reuse and f.exists():
        old = json.loads(f.read_text("utf-8"))
        if [m["file"] for m in old] == [m["file"] for m in manifest]:
            print(f"reads reused: {f}", flush=True)
            return old
    rd = Readers(device)
    t0 = time.time()
    for n, m in enumerate(manifest):
        reads = read_scored(rd, m)
        m["hit_sfx"], m["hit_vl"] = (
            hit(reads, m["text"], "sfx"),
            hit(reads, m["text"], "vl"),
        )
        if n % 20 == 0:
            print(
                f"  read {n}/{len(manifest)} · {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
    del rd
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def _area(r: dict) -> float:
    return (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])


def classify(m: dict) -> dict:
    """The render's class and its main box (module docstring § Selection)."""
    from common.text import lev, norm

    t = norm(m["text"])
    boxes = [
        r
        for r in m["reads"]
        if not r.get("whole")
        and r.get("box")
        and CJK.search((r.get("sfx") or "") + (r.get("vl") or ""))
    ]
    if not boxes:
        return {"cls": "no_text"}

    def edits(r, collapse=True):
        rd = [norm(r.get(x) or "") for x in ("sfx", "vl")]
        if collapse:
            rd = [re.sub(r"(.)\1+", r"\1", x) for x in rd]
        return min(lev(x, t) for x in rd)

    b = min(boxes, key=lambda r: (edits(r), -_area(r)))
    x0, y0, x1, y1 = b["box"]
    out = {
        "box": [int(v) for v in b["box"]],
        "read": b.get("sfx") or b.get("vl"),
        "edits": edits(b),
        "dup": any(re.search(r"(.)\1", norm(b.get(x) or "")) for x in ("sfx", "vl")),
        "long": max(x1 - x0, y1 - y0),
        "short": min(x1 - x0, y1 - y0),
        "vertical": (y1 - y0) > (x1 - x0),
        "boxes": len(boxes),
    }
    if m["hit_sfx"] and m["hit_vl"]:
        cls = "exact"
    elif out["edits"] > min(2, len(t) - 1):
        cls = "far"
    elif any(_area(r) > AREA_RULE * _area(b) for r in boxes):
        cls = "second_box"
    else:
        cls = "near"
    return out | {"cls": cls}


def drawn_grid(w: float, h: float, glyphs: int) -> tuple[int, int]:
    """``(rows, columns)`` of the grid that holds ``glyphs`` in a ``w`` × ``h``
    box with the squarest cells."""
    best = None
    for rows in range(1, min(glyphs, 4) + 1):
        cols = -(-glyphs // rows)
        d = abs(math.log((w / cols) / (h / rows)))
        if best is None or d < best[0]:
            best = (d, rows, cols)
    # a line's cells are not square (pitch, a bold face, a tight detector
    # box): one row / one column wins whenever its cells are within 2 : 1
    line = min(
        (abs(math.log((w / glyphs) / h)), 1, glyphs),
        (abs(math.log(w / (h / glyphs))), glyphs, 1),
    )
    if line[0] <= math.log(2):
        best = line
    return best[1], best[2]


def redraw(m: dict, dst: Path, rng: random.Random) -> dict:
    """``m``'s render with its main box erased and ``m['text']`` drawn in its
    bubble (module docstring § Redraw); the drawn geometry, or ``{"why": …}``
    when there is no bubble or the word does not fit inside it."""
    import cv2
    import numpy as np
    from PIL import Image

    from common.bubble import bubble_interior, bubble_mask, ring_median
    from common.readers import load_bgr
    from common.render.flat import find_fonts, pick_font
    from common.render.scene import NO_HEAD, NO_TAIL, V_GAP, render_into_scene
    from common.text import norm
    from scenes.judge import bubble_region

    box, text = m["sel"]["box"], m["text"]
    n = len(text)
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    bubble, usable = bubble_region(load_bgr(Path(m["file"])), box)
    arr = np.asarray(Image.open(m["file"]).convert("RGB"))
    mask = bubble_mask(arr, box) if bubble is not None else None
    if mask is None:
        return {"why": "no_bubble"}
    inside = bubble_interior(mask).astype(np.uint8)
    # the interior closed over a stroke: a glyph run into the outline is a
    # notch in it, not a hole
    k = 2 * int(0.25 * (w * h / max(1, len(norm(m["sel"]["read"] or "")))) ** 0.5) + 1
    wipe = cv2.morphologyEx(
        inside, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    wipe[:y0], wipe[y1:], wipe[:, :x0], wipe[:, x1:] = 0, 0, 0, 0
    wipe = (wipe > 0) | (inside > 0)
    fill = np.array(ring_median(arr, box), np.int16)
    fonts = [f for f in find_fonts() if "Light" not in f] or find_fonts()
    font = pick_font(text, fonts, rng)
    draw_seed = rng.random()  # the colour draw, the same on every attempt

    def attempt(horiz: bool, reg: list, lines: int, cuts=None):
        reg = [
            int(max(reg[0], usable[0])),
            int(max(reg[1], usable[1])),
            int(min(reg[2], usable[2])),
            int(min(reg[3], usable[3])),
        ]
        if min(reg[2] - reg[0], reg[3] - reg[1]) < MIN_GLYPH:
            return None
        scene = {
            "file": m["file"],
            "regions": [reg],
            "region": reg,
            "boxes_anchor": [list(box)],
            "bubbles": [bubble],
        }
        for frac in SHRINK:
            out = render_into_scene(
                scene,
                text,
                font,
                random.Random(draw_seed),
                min_glyph=MIN_GLYPH,
                fill_frac=frac,
                tilt_frac=0.0,
                max_lines=lines,
                cuts=cuts,
                tategaki=True,
                vertical_only=not horiz,
                horizontal=horiz,
            )
            if out is None:
                return None
            im, d = out
            rest = wipe.copy()
            rest[d[1] : d[3], d[0] : d[2]] = False
            a = np.array(im)
            a[rest] = fill
            im = Image.fromarray(a)
            px = ((d[2] - d[0]) * (d[3] - d[1]) / n) ** 0.5
            k = max(3, int(MARGIN * px))
            safe = cv2.erode(inside, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
            crop = np.asarray(im, np.int16)[d[1] : d[3], d[0] : d[2]]
            ink = np.abs(crop - fill).max(-1) > 60  # glyphs, or the outline / scene
            if not (ink & ~safe[d[1] : d[3], d[0] : d[2]]).any():
                return {"im": im, "box": d, "px": px, "region": reg, "horiz": horiz}
        return None

    rows, cols = drawn_grid(w, h, len(norm(m["sel"]["read"] or "")) or n)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if 1 in (rows, cols):  # the model's line: its extent, its orientation
        horiz = w >= h if rows == cols else rows == 1
        got = attempt(
            horiz,
            [x0, cy - GROW * h / 2, x1, cy + GROW * h / 2]
            if horiz
            else [cx - GROW * w / 2, y0, cx + GROW * w / 2, y1],
            1,
        )
    else:  # the model split the word: a line or a column about its text centre
        cell = min(w / cols, h / rows)
        hx = min(cx - usable[0], usable[2] - cx)
        hy = min(cy - usable[1], usable[3] - cy)
        line = attempt(True, [cx - hx, cy - cell / 2, cx + hx, cy + cell / 2], 1)
        col = attempt(False, [cx - cell / 2, cy - hy, cx + cell / 2, cy + hy], 1)
        cut = n // 2 + 1  # the first column the longer: 3 + 2, 4 + 2
        if (
            n >= TWO_COLS
            and (col is None or col["px"] < SPLIT * cell)
            and text[cut] not in NO_HEAD
            and text[cut - 1] not in NO_TAIL
        ):
            cw = cell * (1 + V_GAP)
            col2 = attempt(
                False, [cx - cw / 2, cy - hy, cx + cw / 2, cy + hy], 2, cuts=[cut]
            )
            if col2 is not None and (col is None or col2["px"] > col["px"]):
                col = col2
        first, other = (line, col) if m["horizontal"] else (col, line)
        got = (
            first
            if first is not None and (other is None or first["px"] >= 0.9 * other["px"])
            else other
        )
    if got is None:
        return {"why": "no_fit"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    got["im"].save(dst)
    return {
        "box": got["box"],
        "lines_horizontal": got["horiz"],
        "region": got["region"],
        "bubble": bubble,
        "model_grid": [rows, cols],
    }


def quart(xs: list) -> list:
    return [round(x) for x in st.quantiles(xs, n=4)] if len(xs) > 3 else sorted(xs)


def sheets(recs: list, items: list, sheet_n: int) -> None:
    from PIL import Image, ImageDraw

    from common.readers import contact_sheet

    per = 60
    for k in range(0, len(recs), per):
        rows = [
            (
                Image.open(m["file"]).convert("RGB"),
                [
                    f"{m['text']}  [{m['sel']['cls']}]",
                    f"{m['frame']}{' hz' if m['horizontal'] else ''}  sfx {(m['sel'].get('read') or '')[:12]}",
                ],
            )
            for m in recs[k : k + per]
        ]
        contact_sheet(rows, ROOT / f"sheet_all_{k // per}.png", thumb=224, cols=10)
    rows = []
    for it in items[:sheet_n]:
        im = Image.open(it["file"]).convert("RGB")
        ImageDraw.Draw(im).rectangle(it["box"], outline=(255, 0, 0), width=2)
        rows.append((im, [f"{it['text']}  render", f"sfx {(it['read'] or '')[:14]}"]))
        rows.append(
            (
                Image.open(it["canvas"]).convert("RGB"),
                [
                    f"canvas{'' if it['kept'] else '  [dropped]'}  px {it['px']:.0f}  × {it['grow']:.2f}",
                    f"sfx {(it['canvas_read'] or '')[:14]}",
                ],
            )
        )
    if rows:
        contact_sheet(rows, ROOT / "sheet_items.png", thumb=320, cols=6)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--lens", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--seed", type=int, default=STREAM_SEED)
    p.add_argument("--sheet_n", type=int, default=24)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    items, info = plan(args.n, args.lens, args.seed)
    print(
        f"{len(items)} renders; word pool {info['pool']} (dropped {info['dropped']})",
        flush=True,
    )
    for it in items[:8]:
        print(f"  {it['caption']}", flush=True)
    if args.dry_run:
        return
    from common.text import lev, norm

    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    device = render(items)
    recs = read(items, ROOT / "render", device)
    for m in recs:
        m["sel"] = classify(m)
    near = [m for m in recs if m["sel"]["cls"] == "near"]
    rng = random.Random(args.seed)
    cdir = ROOT / "canvas" / "img"
    for old in cdir.glob("*.png") if cdir.exists() else ():
        old.unlink()  # an earlier redraw's canvases
    canvas, missed = [], Counter()
    for m in near:
        fa = cdir / f"c_{m['pi']:04d}_{m['text']}.png"
        geo = redraw(m, fa, rng)
        if "why" in geo:
            missed[geo["why"]] += 1
            continue
        canvas.append(
            {
                k: m[k]
                for k in (
                    "pi",
                    "seed",
                    "prompt",
                    "text",
                    "clause",
                    "caption",
                    "frame",
                    "shape",
                    "horizontal",
                )
            }
            | {"file": str(fa), "cond": "canvas", "geo": geo, "src": m["file"]}
            | {
                k: m["sel"][k]
                for k in ("box", "read", "edits", "dup", "long", "short", "vertical")
            }
        )
    cread = read(canvas, ROOT / "canvas", device, reuse=False) if canvas else []
    out_items = []
    for c in cread:
        t = norm(c["text"])
        best = min(
            (lev(norm(r.get(x) or ""), t) for r in c["reads"] for x in ("sfx", "vl")),
            default=len(t),
        )
        main_read = next((r.get("sfx") for r in c["reads"] if not r.get("whole")), "")
        d = c["geo"]["box"]
        dw, dh = d[2] - d[0], d[3] - d[1]
        px = (dw * dh / len(c["text"])) ** 0.5  # the reports' px: √(box area / glyphs)
        px_model = (c["long"] * c["short"] / max(1, len(norm(c["read"] or "")))) ** 0.5
        out_items.append(
            {k: v for k, v in c.items() if k not in ("reads", "file")}
            | {
                "file": c["src"],
                "canvas": c["file"],
                "loss_box": d,
                "canvas_read": main_read,
                "canvas_edits": best,
                "kept": best <= 1,
                "px": round(px, 1),
                "grow": round(px / px_model, 3),  # the redrawn glyph / the model's
                "span": round(max(dw, dh) / c["long"], 3),  # drawn extent / the model's
            }
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "items.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_items),
        encoding="utf-8",
    )
    kept = [r for r in out_items if r["kept"]]
    cls = Counter(m["sel"]["cls"] for m in recs)
    by_len = {
        n: dict(Counter(m["sel"]["cls"] for m in recs if len(m["text"]) == n))
        for n in args.lens
    }
    by_frame = {
        f: dict(Counter(m["sel"]["cls"] for m in recs if m["frame"] == f))
        for f in FRAMES.split(",")
    }
    boxed = [m["sel"] for m in recs if "box" in m["sel"]]
    metrics = {
        "renders": len(recs),
        "plan": info,
        "class": dict(cls),
        "class_by_glyphs": by_len,
        "class_by_frame": by_frame,
        "dup_main_box": sum(s["dup"] for s in boxed),
        "near": len(near),
        "redraw_missed": dict(missed),
        "items": len(out_items),
        "kept": len(kept),
        "kept_by_glyphs": dict(sorted(Counter(len(r["text"]) for r in kept).items())),
        "kept_by_frame": dict(Counter(r["frame"] for r in kept)),
        "all_boxed": {
            "n": len(boxed),
            "short_q": quart([s["short"] for s in boxed]),
            "long_q": quart([s["long"] for s in boxed]),
            "vertical": round(
                sum(s["vertical"] for s in boxed) / max(1, len(boxed)), 3
            ),
        },
        "kept_region": {
            "short_q": quart([r["short"] for r in kept]),
            "long_q": quart([r["long"] for r in kept]),
            "vertical": round(sum(r["vertical"] for r in kept) / max(1, len(kept)), 3),
            "px_q": quart([r["px"] for r in kept]),
            "grow_q": [
                round(x, 2) for x in st.quantiles([r["grow"] for r in kept], n=4)
            ]
            if len(kept) > 3
            else [],
            "span_q": [
                round(x, 2) for x in st.quantiles([r["span"] for r in kept], n=4)
            ]
            if len(kept) > 3
            else [],
            "dup": sum(r["dup"] for r in kept),
        },
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=1), flush=True)
    random.Random(0).shuffle(out_items)
    sheets(recs, out_items, args.sheet_n)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(ROOT)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
