"""The pools every recipe draws from: the rows (single glyphs), the scenes,
the windowed word pool, the dialogue lines, the lone canvases."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import table as T

WINDOW_LEN = (2, 6)  # P1b's in-word lines were 2–6 glyphs


@dataclass
class Pools:
    fonts: list
    tokq: tuple
    inv: object  # data.vocabs.Inventory: the eval groups
    singles: list  # the rows' glyphs
    scenes: list
    single_idx: set  # scenes a lone glyph may go to
    horiz_idx: set  # scenes a left-to-right window may go to
    opt_in: dict  # pool tag → scenes only a tier naming it in scene_pools draws
    mono: set  # greyscale / line-art scenes
    shapes: object  # data.stage.ShapePool
    windows: dict = field(default_factory=dict)  # glyph → its windows
    windows_len: dict = field(default_factory=dict)  # glyph → length → its windows
    sentences: dict = field(default_factory=dict)  # cells → dialogue lines
    used: Counter = field(default_factory=Counter)  # scene → items drawn on it
    tier_used: Counter = field(
        default_factory=Counter
    )  # the draw loop's: scene → items
    scene_cap: int | None = None  # the draw loop's: items a scene may take
    decks: dict = field(default_factory=dict)


def _quietly(fn, *args):
    """A stage resolver with its stdout cut to 160 columns a line."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args)
    for ln in buf.getvalue().splitlines():
        if ln.strip():
            print(ln if len(ln) <= 160 else ln[:159] + "…", flush=True)
    return out


def build_pools(rows: list, rng: random.Random) -> Pools:
    """``rows``: ``data.vocabs`` specs (``chars:…``), single glyphs only —
    the stage resolvers on them, so a row means what it meant in every read
    of record (and the eval groups ``eval.json`` carries)."""
    import tempfile
    from types import SimpleNamespace

    from cjk_scale.config import DATA
    from common.render.flat import find_fonts
    from data.inventory import pieces as qpieces
    from data.inventory import qwen_pieces
    from data.stage import (
        ShapePool,
        _base_inventory,
        _eval_strings,
        _resolve_singles,
        _word_set,
    )
    from data.synth import load_scenes
    from cjk_scale.windows import glyph_count

    a = SimpleNamespace(
        vocabs=list(rows),
        balanced=0,
        seed=T.SEED,
        phrase_min_pieces=int(DATA["phrase_min_pieces"]),
        phrase_max_pieces=int(DATA["phrase_max_pieces"]),
        phrase_norm=bool(DATA["phrase_norm"]),
        word_min_len=2,
        n_word_eval=0,
        line_max_len=16,
        n_line_eval=0,
    )
    inv = _base_inventory(a)
    tokq = qwen_pieces(char_rows=True)
    _eval_strings(a, rng, inv)
    with tempfile.TemporaryDirectory() as scratch:
        _quietly(_resolve_singles, a, Path(scratch), tokq, inv)
        _quietly(_word_set, a, Path(scratch), tokq, inv)
    for g in ("combo", "corpus", "line", "word", "word_held"):
        inv.evals.pop(g, None)
    tok, qmap = tokq
    pool = inv.pool()
    other = [u for u in pool if len(qpieces(tok, qmap, u)) != 1 or glyph_count(u) != 1]
    assert not other, f"reseed draws single glyphs only: {sorted(set(other))[:10]}"
    singles = list(dict.fromkeys(pool))
    if not inv.evals.get("single"):
        srng = random.Random(T.SEED + 47)
        inv.evals["single"] = sorted(srng.sample(singles, min(18, len(singles))))

    scenes = whole_bubbles(load_scenes(T.SCENES, 0.0, 0, "", T.ONE_BUBBLE))
    single_pools = set(T.SINGLE_SCENES.split(","))

    def single_ok(sc) -> bool:
        w = sc["region"][2] - sc["region"][0]
        h = sc["region"][3] - sc["region"][1]
        return sc["pool"] in single_pools and max(w, h) <= T.SINGLE_MAX_AR * max(
            1, min(w, h)
        )

    single_idx = {j for j, sc in enumerate(scenes) if single_ok(sc)}
    horiz = set(T.HORIZONTAL_SCENES.split(","))
    horiz_idx = {j for j, sc in enumerate(scenes) if sc["pool"] in horiz}
    n0 = len(scenes)
    scenes += whole_bubbles(load_scenes(T.SMALL_POOL, 0.0, 0, "", ""))
    opt_in = {T.SMALL_POOL: set(range(n0, len(scenes)))}
    mono = mono_scenes(scenes)
    print(
        f"pools: {len(singles)} rows; {n0} scenes ({len(single_idx)} take a lone "
        f"glyph, {len(horiz_idx)} a line) + {len(scenes) - n0} {T.SMALL_POOL}; "
        f"mono {len(mono)} / {len(scenes)} drawn at {T.MONO_SHARE}",
        flush=True,
    )
    return Pools(
        fonts=find_fonts(),
        tokq=tokq,
        inv=inv,
        singles=singles,
        scenes=scenes,
        single_idx=single_idx,
        horiz_idx=horiz_idx,
        opt_in=opt_in,
        mono=mono,
        shapes=ShapePool(T.SHAPES, T.SEED),
    )


# ----------------------------------------------------------------------------
# whole bubbles: the outline inside the canvas, the erase sparing it


def bubble_check(sc: dict) -> dict:
    """``edge``: the least px between an anchor bubble's interior and the
    canvas edge (``None``: no anchor bubble); ``left``: the largest share of
    an anchor's letter ink the outline-keeping erase leaves."""
    import numpy as np
    from common.bubble import ring_median
    from common.render.scene import erase_paint, outline_ink
    from PIL import Image

    arr = np.asarray(Image.open(sc["file"]).convert("RGB")).copy()
    H, W = arr.shape[:2]
    bubbles = sc.get("bubbles") or [None] * len(sc["regions"])
    edge = [min(b[0], b[1], W - b[2], H - b[3]) for b in bubbles if b]
    left = 0.0
    for tb, reg, bub in zip(sc["boxes_anchor"], sc["regions"], bubbles):
        paint = erase_paint(arr, tb, reg, open_ok=bub is None)
        if paint is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in tb)
        box = np.zeros((H, W), dtype=bool)
        box[y0:y1, x0:x1] = True
        fill = np.array(ring_median(arr, tb), dtype=np.int16)
        ink = box & (np.abs(arr.astype(np.int16) - fill).max(axis=2) > 24)
        kept = outline_ink(arr, tb, paint) & ink
        left = max(left, float(kept.sum() / max(1, ink.sum())))
    return {"edge": min(edge) if edge else None, "left": round(left, 4)}


def whole_bubbles(scenes: list) -> list:
    """``scenes`` less those whose bubble the canvas cuts (``BUBBLE_EDGE_MIN``)
    or whose erase would leave the letters (``ERASE_LEFT_MAX``)."""
    from cjk_scale.paths import OUT

    path = OUT / "experiments" / "scene_bubble_check.json"
    cache = json.loads(path.read_text("utf-8")) if path.exists() else {}
    miss = [s for s in scenes if s["file"] not in cache]
    for s in miss:
        cache[s["file"]] = bubble_check(s)
    if miss:
        path.write_text(json.dumps(cache, indent=0), encoding="utf-8")

    def cut(s) -> bool:
        e = cache[s["file"]]["edge"]
        return e is not None and e < T.BUBBLE_EDGE_MIN

    def left(s) -> bool:
        return cache[s["file"]]["left"] > T.ERASE_LEFT_MAX

    n_cut = sum(map(cut, scenes))
    n_left = sum(left(s) and not cut(s) for s in scenes)
    keep = [s for s in scenes if not cut(s) and not left(s)]
    print(
        f"whole bubbles: {len(keep)} / {len(scenes)} scenes kept — {n_cut} cut by "
        f"the canvas (< {T.BUBBLE_EDGE_MIN} px), {n_left} the erase leaves "
        f"lettered (> {T.ERASE_LEFT_MAX:g})",
        flush=True,
    )
    return keep


# ----------------------------------------------------------------------------
# scene colour (polish_seed's test; its cache is shared)


def colorful(file: str) -> float:
    """The share of a 128² thumbnail's pixels with HSV saturation and value
    over 0.15."""
    import numpy as np
    from PIL import Image

    im = np.asarray(Image.open(file).convert("RGB").resize((128, 128)), np.float32)
    im /= 255
    mx, mn = im.max(-1), im.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    return float(((sat > 0.15) & (mx > 0.15)).mean())


def mono_scenes(scenes: list) -> set:
    from cjk_scale.paths import OUT

    path = OUT / "experiments" / "scene_colorful.json"
    cache = json.loads(path.read_text("utf-8")) if path.exists() else {}
    miss = [s["file"] for s in scenes if s["file"] not in cache]
    for f in miss:
        cache[f] = colorful(f)
    if miss:
        path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return {j for j, s in enumerate(scenes) if cache[s["file"]] < T.COLOR_MIN}


# ----------------------------------------------------------------------------
# the windowed word pool


def window_glyphs(singles) -> set:
    """The rows a window may hold: letters (kana, kanji, ー), not
    punctuation (``・`` trains lone only)."""
    import unicodedata

    return {g for g in set(singles) if unicodedata.category(g) in ("Lo", "Lm")}


def window_pool(glyphs: set, lines, held=(), length: tuple = WINDOW_LEN) -> list:
    """Every substring of ``lines`` of ``length`` glyphs, all in ``glyphs``,
    no glyph repeated (training must not teach doubling), none opening on
    ``scene.NO_HEAD`` or a small kana (``V_SMALL``: NO_HEAD lacks ぁぃぅぇぉゎ),
    none holding a trigram of a ``held`` string. A window may cross a word
    boundary (C3)."""
    from common.render.scene import NO_HEAD, V_SMALL

    no_head = NO_HEAD | V_SMALL
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
            for i in range(len(run)):
                if run[i] in no_head:
                    continue
                for n in range(lo, hi + 1):
                    w = run[i : i + n]
                    if len(w) < n:
                        break
                    if len(set(w)) == n and not any(g in w for g in grams):
                        out.add(w)
            run = ""
    return sorted(out)


def ext_encoder():
    """``ext(route, text)``: the ext rows the pack's encoder gives ``text``
    in a caption clause, routed per glyph or not."""
    from transformers import AutoTokenizer

    from common.models import checkpoints
    from library.anima import ext_vocab
    from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder
    from library.anima.vocab_pack import resolve_pack_prefix
    from library.env import resolve_under_home

    t5 = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/t5_old")
    )
    qw = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    _, mapping = ext_vocab.load_ext_assets(
        resolve_pack_prefix(checkpoints().vocab_pack)
    )
    encs = {
        r: HybridT5Encoder.from_mapping(t5, qw, mapping, glyph_route=r)
        for r in (False, True)
    }

    def ext(route: bool, text: str) -> list:
        ids, mask = encs[route].encode(f'Japanese text reads as "{text}".', 512)
        return [
            i - T5_TABLE_SIZE for i, m in zip(ids, mask) if m and i >= T5_TABLE_SIZE
        ]

    return ext


def add_windows(pools: Pools, read: tuple, out: Path) -> dict:
    """``pools.windows`` (glyph → its windows) over the dialogue lines and
    the training set's own JA text, the read strings held out by trigram,
    every window routed to its glyphs' rows and nothing else (else dropped).
    Writes ``windows.json``; returns the stats for ``build.json``."""
    from cjk_scale.config import dataset_ja_lines, phrase_file

    glyphs = window_glyphs(pools.singles)
    lines = [
        ln.split("\t")[0]
        for ln in Path(phrase_file()).read_text(encoding="utf-8").splitlines()
    ]
    ds = dataset_ja_lines()
    ws = window_pool(glyphs, lines + ds, read)
    ext = ext_encoder()
    ids = {}
    for c in sorted(glyphs):
        a, b = ext(False, c), ext(True, c)
        assert len(a) == 1 and a == b, (c, a, b)
        ids[c] = a[0]
    ok = [w for w in ws if ext(True, w) == [ids[c] for c in w]]
    pools.windows = {g: v for g in sorted(glyphs) if (v := [w for w in ok if g in w])}
    pools.windows_len = {
        g: {k: [w for w in v if len(w) == k] for k in sorted({len(w) for w in v})}
        for g, v in pools.windows.items()
    }
    n = sorted(len(v) for v in pools.windows.values())
    stats = {
        "length": list(WINDOW_LEN),
        "held": list(read),
        "lines": {"dialogue": len(lines), "dataset": len(ds)},
        "n": len(ok),
        "dropped_by_encoding": len(ws) - len(ok),
        "glyphs": len(pools.windows),
        "glyphs_without": sorted(glyphs - set(pools.windows)),
        "per_glyph_min": n[0] if n else 0,
        "per_glyph_median": n[len(n) // 2] if n else 0,
        "by_length": dict(sorted(Counter(map(len, ok)).items())),
    }
    (out / "windows.json").write_text(
        json.dumps(ok, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    print(
        f"windows: {len(ok)} ({stats['dropped_by_encoding']} dropped by the "
        f"encoding check), {len(pools.windows)} glyphs, per glyph min "
        f"{stats['per_glyph_min']} median {stats['per_glyph_median']}; none for "
        f"{''.join(stats['glyphs_without']) or '-'} (lone only)",
        flush=True,
    )
    return stats


# ----------------------------------------------------------------------------
# the dialogue lines (``sent``)

# an ellipsis is drawn ``…`` under 4 dots, ``……`` at 4 or more (user, 10-05:
# Manga109 spells it ・・ / ･･･ / ・・・・・・, the page draws the leader)
_DOT_RUN = re.compile("[・･.．‥…]+")
_DOTS = {"‥": 2, "…": 3}
# what a line may hold off the pack's rows: the leader (T5's `...`) and the
# marks the pack's fold sends to T5's ! / ?
SENT_BASE = set("…！？!?")


def norm_ellipsis(t: str) -> str | None:
    """``t`` with every dot run an ellipsis (``…`` under 4 dots, ``……`` at 4
    or more); a lone ・ / ･ is a 中黒 and stays ・; ``None`` for a lone . / ．
    (not a mark dialogue uses)."""
    out, at = [], 0
    for m in _DOT_RUN.finditer(t):
        run = m.group()
        if len(run) == 1 and run in "・･":
            rep = "・"
        elif len(run) == 1 and run in ".．":
            return None
        else:
            rep = "…" if sum(_DOTS.get(c, 1) for c in run) < 4 else "……"
        out += [t[at : m.start()], rep]
        at = m.end()
    return "".join(out) + t[at:]


def sentence_ok(s: str, lengths: tuple) -> str | None:
    """Why ``s`` (normalised) is not a ``sent`` line, else ``None``."""
    from common.render.scene import NO_HEAD
    from data.synth import _SENT_DISTINCT, _letters

    lo, hi = lengths
    if not lo <= len(s) <= hi:
        return "length"
    ls = _letters(s)
    if len(ls) < T.SENT_MIN_LETTERS or len(set(ls)) < _SENT_DISTINCT:
        return "letters"
    if s[0] in NO_HEAD:
        return "head"
    if re.search(r"([^…])\1\1", s):
        return "run"  # ああああ: a glyph three times running
    return None


def add_sentences(pools: Pools, read: tuple, lengths: tuple, out: Path) -> dict:
    """``pools.sentences`` (cells → lines): the dialogue lines with their
    ellipses normalised, every char routed to its own single row (per glyph,
    as the windows are) or one of ``SENT_BASE``; held out: a ``read`` string
    by trigram (``window_pool``'s rule) and the dialogue ruler's 5+ glyph
    strings by 5-gram. Writes ``sentences.json``; returns the stats."""
    from cjk_scale.config import phrase_file

    from . import OUT

    lines = [
        ln.split("\t")[0].strip()
        for ln in Path(phrase_file()).read_text(encoding="utf-8").splitlines()
    ]
    grams = set()
    for h in read:
        n = min(3, len(h))
        grams |= {h[i : i + n] for i in range(len(h) - n + 1)}
    ruler_file = OUT / "ruler" / "ruler.json"
    ruler = (
        [r["text"] for r in json.loads(ruler_file.read_text("utf-8"))["items"]]
        if ruler_file.exists()
        else []
    )
    r5 = {r[i : i + 5] for r in ruler for i in range(len(r) - 4)}
    ext = ext_encoder()
    single: dict = {}

    def rows_of(c: str):
        if c not in single:
            single[c] = ext(True, c)
        return single[c]

    drop, keep = Counter(), set()
    for t in dict.fromkeys(lines):
        s = norm_ellipsis(t)
        why = "dot" if s is None else sentence_ok(s, lengths)
        if why is None:
            ids = [rows_of(c) for c in s]
            if any((c in SENT_BASE) != (not i) or len(i) > 1 for c, i in zip(s, ids)):
                why = "char"
            elif any(g in s for g in grams):
                why = "read"
            elif any(s[i : i + 5] in r5 for i in range(len(s) - 4)):
                why = "ruler"
            elif ext(True, s) != [x for i in ids for x in i]:
                why = "route"
        if why is None:
            keep.add(s)
        else:
            drop[why] += 1
    pools.sentences = {}
    for s in sorted(keep):
        pools.sentences.setdefault(len(s), []).append(s)
    stats = {
        "lengths": list(lengths),
        "lines": len(lines),
        "n": len(keep),
        "by_length": {n: len(v) for n, v in sorted(pools.sentences.items())},
        "ellipsis": sum("…" in s for s in keep),
        "dropped": dict(drop),
        "ruler_strings": len(ruler),
    }
    (out / "sentences.json").write_text(
        json.dumps(sorted(keep), ensure_ascii=False, indent=0), encoding="utf-8"
    )
    print(
        f"sentences: {len(keep)} of {len(lines)} lines ({stats['ellipsis']} with "
        f"an ellipsis), {lengths[0]}–{lengths[1]} cells; dropped {dict(drop)}",
        flush=True,
    )
    return stats
