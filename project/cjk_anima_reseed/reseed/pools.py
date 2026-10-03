"""The pools every recipe draws from: the rows (single glyphs), the scenes,
the windowed word pool, the lone canvases."""

from __future__ import annotations

import json
import random
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
    used: Counter = field(default_factory=Counter)  # scene → items drawn on it
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

    scenes = load_scenes(T.SCENES, 0.0, 0, "", T.ONE_BUBBLE)
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
    scenes += load_scenes(T.SMALL_POOL, 0.0, 0, "", "")
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
