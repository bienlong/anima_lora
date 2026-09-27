#!/usr/bin/env python
"""ctx_trigger — where does EN's in-word context come from? (2026-09-27)

F2a / F2a′ (proposal.md § 2.0) found no ungated ``v_line``, and F0 found the
adapter nearly blind to a pack row's neighbours: the seed's own context cos
(a glyph's adapter output in a spelled word vs alone) is 0.965–0.975. Yet
the base renders EN words from fragments — inside quotes EN is always split
(``"hello"`` → ``▁" | h | ello``, ``"wonderful"`` → ``w | on | der | ful``),
and the same T5 id stands alone (``"h"``) and inside the word. One
difference: the adapter's cross-attention reads the Qwen3 hidden states, and
Qwen sees ``hello`` as one word, while our spelled captions (``"ひ ま わ
り"``) hand Qwen spaced glyphs and byte fragments (``'ĠãĤ', 'ı'``).

No DiT, no renders: TE + ``llm_adapter`` (fp32, the seed rows hooked), and
per (word, piece, prompt, clause) the cos between the piece's hidden state
in the word and the same T5 id alone, per block boundary (``L0`` … ``out``).
A low cos = the adapter reads the piece by its context. Conditions (``T5``
ids / ``Qwen`` text):

    en_word        "hello"        / "hello"         EN as the base reads it
    en_spaced      "h e l l o"    / "h e l l o"     EN spelled like our JA
    en_qwen_sp     "hello"        / "h e l l o"     EN T5, Qwen word context removed
    ja_spaced      "ひ ま わ り"   / "ひ ま わ り"    the line's spelled form (F0)
    ja_hybrid      "ひ ま わ り"   / "ひまわり"       single rows, Qwen word context
    ja_word        "ひまわり"      / "ひまわり"       the natural form (piece rows
                                                     where the pack has them; the
                                                     single-row glyphs are read)

Hybrid captions take (prompt_embeds, attn mask) from the Qwen-text caption
and (t5 ids, t5 mask) from the T5-text caption. Alone = the piece's own
text in the same prompt and clause, and it must encode to the same id.

Decision: en_word far below ja_spaced, and ja_hybrid moving toward en_word
(or en_qwen_sp up toward ja_spaced) → the in-word context comes through
Qwen, and the spelled captions switched it off; the next read renders
ja_hybrid on the seed rows. All alike → the context is a property of the
pretrained EN rows, and the gate stays.

``--probe c2`` (after c1: EN 0.51 · 0.44 · 0.53, JA 0.965 · 0.965 · 0.973 —
the context is T5-side, Qwen is not the trigger): is the JA rows' context
immunity the rows or the pack? ``ja_spaced`` / ``ja_word`` under five row
arms (effective row = pack row + delta, every seed row):

    seed          the seed rows (c1)
    raw           the pack rows alone (delta scale 0, no identity training)
    seed_n200     the seed rows rescaled to norm 200 (≈ the T5 table's 212)
    seed_at_pack  the seed rows' direction at each pack row's own norm
    raw_at_seed   the pack rows' direction at the seed rows' norm

and ``en_word`` over ``N_EN_BIG`` sampled Qwen word tokens, each T5 piece
tagged with its share (how many Qwen Latin tokens' T5 spelling holds it):
does an ambiguous piece take more context?

``--probe c3`` (hypothesis.md § 4 P0b, after P0: × 0.8 / × 0.65 of the
seed rows lost identity with no composition gain): in EN, share and norm are
one axis (r −0.87), so is JA's steep norm response generic to the pre-norm
adapter, or learned by token? The same α on both sides, the prompt's rows
untouched:

    en_word    @ x<a>   the T5 rows of the quoted words' pieces × a
    ja_spaced  @ x<a>   the seed's effective rows × a (c2's arms, as α)

``ALPHAS`` 0.65 / 0.8 / 1 / 1.2 / 1.5. EN moving as steeply as JA → the norm
is a generic lever (H1); EN barely moving → the context read is learned by
token (H2), and a cap alone will not reach it.

``--probe c4`` (hypothesis.md § 4 P1, after training): ``ja_spaced`` on the
Stage B donor's own words (``HELD_IN`` + ``N_C4_WORDS`` donor words) under
trained rows files — every arm's effective rows on the hook, whole:

    seed      the seed rows
    raw       the pack rows (no identity training)
    stage_b   the Stage B donor (warm from the seed, uncapped, in-word)
    <name>    ``OUT/experiments/<name>/trained.pt`` (``--arms``; P1's
              ``p1_cold`` / ``p1_cap``)

A trained arm near ``raw`` (≈ 0.66, c2) reads its rows in context; near
``seed`` / ``stage_b`` (≈ 0.97) it does not.

``--dry_run`` prints the pieces per condition (tokenizers only).
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics as st
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "f0_exp", LINE / "experiments" / "f0_interaction" / "run_exp.py"
)
F0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F0)
SB = F0.SB

EN_WORDS = (
    "hello", "wonderful", "sunflower", "morning", "thanks",
    "beautiful", "together", "yesterday", "chocolate", "adventure",
)  # fmt: skip
JA_WORDS = SB.HELD_WORDS  # ひまわり さくら みどり くもり まくら
CLAUSES = ("en", "swap")
LAYERS = F0.LAYERS
CONDS = ("en_word", "en_spaced", "en_qwen_sp", "ja_spaced", "ja_hybrid", "ja_word")
C2_CONDS = ("en_word", "ja_spaced", "ja_word")
C3_CONDS = ("en_word", "ja_spaced")
ALPHAS = (0.65, 0.8, 1.0, 1.2, 1.5)  # c3: one scale for both sides
ARMS = ("seed", "raw", "seed_n200", "seed_at_pack", "raw_at_seed")
C4_CONDS = ("ja_spaced",)
C4_ARMS = ("seed", "raw", "stage_b", "p1_cold", "p1_cap")
N_C4_WORDS = 7  # donor words beside HELD_IN, 3–5 glyphs, drawn with seed 0
_JA_WORDS: list = []  # set in main for c4
N_EN_BIG = 60  # c2: Qwen word tokens sampled for the share read
_EN_BIG: list = []  # set in main for c2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--probe", choices=("c1", "c2", "c3", "c4"), default="c1")
    p.add_argument("--arms", nargs="+", default=list(C4_ARMS), help="c4: rows arms")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def forms(cond: str, w: str) -> tuple[str, str]:
    """``(t5 text, qwen text)`` of word ``w`` under ``cond``."""
    sp = " ".join(w)
    return {
        "en_word": (w, w),
        "en_spaced": (sp, sp),
        "en_qwen_sp": (w, sp),
        "ja_spaced": (sp, sp),
        "ja_hybrid": (sp, w),
        "ja_word": (w, w),
    }[cond]


def words(cond: str):
    if cond.startswith("en"):
        return _EN_BIG or EN_WORDS
    return _JA_WORDS or JA_WORDS


def c4_words() -> list[str]:
    """``HELD_IN`` + ``N_C4_WORDS`` of the Stage B donor words (3–5 glyphs)."""
    import json
    import random

    ws = json.loads(
        (OUT / SB.NAME / "donor_words.json").read_text(encoding="utf-8")
    )
    ws = [w for w in ws if 3 <= len(w) <= 5]
    random.Random(0).shuffle(ws)
    return [SB.HELD_IN, *sorted(ws[:N_C4_WORDS])]


def c4_path(arm: str) -> Path:
    if arm == "seed":
        return SEED_ROWS
    if arm == "stage_b":
        return OUT / SB.NAME / "trained.pt"
    return OUT / "experiments" / arm / "trained.pt"


def en_big(tok) -> tuple[list[str], dict]:
    """``N_EN_BIG`` Qwen word-start tokens (lowercase, 6–10 letters) whose
    quoted T5 spelling is ≥ 2 pieces, and per T5 piece id its share: how many
    of Qwen's Latin tokens hold it in their quoted T5 spelling."""
    import random
    import re

    from transformers import AutoTokenizer

    from library.env import resolve_under_home

    qw = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    lat = re.compile(r"^[A-Za-z]+$")
    bodies, starts = [], []
    for t in qw.get_vocab():
        s = qw.convert_tokens_to_string([t])
        b = s[1:] if s.startswith(" ") else s
        if lat.match(b):
            bodies.append(b)
            if s.startswith(" ") and b.islower() and 6 <= len(b) <= 10:
                starts.append(b)
    share: dict = {}
    for b in bodies:
        for i in set(tok.t5('"' + b, add_special_tokens=False)["input_ids"][1:]):
            share[i] = share.get(i, 0) + 1
    rng = random.Random(0)
    rng.shuffle(starts)
    picked = [
        w
        for w in sorted(set(starts))
        if len(tok.t5('"' + w, add_special_tokens=False)["input_ids"]) >= 3
    ]
    rng.shuffle(picked)
    return sorted(picked[:N_EN_BIG]), share


class Tok:
    """The pipeline's T5 side on CPU (``HybridT5Encoder``, the pack's ext ids)."""

    def __init__(self):
        import os

        from transformers import AutoTokenizer

        from library.anima import ext_vocab
        from library.anima.ext_vocab import HybridT5Encoder
        from library.anima.vocab_pack import resolve_pack_prefix
        from library.env import resolve_under_home

        self.t5 = AutoTokenizer.from_pretrained(
            resolve_under_home("library/anima/configs/t5_old")
        )
        qw = AutoTokenizer.from_pretrained(
            resolve_under_home("library/anima/configs/qwen3_06b")
        )
        _, mapping = ext_vocab.load_ext_assets(
            resolve_pack_prefix(os.environ["ANIMA_VOCAB_PACK"])
        )
        self.enc = HybridT5Encoder.from_mapping(self.t5, qw, mapping)

    def ids(self, caption: str) -> list[int]:
        ids, mask = self.enc.encode(caption, 512)
        return [i for i, m in zip(ids, mask) if m]

    def span(self, p: str, cl: str, k: str) -> tuple[list[int], int]:
        """The quoted text's T5 ids and the index where they start: after
        the clause's ``▁as ▁"``, up to the closing quote (EN closes with
        ``".``, a pack span with ``▁"``)."""
        a = self.ids(F0.caption(p, k, cl))
        names = [self.name(x) for x in a]
        i = (
            max(
                q
                for q in range(1, len(a))
                if names[q - 1] == "▁as" and names[q] == '▁"'
            )
            + 1
        )
        j = next(q for q in range(i, len(a)) if names[q] in ('".', '▁"', '"'))
        return a[i:j], i

    def name(self, i: int) -> str:
        from library.anima.ext_vocab import T5_TABLE_SIZE

        return (
            f"<ext{i - T5_TABLE_SIZE}>"
            if i >= T5_TABLE_SIZE
            else self.t5.convert_ids_to_tokens(i)
        )


def plan(tok: Tok, ps: list[str], conds=CONDS) -> dict:
    """Per condition: ``[(word, t5 text, qwen text, piece idx, piece text,
    alone text)]`` for the pieces whose alone form is the same single id
    (the ``▁`` separators of a spaced form are not pieces). Checked on the
    first prompt / clause; every other one is asserted at run time."""
    p, cl = ps[0], "swap"
    out = {}
    for cond in conds:
        rows = []
        for w in words(cond):
            t5_text, q_text = forms(cond, w)
            span, _ = tok.span(p, cl, t5_text)
            if sum(tok.name(i) != "▁" for i in span) < 2:
                continue  # one piece: "in the word" is the alone caption
            for j, i in enumerate(span):
                name = tok.name(i)
                if name == "▁":
                    continue
                # the alone text: an EN piece is its own string; a JA id is
                # the glyph that encodes to it alone (a piece row has none)
                if cond.startswith("en"):
                    alone = name.lstrip("▁")
                else:
                    alone = next((g for g in w if tok.span(p, cl, g)[0] == [i]), None)
                if not alone or tok.span(p, cl, alone)[0] != [i]:
                    continue
                rows.append((w, t5_text, q_text, j, name, alone))
        out[cond] = rows
    return out


def main():
    args = parse_args()
    ps = F0.prompts()
    tok = Tok()
    share: dict = {}
    if args.probe in ("c2", "c3"):
        big, share = en_big(tok)
        _EN_BIG[:] = big
        print(f"en_word: {len(big)} Qwen word tokens: {' '.join(big)}", flush=True)
    if args.probe == "c4":
        _JA_WORDS[:] = c4_words()
        miss = [a for a in args.arms if a != "raw" and not c4_path(a).exists()]
        assert not miss, f"no trained.pt for {miss}"
    pl = plan(
        tok,
        ps,
        {"c2": C2_CONDS, "c3": C3_CONDS, "c4": C4_CONDS}.get(args.probe, CONDS),
    )
    for cond, rows in pl.items():
        by_w: dict = {}
        for w, *_r, name, alone in rows:
            by_w.setdefault(w, []).append(name)
        print(
            f"{cond}: {len(rows)} pieces — "
            + "; ".join(f"{w} {' '.join(v)}" for w, v in by_w.items()),
            flush=True,
        )
    metrics: dict = {
        "prompts": len(ps),
        "clauses": CLAUSES,
        "pieces": {c: [list(r) for r in rows] for c, rows in pl.items()},
    }
    if args.dry_run:
        return

    import torch

    from common.hooks import ExtDelta
    from common.models import checkpoints, encode_captions
    from library.anima.weights import load_llm_adapter

    run_dir = make_run_dir(
        "ctx_trigger",
        label=args.label,
        root=LINE / "experiments" / "ctx_trigger" / "results",
    )
    dev = torch.device(args.device)
    ck = checkpoints()
    assert ck.vocab_pack, "set ANIMA_VOCAB_PACK"
    caps = set()
    for rows in pl.values():
        for _w, t5_text, q_text, _j, _n, alone in rows:
            for p in ps:
                for cl in CLAUSES:
                    caps.update(
                        F0.caption(p, x, cl) for x in (t5_text, q_text, alone, "")
                    )
    enc = encode_captions(
        sorted(caps), dev, cache_dir=OUT / f"ctx_trigger_{args.label}" / "te_cache"
    )
    adapter = load_llm_adapter(
        ck.dit, dtype=torch.float32, device=dev, vocab_pack=ck.vocab_pack
    )
    seed = torch.load(SEED_ROWS, map_location="cpu", weights_only=False)["delta"]
    # the seed rows on the embed hook (kept referenced for the whole run)
    delta = ExtDelta.from_state(type("A", (), {"llm_adapter": adapter})(), seed, dev)
    cap = F0.Capture(adapter)
    import os

    from library.anima import ext_vocab
    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.anima.vocab_pack import resolve_pack_prefix

    table, _ = ext_vocab.load_ext_assets(
        resolve_pack_prefix(os.environ["ANIMA_VOCAB_PACK"])
    )
    pack = torch.as_tensor(table)[delta.ext_ids].float().to(dev)
    raw0 = delta.raw.detach().clone()
    rs = float(delta.row_scale)
    eff0 = pack + raw0 * rs

    # c3: the T5 rows of every quoted EN word's pieces (the prompt's rows are
    # not among them), and their stock values
    W = adapter.embed.weight
    en_ids = sorted(
        {
            i
            for w, t5_text, *_r in pl.get("en_word", [])
            for i in tok.span(ps[0], "swap", t5_text)[0]
        }
    )
    en_idx = torch.tensor(en_ids, dtype=torch.long, device=W.device)
    en0 = W.data[en_idx].clone() if en_ids else None

    def set_scale(arm: str, cond: str) -> dict:
        """c3: ``x<a>`` on ``cond``'s side, the other side stock."""
        a = float(arm[1:])
        a_en, a_ja = (a, 1.0) if cond.startswith("en") else (1.0, a)
        with torch.no_grad():
            W.data[en_idx] = en0 * a_en
            delta.raw.copy_((eff0 * a_ja - pack) / rs)
        delta.scale = 1.0
        rows = en0 * a_en if cond.startswith("en") else eff0 * a_ja
        return {"alpha": a, "row_norm": round(float(rows.norm(dim=1).mean()), 1)}

    table_t = torch.as_tensor(table).float()
    words_ids = sorted(
        {
            i - T5_TABLE_SIZE
            for rows_ in pl.values()
            for _w, t5_text, *_r in rows_
            for i in tok.span(ps[0], "swap", t5_text)[0]
            if i >= T5_TABLE_SIZE
        }
    )

    def set_file(arm: str) -> dict:
        """c4: ``arm``'s rows file on the hook — its effective rows (pack +
        raw × its row_scale) on every id it carries, the seed's elsewhere."""
        if arm == "raw":
            v = pack
        else:
            d = torch.load(c4_path(arm), map_location="cpu", weights_only=False)
            d = d["delta"]
            ids = [int(e) for e in d["ext_ids"]]
            eff = table_t[ids] + d["raw"].float() * float(d["row_scale"])
            at = {e: i for i, e in enumerate(ids)}
            v = eff0.clone()
            for k, e in enumerate(delta.ext_ids):
                if int(e) in at:
                    v[k] = eff[at[int(e)]].to(dev)
        with torch.no_grad():
            delta.raw.copy_((v - pack) / rs)
        delta.scale = 1.0
        loc = [delta.index[e] for e in words_ids if e in delta.index]
        return {
            "path": "pack" if arm == "raw" else str(c4_path(arm)),
            "row_norm": round(float(v[loc].norm(dim=1).mean()), 1),
        }

    def set_arm(arm: str, cond: str = "") -> dict:
        """Put ``arm``'s effective rows on the hook; returns their mean norm."""
        if args.probe == "c3":
            return set_scale(arm, cond)
        if args.probe == "c4":
            return set_file(arm)
        pn, en = pack.norm(dim=1, keepdim=True), eff0.norm(dim=1, keepdim=True)
        v = {
            "seed": eff0,
            "raw": pack,
            "seed_n200": eff0 * (200.0 / en),
            "seed_at_pack": eff0 * (pn / en),
            "raw_at_seed": pack * (en / pn),
        }[arm]
        with torch.no_grad():
            delta.raw.copy_((v - pack) / rs)
        delta.scale = 1.0
        return {"row_norm": round(float(v.norm(dim=1).mean()), 1)}

    def run(pairs):
        """``pairs``: [(t5 caption, qwen caption)] → ``{layer: (B, L, D)}``."""
        pe = torch.stack([torch.as_tensor(enc[q][0]) for _, q in pairs]).to(dev).float()
        am = torch.stack([torch.as_tensor(enc[q][1]) for _, q in pairs]).to(dev)
        t5 = torch.stack([torch.as_tensor(enc[t][2]) for t, _ in pairs]).to(dev).long()
        t5m = torch.stack([torch.as_tensor(enc[t][3]) for t, _ in pairs]).to(dev)
        cap.h.clear()
        with torch.no_grad():
            out = adapter(
                source_hidden_states=pe,
                target_input_ids=t5,
                target_attention_mask=t5m,
                source_attention_mask=am,
            )
        h = dict(cap.h)
        h["out"] = out.float()
        return {k: v for k, v in h.items() if k in LAYERS}

    results: dict = {}
    if args.probe == "c3":
        jobs = [(cond, f"x{a:g}") for cond in pl for a in ALPHAS]
    elif args.probe == "c4":
        jobs = [(cond, arm) for cond in pl for arm in args.arms]
    else:
        jobs = [
            (cond, arm)
            for cond in pl
            for arm in (
                ARMS if args.probe == "c2" and cond.startswith("ja") else ("seed",)
            )
        ]
    for cond, arm in jobs:
        rows = pl[cond]
        arm_info = set_arm(arm, cond)
        by_piece: dict = {}
        vals: dict = {cl: {ly: [] for ly in LAYERS} for cl in CLAUSES}
        by_pos: dict = {"first": [], "inner": []}
        by_word: dict = {}
        for w, t5_text, q_text, j, name, alone in rows:
            for cl in CLAUSES:
                line = [
                    (F0.caption(p, t5_text, cl), F0.caption(p, q_text, cl)) for p in ps
                ]
                lone = [
                    (F0.caption(p, alone, cl), F0.caption(p, alone, cl)) for p in ps
                ]
                hl, ha = run(line), run(lone)
                for i, p in enumerate(ps):
                    span, s0 = tok.span(p, cl, t5_text)
                    aspan, a0 = tok.span(p, cl, alone)
                    assert tok.name(span[j]) == name and aspan == [span[j]], (
                        cond,
                        w,
                        p,
                        cl,
                    )
                    for ly in LAYERS:
                        c = F0._cos(hl[ly][i, s0 + j], ha[ly][i, a0])
                        vals[cl][ly].append(c)
                    c = vals[cl]["out"][-1]
                    by_pos["first" if j == 0 else "inner"].append(c)
                    by_word.setdefault(w, []).append(c)
                    by_piece.setdefault(span[j], []).append(c)
        res = {
            "n": len(vals[CLAUSES[0]]["out"]),
            **{
                cl: {ly: round(st.fmean(v), 4) for ly, v in d.items()}
                for cl, d in vals.items()
            },
            "out_by_pos": {k: round(st.fmean(v), 4) for k, v in by_pos.items() if v},
            "out_by_word": {k: round(st.fmean(v), 4) for k, v in by_word.items()},
        }
        res["arm"] = {"name": arm, **arm_info}
        if share and cond.startswith("en"):
            pc = sorted(
                (share.get(i, 0), st.fmean(v), tok.name(i)) for i, v in by_piece.items()
            )
            k = len(pc) // 3
            res["by_share_tercile"] = [
                {
                    "share_range": [pc[a][0], pc[b - 1][0]],
                    "n_pieces": b - a,
                    "out_cos": round(st.fmean(x[1] for x in pc[a:b]), 4),
                }
                for a, b in ((0, k), (k, 2 * k), (2 * k, len(pc)))
            ]
            res["by_piece"] = [[n, sh, round(c, 4)] for sh, c, n in pc]
            print(
                "            by share tercile: "
                + "  ".join(
                    f"{t['share_range'][0]}–{t['share_range'][1]} ({t['n_pieces']}) "
                    f"{t['out_cos']:.3f}"
                    for t in res["by_share_tercile"]
                ),
                flush=True,
            )
        results[f"{cond}@{arm}"] = res
        print(
            f"{cond + '@' + arm:<24} row norm {arm_info['row_norm']:>6} n {res['n']:>4}/clause  out cos "
            + "  ".join(f"{cl} {res[cl]['out']:.3f}" for cl in CLAUSES)
            + "  | by layer (swap) "
            + " ".join(f"{ly} {res['swap'][ly]:.3f}" for ly in LAYERS)
            + f"  | first {res['out_by_pos'].get('first', float('nan')):.3f} "
            f"inner {res['out_by_pos'].get('inner', float('nan')):.3f}",
            flush=True,
        )
        print(
            "            by word: "
            + " ".join(f"{k} {v:.3f}" for k, v in res["out_by_word"].items()),
            flush=True,
        )
    metrics["conds"] = results
    metrics["seed_rows"] = {"path": str(SEED_ROWS), "n": len(delta.ext_ids)}
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
