"""budget — steps and items per vocab by kind × glyph count × warm / cold
(plan_2900 § 5-1).

The trainer's ``STEPS_PER_VOCAB`` (90) and the builder's ``ITEMS_PER_VOCAB``
are the base; a vocab's **factor** scales both, so items keep pace with the
steps (stage_i scaled both). A rule written as rows with provenance, like
``windows.py``: a new read changes a row and its source string. A vocab no
row matches gets factor 1.

A run's vocabs must share one factor. The builder draws a vocab uniformly
within its kind and the batcher draws items uniformly, so a mixed run would
spread its budget evenly over every vocab — ``run_factor`` refuses it. Split
such a run by factor and join the rows with ``scale.py <out> merge``.

Warm = every idx of the vocab has a seed row (``paths.SEED_ROWS``) and its
kind does not start cold. **Singles start cold** (``COLD_KINDS``,
plan_retrain: the singles re-seed from the pack rows on lone + in-word
data), so a single is never warm here. Script (kana / kanji) splits the
cold single row: the kana point is P1b's, the kanji one stage_i's.

``mix_factor``: the steps keep pace with the items a kind's groups draw
(Σ of ``builder.TABLE`` shares) — the single kind's lone 0.5 + in-word
0.5 + 0.5 is × 1.5, so the in-word items keep P1b's exposure (90 → 135).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

BASE_STEPS = 90  # train.STEPS_PER_VOCAB of record (run0925_300f; conflict_joint report)


@dataclass(frozen=True)
class Rule:
    kind: str  # windows.KINDS
    glyphs: tuple  # (lo, hi) glyph count, hi None = open
    warm: bool | None  # None = either
    steps: int  # per vocab, at the base budget
    source: str
    script: str | None = None  # "kana" / "kanji" (``script_of``); None = either


COLD_KINDS = ("single",)  # plan_retrain § 6-4: every single re-seeds cold


RULES = (
    Rule(
        "single",
        (1, 1),
        False,
        90,
        "cold kana: p1_mix (plan_retrain § 4, hypothesis.md P1b), 36 hiragana "
        "cold at 90 × the in-word mix 1.5 = 135 / row: singles official 82 vs "
        "the floor's 91 / 144 (p 0.69), 8 held-in words ≤ 1 edit 80 / 128 "
        "(floor 1). Katakana at this row is unread (retrain_kana reads it)",
        script="kana",
    ),
    Rule(
        "single",
        (1, 1),
        False,
        150,
        "cold kanji: 90 → 270 steps / row doubled contained (59 → 117 / 192, "
        "reports/stage_i_2026_09_26.md § 5); 150 lies between, unmeasured "
        "(plan_2900 § 3, the C-k budget is the user's call). C3 (plan_retrain "
        "§ 4) at 150 × 1.5 = 225: new kanji official 0 → 51 / 192, the seed's "
        "dense kanji 65 → 31 — not set (plan_retrain § 7)",
        script="kanji",
    ),
    Rule(
        "piece",
        (4, 5),
        None,
        270,
        "long_b0 (reports/long_b0_2026_09_27.md): warm, piece table, 90 → 270 "
        "steps / row: 4-glyph contained 7 → 18 / 64 (4 / 4 pieces up), "
        "5-glyph 3 → 4 (near 8 → 13), 3-glyph 13 → 15 (stays at 90); "
        "user 2026-09-27: 270 for 4–5, cold pieces too (plan_2900 § 2 — B0 "
        "sets C-p's; cold is not read)",
    ),
)


def script_of(vocab: str) -> str:
    """``kanji`` when the vocab holds a CJK ideograph, else ``kana`` (kana,
    ー, punctuation)."""
    return "kanji" if any(0x3400 <= ord(c) <= 0x9FFF for c in vocab) else "kana"


def starts_cold(kind: str) -> bool:
    return kind in COLD_KINDS


def rule_for(kind: str, glyphs: int, warm: bool, script: str = "kana") -> Rule | None:
    for r in RULES:
        lo, hi = r.glyphs
        if (
            r.kind == kind
            and glyphs >= lo
            and (hi is None or glyphs <= hi)
            and (r.warm is None or r.warm == warm)
            and (r.script is None or r.script == script)
        ):
            return r
    return None


def factor(kind: str, glyphs: int, warm: bool, script: str = "kana") -> float:
    r = rule_for(kind, glyphs, warm, script)
    return 1.0 if r is None else r.steps / BASE_STEPS


def mix_factor(kinds, table=None) -> float:
    """Σ of the table's shares for the run's trained kinds (one value: a run
    whose kinds draw different totals is refused, like ``run_factor``)."""
    from .builder import TABLE

    table = TABLE if table is None else table
    by = {k: sum(g.share for g in table if g.kind == k) for k in set(kinds)}
    assert len(set(by.values())) <= 1, f"kinds draw different item totals {by}"
    return next(iter(by.values()), 1.0) or 1.0


@cache
def seed_ids() -> frozenset:
    import torch

    from .paths import SEED_ROWS

    sd = torch.load(SEED_ROWS, map_location="cpu", weights_only=False)
    return frozenset(int(e) for e in sd["delta"]["ext_ids"])


def vocab_factors(vocabs, tokq, seeds=None) -> dict:
    """vocab → its factor; a vocab with no pack row (nothing trains) is left
    out."""
    from data.inventory import pieces as qpieces

    from .windows import glyph_count, vocab_kind

    seeds = seed_ids() if seeds is None else seeds
    tok, q = tokq
    out = {}
    for v in vocabs:
        ps = qpieces(tok, q, v)
        idx = [e for _p, e in ps if e is not None]
        if not idx:
            continue
        kind = vocab_kind(v, len(ps))
        warm = not starts_cold(kind) and all(int(e) in seeds for e in idx)
        out[v] = factor(kind, glyph_count(v), warm, script_of(v))
    return out


def run_factor(vocabs, tokq, seeds=None) -> float:
    """The run's one factor; refuses a run whose vocabs fall under different
    factors."""
    f = vocab_factors(vocabs, tokq, seeds)
    by = {}
    for v, x in f.items():
        by.setdefault(x, []).append(v)
    assert len(by) <= 1, (
        "the run's vocabs fall under different budgets "
        + "; ".join(
            f"× {x:g} ({len(vs)}: {' '.join(vs[:6])}{' …' if len(vs) > 6 else ''})"
            for x, vs in sorted(by.items())
        )
        + " — split the run by budget and join the rows with `scale.py <out> merge`"
    )
    return next(iter(by), 1.0)
