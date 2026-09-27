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

Warm = every idx of the vocab has a seed row (``paths.SEED_ROWS``).
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


RULES = (
    Rule(
        "single",
        (1, 1),
        False,
        150,
        "cold kanji: 90 → 270 steps / row doubled contained (59 → 117 / 192, "
        "reports/stage_i_2026_09_26.md § 5); 150 lies between, unmeasured "
        "(plan_2900 § 3, the C-k budget is the user's call)",
    ),
    # 3+-glyph pieces: B0 (plan_2900 § 2) sets this row; until then 90
)


def rule_for(kind: str, glyphs: int, warm: bool) -> Rule | None:
    for r in RULES:
        lo, hi = r.glyphs
        if (
            r.kind == kind
            and glyphs >= lo
            and (hi is None or glyphs <= hi)
            and (r.warm is None or r.warm == warm)
        ):
            return r
    return None


def factor(kind: str, glyphs: int, warm: bool) -> float:
    r = rule_for(kind, glyphs, warm)
    return 1.0 if r is None else r.steps / BASE_STEPS


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
        warm = all(int(e) in seeds for e in idx)
        out[v] = factor(vocab_kind(v, len(ps)), glyph_count(v), warm)
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
