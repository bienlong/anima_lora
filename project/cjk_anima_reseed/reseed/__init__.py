"""reseed — the JA vocab pack's rows re-seeded cold: the data build.

One table (``table.TABLE``): every tier carries its own glyph px, σ band
and share; the builder draws each tier once and stamps its band — no band
gate, no group, no rebuild pass. Built 2026-10-03 from
``cjk_anima_scale/experiments/reseed_anchor --variant fit`` (the
``grid_small`` → ``grid_lone`` → ``grid_44`` → ``reseed_recap`` chain
flattened), without its ！ / ？ marks and EN cells.

Shared with ``../cjk_anima_scale`` and read-only from here: its ``src/``
(renderers, scene pools, fonts, inventory) and ``cjk_scale.{paths, config,
budget, train}`` (the trainer). Its ``cjk_scale.builder`` / ``recipes``
rebuild the seed of record and are never imported (``tests/``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]  # project/cjk_anima_reseed
SCALE = HOME.parent / "cjk_anima_scale"
REPO = HOME.parents[1]
OUT = REPO / "output" / "cjk_anima_reseed"
CONFIGS = HOME / "configs"


def bootstrap() -> None:
    """``cjk_scale`` importable, its ``src/`` first on ``sys.path`` (its
    ``train`` must win over the repo's ``train.py``), the pack named."""
    os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")
    s = str(SCALE)
    if s not in sys.path:
        sys.path.insert(0, s)
    from cjk_scale.paths import bootstrap as scale_bootstrap

    scale_bootstrap()
