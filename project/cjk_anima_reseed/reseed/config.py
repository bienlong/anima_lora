"""A reseed run is one file, ``configs/<run>.toml``::

    rows = ["chars:あい…"]       # the rows trained cold (data.vocabs specs, single glyphs)
    read = ["こんにちは", …]      # held out of the windows by trigram
    seed = "0921"                # the rows every other row rides frozen at: "0921" (the
                                 # old seed, the kana run's) or "0930" (seed_retrain_0930)
    steps_per_row = 135
    shares = { grid_44 = 10, … }  # optional: % of the items per tier, every tier
                                  # named, Σ 100; else table.TABLE's shares

Its outputs land in ``output/cjk_anima_reseed/<run>/`` (``data/``,
``trained.pt``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from . import CONFIGS, OUT

KEYS = ("rows", "read", "seed", "steps_per_row", "shares")


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    rows: tuple
    read: tuple
    seed: str
    steps_per_row: int
    shares: dict | None = None  # tier → % of the items

    def table(self) -> tuple:
        """``table.TABLE``, its shares the run's when it gives them (Σ kept at
        the table's, so the items per row stay)."""
        from .table import TABLE

        if not self.shares:
            return TABLE
        total = sum(t.share for t in TABLE)
        return tuple(replace(t, share=total * self.shares[t.name] / 100) for t in TABLE)

    @property
    def dir(self) -> Path:
        return OUT / self.name

    @property
    def data(self) -> Path:
        return self.dir / "data"

    def seed_rows(self) -> Path:
        from cjk_scale import paths

        return {"0921": paths.SEED_ROWS_0921, "0930": paths.SEED_ROWS}[self.seed]

    def scale_config(self):
        """The ``cjk_scale.train`` view of the run."""
        from cjk_scale.config import RunConfig

        return RunConfig(
            name=self.name, path=self.path, vocabs=self.rows, read=self.read
        )


def load(run: str) -> Run:
    path = CONFIGS / f"{run}.toml" if "/" not in run else Path(run)
    assert path.is_file(), f"no run config {path}"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    extra = sorted(set(raw) - set(KEYS))
    assert not extra, f"{path}: a run is {{{', '.join(KEYS)}}} — not {extra}"
    assert raw.get("seed") in ("0921", "0930"), f'{path}: seed is "0921" or "0930"'
    shares = raw.get("shares")
    if shares is not None:
        from .table import TABLE

        names = {t.name for t in TABLE}
        assert set(shares) == names, (
            f"{path}: shares names every tier — missing {sorted(names - set(shares))}, "
            f"unknown {sorted(set(shares) - names)}"
        )
        assert abs(sum(shares.values()) - 100) < 1e-9, f"{path}: shares sum to 100"
    return Run(
        name=path.stem,
        path=path,
        rows=tuple(raw["rows"]),
        read=tuple(raw.get("read", ())),
        seed=raw["seed"],
        steps_per_row=int(raw["steps_per_row"]),
        shares=dict(shares) if shares is not None else None,
    )
