"""A reseed run is one file, ``configs/<run>.toml``::

    rows = ["chars:あい…"]       # the rows trained cold (data.vocabs specs, single glyphs)
    read = ["こんにちは", …]      # held out of the windows by trigram
    seed = "0921"                # the rows every other row rides frozen at: "0921" (the
                                 # old seed, the kana run's) or "0930" (seed_retrain_0930)
    steps_per_row = 135
    shares = { grid_44 = 10, … }  # optional: % of the items per tier, every tier
                                  # named, Σ 100; else table.TABLE's shares
    upper_shift = 0.1             # optional: every tier's upper σ edge moved by this
                                  # (capped at 0.9)
    stick_from = "kana_up"        # optional: a stick run — that run's rows and data,
                                  # its rows' shared mean trained only (no data verb)
    rows_from = "retrain_kana"    # optional, stick runs: warm from this scale-line run's
                                  # merged rows (``output/cjk_anima_scale/<it>/trained.pt``)
                                  # instead of ``stick_from``'s; the data stays its
    drop_tiers = ["lone_44", …]   # optional: tiers left out of the data at train
    band = [0.75, 0.95]           # optional, stick runs: every kept item's σ band at
                                  # train (the data's stamped bands replaced; not
                                  # the table, so UPPER_MAX does not apply)
    tag_drop = ["japanese text", 0.5]  # optional, stick runs: the tag out of an item's
                                  # caption at this p, drawn per item per step
    ball_on = "retrain_kana"      # optional: a ball run — the rows cold at this
                                  # scale-line run's mean over them, the mean held, the
                                  # rows less it trained; its merged rows the context
    data_from = "run1002_grid_small/data"  # optional, ball runs: a scale-line data dir
                                  # (under ``output/cjk_anima_scale``) instead of a build

Its outputs land in ``output/cjk_anima_reseed/<run>/`` (``data/``,
``trained.pt``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from . import CONFIGS, OUT

KEYS = (
    "rows",
    "read",
    "seed",
    "steps_per_row",
    "shares",
    "upper_shift",
    "stick_from",
    "rows_from",
    "drop_tiers",
    "band",
    "tag_drop",
    "ball_on",
    "data_from",
)
UPPER_MAX = 0.9  # tests/test_boundary.py: no band past it


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    rows: tuple
    read: tuple
    seed: str
    steps_per_row: int
    shares: dict | None = None  # tier → % of the items
    upper_shift: float = 0.0  # added to every tier's upper σ edge
    stick_from: str = ""  # a stick run: warm from this run, its data, the mean trained
    rows_from: str = ""  # a stick run: warm from this scale-line run instead
    drop_tiers: tuple = ()  # left out of the data at train
    band: tuple | None = None  # every kept item's σ band at train (stick runs)
    tag_drop: tuple | None = None  # (tag, p): out of a caption at p (stick runs)
    ball_on: str = ""  # a ball run: the mean held at this scale-line run's
    data_from: str = ""  # a ball run: this scale-line data dir

    def table(self) -> tuple:
        """``table.TABLE``, its shares the run's when it gives them (Σ kept at
        the table's, so the items per row stay), every upper edge moved by
        ``upper_shift``."""
        from .table import TABLE

        tbl = TABLE
        if self.shares:
            total = sum(t.share for t in TABLE)
            tbl = tuple(
                replace(t, share=total * self.shares[t.name] / 100) for t in tbl
            )
        if self.upper_shift:
            tbl = tuple(
                replace(
                    t,
                    band=(
                        t.band[0],
                        round(min(t.band[1] + self.upper_shift, UPPER_MAX), 4),
                    ),
                )
                for t in tbl
            )
        return tbl

    @property
    def dir(self) -> Path:
        return OUT / self.name

    @property
    def data(self) -> Path:
        if self.data_from:
            from cjk_scale import paths

            return paths.OUT / self.data_from
        return (OUT / self.stick_from if self.stick_from else self.dir) / "data"

    def seed_rows(self) -> Path:
        """The rows the run sits on: a stick run's source rows (merged), else
        the seed's."""
        from cjk_scale import paths

        if self.rows_from:
            return paths.OUT / self.rows_from / "trained.pt"
        if self.ball_on:
            return paths.OUT / self.ball_on / "trained.pt"
        if self.stick_from:
            return OUT / self.stick_from / "trained.pt"

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
    from .table import TABLE

    drop = tuple(raw.get("drop_tiers", ()))
    assert set(drop) <= {t.name for t in TABLE}, f"{path}: drop_tiers {drop}"
    rows_from = raw.get("rows_from", "")
    if rows_from:
        assert raw.get("stick_from"), f"{path}: rows_from is a stick run's"
    band = raw.get("band")
    if band is not None:
        assert raw.get("stick_from"), (
            f"{path}: band is a stick run's (its data is built)"
        )
        assert len(band) == 2 and 0 <= band[0] < band[1] < 1, f"{path}: band {band}"
    tag_drop = raw.get("tag_drop")
    if tag_drop is not None:
        assert raw.get("stick_from"), f"{path}: tag_drop is a stick run's"
        assert (
            len(tag_drop) == 2 and isinstance(tag_drop[0], str) and 0 < tag_drop[1] < 1
        ), f"{path}: tag_drop {tag_drop}"
    ball_on, data_from = raw.get("ball_on", ""), raw.get("data_from", "")
    if ball_on:
        assert not raw.get("stick_from"), f"{path}: ball_on or stick_from, not both"
    if data_from:
        assert ball_on, f"{path}: data_from is a ball run's"
    return Run(
        name=path.stem,
        path=path,
        rows=tuple(raw["rows"]),
        read=tuple(raw.get("read", ())),
        seed=raw["seed"],
        steps_per_row=int(raw["steps_per_row"]),
        shares=dict(shares) if shares is not None else None,
        upper_shift=float(raw.get("upper_shift", 0.0)),
        stick_from=raw.get("stick_from", ""),
        rows_from=rows_from,
        drop_tiers=drop,
        band=tuple(band) if band is not None else None,
        tag_drop=tuple(tag_drop) if tag_drop is not None else None,
        ball_on=ball_on,
        data_from=data_from,
    )
