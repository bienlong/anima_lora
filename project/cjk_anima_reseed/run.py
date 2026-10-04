#!/usr/bin/env python
"""reseed front door: ``run.py <run> data | train | read``.

    .venv/bin/python project/cjk_anima_reseed/run.py kana data              # CPU
    .venv/bin/python project/cjk_anima_reseed/run.py kana data --frac 0.02  # a look at the sizes
    make daemon-run ARGS="project/cjk_anima_reseed/run.py kana train"
    make daemon-run ARGS="project/cjk_anima_reseed/run.py kana read"

``data`` → ``output/cjk_anima_reseed/<run>/data``; ``train`` →
``…/<run>/trained.pt`` (the whole merged rows, ``cjk_scale.train``: the
run's rows cold, every other row frozen at its ``seed``). ``read`` (GPU) is
the scale line's ``experiments/grid_lone`` ``read_plain`` on the kana run's
13 words + 14 singles: the run paired against every reseed run read
before it and ``READ_AGAINST`` (renders cached in their dirs; the run's land in ``…/<run>/native_r4_plain/``) →
``results/<YYYYMMDD-HHMM>-<run>/result.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reseed import bootstrap  # noqa: E402

bootstrap()

# the arms of record a read pairs against (``output/cjk_anima_scale/…``)
READ_AGAINST = (
    "experiments/reseed_anchor_cold_kana_anchor",
    "experiments/reseed_recap_cold_kana_hp",
    "retrain_kana",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("verb", choices=["data", "train", "read"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--frac", type=float, default=1.0, help="data: share of every tier")
    a = p.parse_args()
    from reseed.config import load

    run = load(a.run)
    if a.verb == "data":
        assert not run.stick_from, f"{run.name}: a stick run trains on {run.data}"
        from reseed.builder import build

        build(run, a.workers, a.frac)
    elif a.verb == "read":
        read(run)
    else:
        assert a.frac == 1.0, "--frac is the data verb's"
        from cjk_scale import train as T

        T.train(
            run.scale_config(),
            data=run.data,
            out=run.dir,
            cold=not run.stick_from,
            steps_per_row=run.steps_per_row,
            context=run.seed_rows(),
            drop_tiers=run.drop_tiers,
            stick_only=bool(run.stick_from),
            band=run.band,
        )


def read(run) -> None:
    import os

    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # every render is routed
    from bench._common import make_run_dir, write_result
    from cjk_scale.paths import OUT as SCALE_OUT
    from cjk_scale.paths import load_experiment
    from reseed import HOME, OUT

    # every other reseed run already read (its plain renders cached), then the
    # scale line's arms of record
    read_before = {
        d.name: d
        for d in sorted(OUT.iterdir())
        if d != run.dir and (d / "native_r4_plain" / "native_reads.json").exists()
    }
    arms = (
        {run.name: run.dir}
        | read_before
        | {Path(d).name: SCALE_OUT / d for d in READ_AGAINST}
    )
    metrics = {"read_plain": load_experiment("grid_lone").read_plain(arms, "all")}
    run_dir = make_run_dir("cjk_anima_reseed", label=run.name, root=HOME / "results")
    write_result(
        run_dir,
        script=__file__,
        args={"run": run.name, "verb": "read"},
        label=run.name,
        metrics=metrics,
        artifacts=[str(run.dir)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
