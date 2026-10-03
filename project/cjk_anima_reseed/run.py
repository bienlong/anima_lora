#!/usr/bin/env python
"""reseed front door: ``run.py <run> data | train``.

    .venv/bin/python project/cjk_anima_reseed/run.py kana data              # CPU
    .venv/bin/python project/cjk_anima_reseed/run.py kana data --frac 0.02  # a look at the sizes
    make daemon-run ARGS="project/cjk_anima_reseed/run.py kana train"

``data`` → ``output/cjk_anima_reseed/<run>/data``; ``train`` →
``…/<run>/trained.pt`` (the whole merged rows, ``cjk_scale.train``: the
run's rows cold, every other row frozen at its ``seed``). The reads stay
the scale line's (``experiments/grid_lone`` ``read_plain`` takes any arm dir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reseed import bootstrap  # noqa: E402

bootstrap()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("verb", choices=["data", "train"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--frac", type=float, default=1.0, help="data: share of every tier")
    a = p.parse_args()
    from reseed.config import load

    run = load(a.run)
    if a.verb == "data":
        from reseed.builder import build

        build(run, a.workers, a.frac)
    else:
        assert a.frac == 1.0, "--frac is the data verb's"
        from cjk_scale import train as T

        T.train(
            run.scale_config(),
            data=run.data,
            out=run.dir,
            cold=True,
            steps_per_row=run.steps_per_row,
            context=run.seed_rows(),
        )


if __name__ == "__main__":
    main()
