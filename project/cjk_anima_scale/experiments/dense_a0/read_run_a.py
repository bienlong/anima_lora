#!/usr/bin/env python
"""Run A (run0927_dense_k) on dense_a0's 12 glyphs, `en`, 16 each — trained side
only; scored against the cached floor (seed dir native_densea0/) and A0's arms."""

import importlib.util
import json
from pathlib import Path

REPO = Path("/home/sorryhyun/anima/anima_lora")
spec = importlib.util.spec_from_file_location(
    "dense_a0_exp", REPO / "project/cjk_anima_scale/experiments/dense_a0/run_exp.py"
)
A0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A0)  # bootstraps the line's sys.path
SI, OUT, TAG, MICRO = A0.SI, A0.OUT, A0.TAG, A0.MICRO
from cjk_scale.paths import floor_dir  # noqa: E402

RUN = OUT / "run0927_dense_k"
singles = list(MICRO)
SI.native_read(RUN, RUN / "data", singles)

src = {
    "floor": floor_dir() / f"native_{TAG}" / "native_reads.json",
    "a0": A0.arm_dir(1) / f"native_{TAG}" / "native_reads.json",
    "a0_x3": A0.arm_dir(3) / f"native_{TAG}" / "native_reads.json",
    "runA": RUN / f"native_{TAG}" / "native_reads.json",
}
H, out = {}, {}
for arm, f in src.items():
    if f.exists() and (h := SI.hits(f, singles)):
        H[arm] = h
        out[arm] = SI.tally(f"singles:{arm}", h)
for ref in ("floor", "a0", "a0_x3"):
    if ref in H:
        out[f"runA_vs_{ref}"] = pr = SI.paired(H["runA"], H[ref], SI.SINGLE_M)
        print(f"  runA vs {ref}: {pr}", flush=True)
for arm in H:
    per = out[arm]["per_key"]
    print(
        f"  per glyph {arm:<6} official/contained: "
        + " ".join(f"{t}{per[t]['official']}/{per[t]['contained']}" for t in MICRO if t in per),
        flush=True,
    )
(RUN / f"native_{TAG}" / "score.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
)
print(f"→ {RUN / f'native_{TAG}' / 'score.json'}", flush=True)
