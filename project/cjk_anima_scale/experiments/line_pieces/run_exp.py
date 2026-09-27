#!/usr/bin/env python
"""line_pieces — proposal.md § 2.4: does the run gate hurt pieces? (2026-09-26)

``v_line`` was trained on singles, but the gate fires on any run of pack rows,
so a caption with a piece next to another pack row gets it too. This reads
the operating point (the seed rows + 0.5 · ``v_line``, arm ``tf_l1_line0.5``,
built by ``f1_line``'s ``dose`` leg) on the rulers whose floor the seed dir
already caches, against that floor:

- **piece ruler** (``native_piece``): every key is one ext id (a lone piece),
  so the gate never fires and the arm renders the seed rows — the floor by
  construction. The ``plan`` leg asserts that; nothing is rendered.
- **sent ruler** (``native_sent``, ``en``): はい · おしい · やったネ ·
  ちょっと来い are runs (お+しい, や+った+ネ, ちょっと+来+い — pieces inside
  a run); こんにちは · ありがとう are one id each (gate off: the render-noise
  control).
- **target ruler** (``target``, the user's ComfyUI captions): はい × 4 clause
  shapes gate on, こんにちは × 3 gate off.

Legs:
  plan  (CPU) the ext ids of every key and whether the gate fires
  read  (GPU) sent + target on the arm, paired against the floor cache

``--dry_run`` = the plan leg only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, floor_dir  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage_b_exp", LINE / "experiments" / "stage_b" / "run_exp.py"
)
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)

ARM = OUT / "experiments" / "tf_l1_line0.5"
SENT_CLAUSES = "en"
DUP = re.compile(r"(.)\1")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--legs", nargs="+", default=["plan"], choices=["plan", "read"])
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def floor_texts(rel: str) -> list[str]:
    ms = json.loads((floor_dir() / rel).read_text("utf-8"))
    return list(dict.fromkeys(m["text"] for m in ms))


def plan(ext) -> dict:
    out = {}
    for ruler, rel in (
        ("piece", "native_piece/native_reads.json"),
        ("sent", "native_sent/native_reads.json"),
        ("target", "target/native_reads.json"),
    ):
        out[ruler] = {}
        for t in floor_texts(rel):
            ids = ext(t)
            out[ruler][t] = {"ids": ids, "gate": len(ids) >= 2}
            print(f"  {ruler:<6} {t:<8} ids {ids}  gate {'on' if len(ids) >= 2 else 'off'}")
    assert not any(v["gate"] for v in out["piece"].values()), out["piece"]
    return out


def hits(path: Path, texts, clause: str | None) -> dict:
    """``stage_b.hits`` per render plus ``wdup`` (a glyph of the text read
    doubled — in-word B, not reader noise) and ``exact``; keyed
    ``(text, clause, pi, seed)``."""
    from common.readers import norm

    ms = json.loads(path.read_text("utf-8"))
    cl = clause or "verbatim"
    base = SB.hits(path, texts, cl)
    out = {}
    for m in ms:
        if m["text"] not in texts or m.get("clause", "verbatim") != cl:
            continue
        k = (m["text"], m.get("clause", "verbatim"), m["pi"], m["seed"])
        t = norm(m["text"])
        reads = [
            norm(r.get(x) or "") for r in m.get("reads", []) for x in ("sfx", "vl")
        ]
        v = dict(base.get(k) or {})
        v["exact"] = any(r == t for r in reads)
        v["wdup"] = any(
            mt.group(1) in t for r in reads for mt in DUP.finditer(r)
        )
        out[k] = v
    return out


METRICS = ("official", "contained", "exact", "le1", "le2", "wdup", "dup")


def tally(h: dict, gate: dict) -> dict:
    per = {}
    for (text, _c, _pi, _s), v in h.items():
        c = per.setdefault(text, dict.fromkeys(("n", *METRICS), 0))
        c["n"] += 1
        for k in METRICS:
            c[k] += bool(v.get(k))
    for t, c in per.items():
        print(
            f"  {t:<8} gate {'on ' if gate[t] else 'off'}  "
            + "  ".join(f"{k} {c[k]:>2}" for k in METRICS)
            + f" / {c['n']}",
            flush=True,
        )
    return per


def paired(a: dict, b: dict, keep) -> dict:
    from math import comb

    keys = sorted(k for k in set(a) & set(b) if keep(k[0]))
    out = {"n": len(keys)}
    for m in METRICS:
        g = sum(bool(a[x].get(m)) and not b[x].get(m) for x in keys)
        lo = sum(bool(b[x].get(m)) and not a[x].get(m) for x in keys)
        n = g + lo
        p = (
            min(1.0, 2 * sum(comb(n, i) for i in range(min(g, lo) + 1)) / 2**n)
            if n
            else 1.0
        )
        out[m] = [g, lo, float(f"{p:.2g}")]
    return out


def read_arm(ruler: str, texts: list[str]) -> Path:
    from cjk_scale.config import RunConfig
    from cjk_scale.eval import READ_FILES, TRAINED_ARM, probe_args
    from stages import run as run_stage

    reads = ARM / READ_FILES[ruler]
    if reads.exists():
        print(f"  (read from disk: {reads})", flush=True)
        return reads
    rc = RunConfig(name="line_pieces", path=Path(__file__), vocabs=(), read=())
    if ruler == "sent":
        a = probe_args(
            rc,
            TRAINED_ARM,
            ["native"],
            [
                "--eval_tag",
                "sent",
                "--native_chars",
                ",".join(texts),
                "--native_clauses",
                SENT_CLAUSES,
            ],
        )
        stage = "native"
    else:
        a = probe_args(rc, TRAINED_ARM, ["target"])
        stage = "target"
    a.arm_path, a.data_path = str(ARM), str(ARM / "data")
    run_stage(stage, a)
    return reads


def main():
    args = parse_args()
    assert (ARM / "trained.pt").exists(), f"{ARM} — run f1_line's dose leg first"
    print("plan:", flush=True)
    pl = plan(SB.encoder())
    metrics: dict = {"arm": str(ARM), "plan": pl}
    if args.dry_run or "read" not in args.legs:
        return
    run_dir = make_run_dir(
        "line_pieces", label=args.label, root=LINE / "experiments" / "line_pieces" / "results"
    )
    from cjk_scale.eval import READ_FILES

    for ruler, clause in (("sent", SENT_CLAUSES), ("target", None)):
        texts = list(pl[ruler])
        gate = {t: v["gate"] for t, v in pl[ruler].items()}
        fh = hits(floor_dir() / READ_FILES[ruler], texts, clause)
        print(f"{ruler}: floor", flush=True)
        metrics[f"{ruler}_floor"] = tally(fh, gate)
        th = hits(read_arm(ruler, texts), texts, clause)
        print(f"{ruler}: seed + 0.5 · v_line", flush=True)
        metrics[f"{ruler}_line"] = tally(th, gate)
        for name, keep in (
            ("gate_on", lambda t, g=gate: g[t]),
            ("gate_off", lambda t, g=gate: not g[t]),
        ):
            pr = paired(th, fh, keep)
            metrics[f"{ruler}_paired_{name}"] = pr
            print(f"  paired vs floor, {name}: {pr}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(ARM)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
