#!/usr/bin/env python
"""norm_p0 — hypothesis.md § 4 P0: render the norm arms (2026-09-27)

ctx_trigger c2 found the seed rows context-immune at the adapter, and most
of it norm: the seed rows' effective norm (≈ 252) is 1.2× the T5 table's
(212), and the llm_adapter blocks are pre-norm (a residual update's size
does not grow with the row's), so a loud row stays itself in a word. c2 is
adapter cos, not a render. P0 renders it, training-free, on Stage B's
held-out 10 (5 spelled words + 10 singles, en + swap — the floor keys
cached in ``native_spell/``), no ``v_line``.

Arms (every seed row; effective row = pack row + delta, as c2 built them;
the arm's delta is written so the hook sums to that row):

    x<a>      the seed rows × a (``--alphas``, default 0.8 0.65): direction
              kept, norm scaled — the roll-up's ``row_blocks_alpha`` crossing
              sat near × 0.7, on the delta of an older table
    raw       the pack rows alone (delta 0): the lower anchor, no identity
              training (c2 ``ja_spaced`` 0.657)

The seed itself (× 1) is the floor, already cached.

Decision (hypothesis.md § 4): words ≤ 1 edit up from the floor's 11 / 160
with singles official near 149 / 320 → H1 has render support, P1 next
(norm-bounded identity training; a norm clamp is not a closed geometry
penalty, user 2026-09-27). Words flat, singles down → the norm buys
identity, not a context read the DiT uses; the gate stands.

``--dry_run`` builds nothing and prints the arms' norms.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "f1_exp", LINE / "experiments" / "f1_line" / "run_exp.py"
)
F1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F1)
SB = F1.SB

EXP = OUT / "experiments"
CLAUSES = "en,swap"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label", default="p0")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.8, 0.65])
    p.add_argument("--no_raw", action="store_true", help="skip the raw anchor arm")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def arm_names(label: str, alphas, raw: bool) -> dict:
    out = {f"np_{label}_x{a:g}": a for a in alphas}
    if raw:
        out[f"np_{label}_raw"] = None
    return out


def seed_rows():
    """``(seed file, pack rows, effective rows)`` over the seed's ext ids."""
    import torch

    from library.anima import ext_vocab
    from library.anima.vocab_pack import resolve_pack_prefix

    sd = torch.load(SEED_ROWS, map_location="cpu", weights_only=False)
    d = sd["delta"]
    table, _ = ext_vocab.load_ext_assets(
        resolve_pack_prefix(os.environ["ANIMA_VOCAB_PACK"])
    )
    ids = torch.tensor([int(e) for e in d["ext_ids"]])
    pack = torch.as_tensor(table)[ids].float()
    eff = pack + d["raw"].float() * float(d["row_scale"])
    return sd, pack, eff


def build_arm(name: str, alpha, sd: dict, pack, eff) -> dict:
    """The seed file with its delta set so pack + delta = ``alpha`` × the
    seed's effective row (``alpha`` None: delta 0, the pack rows)."""
    import torch

    d = sd["delta"]
    s = float(d["row_scale"])
    v = pack if alpha is None else eff * float(alpha)
    raw = ((v - pack) / s).to(d["raw"].dtype)
    info = {
        "alpha": alpha,
        "row_norm_mean": round(float(v.norm(dim=1).mean()), 1),
        "seed_norm_mean": round(float(eff.norm(dim=1).mean()), 1),
        "pack_norm_mean": round(float(pack.norm(dim=1).mean()), 1),
    }
    dst = EXP / name
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **sd,
            "delta": {**d, "raw": raw},
            "arm": "rows",
            "seed_merged": str(SEED_ROWS),
            "norm_arm": info,
        },
        dst / "trained.pt",
    )
    return info


def main():
    args = parse_args()
    words = json.loads((OUT / SB.NAME / "donor_words.json").read_text("utf-8"))
    SB.set_words(words)
    ids = SB.check_spelling(SB.encoder(), words)
    names = arm_names(args.label, args.alphas, not args.no_raw)
    sd, pack, eff = seed_rows()
    held = [ids[c] for c in SB.HELD]
    at = {int(e): i for i, e in enumerate(sd["delta"]["ext_ids"])}
    hn = {c: round(float(eff[at[ids[c]]].norm()), 1) for c in SB.HELD}
    print(
        f"norm_p0: arms {names}; held-out seed norms {hn}; seed mean "
        f"{float(eff.norm(dim=1).mean()):.1f}, pack mean {float(pack.norm(dim=1).mean()):.1f}",
        flush=True,
    )
    metrics: dict = {"arms": {}, "held_seed_norm": hn, "held_ids": held}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "norm_p0", label=args.label, root=LINE / "experiments" / "norm_p0" / "results"
    )
    chars = SB.held_keys()
    floor = SB.check_floor(chars, CLAUSES)
    fh = SB.hits(floor, chars, CLAUSES)
    fl = F1.alone_line(floor, chars, CLAUSES)
    print("floor (the seed, × 1), held-out keys:", flush=True)
    metrics["floor"] = SB.tally(fh)
    metrics["floor"]["singles_line"] = sum(fl.values())
    print(f"  SINGLES alone as a line {sum(fl.values())} / {len(fl)}", flush=True)
    for name, alpha in names.items():
        info = build_arm(name, alpha, sd, pack, eff)
        print(f"{name}: {info}", flush=True)
        path = EXP / name
        reads = SB.native_read(path, path / "data", chars, CLAUSES)
        h = SB.hits(reads, chars, CLAUSES)
        ln = F1.alone_line(reads, chars, CLAUSES)
        m = {**SB.tally(h), "build": info}
        m["paired_vs_floor"] = SB.paired(h, fh)
        m["singles_line"] = sum(ln.values())
        m["singles_line_vs_floor"] = F1.paired_line(ln, fl)
        print(
            f"  vs floor {m['paired_vs_floor']}\n"
            f"  SINGLES alone as a line {m['singles_line']} / {len(ln)} "
            f"(vs floor {m['singles_line_vs_floor']})",
            flush=True,
        )
        metrics["arms"][name] = m
        write_result(  # after every arm: a killed job keeps what it read
            run_dir,
            script=__file__,
            args=args,
            label=args.label,
            metrics=metrics,
            artifacts=[str(EXP / n) for n in metrics["arms"]],
        )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
