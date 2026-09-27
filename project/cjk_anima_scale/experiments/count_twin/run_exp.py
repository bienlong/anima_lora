#!/usr/bin/env python
"""count_twin — proposal.md § 2.2: the count twin (2026-09-26)

Stage B's donor trained its 36 kana on in-line words plus a **count tier**
(one donor glyph alone at 24–40 px, weight 0.3 of b0507, ≈ 10 items / glyph)
and still rendered a line when a single was alone (102 / 144, floor 47). The
twin is the same donor with the count tier off: same donors, words, seed rows,
trainer and 90 steps / row; b0507 is all ``scene_spelled`` (b0305 is the same
group as Stage B's and draws the same items). The difference of the two
donors' Δ, each with its own-seed-row component removed, is the **count
direction** ``c`` — the first run of the mode-discovery recipe (§ 2.6):

- ``c`` ⟂ ``u_S`` → a separate component the count tier wrote, transplantable
  beside the line mode;
- ``c`` ≈ −``u_S`` → the count tier only shrinks the line mode.

Legs:
  data   (CPU) the twin's data dir (Stage B's recipes, no count tier)
  train  (GPU) the twin's rows → ``run0926_count_twin/trained.pt``
  build  (CPU) ``c`` per donor and its mean, split-half cos, cos to ``u_S``
         and to F1's ``v_line``, the twin's own shared direction
  read   (GPU) the twin on the donor keys (en: こんにちは spelled + 9
         singles) vs the floor and the plain Stage B donor, with the
         singles scored alone-as-a-line (either reader reads ≥ 3 kana /
         kanji) beside ``repeat``

``--dry_run`` prints the table and checks the encodings; ``--legs score``
(CPU) re-scores the reads on disk (the line metric on floor / Stage B / F1,
F1 report § 2: 47 / 102 / 94).
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
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage_b_exp", LINE / "experiments" / "stage_b" / "run_exp.py"
)
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)

NAME = "run0926_count_twin"
PLAIN = OUT / SB.NAME  # Stage B's donor, the count tier on
F1 = OUT / "run0926_f1_line"
GLYPH = re.compile(r"[ぁ-ゟァ-ヿ一-鿿]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "train", "build", "read", "score"],
    )
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def rc():
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=NAME, path=Path(__file__), vocabs=(f"chars:{SB.DONOR}",), read=()
    )


def table() -> tuple:
    """Stage B's table with the count tier folded into ``scene_spelled``."""
    from cjk_scale.builder import Group, Tier

    b0507, b0305 = SB.table()
    spelled = next(t for t in b0507.tiers if t.recipe == "scene_spelled")
    return (
        Group(
            b0507.name,
            b0507.kind,
            b0507.band,
            b0507.share,
            (Tier("scene_spelled", 1.0, spelled.params),),
        ),
        b0305,
    )


# ----------------------------------------------------------------------------
# the direction


def build(ids: dict) -> dict:
    import torch
    import torch.nn.functional as F

    seed, _ = SB.rows(SEED_ROWS)
    plain, _ = SB.rows(PLAIN / "trained.pt")
    twin, _ = SB.rows(OUT / NAME / "trained.pt")
    d_ids = [ids[c] for c in SB.DONOR]
    Tp = SB.tangential(plain, seed, d_ids)
    Tt = SB.tangential(twin, seed, d_ids)
    C = Tp - Tt  # the count tier's part, per donor
    u = SB.direction(ids)["u"]  # u_S, the plain donor's shared direction
    u_t = F.normalize(Tt.mean(0), dim=0)  # the twin's own
    c = C.mean(0)
    ch = F.normalize(c, dim=0)
    sd = torch.load(F1 / "trained.pt", map_location="cpu", weights_only=False)
    v = F.normalize(sd["delta"]["line"].float(), dim=0)
    Cn = F.normalize(C, dim=1)
    n = len(d_ids)
    info = {
        "c_norm": round(float(c.norm()), 2),
        "c_row_norm_mean": round(float(C.norm(dim=1).mean()), 2),
        "plain_row_norm_mean": round(float(Tp.norm(dim=1).mean()), 2),
        "twin_row_norm_mean": round(float(Tt.norm(dim=1).mean()), 2),
        # how much of the per-donor difference is shared
        "c_energy_frac": round(float(((C @ ch) ** 2).sum() / (C**2).sum()), 4),
        "c_pairwise_cos": round(float(((Cn @ Cn.T).sum() - n) / (n * (n - 1))), 4),
        "c_split_half_cos": round(
            float(
                F.normalize(C[0::2].mean(0), dim=0)
                @ F.normalize(C[1::2].mean(0), dim=0)
            ),
            4,
        ),
        "cos_c_uS": round(float(ch @ u), 4),
        "cos_c_v_line": round(float(ch @ v), 4),
        "cos_c_u_twin": round(float(ch @ u_t), 4),
        "cos_uS_u_twin": round(float(u @ u_t), 4),
        "c_proj_uS": round(float(c @ u), 2),  # < 0: the tier shrinks u_S
        "plain_step_uS": round(float((Tp @ u).mean()), 2),
        "twin_step_uS": round(float((Tt @ u).mean()), 2),
        "twin_step_u_twin": round(float((Tt @ u_t).mean()), 2),
        "twin_rel_drift": round(
            sum(float((twin[e] - seed[e]).norm() / seed[e].norm()) for e in d_ids)
            / n,
            3,
        ),
    }
    torch.save({"c": c, "C": C, "u_twin": u_t}, OUT / NAME / "count_dir.pt")
    return info


# ----------------------------------------------------------------------------
# scoring


def alone_line(path: Path, chars) -> dict:
    """Per single render: either reader reads ≥ 3 kana / kanji (a line of
    other glyphs, F1 report § 2) — keyed like ``stage_b.hits``."""
    from common.readers import norm

    out = {}
    for m in json.loads(path.read_text("utf-8")):
        if m["text"] not in chars or len(m["text"]) != 1 or m["clause"] != "en":
            continue
        reads = [
            norm(r.get(x) or "") for r in m.get("reads", []) for x in ("sfx", "vl")
        ]
        out[(m["text"], m["clause"], m["pi"], m["seed"])] = any(
            len(GLYPH.findall(r)) >= 3 for r in reads
        )
    return out


def score(name: str, path: Path, chars, fh=None, ph=None) -> dict:
    print(f"{name}, donor keys:", flush=True)
    h = SB.hits(path, chars, "en")
    out = SB.tally(h)
    ln = alone_line(path, chars)
    out["singles_line"] = sum(ln.values())
    out["singles_n"] = len(ln)
    print(f"  SINGLES alone as a line {out['singles_line']} / {len(ln)}", flush=True)
    for tag, other in (("floor", fh), ("plain", ph)):
        if other is not None:
            out[f"paired_vs_{tag}"] = SB.paired(h, other[0])
            o = other[1]
            keys = sorted(set(ln) & set(o))
            g = sum(ln[k] and not o[k] for k in keys)
            lo = sum(o[k] and not ln[k] for k in keys)
            out[f"paired_vs_{tag}"]["line"] = [g, lo]
            print(f"  vs {tag} {out[f'paired_vs_{tag}']}", flush=True)
    return out, (h, ln)


def main():
    args = parse_args()
    from cjk_scale import recipes

    recipes.RECIPES["scene_spelled"] = SB.scene_spelled
    recipes.RECIPES["scene_single_small"] = SB.scene_single_small
    words = json.loads((PLAIN / "donor_words.json").read_text("utf-8"))
    SB.set_words(words)
    tb = table()
    for g in tb:
        print(
            f"{g.name} σ {g.band}: {[(t.recipe, t.weight, t.params) for t in g.tiers]}",
            flush=True,
        )
    ids = SB.check_spelling(SB.encoder(), words)
    print(f"{NAME}: {len(words)} donor words (Stage B's), encodings ok", flush=True)
    metrics: dict = {"name": NAME, "n_words": len(words)}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "count_twin", label=args.label, root=LINE / "experiments" / "count_twin" / "results"
    )
    if "data" in args.legs:
        from cjk_scale.builder import build as build_data

        build_data(rc(), workers=args.workers, table=tb)
    if "train" in args.legs:
        from cjk_scale.train import train

        train(rc())
    if "build" in args.legs:
        info = build(ids)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        metrics["build"] = info
    if {"read", "score"} & set(args.legs):
        chars = SB.donor_keys()
        floor = SB.check_floor(chars, "en")
        metrics["floor"], fh = score("floor", floor, chars)
        rel = Path(f"native_{SB.TAG}") / "native_reads.json"
        metrics["plain"], ph = score("plain Stage B donor", PLAIN / rel, chars, fh)
        if "score" in args.legs:
            metrics["f1"], _ = score("F1 (rows + v_line)", F1 / rel, chars, fh, ph)
        if "read" in args.legs:
            path = SB.native_read(OUT / NAME, OUT / NAME / "data", chars, "en")
            metrics["twin"], _ = score("count twin", path, chars, fh, ph)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / NAME)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
