"""merge — runs trained from the same seed on disjoint vocabs → one merged
rows file (plan_2900 § 4, § 5-3).

    scale.py <out> merge <base run> <run> …

The first run is the base: its ``trained.pt`` (the whole merged rows, one
``row_scale``) is taken as is, and every other run's own rows — the idx of
its ``vocabs.json``, the rows its training moved — overwrite the base's at
those idx (appended, ids sorted, where the base lacks one: a vocab cold in
its run), rescaled into the base's units (raw × src ``row_scale`` / base
``row_scale``; the probe line's ``merge_tables`` rule, ported). Refused: a
run with no merged rows, runs from different seeds, a ``line`` mode (a
per-run vector, not a row), and idx two runs both trained.

Out: ``output/cjk_anima_scale/<out>/trained.pt`` + ``merge.json``; ``<out>``
must not hold rows yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import data_dir, run_dir, trained_path


def run_idx(run: str, tokq) -> set[int]:
    """The idx a run trained: its ``vocabs.json`` through the Qwen tokenizer
    (``train.vocab_idx``, the trainer's own split)."""
    from .train import vocab_idx

    f = data_dir(run) / "vocabs.json"
    assert f.is_file(), f"{run}: no {f} — the run's data dir names what it trained"
    return vocab_idx(json.loads(f.read_text(encoding="utf-8")), tokq)


def merge(out: str, runs: list[str]) -> Path:
    import torch
    from data.inventory import qwen_pieces

    assert len(runs) >= 2, "merge takes a base run and at least one more"
    assert len(set(runs)) == len(runs), f"a run named twice: {runs}"
    dst = run_dir(out)
    assert not (dst / "trained.pt").exists(), f"{dst} already holds rows"
    tokq = qwen_pieces()
    sds, idxs = {}, {}
    for r in runs:
        p = trained_path(r)
        assert p.is_file(), f"no rows at {p}"
        sd = torch.load(p, map_location="cpu", weights_only=False)
        assert sd.get("seed_merged"), f"{p}: not the whole merged rows — retrain"
        assert "line" not in sd["delta"], f"{p}: carries a line mode — not a row"
        assert sd.get("step") is None, f"{p}: stopped early at step {sd['step']}"
        sds[r], idxs[r] = sd, run_idx(r, tokq)
    seeds = {str(sd["seed_merged"]) for sd in sds.values()}
    assert len(seeds) == 1, f"runs from different seeds: {seeds}"
    seen: dict = {}
    for r in runs:
        both = {i: seen[i] for i in idxs[r] if i in seen}
        assert not both, (
            f"{r} and {sorted(set(both.values()))} both trained "
            f"{len(both)} idx (e.g. {sorted(both)[:8]}) — merge takes disjoint runs"
        )
        seen.update(dict.fromkeys(idxs[r], r))

    base = runs[0]
    d = sds[base]["delta"]
    ids = [int(e) for e in d["ext_ids"]]
    at = {e: j for j, e in enumerate(ids)}
    rows = list(d["raw"].float().clone())
    s0 = float(d["row_scale"])
    record = {
        "base": {
            "run": base,
            "path": str(trained_path(base)),
            "n_rows": len(idxs[base]),
            "row_scale": s0,
        },
        "seed_merged": next(iter(seeds)),
        "sources": [],
    }
    for r in runs[1:]:
        sd = sds[r]["delta"]
        k = float(sd["row_scale"]) / s0
        src = {int(e): j for j, e in enumerate(sd["ext_ids"])}
        miss = sorted(i for i in idxs[r] if i not in src)
        assert not miss, f"{r}: idx missing from its rows file: {miss[:8]}"
        n_new = 0
        for i in idxs[r]:
            row = sd["raw"][src[i]].float() * k
            if i in at:
                rows[at[i]] = row
            else:  # a vocab the seed (so the base) lacks: cold in its run
                at[i] = len(ids)
                ids.append(i)
                rows.append(row)
                n_new += 1
        record["sources"].append(
            {
                "run": r,
                "path": str(trained_path(r)),
                "n_rows": len(idxs[r]),
                "row_scale": float(sd["row_scale"]),
                "rescale": k,
                "added": n_new,
            }
        )
    order = sorted(range(len(ids)), key=ids.__getitem__)
    ids = [ids[j] for j in order]
    raw = torch.stack([rows[j] for j in order])
    record["n_rows_out"] = len(ids)
    record["n_trained"] = len(seen)
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **sds[base],
            "delta": {**d, "ext_ids": ids, "raw": raw},
            "merged_from": [str(trained_path(r)) for r in runs],
        },
        dst / "trained.pt",
    )
    (dst / "merge.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"merge → {dst / 'trained.pt'}: base {base} + "
        + ", ".join(f"{s['run']} ({s['n_rows']}, × {s['rescale']:.4f})" for s in record["sources"])
        + f"; {record['n_trained']} trained rows of {len(ids)}",
        flush=True,
    )
    return dst
