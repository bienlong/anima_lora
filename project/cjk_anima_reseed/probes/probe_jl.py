#!/usr/bin/env python
"""probe_jl — the rows' Jacobian lens, probe 0 (`idea2.md`, 2026-10-07)

M = E[JᵀJ], J = ∂ v_θ(x_σ, c_JA; rows) / ∂ row, fitted label-free from
random output probes: a Gaussian v on the prediction, backpropagated once,
gives g = Jᵀv per row and E_v[g gᵀ] = JᵀJ. The probe masked to the item's
text box dilated by ``DIL`` cells (`out_mask`'s complement) gives M_in, on
the rest M_out; the cells are disjoint, so M_in + M_out = M.

No training: the rows sit still and the second moment is read.

- ``fit`` (GPU): the trainer's items in its own order (as `probe_pres`, one
  a batch), dealt round-robin to the bands × the two fits (A / B: disjoint
  items, fresh probes). In fp32, SDPA, eager: bf16 — FA2 or SDPA, compiled
  or not — puts the per-position gradient at cos 0.26–0.61 to fp32's on the
  rows (forward 1 % off), TF32 20 % off; fp32 repeats exactly. Per item one
  forward, then ``--probes`` probes on the box and as many off it, each one
  backward on the retained graph to
  ``llm_adapter.embed``'s output (after `ExtDelta`'s hook: the rows as
  rendered). g per position, summed over a row's occurrences in an item —
  the row parameter's per-item gradient, effective units (× row_scale on
  ``raw``) — kept per (item, row) pair → ``output/cjk_anima_reseed/
  probe_jl/<label>/g.pt``. The deal depends on the step only, and ε, σ and
  the probes come off a per-step generator, so ``--bands s080`` at other
  rows (drift) sees the same batches, noise and probes as the start fit.
- ``read`` (CPU): per band and family, M_in / M_out per fit; the A / B
  overlap ‖UᵀŨ‖²_F / k at k 16 / 64 (and with fit A cut to 5–50 items); the top-k
  trace shares; the generalised eigenproblem M_in u = λ (M_out + εI) u
  fitted on A and scored on B (in-sample λ rises on M_out's noise floor),
  against r = tr(M_in) / tr(M_out); the share of trained moves in the top-k
  of M_in, of M_out and of the high-λ subspace (isotropic: k / 1024) — the
  stick, h32 plain / p10 / Δ(p10 − plain), sent_kanji_f0 / sent_kanji_pres /
  Δ(pres − f0), each as its rows, its shared vector and the residual;
  ``--drift`` labels refit at other rows, their overlap with the start's →
  ``…/<label>/read.json``.

- ``split`` (CPU): the 0.5–0.7 band by the items' own σ — per-pair trace
  by quarter, the halves' lenses, pooled against trace-normalised.
- ``check`` (CPU): does the lens predict a read — per row, f0's move as
  M_in sees it against the glyph's ruler recall f0 − start; per string,
  Δ(pres − f0) as M_out sees it against the page reads; each beside the
  move's plain size.

Stop (idea2.md): A / B overlap under 0.8 at k 16, or no cross-fit λ over
2 r (no text-only subspace to project onto).

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_jl.py fit --label start"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_jl.py fit --label f0 --rows output/cjk_anima_reseed/sent_kanji_f0/trained.pt --bands s080"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_jl.py fit --label pres --rows output/cjk_anima_reseed/sent_kanji_pres/trained.pt --bands s080"
    .venv/bin/python project/cjk_anima_reseed/probes/probe_jl.py read --label start --drift f0,pres
    .venv/bin/python project/cjk_anima_reseed/probes/probe_jl.py read --label start --drift f0,pres --weight pair
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import bootstrap  # noqa: E402

bootstrap()

from reseed import OUT  # noqa: E402

PROBE = OUT / "probe_jl"
STICK080 = "output/cjk_anima_scale/seed_fixed_1005_stick080/trained.pt"
# σ per band: (lo, hi), drawn uniform per batch
BANDS = {
    "b57": (0.5, 0.7),
    "s080": (0.8, 0.8),
    "s090": (0.9, 0.9),
    "s095": (0.95, 0.95),
}
FITS = ("A", "B")
MASKS = ("in", "out")
KS = (16, 64)
# the moves read through the lens: label → (end file, start file or None for
# the file's own ``start``)
H32 = "output/cjk_anima_reseed/probe_pres_train/h32"
F0 = "output/cjk_anima_reseed/sent_kanji_f0/trained.pt"
PRES = "output/cjk_anima_reseed/sent_kanji_pres/trained.pt"
from cjk_scale.loss import PRES_DIL as DIL  # noqa: E402
from cjk_scale.loss import out_mask  # noqa: E402


def _slot(step: int) -> tuple[str, str]:
    k = step - 1
    return tuple(BANDS)[k % len(BANDS)], FITS[(k // len(BANDS)) % len(FITS)]


def _forward(anima, noisy, ts, cache, captions, device):
    """`common.models.dit_forward` in fp32: 4D latents → 4D prediction."""
    import torch

    def stacked(k):
        t = torch.stack([cache[c][k] for c in captions]).to(device)
        return t.float() if t.is_floating_point() else t

    b = len(captions)
    return anima(
        noisy.float().unsqueeze(2),
        ts,
        stacked(0),
        padding_mask=torch.zeros(b, 1, *noisy.shape[-2:], device=device),
        target_input_ids=stacked(2),
        target_attention_mask=stacked(3),
        source_attention_mask=stacked(1),
    ).squeeze(2)


def fit(
    run_name: str, rows_path: str, label: str, bands, items: int, probes: int
) -> None:
    import os
    import time
    from types import SimpleNamespace

    import torch
    from cjk_scale import train as T
    from cjk_scale.loss import glyph_count
    from common.models import checkpoints, encode_captions, ext_ids_of, gen_args
    from library.anima.vocab_pack import strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from reseed import REPO
    from reseed.config import load
    from train.stage import Batcher, LatentStore

    from cjk_scale.rows import Rows

    run = load(run_name)
    run.use_pack()
    rows_path = str(rows_path if Path(rows_path).is_absolute() else REPO / rows_path)
    assert Path(rows_path).is_file(), rows_path
    assert set(bands) <= set(BANDS), bands
    out = PROBE / label
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(T.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 is 20 % off fp32 here
    torch.backends.cudnn.allow_tf32 = False
    data = run.data
    recs, _ev, vocabs = T.load_items(data)
    keep = list(range(len(recs)))
    assert all(r["src"] == "scene" for r in recs), "a scene table: the box is drawn"
    bj = json.loads((data / "build.json").read_text(encoding="utf-8"))
    if bj.get("glyph_route"):
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, out)
    args.compile_blocks = False  # fp32 eager
    device = get_generation_settings(args).device
    # one item a batch: the fp32 model and its retained graph fit 16 GB at 1
    ns = SimpleNamespace(seed=T.SEED, batch=1, train_size=512)
    lat = LatentStore(ns, data, recs, keep, device)

    # the deal: every band's every fit gets ``items`` batches; a --bands
    # subset walks the same steps and skips the others
    n_b = items
    n_steps = len(BANDS) * len(FITS) * n_b
    batcher = Batcher(ns, recs, lat)
    order = [(s, list(batcher.next(s))) for s in range(1, n_steps + 1)]
    flat = [i for _s, b in order for i in b]
    assert len(flat) == len(set(flat)), "an item drawn twice: the fits overlap"
    todo = [(s, b) for s, b in order if _slot(s)[0] in bands]
    used = sorted({i for _s, b in todo for i in b})
    ja = sorted({recs[i]["caption"] for i in used})
    cache = encode_captions(ja, device, out / "te")
    touched = ext_ids_of(cache)
    print(
        f"{len(todo)} items ({n_b} per band × fit, bands {','.join(bands)}), "
        f"{len(touched)} ext rows touched",
        flush=True,
    )
    rc = run.scale_config()
    p = T.plan(rc, data, recs, vocabs, touched, Path(rows_path))

    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    rows = Rows(
        anima,
        device,
        p.idx,
        pack,
        warm=Path(rows_path),
        init_anchor=0.0,
        free_residual=0.0,
        lr=0.0,
        touched=p.touched,
        frozen=p.frozen,
        context=Path(rows_path),
    )
    delta = rows.delta
    n_rows, dim = delta.raw.shape
    live = ~rows.frozen_mask
    # bf16 (FA2 or SDPA, compiled or eager) puts the per-position gradient at
    # cos 0.26–0.61 to fp32's on the rows, its forward 1 % off: the lens is
    # read in fp32 (the bf16 weights are exact in it), SDPA, no autocast
    anima.attn_mode = "torch"
    anima.float()

    # the raw ids from a prepended pre-hook (the pack's own pre-hook remaps
    # the ext ids before a forward hook sees its args); embed's output from a
    # forward hook registered after ExtDelta's, so it sees the rows added
    cap: dict = {}

    def grab_ids(module, a):
        cap["n"] = cap.get("n", 0) + 1
        cap["ids"] = a[0]

    def grab(module, a, output):
        cap["emb"] = output

    embed = anima.llm_adapter.embed
    handles = [
        embed.register_forward_pre_hook(grab_ids, prepend=True),
        embed.register_forward_hook(grab),
    ]
    anima.train()
    got = []
    t0 = time.time()
    for n, (step, idx) in enumerate(todo, start=1):
        band, fit_ = _slot(step)
        gen = torch.Generator(device=device).manual_seed(
            1_000_003 * (T.SEED + 1) + step
        )
        lo, hi = BANDS[band]
        sigma = lo + (hi - lo) * float(torch.rand((), generator=gen, device=device))
        latents = lat[idx].to(device).float()
        noise = torch.randn(latents.shape, generator=gen, device=device)
        brecs = [recs[i] for i in idx]
        ts = torch.full((len(idx),), sigma, device=device, dtype=torch.float32)
        noisy = (1.0 - sigma) * latents + sigma * noise
        cap.clear()
        pred = _forward(anima, noisy, ts, cache, [r["caption"] for r in brecs], device)
        assert cap.get("n") == 1, f"embed ran {cap.get('n')} times in one forward"
        assert pred.dtype == torch.float32, pred.dtype
        ids, emb = cap["ids"], cap["emb"]
        mo = out_mask(pred.shape, brecs, pred.device, T.GRID_BOX)
        mi = 1.0 - mo

        # (item, row) pairs: a row's occurrences in an item, summed
        ext = ids >= delta.T
        bpos, lpos = ext.nonzero(as_tuple=True)
        loc = delta.lut[
            torch.clamp(ids[bpos, lpos] - delta.T, max=delta.lut.numel() - 1)
        ]
        k = loc >= 0
        bpos, lpos, loc = bpos[k], lpos[k], loc[k]
        uk, inv = torch.unique(bpos * n_rows + loc, return_inverse=True)
        occ = torch.bincount(inv, minlength=len(uk))

        def pair_g(v, pred=pred, emb=emb):
            g = torch.autograd.grad(pred, emb, grad_outputs=v, retain_graph=True)[0]
            return torch.zeros(len(uk), dim, device=device).index_add_(
                0, inv, g[bpos, lpos].float()
            )

        if n == 1:  # the hook against autograd on ``raw``, and a repeat
            v = torch.randn(pred.shape, generator=gen, device=device)
            agg = pair_g(v)
            again = float((pair_g(v) - agg).norm() / agg.norm())
            g_raw = torch.autograd.grad(
                pred, delta.raw, grad_outputs=v, retain_graph=True
            )[0]
            per_row = torch.zeros(n_rows, dim, device=device).index_add_(
                0, uk % n_rows, agg
            )
            want = per_row[live] * rows.row_scale * delta.scale
            assert len(uk) and float(g_raw[live].norm()) > 0, "no row in the batch"
            err = float((g_raw[live] - want).norm() / g_raw[live].norm())
            print(
                f"self-check: {len(uk)} pairs, |g_raw − Σ g_pair · row_scale| / |g_raw| "
                f"= {err:.2e}, a repeated backward {again:.2e}",
                flush=True,
            )
            assert err < 1e-4 and again < 1e-5, (err, again)

        G = torch.empty(len(uk), len(MASKS), probes, dim, dtype=torch.bfloat16)
        for j, mask in enumerate((mi, mo)):
            for q in range(probes):
                v = torch.randn(pred.shape, generator=gen, device=device) * mask
                G[:, j, q] = pair_g(v).to(torch.bfloat16).cpu()
        del pred, emb, pair_g  # the retained graph goes before the next forward
        cap.clear()
        got.append(
            {
                "step": step,
                "band": band,
                "fit": fit_,
                "sigma": sigma,
                "items": [int(i) for i in idx],
                "glyphs": [glyph_count(r["text"]) for r in brecs],
                "in_frac": mi.mean(dim=(1, 2, 3)).float().cpu().tolist(),
                "pair_item": (uk // n_rows).cpu(),
                "pair_row": (uk % n_rows).cpu(),
                "pair_occ": occ.cpu(),
                "G": G,
            }
        )
        if n % 50 == 0 or n == 1:
            print(
                f"{n}/{len(todo)} step {step} {band}/{fit_} σ {sigma:.3f}: {len(uk)} pairs, "
                f"|g| in {float(G[:, 0].float().norm(dim=-1).mean()):.3e} "
                f"out {float(G[:, 1].float().norm(dim=-1).mean()):.3e}, "
                f"{(time.time() - t0) / n:.2f} s/item",
                flush=True,
            )
    for h in handles:
        h.remove()
    st = delta.state_dict()
    torch.save(
        {
            "run": run_name,
            "rows": rows_path,
            "ext_ids": [int(e) for e in st["ext_ids"]],
            "live": live.cpu(),
            "raw": st["raw"].float().cpu(),
            "row_scale": rows.row_scale,
            "bands": list(bands),
            "n_b": n_b,
            "probes": probes,
            "dil": DIL,
            "batches": got,
        },
        out / "g.pt",
    )
    print(f"→ {out / 'g.pt'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------- read


CUTS = (5, 10, 25, 50)  # fit A cut to its first n items


def _moments(sd, fam_of: dict, weight: str = "none") -> tuple[dict, dict]:
    """One pass: (band, fit, family) → {"in", "out": E[g gᵀ] per (pair,
    probe), "n": pairs}, and (band, family, n) → the same over fit A's first
    n items (``CUTS``). ``weight`` "pair": each pair's g over its RMS across
    probes and both masks — the pairs' size tail off, their in / out split
    and directions kept."""
    import collections

    import numpy as np

    acc, cuts = {}, {}
    seen = collections.Counter()
    for b in sd["batches"]:
        seen[(b["band"], b["fit"])] += 1
        G = b["G"].float().numpy().astype(np.float64)
        if weight == "pair":
            G = G / np.sqrt((G**2).sum(-1).mean((1, 2)))[:, None, None, None]
        fam = np.array([fam_of.get(int(r), "") for r in b["pair_row"].tolist()])
        for f in ("kana", "kanji"):
            sel = fam == f
            if not sel.any():
                continue
            key = (b["band"], b["fit"], f)
            a = acc.setdefault(key, {"in": 0.0, "out": 0.0, "n": 0})
            for j, m in enumerate(MASKS):
                X = G[sel, j].reshape(-1, G.shape[-1])
                a[m] = a[m] + X.T @ X
            a["n"] += int(sel.sum())
        if b["fit"] == "A" and seen[(b["band"], "A")] in CUTS:
            for f in ("kana", "kanji"):
                a = acc.get((b["band"], "A", f))
                if a:
                    cuts[(b["band"], f, seen[(b["band"], "A")])] = dict(a)
    P = sd["probes"]
    for a in (*acc.values(), *cuts.values()):
        for m in MASKS:
            a[m] = a[m] / (a["n"] * P)
    return acc, cuts


_EIG: dict = {}


def _top(M, k):
    """Eigenvalues descending, the top-k eigenvectors (one eigh per matrix)."""
    import numpy as np

    key = id(M)
    if key not in _EIG or _EIG[key][0] is not M:
        w, V = np.linalg.eigh(M)
        _EIG[key] = (M, w[::-1].copy(), V[:, ::-1].copy())
    _, w, V = _EIG[key]
    return w, V[:, :k]


def _overlap(Q1, Q2) -> float:
    import numpy as np

    return float(np.linalg.norm(Q1.T @ Q2) ** 2 / Q1.shape[1])


def _gen(Mi, Mo, ridge):
    """M_in u = λ (M_out + εI) u, descending; ε = ridge · tr(M_out) / d."""
    import numpy as np
    from scipy.linalg import eigh

    d = Mo.shape[0]
    B = Mo + ridge * np.trace(Mo) / d * np.eye(d)
    w, V = eigh(Mi, B)
    return w[::-1], V[:, ::-1], B


def _orth(V, k):
    import numpy as np

    q, _ = np.linalg.qr(V[:, :k])
    return q


def _offsets(path: str) -> dict:
    """ext id → offset (effective units) of a trained file's rows."""
    import torch
    from reseed import REPO

    p = Path(path) if Path(path).is_absolute() else REPO / path
    d = torch.load(p, map_location="cpu", weights_only=False)
    if "delta" in d:
        d = d["delta"]
    rs = float(d["row_scale"])
    return {int(e): (r.float() * rs).numpy() for e, r in zip(d["ext_ids"], d["raw"])}


def _moves(start: dict) -> dict:
    """label → ext id → move (effective units)."""
    import torch
    from reseed import REPO

    out = {}
    for a in ("plain", "p10", "p10c09"):
        p = REPO / H32 / a / "rows.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        rs = float(d["row_scale"])
        out[f"h32_{a}"] = {
            int(e): ((r - s).float() * rs).numpy()
            for e, r, s in zip(d["ext_ids"], d["raw"], d["start"])
        }
    for name, path in (("f0", F0), ("pres", PRES)):
        if (REPO / path).exists():
            end = _offsets(path)
            out[name] = {e: end[e] - start[e] for e in end if e in start}
    for a, b in (("h32_p10", "h32_plain"), ("pres", "f0")):
        if a in out and b in out:
            out[f"{a}−{b}"] = {e: out[a][e] - out[b][e] for e in out[a] if e in out[b]}
    return out


def _share(X, Q) -> float:
    """Energy share of the rows X (n, d) inside span(Q)."""
    import numpy as np

    return float((np.linalg.norm(X @ Q) ** 2) / (np.linalg.norm(X) ** 2 + 1e-30))


def _sens(X, M) -> float:
    """Σ xᵀMx / (Σ‖x‖² · tr(M) / d): how much the image sees the move against
    an isotropic one of the same size."""
    import numpy as np

    d = M.shape[0]
    return float(_quad(X, M) / ((X**2).sum() * np.trace(M) / d))


def _quad(X, M):
    """Σ_n x_nᵀ M x_n (a matmul, not einsum's n·d² loop)."""
    return float(((X @ M) * X).sum())


def read(label: str, drift: list[str], ridge: float, weight: str = "none") -> None:
    import numpy as np
    import torch
    from probe_geom import _families

    sd = torch.load(PROBE / label / "g.pt", map_location="cpu", weights_only=False)
    ids = sd["ext_ids"]
    fam_of = {k: f for f, ks in _families(ids).items() for k in ks}
    acc, cuts = _moments(sd, fam_of, weight)
    d = sd["raw"].shape[1]
    start = {
        int(e): (r * float(sd["row_scale"])).numpy() for e, r in zip(ids, sd["raw"])
    }
    fam_ext = {
        f: {ids[k] for k, g in fam_of.items() if g == f} for f in ("kana", "kanji")
    }
    moves = _moves(start)
    out = {
        "label": label,
        "weight": weight,
        "ridge": ridge,
        "n_b": sd["n_b"],
        "probes": sd["probes"],
        "bands": {},
    }

    for band in sd["bands"]:
        out["bands"][band] = {}
        for f in ("kana", "kanji"):
            if (band, "A", f) not in acc or (band, "B", f) not in acc:
                continue
            A, B = acc[(band, "A", f)], acc[(band, "B", f)]
            rec = {"pairs": [A["n"], B["n"]]}
            Mp = {m: (A[m] * A["n"] + B[m] * B["n"]) / (A["n"] + B["n"]) for m in MASKS}
            Mp["all"] = Mp["in"] + Mp["out"]
            r = float(np.trace(Mp["in"]) / np.trace(Mp["out"]))
            rec["trace_ratio"] = r
            rec["trace"] = {m: float(np.trace(Mp[m])) for m in Mp}
            # stability: A against B, and fit A's first batches against B
            rec["overlap"] = {}
            for m in ("in", "out"):
                for k in KS:
                    rec["overlap"][f"{m}@{k}"] = _overlap(
                        _top(A[m], k)[1], _top(B[m], k)[1]
                    )
            curve = {}
            for upto in CUTS:
                a_part = cuts.get((band, f, upto))
                if a_part and upto < sd["n_b"]:
                    curve[upto] = {
                        m: _overlap(_top(a_part[m], 16)[1], _top(B[m], 16)[1])
                        for m in MASKS
                    }
            rec["overlap_by_items@16"] = curve
            # spectrum
            rec["topk_trace_share"] = {
                f"{m}@{k}": float(_top(Mp[m], k)[0][:k].sum() / np.trace(Mp[m]))
                for m in MASKS
                for k in KS
            }
            # λ: fitted on A, scored on B (and the reverse), against r
            lam = {}
            for src, dst in ((A, B), (B, A)):
                w, V, _ = _gen(src["in"], src["out"], ridge)
                Bd = dst["out"] + ridge * np.trace(dst["out"]) / d * np.eye(d)
                xs = ((dst["in"] @ V) * V).sum(0) / ((Bd @ V) * V).sum(0)
                for k in (1, 16, 64):
                    lam.setdefault(f"in_sample@{k}", []).append(float(w[:k].mean() / r))
                    lam.setdefault(f"cross@{k}", []).append(float(xs[:k].mean() / r))
            rec["lambda_over_r"] = {k: float(np.mean(v)) for k, v in lam.items()}
            rec["cross_max_over_r"] = rec["lambda_over_r"]["cross@1"]
            # the lens on pooled fits; high-λ from the pooled pair
            w, V, _ = _gen(Mp["in"], Mp["out"], ridge)
            Q = {}
            for k in KS:
                Q[f"in@{k}"] = _top(Mp["in"], k)[1]
                Q[f"out@{k}"] = _top(Mp["out"], k)[1]
                Q[f"hi_lambda@{k}"] = _orth(V, k)
            # the moves
            reading = {}
            fe = fam_ext[f]
            st = np.stack([start[e] for e in fe if np.linalg.norm(start[e]) > 0])
            sets = {"stick": st.mean(0, keepdims=True)}
            for name, mv in moves.items():
                xs = [mv[e] for e in mv if e in fe and np.linalg.norm(mv[e]) > 1e-6]
                if len(xs) < 2:
                    continue
                X = np.stack(xs)
                sets[f"{name}"] = X
                sets[f"{name}:shared"] = X.mean(0, keepdims=True)
                sets[f"{name}:residual"] = X - X.mean(0, keepdims=True)
            for name, X in sets.items():
                reading[name] = {
                    "rows": int(len(X)),
                    **{q: _share(X, Qm) for q, Qm in Q.items()},
                    "sens_in": _sens(X, Mp["in"]),
                    "sens_out": _sens(X, Mp["out"]),
                    "lambda_over_r": float(
                        _quad(X, Mp["in"]) / _quad(X, Mp["out"]) / r
                    ),
                }
            rec["isotropic"] = {k: k / d for k in KS}
            rec["moves"] = reading
            rec["stop"] = {
                "overlap_in@16<0.8": rec["overlap"]["in@16"] < 0.8,
                "overlap_out@16<0.8": rec["overlap"]["out@16"] < 0.8,
                "cross_lambda<2r": rec["cross_max_over_r"] < 2.0,
            }
            out["bands"][band][f] = rec

    # drift: band s080 refit at other rows, same batches / noise / probes
    out["drift"] = {}
    for dl in drift:
        dd = torch.load(PROBE / dl / "g.pt", map_location="cpu", weights_only=False)
        # its own row set (only its band's items encoded): families by ext id
        d_fam = {k: f for f, ks in _families(dd["ext_ids"]).items() for k in ks}
        dacc, _ = _moments(dd, d_fam, weight)
        out["drift"][dl] = {"rows": dd["rows"]}
        for band in dd["bands"]:
            for f in ("kana", "kanji"):
                if (band, "A", f) not in acc or (band, "A", f) not in dacc:
                    continue
                rec = {}
                for m in MASKS:
                    for k in KS:
                        sA, sB = (
                            _top(acc[(band, "A", f)][m], k)[1],
                            _top(acc[(band, "B", f)][m], k)[1],
                        )
                        dA, dB = (
                            _top(dacc[(band, "A", f)][m], k)[1],
                            _top(dacc[(band, "B", f)][m], k)[1],
                        )
                        rec[f"{m}@{k}"] = {
                            "start_A_B": _overlap(sA, sB),
                            "paired": float(
                                np.mean([_overlap(sA, dA), _overlap(sB, dB)])
                            ),
                            "crossed": float(
                                np.mean([_overlap(sA, dB), _overlap(sB, dA)])
                            ),
                        }
                hs, hd = [], []
                for src, tgt in ((acc, hs), (dacc, hd)):
                    for fi in FITS:
                        a = src[(band, fi, f)]
                        tgt.append(_gen(a["in"], a["out"], ridge)[1])
                for k in KS:
                    rec[f"hi_lambda@{k}"] = {
                        "start_A_B": _overlap(_orth(hs[0], k), _orth(hs[1], k)),
                        "paired": float(
                            np.mean(
                                [
                                    _overlap(_orth(hs[i], k), _orth(hd[i], k))
                                    for i in (0, 1)
                                ]
                            )
                        ),
                    }
                rec["trace_ratio"] = float(
                    np.trace(dacc[(band, "A", f)]["in"] + dacc[(band, "B", f)]["in"])
                    / np.trace(
                        dacc[(band, "A", f)]["out"] + dacc[(band, "B", f)]["out"]
                    )
                )
                out["drift"][dl][f"{band}|{f}"] = rec

    (
        PROBE / label / ("read.json" if weight == "none" else f"read_{weight}.json")
    ).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    _print(out)


def _print(out):
    for band, fams in out["bands"].items():
        for f, r in fams.items():
            o = r["overlap"]
            lam = r["lambda_over_r"]
            print(
                f"\n== {band} · {f}: pairs {r['pairs']}, tr(M_in)/tr(M_out) {r['trace_ratio']:.3f}"
            )
            print(
                "   A/B overlap  "
                + "  ".join(f"{k} {v:.3f}" for k, v in o.items())
                + "   fit A cut to n items @16: "
                + ", ".join(
                    f"{u}: in {c['in']:.2f} out {c['out']:.2f}"
                    for u, c in r["overlap_by_items@16"].items()
                )
            )
            print(
                "   top-k trace share  "
                + "  ".join(f"{k} {v:.3f}" for k, v in r["topk_trace_share"].items())
            )
            print("   λ / r  " + "  ".join(f"{k} {v:.2f}" for k, v in lam.items()))
            print(f"   stop: {[k for k, v in r['stop'].items() if v] or 'none'}")
            iso = r["isotropic"]
            print(
                f"   moves (share in top-k; isotropic {iso[16]:.3f} @16, {iso[64]:.3f} @64)"
            )
            for name, x in r["moves"].items():
                print(
                    f"     {name:22s} n {x['rows']:4d}  "
                    + "  ".join(
                        f"{q} {x[q]:.3f}"
                        for q in (
                            "in@16",
                            "out@16",
                            "hi_lambda@16",
                            "in@64",
                            "out@64",
                            "hi_lambda@64",
                        )
                    )
                    + f"  sens in {x['sens_in']:.2f} out {x['sens_out']:.2f}  λ/r {x['lambda_over_r']:.2f}"
                )
    for dl, recs in out["drift"].items():
        print(f"\n== drift {dl} ({recs['rows']})")
        for key, rec in recs.items():
            if key == "rows":
                continue
            print(f"   {key}: tr ratio {rec['trace_ratio']:.3f}")
            for m, v in rec.items():
                if isinstance(v, dict):
                    print(
                        f"     {m:14s} "
                        + "  ".join(f"{a} {b:.3f}" for a, b in v.items())
                    )


def _pooled(sd, fam_of, weight, band, lo=None, hi=None) -> dict:
    """family → {"in", "out", "n"} over both fits of ``band``, items with
    σ in [lo, hi) when given."""
    sub = dict(sd)
    sub["batches"] = [
        b
        for b in sd["batches"]
        if b["band"] == band and (lo is None or lo <= b["sigma"] < hi)
    ]
    acc, _ = _moments(sub, fam_of, weight)
    out = {}
    for (_b, _fit, f), a in acc.items():
        o = out.setdefault(f, {"in": 0.0, "out": 0.0, "n": 0})
        for m in MASKS:
            o[m] = o[m] + a[m] * a["n"]
        o["n"] += a["n"]
    for o in out.values():
        for m in MASKS:
            o[m] = o[m] / o["n"]
    return out


def split(label: str) -> None:
    """σ inside the 0.5–0.7 band: the per-pair trace by σ quarter, the upper
    half's share of the pooled trace, the halves' lenses against each other,
    and the pooled lens against the trace-normalised pool."""
    import numpy as np
    import torch
    from probe_geom import _families

    sd = torch.load(PROBE / label / "g.pt", map_location="cpu", weights_only=False)
    fam_of = {k: f for f, ks in _families(sd["ext_ids"]).items() for k in ks}
    out = {}
    for weight in ("none", "pair"):
        rec = out[weight] = {}
        q = [
            (0.5 + 0.05 * i, 0.5 + 0.05 * (i + 1) + (1e-9 if i == 3 else 0))
            for i in range(4)
        ]
        quart = [_pooled(sd, fam_of, weight, "b57", lo, hi) for lo, hi in q]
        halves = [
            _pooled(sd, fam_of, weight, "b57", 0.5, 0.6),
            _pooled(sd, fam_of, weight, "b57", 0.6, 0.7 + 1e-9),
        ]
        whole = _pooled(sd, fam_of, weight, "b57")
        for f in ("kana", "kanji"):
            r = rec[f] = {}
            r["trace_per_pair_by_quarter"] = {
                f"{lo:.2f}": {m: float(np.trace(Q[f][m])) for m in MASKS}
                | {"n": Q[f]["n"]}
                for (lo, _hi), Q in zip(q, quart)
            }
            lo_, up_ = halves[0][f], halves[1][f]
            for m in MASKS:
                t_lo, t_up = np.trace(lo_[m]) * lo_["n"], np.trace(up_[m]) * up_["n"]
                r[f"upper_share_{m}"] = float(t_up / (t_lo + t_up))
                r[f"halves_overlap_{m}@16"] = _overlap(
                    _top(lo_[m], 16)[1], _top(up_[m], 16)[1]
                )
                norm = lo_[m] / np.trace(lo_[m]) + up_[m] / np.trace(up_[m])
                r[f"pooled_vs_normalised_{m}@16"] = _overlap(
                    _top(whole[f][m], 16)[1], _top(norm, 16)[1]
                )
            r["trace_ratio_halves"] = [
                float(np.trace(h[f]["in"]) / np.trace(h[f]["out"])) for h in halves
            ]
            r["pairs_halves"] = [lo_["n"], up_["n"]]
    (PROBE / label / "split.json").write_text(json.dumps(out, indent=1))
    for w, rec in out.items():
        for f, r in rec.items():
            print(f"\n== {w} · {f}: pairs per half {r['pairs_halves']}")
            print(
                "   per-pair trace by σ quarter (in / out): "
                + ", ".join(
                    f"{k} {v['in']:.3f} / {v['out']:.3f}"
                    for k, v in r["trace_per_pair_by_quarter"].items()
                )
            )
            for m in MASKS:
                print(
                    f"   {m}: σ 0.6–0.7's share of the pooled trace {r[f'upper_share_{m}']:.2f}; "
                    f"halves' overlap @16 {r[f'halves_overlap_{m}@16']:.3f}; "
                    f"pooled vs trace-normalised @16 {r[f'pooled_vs_normalised_{m}@16']:.3f}"
                )
            print(
                f"   tr ratio by half {r['trace_ratio_halves'][0]:.3f} / {r['trace_ratio_halves'][1]:.3f}"
            )


RULER = "project/cjk_anima_reseed/results/20261007-2048-ruler-sensitive-sent_kanji_pres"
START_ARM = "seed_fixed_1005_stick080@punct"


def check(label: str, weight: str, ruler_dir: str) -> None:
    """Does the lens predict a read? Per row: f0's move as the box sees it
    (xᵀ M_in x) against the glyph's ruler recall, f0 − start; per string:
    Δ(pres − f0) over the string's rows as the page sees it (Σ xᵀ M_out x)
    against its page reads, pres − f0. Each beside the move's plain ‖x‖²."""
    import collections

    import numpy as np
    import torch
    from probe_geom import _char_rows, _families
    from reseed import REPO
    from scipy.stats import spearmanr

    sd = torch.load(PROBE / label / "g.pt", map_location="cpu", weights_only=False)
    ids = sd["ext_ids"]
    fam_e = {ids[k]: f for f, ks in _families(ids).items() for k in ks}
    fam_of = {k: f for f, ks in _families(ids).items() for k in ks}
    start = {
        int(e): (r * float(sd["row_scale"])).numpy() for e, r in zip(ids, sd["raw"])
    }
    f0, pres = _offsets(F0), _offsets(PRES)
    c2r = _char_rows()
    rd = Path(ruler_dir) if Path(ruler_dir).is_absolute() else REPO / ruler_dir
    R = json.loads((rd / "renders.json").read_text(encoding="utf-8"))
    arms = {
        a: {r["i"]: r for r in R[a]}
        for a in (START_ARM, "sent_kanji_f0", "sent_kanji_pres")
    }

    def hits(text, best):
        """(ext id, hit) per row occurrence in ``text`` (multiset against the read)."""
        have = collections.Counter(best)
        out = []
        for c in text:
            e = c2r.get(c)
            if e is None or int(e) not in fam_e:
                continue
            ok = have[c] > 0
            have[c] -= ok
            out.append((int(e), bool(ok)))
        return out

    M = {b: _pooled(sd, fam_of, weight, b) for b in ("b57", "s080", "s090")}
    res = {"weight": weight, "ruler": str(rd), "rows": {}, "strings": {}}

    # per row: recall f0 − start against f0's move
    rec = {a: collections.defaultdict(list) for a in arms}
    for a, by_i in arms.items():
        for r in by_i.values():
            for e, ok in hits(r["text"], r["best"]):
                rec[a][e].append(ok)
    for band in ("b57", "s080"):
        for f in ("kana", "kanji"):
            es = [
                e
                for e in rec[START_ARM]
                if fam_e[e] == f and e in f0 and e in rec["sent_kanji_f0"]
            ]
            if len(es) < 8:
                continue
            d_rec = np.array(
                [
                    np.mean(rec["sent_kanji_f0"][e]) - np.mean(rec[START_ARM][e])
                    for e in es
                ]
            )
            X = np.stack([f0[e] - start[e] for e in es])
            Mi = M[band][f]["in"]
            q_in = ((X @ Mi) * X).sum(1)
            nn = (X**2).sum(1)
            out = {"rows": len(es)}
            for name, v in (
                ("xMin x", q_in),
                ("|x|²", nn),
                ("xMin x / |x|²", q_in / nn),
            ):
                rho, pv = spearmanr(v, d_rec)
                out[name] = [float(rho), float(pv)]
            res["rows"][f"{band}|{f}"] = out

    # per string: the page reads pres − f0 against Δ(pres − f0) on its rows
    common = sorted(set(arms["sent_kanji_f0"]) & set(arms["sent_kanji_pres"]))
    for band in ("s080", "s090"):
        q_out, nn, dm = [], [], {k: [] for k in ("en_match", "en_tok_out", "iou_en")}
        for i in common:
            a, b = arms["sent_kanji_f0"][i], arms["sent_kanji_pres"][i]
            es = [e for e, _ok in hits(a["text"], a["best"]) if e in pres and e in f0]
            if not es:
                continue
            X = np.stack([pres[e] - f0[e] for e in es])
            q_out.append(
                sum(float(x @ M[band][fam_e[e]]["out"] @ x) for x, e in zip(X, es))
            )
            nn.append(float((X**2).sum()))
            for k in dm:
                dm[k].append(b[k] - a[k])
        out = {"strings": len(q_out)}
        for k, v in dm.items():
            for name, pred in (("Σ xMout x", q_out), ("Σ|x|²", nn)):
                rho, pv = spearmanr(pred, v)
                out[f"{k} ~ {name}"] = [float(rho), float(pv)]
            rho, pv = spearmanr(np.array(q_out) / np.array(nn), v)
            out[f"{k} ~ Σ xMout x / Σ|x|²"] = [float(rho), float(pv)]
        res["strings"][band] = out

    (PROBE / label / f"check_{weight}.json").write_text(json.dumps(res, indent=1))
    print(f"== rows: f0 − start glyph recall (Spearman ρ, p) — {weight}")
    for k, v in res["rows"].items():
        print(
            f"   {k:12s} n {v['rows']:4d}  "
            + "  ".join(
                f"{n} {rp[0]:+.3f} (p {rp[1]:.2g})"
                for n, rp in v.items()
                if n != "rows"
            )
        )
    print(f"== strings: pres − f0 page reads — {weight}")
    for band, v in res["strings"].items():
        print(f"   {band}: n {v['strings']}")
        for n, rp in v.items():
            if n != "strings":
                print(f"     {n:32s} {rp[0]:+.3f} (p {rp[1]:.2g})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["fit", "read", "split", "check"])
    p.add_argument("--ruler", default=RULER, help="check: a ruler result dir")
    p.add_argument("--run", default="sent_kanji_f0")
    p.add_argument(
        "--rows", default=STICK080, help="merged trained.pt, repo-relative or absolute"
    )
    p.add_argument("--bands", default=",".join(BANDS))
    p.add_argument("--items", type=int, default=100, help="items per band × fit")
    p.add_argument("--probes", type=int, default=8, help="per mask per batch")
    p.add_argument("--label", default="start")
    p.add_argument("--drift", default="", help="read: labels refit at other rows")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--weight", choices=["none", "pair"], default="none")
    a = p.parse_args()
    if a.verb == "fit":
        fit(a.run, a.rows, a.label, a.bands.split(","), a.items, a.probes)
    elif a.verb == "split":
        split(a.label)
    elif a.verb == "check":
        check(a.label, a.weight, a.ruler)
    else:
        read(a.label, [x for x in a.drift.split(",") if x], a.ridge, a.weight)


if __name__ == "__main__":
    main()
