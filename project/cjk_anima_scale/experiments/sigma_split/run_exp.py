#!/usr/bin/env python
"""sigma_split — the seed's rows gated by σ, no training (idea.md § 2b check 1 / § 3)

``idea.md``'s question before any data is built: can the rows act **below** σ
``--switch`` (default 0.5) on a layout the base laid out above it? The
sampler's commitment-σ switch (``generate_body``'s ``context_alt`` +
``tag_drop_sigma``: the conditional pass uses ``context`` while σ ≥ switch,
``context_alt`` below; CFG's negative pass untouched) does it with no new
sampler. The row Δ is added at encode (``ExtDelta`` on ``llm_adapter.embed``),
so each side is the same caption encoded with Δ scale 1 (**seed**) or 0
(**raw** = the pack rows exactly).

Arms, JA caption = the ``sent`` ruler's ``en`` clause
(``…, japanese text. Japanese text reads as "<k>".``):

- ``lo``  raw above, seed below — the product condition (§ 3 b): the base lays
  out with untrained pack rows, the rows only speak at σ < switch;
- ``hi``  seed above, raw below — the mirror;
- ``garble``  the garble caption above (``…, japanese text. She is saying
  something.``: no quote, so the base draws its own pseudo-Japanese — the
  caption idea.md's data would be rendered from), the JA caption with the
  seed rows below — the forced scaffold (§ 3 a): can the rows overwrite the
  base's own garble line?

Seed rows on both sides = the floor, read from the cache of record
(``seed_retrain_0930/routed/native_sent/``, the ``retrain_read`` grid: 4
prompts × 2 seeds × 23 strings = 184). ``--traj [--rows <arm>] [--traj_conds …] [--traj_sigmas …]`` decodes x̂0 per σ
(``--rows``: an arm's rows in place of the seed's, into ``traj_<arm>/``).
``--check`` renders one floor key
through the split path with seed on both sides and diffs it against the
cached file (the plumbing check). The seed trained singles at 0.7–0.9, so
``lo`` is a lower bound on what rows trained at 0.3–0.5 could do.

Reads per arm: official / ≤ 1 edit / dup (``cjk_scale.reads``, paired vs the
floor), EN-ref PE cos, and idea.md § 1's placement measures (``box`` = union
of the non-whole read boxes / canvas, ``box_h`` = tallest box / H,
``flat_white`` = share of 16² patches with std < 6 and mean > 225). One
sheet per string: EN ref | floor | lo | hi | garble.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label sigma_split \\
      project/cjk_anima_scale/experiments/sigma_split/run_exp.py --label s05"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the seed trained routed, reads routed
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

FLOOR = SEED_ROWS.parent / "routed"  # the seed's routed floor cache
FLOOR_READS = FLOOR / "native_sent" / "native_reads.json"
PROMPTS = 4  # retrain_read's grid: the first 4 scene prompts × 2 seeds
CLAUSE = "en"
GARBLE = "{p}, japanese text. She is saying something."  # every grid prompt has a girl
EN_CAPTION = '{p}, english text. English text reads as "{k}".'  # enref_caption's shape
SIZE, STEPS, CFG = 512, 28, 4.0  # the sent ruler's (cjk_scale.eval)
ENREF = OUT / "native_enref" / f"{SIZE}_{STEPS}_{CFG:g}"
# arm → (above the switch, below it); "seed" / "raw" = JA caption at Δ 1 / 0,
# "garble" = GARBLE (no ext row: Δ irrelevant)
ARMS = {"lo": ("raw", "seed"), "hi": ("seed", "raw"), "garble": ("garble", "seed")}


def floor_items() -> list[dict]:
    recs = json.loads(FLOOR_READS.read_text("utf-8"))
    items = [
        {k: m[k] for k in ("pi", "prompt", "text", "clause", "caption", "seed")}
        | {"floor_file": m["file"]}
        for m in recs
        if m["pi"] < PROMPTS and m["clause"] == CLAUSE
    ]
    assert len(items) == 184, f"{FLOOR_READS}: {len(items)} floor keys, not 184"
    return items


def arm_dir(switch: float, arm: str) -> Path:
    return OUT / "experiments" / f"sigma_split_s{switch:g}" / arm


def out_file(switch: float, arm: str, it: dict) -> Path:
    return (
        arm_dir(switch, arm)
        / "img"
        / f"{arm}_p{it['pi']:02d}_{it['text']}_{it['clause']}_s{it['seed']}.png"
    )


class Splitter:
    """The DiT + the seed Δ, rendering a caption pair split at σ."""

    def __init__(self, rows_dir: Path = SEED_ROWS.parent):
        import torch

        from common.hooks import ExtDelta
        from common.models import load_generator, load_trained, load_vae

        self.args, self.gen, self.device, self.shared = load_generator(
            SIZE, STEPS, CFG, OUT / "experiments" / "sigma_split_tmp"
        )
        self.anima = self.shared["model"]
        self.anima.eval()
        sd = load_trained(rows_dir)
        self.delta = ExtDelta.from_state(self.anima, sd["delta"], self.device)
        self.vae = load_vae(self.device)
        self.caches = {"seed": {}, "raw": {}}  # conds_cache per Δ scale
        self.torch = torch

    def encode(self, caption: str, side: str):
        from library.inference.text import prepare_text_inputs

        self.delta.scale = 0.0 if side == "raw" else 1.0
        self.shared["conds_cache"] = self.caches["raw" if side == "raw" else "seed"]
        a2 = self._args(caption, 0)
        return prepare_text_inputs(a2, self.device, self.anima, self.shared)

    def _args(self, caption: str, seed: int):
        import copy

        a2 = copy.deepcopy(self.args)
        a2.prompt, a2.seed = caption, seed
        return a2

    def render(
        self,
        fn: Path,
        it: dict,
        above: str,
        below: str,
        switch: float,
        x0s: dict | None = None,
    ):
        """``x0s``: filled with ``{step: (σ, x̂0 latent on CPU)}`` — the
        (CFG-combined) prediction ``x_t − σ·v`` the Euler step is taken from."""
        from common.models import decode_image
        from library.inference import generation as G
        from library.inference import sampling as S

        if fn.exists() and x0s is None:
            return
        cap = lambda side: (  # noqa: E731
            GARBLE.format(p=it["prompt"])
            if side == "garble"
            else EN_CAPTION.format(p=it["prompt"], k=it["en"])
            if side == "en"
            else it["caption"]
        )
        hi, null = self.encode(cap(above), above)
        lo, _ = self.encode(cap(below), below)
        body, step = G.generate_body, S.step
        G.generate_body = lambda *x, **k: body(
            *x, context_alt=lo, tag_drop_sigma=switch, **k
        )
        if x0s is not None:

            def rec(latents, noise_pred, sigmas, i):
                s = float(sigmas[i])
                x0s[i] = (s, (latents.float() - s * noise_pred.float()).cpu())
                return step(latents, noise_pred, sigmas, i)

            S.step = rec
        try:
            with self.torch.no_grad():
                lat = G.generate(
                    self._args(it["caption"], it["seed"]),
                    self.gen,
                    self.shared,
                    precomputed_text_data={"context": hi, "context_null": null},
                )
        finally:
            G.generate_body, S.step = body, step
        fn.parent.mkdir(parents=True, exist_ok=True)
        decode_image(self.vae, lat, self.device).save(fn)

    def decode(self, lat, fn: Path) -> None:
        from common.models import decode_image

        decode_image(self.vae, lat, self.device).save(fn)

    def free(self):
        del self.anima, self.vae, self.shared, self.delta
        self.torch.cuda.empty_cache()


def check(sp: Splitter, it: dict, switch: float) -> float:
    """Seed on both sides through the split path vs the cached floor render."""
    import numpy as np
    from PIL import Image

    fn = OUT / "experiments" / "sigma_split_tmp" / "check.png"
    fn.unlink(missing_ok=True)
    sp.render(fn, it, "seed", "seed", switch)
    a = np.asarray(Image.open(fn), np.float32)
    b = np.asarray(Image.open(it["floor_file"]), np.float32)
    d = float(np.abs(a - b).mean())
    print(f"check {Path(it['floor_file']).name}: mean |Δpx| {d:.3f}", flush=True)
    return d


def placement(m: dict) -> dict:
    """idea.md § 1: box (union of non-whole boxes / canvas), box_h (tallest /
    H), flat_white (16² patches, std < 6 and mean > 225)."""
    import numpy as np
    from PIL import Image

    im = np.asarray(Image.open(m["file"]).convert("L"), np.float32)
    H, W = im.shape
    mask = np.zeros((H, W), bool)
    hs = [0]
    for r in m.get("reads", []):
        if r.get("whole") or not r.get("box"):
            continue
        x0, y0, x1, y1 = (int(v) for v in r["box"])
        mask[max(0, y0) : y1, max(0, x0) : x1] = True
        hs.append(y1 - y0)
    p = im[: H // 16 * 16, : W // 16 * 16].reshape(H // 16, 16, W // 16, 16)
    std, mean = p.std(axis=(1, 3)), p.mean(axis=(1, 3))
    return {
        "box": float(mask.mean()),
        "box_h": max(hs) / H,
        "flat_white": float(((std < 6) & (mean > 225)).mean()),
    }


def read_arm(manifest: list, out: Path, device) -> None:
    from common.readers import Readers, hit, read_scored
    from common.text import CJK_RE
    from eval.enref import EnRef, enref_boxes

    rd = Readers(device)
    for m in manifest:
        reads = read_scored(rd, m)
        m["hit_sfx"], m["hit_vl"] = (
            hit(reads, m["text"], "sfx"),
            hit(reads, m["text"], "vl"),
        )
        m["exact"] = m["hit_sfx"] and m["hit_vl"]
        m["any_cjk"] = any(CJK_RE.search(r["sfx"] or "") for r in reads)
    enref = EnRef(device, ENREF, enref_boxes(ENREF, rd, device))
    del rd
    for m in manifest:
        sc = enref.score(m)
        m["en_cos"], m["en_cos_out"], m["box_iou"] = sc if sc else (None,) * 3
    (out / "native_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def sheets(items: list, arms: list, reads: dict, out: Path) -> None:
    """One sheet per string: rows = (prompt, seed), cols = EN ref | floor | arms."""
    from PIL import Image

    from common.readers import contact_sheet
    from eval.enref import enref_file

    out.mkdir(parents=True, exist_ok=True)
    by_key = {
        name: {(m["text"], m["pi"], m["seed"]): m for m in recs}
        for name, recs in reads.items()
    }
    for text in dict.fromkeys(it["text"] for it in items):
        rows = []
        for it in (i for i in items if i["text"] == text):
            k = (text, it["pi"], it["seed"])
            ref = enref_file(ENREF, it["pi"], it["seed"])
            rows.append((Image.open(ref).convert("RGB"), [f"EN ref p{k[1]} s{k[2]}"]))
            for name in ("floor", *arms):
                m = by_key[name].get(k)
                if m is None:
                    continue
                r0 = (m.get("reads") or [{}])[0]
                rows.append(
                    (
                        Image.open(m["file"]).convert("RGB"),
                        [
                            f"{name}{' ✓' if m.get('exact') else ''}",
                            f"sfx {(r0.get('sfx') or '')[:14]}",
                            f"vl {(r0.get('vl') or '')[:14]}",
                        ],
                    )
                )
        contact_sheet(rows, out / f"sheet_{text}.png", thumb=224, cols=2 + len(arms))


def summarize(
    items: list, arms: list, switch: float, label: str, check_d: float | None
) -> None:
    import statistics as st

    from cjk_scale import reads as R

    chars = sorted({it["text"] for it in items})
    keys = {(it["text"], it["clause"], it["pi"], it["seed"]) for it in items}
    floor_h = {k: v for k, v in R.hits(FLOOR_READS, chars, CLAUSE).items() if k in keys}
    recs = {
        "floor": [
            m
            for m in json.loads(FLOOR_READS.read_text("utf-8"))
            if (m["text"], m["clause"], m["pi"], m["seed"]) in keys
        ]
    }
    metrics: dict = {
        "switch": switch,
        "check_mean_abs_px": check_d,
        "floor": str(FLOOR_READS),
    }
    print("===== floor", flush=True)
    metrics["floor_tally"] = R.tally(floor_h)
    for arm in arms:
        f = arm_dir(switch, arm) / "native_reads.json"
        recs[arm] = json.loads(f.read_text("utf-8"))
        h = R.hits(f, chars, CLAUSE)
        print(f"===== {arm} {ARMS[arm]} @ σ {switch}", flush=True)
        metrics[arm] = {"tally": R.tally(h), "vs_floor": R.paired(h, floor_h)}
        print(f"  vs floor {metrics[arm]['vs_floor']}", flush=True)
    place = {}
    for name, ms in recs.items():
        ps = [placement(m) for m in ms]
        place[name] = {
            k: round(st.mean(p[k] for p in ps), 4)
            for k in ("box", "box_h", "flat_white")
        } | {
            k: round(st.mean(m[k] for m in ms if m.get(k) is not None), 4)
            for k in ("en_cos", "en_cos_out", "box_iou")
        }
        print(f"  placement {name:<6} {place[name]}", flush=True)
    metrics["placement"] = place
    sheet_dir = OUT / "experiments" / f"sigma_split_s{switch:g}" / "sheets"
    sheets(items, arms, recs, sheet_dir)
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(label=label, switch=switch),
        label=label,
        metrics=metrics,
        artifacts=[str(arm_dir(switch, a)) for a in arms] + [str(sheet_dir)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


# --traj: x̂0 per σ on a few samples — when layout / garble / identity appear
TRAJ_SIGMAS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0)  # 0 = the final image
TRAJ_TEXTS = {"こんにちは": "hello", "やったネ": "we did it"}  # JA → its EN pair
TRAJ_SEED = 0
# condition → (above the switch, below it)
TRAJ = {
    "ja_seed": ("seed", "seed"),  # the floor condition
    "ja_raw": ("raw", "raw"),  # the untrained pack rows throughout
    "lo": ("raw", "seed"),
    "garble_ja": ("garble", "seed"),
    "garble": ("garble", "garble"),  # the base's own line, no quote
    "en": ("en", "en"),  # EN identity: TRAJ_TEXTS' EN side
}


def rows_dir(rows: str) -> Path:
    """``""`` / ``seed`` = the seed rows, else an arm under ``OUT/experiments``."""
    return SEED_ROWS.parent if rows in ("", "seed") else OUT / "experiments" / rows


def traj_dir(switch: float, rows: str = "") -> Path:
    """``rows``: ``seed`` or an arm under ``OUT/experiments`` (``rows_dir``);
    ``""`` = the record's own dir."""
    d = OUT / "experiments" / f"sigma_split_s{switch:g}"
    return d / (f"traj_{rows}" if rows else "traj")


def traj(
    label: str,
    switch: float,
    rows_arm: str = "",
    conds: tuple = tuple(TRAJ),
    sigmas: tuple = TRAJ_SIGMAS,
) -> None:
    """Render each TRAJ condition on the 4 grid prompts × TRAJ_TEXTS (seed
    ``TRAJ_SEED``; ``garble`` once per prompt: its caption has no string),
    decode x̂0 at the step nearest each of ``TRAJ_SIGMAS``, read every decode,
    one sheet per condition (rows = samples, cols = σ)."""
    import statistics as st

    from common.readers import Readers, contact_sheet, hit, read_scored
    from common.text import lev, norm
    from PIL import Image

    items = {
        (it["text"], it["pi"]): it
        for it in floor_items()
        if it["text"] in TRAJ_TEXTS and it["seed"] == TRAJ_SEED
    }
    root = traj_dir(switch, rows_arm)
    sp = Splitter(rows_dir(rows_arm))
    manifest = []
    for cond, (above, below) in ((c, TRAJ[c]) for c in conds):
        for (text, pi), it in sorted(
            items.items(), key=lambda kv: (kv[0][1], kv[0][0])
        ):
            if cond == "garble" and text != next(iter(TRAJ_TEXTS)):
                continue
            it = it | {"en": TRAJ_TEXTS[text]}
            d = root / cond / f"p{pi:02d}_{text}_s{TRAJ_SEED}"
            d.mkdir(parents=True, exist_ok=True)
            x0s: dict = {}
            sp.render(d / "sig0.00.png", it, above, below, switch, x0s)
            steps = sorted(x0s)
            for target in sigmas:
                if target == 0.0:
                    fn, s = d / "sig0.00.png", 0.0
                else:
                    i = min(steps, key=lambda j: abs(x0s[j][0] - target))
                    s = x0s[i][0]
                    fn = d / f"sig{s:.2f}.png"
                    sp.decode(x0s[i][1], fn)
                target_text = (
                    it["en"] if cond == "en" else None if cond == "garble" else text
                )
                manifest.append(
                    {
                        "file": str(fn),
                        "cond": cond,
                        "pi": pi,
                        "text": text,
                        "target": target_text,
                        "sigma_target": target,
                        "sigma": s,
                    }
                )
            print(
                f"  traj {cond} p{pi} {text}: σ {[round(x0s[j][0], 2) for j in steps[:1]]}…",
                flush=True,
            )
    device = sp.device
    sp.free()
    rd = Readers(device)
    for m in manifest:
        reads = read_scored(rd, m | {"text": m["target"] or ""})
        m["reads"] = reads
        t = m["target"]
        if t:
            reader = ("vl",) if m["cond"] == "en" else ("sfx", "vl")
            m["hit"] = all(hit(reads, t, r) for r in reader)
            m["best_edit"] = min(
                (
                    lev(norm(r.get(x) or ""), norm(t))
                    for r in reads
                    for x in reader
                    if r.get(x)
                ),
                default=len(t),
            )
        m.update(placement(m))
    (root / "traj_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    metrics: dict = {
        "switch": switch,
        "rows": rows_arm,
        "sigmas": sigmas,
        "per_cond": {},
    }
    for cond in conds:
        per = {}
        for target in sigmas:
            ms = [
                m for m in manifest if m["cond"] == cond and m["sigma_target"] == target
            ]
            per[f"{target:g}"] = {
                "n": len(ms),
                "sigma": round(ms[0]["sigma"], 3),
                "hit": sum(bool(m.get("hit")) for m in ms),
                "le1": sum(m.get("best_edit", 99) <= 1 for m in ms),
                "box": round(st.mean(m["box"] for m in ms), 4),
                "box_h": round(st.mean(m["box_h"] for m in ms), 4),
            }
        metrics["per_cond"][cond] = per
        print(f"===== traj {cond} {TRAJ[cond]}", flush=True)
        for k, v in per.items():
            print(
                f"  σ {k:>3} (step σ {v['sigma']}): hit {v['hit']}/{v['n']}  ≤1 {v['le1']}  "
                f"box {v['box']}  box_h {v['box_h']}",
                flush=True,
            )
        rows = []
        for m in (m for m in manifest if m["cond"] == cond):
            r0 = max(
                m["reads"] or [{}], key=lambda r: len(r.get("vl") or ""), default={}
            )
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"p{m['pi']} {m['target'] or 'garble'} σ {m['sigma']:.2f}{' ✓' if m.get('hit') else ''}",
                        f"sfx {(r0.get('sfx') or '')[:14]}",
                        f"vl {(r0.get('vl') or '')[:14]}",
                    ],
                )
            )
        contact_sheet(
            rows, root / f"sheet_traj_{cond}.png", thumb=224, cols=len(sigmas)
        )
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(
            label=label, switch=switch, traj=True, rows=rows_arm, conds=conds
        ),
        label=label,
        metrics=metrics,
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--switch", type=float, default=0.5)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--traj", action="store_true", help="the x̂0-per-σ leg alone")
    ap.add_argument(
        "--rows",
        default="",
        help="traj: `seed` or an arm under OUT/experiments (its own traj_<rows> dir)",
    )
    ap.add_argument("--traj_conds", default=",".join(TRAJ))
    ap.add_argument("--traj_sigmas", nargs="+", type=float, default=list(TRAJ_SIGMAS))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.traj:
        conds = tuple(c for c in args.traj_conds.split(",") if c)
        assert set(conds) <= set(TRAJ), conds
        n = sum(1 if c == "garble" else len(TRAJ_TEXTS) for c in conds) * PROMPTS
        print(
            f"traj: {n} renders × {len(args.traj_sigmas)} σ → "
            f"{traj_dir(args.switch, args.rows)}",
            flush=True,
        )
        if not args.dry_run:
            traj(
                args.label,
                args.switch,
                args.rows,
                conds,
                tuple(args.traj_sigmas),
            )
        return
    arms = [a for a in args.arms.split(",") if a]
    assert set(arms) <= set(ARMS), arms
    items = floor_items()
    todo = {
        a: sum(not out_file(args.switch, a, it).exists() for it in items) for a in arms
    }
    print(
        f"{len(items)} floor keys ({len({i['text'] for i in items})} strings) · "
        f"switch σ {args.switch} · to render {todo}",
        flush=True,
    )
    for a in arms:
        print(f"  {a}: above {ARMS[a][0]} / below {ARMS[a][1]}", flush=True)
    if args.dry_run:
        return
    sp = Splitter()
    check_d = check(sp, items[0], args.switch)
    t0 = time.time()
    manifests = {}
    for a in arms:
        manifests[a] = []
        for n, it in enumerate(items):
            fn = out_file(args.switch, a, it)
            sp.render(fn, it, *ARMS[a], args.switch)
            manifests[a].append(
                {
                    "file": str(fn),
                    "cond": a,
                    "seed": it["seed"],
                    "pi": it["pi"],
                    "prompt": it["prompt"],
                    "text": it["text"],
                    "clause": it["clause"],
                    "caption": it["caption"],
                    "switch": args.switch,
                }
            )
            if n % 20 == 0:
                print(
                    f"  {a} {n}/{len(items)} · {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    device = sp.device
    sp.free()
    for a in arms:
        read_arm(manifests[a], arm_dir(args.switch, a), device)
        print(f"read {a}", flush=True)
    summarize(items, arms, args.switch, args.label, check_d)


if __name__ == "__main__":
    main()
