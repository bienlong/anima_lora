# experiments — idea validation for the scale line, bench-style

One directory per experiment; each validates one idea before any
production code changes. This is the line's
`bench/`: same envelope, same discipline, but scoped to the line and free
to import `cjk_scale/` and the line's `src/` primitives. The pre-retrain
experiments (influence, 300f pieces, line mode, stage_i, plan_2900's micros,
the hypothesis probes) moved to `../_archive/experiments/` on 2026-09-28;
their reads are in `../_archive/reports/` and `../hypothesis.md`.

## Contract

- `<exp>/run_exp.py` — the entry point, a thin argparse script. `--dry_run`
  must plan (sample, count, print) without touching a model. An experiment
  that builds on another's loads it with `cjk_scale.paths.load_experiment`
  (fresh each call — its import-time `pin_old_seed()` runs again); scoring
  comes from `cjk_scale.reads`, not from another experiment.
- Results: `<exp>/results/<YYYYMMDD-HHMM>[-<label>]/` with the standard
  `result.json` envelope (`bench/_common.py::write_result`) + `report.md`.
  Always pass `--label` — same-minute runs overwrite the dir
  (`project_bench_run_dir_collision`).
- Heavy artifacts (gradient tensors, latent caches) go under
  `output/cjk_anima_scale/<exp>_<label>/`, referenced from `result.json`,
  never into this tree (it is committed). Row arms (a `trained.pt` the
  eval renders, one dir per arm: `tl_*`, `tp_*`) go under
  `output/cjk_anima_scale/experiments/<arm>/`, with the stage's
  `--arm_path` pointed there.
- GPU work goes through the daemon
  (`make daemon-run ARGS="project/cjk_anima_scale/experiments/<exp>/run_exp.py …"`)
  and every launch names the pack
  (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`).
- Never write into a `data_*` dir: sample records, keep sub-set
  latent/TE caches in the experiment's own output dir.
- A verdict that closes (or opens) an idea is written into
  `../reports/` like any other read; this tree holds
  the machinery and the raw envelopes, not the line's memory.

## Experiments

- `stage_b/` — kept as the module `p1_cap` / `p2_route` / `c3_kanji` import
  (its item builders, donor sets and native reads); its per-render scoring
  moved to `cjk_scale/reads.py` (2026-09-30), which every experiment reads
  arms with. The proposal's Stage B: 36 donor kana trained on 568
  manga109s lines (`scene_spelled`, glyph-balanced draw, no repeated glyph)
  plus a count tier (`scene_single_small`: one glyph at 24–40 px in a bubble
  it fills 0.2–0.4 of, 0.3 of b0507). `u_S` = the donors' mean tangential Δ,
  added at one coefficient to the seed rows of ひ ま わ り さ く ら み ど も
  (arms `tb_*`) plus a random ⟂ control, read on five spelled words made of
  them and the ten alone. The held-out keys' floor was rendered once into
  `native_spell/`. **Ran 2026-09-26** → composition transfers (≤ 1 edit
  11 → 66 / 160, random 9) and doubling with it (repeats 25 → 53 / 320)
  (`../_archive/reports/stage_b_2026_09_26.md`).
- `p2_route/` — retrain_experiments § 4 C0 / C2: per-glyph routing
  (`ANIMA_VOCAB_GLYPH_ROUTE`, set in-process) rendered against the spelled
  and the unrouted (piece-row) caption of the same word, same prompts ×
  seeds; routed renders only in each dir's `native_route/`. **Ran
  2026-09-28 (c0)** → `p1_mix` routed ≤ 1 edit 14 vs spelled 11 / 16, piece
  row 0 (`../retrain_experiments.md` § 4). **c2**: 8 held-in donor words on floor /
  Stage B / `p1_cold` / `p1_mix` / `p1_lone` → routed = spelled on every
  arm, `p1_mix` ≤ 1 edit 80 / 128 (floor 1, `p1_lone` 9).
- `c3_kanji/` — retrain_experiments § 4 C3: 36 kanji, cold, on `p1_mix`'s rows as
  context; windows of dialogue lines (2–4 glyphs, kanji-first draw) as the
  in-word tier, routed captions, `p1_mix`'s table, 225 steps / row.
  **Ran 2026-09-28** → words ≤ 1 edit 3 → 36 / 96; new kanji official
  0 → 51 / 192, the seed's dense kanji 65 → 31 (`../retrain_experiments.md` § 4).
- `retrain_read/` — retrain_experiments § 5: a retrain run's routed read on a
  smaller grid (4 prompts × 2 seeds = 8 renders / key), the floor and
  `p1_mix` from their caches of record (8 × 2 restricted to prompts < 4):
  the run's `read` words (en; C2's eight pair with the cached floor /
  `p1_mix` `native_route/`) + 8 hiragana singles (swap, floor from
  `native_spell/`) + 6 katakana (swap, floor rendered once into
  `native_r4_swap/`); ≈ 260 renders. **Ran 2026-09-28** (`kana`) → C2's
  words ≤ 1 edit 29 / 64 (`p1_mix` 34, floor 1), katakana words 20 / 32,
  singles at the floor, `dup` 42 vs `p1_mix`'s 31 (`../retrain_experiments.md` § 5).
- `row_geometry/` — the retrain rows in row space (CPU, no render):
  `retrain_kana` + `c3_kanji` 225 / 450 vs the seed, the raw pack, T5 and
  `u_S` — shared direction, glyph-pair structure, PR, Procrustes, row vs
  read. **Ran 2026-09-28** (`rg1`) → one shared direction (⟂ the pack,
  common across runs, cos 0.83), centered PR = the seed's, no rotation, no
  read predictor (`../reports/row_geometry_2026_09_28.md`).
- `real_kana/` — `retrain_kana`'s 174 kana rows warm-trained on the
  training set's kanji-free OCR images (captions verbatim, loss box = the
  quoted lines' union); `--native` (own size, batch 1, dynamic-seq compile),
  `--full_sigma` (the LoRA trainer's sigmoid σ, no band law), `--plain_mse`.
  **Ran 2026-09-28** (`native_sig12`, 291 images, 2 088 steps) → every read
  ≈ 0 vs `retrain_kana` (words official 16 → 0 / 104, singles contained
  73 → 14 / 112); plain MSE drifted further and was stopped (`../future.md` § 1).
- `target4k/` — the seed's `target` ruler at 768×1344 (the user's ComfyUI shape,
  4 032 tokens), beside the 512² cache. **Ran 2026-09-30** → 8 / 14 vs 5 / 14 at
  512², but every render is a small scene on a black canvas with the string as a
  subtitle under it. `--raw` (Δ 0) and `--base` (no pack: Japanese → `<unk>`)
  draw the same composition, 0 / 14. The letterbox is the base's, and with no pack
  the text sits in the bubbles (`../reports/polish_seed_2026_09_30.md`).
- `polish_seed/` — `polish_b1` (loaded as a module, constants patched) on
  `seed_retrain_0930`'s 1 362 singles, 4 steps / row, μ 0.1; `--read` = `sent`
  (4 prompts × 2 seeds) + `target` 512² / 4 k against the seed's routed floor;
  `--color` drops monochrome / line-art scenes. **Ran 2026-09-30** → sent official
  29 → 11 / 184, ≤ 1 edit 95 → 60, targets held (`../reports/polish_seed_2026_09_30.md`).
