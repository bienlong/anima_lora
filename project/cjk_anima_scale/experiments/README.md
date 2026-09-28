# experiments — idea validation for the scale line, bench-style

One directory per experiment; each validates one idea before any
production code changes. This is the line's
`bench/`: same envelope, same discipline, but scoped to the line and free
to import `cjk_scale/` and the line's `src/` primitives. The two influence
experiments read the old stage-layout dirs (`data_<stage>_<tag>`) through
`cjk_scale/legacy.py` (the archived stage configs).

## Contract

- `<exp>/run_exp.py` — the entry point, a thin argparse script. `--dry_run`
  must plan (sample, count, print) without touching a model.
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
- **Line mode was removed 2026-09-28** (`v_line` / `ExtDelta.line`,
  `train(line_mode=, rows_frozen=)`, the pack's line block and
  `bake_vocab_pack --line_from / --line_fold`). The scripts that trained or
  read it — `f1_line/`, `f2a_line/`, `stage_i/`, `dense_a0/`,
  `kanji_mode/`, `line_pieces/`, `count_twin/`, `doubling_box/` — stay as
  the records of those reads and no longer run: `ExtDelta.load` refuses a
  delta carrying `line`.
- A verdict that closes (or opens) an idea is written into
  `../reports/` and the wake roll-up like any other read; this tree holds
  the machinery and the raw envelopes, not the line's memory.

## Experiments

- `influence_smoke/` — first contact for the gradient bank +
  validation influence: does `I[c, s] = v_sᵀ ḡ_c` rank the recipes the way
  the rulers did, and does the linearized prediction match the measured
  dev-loss change across the run0925_300f delta? **Ran 2026-09-25** →
  verdict in `../reports/influence_smoke_2026_09_25.md` (bank on hold:
  estimator needs row-conditioned sampling; the dev-loss target can't see
  the acceptance axis).
- `influence_target/` — step 1 after the smoke: does the dev target see what
  piece saw? Value-only (no bank): in-box FM loss on matched
  correct / doubled renders of the 8 piece pieces at the seed, step-5 000
  and final 300f tables. T1 = per-piece identity gain vs the piece gains
  (2-glyph bought, 3+ not); T2 = the doubled-vs-correct margin, the
  acceptance-axis read. **Ran 2026-09-25** (`--label t1`) → verdict in
  `../reports/influence_target_2026_09_25.md`: **both fail** — the loss gain
  ranks the unrendered long pieces first and すごい at zero, the doubling
  margin moves in no pattern. Loss-target influence closed; bank v2 not built.
- `parity_300f/` — plan.md § 6-3 without the retrain: replays run0925_300f
  through the one-file code (`plan` CPU = `--dry_run`: vocabs, trainer
  record, save-time merge; `steps`: the first N steps vs the old log; `eval`:
  the old rows under the new eval vs the old reads). **Ran 2026-09-26** →
  parity holds (`../reports/piece_only_2026_09_26.md` § 1).
- `piece_only/` — reports/next_2026_09_25.md § 4a (2): the 300 pieces on `scene_piece` items
  only (`build(…, table=)` with the piece groups cut to that tier), data →
  train → eval as `run0926_300f_sp`. **Ran 2026-09-26** → not the doubling
  lever (`../reports/piece_only_2026_09_26.md` § 2).
- `spell_b/` — the five singles あ り が と う trained on in-line real words
  (`scene_spelled`: unspaced image, spaced caption → each glyph its single
  row) at the piece bands, ありがとう held out; read spelled + alone.
  **Ran 2026-09-26** → composition bought, count lost
  (`../reports/spell_2026_09_26.md` § 4).
- `transplant_line/` — the shared "line" Δ, training-free: `transplant`
  (300f_sp's piece direction added to the seed singles), `strip` (spell_b's
  rows minus their shared component), `shared` (the seed plus only it); read
  on keys the floor cache holds — the eval refuses a missing floor key.
  **Ran 2026-09-26** → `../reports/transplant_line_2026_09_26.md`.
- `canvas/` — `plan_canvas.md` (the plan and its verdict, beside the
  script): does the base spell on a 512-token canvas? Training-free on the
  seed rows: per `--canvas` WxH, the EN control, native あ / い and the
  `cf_sense ja` identity peak (`d0`), or A.1's EN per-px ceiling (`d1`),
  each beside the 512² reads the seed dir already holds. **Ran 2026-09-26
  (d0)** → 256×512 / 512×256 pass (EN 23/24, identity kept, peak 0.7 ≈ 0.8;
  the tall canvas doubles); taken into `recipes.GRIDS` as `1x2` / `2x1`.
- `transplant_piece/` — the line-mode proposal's Stage A, training-free and out of
  sample: `u_P` = the mean tangential Δ of 292 of 300f_sp's pieces, added at
  one coefficient (their mean projection, 94.1) × α to the seed rows of the
  8 piece-ruler pieces, plus a random ⟂ control; read on the floor cache's
  `native_piece/` keys. 300f_sp is `scene_piece`-only, so no item carries a
  held-out piece beside a donor (leak 0). **Ran 2026-09-26** → transfers:
  contained 27 → 52 / 256 (300f_sp 76), random control 11
  (`../reports/transplant_piece_2026_09_26.md`).
- `stage_b/` — the line-mode proposal's Stage B: 36 donor kana trained on 568
  manga109s lines (`scene_spelled`, glyph-balanced draw, no repeated glyph)
  plus a count tier (`scene_single_small`: one glyph at 24–40 px in a bubble
  it fills 0.2–0.4 of, 0.3 of b0507). `u_S` = the donors' mean tangential Δ,
  added at one coefficient to the seed rows of ひ ま わ り さ く ら み ど も
  (arms `tb_*`) plus a random ⟂ control, read on five spelled words made of
  them and the ten alone. The held-out keys' floor was rendered once into
  `native_spell/`. **Ran 2026-09-26** → composition transfers (≤ 1 edit
  11 → 66 / 160, random 9) and doubling with it (repeats 25 → 53 / 320)
  (`../reports/stage_b_2026_09_26.md`).
- `f0_interaction/` — the factorized-rows proposal's F0, no DiT: the
  Qwen + `llm_adapter` forward at a glyph's position, alone vs in a spelled
  word, with only its own row changed (`self`) or every glyph's (`all`), on
  Stage B's u1 / random ⟂ / donor rows. **Ran 2026-09-26** → the adapter
  passes a row's change through blind to its neighbours (out cos
  0.95–0.97, gain 0.97; donor ≈ random), so the gate lives at the hook
  (`../reports/f0_interaction_2026_09_26.md`).
- `f1_line/` — the factorized-rows proposal's F1: Stage B's donor
  retrained with a gated `v_line` beside the rows (`train(…, line_mode=True)`;
  the hook adds it to pack rows in a run of ≥ 2). Arms `tf_*_ronly` (rows
  without it) and `tf_*_line` (seed rows + `v_line`). **Ran 2026-09-26** →
  `v_line` composes held-out words like `u_S` with singles at the floor,
  in-word `dup` 77; the rows keep the line layout alone
  (`../reports/f1_line_2026_09_26.md`). The `dose` leg (`--doses`) reads
  seed + d · `v_line`: at 0.5, ≤ 1 edit 80 / 160, `dup` 54 (report § 5).
- `doubling_box/` — proposal § 2.1 (a), CPU on the reads on disk: per word
  render the line box, px, orientation, the doubled glyph's class and
  position, per clause. **Ran 2026-09-26** → not a fill prior. B is
  word-final and bound to the `Japanese text` clause (`swap`: ≤ 1 edit
  40 = 40, in-word dup 7 vs 27), and a third of `dup` is reader noise
  (`../reports/doubling_box_2026_09_26.md`).
- `line_pieces/` — proposal § 2.4: seed + 0.5 · `v_line` on the sent and
  target rulers (the piece ruler's lone pieces never fire the gate).
  **Ran 2026-09-26** → runs of singles gain (はい 4 → 14 / 16). A piece in
  a run stays at 0, and the single beside it doubles
  (`../reports/line_pieces_2026_09_26.md`).
- `count_twin/` — proposal § 2.2: Stage B's donor with the count tier off
  (same donors, words, seed, trainer; b0305 item-identical). `C` = the two
  donors' tangential Δ difference, `c` its mean. The singles alone are
  scored with the line metric (≥ 3 glyphs read; `--legs score` reproduces
  F1's 47 / 102 / 94). **Ran 2026-09-26** → no count direction (split-half
  0.15, the twin's shared direction = `u_S` at cos 0.96). Alone-as-a-line
  108 vs 102 (`../reports/count_twin_2026_09_26.md`).
- `stage_i/` — proposal § 2.3: I0 (`b0709`) / I1 (scene, px spread, fill
  0.2–1.0) / I2 (small grid cells) on 12 cold kanji the seed lacks, read
  with their rows + 0.5 · `v_line` alone (12 kanji) and on six spelled
  words, floor in the seed dir's `native_stagei/`. **Ran 2026-09-26** →
  alone I0 ≥ I1 ≫ I2. Words 0 / 96 everywhere: underpowered at 90 steps /
  row (`../reports/stage_i_2026_09_26.md`).
- `long_b0/` — plan_2900 § 2's micro: 12 warm pieces (4 per glyph count)
  on the piece table at 90 vs 270 steps / row, read alone on the piece
  ruler's `native` stage (`en`), floor keys into the seed dir's
  `native_piece/`. **Ran 2026-09-27** → 4 glyphs need 270 (contained
  7 → 18 / 64), 3 glyphs stay at 90, 5 glyphs are not a budget problem
  (`../reports/long_b0_2026_09_27.md`; `budget.RULES`).
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
