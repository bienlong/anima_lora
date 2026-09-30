# cjk_anima_scale — the JA vocab pack at scale

Opened 2026-09-23 out of [`../cjk_renderable_anima/`](../cjk_renderable_anima/).
That line is the research surface (probe code, `reports/`, `findings.md`);
this one is the production line that builds the pack on what it settled:
one loss, one trainer, σ per item from the band law. Since 2026-09-28 the
line is the **singles retrain** (`plan_retrain.md`); what came before it —
the stage chain, the 300-piece runs, line mode, plan_2900 — is under
`_archive/` (see its README).

## The vocab band law

[`band_experiment_results.md`](band_experiment_results.md) is the **vocab
band law**: which σ band a vocab-pack row trains in, keyed on what. It is not
a theory — every line of it is a measured read (EN ceiling on the base model,
JA training arms with two seeds), and it holds only over the sizes, layouts
and units those reads covered. As it stands:

- **The band is keyed on the row's glyph count.** Single-glyph rows train at
  0.7–0.9; multi-glyph one-token rows at 0.5–0.7. The two are two runs (or
  two per-item bands), never one band.
- **Rendered px sets the floor the band may reach**, not the band itself
  (§ 2, the per-px window table: 12–16 px text lives at 0.2–0.6, 48 px at
  0.5–0.7, 128 px at 0.8). A grid cell sits one step higher.
- **Nothing above 0.9** — 0.8–0.95 is dead at 48 px for kana and kanji alike.
- **Ink, stroke density and the bubble ellipse move no band.** Kanji take the
  kana band; density is an exposure / px question (§ 6 item 3, open).
- **16 px glyphs carry more caption leverage than 24 px**, lower in σ — small
  text is a band question, not a capability question.

The plans that produced it (`plan_band.md`, `plan_kanji.md`) closed on
2026-09-23 and were deleted; they are in git history at `f5cd4c0c`. The
probe line's step 1a / 1b / merge / step 2 recipe (`recipe.md`) was retired
the same day — `configs/stage*.toml` carry its settings, the band law § 3–4
its reads (git `ff2f70f9` has the last copy). The
reads themselves are the dated reports under `../cjk_renderable_anima/reports/`
(`cf_rebin_gate0`, `cf_band_a1`, `band_b1`, `cf_kanji_c1`, `band_c2_kanji`).
A new read that changes a row of the law goes into
`band_experiment_results.md`, with its report there.

## Files

| file | what |
|---|---|
| [`plan_retrain.md`](plan_retrain.md) | **the live plan**: the kanji budget, `retrain_kanji_b1..b4` chained by `context`, the new seed + `SEED_ROWS` move + floor re-render, the bake with routing on; open: doubling, the cut |
| [`retrain_experiments.md`](retrain_experiments.md) | the retrain's record (2026-09-28): why the singles re-seed cold (P1 / P1b), per-glyph routing (P2), the windowed word pool, checks C0–C3, `retrain_kana` trained and read, the code that landed |
| [`hypothesis.md`](hypothesis.md) | why a JA row does not compose (2026-09-27) — the P0 / P1 / P2 reads the retrain is built on |
| [`band_experiment_results.md`](band_experiment_results.md) | **the vocab band law** — the verdict, the per-px window table, the training reads, what is left unrun |
| [`plan.md`](plan.md) | the collapse spec the code implements: a run is one file, everything else is a rule |
| [`floor_score.md`](floor_score.md) | the floors on sent / target / word / en: the new seed's (`seed_retrain_0930`, what runs read against) and the old seed's (what the retrain was read against) |
| [`product_criteria.md`](product_criteria.md) | what a pack has to do to ship: the text axis and the page axis, dev set vs acceptance set |
| [`colab.md`](colab.md) | running `data` / `train` on a Colab VM (G4 = the kanji batches) |
| [`future.md`](future.md) | not planned: real images do not train rows, an OCR-reward update, token scaling as the last stage |
| [`idea.md`](idea.md) | not planned (2026-09-29): layout from the base's own text, identity at 0.3–0.5; the paste read (lone data teaches size) that motivated it |
| [`plan_polish.md`](plan_polish.md) | pilot only: `future.md` § 3 at 4 k tokens on a self-generated EN-anchor canvas; planned: SFX / small text on a text-free canvas in the post-b3 polish |
| `reports/` | the reads the live code cites: [`conflict_joint_2026_09_25.md`](reports/conflict_joint_2026_09_25.md) + [`grid_box_2026_09_25.md`](reports/grid_box_2026_09_25.md) (the trainer constants), [`piece_2026_09_25.md`](reports/piece_2026_09_25.md) (the piece ruler), [`long_b0_2026_09_27.md`](reports/long_b0_2026_09_27.md) (the long-piece budget row), [`stage_i_2026_09_26.md`](reports/stage_i_2026_09_26.md) (`b0709`, the cold-kanji budget), [`row_geometry_2026_09_28.md`](reports/row_geometry_2026_09_28.md) (the retrain rows in row space) |
| `configs/runs/*.toml` | the runs — `{vocabs, read[, context]}`: `retrain_kana`, `retrain_kanji_b1..b4`; `run0923_micro` / `run0925_300f` stay for the tests |
| `cjk_scale/` | the code (`windows` = the law, `config` = the run file + data pools, `recipes` + `builder` = data, `rows` + `train` = the fixed trainer, `budget`, `eval` = floor + trained on one sheet, `conflict`, `merge`, `ledger`); `scale.py` is the front door |
| `src/` | the stage packages the line runs on (render, readers, scoring, sheets, eval / native / target / cf_sense / scenes), vendored 2026-09-25 byte-faithful and pruned the same day; `src/run_stage.py` runs one by hand |
| `experiments/` | the retrain's experiments (`stage_b` stays as the module they import) — see its README |
| `assets/` | what `src/` reads: `fonts/` (binaries gitignored, `FONTS.md`), `vocabs/` (vocab files), `glyph_ink.json`, `target_prompts.txt` |
| `_archive/` | gitignored: the retired stage code, and the pre-retrain docs / reports / experiments / run files (see its README) |
| `runs/` | `ledger.jsonl` — every submitted job |

## Where it stands (2026-09-30)

`retrain_kana` (174 cold kana rows, routed) composes like `p1_mix` and holds
the singles; its rows are the kana half of the new seed.
`retrain_kanji_b1` (329 kanji on `retrain_kana`'s rows) trained on a Colab
G4, b2 and b3 locally (all unread); b4 (the training set's JA tail, the
first under the encode fold) trained 2026-09-30 and its rows are the new
seed, `output/cjk_anima_scale/seed_retrain_0930/` (`paths.SEED_ROWS`; the
old one is `SEED_ROWS_0921`). Its floor is in `floor_score.md` (acceptance
8 → 37 / 80), and it is baked with routing on and published as
`anima_cjk_vocab_pack_preview4` (Hub `sorryhyun/anima-vocab-pack-cjk`,
ComfyUI Adapter node ≥ 3.13.0). Details and numbers: `plan_retrain.md`, `retrain_experiments.md`.

## Running a run

```bash
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack   # MANGA109S comes from .env
.venv/bin/python project/cjk_anima_scale/scale.py retrain_kanji_b2 data              # CPU
.venv/bin/python project/cjk_anima_scale/scale.py retrain_kanji_b2 train --submit --queue
.venv/bin/python project/cjk_anima_scale/scale.py retrain_kanji_b2 eval --submit
.venv/bin/python project/cjk_anima_scale/scale.py retrain_kanji_b2 conflict --submit
.venv/bin/python project/cjk_anima_scale/scale.py windows | runs | ledger
```

A run is `configs/runs/<run>.toml` = `vocabs` (a vocabs file: one vocab per
line) + `read` (the `native_sent` strings). `data` draws the items by the
vocabs' kinds (singles → `scene_single` + `grid_single` at 0.7–0.9; pieces →
`scene_piece` at two px tiers, `grid_string`, `scene_short`,
`scene_sentence` at 0.5–0.7 / 0.3–0.5), ≈ 67 items per vocab, every item
stamped with its band; `train` trains the vocabs' rows from the seed rows
with every other row frozen at them, and saves `trained.pt` as the whole
merged rows — the seed's rows with the run's on top (μ 0, lr 1e-3, batch 4,
cosine, warmup 10 %, grid box, 90 steps/row — `cjk_scale/train.py`); `eval`
renders the floor (the seed rows' dir, a read cache shared by every run —
only the keys it lacks render) and the trained rows
(`trained.pt` in place — no `ctx/`) on
`word` / `single` / `en`, あ / い native, up to 8 trained pieces alone in a
native scene (`native_piece/`), the `read` strings and the target
captions, and writes one `sheet.png` + `reads.json`. Everything lands in
`output/cjk_anima_scale/<run>/`; `--submit` records the job in
`runs/ledger.jsonl`. The stage-shaped runs before 2026-09-25 stay on disk
as `{data,rows}_<stage>_<tag>/` records.

## Scene pools

The four pools the recipes draw on (`config.DATA["scenes"] = "s1,s1w,sl1w,ja_comic"`)
are grown, not rebuilt: the prompt stream is deterministic in `--seed`, so a
pool's own argv with a larger `--scene_n` keeps every stored row and renders
only the new indices (`src/scenes/stage.py`). `--scene_prune 1` deletes the
rejected renders (rows stay in `scenes_all.jsonl`); the pools were pruned on
2026-09-24 and every grow run prunes its own rejects. The argv per pool —
`S=project/cjk_anima_scale/src/run_stage.py --stage scenes`, raw pack
in the env, through `make daemon-run --stall-timeout 0` (pools land in
`output/cjk_anima_scale/scenes_<pool>/`):

| pool | argv after `--scene_tag <pool>` | grown to |
|---|---|---|
| `s1` | `--scene_frames reads_as,bubble_reads,saying,sign` | 2000 (2026-09-24) |
| `s1w` | `--seed 3 --scene_shapes 576x448,448x576,640x448,448x640,640x384,384x640 --scene_frames reads_as,bubble_reads,saying,sign` | 2600 |
| `sl1w` | `--seed 1 --scene_shapes 576x448,…,384x640 --scene_frames reads_as,bubble_reads,saying --scene_anchors <the 55 EN sentences: `sorted({r["anchor"]})` over the pool's `prompts.jsonl`>` | 2000 |
| `ja_comic` | `--seed 2 --scene_shapes 384x640,448x640,448x576 --scene_frames ja_reads_as,ja_bubble_reads,ja_saying --scene_extra_tags comic --scene_min_box 40` | 4400 |
| `s1s` | `--seed 4 --scene_frames reads_as,bubble_reads,saying,sign --scene_min_box 20` — the small-bubble pool (one-glyph fit p10 38 px vs s1's 57); not in `config.DATA`, only F2a′ adds it (`experiments/f2a_line`) | 1000 (2026-09-27), 390 kept |

The line's code is `cjk_scale/`; the stage packages it runs on are its
own `src/` (nothing is imported from another line); tests:
`.venv/bin/python -m pytest project/cjk_anima_scale/tests`.
