# cjk_anima_reseed

The JA vocab pack's rows re-seeded cold. Why: `_archive/motivation2.md` (archived 10-05, with `motivation.md`); what an arm is judged on: `criteria.md`; the plan in hand: `sent_plan.md`; the reads:
`reports/`.

## Code

| file | what |
|---|---|
| `run.py` | front door: `run.py <run> data [--frac f] \| train \| read` (`read`: the plain read against `READ_AGAINST` → `results/`) |
| `probe_split.py` | rows swapped at render, no training: two runs' rows split at a σ switch, one run's rows as stick / spikes, x̂0 per σ (`reports/probe_split_2026_10_04.md`); a cold arm's ball on retrain_kana's stick and back (`reports/ball_2026_10_04.md`); the ball runs as trained (`reports/ball_rk_2026_10_04.md`) |
| `stick_fit.py` | CPU: the stick runs' sticks, the kana / kanji burr, the band arms' sticks, the hiragana rows as stick + ball and the swap arms' EN-ref cos (`--legs ball`), scene numbers and sheets on cached renders (`reports/ball_2026_10_04.md`, `reports/stick_fit_2026_10_04.md`, `reports/stick_scene_2026_10_04.md`, `reports/stick_rk_2026_10_04.md`, `reports/stick_rk_jt50_2026_10_04.md`) |
| `ruler.py` | the dialogue ruler (`criteria.md`): `build` (CPU) draws 96 bubble-dialogue strings from the training set's captions, each with its own image's prompt and an EN reference line → `output/cjk_anima_reseed/ruler/ruler.json`; `run [--arms a,b] [--label l]` (GPU) renders what is missing — the floor (EN refs, retrain_kana, seed_retrain_0930) once — and reads every arm against it → `results/<ts>-ruler-<label>/` (`reports/ruler_2026_10_05.md`) |
| `punct_pack.py` | CPU: the punct base pack (10-05, the green leaf) → `models/vocab_packs/anima_cjk_vocab_pack_punct`: the raw pack with `〜`→`～` and `―`→`ー` folded, dot runs (`ext_vocab.Dots`: `・`→`.`, 2–3 dots → `…`, 4+ → `……`, `...` / `…` only beside a routed char) and a `…` row appended (ext 69 558, at T5's `...`) |
| `probe_grad.py` | the scale line's `grad_identity` pass 2 on tiers drawn here (`reports/grid_64_2026_10_03.md`) |
| `configs/<run>.toml` | the run: `rows` (data.vocabs specs, single glyphs), `read` (held out of the windows), `seed` (`"0921"` / `"0930"`), `steps_per_row`, optional `shares` (% of the items per tier, Σ 100), optional `upper_shift` (added to every tier's upper σ edge, capped at 0.9; `kana_up`), optional `stick_from` (a stick run: that run's rows and data, the rows' mean trained only; `stick_*`), optional `rows_from` (warm from that scale-line run's rows — a stick run's instead of `stick_from`'s, `stick_rk_*` / `sent_stick`; without `stick_from` / `ball_on` a plain warm run, every row free — `sent_whole`) and `drop_tiers` (tiers left out at train), optional `band` (a stick run: every kept item's σ band at train), optional `tag_drop` (a stick run: `[tag, p]`, the tag out of an item's caption at p per step; `stick_rk_fb_jt50`), optional `ball_on` (a ball run: the rows cold at that scale-line run's mean over them, the mean held, the rows less it trained; its merged rows the context; `ball_rk*`) and `warm` (a ball run: the rows warm at `ball_on`'s, that mean held — `sent_ball`) and `data_from` (a ball or plain warm run: a scale-line data dir instead of a build; a bare run name = that reseed run's `data/` — `sent_ball_lr2`, `sent_whole`), optional `lr` (the rows' peak lr, `cjk_scale.train.LR` = 1e-3 without it; `sent_ball_lr2`), optional `pack` (`"punct"`: the run sits on `punct_pack.py`'s pack — its routing at build, train and read; a mark row (not a letter, not `・`) then trains in mark windows around seed_retrain_0930's letters and in `sent` lines holding it, hearts synthesised into the lines, and is drawn alone only if its lone spelling hits its row; `punct`) |
| `reseed/table.py` | **the table**: one row per tier — recipe, share, σ band, glyph px, `px_keep` — and the scene knobs |
| `reseed/recipes.py` | `bubble1` / `bubbleN` / `sent` / `grid` |
| `reseed/pools.py` | rows, scenes (+ the `s1s` pool, mono weighting), the windowed word pool, the dialogue lines (`sent`) |
| `reseed/builder.py` | one pass over the table → `output/cjk_anima_reseed/<run>/data` |

```bash
.venv/bin/python project/cjk_anima_reseed/run.py kana data --frac 0.03   # sizes: sheet_<tier>.png
.venv/bin/python project/cjk_anima_reseed/run.py kana data
make daemon-run ARGS="project/cjk_anima_reseed/run.py kana train"
make daemon-run ARGS="project/cjk_anima_reseed/run.py kana read"
make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/ruler.py run --arms kana_up --label kana_up"
.venv/bin/python -m pytest project/cjk_anima_reseed/tests
```

Renderers, scene pools, fonts and the trainer are `../cjk_anima_scale`'s
(`src/`, `cjk_scale.train`). Its `cjk_scale.builder` / `recipes` rebuild the
seed of record and are not imported here (`tests/test_boundary.py`).

## What the table is

`reseed_anchor --variant fit` (2026-10-03) flattened: `grid_44`'s grid / lone
sizes, `reseed_recap`'s shares and `hp` bands, `fit`'s bubble sizing
(`bubble1_32` at 0.5–0.8 of its bubble, `bubbleN_18` on `s1s` with
`cross_min`), mono scenes at 10 %, columns lettered (`tategaki` +
`vert_forms`), no window opening on `scene.NO_HEAD` or a small kana (`V_SMALL` — anchor's rule let ぁぃぅぇぉゎ through), plain grid captions. A
3 % build lands every tier's median px on `run1003_reseed_anchor_fit`'s.

Whole bubbles (10-03, after that build): the erase spares the bubble outline
(`render_into_scene(keep_outline=True)` — the interior mask holds the
outline, and the erase rectangle painted it away at its sides on 186 of
2 690 scenes), and `pools.whole_bubbles` drops the scenes whose anchor
bubble sits under `BUBBLE_EDGE_MIN` px from the canvas edge (500 of 2 480:
the edge cuts the outline) or whose erase would leave the letters (15).

Left out against the scale builder: the ！ / ？ marks and EN cells (the
`anchor` read: singles lower, words tied), the band law's gate and its groups
(each tier carries its band; `px_keep` is the size cut the gate made), the
rebuild passes (`derive` / `reband`), the per-row draw weights (one
`steps_per_row`), the piece / line tiers.

`grid_64` (10-03): 65 px grids at σ 0.65–0.8, the band read off the
gradient (`reports/grid_64_2026_10_03.md`), in the table at share 0 and
last — a run's `shares` turns it on (`kana_big`); a share-0 tier is not
drawn, so the runs before it build what they built.

`sent_34` / `sent_22` (10-05, share 0 — a run's `shares` turns them on): a
Manga109 dialogue line (8–10 / 8–14 cells) lettered in 2–3 columns of a
bubble at font px 26–36 / 16–22 (the ink px counts the column gaps; 20–28 /
13–19 read small). The
line: every ellipsis drawn `…` under 4 dots, `……` at 4 or more (T5 reads it
as `...`, no pack row); every other char routed to its own single row, as the
windows are; the `read` strings held by trigram, the dialogue ruler's 5+
glyph strings by 5-gram. The layout: columns cut at Qwen pieces, never on a
`…`; one column that holds the line, a block under 1.5× as tall as wide
(3×3, 3×4, 2×3), a region over 1.3× as wide as tall, a fill under 0.65 and a
sign frame are all re-drawn (`table.SENT_*`).
