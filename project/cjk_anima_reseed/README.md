# cjk_anima_reseed

The JA vocab pack's rows re-seeded cold. Why: `motivation2.md`; what an arm is judged on: `criteria.md`; the reads:
`reports/`.

## Code

| file | what |
|---|---|
| `run.py` | front door: `run.py <run> data [--frac f] \| train \| read` (`read`: the plain read against `READ_AGAINST` → `results/`) |
| `probe_split.py` | rows swapped at render, no training: two runs' rows split at a σ switch, one run's rows as stick / spikes, x̂0 per σ (`reports/probe_split_2026_10_04.md`); a cold arm's ball on retrain_kana's stick and back (`reports/ball_2026_10_04.md`); the ball runs as trained (`reports/ball_rk_2026_10_04.md`) |
| `stick_fit.py` | CPU: the stick runs' sticks, the kana / kanji burr, the band arms' sticks, the hiragana rows as stick + ball and the swap arms' EN-ref cos (`--legs ball`), scene numbers and sheets on cached renders (`reports/ball_2026_10_04.md`, `reports/stick_fit_2026_10_04.md`, `reports/stick_scene_2026_10_04.md`, `reports/stick_rk_2026_10_04.md`, `reports/stick_rk_jt50_2026_10_04.md`) |
| `ruler.py` | the dialogue ruler (`criteria.md`): `build` (CPU) draws 96 bubble-dialogue strings from the training set's captions, each with its own image's prompt and an EN reference line → `output/cjk_anima_reseed/ruler/ruler.json`; `run [--arms a,b] [--label l]` (GPU) renders what is missing — the floor (EN refs, retrain_kana, seed_retrain_0930) once — and reads every arm against it → `results/<ts>-ruler-<label>/` (`reports/ruler_2026_10_05.md`) |
| `probe_grad.py` | the scale line's `grad_identity` pass 2 on tiers drawn here (`reports/grid_64_2026_10_03.md`) |
| `configs/<run>.toml` | the run: `rows` (data.vocabs specs, single glyphs), `read` (held out of the windows), `seed` (`"0921"` / `"0930"`), `steps_per_row`, optional `shares` (% of the items per tier, Σ 100), optional `upper_shift` (added to every tier's upper σ edge, capped at 0.9; `kana_up`), optional `stick_from` (a stick run: that run's rows and data, the rows' mean trained only; `stick_*`), optional `rows_from` (a stick run: warm from that scale-line run's rows instead; `stick_rk_*`) and `drop_tiers` (tiers left out at train), optional `band` (a stick run: every kept item's σ band at train), optional `tag_drop` (a stick run: `[tag, p]`, the tag out of an item's caption at p per step; `stick_rk_fb_jt50`), optional `ball_on` (a ball run: the rows cold at that scale-line run's mean over them, the mean held, the rows less it trained; its merged rows the context; `ball_rk*`) and `data_from` (a ball run: a scale-line data dir instead of a build) |
| `reseed/table.py` | **the table**: one row per tier — recipe, share, σ band, glyph px, `px_keep` — and the scene knobs |
| `reseed/recipes.py` | `bubble1` / `bubbleN` / `grid` |
| `reseed/pools.py` | rows, scenes (+ the `s1s` pool, mono weighting), the windowed word pool |
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
