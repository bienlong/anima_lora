# cjk_anima_reseed

The JA vocab pack's rows re-seeded cold. Why: `motivation2.md`; the reads:
`reports/`.

## Code

| file | what |
|---|---|
| `run.py` | front door: `run.py <run> data [--frac f] \| train` |
| `configs/<run>.toml` | the run: `rows` (data.vocabs specs, single glyphs), `read` (held out of the windows), `seed` (`"0921"` / `"0930"`), `steps_per_row` |
| `reseed/table.py` | **the table**: one row per tier — recipe, share, σ band, glyph px, `px_keep` — and the scene knobs |
| `reseed/recipes.py` | `bubble1` / `bubbleN` / `grid` |
| `reseed/pools.py` | rows, scenes (+ the `s1s` pool, mono weighting), the windowed word pool |
| `reseed/builder.py` | one pass over the table → `output/cjk_anima_reseed/<run>/data` |

```bash
.venv/bin/python project/cjk_anima_reseed/run.py kana data --frac 0.03   # sizes: sheet_<tier>.png
.venv/bin/python project/cjk_anima_reseed/run.py kana data
make daemon-run ARGS="project/cjk_anima_reseed/run.py kana train"
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

Left out against the scale builder: the ！ / ？ marks and EN cells (the
`anchor` read: singles lower, words tied), the band law's gate and its groups
(each tier carries its band; `px_keep` is the size cut the gate made), the
rebuild passes (`derive` / `reband`), the per-row draw weights (one
`steps_per_row`), the piece / line tiers.
