# count_twin — proposal § 2.2: the count twin (2026-09-26)

Stage B's donor retrained with the count tier off. Same 36 donors, 568
words, seed rows, trainer and 90 steps / row. b0507 is all `scene_spelled`
(1 200 items), and b0305 is item-for-item Stage B's. **Verdict: the count
tier as run wrote no count direction.** The difference between the two
donors is per-row, with no shared part (split-half cos 0.15, shared energy
4 %). The twin's own line direction is `u_S` itself (cos 0.956, the same
step). The tier's only effect on the singles alone is a marginal cut in
same-glyph repeats (twin 64 vs plain 50 / 144, p 0.09). Alone-as-a-line
does not move (108 vs 102, n.s.). The first run of the mode-discovery
recipe (§ 2.6) finds no mode on the count axis at this dose.

One job, `20260926-221234-fec68c` (train 23.8 min + read, 35 min). Script
`experiments/count_twin/run_exp.py`; envelopes
`experiments/count_twin/results/20260926-2212-gpu1/result.json` (train +
build + read) and `…-2211-score1` (the line metric reproducing F1's
47 / 102 / 94 on disk). Rows: `output/cjk_anima_scale/run0926_count_twin/`
(`count_dir.pt` = `c`, the per-donor `C`, the twin's `u`).

## 1. The direction (CPU, `build` leg)

`C` = plain donor's tangential Δ − the twin's, per donor (own seed row
removed from both); `c` = its mean.

| | value |
|---|---|
| per-donor ‖C_i‖ mean | 115 (the donors' Δ: plain 154, twin 158) |
| ‖c‖ | 23 |
| shared energy of `C` along `c` | 0.043 |
| pairwise cos of the C_i | 0.011 |
| split-half cos of `c` | 0.15 |
| cos(`c`, `u_S`) · cos(`c`, `v_line`) | 0.04 · −0.19 |
| cos(`u_S`, the twin's own shared direction) | **0.956** |
| step on `u_S`: plain · twin | 75.2 · 74.3 |

- The two donors differ by about 115 per row, three-quarters of a row's
  whole Δ, but in unrelated directions. That is trajectory divergence (the
  b0507 batch stream differs from item 84 on), not a component the tier
  wrote. `c` is not stable across halves (0.15), so there is nothing to
  transplant.
- **The line mode does not depend on the count tier.** The twin grows the
  same `u_S` at the same step. The tier neither shrinks it (projection
  of `c` on `u_S` +0.9) nor adds a direction beside it.

## 2. The donor keys (en, / 16 per key)

| | floor | plain (count 0.3) | **twin (count off)** | F1 (count 0.3 + `v_line`) |
|---|---|---|---|---|
| singles official / 144 | 91 | 43 | **40** | 50 |
| singles repeat | 29 | 50 | **64** | 35 |
| singles alone as a line (≥ 3 glyphs read) | 47 | 102 | **108** | 94 |
| こんにちは ≤ 1 edit | 0 | 6 | **5** | 8 |
| こんにちは ≤ 2 edits | 0 | 13 | **11** | 11 |
| こんにちは `dup` | 0 | 13 | **11** | 14 |

- Twin vs plain: repeat 36 / 22 (p 0.087), line 13 / 7, official 13 / 14,
  ≤ 2 edits 1 / 3. The only difference near significance is same-glyph
  repeat.
- Twin vs floor: line 62 / 1, official 10 / 59. Without the tier the
  singles break the same way.
- Per glyph the repeat rise is ん (8 → 11), あ (5 → 10), ち (3 → 8),
  と (7 → 9). The tier's ≈ 10 items / glyph bought a few same-glyph
  counts, and nothing about "one glyph, not a line".

## 3. What it decides

- **§ 2.2's count twin closes negative:** no count direction at weight 0.3
  and 24–40 px. The layout the rows carry alone (a line, 102–108 / 144) is
  written by the spelled-word data, whether or not the tier is present.
  F1's gated `v_line` took `u_S` out of the rows and moved it only to 94.
- **F1b** (the tier at b0305's ≈ 18 px, after a band-law row for singles at
  16–24 px) is the last untested count variant. This read lowers its prior:
  the tier moved same-glyph repeats and not the line, and F1b changes the
  tier's px, not its kind.
- The working assumption stands: **the seed row is a single's "alone"
  value, and trained rows are line-only.** Seed + 0.5 · `v_line` + the gate
  keeps that split without clean rows.
- § 2.6's recipe works as machinery (twin, difference, split-half), and the
  first axis it ran on shows no mode.
