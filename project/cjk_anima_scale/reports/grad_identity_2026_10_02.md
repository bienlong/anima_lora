# grad_identity — how much of a cold row's gradient is glyph identity, by px × σ (2026-10-02)

`experiments/grad_identity` — a training-free read of `hypothesis.md` H1 on
the grid_44 train leg's initial point: the 81 hiragana + ー rows cold at the
pack rows, everything else at the old seed, routed, raw pack. Two passes, both
at the gradient of record (the box-share FM loss, `train.BOX_SHARE*`,
`GRID_BOX`, plain MSE on a lone 1×1), σ swept over 0.2 … 0.95 with one ε per
(item, σ).

**Answer.** Where the identity signal leaves the gradient, by px tier:

| tier | slot px | identity peak (σ of max ‖I‖) | identity share at half its plateau | share < 0.1 from | EN ceiling (§ 2: grid cell / flat letter, peak · live) | grid_44 band | share across that band | b7593 (0.75–0.93) |
|---|---|---|---|---|---|---|---|---|
| `grid_16` | 15 | 0.4 | ≈ 0.62 | 0.7 | 16 px: 0.5 · 0.4–0.6 / 0.4 · 0.2–0.6 | 0.3–0.5 | 0.31 → 0.19 | 0.08 → 0.01 |
| `grid_29` | 28 | 0.5 | ≈ 0.72 | 0.8 | 24–32 px: 0.5–0.6 · 0.4–0.7 / 0.5 · 0.35–0.7 | 0.5–0.7 | 0.32 → 0.19 | 0.12 → 0.03 |
| `grid_44` | 44 | 0.6 | ≈ 0.76 | 0.9 | 48 px: 0.8 · 0.7–0.8 / 0.6 · 0.5–0.7 | 0.7–0.9 | 0.36 → 0.07 | 0.25 → 0.07 |
| `lone_16` | 15 | flat (‖I‖ 2–5 × 1e-3) | ≈ 0.53 | 0.75 (0.11 at 0.8) | 16 px flat: 0.4 · 0.2–0.6 | 0.3–0.5 | 0.60 → 0.35 | 0.10 → 0.02 |
| `lone_28` | 28 | ≈ 0.75 (‖I‖ ≤ 8 × 1e-3) | ≈ 0.66 | 0.85 | 24–32 px flat: 0.5 · 0.35–0.7 | 0.5–0.7 | 0.69 → 0.19 | 0.16 → 0.04 |
| `lone_44` | 45 | 0.7–0.8 (‖I‖ ≤ 13 × 1e-3) | ≈ 0.77 | 0.9 | 48 / 64 px flat: 0.6 · 0.5–0.7 / 0.7 · 0.6–0.8 | 0.7–0.9 | 0.66 → 0.07 | 0.42 → 0.07 |

"Identity share" = f, the share of row u's gradient energy that changes when
the glyph drawn in its slot changes (pass 2, § 3); ‖I‖ = ‖g‖·√f, its size.
Ceiling px are font px, slot px are ink px (≈ 0.8 × font px for these
renders); the law keys on ink px, as here.

- **The true glyph never separates from the layout in the brief's sense.**
  In no tier, at no σ, plain or clause, is the drawn glyph's row (pass 1) or
  the drawn glyph's image (pass 2) more distinct from the rest than a wrong
  one is: no (tier, σ) cell has the excess over the null above zero at
  p < 0.01, and pooled the sign is the other way. At the cold start no row
  matches its glyph yet, so "the right one" is not a contrast the gradient
  carries.
- **What the gradient does carry is how much it depends on the glyph drawn**,
  and that is the H1 quantity. Holding the row fixed and swapping the glyph
  in the image (pass 2), f sits on a plateau at low σ and falls with σ; it
  falls **earlier the smaller the glyph**, and its half point (0.62 / 0.72 /
  0.76 for 15 / 28 / 44 px grids; 0.53 / 0.66 / 0.77 lone) **is the EN
  ceiling's upper edge for that px** (0.6 / 0.7 / 0.8). The grids' identity
  peak (0.4 / 0.5 / 0.6) is the ceiling's flat-letter peak, one step under
  its grid-cell peak.
- **There is no lower edge in f.** Below its peak the share holds, but the
  gradient's size collapses (`grid_44`: ‖g‖ 1.4 × 1e-3 at σ 0.2 vs 76 × 1e-3
  at 0.6; `grid_16` only 3× smaller at 0.2). A band's lower edge is an
  exposure edge, not an identity one.
- **Against grid_44's bands:** the 15 and 28 px tiers train at their peak,
  between plateau and half (grids f 0.31 → 0.19); the 44 px tier trains at 0.7–0.9,
  above its ‖I‖ peak, and its upper half (0.8–0.9) gives f ≤ 0.15. **b7593**
  (every item at 0.75–0.93) puts every 15–28 px tier at f ≤ 0.16 and mostly
  under 0.1 — a step-0 gradient over 90 % the same whichever glyph is drawn,
  H1's "layout alone", measured; `grid_small b7593` read 0
  (`../cjk_anima_reseed/reports/grid_small_lone_2026_10_02.md`).

## 1. Setup

| | pass 1 (`r1`) | pass 2 (`render`) |
|---|---|---|
| contrast | different rows: the true caption C vs C^{v_j} (slot k's glyph u → v_j), same image | one row (u), the true caption C, the image re-drawn with slot k's glyph u (A) or v_j (B_j) |
| items | 40 per tier × 9 tiers (360); clause: 16 per tier (144) | 40 per tier × 6 grid / lone tiers (240, none dropped) |
| captions | `data_recap_b7593` (plain) + `data` (position clause) | plain |
| m, σ | 3; 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 | same |
| job | `20261002-195428-37dfb2` (38.5 min) | `20261002-203655-1e4c1d` (18.5 min) |
| result | `experiments/grad_identity/results/20261002-1954-r1/` | `experiments/grad_identity/results/20261002-2036-render/` |

Both: `output/cjk_anima_scale/run1002_grid_44/` items (the recap dir's
latents in pass 1), rows from `Rows(warm=None, frozen=…, context=SEED_ROWS_0921)`
— the 82 rows are the only ext rows the captions carry, so all of them sit at
the pack rows; job logs `pack anima_cjk_vocab_pack raw sha 7b9fce0bb57b;
ANIMA_VOCAB_GLYPH_ROUTE=1`, set in-process. u is one of the 81 hiragana
occurring once in the item; v_j are hiragana absent from the item and from
u's dakuten / small-kana family. Every caption set is checked on its T5 ids
(wrong differs from true at exactly one position, u's row against v_j's).
Each results dir holds `plan.json`, `grads_<mode>.pt` (the 1 + m gradient
rows per item × σ), `per_item.jsonl`, `summary.json` and the full
`report.md` (per tier × σ: id2 · null · excess, direction-only, norm-only,
f / ‖I‖, dloss / ‖g‖ / cos, and per form × slot px).

**Batching (pass 1).** One batch of the 1 + m captions over the repeated
noisy latent gives every row's gradient in one backward. Checked before the
loop on 4 items × σ 0.4 / 0.8 (`verify.json`): a caption never touches
another's row (the others' rows are exactly zero); row u's gradient with the
three wrong captions replaced by three copies of C moves by rel 0.013 —
inside the run-to-run floor (same batch twice 0.018, reversed 0.017, B = 1
twice 0.018). B = 1 against B = 4 differs by rel **0.12** (median; up to 0.8
on small gradients) with matching losses to 1e-3 — bf16 kernel shape, not
batching; the loop runs one shape throughout. (`20261002-194820-45136a`
stopped at the first, stricter gate, B = 1 vs B = 4 at cos > 0.98;
`20261002-195126-3f915f` is the diagnostic that split the two; `…-194547-b86330`
died on a σ assert that did not allow the bf16 cast.)

**Per-sample gradient (pass 2).** All 1 + m samples carry row u, so a zero
`(B, D)` tensor is added at u's position after the rows; its gradient is each
sample's own. Σ_b over it equals ∂raw_u exactly on the 20 checked reads.

**Renders.** `data.grid.render_grid` with the record's units, grid, canvas,
bubble and fill and `grid_small`'s `bubble_fit` / `cell_jitter` (the grid_44
build's), one seeded rng per item: fresh twins of the records, not their
pixels. The font list is cut to the fonts covering every glyph involved
(14–15 of the set), so `pick_font` draws one font for A and every B_j. Kept
iff the pixel difference lies inside slot k's cell and every other cell's ink
box is unchanged: 240 / 240. Peeked: only the slot's glyph (and its fitted
bubble) changes.

## 2. Pass 1 — the true row against other rows in its slot (the brief's design)

| (plain, all 9 tiers) | σ 0.2 | 0.4 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| cos(g_true, g_wrong) | 0.15 | 0.17 | 0.16 | 0.16 | 0.15 | 0.15 | 0.15 |
| cos(g_wrong, g_wrong) | 0.16 | 0.16 | 0.16 | 0.16 | 0.14 | 0.14 | 0.15 |
| items with excess_dir > 0 (of 360) | 163 | 170 | 169 | 169 | 172 | 167 | 175 |

- Different rows' gradients in one slot are **near-orthogonal at every σ**
  (cos ≈ 0.15; f ≈ 0.85): a row's gradient is mostly a function of its own
  vector, whatever the image. The mean of other rows' gradients is therefore
  not "what any row standing in the slot receives" for u — the brief's layout
  term is a different row's Jacobian, and the identity / layout split does not
  exist in row space at the cold start.
- id2 vs null (the brief's quantity) is ≈ 1.1–1.4 vs 1.1–1.4 in every cell
  (clause, 16 items, up to 1.57); direction-only excess per tier, pooled over
  σ, −0.015 … +0.005 plain, −0.015 … +0.017 clause; **no (tier, σ) cell
  above zero** at p < 0.01 on any excess, plain or clause. Pooled, the true row is slightly
  *closer* to the others (plain excess_dir −0.007, 1 820 / 3 960 > 0,
  p 4e-7; clause n.s.).
- Caption leverage at the cold start is nil: dloss = L(wrong) − L(true)
  median 0.0000 in every cell, pooled 2 060 / 3 960 > 0 (p 0.012), clause
  p 0.41; the one cell at p < 0.01 of the 198 is clause `bubble1_32` σ 0.3
  (15 / 16). Plain and clause do not differ.
- The magnitude estimator's **median** excess is negative everywhere
  (−0.05 … −0.10, p to 1e-10) under exchangeability: id2's three terms share
  one denominator ‖g_true‖, null's three do not, so id2 is the more skewed
  (row gradient norms spread with log-sd 0.59). Its mean is +0.02. The
  direction-only and norm-only columns carry the read.

## 3. Pass 2 — one row, the glyph drawn swapped

f = 1 − cos(g(u | B_j), g(u | B_k)), median over items (IQR in the results
report). With g = L + I(glyph), f is I's share of the energy; the bf16 floor
is f ≈ 0.01 (B-shape numerics, rel 0.12), the run-to-run floor 2e-4.

| tier | px | σ 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grid_44` | 44 | 0.48 | 0.58 | 0.45 | 0.44 | 0.40 | 0.36 | 0.25 | 0.15 | 0.11 | 0.07 | 0.01 |
| `grid_29` | 28 | 0.42 | 0.34 | 0.36 | 0.32 | 0.33 | 0.19 | 0.12 | 0.10 | 0.08 | 0.03 | 0.01 |
| `grid_16` | 15 | 0.33 | 0.31 | 0.29 | 0.19 | 0.17 | 0.09 | 0.08 | 0.06 | 0.02 | 0.01 | 0.01 |
| `lone_44` | 45 | 0.55 | 0.71 | 0.76 | 0.85 | 0.71 | 0.66 | 0.42 | 0.30 | 0.17 | 0.07 | 0.01 |
| `lone_28` | 28 | 0.56 | 0.73 | 0.63 | 0.69 | 0.52 | 0.19 | 0.16 | 0.11 | 0.04 | 0.04 | 0.01 |
| `lone_16` | 15 | 0.64 | 0.60 | 0.44 | 0.35 | 0.23 | 0.16 | 0.10 | 0.11 | 0.04 | 0.02 | 0.01 |

‖I‖ = ‖g‖·√f and ‖g_true‖ (× 1e-3, medians):

| tier | | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grid_44` | ‖I‖ | 1.2 | 2.3 | 4.9 | 13.6 | **46.9** | 35.4 | 34.6 | 26.9 | 20.7 | 14.5 | 9.9 |
| | ‖g‖ | 1.4 | 3.1 | 6.2 | 20 | 76 | 65 | 64 | 63 | 66 | 69 | 90 |
| `grid_29` | ‖I‖ | 3.1 | 6.4 | 21.2 | **48.5** | 40.5 | 26.1 | 22.1 | 24.0 | 19.2 | 11.3 | 10.1 |
| | ‖g‖ | 4.2 | 12 | 33 | 82 | 53 | 53 | 69 | 63 | 68 | 75 | 88 |
| `grid_16` | ‖I‖ | 20.0 | 28.7 | **34.6** | 23.7 | 16.3 | 23.4 | 27.3 | 21.9 | 13.3 | 10.2 | 10.2 |
| | ‖g‖ | 25 | 58 | 68 | 54 | 46 | 81 | 73 | 93 | 104 | 92 | 115 |
| `lone_44` | ‖I‖ | 0.9 | 1.8 | 4.1 | 8.3 | 10.9 | 12.2 | 12.5 | **12.8** | 6.5 | 7.4 | 10.1 |
| `lone_28` | ‖I‖ | 1.1 | 2.9 | 4.8 | 5.5 | 5.1 | 7.2 | **8.1** | 5.9 | 4.5 | 6.5 | 7.5 |
| `lone_16` | ‖I‖ | 2.4 | 2.4 | 2.0 | 2.3 | 2.9 | **5.1** | 4.4 | 4.3 | 3.3 | 5.0 | 7.4 |

- **Monotone in σ, ordered by px.** At σ 0.7 the grids give f 0.36 / 0.19 /
  0.09 for 44 / 28 / 15 px; at 0.8, 0.15 / 0.10 / 0.06; at 0.95 every tier
  is at the bf16 floor.
- **The lone tiers** have a larger share (0.55–0.85 plateau) but a small
  gradient: their loss is the plain canvas mean (`src` font), where a 15 px
  glyph is ≈ 0.1 % of a 512² canvas; ‖g‖ stays under 36 × 1e-3 up to σ 0.9.
  ‖I‖ is 3–10× under the grids' and has no clear peak at 15 px; the 0.95
  values (‖g‖ ≈ 0.1) are the global composition, at f ≈ 0.01.
- **The drawn glyph being u buys nothing; at 28–45 px it costs a little.**
  The true render's gradient sits *closer* to the common part than a wrong
  render's: direction-only excess per tier, pooled over σ, `lone_44` −0.060
  (150 / 440 > 0, p 2e-11), `grid_44` −0.033 (p 9e-6), `lone_28` −0.033
  (p 0.002), `grid_29` −0.027 (p 3e-4), `grid_16` −0.010 (p 0.08), `lone_16`
  −0.007 (p 0.06). The base (its Qwen side, the pack rows) already expects u
  a little where the glyph is large enough, so the matching image leaves a
  smaller glyph-specific residual; at 15 px it cannot see it.
- dloss (L(B_j) − L(A) under C) is not separable from zero in any cell; the
  grids lean positive at σ ≤ 0.4 (69–72 of 120, p 0.035–0.12).

## 4. What it does not show

- **Training.** A step-0 gradient is necessary, not sufficient: Adam
  rescales per coordinate, the rows move off the pack rows within a few
  hundred steps (`rel` ≈ 1 by the end of grid_44's run), and the identity
  share of a moving row is not read here. f says what a draw *offers* the
  row, not what the run keeps.
- **Different rows start from different vectors.** That is what sinks pass
  1 (§ 2) and why pass 2 holds the row fixed; pass 2's f is per row, medians
  over 40 different u per tier.
- **Bubble tiers in pass 2.** Scenes were not re-drawn; `bubbleN_34`,
  `bubble1_32`, `bubbleN_18` have pass-1 reads only.
- **Clause captions in pass 2**, and any warm or trained rows.
- **The pixels of record.** Pass 2's renders are fresh twins of the records'
  specs (font list cut to the covering fonts); their slot px match the tiers'
  (15 / 28 / 44–45).
- **Numerics.** Gradients are bf16-autocast at B = 4, as in training; a
  kernel-shape change moves a row's gradient by rel 0.12, so f ≤ 0.01 is
  floor.

## Open

- Pass 2 on trained rows (`experiments/grid_44_cold_hira_recap_b7593`,
  `retrain_kana`): where training bought identity, does the true render's
  gradient separate from the wrong renders' (the sign of § 3's matching
  excess flipping), and at which σ?
- Pass 2 for the bubble tiers: re-draw a scene item with one glyph swapped
  (`render_into_scene`, the record's scene / box), same diff-box check.
- A band chosen by ‖I‖ rather than by the ceiling: 0.3–0.5 (15 px), 0.4–0.6
  (28 px), 0.5–0.7 (44 px grid) are the windows around the per-draw identity
  peak; grid_44 trains the 44 px tier one step above it.
- The lone tiers' plain canvas loss gives them 3–10× less identity per draw
  than a grid cell of the same px; whether a lone 1×1 should take the box
  share (`GRID_BOX` covers `src` grid only) is a trainer question this read
  only prices.
