# length proposal — the line as long as the word (2026-10-01, draft)

The live open question of the line. It takes over `plan_retrain.md` § 4
"Doubling" (archived 2026-10-01): `retrain_kana`'s `dup` rose over `p1_mix`'s
(42 vs 31 / 64), and the seed still doubles on the `sent` grid (dup
100 / 184, `かかんがが`, `こんにちちは`).

## Where the line got to

**The concept: attack the base where it is confused, in its own layout.**
Asked for Japanese, the base already draws a text region and fills it with
pseudo-Japanese (`reports/polish_seed_2026_09_30.md`, `sigma_split`: bubbles with vertical lines,
a subtitle bar, a banner). The rows do not have to invent a layout; they
have to make the base's own text region say the word. `garble_replace`
built that literally — the base's garble erased and replaced by a real line
at its px — and it moved the layout the way no fill-rule composite did
(box IoU vs EN ref 0.20 → 0.33), while the strings did not read.

What the garble arms and their follow-ups settled
(`reports/garble_replace_2026_09_30.md`, `reports/delta_scale_2026_10_01.md`,
`reports/sigma_split_2026_09_30.md`):

- **The repeats are leftover slots.** The text region's span is set by σ 0.95
  with or without trained rows; the glyph count inside it at σ ≈ 0.9. The word
  is written into those slots and the slots left over repeat its glyphs
  (traj: warm fills the seed's span with smaller glyphs, `こんにちはは`). A
  weaker Δ lengthens the line in the same banner (Δ 0.9: dup 100 → 124).
- **Order information is not the limit.** The slot share of the adapter output
  falls with the reads across arms, but restoring it (Δ 0.75: slot variance
  2×) does not bring the reads back; identity falls first.
- **The count is decided where only large glyphs resolve.** Garble glyphs are
  ≈ 14 px: below σ 0.85 the rows learn small glyphs (more slots); at 0.8–0.95
  they learn the canvas's bubbles and no string (`windows.py` C.2: 0.8–0.95
  is dead even at 48 px).

Closed by these reads: a Δ scale or row cap; garble items at the base's px
at any band; an order-paired loss on routed singles (concatenated single
rows carry no order leverage at any σ —
`../finished/cjk_renderable_anima/findings.md` § Settled); ΔFM (plain FM's
target with less variance — same findings, closed 2026-09-18).

## Proposal: train the count where the base miscounts

A counterfactual pair on the base's confusion state, not a clean target:

- **A** = the word once, banner-size, filling the region; **B** = the same
  region and span with one slot more — a glyph of the word repeated
  (`こんにちはは`), the leftover-slot error itself.
- Input = B noised, caption = A's, target toward A
  (`../finished/cjk_renderable_anima/idea.md`'s CF; the residual
  (x0_A − x0_B)/σ does not vanish with ε). Plain FM only ever shows the
  rows trajectories that start from a clean A; at inference they meet an x_t
  that already has the extra slot.
- **σ 0.8–0.9**, where the count is decided and a single glyph's caption
  leverage peaks (`cf_sense`: EN 0.197, trained rows 0.155 at σ 0.8).
- Large glyphs only — the base's banner, not its garble.

**Step 0 (no training, minutes):** `src/eval/cf_sense.py` on the seed rows
with A / B pairs as above, a handful of `sent` words, σ 0.8 / 0.85 / 0.9 —
does caption A move x̂0 from B toward A? ≥ 0.1 opens step 1; ≈ 0 closes the
paired route (the count would then follow the span, set at 0.95 without the
rows).

**Step 1:** one micro arm (≈ 25 min) on the 57 `sent` singles, warm, CF
items at 0.8–0.9 beside the seed's own item mix; read on `sent` (dup,
glyphs per read) and the traj leg (`sigma_split --traj --rows`), glyph count
at σ 0.9 against the seed's.
