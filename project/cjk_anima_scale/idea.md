# idea — layout from the base, identity at low σ (2026-09-29)

**Read 2026-09-30** (`reports/sigma_split_2026_09_30.md`): the band half is
closed — below σ 0.5 neither the rows nor the caption move any text, and
layout and string are both decided between 0.9 and 0.7. The data half goes on
as [`plan_garble_replace.md`](plan_garble_replace.md). § 3's σ-split sampler
exists (`generate_body`'s `context_alt` + `tag_drop_sigma`,
`experiments/sigma_split`).

Not planned, no run, no budget. **The plan stands**: `retrain_kanji_b1..b4`
(`plan_retrain.md`), then the warm-started training after b4 as planned. If
this idea is tried, it is tried on the rows after b4, not before and not
instead.

The worry behind it (user): 0.7–0.9 is the
band where the DiT decides the page's composition. The single-glyph data
(`b0709`: a bubble-fit glyph, a 1×1 flat at 150–400 px, grid cells at
51–136 px) may teach a row "one big glyph on a plain canvas" together with
the glyph. A singles ruler that renders a lone glyph in a bubble
(`native_single`) cannot tell the two apart.

## 1. The read that motivated it (CPU, existing renders)

The renders of `p1_cold` (in-word only), `p1_lone` (`b0709` only), `p1_mix`
and the floor (`rows_step1_0921_merged`), paired by (string, prompt, seed)
with the EN reference (`native_enref/512_28_4/`). The measures:

- `box`: the union of the reader's non-whole text boxes / canvas;
- `box h`: the tallest box / 512;
- `flat white`: the share of 16×16 patches with std < 6 and mean > 225.

The script was a one-off in the session scratchpad and is not kept.

| `native_spell` words (8 × 16) | box | box h | flat white |
|---|---|---|---|
| floor (seed) | 0.120 | 0.289 | **0.496** |
| `p1_cold` | 0.117 | 0.151 | 0.320 |
| `p1_lone` | **0.174** | **0.287** | 0.245 |
| `p1_mix` | 0.132 | 0.185 | 0.299 |

- **Lone data teaches size, not the white canvas.** `p1_lone` vs `p1_cold`,
  box larger in 91 / 37 pairs (p 2e-6). The routed words (p 4e-8) and the
  singles (p 3.5e-6) go the same way. Its flat-white share is the lowest of
  the four.
- **`p1_mix` sits between the two.** The in-word items shrink the size
  prior but do not remove it (box vs `p1_cold`: 85 / 43, p 3e-4). The
  production recipe is `p1_mix`'s, so `retrain_kana` and the kanji batches
  likely carry a weak one. Whether it compounds along the `context` chain
  was not read.
- **The floor is the full paste.** On the contact sheet it draws one giant
  glyph on a white canvas or a disc and wipes the scene. It does this even
  for a word (ぱ / い / え for a 4–5-glyph word). This is the seed trained on
  lone data for ≈ 152 k steps.
- The in-word items differ from the lone ones in band, px and composition
  at once. This read cannot say which of the three shrinks the prior.

## 2. The idea

- **Data.** Let the base draw the text: a scene prompt that makes it draw
  dialogue ("she is saying something in japanese" → pseudo-Japanese, or an
  EN anchor). Find that text's box and replace it with real Japanese **at
  the base's own glyph size and line layout**. Placement and size are then
  the base's, not a fill rule's.
- **Band.** Train at 0.3–0.5 (the user also floated 0.1–0.3), so the row is
  never asked to decide the layout.

### What exists already

The scene pools (`s1`, `s1w`, `sl1w`, `ja_comic`) already work this way:
the base draws a bubble with text in it, the judge finds the anchor bubble,
the anchor is erased with a ring-median fill, and JA is rendered in. Two
things are new:

- the px comes from the anchor's glyph height (the base's lines are
  ≈ 14–22 px at 512, `_archive/reports/next_2026_09_25.md` § 4), not from
  the bubble fit (48–53 px);
- the band.

`scene_single_small` (one glyph at line px) is the nearest recipe. The
uncommitted `experiments/polish_b1` (`p0305` `scene_line`, 14–22 px)
overlaps in part.

### Why this one (user, 2026-09-29)

Of the options on the table, this is the one that does all three at once:

1. **It turns the base's garble into Japanese text.** The base already
   decides "there is a line of dialogue here, this size, this layout"; the
   row only has to change what the line says.
2. **It keeps the layout out of the row.** Every layout-bearing decision in
   an item is the base's own, so nothing in the data asks the row to carry
   one. The lone data's size prior (§ 1) is the contamination this avoids.
3. **It keeps the self-generated FM benefit** (`future.md` § 1). Outside the
   text the item is the DiT's own render, so the loss there is low and the
   residual left for the row is the text.

Whether it is also the alternative that conflicts least with the record's
failures (§ 2b) is not established.

EN anchor or garble: an EN anchor can be read back, so its box and glyph
height can be checked. Garble has no read-back, and outside a bubble the
erase leaves a blot (`plan_polish.md`). The first version would reuse the
pools' anchor boxes.

### The band argument (as first written; contested in § 2b)

- **0.3–0.5 is the live window for text at the base's size.** On the
  ceiling (`band_experiment_results.md` § 2), 16 px is live at 0.2–0.6
  (peak 0.4) and 20 px at 0.25–0.6.
- **0.1–0.3 is below every floor in the table.** In training, x_t is the
  JA target noised. At inference with the layout from the base, x_t is the
  base's text noised. Inside the live window the two are not told apart;
  that is what live leverage means, so training there carries over. Below
  it, x_t already holds the glyph shape. A row trained at 0.1–0.3 learns to
  sharpen JA that is already there, not to overwrite the base's text.
- So the band is 0.3–0.5, or 0.2–0.5 at the widest.
- px windows are render px: 18 px at 512 is 36 px at 1024-tier, one band
  step up.

### 2b. Review (Fable 5.1, 2026-09-29) and the user's reading

**Against the band argument above.**

- "Live" on the ceiling means mean move ≥ 0.1. At 16 px the largest move is
  0.41 for a letter and 0.28 for a string (`cf_band_a1` A.2), so the DiT
  still reads 60–70 % from x_t. "Not told apart" overstates it, and the
  0.1–0.3 objection ("it only learns to sharpen") applies, weaker, to
  0.3–0.5 too.
- The ceiling is EN, on the base, on one layout. JA rows sit one step higher
  (48 px: 0.7 vs EN 0.6; `bs_lo` trained at 0.5–0.7 still has leverage only
  0.08 at σ 0.5, `band_b1`). JA leverage at 16–20 px is unmeasured. No table
  has leverage for *overwriting* a different count or layout.
- The count is decided above 0.5 (band law H-count; low-band singles miss as
  runs), and there the row is the untrained pack row. This design hands
  count to the base, and doubling is already the largest cost
  (`retrain_kana` dup 42 / 64).
- Δ is σ-blind. What a Δ learned at 0.3–0.5 does at σ 0.9 is unread.

**Precedents, as the review lists them.** Small-px / low-band single data
lost identity and learned a line mode three times:

- F2a′ (`_archive/proposal.md` § 0): lone glyphs in the `s1s` small-bubble
  pool at the words' bands and px; donors official 91 → 59, alone as a line
  47 → 97 / 144.
- `micro_chain_result.md` § 3: 24–32 px singles at 0.5–0.7 undid 0709's
  identity.
- `run0925_300f`: 0.3–0.7 rows learned "a small line of text in the bubble".

The review also cites `transplant_line` (a shared line direction grows from
low-band line data and is absent on single canvases): layout reaches the
row at low σ too, a counterexample to "layout from the base".

**The user's reading.** Those three failures are glyph drawing learned
ambiguously. In each of them the row was being taught both what the glyph
is and how the text sits, at sizes and positions the base did not choose
itself. This idea takes the second job away, so they are not the same test.
Unread either way.

**Confounds in § 1 (review).**

- `p1_lone` composes 9 / 128, so its word `box` may be the union of boxes
  over pseudo-text. `box h` is better, but merged detections in one bubble
  give the same number.
- The white canvas belongs to the seed. `p1_mix`'s residual size prior is
  ≈ 13 % over `p1_cold`. Whether it reaches the named failure shapes (paste
  / wipe on `native_sent`) was not read.
- The most direct suspect is the 1×1 flat, which `loss.py` leaves under
  plain canvas MSE, rather than the band.

**Cheaper checks, in the review's order.**

1. **A training-free σ-gate** (GPU, ≈ 25 min). On `retrain_kana`, with the
   C2 eight words and the singles, run two arms: Δ off above 0.5, and the
   mirror (Δ off at σ ≤ 0.5). If the ≤ 0.5-only arm cannot hold half of the
   rows' reads, the idea is dead without building data.
2. `cf_sense --cf_lang ja --cf_glyph_px 16,20` on `retrain_kana` (GPU): are
   the JA rows live at 0.3–0.5 at that px?
3. **CPU.** The § 1 read on `band_s_0923`'s `bs_lo` / `bs_hi` native renders
   (same data, two bands) and on Stage B's count tier. This separates band
   from px and composition for free.
4. A data micro-arm (GPU, ≈ 25 min): `p1_mix` without the 1×1 flat (one
   TABLE line), 36 donors.

### 2c. A read for the premise (2026-09-30)

The 14 `target` captions (hoshino ai at the bar, はい / こんにちは) at 768×1344, on
Anima with **no vocab pack**. T5 sees one `<unk>` per quoted span, and Qwen3 reads
the Japanese.

- The base draws **Japanese-looking lines inside its own speech bubbles**, mostly
  vertical (まだよ, えんよ, こげきたばいよ). It draws no subtitle or only a small
  one, and 0 / 14 are correct.
- With pack rows the text leaves the bubble for a subtitle band under the scene.
  The seed's rows make that band large and correct (8 / 14).
- So without a row the base already decides "a Japanese line, this size, in this
  bubble". The rows add the string, and with it the lone data's size and paste
  prior.
- This supports § 2's premise, that the layout can be the base's and the row only
  changes what the line says. It is a small read (7 prompts × 2 seeds, placement
  judged by eye): `reports/polish_seed_2026_09_30.md` § The 4 k letterbox is the
  base's.

## 3. The test (GPU, later)

`src/eval/cf_sense.py` is a one-step x̂0 leverage read on a noised render,
not a sampler that switches captions at a σ. `src` has no caption switch
(corrected 2026-09-29; an earlier draft said it existed). Both conditions
need a σ-split sampler, or the Δ gate of § 2b check 1:

- **(a) forced scaffold.** High σ on an EN (or garble) caption, the JA row's
  caption from σ ≤ 0.5. Does the row have the leverage to overwrite?
- **(b) the product condition.** One caption, the row throughout. At high σ
  the row is the untrained pack row, so the premise is that a raw pack row
  lets the base lay out an ordinary text line. P0's scaled-down rows
  drifting toward generic text points that way, but it was not read.

Read (a) and (b) on the seed and on the pack rows before any training, so
the trained rows have a floor. Judge on `native_sent` and words. The singles
ruler (48 px lone) will fall by construction and mixes in composition (§ 1).

## 4. Risks

- Dense kanji at 16–20 px may not be legible at all (`band_experiment_results.md`
  § 6 item 3, px vs exposure, open).
- The old piece run learned "a small line of text in the bubble"
  (`next_2026_09_25` § 4). At low σ the line belongs to the base, which may
  remove that failure or may not; unread.

## 5. First step (CPU)

From the pools' stored anchor boxes: the distribution of the base's glyph
height per pool. How many items could take JA at line px, and does that px
fall in the 0.3–0.5 window at the training shape? If yes, this is a recipe
beside `scene_single_small`, not a new pool.
