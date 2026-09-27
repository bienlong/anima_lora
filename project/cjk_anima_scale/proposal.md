# proposal — the line mode: what is left (2026-09-26)

This file merges `proposal.md` (the line mode as a transferable direction,
and the doubling track) with `proposal_factorizedrows.md` (rows = identity,
modes by context). It keeps only the open items. What ran is in the reports.
Section numbers that reports and experiment docstrings cite refer to the
pre-merge files at git `48e1d6ab`.

## 0. Where it stands

| read | verdict | report |
|---|---|---|
| Stage A: a piece direction `u_P` on held-out pieces | transfers (contained 27 → 52 / 256) | `reports/transplant_piece_2026_09_26.md` |
| Stage B: a 36-kana donor's `u_S` on 10 held-out kana | composition transfers (≤ 1 edit 11 → 66 / 160), and doubling travels with it | `reports/stage_b_2026_09_26.md` |
| α sweep of `u_S` | composition peaks at α 1; no dose bounds either doubling | `reports/stage_b_2026_09_26.md` § 7 |
| F0: does the adapter modulate a row by its neighbours? | no (out cos 0.95–0.97, gain 0.97), so the gate lives at the embed hook | `reports/f0_interaction_2026_09_26.md` |
| F1: a factorized donor, rows + a gated `v_line` | `v_line` is a working mode. The rows still render a line when alone | `reports/f1_line_2026_09_26.md` § 1–4 |
| `v_line` dose 0.5 on the held-out 10 | ≤ 1 edit 80 / 160 (u1 66), ≤ 2 edits 131, singles at the floor, in-word `dup` 54 (floor 26) | `reports/f1_line_2026_09_26.md` § 5 |
| § 2.1 (a): what B is | not a fill prior. It is word-final, and it is bound to the `Japanese text` clause: under `English text` ≤ 1 edit holds (40 = 40 / 80) and in-word doubling drops to the floor (7 vs en 27, floor 4). A third of `dup` is reader noise | `reports/doubling_box_2026_09_26.md` |
| § 2.2 count twin: the Stage B donor, count tier off | no count direction: the donor difference has no shared part (split-half cos 0.15, energy 4 %), and the twin grows `u_S` again (cos 0.96). Alone-as-a-line 108 vs 102, repeat 64 vs 50 (p 0.09) | `reports/count_twin_2026_09_26.md` |
| § 2.3 Stage I on 12 cold kanji | alone: I0 wins at 90 and at 270 steps / row (x3: contained 117 vs I1 84 / 192, p 9e-6), I2 out. **Under the mode, the kanji words never compose** (≥ 2 glyphs in order ≤ 3 / 96 on every arm, x3 included), so criterion 2 cannot rank recipes. `b0709` stays. Cold kanji need more than 90 steps / row | `reports/stage_i_2026_09_26.md` § 5 |
| § 2.4: the gate on runs with pieces | a piece in a run renders 0 before and after. はい contained 4 → 14 / 16, target はい 0 → 6 / 8. The single beside a piece doubles (やったネ `wdup` 2 → 9) | `reports/line_pieces_2026_09_26.md` |

The best configuration on record for a vocab without line training is
**the seed rows + 0.5 · `v_line` + the run gate**. Two kinds of doubling
remain:
- **A, a glyph alone repeating itself (あ → ああ):** bounded. The gate
  keeps a lone row at its seed value.
- **B, a doubled glyph inside a word (ひまわり → ひまわりり):** open. Counted
  as a glyph of the word doubled (`wdup`), it is 34 / 160 vs the floor's 8
  (the `dup` of record, 54 vs 26, includes reader noise). It is mostly the
  word-final glyph, and it lives under the `en` clause (§ 2.1). Collapsing
  doubled glyphs lifts the words from 80 to 89, so B costs about a tenth
  of the words.

Bound by the wake roll-up:
- Transplant + a pinned trigger is not a step saver, so nothing here
  claims fewer steps per row.
- No row-space geometry penalties (decorrelation, orthogonality,
  whitening).

## 1. The model, and what is built

```
row_eff(i, ctx) = r_i + g_line(ctx) · v_line
```

- `g_line = 1` iff the pack row has a pack neighbour in the T5 ids. A
  spelled word's glyphs are adjacent ext ids. A lone glyph or a lone piece
  has no pack neighbour.
- **Built:**
  - `src/common/hooks.py::ExtDelta.line` holds the vector and applies the
    gate at the hook. It is saved as `delta['line']`, and a state without
    it reads exactly as before.
  - `cjk_scale/rows.py` / `train.py` take `line_mode`, which trains it
    beside the rows. It is for experiments only; `scale.py` never passes
    it.
  - `v_line` itself: `output/cjk_anima_scale/run0926_f1_line/trained.pt`
    (norm 204, cos 0.79 to `u_S`), used at 0.5.
- **Not built:** the inference side (§ 2.5).

## 2. Open items

### 2.1 In-word doubling (B) — the main open cost

What is known:
- B rides the line mode. It jumps above the floor at the first dose
  (`u_S` α 0.5: `dup` 49) and does not fall below +18 at any dose tried
  (`u_S` α 1: 44; `v_line` 0.5: 54; `v_line` 1: 77). Composition and B
  have not separated on any dose axis.
- A reading, not a verdict (Stage B § 7): the line is switched on before
  the sequence is resolved. Doubling fills the line with the glyphs it
  has, and it falls only as composition gets stronger.
- The recipes fill the box. `scene_spelled` / `scene_piece` draw `fill`
  0.7–1.0 of the bubble, so a word has never been trained in a bubble
  wider than itself.

Hypotheses for B:
- **H1, a fill prior:** the line fills its box, and a word shorter than
  the box's capacity pads by doubling. Predicts that `dup` rises with the
  box's capacity in glyphs minus the word's length, and that doubled
  glyphs sit at the trained line px.
- **H2, a sequence failure:** the mode is on but the order is weak.
  Predicts that `dup` is independent of box size, concentrates on
  particular positions or bigrams, and falls with composition strength.

**(a) ran (`reports/doubling_box_2026_09_26.md`): neither hypothesis as
written.** Doubled and clean renders sit at the same px (55–85, against
trained word px 35 / 18). The extra glyph lengthens the line at a fixed
px, so no fixed box is being filled (H1 fails). The doubling concentrates
by position (H2), but on the word-final glyph (24 / 34 at v0.5). The
separating axis is the caption clause. `swap` (`English text reads as`)
keeps ≤ 1 edit at 40 / 80 with in-word doubling at 7 (floor 4). `en`
(`Japanese text reads as`, the clause every recipe trains on) has 27.
Read B with `wdup`, not `dup`.

Analyses:
- **(a) px, box and position of the doubled renders (CPU, reads on
  disk).** Done, above.
  - Renders: every word render in floor / u1 / `v_line` 1 / `v_line` 0.5,
    and spell_b's and the Stage B donor's own words.
  - Per render: the reader box (`reads[].box`), the glyphs read,
    px = √(box area / glyphs), the bubble where the sheet shows it, and
    which position doubled (first / inner / last; the preceding glyph).
  - H1 predicts box-driven doubling, H2 position- or bigram-driven.
  - This is the old § 3.3 b, extended from singles to words.
- **(b) Cross-attention placement (costs code).** Where the doubled glyph's
  token lands in the image, per block: one blob or two. Only if (a) leaves
  H1 vs H2 open.

Levers, chosen by (a). (a) points past both levers below, to the
caption:
- **Clause split (training-free, new floor keys):** the held-out words
  under the `japanese text` tag alone, and under the `Japanese text reads
  as` sentence alone, at 0.5 · `v_line` against the floor. It shows which
  part carries B.
- **Clause-varied `scene_spelled` captions** (data, if the split
  localizes it): retrain `v_line` on Stage B's words with the clause drawn
  over variants, then read `en` / `swap` at dose 0.5.

The levers as first written:
- **H1 → data:** a `scene_spelled` tier at low fill (0.3–0.6, a word in a
  bubble wider than itself). Retrain `v_line` on Stage B's words with it,
  then read the held-out 10 at dose 0.5. It is the same data axis as
  F1b's alone items (§ 2.2), but for words.
- **H2 → the mode side:** a finer `v_line` dose (0.35 / 0.75 around 0.5,
  training-free, ≈ 25 min each) to find the `dup` minimum. Then a caption
  marker (§ 2.7), which is the last resort.
- **Not** a training-time cap on a row's projection. It sits in the
  closed geometry-penalty family and would act on `r_i`, while B lives in
  the mode.

### 2.2 Trained rows alone still render a line

F1's rows, gate off, read as a line of other glyphs in 94 / 144 renders
(floor 47), and official is 50 (floor 91). `u_S` left them (energy 24 % →
5 %), but the layout did not. Until this moves, **the "alone" value of a
trained single is its seed row**, and trained rows are line-only.

- **F1b:** alone items at the b0305 px (≈ 18 px). The count tier covers
  24–40 px only, because a single under 24 px has no window in the band
  law. **It needs a band-law row first:** a read of the single's window at
  16–24 px (`band_experiment_results.md`). Then retrain F1 with the tier
  and read the donor singles alone (line ≥ 3 glyphs, official, repeat)
  against F1's 94 · 50 · 35.
- **The count twin: ran, negative** (`reports/count_twin_2026_09_26.md`).
  The count tier at 0.3 / 24–40 px wrote no direction, and the line mode
  is the same without it. Alone-as-a-line is written by the word data.
  This lowers F1b's prior: the tier moved same-glyph repeats, not the
  line. The original item:
- **The count twin** (old § 3.3 d): the Stage B donor with the count tier
  off, same words and seed. The Δ difference between the twins, own-row
  part removed, is the count direction: roughly ⟂ `u_S` means a separate
  transplantable component, roughly −`u_S` means the tier only shrinks the
  line mode. It is also the first run of the mode-discovery recipe (§ 2.6).
  Data CPU + ≈ 25 min train.
- Score the singles with the line metric (either reader reads ≥ 3 kana /
  kanji) beside `repeat`. `repeat` counts あ → ああ only and missed F1's
  misses.

### 2.3 Stage I — which scene makes a good identity

With the line carried by `v_line`, the question for `r_i` is **which
training context gives the identity with the least layout baked in.** Grid
50 % : scene 50 % (the seed's `b0709`) is the incumbent. It buys identity
and native together, and it binds "one big glyph filling the bubble" into
the row: the seed's spelled string renders as one big first glyph.

A good `r_i`:
1. **Alone:** renders its glyph once in a native scene (official, repeat
   and line at floor).
2. **Under the mode:** composes with 0.5 · `v_line` in a run (≤ 1 edit on
   spelled words, `dup` at the floor).
3. **Modular:** renders on a foreign shared direction (the 09-16 test).

Candidates. Each is `r_i` data only (gate off, one glyph per item), and
the recipes exist unless marked:
- **I0**, the incumbent: `b0709` as is, `scene_single` 0.5 (fill 0.7,
  ≈ 50 px) + `grid_single` 0.5 (1×1–3×3).
- **I1**, scene only with a px spread: `scene_single` at bubble fill
  0.2–1.0, so one glyph at 20–60 px and never always filling.
- **I2**, grid only at small cells (3×3 / 2×3, 25–85 px). The negative
  control for "native needs scene" (a grid alone was never a seed table).
- **I3**: I0 + I1's small-fill tier at 1 : 1 : 1.
- **I4**, pool diversity: I0 on the non-manga / `sl1w` pools as well. It
  needs a pool check, not code.

Setup:
- **Cold rows on a micro set** (≈ 12 vocabs the seed lacks: katakana or
  common kanji, pack raw rows). A warm start would measure the candidate
  on top of `b0709`'s identity, which is the confound.
- Micro arms (≈ 25 min each), same draws per row, singles band 0.7–0.9.
- Read per glyph with sheets.
- For criterion 2, read 0.5 · `v_line` on spelled strings of the micro
  set. Those strings need new floor keys, which is the one floor render
  here.
- Pick the recipe that wins 2 without losing 1.

**Ran (`reports/stage_i_2026_09_26.md`):** on the kanji micro set
(山 田 野 郎 太 道 場 星 空 天 地 小, cold):
- I2 is out. I0 beats I1 at 90 and at 270 steps / row, so `b0709` stays.
- Criterion 2 is 0 at both budgets. `v_line` (kana-trained) does not
  compose kanji rows: a word renders its first glyph, then unrelated text.
- Open: kanji vs kana, or row maturity. The seed's kana rows are
  `b0709`-trained too, and they compose.

Order: I0 / I1 / I2 first; I3 / I4 only if I1 moves 1 or 2. It closes if
every candidate ties on 2: then the identity source does not matter beyond
scene vs flat (09-16), `b0709` stays, and the gain is all in the mode.

### 2.4 The mode for pieces

`v_line` was trained on singles. The gate fires on any run, so a caption
with adjacent piece rows (って ください) gets it too. Nothing has read that.
- **Read (`reports/line_pieces_2026_09_26.md`):**
  - The piece ruler is lone pieces (one id each), so the gate never fires
    there.
  - On the sent / target rulers, runs of singles gain (はい contained
    4 → 14 / 16, the user's target はい 0 → 6 / 8, おしい ≤ 1 edit
    8 → 14).
  - A piece inside a longer run renders 0 before and after (った,
    ちょっと). The single beside it doubles (やったネ `wdup` 2 → 9: やや…).
  - Gate-off keys are identical to the floor.
- **First look, in ComfyUI (2026-09-26, the user's renders, not scored).**
  The pack `anima_cjk_vocab_pack_300fsp_line05` (300f_sp rows +
  0.5 · `v_line` as a line block) on a scene prompt ending
  `She is saying "へんたい"`: the prompt tokenizes as へ ん (seed singles)
  + たい (a 300f piece), all three on the line block. It rendered
  **へんてだ♥**: the singles are right, and the piece comes out as two
  wrong glyphs plus a trailing ♥. Two further tries (prompts not recorded
  here) rendered へい and んぃいい: dropped glyphs and in-word doubling
  (§ 2.1).
- **Next: the kind gate** (`v_line` on single-kind rows only). Pieces
  lost no identity but gained doubling beside them. The kind gate is a hook
  change plus the same ≈ 3 min read, and it also shows whether B rides on
  the piece row or on its neighbours. It holds until a piece mode exists. `u_P` (Stage A) is the post-hoc candidate for one, and a
  `v_line` trained on pieces is the learned one.

### 2.5 Shipping the mode

Once § 2.1 and § 2.4 settle the dose and the kind rule:
- **The inference hook:** the same run gate in
  `library/anima/vocab_pack.py`'s hook, with the pack carrying `line` (a
  pack without it reads as today).
- **ComfyUI:** the node's `_vendor/` follows through `make vendor-sync`.
- **Baking:** `scripts/toolkits/bake_vocab_pack.py` carries
  `delta['line']` into the pack.
- **The post-train comparison** Stage B deferred: seed + mode + a short
  post-train against the post-train alone, on a vocab set with line
  data. It decides whether the mode is worth adding to a production run's
  rows, or only to vocabs without line data.
- **The sent test:** `300f_sp`'s rows + 0.5 · `v_line` on the sent ruler.
  Does sent contained recover 4 → 11? It tells whether the sentence mix
  can shrink per vocab once the mode carries line composition.

### 2.6 Modes beyond line (discovery)

The recipe is twin donors: same vocabs and seed, one context axis
changed. Then the Δ difference with the own-row part removed, split-half
stability, a transplant onto held-out rows and the ruler. A mode that
passes becomes a gated `v_m`.

| mode | gate | first read | note |
|---|---|---|---|
| count / alone | no pack neighbour | the count twin (§ 2.2) | the first run of the recipe |
| horizontal | the existing caption marker | a horizontal ruler on the seed: is there a failure? | no horizontal read on record; skip the mode if there is no failure |
| manga vs other surfaces | a new caption marker | twin: manga-bubble pools vs sign / cloth / UI pools | the data is all manga bubble today |
| SFX | a new caption marker | none possible yet | no SFX data recipe (the renderers are font-based); the reader and corpus exist (Manga109-s + COO) |

Order by the rulers' failure list (doubling, sentence assembly, 3+ glyph
pieces), not by the taxonomy. A mode is worth training only if it explains
one of those failures.

### 2.7 Parked

- **A caption marker for the line mode** (spelled items captioned
  `, spelled.` or similar). It touches the caption convention and the TE
  path, so it is the last resort for B.
- **The outside opinion's adapter replay / distillation**
  (`opinion_factorizedrows.md` § 2–4). F0 found no context-dependent
  component in the adapter worth localizing, and adapter distances have
  not predicted renders (piece ↔ glyph R² ≈ 0.03).
- **The `u_P` → こんにちは-singles bridge** (no training, cached floor):
  low priority.

## 3. Order

1. ~~§ 2.1 (a)~~ done: B is clause-bound and word-final.
2. ~~§ 2.4 read~~ done: pieces in a run gain nothing, their neighbours
   double.
3. ~~§ 2.2 the count twin~~ done, negative: no count direction.
   ~~§ 2.3 Stage I~~ done: `b0709` stays (criterion 1). The mode does not
   compose kanji rows, which is a new open item.
   Parked (user, 2026-09-26: no more time on tokenization / caption
   tricks): the § 2.1 clause split and clause-varied captions, and the
   § 2.4 kind gate. B stays open and is read as `wdup`.
4. **§ 2.3 Stage I**: I0 / I1 / I2 against 0.5 · `v_line`.
5. **§ 2.2 F1b**, once its band-law row exists.
6. **§ 2.5**: shipping, then the sent test.

## 4. What closes it

- **B survives both levers** (a low-fill tier and the dose minimum) →
  doubling inside a word is part of the line mode in this
  parameterization. The fix moves to a caption marker, or B is accepted as
  a render-time filter.
- **F1b leaves the rows rendering a line alone** → the factorization does
  not clean `r_i` by gradient. The seed row stays the "alone" value, and
  trained rows are line-only (already the working assumption).
- **Stage I ties** → `b0709` stays the identity recipe.
