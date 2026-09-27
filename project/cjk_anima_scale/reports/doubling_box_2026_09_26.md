# doubling_box — proposal § 2.1 (a): what in-word doubling (B) is (2026-09-26)

CPU only, the reads on disk: every spelled-word render of floor, Stage B's
α sweep (u0.5 / u1 / u2, rand1), `v_line` 1 and 0.5, spell_b, and the
Stage B / F1 donors' こんにちは. **Verdict: neither H1 nor H2 as written.
B is word-final and caption-conditioned.** The doubled glyph is an extra
glyph appended to a line at the line's own px. It is not padding a fixed
box. It is mostly the word's last glyph, and it lives under the `en` clause
(`japanese text. Japanese text reads as "…"`). Under `swap`
(`english text. English text reads as "…"`), the same rows compose as well
with in-word doubling at the floor. **That is the first axis that separates
composition from B.** The dose axis never did. Also: a third of the `dup`
metric of record is not in-word doubling.

Script: `experiments/doubling_box/run_exp.py`; envelope + per-render table
`experiments/doubling_box/results/20260926-2157-a1/{result.json,renders.json}`.

## 1. `dup` of record is a third reader noise

`stage_b.hits`' `dup` counts any doubled character in any read, from any
box. Split by what doubled (held-out words, / 160):

| | `dup` of record | a glyph of the word (B) | another kana | latin / digits / punct. |
|---|---|---|---|---|
| floor | 26 | **8** | 12 | 6 |
| rand1 | 20 | 1 | 11 | 8 |
| u1 | 44 | 25 | 13 | 6 |
| u2 | 73 | 45 | 15 | 13 |
| `v_line` 1 | 77 | 54 | 8 | 15 |
| `v_line` 0.5 | 54 | **34** | 5 | 15 |

- The non-B part (≈ 15–25) is flat across arms: `\\`, `mm`, `ee` and
  background text in other boxes. B proper is floor 8 → 34 at the
  operating point, so it is 4× the floor, not the 2× the metric of record
  showed. The +26 excess is unchanged.
- Below, "B" and "in-word dup" mean the word-glyph class (`wdup`).

## 2. H1 (a fill prior) — its predictions fail

- **px:** Stage B's word items train at px 35 (b0507, p10–p90 30–48) and
  18 (b0305, 15–22). The renders' line boxes (tight on the ink: checked on 16
  `v_line` 0.5 renders, `output/cjk_anima_scale/doubling_box_a1/boxes.png`) give px 55–85 median, and
  doubled ≈ clean in every arm (`v_line` 0.5: 85 vs 80; `v_line` 1: 56 vs
  61; u1: 58 vs 60; spell_b: 69 vs 75). No render sits at the trained line
  px. The native prompts draw the word as headline-size text, and there is
  no bubble to fill.
- **Box:** within one word × clause × prompt, a doubled seed's box is
  longer than a clean seed's at the same px (`v_line` 1: 20 / 27 keys
  longer, median +137 px, Δpx +2; `v_line` 0.5: 10 / 17, +40, Δpx −5;
  spell_b: 15 / 24, +21, Δpx +1). The extra glyph extends the line. A
  fixed-capacity box would hold the length and shrink the px.
- **Prompt:** B by scene prompt at `v_line` 0.5 is 8 2 5 4 5 6 2 2 / 20, and
  at `v_line` 1 it is 8 7 10 7 8 5 3 6. It is lowest on the portrait /
  simple-background prompt in every arm, but nowhere near the clause effect
  below.

## 3. H2 (a sequence failure) — position concentrates, but on the end

In-word doubling by position of the doubled glyph in the word:

| | first | inner | last |
|---|---|---|---|
| u1 | 6 | 6 | 13 |
| u2 | 21 | 8 | 16 |
| `v_line` 1 | 9 | 20 | 25 |
| `v_line` 0.5 | 4 | 6 | **24** |
| spell_b | 16 | 3 | 24 |
| donors' こんにちは (Stage B + F1) | 5 | 16 | 4 |

- At the operating point 24 / 34 are the word-final glyph. り is 18 of those
  (ひまわり みどり くもり end in it) and ら is 6. The renders read
  みとりり, びまりり, あどり + a mangled り, ぎとるりり: a trailing copy
  of the last glyph.
- spell_b's first-position count is あ in あり / あがり / ありがとう, and
  its 19 "pure padding" renders (collapse the run → exactly the word) are
  あありがとう-type. こんにちは doubles inside, at に after ん (んに 11 / 25).
  That is a bigram site, not the end.
- So B concentrates by position (H2's prediction). But the dominant position
  is the end of the word, where "the sequence ran out and the line went
  on" and "the order is weak" look the same. The clause read decides which.

## 4. The clause axis — composition and B separate

The held-out words per clause (/ 80 each; `le1` / `le2` per `stage_b.hits`):

| | clause | ≤ 1 edit | ≤ 2 edits | ≤ 1 collapsed | **in-word dup** | vertical |
|---|---|---|---|---|---|---|
| floor | en | 5 | 44 | 5 | 4 | 45 |
| | swap | 6 | 49 | 6 | 4 | 45 |
| u1 | en | 30 | 53 | 33 | 19 | 25 |
| | swap | 36 | 60 | 36 | **6** | 37 |
| u2 | en | 8 | 26 | 11 | 24 | 5 |
| | swap | 26 | 47 | 33 | 21 | 21 |
| `v_line` 1 | en | 26 | 46 | 36 | 37 | 2 |
| | swap | 38 | 59 | 42 | 17 | 14 |
| **`v_line` 0.5** | en | 40 | 64 | 49 | 27 | 11 |
| | swap | **40** | **67** | 40 | **7** | 23 |
| spell_b | en | 13 | 19 | 16 | 29 | 9 |
| | swap | 19 | 24 | 20 | 14 | 5 |

- **At the operating point the `swap` clause keeps all the composition
  (≤ 1 edit 40 = 40, ≤ 2 edits 67 vs 64) and drops in-word doubling to
  the floor (7 vs floor 4, en 27).** On every arm that composes, `swap` is ≥
  `en` on composition and ≤ on B.
- `en` is the clause the recipes train on: every `scene_spelled` item is
  captioned `japanese text` + `Japanese text reads as "…"`. The floor does
  not double under either clause, so B is the mode read in the Japanese-text
  context. It is neither the base model's Japanese prior alone nor the mode
  alone.
- `en` also renders fewer vertical lines at every dose (v0.5 11 vs 23), so
  the clause moves the layout as well as the count.
- u2 is the exception on the B side (24 · 21): at 2× dose the doubling
  saturates under both clauses.

## 5. What it decides

- **H1's lever (a low-fill `scene_spelled` tier) is not indicated.** B is
  not a box being filled at the trained px.
- **H2's lever (a finer dose) is not the separator either.** Dose never
  separated composition from B, and the clause does at a fixed dose.
- **The lever moves to the caption axis.** It is still open which part
  carries B: the `japanese text` tag, the `Japanese text reads as`
  sentence, or both. That read is training-free, but it needs new floor keys
  (the held-out words under the split clauses). Two follow-ons, if the
  split localizes it:
  - a data lever: `scene_spelled` captions drawn over clause variants, so
    `v_line` does not bind to the Japanese-text context;
  - the § 2.7 caption marker, re-scoped: the clause itself is already a
    marker the mode reads.
- **Metric:** future reads of B count `wdup` (a glyph of the word
  doubled), not `dup`.
