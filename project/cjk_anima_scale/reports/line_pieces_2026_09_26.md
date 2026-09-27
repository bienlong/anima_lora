# line_pieces — proposal § 2.4: the run gate on pieces (2026-09-26)

The operating point (the seed rows + 0.5 · `v_line`, arm `tf_l1_line0.5`),
read on the rulers whose floor the seed dir caches. **Verdict: the gate
helps runs of singles, gives nothing to the pieces inside a run, and adds
doubling on the neighbouring singles.** A piece that sits in a run
(った in やったネ, ちょっと in ちょっと来い) renders 0 / 16 on the floor
and 0 / 16 under the mode. The one short run with a piece (おしい) gains.
The two longer ones gain only in-word doubling of a single beside the
piece (やったネ `wdup` 2 → 9 / 16, read ややたえ…, ややろ). The user's
ComfyUI へんたい (へ ん + たい) → へんてだ♥ is the same shape: the singles
are right and the piece is lost.

One job, `20260926-215922-ec3bac` (≈ 3 min of renders). Script
`experiments/line_pieces/run_exp.py`; envelope
`experiments/line_pieces/results/20260926-2159-s1/result.json`. Reads:
`output/cjk_anima_scale/experiments/tf_l1_line0.5/{native_sent,target}/`.

## 1. Which keys the gate sees (`plan` leg, CPU)

- **Piece ruler: every key is one ext id** (しい った です メン すごい
  ちょっと ありがとう こんにちは あと きて こう こと). A lone piece has no
  pack neighbour, so the arm renders the seed rows there: the floor by
  construction. Nothing rendered.
- **Sent ruler:** はい = は + い (two singles), おしい = お + しい,
  やったネ = や + った + ネ, ちょっと来い = ちょっと + 来 + い: gate on.
  こんにちは and ありがとう are one id each: gate off, the render-noise
  control.
- **Target ruler** (the user's hoshino-ai captions): はい × 4 clause shapes
  gate on, こんにちは × 3 gate off.

## 2. The reads (/ 16 per sent key, / 8 · / 6 target)

| key | ids | floor contained · exact · ≤ 1 · `wdup` | + 0.5 · `v_line` |
|---|---|---|---|
| はい | single + single | 4 · 4 · – · 2 | **14 · 12** · – · 2 |
| おしい | single + piece | 6 · 5 · 8 · 1 | 8 · 7 · **14** · 3 |
| やったネ | single + piece + single | 0 · 0 · 0 · 2 | 0 · 0 · 0 · **9** |
| ちょっと来い | piece + kanji + single | 0 · 0 · 0 · 4 | 0 · 0 · 0 · 6 |
| こんにちは (gate off) | piece | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 |
| ありがとう (gate off) | piece | 0 · 0 · 0 · 1 | 0 · 0 · 0 · 1 |
| target はい (4 clauses) | single + single | 0 · 0 · – · 0 / 8 | **6 · 5** · – · 1 / 8 |

- Paired vs floor, gate-on sent renders (n 64): contained 15 / 3
  (p 0.008), exact 13 / 3 (p 0.02), ≤ 1 edit 6 / 0 (p 0.03), `wdup` 14 / 3
  (p 0.013).
- **Gate off is identical to the floor** (n 32 + 6: 0 changes on every
  metric but one `dup` read). The renders are deterministic, so every
  difference above is the mode.
- The target はい includes the clause shapes the recipes never train
  (`Japanese text that reads "はい"`, `Text that reads as "はい"`). The
  mode reaches them: 0 → 6 / 8 contained.

## 3. What the renders show

- **やったネ:** the floor reads や alone or ややふ. Under the mode 9 / 16
  open with やや, and った never appears (ややたえ女本, ややきくにえ,
  ややろ, ややふ). The piece row's slot comes out as other glyphs, and the
  single before it doubles.
- **ちょっと来い:** ちょっと is absent on both arms. The mode lengthens
  the tail (来いい, 来来いい, 東いい, 柔率いい): the doubled glyph is い,
  a single, at the end, the word-final pattern of
  `doubling_box_2026_09_26.md` § 3.
- **おしい:** reads おしい more often (≤ 1 edit 8 → 14). The piece しい
  renders when it closes a two-token run.

## 4. What it decides

- **Pieces do not lose identity to the gate. They had none in a run to
  lose** (った / ちょっと 0 on the floor). What the gate adds to a run with
  a piece is B on the singles beside it.
- The proposal's branch "if pieces lose → gate by kind" is triggered on
  the doubling side, not on identity. The kind gate (`v_line` only on
  single-kind rows) is the next read: a hook change plus the same
  ≈ 3 min read. Which row B rides on is open: `v_line` on the piece row, or
  on や and ネ. The kind gate answers it (`v_line` stays on や and ネ and
  leaves った).
- A piece mode (`u_P` post hoc, or a `v_line` trained on piece runs) is what
  would make った render in a run. Nothing here reads it.
- The piece ruler as built (lone pieces) cannot see the line mode. A piece
  read under the mode needs run keys (piece + piece, piece + single), and
  those need new floor keys.
