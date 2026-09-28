# plan_retrain — what is left (2026-09-28)

Target: a new seed whose singles are cold-trained on lone + in-word
(routed windows) data, replacing `paths.SEED_ROWS` for every later run, and
every JA caption addressed through those singles by per-glyph routing.
Everything read so far — why (P1 / P1b), the routing case, the windowed
word pool, checks C0–C3, `retrain_kana` and its read, the code that landed
— is in `retrain_experiments.md`.

Where it stands: routing holds and pieces stay out (C0 / C2); the in-word
tier is load-bearing (C1); windows compose kanji (C3); `retrain_kana`
(174 cold kana rows) composes like `p1_mix` and holds the singles, so its
rows are the kana half of the new seed. `retrain_kanji_b1` is trained
(2026-09-28, unread); left: b2, b3 (§ 2) and the bake.

## 1. The kanji budget — set 2026-09-28

**Ink < 10 → 225 / row, ink ≥ 10 → 337 / row** (user). C3 at 450 brought
the seed's dense kanji back to the floor (official 31 → 57 / 192, floor
65, p 0.38 vs the floor), and the gain over 225 went to ink-dense glyphs
(ink ≥ 10.5: 44 → 86; ink < 10: 38 → 34). 337 is the midpoint, unmeasured.
Read, split and the step totals: `retrain_experiments.md` § 4 (C3 at 450).
In code (2026-09-28): `budget.RULES` splits the cold kanji row by ink
(150 / 225 × 1.5 = 225 / 337.5 per row; ink pinned in
`assets/glyph_ink.json`), and a run's singles may mix budgets
(`budget.run_budget`): the builder repeats each single 2 : 3 in the draw
pools (`draw_weights`, lone / count tiers and the window keys) and the
trainer sums each row's own steps. Piece runs still take one factor.

The record before the 450 read:

At 225 / row cold (C3), new kanji gain identity and the seed's dense kanji
lose half (65 → 31 / 192). Candidates: more steps / row (C3 at 450 on the
same data, ≈ 2 h), a larger lone share for kanji, or both.

What is on record (2026-09-28):

| read | setting | result |
|---|---|---|
| stage_i I0 | 12 kanji with no seed row, cold, lone only, 90 → 270 / row | contained 59 → 117 / 192 |
| C3 | 36 kanji, cold, lone 0.5 + windows, 225 / row (150 × 1.5) | new kanji official 0 → 51, contained 4 → 104 / 192 |
| C3 | the seed's dense kanji (dense_a0's 12) in the same run | official 65 → 31, contained 110 → 93 (感 愛 飲 最 様 at 0) |
| C3 | six kanji-bearing words | ≤ 1 edit 3 → 36 / 96 |

`budget.py`'s kanji row (150) is a pick between stage_i's 90 and 270,
unmeasured (its source string says so). **A caution on "more steps"**:
the seed's rows had ≈ 80 steps / row of lone-style data (`step1_0921`
374 rows at 30 k, `step1_0921z` 1 900 at 152 k), and C3's lone share is a
third of 225 ≈ 75 / row, about the same. The dense kanji still halved,
so total steps alone may not bring them back: the in-word items may
compete with identity, or the seed's history holds more than that count.
Both are unread.

**Order** (done 2026-09-28): C3 at 450 / row ran first and brought the dense
kanji back; the lone-share arm was not needed.

## 2. `retrain_kanji` — three chained batches (set 2026-09-28)

The seed's 783 kanji + the 158 kanji in the top 1 000 glyphs with no seed
row, 941 rows, 271 k steps, cold. 緒 単 戻 are among the 158: Qwen splits
them into byte fragments and their pack row is a `char` row. Routing
already sent them there at encode; the line's vocab → idx lookup now does
too (`data.inventory.qwen_pieces(char_rows=True)`, every `cjk_scale`
caller), so they train like any single (b1 / b2 / b3 hold one each). Ranked by `dialogue_2_10` count and cut into three batches at
equal steps (user: three, not four):

| run | kanji | new | ink < 10 | steps | local (8.3 k/h) | Colab G4 (≈ 25 k/h) | status |
|---|---|---|---|---|---|---|---|
| `retrain_kanji_b1` | 329 (count 1 171 → 45) | 91 | 181 | 90.7 k | ≈ 10.9 h | ≈ 3.6 h | **trained** 2026-09-28 on a G4: 90 749 steps in 225.3 min (6.7 it/s); `trained.pt` pulled (md5 `db05c109…`), unread |
| `retrain_kanji_b2` | 307 (45 → 18) | 22 | 119 | 90.3 k | ≈ 10.9 h | ≈ 3.6 h | planned |
| `retrain_kanji_b3` | 305 (18 → 0) | 45 | 113 | 90.3 k | ≈ 10.9 h | ≈ 3.6 h | planned |

Vocabs: `assets/vocabs/ja_retrain_kanji_b{1,2,3}.txt` (glyph, count, ink,
seed row). Read: C3's six kanji-bearing words (held out of every batch's
windows).

**Chained by `context`** (the run file's third key): b1 sits on
`retrain_kana`, b2 on b1, b3 on b2. A batch warms from / freezes / merges
onto its context's rows, and its windows may carry every single trained
down the chain (`builder.context_singles`: the kana, then the earlier
batches' kanji) beside its own. A kanji of a later batch still has the
seed's row, so it stays out of the windows. b1 dry check (CPU, before the
encoding check, before 緒 joined): 200 792 windows over its 328 kanji + 165
context kana, per kanji min 77 / median 611; the same kanji alone give
4 426 (median 23, one at 0). Draw weights 2 : 3 (181 : 148 with 緒).

The chain replaces the merge: b3's `trained.pt` holds `retrain_kana`'s rows
+ all three batches. The batches run in order (b2 needs b1's rows). A VM
needs each context run's `trained.pt` **and** `data/vocabs.json`
(`merge.idx_source`: `retrain_kana` names specs, not a file).

`train.py` refuses an attached pack other than the raw one
(`paths.RAW_PACK_SHA`, `7b9fce0bb57b…`): cold rows start at its rows.

## 3. The new seed and the bake

1. New seed = `retrain_kanji_b3`'s `trained.pt` (the chain holds
   `retrain_kana` + b1–b3; no merge). The old seed's piece rows ride along
   unused.
2. `paths.SEED_ROWS` moves, and the floor is re-rendered once on the new
   seed (every later run reads against it). The old seed's floor stays as
   the record the retrain itself is read against.
3. Bake with routing on: the pack ships the flag as its default, which is
   a pack-format decision.

## 4. Open

- **Doubling.** `retrain_kana`'s `dup` rose over `p1_mix`'s (42 vs 31,
  p 0.06; collapsing doubles lifts ≤ 1 edit 29 → 48 / 64). Why is unread:
  window length 2–6 vs whole lines, row count, window variety.
- **plan_2900.** Run A (dense kanji, warm from the old seed) is superseded
  by `retrain_kanji`. C-k (388 cold kanji, lone `b0709`) is 386 of the
  ≈ 560 frequent glyphs with no row, so it folds into `retrain_kanji`
  instead of running on its own. B and C-p (pieces) stop: C0 passed.
- **Where the kanji batches train** (≈ 10.9 h local vs ≈ 3.6 h on a G4
  each). (user) b1 ran on a Colab G4 (3.75 h).
- **The cut** is top 1 000 (99.06 %; user 2026-09-28). The 1 500 cut's
  ≈ 400 more kanji would be a fourth batch on b3's rows.
