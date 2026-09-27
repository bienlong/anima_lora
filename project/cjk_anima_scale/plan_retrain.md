# plan_retrain — the singles re-seeded cold, with in-word data (2026-09-28)

Why: hypothesis.md § 4 P1 / P1b. On Stage B's 36 donor kana, rows trained
**cold from the pack rows** on in-word items + the lone singles group
(`p1_mix`) hold the floor's singles (official 82 vs 91 / 144, p 0.69;
repeat 30 vs 29) and compose (こんにちは ≤ 1 edit 0 → 11 / 16, exact
0 → 5, `dup` 4). The adapter reads them in context (c4 0.817; seed 0.977).
The same items warm from the seed (Stage B) compose less (6 / 16) and
double (`dup` 13), and a norm cap is nearly redundant (cold rows stop at
≈ 230–240 on their own). The seed's rows carry a context-immune direction.
Training on top of it keeps that direction.

Target: a new seed whose singles are cold-trained on lone + in-word data,
replacing `paths.SEED_ROWS` for every later run.

## 1. Scope

The seed (`rows_step1_0921_merged`, 2 274 rows; `step1_0921` 374 at 30 k
steps + `step1_0921z` 1 900 at 152 k, the research line's lone-style data):

| kind | rows | in this plan |
|---|---|---|
| hiragana | 80 | retrain cold |
| katakana | 81 | retrain cold |
| kanji | 783 | retrain cold |
| other singles (punctuation etc.) | 9 | retrain cold, lone only |
| pieces (2 / 3 / 4 / 5 glyphs) | 797 / 354 / 140 / 29 | **kept at the seed** (§ 6 open) |

P1 read only singles: a spelled caption addresses single rows, and a piece
row is the only address of its string (hypothesis.md H2). Pieces stay at
their seed values until a piece read says otherwise.

## 2. Data: what is new

Two of the three tiers exist already:

- **lone** = the production `b0709` group (`scene_single` + `grid_single`,
  σ 0.7–0.9), at share 0.5 of the kind's items (P1b's 1 : 2 lone to in-word).
- **in-word, count** = Stage B's `scene_single_small` in `b0507`
  (weight 0.3).
- **in-word, spelled** = Stage B's `scene_spelled` (unspaced image, spaced
  caption) in `b0507` / `b0305`. **Its word source is the new part.**

Stage B took whole dialogue lines of 2–6 glyphs, every glyph a donor, none
repeated. Over the seed's 953 singles (の / を excluded: their
space-prefixed form is another Qwen token, so a spelled caption cannot
address them), the manga109s dialogue pool gives:

| word source | words | kana median words / glyph | kanji median | kanji with 0 | kanji < 5 |
|---|---|---|---|---|---|
| whole lines, 2–6 glyphs | 7 607 | 188 (hira), 39 (kata) | **3** | **101** | **496** |
| windows of lines, 2–4 glyphs | 204 053 | 725 (kata) | 90 | 2 | 4 |
| windows of lines, 2–6 glyphs | 380 447 | 1 406 (kata) | 162 | 2 | 3 |

(CPU, 2026-09-28; the two kanji at 0 do not occur in the corpus at all.)

Whole lines cover kana only. Kanji need **windows**: any substring of a
dialogue line whose glyphs are all singles of the run, none repeated. A
window can cross a word boundary. Nothing on record says the DiT needs
real words (JA is rendered, never read for meaning), and M2 is the check.
Rules carried over from Stage B:
- A word draw picks a glyph uniformly, then one of its words, so exposure
  is per row.
- No glyph repeated inside a word, so training does not teach doubling.
- The read words are trigram-held-out from the pool.
- Every spelled word must encode to its glyphs' single ids and nothing
  else (`stage_b.check_spelling`). A word that does not is dropped.

So the retrain needs one production change, not a new data pipeline: a
`scene_spelled` recipe + a windowed word pool in `recipes.py`, and the
single kind's groups in `builder.TABLE` become lone `b0709` + spelled/count
`b0507` + spelled `b0305` (a rule change, with this plan and the P1 read as
its report). The builder, pools, trainer, eval and floor stay as they are.
That is why this plan lives in the line and not in a new project.

## 3. Micros before the full run

Each is launched by hand after the previous read. Only M1 is cheap
enough to run on Stage B's items as they stand.

| # | what | answers | GPU |
|---|---|---|---|
| M1 | `p1_lone`: 36 donors, cold, `b0709` only, 90 steps / row | Is the in-word tier load-bearing, or is "cold" alone enough? If lone-only cold composes like `p1_mix`, § 2's new tier is not needed and the retrain is the existing TABLE, cold. If it ends context-immune (c4 ≫ 0.82) with no composition, the in-word tier stays. | ≈ 25 min + read |
| M2 | the composition read widened: 6–8 held-in kana words (donor glyphs, trigram-held-out), floor rendered **once**; re-read seed / Stage B / `p1_cold` / `p1_mix` (/ `p1_lone`) | `p1_mix`'s composition on more than one word × 16 renders | floor ≈ 128 renders + ≈ 128 per arm |
| M3 | 36 cold kanji donors (mixed density), windowed in-word pool + lone, budget as § 4, read: the kanji singles + 4–6 held-in kanji-bearing words | Does the recipe carry from kana to kanji (identity at the cold-kanji budget, composition with windows)? | ≈ 36 × 225 steps ≈ 1 h + read |

## 4. The full run

Cold singles use `budget.py`'s cold single row (150). The mix multiplies
by (all items / in-word items) = 1.5, so the in-word items keep their
exposure (P1b's 90 → 135):

| run | rows | steps / row | steps | local (8.3 k/h) | Colab G4 (≈ 25 k/h) |
|---|---|---|---|---|---|
| `retrain_kana` (hira + kata + other) | 170 | 135 (P1b) or 225 | 23 k / 38 k | ≈ 2.8 / 4.6 h | ≈ 1–1.5 h |
| `retrain_kanji` | 783 | 225 | 176 k | ≈ 21 h | ≈ 7 h |

The kana run's 135 is P1b's point (cold hiragana, singles at the floor).
Katakana at 135 is unread, so M2 / M1 reads decide it. The kanji budget is
the cold-kanji row × 1.5 until M3 sets it.

**Order and context.** A spelled window mixes kana and kanji. A glyph
outside the training run rides frozen at the seed, which is the
context-immune value this plan replaces. So kana trains first, and the
kanji run takes the kana run's merged rows as its context/seed file.
`paths.SEED_ROWS` is a constant today, so this needs a per-run seed
override (§ 5).

## 5. Code

1. `recipes.py`: `scene_spelled` + `scene_single_small` promoted from
   `experiments/stage_b` (byte-faithful), and the windowed word pool
   (glyph-first draw, no repeats, の / を out, trigram hold-out of the read
   words, spelling check).
2. `builder.TABLE`: the single kind's three groups (§ 2).
3. `budget.py`: the × 1.5 for the in-word mix, as a row with provenance
   (P1b).
4. `train.py`: `cold` becomes the rule for this run (it exists as an
   experiment flag today), and the seed/context file becomes a per-run
   override so `retrain_kanji` can sit on `retrain_kana`'s rows.
5. New seed = `retrain_kana` + `retrain_kanji` singles over the old seed's
   pieces (`scale.py <out> merge`), → `rows_retrain_merged`. Then
   `paths.SEED_ROWS` moves, and the floor is re-rendered once on the new
   seed (every later run reads against it). The old seed's floor stays as
   the record the retrain itself is read against.

## 6. Open (user)

- **Pieces.** Kept at the seed here. Alternatives: a piece micro (cold
  pieces, P1's shape) before deciding, or P2 (per-glyph routing), which
  leaves piece rows unused and makes the question moot.
- **plan_2900.** Run A (dense kanji, warm from the old seed) is superseded
  by `retrain_kanji`. C-k (388 new cold kanji, lone `b0709`) should wait
  for M1: if the in-word tier is load-bearing, C-k's data needs it too.
  B and C-p are pieces, so they are unaffected while pieces stay out of scope.
- **Where the kanji run trains** (≈ 21 h local vs ≈ 7 h on a G4).
