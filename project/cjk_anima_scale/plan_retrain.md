# plan_retrain — the singles re-seeded cold, with in-word data; pieces by per-glyph routing (2026-09-28)

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
replacing `paths.SEED_ROWS` for every later run, and every JA caption
addressed through those singles (§ 2). Nothing trains until the checks in
§ 4 are read.

## 1. Scope: singles only

The seed (`rows_step1_0921_merged`, 2 274 rows; `step1_0921` 374 at 30 k
steps + `step1_0921z` 1 900 at 152 k, the research line's lone-style data):

| kind | rows | in this plan |
|---|---|---|
| hiragana | 80 | retrain cold |
| katakana | 81 | retrain cold |
| kanji | 783 | retrain cold |
| other singles (punctuation etc.) | 9 | retrain cold, lone only |
| frequent glyphs with no single row yet | ≈ 560 (§ 2) | train cold (C-k's 388 are 386 of them) |
| pieces (2 / 3 / 4 / 5 glyphs) | 797 / 354 / 140 / 29 | **not trained**: § 2 routes around them |

## 2. Pieces: per-glyph routing (hypothesis.md § 3, P2)

**Per-glyph routing** is an encoder flag. Every JA Qwen token goes, on the
T5 side, to its glyphs' single rows; the Qwen text is untouched.
`HybridT5Encoder` already regroups byte fragments per char, and the flag
makes that the only path. The T5 side becomes ≈ 1 500 single rows under
27 k Qwen JA tokens (**Qwen / T5 ≈ 3**, EN's is 2.4), and every row is
shared by every word it appears in. Piece rows go unused, so the piece
question disappears. It was gated on P1 ("P2 follows P0 / P1 … only if P1
composes"), and P1b composes.

For it:
- **こんにちは**: the piece row renders 0 on every arm on record (300f
  0 → 0; B0 at × 1 and × 3). Spelled on `p1_mix`'s singles: ≤ 1 edit
  11 / 16, exact 5. Spell 2026-09-26 had the same direction on the seed
  (trained singles 13 / 32 within 2 edits vs the piece row's 4).
- **Coverage** (manga109s dialogue pool, CPU 2026-09-28). The seed's 953
  singles already hold **96.6 %** of the JA glyph occurrences (941 of 1 890
  distinct glyphs):

  | most frequent glyphs | occurrences covered | not in the seed's singles | of them in C-k |
  |---|---|---|---|
  | 1 000 | 99.06 % | 161 (159 kanji) | 46 |
  | 1 500 | 99.85 % | 563 (559 kanji) | 386 |
  | 2 000 | 100 % | — | — |

  Pieces cover ≈ 85 % of lines with 1 900 cold rows
  (`step1_0921z`). Singles reach 99.85 % with ≈ 1 500.
- **の / を** are addressable. The spelled-caption problem (their
  space-prefixed form is another Qwen token) is a tokenization artefact of
  spelling. Routing maps the glyph to its single row at the encoder.
- plan_2900 B and C-p (the piece runs, ≈ 18 h) are no longer needed.

Against it, or unread:
- **The routed caption has not been rendered.** T5 gets per-glyph ids and
  Qwen gets the whole word (c1's `ja_hybrid`). c1 found Qwen's word context
  moves the adapter's context cos by nothing (0.965 → 0.965), but that is
  an adapter read, not a render (§ 4 C0).
- **Doubling reaches every caption.** Every JA string becomes a run of
  single rows, so the in-word `dup` of the singles (4 / 16 on `p1_mix`) is
  what every caption carries.
- The piece training on record (300f, `300f_sp`, the long pieces) is sunk.

## 3. Data: what is new

Two of the three tiers exist already:

- **lone** = the production `b0709` group (`scene_single` + `grid_single`,
  σ 0.7–0.9), at share 0.5 of the kind's items (P1b's 1 : 2 lone to in-word).
- **in-word, count** = Stage B's `scene_single_small` in `b0507`
  (weight 0.3).
- **in-word, spelled** = Stage B's `scene_spelled` (unspaced image, spaced
  caption) in `b0507` / `b0305`. **Its word source is the new part.** Once
  C0 has passed, the caption can be the routed unspaced form instead of the
  spaced one. Both reach the same T5 rows.

Stage B took whole dialogue lines of 2–6 glyphs, every glyph a donor, none
repeated. Over the seed's 953 singles (の / を excluded, as spelled), the
dialogue pool gives:

| word source | words | kana median words / glyph | kanji median | kanji with 0 | kanji < 5 |
|---|---|---|---|---|---|
| whole lines, 2–6 glyphs | 7 607 | 188 (hira), 39 (kata) | **3** | **101** | **496** |
| windows of lines, 2–4 glyphs | 204 053 | 725 (kata) | 90 | 2 | 4 |
| windows of lines, 2–6 glyphs | 380 447 | 1 406 (kata) | 162 | 2 | 3 |

(CPU, 2026-09-28; the two kanji at 0 do not occur in the corpus at all.)

Whole lines cover kana only. Kanji need **windows**: any substring of a
dialogue line whose glyphs are all singles of the run, none repeated. A
window can cross a word boundary. Nothing on record says the DiT needs
real words (JA is rendered, never read for meaning), and C3 is the check.
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

## 4. Checks before the retrain

Each is launched by hand after the previous read. C0 and C1 run on rows
that exist (`p1_mix`) or on Stage B's items as they stand.

| # | what | answers | GPU |
|---|---|---|---|
| **C0** | **P2 render.** The routing flag (encoder + the eval path), then `p1_mix`'s rows on the unspaced こんにちは: routed vs spelled (cached, 11 / 16) vs unrouted (the seed's piece row, cached on the floor) | Does a routed caption render what the spelled one does? Routed ≈ spelled (≤ 1 edit within noise of 11 / 16) → § 2 holds and pieces stay out. Routed ≪ spelled → Qwen's word context interferes at render, and the piece question reopens (a cold-piece micro in P1's shape). | ≈ 16–32 renders |
| C1 | `p1_lone`: 36 donors, cold, `b0709` only, 90 steps / row; render read + c4 | Is the in-word tier load-bearing, or is cold alone enough? Lone-only cold composes like `p1_mix` → § 3's new tier is not needed, and the retrain is the existing TABLE, cold. Ends context-immune (c4 ≫ 0.82) with no composition → the in-word tier stays. | ≈ 25 min + read |
| C2 | the composition read widened: 6–8 held-in kana words (donor glyphs, trigram-held-out), spelled **and routed**, floor rendered **once**; re-read seed / Stage B / `p1_cold` / `p1_mix` (/ `p1_lone`) | `p1_mix`'s composition on more than one word × 16 renders, and C0 on more than one word; in-word `dup` with it | floor ≈ 256 renders + ≈ 256 per arm |
| C3 | 36 cold kanji donors (mixed density, some from the ≈ 560 with no row yet), windowed in-word pool + lone, the § 5 budget; read the kanji singles + 4–6 held-in kanji-bearing words (routed) | Does the recipe carry from kana to kanji: identity at the cold-kanji budget, composition with windows? It also sets the kanji steps / row. | ≈ 36 × 225 steps ≈ 1 h + read |

## 5. The full run

Cold singles use `budget.py`'s cold single row (150). The mix multiplies
by (all items / in-word items) = 1.5, so the in-word items keep their
exposure (P1b's 90 → 135):

| run | rows | steps / row | steps | local (8.3 k/h) | Colab G4 (≈ 25 k/h) |
|---|---|---|---|---|---|
| `retrain_kana` (hira + kata + other + the 4 missing kana) | ≈ 175 | 135 (P1b) or 225 | 24 k / 39 k | ≈ 2.9 / 4.7 h | ≈ 1–1.6 h |
| `retrain_kanji` (the seed's 783 + ≈ 560 frequent, C-k's 386 among them) | ≈ 1 340 | 225 | ≈ 300 k | ≈ 36 h | ≈ 12 h |

The kana run's 135 is P1b's point (cold hiragana, singles at the floor).
Katakana at 135 is unread, so C1 / C2 decide it. The kanji budget is
the cold-kanji row × 1.5 until C3 sets it. The kanji run can split by
frequency (the seed's 783 first, then the ≈ 560) and merge. The top-1 000
cut (161 new, 99.06 %) is the smaller option.

**Order and context.** A spelled window mixes kana and kanji. A glyph
outside the training run rides frozen at the seed, which is the
context-immune value this plan replaces. So kana trains first, and the
kanji run takes the kana run's merged rows as its context/seed file.
`paths.SEED_ROWS` is a constant today, so this needs a per-run seed
override (§ 6).

## 6. Code

0. (C0) Per-glyph routing: an encoder flag in
   `library/anima/ext_vocab.py::HybridT5Encoder`, reached from inference
   and from the line's eval path (the `native` stage encodes through the
   pipeline). Default off until the new seed is baked.
1. `recipes.py`: `scene_spelled` + `scene_single_small` promoted from
   `experiments/stage_b` (byte-faithful), and the windowed word pool
   (glyph-first draw, no repeats, trigram hold-out of the read words,
   spelling check).
2. `builder.TABLE`: the single kind's three groups (§ 3).
3. `budget.py`: the × 1.5 for the in-word mix, as a row with provenance
   (P1b).
4. `train.py`: `cold` becomes the rule for this run (it exists as an
   experiment flag today), and the seed/context file becomes a per-run
   override so `retrain_kanji` can sit on `retrain_kana`'s rows.
5. New seed = `retrain_kana` + `retrain_kanji` (`scale.py <out> merge`),
   → `rows_retrain_merged`. The old seed's piece rows ride along unused.
   Then `paths.SEED_ROWS` moves, and the floor is re-rendered once on the
   new seed (every later run reads against it). The old seed's floor stays
   as the record the retrain itself is read against.
6. Bake with routing on: the pack ships the flag as its default, which is
   a pack-format decision.

## 7. Open (user)

- **plan_2900.** Run A (dense kanji, warm from the old seed) is superseded
  by `retrain_kanji`. C-k (388 cold kanji, lone `b0709`) is 386 of the
  ≈ 560 frequent glyphs with no row, so it folds into `retrain_kanji`
  instead of running on its own. B and C-p (pieces) stop if C0 passes.
- **Where the kanji run trains** (≈ 36 h local vs ≈ 12 h on a G4; half
  of each at the top-1 000 cut).
- **How far the cut goes**: top 1 000 glyphs (99.06 %) or 1 500 (99.85 %).
