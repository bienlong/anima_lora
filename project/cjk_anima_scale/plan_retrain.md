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

**C0 read (2026-09-28, `experiments/p2_route/results/20260928-0100-c0/`,
job `20260928-010053-5dd1b7`): routing holds, pieces stay out.**
こんにちは (en, 16 renders), ≤ 1 edit / official / contained:

| rows | routed | spelled | unrouted (piece row) |
|---|---|---|---|
| `p1_mix` | **14** / 8 / 11 | 11 / 5 / 6 | 0 / 0 / 0 |
| seed (floor) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 (`dup` 6) |

Routed vs spelled on `p1_mix`: ≤ 1 edit 3 / 0 (p 0.25), contained 5 / 0
(p 0.06), `dup` 3 vs 4. Every routed render carried one ext row per glyph.
Qwen's whole-word context does not interfere at render, and may help. One
word; C2 widens it.

- Spelling is not a neutral address beyond の / を: a spaced kanji is a
  space-prefixed Qwen token with its own row (`" 名"` → row 675, 名 → 22).
  Routing sends both to the glyph's own row, so kanji data (C3, the
  retrain) uses routed captions.
- The flag: `mapping["glyph_route"]` (in the digest only when set) or
  `ANIMA_VOCAB_GLYPH_ROUTE=1|0` (overrides, read in
  `VocabPack.build_encoder`; the line's TE cache key carries it). Every JA
  Qwen token (27 143) splits; none keeps its piece row. Experiments set the
  env var in-process, never in the submit shell (a daemon booted from that
  shell would route every later job).

**C1 read (2026-09-28, `experiments/p1_cap/results/20260928-0103-lone1/`,
job `20260928-010327-fab1ec`): the in-word tier is load-bearing.**
`p1_lone` = the 36 donors cold on the production table alone (`b0709`,
2 400 lone items, 90 steps / row, data `run0928_p1_lone`):

| donor keys (en) | floor | Stage B | `p1_cold` | `p1_mix` | **`p1_lone`** |
|---|---|---|---|---|---|
| end row norm | ≈ 320 | 311 | 229 | 240 | **230** |
| こんにちは ≤ 1 edit / 16 (spelled) | 0 | 6 | 12 | 11 | **0** |
| こんにちは ≤ 2 edits | 0 | 13 | 14 | 12 | 1 |
| singles official / 144 | 91 | 43 | 29 | 82 | **74** |
| singles repeat | 29 | 50 | 43 | 30 | 25 |

A cold start alone stops the rows at the same norm as the in-word arms and
composes nothing. The norm is not what composes; the in-word items are.
Singles: 74 vs the floor's 91 (paired 18 / 35, p 0.027), near `p1_mix`'s
82. So § 3's new tier stays. At the adapter (`ctx_trigger --probe c4`,
`results/20260928-0431-c4lone/`) the lone rows read less context than
`p1_mix`'s at nearly the same norm: out cos 0.893 at 228 vs 0.817 at 238
(seed 0.977 at 321, pack rows 0.624 at 182). The in-word items move the
rows' direction, not their size.

**C2 read (2026-09-28, `experiments/p2_route/results/20260928-0138-c2/`,
job `20260928-010523-19dc00`): `p1_mix` composes on eight words, and
routed = spelled everywhere.** こんにちは + たいせつ かなしい かんがえ
たすけて こうえん てつだう ことば (donor glyphs, trigram-held-out from the
donor words), en, 8 × 16 renders per condition:

| arm | ≤ 1 edit routed / spelled | ≤ 2 (routed) | official | `dup` | ≤ 1 with doubles collapsed |
|---|---|---|---|---|---|
| floor (seed) | 1 / 1 | 14 | 0 | 9 | 1 |
| Stage B (warm) | 23 / 24 | 55 | 2 | 81 | 46 |
| `p1_cold` | 65 / 65 | 96 | 23 | 60 | 79 |
| **`p1_mix`** | **80 / 75** | **110** | **26** | 55 | **105** |
| `p1_lone` | 9 / 10 | 37 | 0 | 40 | 10 |

- **Routing**: routed vs spelled within noise on every arm (`p1_mix` ≤ 1
  edit 15 / 10, p 0.42; floor ≤ 2 edits 2 / 10, p 0.04, both near zero).
  C0 generalises: § 2 holds.
- **`p1_mix` composes on every word** (≤ 1 edit 6–15 / 16 per word; vs the
  floor 79 / 0, p 3e-24), ahead of Stage B (62 / 5) and at or above
  `p1_cold` (34 / 19, p 0.053; ≤ 2 edits p 0.034) while holding the
  singles (C1 table).
- **`p1_lone` barely moves** (9 / 128): C1's one-word read holds on eight.
- **Doubling is the remaining cost**: 55 / 128 `p1_mix` words carry a
  doubled glyph, and collapsing them lifts ≤ 1 edit 80 → 105. It is below
  Stage B's (81) but far above the floor's 9, whose renders are mostly not
  the word.

**C3 read (2026-09-28, `experiments/c3_kanji/results/20260928-0246-c3/`,
job `20260928-011133-aa3912`): composition carries to kanji; identity is
bought for new kanji and lost for the seed's dense ones at 225 / row.**
36 kanji (stage_i's 12 with no seed row, dense_a0's 12 seed rows, 日本人大丈夫何時
(seed), 死父誰名 (no row)), cold, on `p1_mix`'s merged rows as context
(`train(…, context=)`), `p1_mix`'s table with windows (2–4 glyphs of
dialogue lines, 4 370, kanji-first draw; 輩 12, 精 / 奥 17 the thinnest) as
the in-word tier, routed captions, 6 000 items, 225 steps / row (59.5 min).
End row norm ≈ 270 (the kanji pack rows start at ≈ 203).

| read (en, 16 renders / key) | floor | `p1_mix` (kana context only) | **`c3_kanji`** |
|---|---|---|---|
| words ≤ 1 edit / 96 (routed) | 3 | 7 | **36** |
| words ≤ 2 edits | 28 | 33 | 68 |
| words official | 0 | 3 | 13 |
| words `dup` | 22 | 24 | 39 |
| new kanji (stage_i 12) official / 192 | 0 | = floor | **51** |
| new kanji contained | 4 | = floor | 104 |
| seed kanji (dense_a0 12) official / 192 | 65 | = floor | **31** |
| seed kanji contained | 110 | = floor | 93 |

Words, per word (≤ 1 edit / 16): 日本人 14, 大丈夫 9, 小山田 6, 愛してる 4,
何時間 3, 山田太郎 0 (≤ 2 edits 7). Paired vs the floor: words ≤ 1 edit
35 / 2 (p 1e-8); vs `p1_mix` 33 / 4 (p 1e-6), so it is the kanji rows, not
the kana context.

- **Windows compose.** Substrings that cross word boundaries were enough;
  nothing here needs real words.
- **New kanji get identity** at 225 / row (stage_i's lone-only I0 at
  90 / 270 contained 59 / 117, `budget.RULES`); 郎 野 太 stay at 0.
- **The seed's dense kanji lose half their identity** (official 65 → 31,
  paired 14 / 48, p 2e-5; 感 愛 飲 最 様 at 0). The kana run at 135 held
  the floor (C1 table: 82 vs 91). For kanji, 225 / row does not, so the
  kanji steps / row are not set yet (§ 5, § 7).
- Doubling rises with composition, as on kana (`dup` 39 vs 22).

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
   pipeline). Default off until the new seed is baked. **Done 2026-09-28**
   (C0 read above; `tests/test_ext_vocab_glyph_route.py`).
1. `recipes.py`: `scene_spelled` + `scene_single_small` promoted from
   `experiments/stage_b` (byte-faithful), and the windowed word pool
   (glyph-first draw, no repeats, trigram hold-out of the read words,
   spelling check).
2. `builder.TABLE`: the single kind's three groups (§ 3).
3. `budget.py`: the × 1.5 for the in-word mix, as a row with provenance
   (P1b).
4. `train.py`: `cold` becomes the rule for this run (it exists as an
   experiment flag today), and the seed/context file becomes a per-run
   override so `retrain_kanji` can sit on `retrain_kana`'s rows. The
   override exists (`train(…, context=)`, 2026-09-28, used by C3); `cold`
   and `budget.run_factor` still do not know each other (C3 patches the
   factor: seed and new kanji both train cold, one budget).
5. New seed = `retrain_kana` + `retrain_kanji` (`scale.py <out> merge`),
   → `rows_retrain_merged`. The old seed's piece rows ride along unused.
   Then `paths.SEED_ROWS` moves, and the floor is re-rendered once on the
   new seed (every later run reads against it). The old seed's floor stays
   as the record the retrain itself is read against.
6. Bake with routing on: the pack ships the flag as its default, which is
   a pack-format decision.

## 7. Open (user)

- **Kanji budget after C3.** At 225 / row cold, new kanji gain identity and
  the seed's dense kanji lose half (65 → 31 / 192). Candidates: more steps
  / row (C3 at 450 on the same data, ≈ 2 h), a larger lone share for
  kanji, or both. `retrain_kanji` waits on it.

- **plan_2900.** Run A (dense kanji, warm from the old seed) is superseded
  by `retrain_kanji`. C-k (388 cold kanji, lone `b0709`) is 386 of the
  ≈ 560 frequent glyphs with no row, so it folds into `retrain_kanji`
  instead of running on its own. B and C-p (pieces) stop if C0 passes.
- **Where the kanji run trains** (≈ 36 h local vs ≈ 12 h on a G4; half
  of each at the top-1 000 cut).
- **How far the cut goes**: top 1 000 glyphs (99.06 %) or 1 500 (99.85 %).
