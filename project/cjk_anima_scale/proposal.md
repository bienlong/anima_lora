# proposal — the line mode: what is left (2026-09-27)

**The target (user, 2026-09-27): one `v_line` that is summed, ungated, onto
any row trained by `b0709`**, so that the row still renders its glyph
alone and composes in a word, kana and kanji. The main line is the
quality of `v_line` itself (§ 2.0). The run gate, the dose and the kind
gate are the fallback that ships until then.

Open items only; what ran is in the reports (the verdicts are indexed in
§ 0). Section numbers stay as reports and experiment docstrings cite them,
so a closed section keeps a one-line stub. Numbers cited from before the
2026-09-26 merge refer to the pre-merge files at git `48e1d6ab`.

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
| § 2.3 Stage I on 12 cold kanji | alone: I0 wins at 90 and at 270 steps / row (x3: contained 117 vs I1 84 / 192, p 9e-6), I2 out; `b0709` stays. Under the mode the kanji words never compose (≥ 2 glyphs in order ≤ 3 / 96 on every arm) | `reports/stage_i_2026_09_26.md` § 5 |
| § 2.3 kanji_mode: warm seed kanji under 0.5 · `v_line` | barely compose: ≤ 1 edit 2 → 10 / 96 (p 0.04), ≥ 2 in order 3 → 14, against kana 5 → 40 / 80 and 72 / 80. Kanji vs kana, not row maturity | `experiments/kanji_mode/results/20260927-1138-k1-score/` (no report) |
| § 2.4: the gate on runs with pieces | a piece in a run renders 0 before and after. はい contained 4 → 14 / 16, target はい 0 → 6 / 8. The single beside a piece doubles (やったネ `wdup` 2 → 9). A line-block pack in ComfyUI rendered へんたい as へんてだ♥ | `reports/line_pieces_2026_09_26.md` |

The best configuration on record for a vocab without line training is
**the seed rows + 0.5 · `v_line` + the run gate**. It is gated because the
ungated sum costs the glyph alone: `u_S` (cos 0.79 to `v_line`) added to
every row takes singles official 149 → 139 → 109 → 49 and repeat
25 → 36 → 53 → 60 over α 0.5 / 1 / 2 (Stage B § 7). Two kinds of doubling
remain:
- **A, a glyph alone repeating itself (あ → ああ):** bounded. The gate
  keeps a lone row at its seed value.
- **B, a doubled glyph inside a word (ひまわり → ひまわりり):** open. Counted
  as a glyph of the word doubled (`wdup`), it is 34 / 160 vs the floor's 8.
  It is mostly the word-final glyph, and it lives under the `en` clause
  (§ 2.1). Collapsing doubled glyphs lifts the words from 80 to 89, so B
  costs about a tenth of the words.

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
  - The pack side (commit `c0fc7414`): the line block — the gate in the
    encoder (`HybridT5Encoder.apply_line`, a pack without `line` encodes
    as before), `bake_vocab_pack.py --line_from … --line_dose`, the node's
    `_vendor/` synced. First pack: `anima_cjk_vocab_pack_300fsp_line05`
    (local).

## 2. Open items

### 2.0 An ungated `v_line` — the main line

```
row_eff(i) = r_i + v_line        for every r_i trained by b0709, no gate
```

Identity runs stay `b0709`-only and cheap. Composition is one shared
vector, trained once, and the pack carries no gate and no kind rule.

**Why the current `v_line` is not it.** F1 trained it gated: it was on
only in spelled words, and the count tier's lone items saw the rows
without it. Nothing asked it to be harmless on a lone glyph, and the
ungated `u_S` reads (§ 0) say it is not. Its data was also 36 kana in
single-only words under one clause, and it composes nothing else (kanji
§ 2.3, a piece in a run § 2.4). F0 found the adapter row-wise, so a
vector on every row reaches a lone glyph unchanged; whether it acts
differently alone and in a word depends on the DiT reading several tokens
at once. Whether such a direction exists is the open question, and F2a
reads it first.

**Acceptance** (on `b0709` rows that `v_line` never trained against):
1. **Alone at the floor:** the single ruler (official / repeat / line
   ≥ 3 glyphs) and the piece ruler, ungated.
2. **Composes:** the held-out kana words (≤ 1 edit, gated 0.5 = 80 / 160)
   and kanji_mode's six words (floor ≤ 1 edit 2 / 96), with `wdup` no
   worse than the gated operating point (34 / 160).
3. **Transfers:** the same reads on rows from another run (A's dense
   kanji, the cold 1 900).

**Arms** (micro first, the reads' floors cached):
- **F2a, does it exist:** `v_line` trained **ungated with the rows frozen
  at the seed** (the `b0709` rows it will be summed onto; the only
  trainable is `v_line`). Data: Stage B's kana words, plus lone items of
  the same glyphs (the count tier, and `b0709`'s scene / grid singles), so
  every lone item carries `v_line` and must still render one glyph. Read
  on the held-out 10 kana (the same keys as F1, cached floor): alone and
  words, against F1's gated 0.5.
- **F2b, the data axes:** F2a's mix plus spelled kanji words (§ 2.3) and
  runs with a piece in them (§ 2.4), with lone pieces. Read adds
  kanji_mode's words and the piece ruler.
- Clause variety stays parked (§ 2.7), so `wdup` is read, not targeted.

**Code:** an ungated line mode (the hook adds `line` to every pack row)
and a rows-frozen trainer (every vocab rides as a frozen seed row, which
`rows.Rows` already does for context rows). The pack side needs only the
gate removed (§ 1).

Not a geometry penalty (the roll-up closes those): `v_line` is fit by the
data, and the rows do not move.

### 2.1 In-word doubling (B) — the main open cost

B rides the line mode. It jumps above the floor at the first dose
(`u_S` α 0.5: `dup` 49) and does not fall below +18 at any dose tried
(`u_S` α 1: 44; `v_line` 0.5: 54; `v_line` 1: 77). Composition and B have
not separated on any dose axis. It is word-final and bound to the
`Japanese text reads as` clause every recipe trains on (§ 0). Read B with
`wdup`, not `dup`.

The one lever left outside caption / tokenization work (parked, § 2.7):
- **A finer `v_line` dose** (0.35 / 0.75 around 0.5, training-free,
  ≈ 25 min each) to find the `wdup` minimum.

Not a training-time cap on a row's projection: it sits in the closed
geometry-penalty family and would act on `r_i`, while B lives in the mode.

### 2.2 Trained rows alone still render a line

(Under § 2.0 the rows never see line data, so this concerns rows trained
on words: F1's, and the piece table's sentence / grid-string tiers.)

F1's rows, gate off, read as a line of other glyphs in 94 / 144 renders
(floor 47), and official is 50 (floor 91). `u_S` left them (energy 24 % →
5 %), but the layout did not. Until this moves, **the "alone" value of a
trained single is its seed row**, and trained rows are line-only.

- **F1b:** alone items at the b0305 px (≈ 18 px). The count tier covers
  24–40 px only, because a single under 24 px has no window in the band
  law. **It needs a band-law row first:** a read of the single's window at
  16–24 px (`band_experiment_results.md`). Then retrain F1 with the tier
  and read the donor singles alone (line ≥ 3 glyphs, official, repeat)
  against F1's 94 · 50 · 35. The count twin lowered its prior: the tier
  moved same-glyph repeats, not the line.

### 2.3 The mode does not compose kanji

`v_line` is kana-trained. Kanji words do not compose under it, cold
(Stage I) or warm (kanji_mode), and retraining the dense glyph rows does
not lift them (plan_2900 § 1, A0: 何時間 先輩 最高 stay at the mode
floor). The fix is data for `v_line`: spelled kanji words in F2b's mix
(§ 2.0), read on kanji_mode's six words (floor keys cached in
`native_kmode/`). A gated kanji-only `line_mode` run is the fallback if
F2a finds no ungated direction.

### 2.4 The mode for pieces

A piece inside a run renders 0 with the gate on or off, and the single
beside it doubles (§ 0). **The kind gate** (`v_line` on single-kind rows
only) is a hook change plus the same ≈ 3 min read, and it also shows
whether B rides on the piece row or on its neighbours. Parked with the
caption work (§ 3). `u_P` (Stage A) is the post-hoc candidate for a piece
mode, and a `v_line` trained on piece runs is the learned one: F2b's
piece axis (§ 2.0).

### 2.5 Shipping the mode

The pack side is built (§ 1) for the gated mode; an ungated `v_line`
(§ 2.0) drops the gate. Until then, the dose and the kind rule wait on
§ 2.1 and § 2.4. Open:
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
passes becomes a gated `v_m`. The count twin was its first run, negative
(§ 0).

| mode | gate | first read | note |
|---|---|---|---|
| horizontal | the existing caption marker | a horizontal ruler on the seed: is there a failure? | no horizontal read on record; skip the mode if there is no failure |
| manga vs other surfaces | a new caption marker | twin: manga-bubble pools vs sign / cloth / UI pools | the data is all manga bubble today |
| SFX | a new caption marker | none possible yet | no SFX data recipe (the renderers are font-based); the reader and corpus exist (Manga109-s + COO) |

Order by the rulers' failure list (doubling, sentence assembly, 3+ glyph
pieces), not by the taxonomy. A mode is worth training only if it explains
one of those failures.

### 2.7 Parked

- **Caption / tokenization levers for B** (user, 2026-09-26: no more time
  on them): the clause split (the held-out words under the `japanese text`
  tag alone and under the `Japanese text reads as` sentence alone), then
  clause-varied `scene_spelled` captions if the split localizes B; and a
  caption marker for the line mode (spelled items captioned `, spelled.`),
  the last resort.
- **The outside opinion's adapter replay / distillation**
  (`opinion_factorizedrows.md` § 2–4). F0 found no context-dependent
  component in the adapter worth localizing, and adapter distances have
  not predicted renders (piece ↔ glyph R² ≈ 0.03).
- **The `u_P` → こんにちは-singles bridge** (no training, cached floor):
  low priority.

## 3. Order

1. **§ 2.0 F2a**: the ungated, rows-frozen `v_line` on kana (code first).
2. **§ 2.0 F2b**: + kanji words (§ 2.3) + piece runs (§ 2.4).
3. The gated fallback: **§ 2.1** the dose sweep, **§ 2.5** the sent test,
   then the post-train comparison.
4. **§ 2.2 F1b**, once its band-law row exists.

Parked: § 2.4's kind gate and § 2.7's caption levers.

## 4. What closes it

- **F2a finds no ungated direction** (every `v_line` that composes also
  moves the lone glyph off the floor) → the gate is part of the design,
  and the main line becomes the gated `v_line`'s data (kanji, pieces)
  with the kind rule.
- **F2b holds all three acceptance reads** → the gate and kind rule come
  out of the pack; § 2.1's dose and § 2.4's kind gate close.

- **B survives the dose minimum** → doubling inside a word is part of the
  line mode in this parameterization. The fix moves to a caption marker,
  or B is accepted as a render-time filter.
- **F1b leaves the rows rendering a line alone** → the factorization does
  not clean `r_i` by gradient. The seed row stays the "alone" value, and
  trained rows are line-only (already the working assumption).
- **No `v_line` (ungated or gated) composes kanji words** → kanji ship
  without the mode (identity only), and the mode stays kana-only.
