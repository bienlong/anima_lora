# CJK vocab ext rows — the training recipe

How the Japanese rows of the [CJK vocab pack](../methods/cjk_vocab_pack.md) are
trained so that the base model **renders** them as glyphs: which rows a
caption reaches (Qwen : T5 granularity and per-glyph routing), where the
images come from (a self-generated canvas with pasted text), which σ each
item trains at (a band law and a per-band data mix), and how the loss
weighs the text (in-box share). Status 2026-09-28: the kana rows
(`retrain_kana`) and the first kanji batch (`retrain_kanji_b1`) are trained;
two kanji batches and the bake of the new seed are left.

The line's code and records live in `project/cjk_anima_scale/`. Its
`README.md` holds the current state, `proposal_length.md` the open question,
`retrain_experiments.md` the reads behind this page, and
`band_experiment_results.md` the band law. Run it through that project's
`scale.py <run> data | train | eval`. There is no `make` target.

## What trains

Only the pack's ext rows train. The DiT, the LLM adapter and the T5 table stay
frozen. A run stores a delta per row (`raw · row_scale`), summed onto the
pack row at lookup, and `scripts/toolkits/bake_vocab_pack.py` folds it into a
new pack pair. A trained row ends near norm 250, against the T5 table's mean
212.

A run's vocabs start **cold**: Δ = 0 on the raw pack row. Every other row a
caption touches rides frozen at the context rows (the seed, or the previous
run of a chain). Cold is deliberate. On the same in-word data, rows warm
from the old seed composed less than cold rows (こんにちは ≤ 1 edit 6 vs
11 / 16) and doubled more (`dup` 13 vs 4). The old seed carries a direction
the adapter treats as context-free, and training on top of it keeps that
direction (`retrain_experiments.md` § 1).

## Addressing: Qwen : T5 granularity and per-glyph routing

Anima's text path sends each caption through Qwen and through a T5-side query
table. The table's granularity decides whether a row is shared by many words
or is a whole string of its own.

| | T5 side | Qwen | Qwen / T5 |
|---|---|---|---|
| Latin | 28 438 pieces | 68 923 | **2.4** |
| JA, pack as built | one ext row per Qwen token (8 734 one-glyph, 18 288 multi-glyph "piece" rows) | 27 022 | **1.0** |
| JA, per-glyph routing | ≈ 1 500 single rows in use (up to ≈ 8.7 k) | 27 022 | **≈ 3** |

EN sits at 2.4. A T5 piece is shared by many words, and the adapter reads it
in context. A JA piece row is the only address of its string, and nothing
ever asked the adapter to read it by its neighbours.

**Per-glyph routing** is an encoder flag (`HybridT5Encoder`,
`library/anima/ext_vocab.py`). On the T5 side, every JA Qwen token goes to its
glyphs' single rows. The Qwen text is untouched, and so is EN, which stays
bit-exact. Each single row is then shared by every word it appears in, as EN
pieces are. The piece rows go unused.

On こんにちは (16 renders), the routed caption beat both the spelled one and
the piece row:

| | ≤ 1 edit | official |
|---|---|---|
| routed | 14 | 8 |
| spelled | 11 | 5 |
| piece row | 0 | 0 |

Singles reach 99.85 % of JA glyph occurrences in the dialogue corpus with
≈ 1 500 rows. Pieces covered ≈ 85 % of lines with 1 900 rows.

The flag lives in the pack json (`"glyph_route": true`).
`ANIMA_VOCAB_GLYPH_ROUTE=1/0` overrides it. A data dir built with routed
windows records `glyph_route` in `build.json`. It is then trained and read
routed, against a routed floor cache.

## Data: a self-generated canvas with pasted text

**Real images do not train rows.** The kana rows were warm-trained on 291 real
kanji-free pages with their verbatim captions and the loss box on the quoted
lines (`experiments/real_kana`). Every read fell to ≈ 0: words official
16 → 0 / 104, singles contained 73 → 14 / 112. The frozen DiT cannot
reproduce a real page, so its FM loss is high everywhere. The rows are the
only trainable place for that residual, and they absorb the image's whole
mismatch, not just the text's.

So the canvas is the base model's own render, and only the text is foreign.
The `scenes` stage (`project/cjk_anima_scale/src/scenes/stage.py`) works like
this:

1. **Generate.** The base model draws a scene from a combinatorial tag
   prompt plus an EN anchor clause:
   `<tags incl. speech bubble>, english text. English text reads as "hi".`
   Every token is pretrained, and no ext row is touched.
2. **Judge.** A detector and a reader keep an image only if **exactly one**
   text box is found and the anchor is read back. The box must reach a
   minimum size, and the erase must clear it. Stray specks are erased or
   the image is rejected. The 1 k pools kept 13–24 % of renders.
3. **Composite.** The data stage erases the anchor's box, draws the JA text
   into the bubble with a font, and swaps only the quote in the caption. For
   example, `English text reads as "hi"` becomes
   `Japanese text reads as "…"`, with the `english text` tag becoming
   `japanese text`.

Everything outside the box is the DiT's own output, so the loss there is
low. What is left for the rows is the glyph.

The 1 k-token pools are 384–640 px shapes: `s1` and `s1w` (one-word anchors,
four caption frames), `sl1w` (EN sentences, the only pool for left-to-right
items), and `ja_comic` (one bubble). Lone glyphs go to `s1` / `s1w` scenes
whose bubble aspect is at most 2.

`scenes_t4k` is a 1024-tier pilot pool (3 840–4 480 tokens, `plan_polish.md`).
It uses the frame `Text reads as "<short EN sentence>".` with no
`english text` tag, and keeps 51 of 200 renders.

Orientation is drawn, not fitted. 30 % of multi-glyph items are
left-to-right lines, marked in the caption (`horizontal Japanese text reads
as` / `, written horizontally.`). The rest are columns, the manga default.

## σ per item: the band law and the per-band mix

Every item is stamped with a σ band. The trainer draws σ inside that band
(`fm_training_batch` with the band's `t_min` / `t_max`; a batch is split by
band). The band comes from a **band law** keyed on the item's kind and px,
where px = √(ink box area / glyphs). The law is written as rows with
provenance in `cjk_scale/windows.py`, each row naming the read that set it.

| kind | px | band | read |
|---|---|---|---|
| single (one token, one glyph) | ≥ 40 | 0.7–0.9 | trained: 24 kana at bubble fit, native 84 / 55 vs 49 / 30 of 192 at 0.5–0.7 |
| single | 24–40 | 0.5–0.7 | ceiling only |
| single | 12–24 | 0.3–0.5 | ceiling only |
| piece / multi (≥ 2 glyphs) | 24–64 | 0.5–0.7 | trained: 16 pieces at 35 px, exact 7 vs 1 / 32 against 0.7–0.9 |
| piece / multi | 12–24 | 0.3–0.5 | ceiling only |

The band law's other findings:

- **Glyph count sets the band.** Single-glyph rows train higher than
  multi-glyph rows.
- **Px sets the floor.** Px decides how low a band may reach: 12–16 px text
  lives at 0.2–0.6, 128 px at 0.8.
- **Nothing above 0.9.** 0.8–0.95 is dead at 48 px, for kana and kanji alike.
- **What moves no band:** ink, stroke density and the bubble ellipse. Kanji
  take the kana band.

**The singles' mix** (`builder.TABLE`) has three band groups. Each group
takes half of the kind's item budget; the tiers split the group by weight.
An item is kept only if the group's band sits inside its own window (or
covers 0.8 of it), so px and band stay consistent.

| group | band | tier | weight | what |
|---|---|---|---|---|
| lone | 0.7–0.9 | `scene_single` | 0.5 | one glyph in a bubble, fill 0.7 (≈ 50 px) |
| | | `grid_single` | 0.5 | 1×1 … 3×3 grids, one glyph per cell, half in bubbles; captions name each cell |
| in-word | 0.5–0.7 | `scene_window` | 0.7 | a 2–6-glyph window of a dialogue line in a bubble (≈ 40 px), routed caption |
| | | `scene_single_small` | 0.3 | one glyph at line px (28–40) in a bubble it fills 0.2–0.4 of: the count tier |
| small in-word | 0.3–0.5 | `scene_window` | 1.0 | windows at 12–24 px in small bubbles (fill ≥ 0.5) |

**Windows** are any substring of a manga dialogue line whose glyphs are all
the run's singles, with no glyph repeated and none crossing a held-out read
word's trigram. Whole lines cover kana only: the kanji median is 3 lines per
glyph, and 101 kanji have none. Windows of 2–6 glyphs give a kanji median of
162 and reach every kanji the corpus holds. A window may cross a word
boundary; the DiT renders JA and never reads it for meaning. Each draw picks
a glyph uniformly, then one of its windows, so exposure is per row.

Every window must encode to exactly its glyphs' single rows, or it is
dropped.

Two reads settled the mix:

- **Lone alone composes nothing** (C1). Lone and in-word are 1 : 2.
- **The in-word tier is load-bearing.** Kana rows trained this way hold their
  singles and compose.

**Budget.** The base is 90 steps per vocab, scaled by kind, glyph count,
cold/warm and ink (`cjk_scale/budget.py`), times the mix factor 1.5 for the
in-word share:

| rows | steps / row |
|---|---|
| cold kana | 135 |
| cold kanji, ink < 10 | 225 |
| cold kanji, ink ≥ 10 | 337 |

Dense kanji at 225 lost half their reads. 450 brought them back, and 337 is
the midpoint. Items scale with steps, at ≈ 67 per vocab-factor.

## In-box loss

A glyph is a few percent of the canvas. Under plain MSE, a 24–32 px grid cell
is 0.3–0.8 % of the loss. `cjk_scale/loss.py` gives each item an **in-box
share** `s` and averages the in-box and out-of-box cells separately:

```
loss_item = s · mean(se over box cells) + (1 − s) · mean(se over the rest)
s(n) = s1 + (s_cap − s1) · min(1, ln n / ln n_cap)
     s1 = 0.25 (one glyph), s_cap = 0.5, n_cap = 8 glyphs
```

The share rises with the glyph count on a log curve and is paid in glyphs, not
tokens. Under the probe's linear rule, a 4-glyph piece already hit a 0.75
cap. The pixel box maps to latent cells at 8× (`box_mask`).

A grid item takes the **union of its cells** as one box (`grid_box`, on),
with the share from the joined glyph count. Each row learns from its own
cell because the caption's position clause binds it there. The loss does
not pair a row with its cell. Without the union, a grid cell was priced
15–50× below a scene box per draw (`reports/grid_box_2026_09_25.md`).

A flat 1×1 and any item without a box stay plain MSE.

## Trainer

| setting | value |
|---|---|
| loss | plain flow matching on the frozen DiT, with the in-box share above |
| lr | 1e-3, cosine, warmup 0.1 |
| batch | 4 |
| anchor μ | 0 (the rest is frozen, not anchored) |
| free residual | 1e-3 · ‖f‖² on the touched rows |
| σ | per item, in its band |

The trainer is fixed: each constant in `cjk_scale/train.py` names the read
that set it, and a change is a code change with a report beside it.

The kana runs at 1 k tokens: 12 shapes, 384–640 px on a side. The local GPU does
≈ 2.2 it/s at batch 4, and a Colab G4 ≈ 6.7.

## Results so far

`retrain_kana` trained 174 cold kana and punctuation rows, 23 490 steps on
17 400 items (183 min). It was read routed, 4 prompts × 2 seeds per key:

| words (`en`, / 8 per word) | ≤ 1 edit | official | `dup` |
|---|---|---|---|
| floor (old seed), 8 words | 1 / 64 | 0 | 4 |
| `retrain_kana`, same 8 | **29** / 64 | 10 | 42 |
| `retrain_kana`, 4 katakana words | 20 / 32 | 6 | 15 |

- **Composition.** Against the floor, ≤ 1 edit rose 28 / 0 (p 7e-9).
- **Singles** hold at the floor: official 17 vs 23 / 64 hiragana, not
  significant; contained flat.
- **Doubling rose** (`dup` 42, こんんにちちぱ-style). Collapsing doubles lifts
  ≤ 1 edit 29 → 48 / 64, and why doubling rose is unread.

`retrain_kanji_b1` trained 329 kanji, 91 of them new, cold on
`retrain_kana`'s rows: 90 749 steps on a Colab G4 (225 min). Its singles
were read only on the kanji the seed floor already had cached: 21 kanji,
2 prompts × 2 seeds, paired (`experiments/kanji_read`).

| kanji | floor contained | `b1` contained | floor official | `b1` official |
|---|---|---|---|---|
| 12 with no seed row | 0 / 48 | **29** (p 4e-9) | 0 | 15 |
| 9 dense seed kanji | 20 / 36 | 17 (n.s.) | 10 | 6 (n.s.) |

`repeat` (the glyph drawn more than once) went 0 → 7.

It and the kana are baked, routing on, as
`models/vocab_packs/anima_cjk_vocab_pack_retrained0928_kanj1b`. The same
table is on the Hub as `anima_cjk_vocab_pack_preview3`. Its trained glyphs
are listed in `anima_cjk_vocab_pack_preview3_trained.json`. In ComfyUI it
needs ComfyUI-Anima_lora-Adapter ≥ 3.12.0; older nodes ignore the routing
flag.

**A LoRA trained through the pack.** Two LoRAs were trained on one artist's
30 images (`@channel (caststation)`), 12 of them with Japanese text
clauses. The recipe was `make lora` defaults (8 epochs, 240 steps), and the
two differ only in the text path. One ran through preview3 with routing
on; the other ran with no pack. Compared by eye, the pack-trained LoRA
shows **no visible degradation** against the stock one. Nothing was
scored. Configs and TE caches: `output/vocab_cmp/`.

## Open

- **Doubling.** Candidate causes: window length 2–6 vs whole lines, the row
  count, window variety, the shared direction in row space.
- **Kanji batches b2, b3**, then the new seed: `paths.SEED_ROWS` moves, the
  floor is re-rendered once, and the bake ships routing on.
- **Token scaling.** Users render at the 1024 tier (≈ 4 k tokens), but the
  rows train at 1 k. A pilot (`plan_polish.md`,
  `experiments/polish_rows`) warm-trains the kana rows on `scenes_t4k`
  composites. It runs at batch 1, native size, σ over the full [0, 1], with
  the kana px set by the erased EN anchor's px.
- **An OCR reward on the rows** (`future.md` § 2). It is an option because
  real pages cannot supervise a row through the FM loss.
