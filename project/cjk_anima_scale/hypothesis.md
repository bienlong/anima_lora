# hypothesis — why a JA row does not compose (2026-09-27)

F2a / F2a′ (proposal.md § 2.0) found no ungated `v_line`: lone items at the
words' own bands and px did not keep the lone glyph (donors alone: official
91 → 59, line 47 → 97 / 144). The question behind it: EN composes words from
T5 fragments with no mode vector at all, so what does EN have that a JA row
lacks? Two adapter-only probes (`experiments/ctx_trigger/`, TE + `llm_adapter`,
no DiT, no renders) point at the rows themselves, and at their norm. Everything
below is adapter-level; the roll-up rule stands (judge composition by render,
never by row geometry), so § 4 starts with a render read.

## 1. What the probes measured

The read: the cos between one T5 id's adapter output inside a word and the
same id alone, same prompt and clause (8 native prompts × `en` / `swap`). Low
cos = the adapter rewrites the piece from its neighbours.

**c1** (`results/20260927-2058-c1/`, job `20260927-205811-2f2f96`), seed rows:

| condition | T5 text / Qwen text | out cos |
|---|---|---|
| `en_word` | `"hello"` / same | **0.51** |
| `en_spaced` | `"h e l l o"` / same | 0.44 |
| `en_qwen_sp` | `"hello"` / `"h e l l o"` | 0.53 |
| `ja_spaced` | `"ひ ま わ り"` / same | **0.965** |
| `ja_hybrid` | `"ひ ま わ り"` / `"ひまわり"` | 0.965 |
| `ja_word` | `"ひまわり"` / same | 0.973 |

- Inside quotes EN is always fragments (`"hello"` → `▁" | h | ello`,
  `"wonderful"` → `w | on | der | ful`), and the same id stands alone. The
  adapter reads a context-dependent row when the row lets it.
- **Qwen is not the trigger.** Removing Qwen's word context from EN moves it
  0.51 → 0.53; giving it to JA moves nothing (0.965 → 0.965). The context is
  T5-side: EN starts falling at block 1 (0.91 → 0.78 → 0.64), JA stays ≥ 0.97
  through block 6.

**c2** (`results/20260927-2106-c2/`, job `20260927-210648-b0aa99`): the JA
conditions under five row arms (effective row = pack row + delta, every seed
row), and EN over 60 sampled Qwen word tokens (158 pieces):

| JA row arm | mean row norm | `ja_spaced` | `ja_word` |
|---|---|---|---|
| `seed` (the seed rows) | 252 | 0.965 | 0.973 |
| `raw_at_seed` (pack direction, seed norm) | 252 | 0.939 | 0.936 |
| `seed_n200` (seed direction, norm 200) | 200 | 0.824 | 0.854 |
| `seed_at_pack` (seed direction, the pack row's norm) | 206 | 0.780 | 0.807 |
| **`raw`** (the pack rows, no identity training) | 206 | **0.657** | **0.617** |
| (EN, 158 pieces) | T5 table mean 212 | 0.56 | |

- **Norm is most of it.** The pack direction scaled up to the seed's norm
  goes 0.66 → 0.94; the seed direction scaled down to the pack's norm goes
  0.97 → 0.78. Direction is the rest (≈ 0.1–0.15 at equal norm).
- The glyphs read here sit at effective norm 270–320 (ひ 322, ま 271, わ 282,
  り 296, さ 322) on pack rows of 155–208: identity training put a Δ of
  240–290 on top. EN pieces are 160–225 (T5 table mean 212, pack mean 204).
- **Ambiguity takes context in EN.** Pieces by share (how many of Qwen's
  68 923 Latin tokens hold the piece in their quoted T5 spelling), terciles:
  share 1–20 → 0.74, 20–89 → 0.67, 102–10 159 → 0.46.
- **In EN, share and norm are one axis** (post-hoc, 2026-09-27: c2's 114
  `en_word` pieces, each piece's row norm from the base DiT's
  `net.llm_adapter.embed.weight`, table mean 212; not a script):

  | | r |
  |---|---|
  | log share ↔ norm | **−0.87** (a common piece has a small row) |
  | log share ↔ cos | −0.74 |
  | norm ↔ cos | +0.68 |
  | share, norm held (partial) | **−0.42** |
  | norm, share held (partial) | 0.10 |

  By norm tercile: 139–196 → 0.47, 197–222 → 0.68, 222–265 → 0.72. EN's
  natural variation is organised by share. Norm follows it and adds little
  beyond it, and EN's loudest pieces (≈ the seed's 252) still read context at
  0.72, not 0.965. The JA arms respond far more steeply to the same norm range
  (`raw` 206 → `raw_at_seed` 252: 0.66 → 0.94). So c2's norm effect is shown
  only by rescaling JA rows; EN's own context read is not shown to be a norm
  effect.

**Vocab granularity** (tokenizers, CPU):

| | T5 | Qwen | Qwen / T5 |
|---|---|---|---|
| Latin word-start | 20 148 | 41 547 | 2.1 |
| Latin continuation | 8 290 | 27 376 | 3.3 |
| Latin total | 28 438 | 68 923 | **2.4** |
| JA tokens | 0 | 27 022 (8 734 one-glyph, 18 288 multi) | pack: 1 row per Qwen token → **1.0** |

The pack mirrors Qwen: every JA Qwen token is one ext row, and 18 288 of them
are piece rows holding a whole multi-glyph string (+ 28 017 char rows for the
byte-fragment path, + 6 118 symbol rows; 69 558 in all). EN's T5 side is 2.4×
coarser than Qwen, so a T5 piece is shared by many words and cannot be read
without its neighbours.

## 2. Hypotheses

- **H1, loud rows.** Identity training grows a row's effective norm to
  1.3–1.9× the T5 table's. At that norm the adapter's residual updates (self-
  attn and MLP over the neighbours) barely move the row, so the DiT receives
  the same embedding alone and in a word. F0's "the adapter is context-blind
  to a row" is this: the rows it measured were trained rows. Composition then
  has to be injected (`u_S`, `v_line`), and an injected vector reaches a lone
  glyph unchanged (F2a).
- **H2, a row that is the only address of its string needs no context.** EN
  pieces are ambiguous (share) and the adapter reads them in context; a JA
  piece row is a whole string and a single row is trained to render alone.
  Nothing in the table asks the adapter to read a JA row by its neighbours.
- **Not H0 (Qwen).** c1 rules out the Qwen word context as the lever.

H1 is the measured one on JA rows (c2's arms). H2 is EN-side evidence plus
the design of the pack, not a JA read, and in EN the two cannot be told
apart: share and norm correlate at −0.87, and share keeps a partial effect
(−0.42) that norm does not (0.10) (§ 1). Two readings remain:
- **H1 generic:** a pre-norm residual (`LLMAdapterTransformerBlock`:
  `x + f(norm(x))`, so an update's size does not grow with ‖x‖) makes any
  row's context read fall with its norm. EN's weak norm slope is then because
  EN rows vary little in norm for a given share. A norm cap is enough.
- **H2 learned:** the adapter learned how large an update to give by token
  (update size tracks the row's direction; common pieces get large ones).
  JA rows, pack or seed, never took part in that learning, so a cap brings
  a JA row down to EN's norm but not to EN's context read. A cap is
  necessary, not sufficient.

P0b (§ 4) separates them at the adapter; P1's 2 × 2 at training.

### Against the record

- **Roll-up, "trained rows are context-free at the adapter"**: H1 is the
  mechanism for it, with a control the old read lacked (the raw rows are not
  context-free).
- **Roll-up, "row norm is the wipe ↔ hit lever"** (`row_blocks_alpha`
  2026-09-18): native hits rose 17 → 38 / 64 as the delta was scaled × 1.0 →
  × 0.2, while `en cos` fell 0.921 → 0.875. That read scaled the delta, not
  the whole row, on an older table, and scored single hits, not words. It is
  the one render read on record that moves along this axis, and it moved the
  right way.
- **Roll-up, "judge composition by render, never by row geometry"** (spell
  2026-09-26: piece ↔ glyph R² ≈ 0.03). An adapter cos is not a render. Every
  proposal below is gated on § 4 P0.
- **Closed: row-space geometry penalties (decorrelation / orthogonality /
  whitening), shape encoders, IDS splits.** A norm bound is none of these
  (user, 2026-09-27: the ban does not cover a norm clamp), so P1 is open. If
  P0 points there, a seed retrain under the bound is acceptable.

## 3. The user's lever: fewer piece rows

CJK in Anima is rendered, never read for meaning, so the T5 side is free to
choose its granularity. The pack today sits at Qwen / T5 = 1.0 (and finer:
piece rows are whole strings); EN sits at 2.4.

- **Per-glyph routing** (every JA Qwen token → its glyphs' single / char
  rows on the T5 side, Qwen untouched): the T5 side becomes ≈ 8.7 k kana /
  kanji rows under 27 k Qwen JA tokens, **Qwen / T5 ≈ 3**, close to EN's 2.4.
  Every row is then shared by every word it appears in (the high-share
  regime, 0.46 in EN). It is an encoder change (`HybridT5Encoder` already
  regroups byte fragments per char; this makes it the only path), the 18 k
  piece rows become unused, and the spelled path is the only path.
- **Coarser than per-glyph is not on the table.** c1 says Qwen barely
  changes the adapter output, so a row shared by two glyphs could not be
  disambiguated, and sub-glyph splits are closed (IDS / shape encoder).
- **What it depends on.** Per-glyph routing only composes if the adapter
  reads the single rows in context, which H1 says the trained rows prevent.
  The spelled path today is exactly per-glyph routing, and it sits at the
  floor (held-out words ≤ 1 edit 11 / 160). So P2 follows P0 / P1, not the
  other way round.
- **What it costs.** The piece rows are what the piece line trained (300f,
  `300f_sp`), and they are the address every unspaced caption uses today;
  per-glyph routing leaves them unused. On the one read that compared the
  two, the trained singles spelled a held-out 5-glyph word within 2 edits
  13 / 32 against the piece row's 4 (spell 2026-09-26), so the piece row is
  not the stronger address by default.

## 4. Proposals, in order

- **P0, render the norm arms (training-free, the cached floor)** —
  `experiments/norm_p0/` (launched 2026-09-27, job
  `20260927-213257-5b19e6`). Every seed row's effective row × 0.8 and × 0.65
  (the held-out rows 271–322 → 217–257 / 176–209), and `raw` as the lower
  anchor, on the held-out 10 (en + swap), no `v_line`: does words ≤ 1 edit
  rise from the floor's 11 / 160 while singles official stays near 149 / 320?
  About 25 min per arm.
  - Words up, singles held → H1 has render support; P1 next.
  - Words flat, singles fall → the norm buys identity, not a context read
    the DiT uses; H1 stays adapter-only and the gate stands.

  **Read (`results/20260927-2133-p0/`; `raw` skipped): the second branch.**

  | held-out 10 | floor × 1 | × 0.8 | × 0.65 |
  |---|---|---|---|
  | singles official / 320 | 149 | 89 (p 2e-11) | **20** |
  | singles contained | 280 | 206 | 87 |
  | singles alone as a line | 78 | 94 | 143 |
  | words ≤ 1 edit / 160 | 11 | 16 (p 0.42) | 1 |
  | words ≤ 2 edits | 93 | 63 (p 8e-4) | 5 |
  | words `dup` | 26 | 60 | 65 |

  A trained row scaled down loses its glyph and drifts toward generic text
  (Latin gibberish in the renders), and it gains no composition. In-word
  doubling rises. Identity is carried partly by the norm of a trained row.
- **P0b, EN rows rescaled (adapter only, minutes).** c2's `en_word` with the
  EN pieces' T5 rows × 0.8 / × 1.2 (the probe c2 ran on JA, run on EN):
  - EN cos moves as steeply as JA's → H1 generic; the cap is the lever.
  - EN barely moves → H2 learned; the cap is necessary but will not bring a
    JA row to EN's context read, and P1's in-word column carries the test.

  **Read (`ctx_trigger --probe c3`, `results/20260927-2226-c3/`, job
  `20260927-222625-25e457`): both, and direction is the larger part.** Out
  cos (en clause), the rows' mean norm:

  | α | EN pieces | JA held-out glyphs (≈ 300 × α) |
  |---|---|---|
  | 0.65 | 135 → 0.49 | ≈ 195 → 0.81 |
  | 0.8 | 166 → 0.50 | ≈ 240 → 0.91 |
  | 1 | 207 → 0.55 | ≈ 300 → 0.965 |
  | 1.2 | 249 → 0.65 | ≈ 360 → 0.98 |
  | 1.5 | 311 → 0.82 | ≈ 450 → 0.99 |

  - **The norm lever is generic, upward.** EN pieces made loud go
    context-immune too (0.55 → 0.82 at × 1.5), every share tercile alike.
    Below × 1, EN flattens at ≈ 0.5: a quieter row does not read more
    context.
  - **At matched norm the seed rows sit 0.25–0.3 above EN** (≈ 240: JA 0.91
    vs EN 0.65; ≈ 200: 0.81 vs 0.55). c2's pack rows at their own norm (206
    → 0.66) sit on EN's curve, so the offset came with identity training,
    and it is the rows' direction: c2's 0.1–0.15 "direction" understated it.
  - With P0: taking a trained row down to EN's norm leaves it context-free
    (0.81) and loses its glyph. **A norm cap alone is unlikely to be the
    lever.** A capped retrain has to find identity in a direction the adapter
    reads in context, and nothing in the lone `b0709` data asks for one.
    P1's in-word column is the only arm that could, and Stage B's uncapped
    in-word donors read 0.95–0.97.
- **P1, norm-bounded identity training.** The effective row (pack + delta)
  projected back to a cap after every optimizer step (a hard clamp, no
  penalty weight; the cap value is P0's winning arm), everything else as
  trained. Micro first, on a set whose floor is cached (stage_i's or A0's 12),
  at × 1 and, if identity falls short, × 3. As a 2 × 2:

  | | lone (`b0709`) | in-word, shared (Stage B's data) |
  |---|---|---|
  | uncapped | the reads on record | Stage B on record (F0: 0.95–0.97) |
  | capped | H1: identity under the cap? | **H2 at training: does a shared, capped row read context?** |

  Plus a warm arm: the seed projected to the cap, then capped `b0709`
  fine-tuning. If it holds identity, the cap does not need a cold 2 274-row
  seed retrain (≈ 200 k steps, ≈ 25 h local / ≈ 8 h G4). Singles only: a
  piece row is the only address of its string (H2), and P2 leaves it unused.
  No word items or `v_line` beside the cap in the lone column (one variable).
  Stage B's `data/` keeps `img/` + `train.jsonl`; its caches were cleared
  2026-09-27 and rebuild at train.
- **P2, per-glyph routing** (§ 3): an encoder flag, read on `native_sent`
  and the held-out words, on the rows from P1. Only if P1 composes.
- **Not proposed:** retraining the adapter (outside the frozen-DiT
  contract), Qwen-side changes (c1).

## 5. What would falsify it

- P0's norm arms compose no better than the seed rows (words ≤ 1 edit within
  noise of 11 / 160): the adapter-level context does not reach the render.
- P1's bounded rows lose identity (singles official well under `b0709`'s)
  without composing: norm is how identity is bought, and H1 is a trade, not a
  lever.
- P0b: EN barely moves under rescaling, and P1's capped in-word rows stay
  context-free: context is learned by token, and no row-side lever reaches it
  under the frozen adapter.
- A per-glyph-routed caption of a trained word renders worse than the piece
  row (P2): granularity is not the constraint.

Scripts: `experiments/ctx_trigger/run_exp.py` (`--probe c1 | c2`),
`experiments/norm_p0/run_exp.py` (P0). F2a /
F2a′: `experiments/f2a_line/` (`20260927-1644-a1`, `20260927-1950-a2`).
