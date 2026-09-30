# plan_garble_replace — the base's own garble, only the string replaced (proposal, 2026-09-30)

Train rows on the base's own text layout. The base renders a scene where
someone speaks Japanese and draws pseudo-Japanese in one or two places; that
garble is erased and a real line is drawn in its place, at the base's glyph
size, column count and orientation. The caption adds `Japanese text reads as
"…"`. What `idea.md` § 2 proposed for data; its band half is closed
(`reports/sigma_split_2026_09_30.md`).

## Why

- **The rows carry the paste, the base does not** (`reports/sigma_split_2026_09_30.md`).
  - With the pack rows above σ 0.5 instead of the seed's, the banner shrinks
    by a third and flat white halves. The text also lands where the base's EN
    word lands (IoU 0.20 → 0.45).
  - The seed's rows commit a large string at σ ≥ 0.9, before the scene forms.
  - Asked for Japanese without a quote, the base draws bubbles with vertical
    lines on its own (also `reports/polish_seed_2026_09_30.md` § The 4 k
    letterbox). With that caption above σ 0.8 and the seed's JA caption
    below, the seed's rows already fill those bubbles with their string
    (16 / 29 official, 61 / 92 ≤ 1 edit).
- **Layout has to leave the row through the data, not the band.**
  - Text placement is decided at σ ≥ 0.9 and the string between 0.9 and 0.7.
    Below 0.5 nothing in the caption moves the text.
  - If the target is the base's own canvas with only the glyphs changed, the
    FM residual outside the glyphs is ≈ 0 at every σ. The row then gets a
    glyph-identity gradient and no layout gradient, even trained where layout
    is decided.
- **The FM signal stays low outside the text** (`future.md` § 1), and every
  string is a real line (the line's rule).

## 1. The canvas pool `scenes_garble`

- **Tooling.** The `scenes` stage as the other pools run it: same prompt
  vocabulary, the 512-tier shapes of `s1w` / `ja_comic`, `--scene_prune 1`.
- **Two new frames** in `src/scenes/stage.py` `FRAMES`. Neither has a quote or
  an anchor.
  - `ja_garble_saying`: generals `{bubble}`, `japanese text`; clause
    `{pro} is saying something.` (solo counts only, as with every pronoun
    frame).
  - `ja_garble_tag`: generals `{bubble}`, `japanese text`; no clause.
- **The judge** follows the JA-frame rule: every detector box is text to be
  replaced, and there is no read match. Beyond that:
  - after merging, **one or two text regions**, else `multi_box`;
  - each region passes the existing bubble / open-region checks
    (closed bubble, or a uniform open fill; leaks and specks as now).
- **Size.** A first 200 prompts, grown by `--scene_n` on the keep rate (pools
  are grown, not rebuilt).

## 2. The items: `garble_replace`

Per kept region:

- **Orientation**: vertical when the box is taller than wide, else horizontal.
- **Glyph px and line count** come from the garble's ink: runs in the
  projection profile across the box (columns when vertical, lines when
  horizontal). Glyphs per line = box length / px.
- **String**: a real dialogue line from `polish_b1`'s `line_pool`
  (manga109s dialogue, normalized, routed to its glyphs' rows, read words held
  out by trigram, no glyph doubled in a row). Its length must fit
  lines × glyphs per line. Two regions take two lines.
- **Render**: `render_into_scene`'s erase (ring-median, inside the bubble
  interior) and draw, with **px fixed to the measured px** (no bubble fit)
  and the measured orientation and line count. Fonts: the render set's
  regular faces.
- **Caption**: the canvas prompt + ` Japanese text reads as "X".`
  (two regions: `… reads as "X". Japanese text reads as "Y".`).
- **σ band: 0.6–0.85.** From `reports/sigma_split_2026_09_30.md`:
  - The text's place is set by σ ≈ 0.9 and its string between 0.9 and 0.7
    (traj).
  - Rows acting only below 0.8 write their string into the base's garble
    layout (16 / 29 official, 61 / 92 ≤ 1 edit, untrained for it).
  - Rows acting only below 0.5 move nothing.

  The band therefore sits where the layout is already the base's and the
  string is still open.

## 3. Step 1 — the data sheet (CPU after the pool)

Build 64 items and one contact sheet per region count, one row per item:
canvas | garble box + measured px / lines | composite | caption. Print the
yield: pool keep rate, regions that take a line, px distribution. Reviewed
before anything trains.

## 4. Step 2 — warm vs cold on the same data

- **Rows**: the 57 singles of the `sent` strings (い う え お か が け こ し す
  せ た だ ち っ つ て と な に は ば や ょ る ん ア イ カ コ ソ テ ド ネ パ ビ メ
  ラ ル レ ン 丈 人 何 大 太 夫 小 山 愛 日 時 本 来 田 郎 間), routed. Every
  other row stays frozen at the seed. Items are drawn so each of the 57
  appears; a line may carry other seed singles as frozen context.
- **Arms**, with the same items, steps and trainer constants
  (`cjk_scale/train.py`: lr 1e-3 cosine, batch 4, 90 steps / row = 5 130
  steps):
  - **warm**: the 57 start from the seed rows. Its anchor μ is open;
    `polish_seed` ran 0.1.
  - **cold**: the 57 start from the pack rows, as the retrain's singles did.
- **Read**, paired per render against the seed's routed floor cache:
  - `sent` on the `retrain_read` grid (184), per string: official, ≤ 1 edit,
    dup;
  - placement (box, box h, flat white, EN-ref IoU), and bubble vs banner on
    the sheets;
  - the `--traj` leg on the trained rows: at which σ the string commits.
- **Context on warm.** Every warm pass so far lost identity
  (`reports/polish_seed_2026_09_30.md` § Verdict). Those passes used
  composites with a fill-rule size. This data takes size and place from the
  base.

## Code

`experiments/garble_replace/run_exp.py`, with legs `pool` (GPU, the scenes
stage with the new frames), `data` (CPU: measure, fit, render, sheet),
`train --arm warm|cold` (GPU) and `read` (GPU; `sigma_split`'s read and traj
code, loaded with `paths.load_experiment`). The new frames and the judge's
region cap are the only edits to `src/`.
