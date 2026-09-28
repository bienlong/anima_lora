# plan_polish.md — the 4 k-token polish on a self-generated canvas, pilot only

`future.md` § 3 (token scaling on a self-generated set) cut to its 4 k-token
step, as a pilot on the kana rows. No production run is planned from it; the
pilot decides whether § 3 is worth a plan.

## Why a self-generated canvas

`experiments/real_kana` (`future.md` § 1): real pages sit outside what the
frozen DiT can reach, their FM loss is high everywhere, and the trainable rows
absorb all of it — identity lost. The loss outside the text has to be low, so
the canvas must be the DiT's own render.

The text must not be the DiT's own. On a render of the rows' own kana the
in-box loss is the model's too (FM gradient ≈ 0 in expectation, the signal
only a reader filter). The line's 1 k data already has both properties: the
base draws a scene with an EN anchor in its bubble (every token pretrained, no
ext row touched), and the data stage erases the anchor and pastes the kana
word. The DiT made everything outside the box, and the font made the text
inside it. This pilot builds the same composite at the 1024 tier.

A first cut (2026-09-28, stopped at 28 renders, `OUT/polish_yield/img/`)
rendered `retrain_kana`'s kana directly and kept the reader-exact ones. It
is superseded by the composite for the reason above.

## Pilot

Rows: `retrain_kana` (174 kana, routed). Resolution: the 1024 tier
(3 840–4 480 tokens: 1024², 896×1152, 768×1280, 896×1280 and their
transposes), where users render.

1. **Scene pool (`experiments/polish`, `--label t4k` → `OUT/scenes_t4k/`).**
   The `scenes` stage as the 1 k pools ran it (one closed bubble, the
   anchor read back, the erase clears it, specks erased or rejected), with
   three changes:
   - The frame is `<tags incl. speech bubble>. Text reads as "{a}".`, with
     no `english text` tag. The composite caption keeps the frame and swaps
     only the quote.
   - The anchors are short EN sentences or several words (40, e.g. `wait for
     me`, `thank you so much`), not one word, so the bubble is sized for a
     line.
   - The canvas is rendered at its own size (min box: below).

   200 prompts; the 1 k pools kept ≈ 23 %. **Ran 2026-09-28: 32 / 200
   under the stage's judge, 69 / 200 under the loosened one** (a Latin read
   within ⅓ of the anchor counts, since the anchor is erased; min box 72;
   open bubble at seam ≥ 0.85, lost ≤ 0.12; the counts per setting are in
   the script's docstring), **51 / 200** once white-background line art is
   dropped (near-white ≥ 0.8, saturation < 0.05: 18 of the 69). What is left is mostly the base drawing
   pseudo-Japanese without the `english text` tag (multi_box 57, read_miss 39).
2. **Data.** Kana words (`retrain_kana`'s windows, its read words held out
   by trigram) composited into the `t4k` pool at the pool's own size. Open:
   - which recipes (`scene_window` / `scene_single`) and how many items;
   - the σ band. The glyphs render at ≈ 2× the 1 k px, and the band law's
     per-px table (`band_experiment_results.md` § 2) stops at 128 px.
3. **Train.** Batch 1, warm from `retrain_kana` with a light anchor (it is a
   polish), the rows only.
4. **Read**, paired against `retrain_kana`, at 1 k (`retrain_read`'s grid)
   and at 4 k. This answers `future.md` § 3's first question: do the rows
   read at 4 k as at 1 k (words official 16 / 104 at 1 k)?

Step 1 runs alone first; step 2 is sized on its keep count.
