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

**On the new seed (2026-09-30):** `polish_b1`'s table on `seed_retrain_0930`
(1 362 singles, 4 steps / row, μ 0.1) cost identity: sent official 29 → 11 / 184,
and dropping the monochrome scenes did not help (10). Warm polish in this shape
stops. The SFX plan below waits on a cold micro arm (the report's § Verdict and
next). Read: [`reports/polish_seed_2026_09_30.md`](reports/polish_seed_2026_09_30.md).

## SFX and small text: a text-free canvas in the post-b3 polish (planned 2026-09-29)

Not run; it waits for `retrain_kanji_b3` (the polish follows the chain,
`plan_retrain` § 3; `experiments/polish_b1` is that polish's pilot on b1).

**Why.** Every item the rows have seen is bubble-interior lettering: the
base's bubble, the anchor erased with the ring-median fill, a dark upright
glyph (`render_into_scene`: the anchor's ink or black / (30,30,30) /
(60,40,40), no stroke), in the `reads_as / bubble_reads / saying / sign`
frames, singles at ≥ 28 px. So the rows render only that: a white or
lightly outlined bubble, near-black regular glyphs. SFX (ビクッ, ドキドキ)
and small lettering never appear, and `Japanese SFX reads as "…"` (the
clause `anime_tools` writes into real captions) never trained. The fix is
the style mix, not the seed: identity is the chain's job, and SFX is mostly
kana (COO: 70 % katakana, p50 2 glyphs), which `retrain_kana` already holds.

**The canvas: a text-free pool `sfx` (new; user, 2026-09-29).** Not the EN-
anchor pools: an erase over art (not a flat bubble) leaves a ring-median
blot, and a bubble in every SFX item ties SFX to bubbles. The prompts carry
no `english text`, no `speech bubble`, no text frame; they carry scene tags
SFX goes with (`surprised`, `flinch`, `running`, `motion lines`, `heart`,
…), and the negative carries `text, speech bubble, sound effects`. The
judge is one rule: the text detector (`common.readers`) finds no box.
Without the `english text` tag the base draws pseudo-Japanese (t4k:
multi_box 57 / 200), so the rule is load-bearing. No anchor, no erase, no
read-back: the whole canvas is the DiT's. A 512-tier pool for the 1 k
polish; a 1024-tier one only if the 4 k step goes ahead.

**The items: SFX pasted over the art.**
- Strings: real SFX, from the sincos SFX reads (`dedupe_sfx` keys) and the
  COO text frequencies (Manga109 strings never enter the repo), kept when
  every glyph is a trained single (routed, as `scene_line` does), the read
  words held out by trigram.
- Render: the SFX faces of `assets/fonts/` (TanukiMagic, 破線G, 源真ゴシック
  Bold, Corporate Logo); a thick contrasting stroke (white fill / black
  stroke, or a colour fill / white stroke — `flat.py`'s jitter inks, stroke
  2–6 scaled by px); tilt up to ± 20°, per-glyph scale and offset jitter,
  diagonal runs; anywhere on the canvas.
- Two sizes, banded by the law's px table (`band_experiment_results.md`
  § 2): **large** 80–200 px at p0507 (> 128 px is outside the table,
  unmeasured), **small** 16–24 px (ドキ, ぎゅっ beside a character) at
  p0305.
- Caption: the canvas's own tags + `sound effects` + `Japanese SFX reads as
  "ビクッ".` — the real-caption clause, not a new frame.

**The mix.** The SFX tiers take a large share of the polish's items (user,
2026-09-29; the number is open), beside `polish_b1`'s bubble singles,
windows and lines, which stay: the same row has to appear in both styles,
so the caption carries the style and the row only the glyph.

**Read.** `polish_b1`'s reads, plus an SFX read: SFX prompts on text-free
canvases, read by `SfxReader` (`anime_tools.ocr.sfx`), against the
unpolished rows.

**Round 2 (optional).** Render SFX with the polished rows and keep only
the `SfxReader`-exact ones (`future.md` § 3's filter). The text is then the
DiT's own too, so the FM signal is small and the filter carries it (§ Why a
self-generated canvas above); round 1's composites come first.
