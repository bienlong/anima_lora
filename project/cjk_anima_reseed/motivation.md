# cjk_anima_reseed — motivation (2026-10-02)

> **Draft of the morning, overtaken the same day** — read
> `../cjk_anima_scale/reports/band_size_2026_10_02.md` first. § 1 does not
> stand as worded: the seed's items sat inside the ceiling window of their
> px, and what is missing is an item that could train above 0.7, which § 4
> says better. The target is manga-size dialogue in speech bubbles, not the
> `sent` banner (user, 10-02), and identity does not need large glyphs
> (`p1_cold`). § 2 has no read behind it as a cause; § 3 is renderer work the
> dialogue target needs. Not rewritten yet: `grid_small` is still training.

The JA vocab pack's seed (`seed_retrain_0930`, `project/cjk_anima_scale/`) was
trained on data that four later reads call into question. Each is a property of
the seed's training items, measured on the data dirs of record
(`output/cjk_anima_scale/retrain_kana/data`, `…/retrain_kanji_b4/data`; b1–b3
are the same recipes). This line re-seeds the rows cold on data without them.

Where the seed stands (`sent`, 184 renders, routed): official 29, ≤ 1 edit 92,
a doubled glyph in 100 (`cjk_anima_scale/proposal_seed_synthesis.md`).

## 1. Two thirds of the items trained below where the text is decided

Every run in the seed's chain (`retrain_kana`, `retrain_kanji_b1`–`b4`) drew
`builder.TABLE`'s three groups 1 : 1 : 1:

| group | σ | recipes | items, `retrain_kana` / `b4` |
|---|---|---|---|
| `b0709` | 0.7–0.9 | `scene_single`, `grid_single` | 5 800 / 18 906 |
| `b0507` | 0.5–0.7 | `scene_window` 0.7, `scene_single_small` 0.3 | 5 800 / 18 906 |
| `b0305` | 0.3–0.5 | `scene_window` | 5 800 / 18 906 |

The bands were the band law's rows at the time, not a mistake in the build.
The trajectory reads of 10-01 came after (`cjk_anima_scale/findings.md`):

- slot count and layout are set at σ ≥ 0.9, glyph identity at 0.85–0.7;
- `b0507` for words was borrowed from the piece read, which per-glyph routing
  no longer uses; `b0305`'s 12–24 px row is ceiling-only (teacher-forced);
- with the conditional dropped at 0.8, the seed's `b0305` strings do not
  survive, and below σ 0.5 nothing moves the text.

So the word items — every multi-glyph item the singles have — sit entirely
below 0.7, and nothing trains above 0.9.

Not settled by this: where the items should go instead. A row is one vector
at every σ (`hypothesis.md` H1), so an item trained above where its glyphs
resolve teaches its layout alone — `b0305_reband` (the 12–24 px windows at
0.75–0.93, warm) turned the banner into small columns. Band and glyph size
are one choice (§ 4).

The first cold read says the same (`cjk_anima_scale/experiments/kana_reband`,
`--rows hira`, 10-02): the 81 hiragana rows cold on the kana run's
all-hiragana items, every one at 0.75–0.93, 135 / row. Against
`retrain_kana` on the same renders: words official 10 → 0 / 72, ≤ 2 edits
53 → 0; singles official 17 → 0 / 64, contained 44 → 1. The renders take the
items' layout — small columns in bubbles — with no identity.

Confounded, so not a verdict on the band: the hiragana filter dropped 2 272
of the 2 314 multi-cell grids, the row set is 81 against the kana run's 174,
and the lone glyphs moved (0.7–0.9 → 0.75–0.93) together with the words. The
same items at the seed's own bands have not been run. What it does show is
what the ceiling table predicts for 18–34 px words above σ 0.6; no arm has
trained a word of 48 px or more at a band that reaches 0.8.

## 2. Half the scenes are monochrome or line art

Coloured share of a scene's 128² thumbnail (`polish_seed`'s `colorful`,
saturation > 0.15 and value > 0.15; a scene under 0.08 is greyscale or line
art with a speck of colour):

| | scene items on a scene < 0.08 | flat / grid canvas | both, of all items |
|---|---|---|---|
| `retrain_kana` | 7 210 / 14 500 (49.7 %) | 2 900 / 17 400 | 58 % |
| `retrain_kanji_b4` | 23 527 / 47 265 (49.8 %) | 9 453 / 56 718 | 58 % |

By pool, of 1 897 scenes: `s1` 45 %, `s1w` 49 %, `sl1w` 48 %, `ja_comic` 53 %
under 0.08.

The one read on it does not support it as a cause: `polish_seed --color`
(973 coloured scenes only, 4 steps / row, warm μ 0.1) read no better than the
same polish on every scene — sent official 29 → 10 vs 11, ≤ 1 edit 95 → 46 vs
60 (`cjk_anima_scale/reports/polish_seed_2026_09_30.md`: "the canvases were
not the cause"). That arm was a warm polish on the finished seed, not a cold
run, so the question is open for a reseed; the share above is what is
measured.

## 3. The lettering is not the lettering the model is asked for

- **ー and the other turned marks sit off the column axis.** Every data dir
  of record was drawn with `render_into_scene(tategaki=False)`: a turned
  `ー 〜 …` lands where the font's ascent puts the horizontal glyph, up to
  0.2 em left of the axis (`src/common/render/scene.py`). Vertical windows
  with one: 860 / 7 004 in `retrain_kana`, 606 / 22 931 in `b4`. The fix
  exists as the opt-in `tategaki=True`; only `seed_synth`'s redraw passes it.
- **No item has a second column or line.** `scene_window` draws with
  `max_lines=1`, and the other recipes of the single kind are one glyph, so
  every word the rows saw is one column or one line of 2–6 glyphs. The base
  breaks Japanese into columns by itself (its columns are placed by σ 0.69 —
  `findings.md`), and `seed_synth`'s near-misses needed two columns from
  5 glyphs.

Same family, read off the code and not yet checked on a sheet:

- small kana (っ ゃ ァ …) in a column are the horizontal glyph, centred — not
  the vertical form, up and to the right (2 278 / 7 004 vertical windows in
  `retrain_kana`, 2 027 / 22 931 in `b4`);
- a window is a slice of a dialogue line with no line-head rule: 774 / 9 860
  (`retrain_kana`) and 929 / 32 141 (`b4`) start with ー, a small kana or a
  closing mark (`scene.py`'s `NO_HEAD`).

## 4. Glyph size is tied to the recipe and the band

Drawn px (√(box area / glyphs)), p5 – p50 – p95, `retrain_kana` / `b4`:

| group, recipe | text | `retrain_kana` | `b4` |
|---|---|---|---|
| `b0709` `grid_single` | one glyph per cell | 52 – 91 – 232 | 62 – 108 – 271 |
| `b0709` `scene_single` | one glyph | 41 – 52 – 82 | 42 – 55 – 92 |
| `b0507` `scene_window` | word | 28 – 34 – 51 | 30 – 35 – 53 |
| `b0507` `scene_single_small` | one glyph | 26 – 32 – 38 | 31 – 36 – 40 |
| `b0305` `scene_window` | word | 14 – 18 – 23 | 14 – 18 – 23 |

The sizes span 12–387 px, but each size has one recipe and one band: a word
is never above 64 px and never above σ 0.7; a glyph above 64 px is always a
lone glyph, in a bubble or a grid cell. The seed's `sent` renders write the word as a banner at ≈ 78 px
(long side 401 px) — a size no word item was drawn at.

## Why a reseed and not another arm on the seed

Every arm that changed one of these on the finished seed lost reads: the
polish (official 29 → 11), `b0305_reband` (the banner turned into small
columns; its read was stopped at 158 / 184), `span_reband` (dup 100 → 113),
`seed_synth` (dup 100 → 116), all warm at μ 0.02–0.1 with 4–23 steps / row.
The seed's rows already carry what the low bands taught, and a row is one
vector at every σ; a warm arm adds to it. None of the four points above has
been read cold.
