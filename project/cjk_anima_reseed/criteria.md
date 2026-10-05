# criteria — what a reseed arm is judged on (2026-10-04)

**The target** (user, 10-02 / 10-04): Japanese dialogue in speech bubbles at
manga size — multi-panel and single-panel manga, illustrations — and in the
end dense dialogue and sentences over several columns (the `sincos` image
set has them). Not the `sent` banner, not a glyph alone on a page.

## The unit is the word

A word is the smallest thing a read counts; a sentence is the target.
Singles (one glyph alone) are diagnosis: they say whether a row has an
identity, never which arm is better. `product_criteria.md` (scale line)
already had it: "the product draws words".

## The ruler: dialogue

Strings: real JA words and phrases (the standing rule — never random
strings), word length up to sentence length, in a speech bubble, one or two
columns. Captions name the bubble the way the target's prompts do; the
caption is the same for every arm.

Read per render, paired on the same prompt × seed against the floor:

- **text** — the readers' exact / contained / ≤ 1 / ≤ 2 edits and `dup` (a
  doubled glyph or a read longer than the string: the slot count against the
  string) per string;
- **page** — what the base draws for the same prompt with an EN string of
  the same length (`product_criteria.md` Axis 2): EN-ref cos outside the text
  box, and the sheet by eye for the two failure shapes, **paste** (a flat
  white box or block with the text, the scene overridden) and **wipe** (the
  scene's bubble with the base's guess in it), and a third, **banner** (the
  text out of the bubble as a title line).

Power is in strings, not seeds: more strings over prompts, and a direction
called by the sign over strings that share no row, never off a total alone
(`project_cjk_blind_pairs_protocol`).

## Floor

The dialogue ruler has no floor yet (`_archive/motivation2.md` § 6: "sentence-length
captions have no floor"). It is rendered once — the EN references and
`retrain_kana` / `seed_retrain_0930` on the ruler's strings and prompts —
and every arm after reads against that cache; it is not re-rendered per
experiment.

## What the banner grid is now

`probe_split`'s plain read (9 hiragana words + 8 singles × p00–p03, seed 0,
`Text reads as "…"`) draws a top banner. Its verdicts (`reports/`, 10-02 –
10-04) stand as verdicts about the banner: "word fit" there is a word
filling the banner's slots. An arm that lost on it is not lost on the
target until it is read on the dialogue ruler — first `retrain_kana`,
`kana_up`, `ball_rk_bubble` and `stick_nlg_high` (the one arm that drew
the bubble layout and lost the banner read). Read 10-05
(`reports/ruler_2026_10_05.md`): none beats the floor; ball_rk_bubble leans
up on ≤ 2 edits, stick_nlg_high loses the text. On the sensitive prompts
(§ 4) gs_rkstick, ball_rk_bubble and kana_up tie retrain_kana,
stick_nlg_high loses, seed_retrain_0930 (its kanji rows) is the best table.

## The ruler as built (10-05, `ruler.py build`)

- **Strings** (user, 10-05): the training set's own bubble dialogue —
  captions tagged `speech bubble` / `comic` / `dialogue`, their `Japanese
  text reads as` clause, not SFX — one string per image, 32 per bin: short
  2–4 glyphs (16 interjections, 16 lexical), mid 5–9, long 10–20. OCR
  misreads, SFX filed as dialogue and non-word fragments dropped by eye
  (`DROP`). 96 strings, 96 images.
- **Prompts** (user, 10-05): each string's own image caption, text clauses
  and other-language text tags out, `speech bubble, japanese text` in; the
  rating `sensitive` and the explicit tags out (`ruler.py` `EXPLICIT`);
  characters and artist as captioned. Source images tagged as a child leave
  the pool; an image gives up to two strings.
- **Seen / unseen**: the reseed windows came from these same lines, so
  every string carries `cov3` (its trigrams' share in the arms of record's
  training text). A read reports all strings and `cov3 ≤ 0.15` apart; the
  unseen side is 25 short / 9 mid / 4 long, so only the short bin can call
  a direction on it. Not yet held out of later builds: `window_pool` holds
  a string by its trigrams (a 2-glyph one by its bigram), so the short
  bin's `うっ` / `おっ` / `はっ` would strip every window holding them —
  open, decided before the next build.
- **EN reference** (user, 10-05): each line's EN rendering (hand-written,
  `en.json`), same prompt with `english text`, `English text reads as`.

Still open: whether the readers read a two-column bubble as one string.
