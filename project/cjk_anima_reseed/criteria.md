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

The dialogue ruler has no floor yet (motivation2 § 6: "sentence-length
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
the bubble layout and lost the banner read).

## Open, set when the ruler is built

- the string set — lengths, words vs phrases, the sources (scene pools and
  line pools of record; no new pool without asking);
- the prompts — the scene pools' manga / comic prompts;
- whether the readers read a two-column bubble as one string.
