# probe_pres: the page-preservation term on the rows (2026-10-06)

From the user (10-06, after f0 in ComfyUI: the longer the JA line, the more
often it lands as a column across the page and the subject goes): one glyph
or a sentence, the page outside the text should stay as the base draws it —
`../cjk_anima_scale/product_criteria.md` Axis 2, read so far only as a
ruler column (`en_match`, `en_tok_out`). Can it be a training term?

    L_pres = mean_out (v_θ(x_σ, c_JA; rows) − sg[v_base(x_σ, c_EN)])²

Student: the DiT with the rows under the item's caption. Teacher: the same
frozen DiT at the same x_σ (latent and ε) under the caption with its JA
string swapped for an EN line of its length bin (one / 2–6 / 7+ glyphs,
fixed pools in `probe_pres.py`) and `japanese text` → `english text`; the
EN caption holds no ext id, so `ExtDelta` stays out and the teacher sees no
row. `mean_out`: the cells outside the item's text box dilated by 2 latent
cells (88–98 % of the page). The data term's out-box residual is ~82 % the
draw's own (σ, ε) (`probe_scene_2026_10_06.md`); here student and teacher
share them.

`probe_pres.py grad`: the rows at `seed_fixed_1005_stick080` (f0's start),
the trainer's first 1 200 batches of `sent_kanji`'s data (4 800 items), each
batch at one σ of 0.5 / 0.6 / 0.7 / 0.8 / 0.9 / 0.95 in turn (200 each, the
bands not used). Per batch the data term's in-box and out-box summands
(`probe_geom.split_terms`) and L_pres, each one's gradient on the rows; no
step. Job `20261006-203323-8e685d` (daemon, 21.4 min, 1.07 s / batch,
dynamic-seq compile) → `output/cjk_anima_reseed/probe_pres/s080/`
(`grads.pt`, `read.json`).

- **L_pres is the most coherent of the three terms.** Per-draw signal
  share ρ (Spearman–Brown on split halves, median over rows): kana in-box
  0.009 / out-box 0.004 / **pres 0.016**, kanji 0.011 / 0.007 / **0.028**;
  the order holds in every σ cut (σ ≤ 0.7 kanji: pres 0.056 against in-box
  0.014). Still 2–8 % a draw: less noise than the data term, not little.
- **It is tiny where the rows train.** |g_pres| / |g_in| per draw: 0.003–
  0.009 at σ 0.5–0.7, 0.03 at 0.8, 0.08–0.09 at 0.9, 0.13–0.15 at 0.95.
- **It pulls with the in-box term up to 0.9 and against it at 0.95**:
  cos(mean pres, mean in) per row +0.02 – +0.08 through σ 0.9, −0.03 (kana)
  / −0.12 (kanji) at 0.95, where the data term asks for the text and L_pres
  for the EN page.
- **The rows share its direction, and the direction is not the stick.**
  ‖mean of the rows' unit mean gradients‖²: pres kana 0.24 / kanji 0.12
  (in-box 0.15 / 0.04; chance 1 / N ≈ 0.007 / 0.003). Its cos with the
  family stick −0.13 – −0.18: a step shrinks the stick (the sign the × 0.8
  read had), but 2–3 % of the common direction's energy is on it.
- **Longer lines and higher σ move the page more.** L_pres per item (JA against
  EN outside the box) by σ and length: 7+ glyphs over one glyph ×1.6 at
  σ 0.5–0.7, ×2.0–2.2 at 0.8–0.95; and ~10× from σ 0.5 to 0.95 at every
  length.

## 1. The terms

Medians over rows with ≥ 6 draws in the cut.

| cut · family | rows | draws | \|pres\|/\|in\| | cos(pres, in) | ρ in | ρ out | ρ pres | common in / pres | cos(pres common, stick) |
|---|---|---|---|---|---|---|---|---|---|
| all · kana | 150 | 45 | 0.035 | +0.096 | 0.009 | 0.004 | 0.016 | 0.150 / 0.239 | −0.140 |
| all · kanji | 375 | 10 | 0.038 | +0.032 | 0.011 | 0.007 | 0.028 | 0.042 / 0.120 | −0.176 |
| σ ≤ 0.7 · kana | 133 | 24 | 0.006 | +0.083 | 0.014 | 0.001 | 0.024 | 0.145 / 0.185 | −0.124 |
| σ ≤ 0.7 · kanji | 184 | 8 | 0.007 | +0.040 | 0.014 | 0.003 | 0.056 | 0.059 / 0.172 | −0.214 |
| σ ≥ 0.8 · kana | 137 | 30 | 0.078 | +0.066 | 0.017 | 0.009 | 0.028 | 0.148 / 0.244 | −0.139 |
| σ ≥ 0.8 · kanji | 172 | 9 | 0.081 | +0.027 | 0.020 | 0.017 | 0.044 | 0.060 / 0.161 | −0.145 |
| σ 0.95 · kana | 87 | 25 | 0.133 | −0.028 | 0.030 | 0.016 | 0.040 | 0.182 / 0.241 | −0.133 |
| σ 0.95 · kanji | 31 | 9 | 0.152 | −0.124 | 0.046 | 0.023 | 0.084 | 0.142 / 0.206 | −0.131 |

## 2. L_pres by σ and length

Mean per item (n ≈ 32–52 one-glyph, 170–204 window, 549–590 line items per σ).

| σ | one | 2–6 | 7+ | 7+ / one |
|---|---|---|---|---|
| 0.5 | 0.00012 | 0.00015 | 0.00019 | 1.6 |
| 0.6 | 0.00014 | 0.00020 | 0.00027 | 1.9 |
| 0.7 | 0.00025 | 0.00032 | 0.00040 | 1.6 |
| 0.8 | 0.00048 | 0.00083 | 0.00094 | 2.0 |
| 0.9 | 0.00078 | 0.00152 | 0.00173 | 2.2 |
| 0.95 | 0.00105 | 0.00193 | 0.00212 | 2.0 |

Not apart here: the share of the gap the rows can close (the
`japanese text` / `english text` tag and the clause wording move the page
too, and no row holds them), and the EN line's own length (the pools grow
with the bin, so the read is JA against an EN line of its length band).

## Read

Axis 2 as a loss carries direction: per draw more than the data term does,
and the rows share it. Where it has weight is above the rows' bands — at
σ ≤ 0.7 it is under 1 % of the in-box term, so a run that wants it trains
L_pres at σ 0.8–0.95 (λ ≈ 5–10 matches the in-box term at 0.9), and at 0.95
it pulls against the in-box term. The 09-17 ΔFM arm (EN sibling, whole
canvas: wipe tail gone, reader hits halved,
`../finished/cjk_renderable_anima/reports/synth_pair_2026_09_17.md`) is the
text cost to watch.

## Trained: 16 hiragana, λ 5 (10-06)

`probes/probe_pres_train.py`, label `h16`: 16 hiragana of f0's
(ごみろつきをじやりそさおらこるよ, 151–999 draws each) live at
`seed_fixed_1005_stick080`, every other row frozen; plain AdamW lr 2e-4,
1 500 × 4, and `p5` = the same batches plus 5 · L_pres at σ ~ U(0.8, 0.95)
(own generator). Held out: 400 items, teacher-forced.

| p5 − plain | σ 0.8 | 0.9 | 0.95 | item band |
|---|---|---|---|---|
| L_pres | −20 % (16/16 rows) | −27 % (16/16) | −31 % (16/16) | — |
| in-box | +0.0001 (n.s.) | +0.0010 (z 3.2) | +0.0014 (z 3.1) | +0.0015 (z 9; 1/16 rows lower) |

Plain leaves L_pres where start had it (σ 0.8–0.95 within ±0.00006) and
takes the in-box term down 0.0093; p5 gives back 16 % of that. Δ(p5 − plain)
is 31 % one vector shared by the 16 rows, cos +0.68 with § 1's common pres
step and −0.09 with the stick; p5's move keeps cos 0.80 with plain's.

Renders: `ruler.py --rows_pt` (`h16_plain@punct`, `h16_p5@punct`), the 56 of
96 ruler strings holding a live kana (the other 40 render the same pixels in
both arms), sheet EN | p5 | plain →
`output/cjk_anima_reseed/probe_pres_train/h16/sheet_changed.png`. User's
read of the sheet (19 of the 56 rendered at the time): **p5 keeps the page
closer to the EN render than plain.**
The cached start renders (`seed_fixed_1005_stick080@punct`, 10-05) are not
comparable: the render path's retrain_kana check drifted 0.0 → 5.8 mean
|Δpx| since, and start vs plain differs on strings with no live row.

## Open

- More λ (2, 10) and the band without 0.95, read on the 56 strings.
- How much of the gap is the rows': the same read with the rows at the
  punct pack's raw rows (no training) as the student.
