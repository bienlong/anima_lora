# DoRA / DoKr / LoKr / rs-LoRA — plain-family adapter extensions

Additions to the plain (non-routed) LoRA family, selectable from the GUI
Config tab (LoRA family dropdown) or via TOML/CLI network kwargs. All of them
are dispatched by `resolve_network_spec` (`networks/__init__.py`) and validate
their combos there — routed/ortho cells (Hydra, FeRA, Chimera, Ortho*,
step-expert) refuse them with a `ValueError` rather than silently ignoring
the flags.

Combo matrix (`use_dora` × `use_lokr`):

| use_dora | use_lokr | adapter | direction | magnitude (`dora_scale`) | ComfyUI file |
|---|---|---|---|---|---|
| – | – | LoRA | low-rank `up @ down` | – | native |
| ✓ | – | DoRA | low-rank `up @ down` | ✓ | native |
| – | ✓ | LoKr | `kron(w1, w2)` | – | in-repo; `scripts/export_comfyui_lora.py` → ComfyUI |
| ✓ | ✓ | **DoKr** (LyCORIS `dokr`) | `kron(w1, w2)` | ✓ | in-repo; `scripts/export_comfyui_lora.py` → ComfyUI |

`rs_lora` composes with all four (see §rs-LoRA).

## Exporting a trained file to ComfyUI

The training save uses kohya `lora_unet_*` underscore keys with the fused
`self_attn_qkv_proj` / `cross_attn_kv_proj` matrices, while ComfyUI's anima
DiT expects `diffusion_model.*` dotted keys with separated q/k/v_proj — a raw
save attaches only a fraction of its patches there and renders pure noise.
`scripts/export_comfyui_lora.py` rewrites the file:

```bash
python scripts/export_comfyui_lora.py --lora output/ckpt/<name>.safetensors
# --dit defaults to configs/base.toml pretrained_model_name_or_path
# --out defaults to <name>_comfyui.safetensors next to the input
```

It (1) rewrites keys to `diffusion_model.*`; (2) splits the fused matrices —
kv side is a lossless kron row split of `lokr_w1`, qkv side is lossless when
`out_k` divides the per-component rows and otherwise falls back to a rank ≤
`--max_svd_rank` (default 64) SVD per component; and (3) rescales `dora_scale` to the consumer norm convention
(`m·‖W0‖/‖V‖` — ComfyUI's `weight_decompose` normalizes by the *original*
weight norm `‖W0‖` while training normalizes by `‖V‖`; the reciprocal only
cancels when `‖V‖≈‖W0‖` and otherwise leaves a `2ε` row-scale error) and
stores it 2-D `[out, 1]` per the LyCORIS disk convention — a 1-D vector
broadcasts into an `[out, out]` outer product inside `weight_decompose` and
silently corrupts square weights. The run ends with a per-module error
report; a count of 0 modules over 2% means the file is good to use. Scope:
LoKr / DoKr files (modules carrying `lokr_w1`); plain LoRA / DoRA saves are
not converted by this tool. Branch coverage is pinned by
`tests/test_export_comfyui_lora.py`.

## DoRA (`use_dora = true`)

Weight-Decomposed Low-Rank Adaptation ([arXiv:2402.09353](https://arxiv.org/abs/2402.09353)):
the adapted weight is `W' = m ⊙ V / ‖V‖_row` with `V = W0 + scale·(up @ down)` —
the low-rank product trains the per-row *direction* while a trainable magnitude
vector `m` (one float per output channel) absorbs the norm drift that makes
plain LoRA's update geometry anisotropic. Convergence is typically steadier;
VRAM increases slightly (the fp32 `V` intermediates are `(out × in)`).

Implementation: `networks/lora_modules/dora.py::DoRALoRAModule` (Linear-only —
the Anima DiT is all Linear). Invariants:

- `m` starts at `‖W0‖_row`, so `ΔW = 0` at step 0 (same identity-start
  convention as every variant in the package).
- The `(out × in)` delta intermediates are computed under
  `torch.utils.checkpoint` (recompute-in-backward), so per-module retained
  memory stays at the two rank matrices; DiT-block gradient checkpointing
  recomputes them either way.
- `rank_dropout` is unsupported (warned and ignored) — there are no rank
  activations to mask. `use_timestep_mask` (T-LoRA) raises: DoRA's forward is
  weight-space and never builds the gated activations.
- `scale_weight_norms` still applies (it scales the up/down products; DoRA's
  norm path recomputes from the current weights, staying consistent).

Save format: standard `lora_down/up/alpha` keys **plus `<name>.dora_scale`**
(the magnitude vector, chunked along the output axis when qkv-fused modules
are defused). This is the key ComfyUI's native DoRA path consumes, so the file
loads in stock ComfyUI without a converter. `save_variant` stays `"standard"`.

## LoKr (`use_lokr = true`)

LoRA + Kronecker product ([arXiv:2309.14859](https://arxiv.org/abs/2309.14859)):
`ΔW = kron(w1, w2) * scale` where `w1` is a small full matrix and `w2` is either
a second full matrix (`use_w2`, forced `scale = 1` and `alpha = lora_dim`) or
itself low-rank `w2_a @ w2_b`. More expressive per parameter than a rank-r
two-factor LoRA; `w2_b` starts at zero so `ΔW = 0` at init.

Implementation: `networks/lora_modules/lokr.py::LoKrModule` (Linear-only).
`lokr_factor` (int, default `-1`) pins the w1 axis factorization — `-1` picks
the most balanced factors (LyCORIS convention, `factorization()`). On reload
the saved `w1_shape` + `use_w2` split is sniffed from the checkpoint keys
(`create_network_from_weights`), so the exact trained factorization is
restored regardless of the current `lokr_factor`.

Save format: `lokr_w1.weight` + `lokr_w2.weight` (or `lokr_w2_a/w2_b`).
**In-repo loading only** — the standard writer does not defuse the
runtime-fused qkv keys for this variant, and ComfyUI's stock LoRA path is not
wired for Anima LoKr files. Train/merge/test through the repo tooling
(`inference.py`, `make merge`, …).

## DoKr (`use_lokr = true` + `use_dora = true`)

DoRA weight-decomposition applied to a LoKr direction (LyCORIS `dokr`):
`W' = m ⊙ V / ‖V‖_row` with `V = W0 + scale·kron(w1, w2)` — the Kronecker
product trains the per-row *direction*, the trainable magnitude vector `m`
(one float per output channel) absorbs the norm drift. `m` starts at
`‖W0‖_row` (ΔW = 0 at init); the `(out × in)` intermediates are
checkpoint-recomputed, same memory discipline as LoKr/DoRA.

Save format: `lokr_w1 / lokr_w2[_a/_b]` plus `<name>.dora_scale` through the
lokr write path. In-repo loading only (same constraint as plain LoKr —
runtime-fused qkv keys are not defused for this variant).

## rs-LoRA (`rs_lora = true`)

Rank-stabilized scaling ([rs-LoRA](https://arxiv.org/abs/2312.03732)): the
effective per-module scale becomes `alpha / √r` instead of `alpha / r`, which
keeps the update magnitude sane at high rank (> 32). Implementation is a
construction-time alpha transform in `LoRANetwork.create_modules`:
`alpha_val *= sqrt(dim)` — the **saved alpha buffer therefore carries `√r`**,
so every kohya-format loader (in-repo reload, ComfyUI, sd-scripts) reproduces
the trained scale from plain `alpha / dim` with no metadata sniffing.

Caveats:

- Plain LoRA / DoRA / LoKr / DoKr (`module_class` check in
  `LoRANetworkCfg.from_kwargs`); the routed families compute their own
  scaling and refuse the flag. use_w2 LoKr forces `scale = 1`, so rs is
  inert there by construction.
- Resume/warm-start must keep `rs_lora = true` in the TOML: the transform is
  applied at construction from the config alpha, so a resume that drops the
  flag trains at `alpha / r` against a checkpoint scaled for `alpha / √r`.
  (`load_weights` refuses cross-family mismatches — dora/lokr vs plain — but
  cannot detect a dropped `rs_lora`.)

## `timestep_sampling = "qinglong_flux"`

Port of the qinglong triple-hybrid sampler (sdbds, musubi-tuner PR #407):
each batch sample independently picks one of three samplers —

| share | sampler | draw |
|---|---|---|
| 79% | flux_shift (mid_shift) | `σ(sigmoid_scale·randn + sigmoid_bias)` shifted by `exp(mu)`, `mu = get_lin_function(y1=0.5, y2=1.15)((h/2)·(w/2))` |
| 11% | logsnr (Style-Friendly, [arXiv:2411.14793](https://arxiv.org/abs/2411.14793)) | `t = σ(-logsnr/2)`, `logsnr ~ N(logit_mean, logit_std)` |
| 10% | logsnr2 | same form with fixed `N(5.36, 1.0)` — the low-noise / detail regime |

`sigmoid_scale`/`sigmoid_bias` apply to the mid (flux_shift) branch only, as in
the other logit-normal branches; the logsnr draws stay pure per the musubi
implementation. `t_min`/`t_max` restriction applies to the mixture as a whole.
Provides a flat-σ draw, so `sigma_lowres` composes with it.

## GUI surface

- New LoRA-family variants: **DoRA** / **rs-LoRA** / **LoKr**
  (`configs/gui-methods/{dora,rs_lora,lokr}.toml`), plus `use_dora` / `rs_lora`
  switches on the plain **LoRA** variant. The **LoKr** variant carries all
  three knobs (`use_lokr` / `lokr_factor` / `use_dora` / `rs_lora`) — tick
  `use_dora` (+ optionally `rs_lora`) on the LoKr variant to train DoKr,
  matching other trainers' "Lora Type = LoKr + Lora Dora / Lora Rs" layout.
- `timestep_sampling` and `scale_weight_norms` are promoted into the Basic
  section and get the accent highlight (★ + bold link-colored label) per user
  request; `scale_weight_norms` now ships a default of `5.0` in `base.toml`
  (0 disables; inert on LoKr keys — no up/down pairs to scale).
- Field tooltips for all new keys live in `gui/explanations/guides/<lang>/_fields.json`
  (en/cn/ja/ko).

## Tests

`tests/test_dora_lokr_rs.py` pins: spec dispatch (incl. dora+lokr → dokr) +
combo refusals, LoKr zero-init / kron-forward / full-matrix scale=1 rules,
DoRA magnitude seeding and the weight-decompose forward formula (magnitude
frozen, multiplier honored), DoKr zero-init / forward formula / factor
gradients, the rs alpha convention + cfg module_class acceptance matrix,
qinglong_flux bounds/determinism/range restriction + end-to-end timestep
path, and dora_scale chunking through the qkv defuse.
