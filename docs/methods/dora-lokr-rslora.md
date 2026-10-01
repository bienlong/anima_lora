# DoRA / LoKr / rs-LoRA — plain-family adapter extensions

Three additions to the plain (non-routed) LoRA family, selectable from the GUI
Config tab (LoRA family dropdown) or via TOML/CLI network kwargs. All three are
dispatched by `resolve_network_spec` (`networks/__init__.py`) and validate their
combos there — routed/ortho cells (Hydra, FeRA, Chimera, Ortho*, step-expert)
refuse them with a `ValueError` rather than silently ignoring the flags.

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

## rs-LoRA (`rs_lora = true`)

Rank-stabilized scaling ([rs-LoRA](https://arxiv.org/abs/2312.03732)): the
effective per-module scale becomes `alpha / √r` instead of `alpha / r`, which
keeps the update magnitude sane at high rank (> 32). Implementation is a
construction-time alpha transform in `LoRANetwork.create_modules`:
`alpha_val *= sqrt(dim)` — the **saved alpha buffer therefore carries `√r`**,
so every kohya-format loader (in-repo reload, ComfyUI, sd-scripts) reproduces
the trained scale from plain `alpha / dim` with no metadata sniffing.

Caveats:

- Plain LoRA / DoRA only (`module_class` check in `LoRANetworkCfg.from_kwargs`);
  LoKr and the routed families compute their own scaling and refuse the flag.
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
  switches on the plain **LoRA** variant.
- `timestep_sampling` and `scale_weight_norms` are promoted into the Basic
  section and get the accent highlight (★ + bold link-colored label) per user
  request; `scale_weight_norms` now ships a default of `5.0` in `base.toml`
  (0 disables; inert on LoKr keys — no up/down pairs to scale).
- Field tooltips for all new keys live in `gui/explanations/guides/<lang>/_fields.json`
  (en/cn/ja/ko).

## Tests

`tests/test_dora_lokr_rs.py` pins: spec dispatch + combo refusals, LoKr
zero-init / kron-forward / full-matrix scale=1 rules, DoRA magnitude seeding
and the weight-decompose forward formula (magnitude frozen), the rs alpha
convention, qinglong_flux bounds/determinism/range restriction + end-to-end
timestep path, and dora_scale chunking through the qkv defuse.
