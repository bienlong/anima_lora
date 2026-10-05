"""DoRA / LoKr / rs-LoRA invariants — CPU-only, no model downloads.

Pins the math the three new plain-family adapter knobs ride on:

* spec dispatch + combo exclusion (``resolve_network_spec``),
* LoKr module: ΔW = 0 at init, forward == org + F.linear(x, kron delta),
  gradients reach every factor, zero-init convention,
* DoRA module: magnitude seeds from W0's row norms (ΔW = 0 at init),
  forward == org + F.linear(x, W' − W0), magnitude stays frozen,
* rs-LoRA: the alpha-scaled-by-sqrt(r) convention reproduces scale=α/√r,
* qinglong_flux: flat-σ draw stays in [0,1], is generator-deterministic,
  and respects the t_min/t_max restriction,
* the standard save path carries ``.dora_scale`` through the qkv defuse.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from networks import NETWORK_REGISTRY, resolve_network_spec
from networks.lora_modules.dokr import DoKrLoRAModule
from networks.lora_modules.lokr import LoKrModule
from networks.lora_modules.dora import DoRALoRAModule
from networks.lora_modules.lokr import LoKrModule, factorization, make_kron
from networks.lora_modules.lora import defuse_standard_qkv


def _wire(module, org: torch.nn.Module) -> None:
    """Stand in for ``apply_to``: forward needs org_forward/org_module set."""
    module.org_forward = org.forward
    module.org_module = org
    module.org_module_ref = [org]


# ---------------------------------------------------------------------------
# Spec dispatch
# ---------------------------------------------------------------------------


def test_spec_dispatch_selects_dora_lokr_and_dokr():
    assert resolve_network_spec({"use_dora": "true"}).name == "dora"
    assert resolve_network_spec({"use_dora": True}).name == "dora"
    assert resolve_network_spec({"use_lokr": True}).name == "lokr"
    assert resolve_network_spec({"use_lokr": "true", "lokr_factor": 4}).name == "lokr"
    # dora + lokr together = DoKr (the user-facing "LoKr + Lora Dora" combo)
    assert resolve_network_spec({"use_lokr": True, "use_dora": True}).name == "dokr"
    assert resolve_network_spec({}).name == "lora"
    assert resolve_network_spec({"use_dora": "false"}).name == "lora"


def test_spec_rejects_routed_combos():
    for routed in (
        {"use_moe_style": "shared_A"},
        {"step_expert_K": 2},
    ):
        with pytest.raises(ValueError, match="plain"):
            resolve_network_spec({"use_dora": True, **routed})
        with pytest.raises(ValueError, match="plain"):
            resolve_network_spec({"use_lokr": True, **routed})
        with pytest.raises(ValueError, match="plain"):
            resolve_network_spec({"use_dora": True, "use_lokr": True, **routed})


def test_registry_entries():
    assert NETWORK_REGISTRY["dora"].save_variant == "standard"
    assert NETWORK_REGISTRY["lokr"].save_variant == "lokr"
    assert NETWORK_REGISTRY["dokr"].save_variant == "lokr"


def test_cfg_rs_lora_accepts_kron_family_and_refuses_routed():
    from networks.lora_anima.config import LoRANetworkCfg
    from networks.lora_modules import HydraLoRAModule  # routed 家族（v2 保留）

    cfg = LoRANetworkCfg.from_kwargs(
        {"rs_lora": "true", "use_lokr": "true", "use_dora": "true"},
        network_dim=8,
        network_alpha=8,
        neuron_dropout=None,
        module_class=NETWORK_REGISTRY["dokr"].module_class,
    )
    assert cfg.rs_lora is True
    with pytest.raises(ValueError, match="rs_lora"):
        LoRANetworkCfg.from_kwargs(
            {"rs_lora": "true", "use_moe_style": "shared_A"},
            network_dim=8,
            network_alpha=8,
            neuron_dropout=None,
            module_class=HydraLoRAModule,
        )


# ---------------------------------------------------------------------------
# LoKr module
# ---------------------------------------------------------------------------


def test_factorization_matches_lycoris_reference():
    assert factorization(128) == (8, 16)
    assert factorization(512) == (16, 32)
    assert factorization(1024) == (32, 32)
    assert factorization(128, 4) == (4, 32)
    for d in (64, 96, 256, 512):
        m, n = factorization(d)
        assert m * n == d and m <= n


def test_lokr_delta_zero_at_init_and_forward_matches_kron():
    torch.manual_seed(0)
    org = torch.nn.Linear(256, 512, bias=False)
    module = LoKrModule("t_lokr", org, 1.0, 4, 16)
    _wire(module, org)
    assert module.use_w2 is False  # (out_k=32, in_n=16) — rank 4 < max/2 = 16

    # Zero-init convention: kron(w1, w2_b=0) = 0 → no delta at step 0.
    assert module.get_weight().abs().sum() == 0

    with torch.no_grad():
        module.lokr_w2_b.normal_()
    x = torch.randn(2, 256)
    org.training = False
    expected = org(x) + torch.nn.functional.linear(
        x,
        make_kron(
            module.lokr_w1.detach().float(),
            (module.lokr_w2_a @ module.lokr_w2_b).detach().float(),
            module.scale,
        ),
    )
    assert torch.allclose(module(x), expected, atol=1e-4)


def test_lokr_gradients_reach_every_factor():
    torch.manual_seed(0)
    org = torch.nn.Linear(256, 512, bias=False)
    module = LoKrModule("t_lokr", org, 1.0, 4, 16)
    _wire(module, org)
    with torch.no_grad():
        module.lokr_w2_b.normal_()
    org.training = True
    module(torch.randn(2, 256)).sum().backward()
    for p in (module.lokr_w1, module.lokr_w2_a, module.lokr_w2_b):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_lokr_full_matrix_w2_forces_scale_one():
    torch.manual_seed(0)
    org = torch.nn.Linear(64, 96, bias=False)
    module = LoKrModule("t_lokr_full", org, 1.0, 8, 16)
    _wire(module, org)
    assert module.use_w2 is True
    assert module.scale == 1.0
    assert float(module.alpha) == 8.0
    org.training = True
    module(torch.randn(2, 64)).sum().backward()
    assert module.lokr_w2.grad is not None


def test_lokr_rejects_conv2d():
    conv = torch.nn.Conv2d(4, 8, 3)
    with pytest.raises(ValueError, match="Linear"):
        LoKrModule("t_conv", conv, 1.0, 4, 16)


# ---------------------------------------------------------------------------
# DoRA module
# ---------------------------------------------------------------------------


def test_dora_magnitude_seeds_from_w0_row_norms():
    torch.manual_seed(0)
    org = torch.nn.Linear(48, 32, bias=False)
    module = DoRALoRAModule("t_dora", org, 1.0, 8, 16)
    _wire(module, org)
    assert torch.allclose(
        module.dora_scale, org.weight.data.float().norm(p=2, dim=1), atol=1e-6
    )
    # Identity start: with up=0 the direction is W0 and s = m/||W0|| = 1.
    assert module.get_weight().abs().sum() == 0


def test_dora_forward_matches_weight_decompose_formula():
    torch.manual_seed(0)
    org = torch.nn.Linear(48, 32, bias=False)
    module = DoRALoRAModule("t_dora", org, 1.0, 8, 16)
    _wire(module, org)
    with torch.no_grad():
        module.lora_up.weight.normal_()
    W0 = org.weight.data.float()
    V = W0 + module.scale * (
        module.lora_up.weight.detach().float()
        @ module.lora_down.weight.detach().float()
    )
    expected_delta = (
        module.dora_scale.unsqueeze(1)
        / V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    ) * V - W0
    x = torch.randn(2, 48)
    org.training = False
    assert torch.allclose(
        module(x), org(x) + torch.nn.functional.linear(x, expected_delta), atol=1e-4
    )


def test_dora_magnitude_trains_under_backward():
    """The magnitude vector is DoRA's second trainable quantity — it must be
    an nn.Parameter receiving gradients (a buffer here would freeze it at
    ‖W0‖ and degrade DoRA to fixed-magnitude normalized LoRA)."""
    torch.manual_seed(0)
    org = torch.nn.Linear(48, 32, bias=False)
    module = DoRALoRAModule("t_dora", org, 1.0, 8, 16)
    _wire(module, org)
    with torch.no_grad():
        module.lora_up.weight.normal_()
    org.training = True
    module(torch.randn(2, 48)).sum().backward()
    assert module.lora_up.weight.grad is not None
    assert module.lora_down.weight.grad is not None
    assert module.dora_scale.grad is not None
    assert module.dora_scale.grad.abs().sum() > 0
    assert any(p is module.dora_scale for p in module.parameters())


def test_dora_rejects_conv2d():
    conv = torch.nn.Conv2d(4, 8, 3)
    with pytest.raises(ValueError, match="Linear"):
        DoRALoRAModule("t_conv", conv, 1.0, 4, 16)


def test_dora_forward_honors_multiplier():
    # Regression: the forward path must scale the weight-decomposed delta by
    # self.multiplier (merge_to/get_weight always did).
    torch.manual_seed(0)
    org = torch.nn.Linear(48, 32, bias=False)
    module = DoRALoRAModule("t_dora_m2", org, 2.0, 8, 16)
    _wire(module, org)
    with torch.no_grad():
        module.lora_up.weight.normal_()
    W0 = org.weight.data.float()
    V = W0 + module.scale * (
        module.lora_up.weight.detach().float()
        @ module.lora_down.weight.detach().float()
    )
    delta = (
        module.dora_scale.unsqueeze(1)
        / V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    ) * V - W0
    x = torch.randn(2, 48)
    org.training = False
    assert torch.allclose(
        module(x),
        org(x) + 2.0 * torch.nn.functional.linear(x, delta),
        atol=1e-4,
    )


def test_lokr_forward_honors_multiplier():
    torch.manual_seed(0)
    org = torch.nn.Linear(256, 512, bias=False)
    module = LoKrModule("t_lokr_m2", org, 2.0, 4, 16)
    _wire(module, org)
    with torch.no_grad():
        module.lokr_w2_b.normal_()
    x = torch.randn(2, 256)
    org.training = False
    expected = org(x) + 2.0 * torch.nn.functional.linear(
        x,
        make_kron(
            module.lokr_w1.detach().float(),
            (module.lokr_w2_a @ module.lokr_w2_b).detach().float(),
            module.scale,
        ),
    )
    assert torch.allclose(module(x), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# DoKr module (LoKr direction + DoRA magnitude)
# ---------------------------------------------------------------------------


def test_dokr_zero_init_and_forward_matches_formula():
    torch.manual_seed(0)
    org = torch.nn.Linear(256, 512, bias=False)
    module = DoKrLoRAModule("t_dokr", org, 1.0, 4, 16)
    _wire(module, org)
    assert module.use_w2 is False
    # Magnitude seeds from W0 row norms → identity start.
    assert torch.allclose(
        module.dora_scale, org.weight.data.float().norm(dim=1), atol=1e-6
    )
    assert module.get_weight().abs().sum() == 0

    with torch.no_grad():
        module.lokr_w2_b.normal_()
    W0 = org.weight.data.float()
    V = W0 + module.scale * make_kron(
        module.lokr_w1.detach().float(),
        (module.lokr_w2_a @ module.lokr_w2_b).detach().float(),
        1.0,
    )
    expected_delta = (
        module.dora_scale.unsqueeze(1)
        / V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    ) * V - W0
    x = torch.randn(2, 256)
    org.training = False
    assert torch.allclose(
        module(x), org(x) + torch.nn.functional.linear(x, expected_delta), atol=1e-4
    )


def test_dokr_gradients_reach_factors_and_magnitude():
    torch.manual_seed(0)
    org = torch.nn.Linear(256, 512, bias=False)
    module = DoKrLoRAModule("t_dokr", org, 1.0, 4, 16)
    _wire(module, org)
    with torch.no_grad():
        module.lokr_w2_b.normal_()
    org.training = True
    module(torch.randn(2, 256)).sum().backward()
    for p in (module.lokr_w1, module.lokr_w2_a, module.lokr_w2_b):
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert module.dora_scale.grad is not None
    assert module.dora_scale.grad.abs().sum() > 0


def test_kron_modules_accept_and_ignore_channel_scale():
    """channel_scaling_alpha>0 时工厂给每个模块带 channel_scale kwarg；Kronecker
    家族没有 lora_down 可吸收，必须显式接受并忽略（回归：纯 LoKr 曾直接
    TypeError，1727f0a 只修了 DoKr 漏了 LoKr）。"""
    scale = torch.ones(256)
    org = torch.nn.Linear(256, 512, bias=False)
    dokr = DoKrLoRAModule("t_dokr", org, 1.0, 4, 16, channel_scale=scale)
    _wire(dokr, org)
    assert dokr(torch.randn(2, 256)).shape == (2, 512)
    org2 = torch.nn.Linear(256, 512, bias=False)
    lokr = LoKrModule("t_lokr", org2, 1.0, 4, 16, channel_scale=scale)
    _wire(lokr, org2)
    assert lokr(torch.randn(2, 256)).shape == (2, 512)


def test_dokr_rejects_conv2d():
    conv = torch.nn.Conv2d(4, 8, 3)
    with pytest.raises(ValueError, match="Linear"):
        DoKrLoRAModule("t_conv", conv, 1.0, 4, 16)


# ---------------------------------------------------------------------------
# rs-LoRA alpha convention
# ---------------------------------------------------------------------------


def test_rs_lora_alpha_convention_reproduces_sqrt_scale():
    # network.py scales the per-module alpha by sqrt(dim); loaders then
    # compute scale = alpha_stored / dim — which must equal alpha / sqrt(r).
    alpha, rank = 16, 32
    stored = alpha * math.sqrt(rank)
    assert math.isclose(stored / rank, alpha / math.sqrt(rank))


# ---------------------------------------------------------------------------
# qinglong_flux
# ---------------------------------------------------------------------------


def _qinglong_args(**overrides) -> SimpleNamespace:
    base = dict(
        timestep_sampling="qinglong_flux",
        discrete_flow_shift=1.0,
        sigmoid_scale=1.0,
        sigmoid_bias=0.0,
        logit_mean=0.0,
        logit_std=1.0,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        t_min=None,
        t_max=None,
        weighting_scheme="uniform",
        mode_scale=1.29,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_qinglong_flux_draw_is_bounded_and_deterministic():
    from library.runtime.noise import draw_flat_sigmas

    args = _qinglong_args()
    gen = torch.Generator().manual_seed(42)
    sigmas = draw_flat_sigmas(args, 4096, 64, 64, "cpu", generator=gen)
    assert sigmas.shape == (4096,)
    assert (sigmas >= 0).all() and (sigmas <= 1).all()

    gen = torch.Generator().manual_seed(42)
    again = draw_flat_sigmas(args, 4096, 64, 64, "cpu", generator=gen)
    assert torch.equal(sigmas, again)


def test_qinglong_flux_respects_sigma_range_restriction():
    from library.runtime.noise import draw_flat_sigmas

    args = _qinglong_args(t_min=0.2, t_max=0.8)
    sigmas = draw_flat_sigmas(args, 512, 64, 64, "cpu")
    assert sigmas.min() >= 0.2 - 1e-6 and sigmas.max() <= 0.8 + 1e-6


def test_qinglong_flux_feeds_the_training_step():
    from library.runtime.noise import (
        FlowMatchEulerDiscreteScheduler,
        get_noisy_model_input_and_timesteps,
    )

    args = _qinglong_args()
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    latents = torch.randn(2, 4, 16, 16)
    noise = torch.randn_like(latents)
    noisy, timesteps, sigmas = get_noisy_model_input_and_timesteps(
        args, sched, latents, noise, "cpu", torch.float32
    )
    assert noisy.shape == latents.shape
    assert timesteps.shape == (2,)
    assert (timesteps >= 0).all() and (timesteps <= 1).all()


# ---------------------------------------------------------------------------
# Save path: DoRA rides the standard writer
# ---------------------------------------------------------------------------


def test_defuse_carries_dora_scale_chunks():
    torch.manual_seed(0)
    down = torch.randn(4, 16)
    up = torch.randn(48, 4)  # 3 fused components x 16 rows
    dora_scale = torch.randn(48)
    sd = {
        "lora_unet_blocks_0_self_attn_qkv_proj.lora_down.weight": down,
        "lora_unet_blocks_0_self_attn_qkv_proj.lora_up.weight": up,
        "lora_unet_blocks_0_self_attn_qkv_proj.alpha": torch.tensor(4.0),
        "lora_unet_blocks_0_self_attn_qkv_proj.dora_scale": dora_scale,
    }
    defuse_standard_qkv(sd)
    chunks = dora_scale.chunk(3, dim=0)
    for i, letter in enumerate("qkv"):
        assert f"lora_unet_blocks_0_self_attn_{letter}_proj.lora_down.weight" in sd
        ds = sd[f"lora_unet_blocks_0_self_attn_{letter}_proj.dora_scale"]
        assert torch.equal(ds, chunks[i])
    assert not any("qkv_proj" in k for k in sd)


def test_max_norm_regularization_noops_on_kron_networks():
    # Regression: scale_weight_norms on an all-LoKr network used to raise
    # ZeroDivisionError (no lora_down keys → empty norms list).
    import types

    from networks.lora_anima.network import LoRANetwork

    fake = types.SimpleNamespace(
        state_dict=lambda: {
            "lora_unet_blocks_0_self_attn_qkv_proj.lokr_w1.weight": torch.randn(4, 4),
            "lora_unet_blocks_0_self_attn_qkv_proj.lokr_w2_b.weight": torch.randn(4, 4),
            "lora_unet_blocks_0_self_attn_qkv_proj.alpha": torch.tensor(4.0),
        }
    )
    keys_scaled, mean_norm, max_norm = LoRANetwork.apply_max_norm_regularization(
        fake, 5.0, torch.device("cpu")
    )
    assert keys_scaled == 0 and mean_norm == 0.0 and max_norm == 0.0
