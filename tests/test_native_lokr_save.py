"""Native save emission + reassembly roundtrip — CPU-only, no model downloads.

Pins the save-side native ComfyUI emission (exact LoKr qkv/kv split, DoRA
magnitude rescale ``m' = m·‖W0‖/‖V‖`` stored 2-D, ``diffusion_model.*``
reprefix) and its inverse (:func:`reassemble_comfyui_native_sd`) so that:

* the main training artifact IS ComfyUI-native with no post-conversion step,
* the exact split cases reconstruct the fused kron rows element-for-element,
* the ComfyUI math applied to the emitted file equals the training semantics,
* resume/merge can read the native file back losslessly (cases 1/2),
* SVD-fallback components refuse re-fusion loudly instead of corrupting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from networks.lora_save import (
    build_standard_state_dict,
    export_comfyui_sidecar,
    reassemble_comfyui_native_sd,
)

FUSED = "lora_unet_blocks_0_self_attn_qkv_proj"
PLAIN = "lora_unet_blocks_0_mlp_layer1"
LETTERS = ("q", "k", "v")


def _base_weights():
    g = torch.Generator().manual_seed(11)
    return {
        FUSED: torch.randn(24, 8, generator=g),
        PLAIN: torch.randn(8, 8, generator=g),
    }


def _fused_sd(case: str):
    g = torch.Generator().manual_seed(5)
    if case == "case1":  # comp_dim % c == 0 → w1 row-block split, w2 shared
        w1, w2_a, w2_b = torch.randn(6, 2, generator=g), torch.randn(4, 2, generator=g), torch.randn(2, 4, generator=g)
    elif case == "case2":  # factor=0 degenerate: w1 1×1, components inside one w1 row
        w1, w2_a, w2_b = torch.randn(1, 1, generator=g), torch.randn(24, 3, generator=g), torch.randn(3, 8, generator=g)
    else:  # case3: a=2 → middle component spans two w1 rows → SVD fallback
        w1, w2_a, w2_b = torch.randn(2, 2, generator=g), torch.randn(12, 2, generator=g), torch.randn(2, 4, generator=g)
    return {
        f"{FUSED}.lokr_w1": w1.to(torch.bfloat16),
        f"{FUSED}.lokr_w2_a": w2_a.to(torch.bfloat16),
        f"{FUSED}.lokr_w2_b": w2_b.to(torch.bfloat16),
        f"{FUSED}.alpha": torch.tensor(4.0),
        f"{FUSED}.dora_scale": (torch.randn(24, generator=g).abs() + 0.5).to(torch.bfloat16),
    }


def _plain_sd(g):
    return {
        f"{PLAIN}.lora_down.weight": torch.randn(2, 8, generator=g).to(torch.bfloat16),
        f"{PLAIN}.lora_up.weight": torch.randn(8, 2, generator=g).to(torch.bfloat16),
        f"{PLAIN}.alpha": torch.tensor(4.0),
        f"{PLAIN}.dora_scale": (torch.randn(8, generator=g).abs() + 0.5).to(torch.bfloat16),
    }


def _factors(native, letter):
    p = f"diffusion_model.blocks.0.self_attn.{letter}_proj"
    if f"{p}.lokr_w1" in native:
        w1 = native[f"{p}.lokr_w1"].to(torch.float)
        w2 = native[f"{p}.lokr_w2_a"].to(torch.float) @ native[f"{p}.lokr_w2_b"].to(torch.float)
        s = float(native[f"{p}.alpha"].item()) / native[f"{p}.lokr_w2_b"].shape[0]
        return p, torch.kron(w1, w2) * s
    a_down = native[f"{p}.lora_A.weight"].to(torch.float)
    b_up = native[f"{p}.lora_B.weight"].to(torch.float)
    s = float(native[f"{p}.alpha"].item()) / a_down.shape[0]
    return p, (b_up @ a_down) * s


def _assert_comfy_parity(native, base_weights, fused_w1, fused_w2_a, fused_w2_b, m_fused, tol):
    """ComfyUI math from the emitted file == training semantics from the inputs."""
    w0_full = base_weights[FUSED].to(torch.float)
    w1 = fused_w1.to(torch.float)
    w2 = fused_w2_a.to(torch.float) @ fused_w2_b.to(torch.float)
    s = 4.0 / fused_w2_b.shape[0]
    v_full = w0_full + s * torch.kron(w1, w2)
    w_train_full = (m_fused.to(torch.float).unsqueeze(1) / v_full.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)) * v_full
    comp_dim = w0_full.shape[0] // 3
    for i, letter in enumerate(LETTERS):
        sl = slice(i * comp_dim, (i + 1) * comp_dim)
        p, delta = _factors(native, letter)
        w0_i = w0_full[sl]
        m2d = native[f"{p}.dora_scale"].to(torch.float)
        assert m2d.dim() == 2 and m2d.shape == (w0_i.shape[0], 1), (p, m2d.shape)
        w_comfy = (w0_i + delta) * (m2d / (w0_i.norm(p=2, dim=1, keepdim=True) + 1e-8))
        err = ((w_comfy - w_train_full[sl]).norm() / w0_i.norm()).item()
        assert err < tol, f"{p}: ComfyUI-vs-training error {err:.5f} (tol {tol})"


@pytest.mark.parametrize("case,tol", [("case1", 1e-5), ("case2", 1e-5), ("case3", 0.05)])
def test_native_save_and_comfy_parity(case, tol):
    g = torch.Generator().manual_seed(3)
    base_weights = _base_weights()
    original = _fused_sd(case)
    original.update(_plain_sd(g))
    m_fused = original[f"{FUSED}.dora_scale"]
    fused_w1, fused_w2_a, fused_w2_b = (
        original[f"{FUSED}.lokr_w1"],
        original[f"{FUSED}.lokr_w2_a"],
        original[f"{FUSED}.lokr_w2_b"],
    )

    native, meta = build_standard_state_dict(dict(original), None, None, base_weights)

    assert meta is not None and "ss_export_note" in meta
    assert native and all(k.startswith("diffusion_model.") for k in native)
    assert not any(k.startswith("lora_unet_") for k in native)
    assert not any("qkv_proj" in k or "_kv_proj" in k for k in native)
    for letter in LETTERS:
        p = f"diffusion_model.blocks.0.self_attn.{letter}_proj"
        assert f"{p}.alpha" in native, p
    assert "diffusion_model.blocks.0.mlp.layer1.lora_down.weight" in native

    # exact split: per-component kron rows == the fused kron's rows
    if case != "case3":
        w1 = fused_w1.to(torch.float)
        w2 = fused_w2_a.to(torch.float) @ fused_w2_b.to(torch.float)
        kron_full = torch.kron(w1, w2)
        comp_dim = kron_full.shape[0] // 3
        for i, letter in enumerate(LETTERS):
            p, delta = _factors(native, letter)
            # case1/2 keep alpha (consumer scale == training scale)
            sl = slice(i * comp_dim, (i + 1) * comp_dim)
            assert torch.allclose(delta, kron_full[sl] * (4.0 / fused_w2_b.shape[0]), rtol=1e-4, atol=1e-4), p

    _assert_comfy_parity(native, base_weights, fused_w1, fused_w2_a, fused_w2_b, m_fused, tol)


@pytest.mark.parametrize("case", ["case1", "case2"])
def test_reassembly_roundtrip(case):
    g = torch.Generator().manual_seed(7)
    base_weights = _base_weights()
    original = _fused_sd(case)
    original.update(_plain_sd(g))
    native, _ = build_standard_state_dict(dict(original), None, None, base_weights)

    def fake_lora(name, w0, w1_shape=None, w2a_shape=None):
        extra = {}
        if w1_shape is not None:
            extra["lokr_w1"] = torch.zeros(*w1_shape)
            extra["lokr_w2_a"] = torch.zeros(*w2a_shape)
        return SimpleNamespace(
            lora_name=name, org_module_ref=(SimpleNamespace(weight=w0),), **extra
        )

    loras = [
        fake_lora(FUSED, base_weights[FUSED], (6, 2) if case == "case1" else (1, 1), (4, 2) if case == "case1" else (24, 3)),
        fake_lora(PLAIN, base_weights[PLAIN]),
    ]
    back = reassemble_comfyui_native_sd(native, loras)

    assert f"{FUSED}.lokr_w1" in back and f"{PLAIN}.lora_down.weight" in back
    assert torch.allclose(
        back[f"{FUSED}.lokr_w1"].to(torch.float), original[f"{FUSED}.lokr_w1"].to(torch.float), rtol=1e-4, atol=1e-4
    )
    assert torch.allclose(
        back[f"{FUSED}.lokr_w2_a"].to(torch.float), original[f"{FUSED}.lokr_w2_a"].to(torch.float), rtol=1e-4, atol=1e-4
    )
    assert back[f"{FUSED}.lokr_w2_b"].shape == original[f"{FUSED}.lokr_w2_b"].shape
    m_err = (
        (back[f"{FUSED}.dora_scale"].to(torch.float) - original[f"{FUSED}.dora_scale"].to(torch.float))
        .norm()
        / original[f"{FUSED}.dora_scale"].to(torch.float).norm()
    ).item()
    assert m_err < 1e-4, f"magnitude roundtrip error {m_err}"
    pm_err = (
        (back[f"{PLAIN}.dora_scale"].to(torch.float) - original[f"{PLAIN}.dora_scale"].to(torch.float))
        .norm()
        / original[f"{PLAIN}.dora_scale"].to(torch.float).norm()
    ).item()
    assert pm_err < 1e-4, f"plain magnitude roundtrip error {pm_err}"


def test_reassembly_refuses_svd_components():
    base_weights = _base_weights()
    original = _fused_sd("case3")
    native, _ = build_standard_state_dict(dict(original), None, None, base_weights)
    assert any("lora_A.weight" in k for k in native), "middle component should fall back to SVD"

    lora = SimpleNamespace(
        lora_name=FUSED,
        org_module_ref=(SimpleNamespace(weight=base_weights[FUSED]),),
        lokr_w1=torch.zeros(2, 2),
        lokr_w2_a=torch.zeros(12, 2),
    )
    with pytest.raises(ValueError, match="SVD"):
        reassemble_comfyui_native_sd(native, [lora])


def test_sidecar_gate_skips_native(tmp_path):
    p = tmp_path / "native.safetensors"
    save_file({"diffusion_model.blocks.0.self_attn.q_proj.lokr_w1": torch.zeros(1, 1).to(torch.bfloat16)}, str(p))
    assert export_comfyui_sidecar(str(p), str(tmp_path / "dit.safetensors")) is None
