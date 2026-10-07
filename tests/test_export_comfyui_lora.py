"""scripts/export_comfyui_lora.py — synthetic mini-DiT round-trip tests.

Builds a tiny anima-shaped base model (separated q/k/v_proj) plus a fake
in-repo LoKr/DoKr save (kohya ``lora_unet_*`` keys, fused qkv/kv modules)
and checks the exported file against both conventions, replicating the real
ComfyUI 0.37.2 consumer math (``comfy/weight_adapter/base.py::weight_decompose``
normalizes by the ORIGINAL weight norm; ``lokr.py`` scales ``kron(w1, w2)``
by ``alpha / w2_b.shape[0]``).

The fixture uses scale ≈ 1 so ‖V‖ drifts far from ‖W0‖ — this pins the DoRA
magnitude rescale DIRECTION (``m·‖W0‖/‖V‖``, not the reciprocal): an inverted
sign of the exponent produces percent-level row-scale errors that fail the
parity thresholds.
"""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.export_comfyui_lora import convert, dotted, fused_kind

# module short name → (fused kind, out_k, scale, has_dora)
MODULES = {
    "blocks_0_self_attn_qkv_proj": ("qkv", 4, 1.0, True),  # divisible → kron row split
    "blocks_0_cross_attn_kv_proj": ("kv", 4, 1.0, True),  # divisible
    "blocks_0_mlp_layer1": ("", 4, 1.0, True),  # plain, non-square module absent — square here
    "blocks_0_mlp_layer2": ("", 4, 1.0, False),  # pure LoKr (no dora_scale)
    "blocks_0_adaln_modulation_mlp_2": ("", 4, 1.0, True),  # non-square [12, 8]
    "blocks_1_self_attn_qkv_proj": ("qkv", 3, 0.25, True),  # 8 % 3 → SVD fallback
    "blocks_1_cross_attn_kv_proj": ("kv", 4, 0.25, True),  # 6 % 4 → SVD fallback (kv side!)
}

_LOKR_SHAPES = {
    # out_k=4 family: kron(w1, w2) builds the fused delta
    "blocks_0_self_attn_qkv_proj": ((6, 2), (4, 2), (2, 4)),
    "blocks_0_cross_attn_kv_proj": ((4, 2), (4, 2), (2, 4)),
    "blocks_0_mlp_layer1": ((2, 2), (4, 2), (2, 4)),
    "blocks_0_mlp_layer2": ((2, 2), (4, 2), (2, 4)),
    "blocks_0_adaln_modulation_mlp_2": ((3, 2), (4, 2), (2, 4)),
    "blocks_1_self_attn_qkv_proj": ((8, 8), (3, 2), (2, 1)),
    "blocks_1_cross_attn_kv_proj": ((3, 4), (4, 2), (2, 2)),
}


def _build_world(tmp_path):
    g = torch.Generator().manual_seed(7)

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    dit_sd = {
        "net.blocks.0.self_attn.q_proj.weight": rnd(8, 8),
        "net.blocks.0.self_attn.k_proj.weight": rnd(8, 8),
        "net.blocks.0.self_attn.v_proj.weight": rnd(8, 8),
        "net.blocks.0.cross_attn.k_proj.weight": rnd(8, 8),
        "net.blocks.0.cross_attn.v_proj.weight": rnd(8, 8),
        "net.blocks.0.mlp.layer1.weight": rnd(8, 8),
        "net.blocks.0.mlp.layer2.weight": rnd(8, 8),
        "net.blocks.0.adaln_modulation_mlp.2.weight": rnd(12, 8),
        "net.blocks.1.self_attn.q_proj.weight": rnd(8, 8),
        "net.blocks.1.self_attn.k_proj.weight": rnd(8, 8),
        "net.blocks.1.self_attn.v_proj.weight": rnd(8, 8),
        "net.blocks.1.cross_attn.k_proj.weight": rnd(6, 8),
        "net.blocks.1.cross_attn.v_proj.weight": rnd(6, 8),
    }

    def fused_w0(short):
        kind = MODULES[short][0]
        d = dotted(short)
        if kind == "qkv":
            parts = ["q", "k", "v"]
            tpl = d.replace("self_attn.qkv_proj", "self_attn.{c}_proj")
        else:
            parts = ["k", "v"]
            tpl = d.replace("cross_attn.kv_proj", "cross_attn.{c}_proj")
        return torch.cat([dit_sd[f"net.{tpl.format(c=c)}.weight"] for c in parts], dim=0)

    lora_sd: dict[str, torch.Tensor] = {}
    mags = {}
    for short, (kind, out_k, scale, has_dora) in MODULES.items():
        w1_shape, w2a_shape, w2b_shape = _LOKR_SHAPES[short]
        pre = f"lora_unet_{short}"
        lora_sd[f"{pre}.lokr_w1"] = rnd(*w1_shape).to(torch.bfloat16)
        lora_sd[f"{pre}.lokr_w2_a"] = rnd(*w2a_shape).to(torch.bfloat16)
        lora_sd[f"{pre}.lokr_w2_b"] = rnd(*w2b_shape).to(torch.bfloat16)
        lora_sd[f"{pre}.alpha"] = torch.tensor(out_k * scale)
        if has_dora:
            w0 = fused_w0(short) if kind else dit_sd[f"net.{dotted(short)}.weight"]
            mags[short] = w0.norm(dim=1) * (1.0 + 0.3 * rnd(w0.shape[0]))  # ~‖W0‖ rows
            lora_sd[f"{pre}.dora_scale"] = mags[short].to(torch.bfloat16)

    dit_path = str(tmp_path / "dit.safetensors")
    lora_path = str(tmp_path / "test_lora.safetensors")
    out_path = str(tmp_path / "out.safetensors")
    save_file(dit_sd, dit_path)
    save_file(lora_sd, lora_path)
    convert(lora_path, dit_path, out_path)
    return dit_sd, lora_sd, out_path


def _comfy_weight(dora_scale, weight, lora_diff, alpha):
    """Replica of comfy/weight_adapter/base.py::weight_decompose (fp32)."""
    lora_diff = lora_diff * alpha
    if dora_scale is None:
        return weight + lora_diff  # non-dora branch of lokr.py calculate_weight
    weight_calc = weight + lora_diff
    weight_norm = (
        weight.reshape(weight.shape[0], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
    )
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps
    return weight_calc * (dora_scale / weight_norm)


def test_lokr_dokr_export_roundtrip(tmp_path):
    dit_sd, lora_sd, out_path = _build_world(tmp_path)
    out = safe_open(out_path, framework="pt", device="cpu")
    keys = set(out.keys())

    assert keys, "export produced no tensors"
    assert all(k.startswith("diffusion_model.") for k in keys)
    assert not any("lora_unet_" in k for k in keys)

    for short, (kind, out_k, scale, has_dora) in MODULES.items():
        d = dotted(short)
        letters = {"qkv": ["q", "k", "v"], "kv": ["k", "v"], "": [""]}[kind]
        if kind == "qkv":
            comp_dim = dit_sd["net.blocks.0.self_attn.q_proj.weight"].shape[0]
        elif kind == "kv":
            comp_dim = dit_sd[
                "net.blocks.1.cross_attn.k_proj.weight" if "blocks_1" in short else "net.blocks.0.cross_attn.k_proj.weight"
            ].shape[0]
        else:
            comp_dim = dit_sd[f"net.{d}.weight"].shape[0]

        # reference training semantics from the INPUT file
        pre_in = f"lora_unet_{short}"
        w1 = lora_sd[f"{pre_in}.lokr_w1"].float()
        w2 = lora_sd[f"{pre_in}.lokr_w2_a"].float() @ lora_sd[f"{pre_in}.lokr_w2_b"].float()
        s = lora_sd[f"{pre_in}.alpha"].item() / lora_sd[f"{pre_in}.lokr_w2_b"].shape[0]
        if kind == "":
            W0_full = dit_sd[f"net.{d}.weight"].float()
        else:
            W0_full = torch.cat(
                [
                    dit_sd[
                        f"net.{d.replace('self_attn.qkv_proj', f'self_attn.{c}_proj').replace('cross_attn.kv_proj', f'cross_attn.{c}_proj')}.weight"
                    ].float()
                    for c in letters
                ],
                dim=0,
            )
        V = W0_full + s * torch.kron(w1, w2)
        if has_dora:
            m = lora_sd[f"{pre_in}.dora_scale"].float()
            W_train = (m.unsqueeze(1) / V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)) * V
        else:
            W_train = V

        for i, comp in enumerate(letters):
            sl = slice(i * comp_dim, (i + 1) * comp_dim)
            W0_c = W0_full[sl]
            pre_out = f"diffusion_model.{d}"
            if kind != "":
                pre_out = pre_out.replace("self_attn.qkv_proj", f"self_attn.{comp}_proj").replace(
                    "cross_attn.kv_proj", f"cross_attn.{comp}_proj"
                )
            if has_dora:
                ds = out.get_tensor(f"{pre_out}.dora_scale")
                assert ds.dim() == 2 and ds.shape == (W0_c.shape[0], 1), (pre_out, ds.shape)

            # what ComfyUI computes from the exported file
            if f"{pre_out}.lokr_w1" in keys:
                w1_s = out.get_tensor(f"{pre_out}.lokr_w1").float()
                w2_s = out.get_tensor(f"{pre_out}.lokr_w2_a").float() @ out.get_tensor(
                    f"{pre_out}.lokr_w2_b"
                ).float()
                alpha = out.get_tensor(f"{pre_out}.alpha").item() / out.get_tensor(f"{pre_out}.lokr_w2_b").shape[0]
                W_comfy = _comfy_weight(
                    out.get_tensor(f"{pre_out}.dora_scale").float() if has_dora else None,
                    W0_c,
                    torch.kron(w1_s, w2_s),
                    alpha,
                )
            else:
                a_s = out.get_tensor(f"{pre_out}.lora_A.weight").float()
                b_s = out.get_tensor(f"{pre_out}.lora_B.weight").float()
                alpha = out.get_tensor(f"{pre_out}.alpha").item() / a_s.shape[0]
                assert has_dora, "SVD fallback only expected on DoKr modules here"
                W_comfy = _comfy_weight(out.get_tensor(f"{pre_out}.dora_scale").float(), W0_c, b_s @ a_s, alpha)

            err = ((W_comfy - W_train[sl]).norm() / W0_c.norm()).item()
            # divisible splits are lossless → tight bound; the SVD fallback
            # truncates energy at 0.999 (plus bf16) → percent-level is expected
            limit = 0.02 if (kind == "" or comp_dim % out_k == 0) else 0.05
            assert err < limit, f"{pre_out}: ComfyUI-vs-training error {err:.4f}"

        # fixture discrimination guard: with scale ≈ 1 the row norms of V must
        # drift well away from W0's — otherwise this test could not tell a
        # rescale-direction bug from bf16 noise.
        if has_dora and scale == 1.0:
            drift = (V.norm(dim=1) / W0_full.norm(dim=1) - 1).abs().max().item()
            assert drift > 0.05, f"{short}: fixture drift too small ({drift:.4f}) to pin the rescale direction"


def test_rejects_plain_lora(tmp_path):
    p = tmp_path / "plain.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_mlp_layer1.lora_down.weight": torch.zeros(2, 8).to(torch.bfloat16),
            "lora_unet_blocks_0_mlp_layer1.lora_up.weight": torch.zeros(8, 2).to(torch.bfloat16),
            "lora_unet_blocks_0_mlp_layer1.alpha": torch.tensor(4.0),
        },
        str(p),
    )
    with pytest.raises(ValueError, match="lokr_w1"):
        convert(str(p), str(p), str(tmp_path / "out.safetensors"))


def test_rejects_incomplete_module(tmp_path):
    p = tmp_path / "broken.safetensors"
    save_file({"lora_unet_blocks_0_mlp_layer1.lokr_w1": torch.zeros(2, 2).to(torch.bfloat16)}, str(p))
    with pytest.raises(ValueError, match="lokr_w2_a"):
        convert(str(p), str(p), str(tmp_path / "out.safetensors"))


def test_unrelated_dit_raises_friendly_error(tmp_path):
    g = torch.Generator().manual_seed(3)
    dit_p = tmp_path / "not_anima.safetensors"
    save_file({"some_other_model.weight": torch.randn(4, 4, generator=g)}, str(dit_p))
    lora_p = tmp_path / "l.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_mlp_layer1.lokr_w1": torch.randn(2, 2, generator=g).to(torch.bfloat16),
            "lora_unet_blocks_0_mlp_layer1.lokr_w2_a": torch.randn(4, 2, generator=g).to(torch.bfloat16),
            "lora_unet_blocks_0_mlp_layer1.lokr_w2_b": torch.randn(2, 4, generator=g).to(torch.bfloat16),
            "lora_unet_blocks_0_mlp_layer1.alpha": torch.tensor(4.0),
        },
        str(lora_p),
    )
    with pytest.raises(ValueError, match="anima"):
        convert(str(lora_p), str(dit_p), str(tmp_path / "out.safetensors"))


def test_key_helpers():
    assert fused_kind("blocks_0_self_attn_qkv_proj") == "qkv"
    assert fused_kind("blocks_0_cross_attn_kv_proj") == "kv"
    assert fused_kind("blocks_0_mlp_layer1") == ""
    assert dotted("blocks_0_adaln_modulation_mlp_2") == "blocks.0.adaln_modulation_mlp.2"
    assert dotted("blocks_3_mlp_layer2") == "blocks.3.mlp.layer2"
    assert dotted("final_layer") == "final_layer"
