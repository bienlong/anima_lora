"""networks/lora_save.export_comfyui_sidecar — training-end auto-export hook.

Pins: LoKr/DoKr files get a ComfyUI-native twin next to the native save;
plain-LoRA files are skipped (already native); the env kill-switch works;
every failure is swallowed (training must never die at the last step).
"""

from __future__ import annotations

import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from networks.lora_save import export_comfyui_sidecar


def _tiny_world(tmp_path):
    g = torch.Generator().manual_seed(11)

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    dit = tmp_path / "dit.safetensors"
    save_file({"net.blocks.0.mlp.layer1.weight": rnd(8, 8)}, str(dit))
    lora = tmp_path / "dokr.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_mlp_layer1.lokr_w1": rnd(2, 2).to(torch.bfloat16),
            "lora_unet_blocks_0_mlp_layer1.lokr_w2_a": rnd(4, 2).to(torch.bfloat16),
            "lora_unet_blocks_0_mlp_layer1.lokr_w2_b": rnd(2, 4).to(torch.bfloat16),
            "lora_unet_blocks_0_mlp_layer1.alpha": torch.tensor(4.0),
            "lora_unet_blocks_0_mlp_layer1.dora_scale": rnd(8).to(torch.bfloat16),
        },
        str(lora),
    )
    return str(dit), str(lora)


def test_sidecar_written_for_lokr(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIMA_COMFYUI_EXPORT", raising=False)
    dit, lora = _tiny_world(tmp_path)
    out = export_comfyui_sidecar(lora, dit)
    assert out is not None and os.path.exists(out)
    keys = safe_open(out, framework="pt", device="cpu").keys()
    assert any(k.startswith("diffusion_model.") for k in keys)
    assert any(k.endswith("dora_scale") for k in keys)


def test_sidecar_skips_plain_lora(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIMA_COMFYUI_EXPORT", raising=False)
    p = tmp_path / "plain.safetensors"
    save_file(
        {
            "lora_unet_x.lora_down.weight": torch.zeros(2, 8).to(torch.bfloat16),
            "lora_unet_x.lora_up.weight": torch.zeros(8, 2).to(torch.bfloat16),
            "lora_unet_x.alpha": torch.tensor(4.0),
        },
        str(p),
    )
    assert export_comfyui_sidecar(str(p), str(tmp_path / "dit.safetensors")) is None


def test_sidecar_env_off(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_COMFYUI_EXPORT", "0")
    dit, lora = _tiny_world(tmp_path)
    assert export_comfyui_sidecar(lora, dit) is None


def test_sidecar_swallows_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIMA_COMFYUI_EXPORT", raising=False)
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not safetensors")
    assert export_comfyui_sidecar(str(bad), str(tmp_path / "dit.safetensors")) is None
