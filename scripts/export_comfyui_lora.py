"""Export a trained LoKr / DoKr checkpoint to ComfyUI-native format.

The training save uses kohya ``lora_unet_*`` underscore keys with the fused
``self_attn_qkv_proj`` / ``cross_attn_kv_proj`` matrices, while ComfyUI's
anima DiT expects ``diffusion_model.*`` dotted keys with separated
q/k/v_proj — feeding the raw save to ComfyUI attaches only a fraction of the
patches and renders pure noise. This tool rewrites the file in three steps
(v3 semantics, validated module-by-module against ComfyUI 0.37.2's own
load path):

1. rewrite keys to ``diffusion_model.*`` dotted form (``blocks_N_`` →
   ``blocks.N.`` etc.);
2. split the fused matrices — kv side is a lossless kron row split of
   ``lokr_w1``; the qkv side is lossless too when ``out_k`` divides the
   per-component rows, otherwise each component gets a rank ≤
   ``--max_svd_rank`` (default 64) SVD approximation;
3. rescale the DoRA magnitude to the consumer norm convention
   (``m·‖W0‖/‖V‖``) and store it 2-D ``[out, 1]`` per the LyCORIS disk
   convention — a 1-D vector broadcasts into an ``[out, out]`` outer product
   inside ComfyUI's ``weight_decompose`` and silently corrupts square
   weights (the "pure noise" root cause). The rescale direction matters:
   ComfyUI's ``weight_decompose`` normalizes by the *original* weight norm
   (``weight.reshape(...).norm(...)`` — verified against
   ``comfy/weight_adapter/base.py``) while training normalizes by ``‖V‖``,
   so the exact consumer-side magnitude is ``m·‖W0‖/‖V‖``.

Scope: files whose modules carry ``lokr_w1`` (LoKr / DoKr). Plain
LoRA / DoRA saves are not converted by this tool.

Usage::

    python scripts/export_comfyui_lora.py --lora output/ckpt/<name>.safetensors
    python scripts/export_comfyui_lora.py --lora a.safetensors --dit b.safetensors --out c.safetensors

``--dit`` defaults to ``configs/base.toml``'s
``pretrained_model_name_or_path`` (the same base model the GUI Paths
section configures). Conversion runs on CPU; the run ends with a per-module
error report (median / worst / count over 2%) comparing the ComfyUI math
against the training semantics.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]

_FUSED_SEG = {"qkv": "self_attn.qkv_proj", "kv": "cross_attn.kv_proj"}


def default_dit_path() -> str:
    base = REPO / "configs" / "base.toml"
    if base.exists():
        with open(base, "rb") as f:
            cfg = tomllib.load(f)
        path = cfg.get("pretrained_model_name_or_path")
        if path and Path(path).exists():
            return str(path)
    raise SystemExit(
        "未找到底模：configs/base.toml 的 pretrained_model_name_or_path 为空或文件不存在，"
        "请用 --dit 显式指定 anima 底模（如 ComfyUI models/diffusion_models/anima_baseV10.safetensors）"
    )


def dotted(short: str) -> str:
    """kohya underscore module name → ComfyUI dotted parameter path."""
    stem = re.sub(r"blocks_(\d+)_", r"blocks.\1.", short)
    stem = re.sub(r"(adaln_modulation_[a-z_]+?)_(\d+)$", r"\1.\2", stem)
    stem = re.sub(r"(self_attn|cross_attn)_", r"\1.", stem)
    stem = re.sub(r"mlp_layer(\d+)$", r"mlp.layer\1", stem)
    return stem


def fused_kind(short: str) -> str:
    if "self_attn_qkv_proj" in short:
        return "qkv"
    if "cross_attn_kv_proj" in short:
        return "kv"
    return ""


def component_prefix(dstem: str, fused: str, comp: str) -> str:
    if fused == "qkv":
        return dstem.replace(_FUSED_SEG["qkv"], f"self_attn.{comp}_proj")
    if fused == "kv":
        return dstem.replace(_FUSED_SEG["kv"], f"cross_attn.{comp}_proj")
    return dstem


def base_weight(dit: safe_open, dstem: str, fused: str) -> torch.Tensor:
    """Training-side fused W0 (rows = q|k|v or k|v concatenated)."""
    try:
        if fused == "":
            return dit.get_tensor(f"net.{dstem}.weight").float()
        parts = {"qkv": ["q", "k", "v"], "kv": ["k", "v"]}[fused]
        tpl = dstem.replace(
            _FUSED_SEG[fused],
            ("self_attn.{c}_proj" if fused == "qkv" else "cross_attn.{c}_proj"),
        )
        return torch.cat([dit.get_tensor(f"net.{tpl.format(c=c)}.weight").float() for c in parts], dim=0)
    except SafetensorError as exc:
        raise ValueError(
            f"底模里找不到键 {exc.args[0]!r}——请确认 --dit 指向 anima 底模本体"
            "（anima_baseV10.safetensors），而不是别的模型"
        ) from exc


def convert(lora_path: str, dit_path: str, out_path: str, max_svd_rank: int = 64) -> dict:
    lora = safe_open(lora_path, framework="pt", device="cpu")
    dit = safe_open(dit_path, framework="pt", device="cpu")

    mods = sorted(set(k.rsplit(".", 1)[0] for k in lora.keys() if k.endswith(".lokr_w1")))
    if not mods:
        raise ValueError(
            "文件里没有 lokr_w1 键——本工具只支持 LoKr/DoKr 成丹；"
            "普通 LoRA/DoRA 不需要（也不经过）此转换"
        )

    out_sd: dict[str, torch.Tensor] = {}
    errors: list[tuple[float, str]] = []
    lora_keys = set(lora.keys())

    for mod in mods:
        for key in (f"{mod}.lokr_w2_a", f"{mod}.lokr_w2_b", f"{mod}.alpha"):
            if key not in lora_keys:
                raise ValueError(f"模块 {mod} 缺少 {key.rsplit('.', 1)[1]}——不是本丹炉保存的 LoKr/DoKr 文件")
        short = mod.replace("lora_unet_", "")
        fused = fused_kind(short)
        dstem = dotted(short)
        W0_full = base_weight(dit, dstem, fused)

        w1 = lora.get_tensor(f"{mod}.lokr_w1").float()
        w2_a = lora.get_tensor(f"{mod}.lokr_w2_a")
        w2_b = lora.get_tensor(f"{mod}.lokr_w2_b")
        w2 = w2_a.float() @ w2_b.float()
        alpha_t = lora.get_tensor(f"{mod}.alpha")
        scale = alpha_t.item() / w2_b.shape[0]
        has_dora = f"{mod}.dora_scale" in lora_keys
        m = lora.get_tensor(f"{mod}.dora_scale").float() if has_dora else None
        delta = torch.kron(w1, w2)
        if delta.shape != W0_full.shape:
            raise ValueError(f"{mod}: kron 增量 {tuple(delta.shape)} 与底模权重 {tuple(W0_full.shape)} 不匹配")
        V = W0_full + scale * delta
        normV = V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        W_train = V if m is None else (m.unsqueeze(1) / normV) * V

        n_comp = {"qkv": 3, "kv": 2, "": 1}[fused]
        letters = {"qkv": ["q", "k", "v"], "kv": ["k", "v"], "": [""]}[fused]
        comp_dim = W0_full.shape[0] // n_comp
        out_k = w2.shape[0]

        for i, comp in enumerate(letters):
            sl = slice(i * comp_dim, (i + 1) * comp_dim)
            W0_c = W0_full[sl]
            Wt = W_train[sl]
            pre = f"diffusion_model.{component_prefix(dstem, fused, comp)}"
            if fused == "" or comp_dim % out_k == 0:
                if fused == "":
                    w1_c = w1
                else:
                    rows = comp_dim // out_k
                    w1_c = w1[i * rows : (i + 1) * rows, :]
                out_sd[f"{pre}.lokr_w1"] = w1_c.contiguous().to(torch.bfloat16)
                out_sd[f"{pre}.lokr_w2_a"] = w2_a.clone()
                out_sd[f"{pre}.lokr_w2_b"] = w2_b.clone()
                out_sd[f"{pre}.alpha"] = alpha_t.clone()
            else:
                delta_c = (scale * delta)[sl, :]
                U, S, Vh = torch.linalg.svd(delta_c, full_matrices=False)
                energy = (S**2).cumsum(0) / (S**2).sum()
                r = min(int(torch.searchsorted(energy, torch.tensor(0.999)).item()) + 1, max_svd_rank)
                sqrtS = S[:r].sqrt()
                down = (Vh[:r, :] * sqrtS.unsqueeze(1)).contiguous()
                up = (U[:, :r] * sqrtS.unsqueeze(0)).contiguous()
                out_sd[f"{pre}.lora_A.weight"] = down.to(torch.bfloat16)
                out_sd[f"{pre}.lora_B.weight"] = up.to(torch.bfloat16)
                out_sd[f"{pre}.alpha"] = torch.tensor(float(r))
            if m is not None:
                V_c = W0_c + (delta * scale)[sl, :]
                normV_c = V_c.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                normW0 = W0_c.norm(p=2, dim=1, keepdim=True) + 1e-8
                m_c = m[sl]
                # ComfyUI divides by ‖W0‖ while training divides by ‖V‖ → the
                # consumer-side magnitude is m·‖W0‖/‖V‖ (NOT the reciprocal —
                # that only cancels when ‖V‖≈‖W0‖ and leaves a 2ε row-scale
                # error; verified against comfy/weight_adapter/base.py).
                m_fix = (m_c.unsqueeze(1) * normW0 / normV_c).squeeze(1)
                # LyCORIS disk convention: 2-D [out, 1]; a 1-D vector broadcasts
                # into [out, out] inside ComfyUI's weight_decompose.
                out_sd[f"{pre}.dora_scale"] = m_fix.unsqueeze(1).contiguous().to(torch.bfloat16)

            # 误差报告从落盘（bf16 量化后）的张量反算 ComfyUI 会算出的结果，
            # 衡量的是文件最终保真度而非中间量。
            if f"{pre}.lokr_w1" in out_sd:
                w1_s = out_sd[f"{pre}.lokr_w1"].float()
                w2_s = out_sd[f"{pre}.lokr_w2_a"].float() @ out_sd[f"{pre}.lokr_w2_b"].float()
                alpha_s = float(out_sd[f"{pre}.alpha"].item())
                delta_s = torch.kron(w1_s, w2_s) * (alpha_s / out_sd[f"{pre}.lokr_w2_b"].shape[0])
            else:
                a_s = out_sd[f"{pre}.lora_A.weight"].float()
                b_s = out_sd[f"{pre}.lora_B.weight"].float()
                delta_s = (b_s @ a_s) * (float(out_sd[f"{pre}.alpha"].item()) / a_s.shape[0])
            if m is None:
                W_comfy = W0_c + delta_s
            else:
                d_s = out_sd[f"{pre}.dora_scale"].float()
                weight_norm = W0_c.norm(p=2, dim=1, keepdim=True) + torch.finfo(torch.float32).eps
                W_comfy = (W0_c + delta_s) * (d_s / weight_norm)
            err = ((W_comfy - Wt).norm() / W0_c.norm()).item()
            errors.append((err, f"{short}{('.' + comp) if comp else ''}"))

    meta = dict(lora.metadata() or {})
    meta["modelspec.title"] = f"{Path(lora_path).stem} (comfyui-native)"
    meta["ss_export_note"] = (
        "diffusion_model.* keys; fused qkv/kv defused; dora_scale [out,1] 2D "
        "(LyCORIS convention) rescaled to consumer norm convention"
    )
    save_file(out_sd, out_path, meta)

    errs = sorted(e for e, _ in errors)
    worst = max(errors)
    return {
        "modules": len(mods),
        "tensors": len(out_sd),
        "median_err": errs[len(errs) // 2],
        "worst_err": worst[0],
        "worst_name": worst[1],
        "over_2pct": sum(1 for e in errs if e > 0.02),
        "out_path": out_path,
    }


def main(argv: list[str] | None = None) -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="把本丹炉的 LoKr/DoKr 成丹转成 ComfyUI 原生格式")
    parser.add_argument("--lora", required=True, help="训练产出的 LoKr/DoKr safetensors 路径")
    parser.add_argument("--dit", default=None, help="anima 底模路径（缺省读 configs/base.toml）")
    parser.add_argument("--out", default=None, help="输出路径（缺省 <lora>_comfyui.safetensors）")
    parser.add_argument("--max_svd_rank", type=int, default=64, help="qkv 不可整除时 SVD 近似的最大秩（默认 64）")
    args = parser.parse_args(argv)

    dit_path = args.dit or default_dit_path()
    out_path = args.out or str(Path(args.lora).with_name(Path(args.lora).stem + "_comfyui.safetensors"))
    stats = convert(args.lora, dit_path, out_path, args.max_svd_rank)

    size_mb = Path(stats["out_path"]).stat().st_size / 1e6
    print(f"已写出: {stats['out_path']}  {size_mb:.1f} MB  {stats['tensors']} 键 / {stats['modules']} 模块")
    print(f"误差报告（ComfyUI 数学 vs 训练语义）: 中位 {stats['median_err']:.5f}  最差 {stats['worst_err']:.4f} ({stats['worst_name']})")
    print(f"超 2% 模块数: {stats['over_2pct']}/{stats['modules']}  ——  为 0 即可放心在 ComfyUI 使用")


if __name__ == "__main__":
    main()
