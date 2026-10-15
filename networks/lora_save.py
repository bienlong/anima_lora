"""Save-pipeline orchestrator for the LoRA / Hydra family.

The per-variant save logic — MoE write layout, qkv defuse — lives on the
variant's module class in ``networks/lora_modules/``
(``HydraLoRAModule.build_moe_state_dict``, ``lora.defuse_and_bake_standard``).
This file is the thin layer that calls them and writes the resulting file.

The standard write path relays adaln keys from the runtime names to the
ComfyUI layout (``_relayout_adaln_to_comfy``), after the qkv defuse and
before hashing — so the shipped file is ComfyUI-native end to end. The Hydra
MoE variant returns early and is not ComfyUI-loadable regardless.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional

import torch

from library.log import setup_logging
from networks.attn_fuse import ATTN_FUSE_SPECS, match_fused_spec
from networks.lora_modules import HydraLoRAModule
from networks.lora_modules.lora import defuse_and_bake_standard

setup_logging()
logger = logging.getLogger(__name__)


def _relayout_adaln_to_comfy(
    state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Rename adaln LoRA keys from the in-repo runtime names
    (``adaln_up_{br}``) to the ComfyUI state-dict layout
    (``adaln_modulation_{br}_2``) and stamp ``ss_adaln_layout`` — see the
    layout note in ``networks/lora_utils.py``. The attn/MLP keys already ship
    in the defused split layout, so only the adaln keys move.

    Presence-gated — an adaln-less checkpoint is untouched, metadata and
    all. Mutates ``state_dict`` in place; returns the metadata to write
    (a dict is allocated if the stamp needs one and none was passed).
    """
    from networks.lora_utils import relayout_adaln_runtime_to_comfy

    renamed = relayout_adaln_runtime_to_comfy(state_dict)
    if renamed.keys() == state_dict.keys():
        return metadata  # no runtime adaln keys present — nothing to relayout

    state_dict.clear()
    state_dict.update(renamed)
    if metadata is None:
        metadata = {}
    metadata["ss_adaln_layout"] = "comfy"
    n_adaln = sum(
        1 for k in renamed if "adaln_modulation_" in k and k.endswith(".alpha")
    )
    logger.info(
        f"relaid {n_adaln} adaln modules to the ComfyUI layout "
        "(loads natively in ComfyUI; in-repo loader renames back on load)"
    )
    return metadata


def build_standard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
    base_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    """Run the standard finalize chain WITHOUT writing a file.

    Defuse fused qkv + bake channel scaling, relay adaln keys to the ComfyUI
    layout, cast dtype. Returns the finalized ``(state_dict, metadata)``. Factored
    out of :func:`save_network_weights` so the dual-pool turbo save can finalize
    each pool to its on-disk plain-LoRA layout and concat the two exactly.

    ``base_weights`` (lora_name → org Linear weight) unlocks the native
    emission: exact LoKr qkv/kv split + DoRA magnitude rescale + full
    ``diffusion_model.*`` reprefix — the file is then ComfyUI-native with NO
    post-conversion step. Without it the legacy layout is kept (old behavior).
    """
    defuse_and_bake_standard(state_dict)
    native_ok = False
    if base_weights:
        # Native math runs on CPU: the training state_dict lives on GPU while
        # base weights are CPU, and the artifact ends up on disk anyway — the
        # factor tensors are only a few MB.
        work = {k: v.detach().to("cpu") for k, v in state_dict.items()}
        try:
            _split_fused_lokr(work)
            _rescale_dora_native(work, base_weights)
        except Exception:  # noqa: BLE001 — native is an upgrade, never a hard dependency
            logger.warning(
                "native ComfyUI layout emission failed; falling back to the "
                "legacy layout (the training-end sidecar still covers it)",
                exc_info=True,
            )
        else:
            state_dict.clear()
            state_dict.update(work)
            native_ok = True
    metadata = _relayout_adaln_to_comfy(state_dict, metadata)
    if native_ok:
        _reprefix_to_diffusion_model(state_dict)
        metadata = dict(metadata or {})
        metadata["ss_export_note"] = (
            "diffusion_model.* keys; fused qkv/kv defused; dora_scale [out,1] "
            "rescaled m·‖W0‖/‖V‖ — native at save time, no post-conversion"
        )
    if dtype is not None:
        for key in list(state_dict.keys()):
            state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)
    return state_dict, metadata


def _split_fused_lokr(state_dict: Dict[str, torch.Tensor]) -> None:
    """Exact per-component split of fused qkv/kv LoKr modules (runtime names).

    kron(w1, w2) row ``R`` is ``w1[R // c] ⊗ w2[R % c]`` (``c`` = w2 rows), so a
    fused module splits exactly when every component's row range either

    * case 1 — covers whole ``w1`` row blocks (``comp_dim % c == 0``):
      slice ``w1`` rows, share ``w2``;
    * case 2 — stays inside a single ``w1`` row (e.g. the degenerate
      ``lokr_factor = 0`` where ``w1`` is 1×1): keep that ``w1`` row, slice
      ``w2_a`` rows (``w2_b``/``alpha`` unchanged — the consumer scale is
      ``alpha / w2_b.shape[0]``, unaffected by the row slice);

    and only otherwise falls back to a per-component SVD (rank ≤ 64, loud
    warning) — for sane configurations that path never fires. Cases 1/2 are
    verified element-exact against the fused kron before emission.
    """
    for key in [k for k in state_dict if k.endswith(".lokr_w1")]:
        prefix = key[: -len(".lokr_w1")]
        spec = match_fused_spec(prefix)
        if spec is None:
            continue
        letters = spec.component_letters
        n = len(letters)
        w1 = state_dict.pop(f"{prefix}.lokr_w1").to(torch.float)
        use_w2 = f"{prefix}.lokr_w2" in state_dict
        if use_w2:
            w2 = state_dict.pop(f"{prefix}.lokr_w2")
            w2_a = None
            w2_b = None
            c = w2.shape[0]
            kron_full = torch.kron(w1, w2.to(torch.float))
        else:
            w2_a = state_dict.pop(f"{prefix}.lokr_w2_a")
            w2_b = state_dict.pop(f"{prefix}.lokr_w2_b")
            c = w2_a.shape[0]
            kron_full = torch.kron(w1, w2_a.to(torch.float) @ w2_b.to(torch.float))
        alpha = state_dict.pop(f"{prefix}.alpha")
        m = state_dict.pop(f"{prefix}.dora_scale", None)

        a_rows = w1.shape[0]
        fused_out = a_rows * c
        comp_dim = fused_out // n
        if comp_dim * n != fused_out:
            raise ValueError(f"{prefix}: fused out {fused_out} not divisible by {n}")
        base = prefix.removesuffix(spec.fused_frag)

        for i, letter in enumerate(letters):
            sl = slice(i * comp_dim, (i + 1) * comp_dim)
            new_prefix = f"{base}{spec.component_frag(letter)}"
            target = kron_full[sl]
            w1_i: Optional[torch.Tensor] = None
            w2_a_i: Optional[torch.Tensor] = None
            w2_i: Optional[torch.Tensor] = None
            if comp_dim % c == 0:  # case 1: whole w1 row blocks per component
                r = comp_dim // c
                w1_i = w1[i * r : (i + 1) * r].contiguous()
                if use_w2:
                    w2_i = w2
                else:
                    w2_a_i = w2_a
            else:
                b0 = (i * comp_dim) // c
                b1 = ((i + 1) * comp_dim - 1) // c
                if b0 == b1:  # case 2: component inside a single w1 row
                    w1_i = w1[b0 : b0 + 1].contiguous()
                    lo = i * comp_dim - b0 * c
                    if use_w2:
                        w2_i = w2[lo : lo + comp_dim].contiguous()
                    else:
                        w2_a_i = w2_a[lo : lo + comp_dim].contiguous()
            if w1_i is not None:
                if use_w2:
                    recon = torch.kron(w1_i, w2_i.to(torch.float))
                else:
                    recon = torch.kron(
                        w1_i, w2_a_i.to(torch.float) @ w2_b.to(torch.float)
                    )
                if not torch.allclose(recon, target, rtol=1e-5, atol=1e-5):
                    raise ValueError(f"{new_prefix}: exact split verification failed")
                state_dict[f"{new_prefix}.lokr_w1"] = w1_i
                if use_w2:
                    state_dict[f"{new_prefix}.lokr_w2"] = w2_i.clone()
                else:
                    state_dict[f"{new_prefix}.lokr_w2_a"] = w2_a_i.clone()
                    state_dict[f"{new_prefix}.lokr_w2_b"] = w2_b.clone()
                state_dict[f"{new_prefix}.alpha"] = alpha.clone()
            else:  # case 3: SVD approximation — of the SCALED rows (the consumer
                # applies alpha/r = 1, so the training scale must be baked in,
                # matching scripts/export_comfyui_lora.py's delta_c)
                scale = 1.0 if use_w2 else (float(alpha.item()) / w2_b.shape[0])
                target_scaled = scale * target
                U, S, Vh = torch.linalg.svd(target_scaled, full_matrices=False)
                energy = (S**2).cumsum(0) / (S**2).sum()
                r = min(
                    int(
                        torch.searchsorted(
                            energy, torch.tensor(0.999, device=energy.device)
                        ).item()
                    )
                    + 1,
                    64,
                )
                sqrt_s = S[:r].sqrt()
                a_down = (Vh[:r, :] * sqrt_s.unsqueeze(1)).contiguous()
                b_up = (U[:, :r] * sqrt_s.unsqueeze(0)).contiguous()
                state_dict[f"{new_prefix}.lora_A.weight"] = a_down
                state_dict[f"{new_prefix}.lora_B.weight"] = b_up
                state_dict[f"{new_prefix}.alpha"] = torch.tensor(float(r))
                err = (
                    (b_up @ a_down - target_scaled).norm() / target_scaled.norm()
                ).item()
                logger.warning(
                    f"{new_prefix}: qkv defuse fell back to SVD rank {r} "
                    f"({err:.4f} rel err) — consider a lokr_factor whose w1 rows "
                    "divide evenly across q/k/v for a lossless split"
                )
            if m is not None:
                state_dict[f"{new_prefix}.dora_scale"] = m[sl].clone()


def _w0_for_prefix(
    prefix: str, base_weights: Dict[str, torch.Tensor]
) -> Optional[torch.Tensor]:
    """Org weight for a (possibly post-split component) prefix.

    base_weights is keyed by the fused runtime lora_name; a split component
    prefix (``…self_attn_q_proj``) resolves to its fused module's weight sliced
    to that component's row range.
    """
    if prefix in base_weights:
        return base_weights[prefix]
    for spec in ATTN_FUSE_SPECS:
        for idx, letter in enumerate(spec.component_letters):
            frag = spec.component_frag(letter)
            if prefix.endswith(frag):
                fused = prefix[: -len(frag)] + spec.fused_frag
                if fused in base_weights:
                    w0 = base_weights[fused]
                    comp_dim = w0.shape[0] // len(spec.component_letters)
                    return w0[idx * comp_dim : (idx + 1) * comp_dim]
    return None


def _rescale_dora_native(
    state_dict: Dict[str, torch.Tensor],
    base_weights: Dict[str, torch.Tensor],
) -> None:
    """DoRA magnitude → the ComfyUI ``weight_decompose`` convention, in place.

    Training normalizes rows by ``‖V‖`` (``V = W0 + s·Δ``); ComfyUI divides by
    the ORIGINAL weight norm (``weight_decompose`` norms ``weight``, i.e. W0 —
    verified against comfy/weight_adapter/base.py). The exact consumer-side
    magnitude is therefore ``m' = m·‖W0‖/‖V‖``, stored 2-D ``[out, 1]`` per the
    LyCORIS disk convention (1-D broadcasts into an ``[out, out]`` outer
    product inside ComfyUI and silently corrupts square weights).
    """
    for key in [k for k in state_dict if k.endswith(".dora_scale")]:
        prefix = key[: -len(".dora_scale")]
        w0 = _w0_for_prefix(prefix, base_weights)
        if w0 is None:
            raise ValueError(
                f"dora module {prefix} has no base weight for the native rescale"
            )
        w0 = w0.to(torch.float)
        m = state_dict[key].to(torch.float)
        if f"{prefix}.lokr_w1" in state_dict:
            w1 = state_dict[f"{prefix}.lokr_w1"].to(torch.float)
            if f"{prefix}.lokr_w2" in state_dict:
                w2 = state_dict[f"{prefix}.lokr_w2"].to(torch.float)
                s = 1.0
            else:
                w2 = state_dict[f"{prefix}.lokr_w2_a"].to(torch.float) @ state_dict[
                    f"{prefix}.lokr_w2_b"
                ].to(torch.float)
                s = (
                    float(state_dict[f"{prefix}.alpha"].item())
                    / state_dict[f"{prefix}.lokr_w2_b"].shape[0]
                )
            delta = torch.kron(w1, w2) * s
        elif f"{prefix}.lora_A.weight" in state_dict:
            a_down = state_dict[f"{prefix}.lora_A.weight"].to(torch.float)
            b_up = state_dict[f"{prefix}.lora_B.weight"].to(torch.float)
            s = float(state_dict[f"{prefix}.alpha"].item()) / a_down.shape[0]
            delta = (b_up @ a_down) * s
        elif f"{prefix}.lora_down.weight" in state_dict:
            down = state_dict[f"{prefix}.lora_down.weight"].to(torch.float)
            up = state_dict[f"{prefix}.lora_up.weight"].to(torch.float)
            s = float(state_dict[f"{prefix}.alpha"].item()) / down.shape[0]
            delta = (up @ down) * s
        else:
            raise ValueError(f"{prefix}: dora_scale without recognizable factors")
        norm_v = (w0 + delta).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        norm_w0 = w0.norm(p=2, dim=1, keepdim=True) + 1e-8
        state_dict[key] = (m.unsqueeze(1) * norm_w0 / norm_v).contiguous()


def _reprefix_to_diffusion_model(state_dict: Dict[str, torch.Tensor]) -> None:
    """``lora_unet_*`` → ``diffusion_model.*`` dotted ComfyUI keys, in place."""
    from networks.lora_utils import diffusion_model_key

    renamed = {
        diffusion_model_key(k): state_dict.pop(k)
        for k in [k for k in state_dict if k.startswith("lora_unet_")]
    }
    state_dict.update(renamed)


def _comfy_module_name(lora_name: str) -> str:
    """Module-name-level adaln runtime→comfy rename (see lora_utils relayout)."""
    m = re.match(
        r"^(lora_unet_blocks_\d+_)adaln_up_(self_attn|cross_attn|mlp)$", lora_name
    )
    return f"{m.group(1)}adaln_modulation_{m.group(2)}_2" if m else lora_name


def _inverse_dora_rescale(
    m2d: torch.Tensor, w0: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Exact inverse of :func:`_rescale_dora_native` → 1-D training magnitude."""
    norm_v = (w0 + delta).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    norm_w0 = w0.norm(p=2, dim=1, keepdim=True) + 1e-8
    return (m2d.to(torch.float) * norm_v / norm_w0).squeeze(1)


def reassemble_comfyui_native_sd(
    weights_sd: Dict[str, torch.Tensor],
    loras: list,
) -> Dict[str, torch.Tensor]:
    """Native ``diffusion_model.*`` file → in-repo runtime fused layout.

    Inverse of the native emission in :func:`build_standard_state_dict`, so
    resume / merge / warm-start can read back what the trainer now writes:
    component factors re-fuse (case-1 ``w1`` concat, case-2 ``w2_a`` concat),
    the DoRA magnitude is un-rescaled per component against the live org
    weights, and plain q/k/v splits re-fuse like the loader's defuse inverse.
    SVD-fallback components cannot be re-fused losslessly and raise.
    """
    from networks.lora_utils import comfy_diffusion_stem

    out: Dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for lora in loras:
        name = lora.lora_name
        native_prefix = "diffusion_model." + comfy_diffusion_stem(
            _comfy_module_name(name)
        )
        spec = match_fused_spec(name)
        w0_full = lora.org_module_ref[0].weight.detach().to("cpu").to(torch.float)
        if spec is None:
            src = native_prefix
            if f"{src}.lokr_w1" in weights_sd:
                has_w2_full = f"{src}.lokr_w2" in weights_sd
                if has_w2_full:
                    for suffix in (".lokr_w1", ".lokr_w2", ".alpha"):
                        out[f"{name}{suffix}"] = weights_sd[f"{src}{suffix}"]
                        consumed.add(f"{src}{suffix}")
                    if f"{src}.dora_scale" in weights_sd:
                        w1 = weights_sd[f"{src}.lokr_w1"].to(torch.float)
                        w2 = weights_sd[f"{src}.lokr_w2"].to(torch.float)
                        out[f"{name}.dora_scale"] = _inverse_dora_rescale(
                            weights_sd[f"{src}.dora_scale"], w0_full, torch.kron(w1, w2)
                        )
                        consumed.add(f"{src}.dora_scale")
                else:
                    for suffix in (".lokr_w1", ".lokr_w2_a", ".lokr_w2_b", ".alpha"):
                        out[f"{name}{suffix}"] = weights_sd[f"{src}{suffix}"]
                        consumed.add(f"{src}{suffix}")
                    if f"{src}.dora_scale" in weights_sd:
                        w1 = weights_sd[f"{src}.lokr_w1"].to(torch.float)
                        w2 = weights_sd[f"{src}.lokr_w2_a"].to(
                            torch.float
                        ) @ weights_sd[f"{src}.lokr_w2_b"].to(torch.float)
                        s = (
                            float(weights_sd[f"{src}.alpha"].item())
                            / weights_sd[f"{src}.lokr_w2_b"].shape[0]
                        )
                        out[f"{name}.dora_scale"] = _inverse_dora_rescale(
                            weights_sd[f"{src}.dora_scale"],
                            w0_full,
                            torch.kron(w1, w2) * s,
                        )
                        consumed.add(f"{src}.dora_scale")
            elif f"{src}.lora_down.weight" in weights_sd:
                for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
                    out[f"{name}{suffix}"] = weights_sd[f"{src}{suffix}"]
                    consumed.add(f"{src}{suffix}")
                if f"{src}.dora_scale" in weights_sd:
                    down = weights_sd[f"{src}.lora_down.weight"].to(torch.float)
                    up = weights_sd[f"{src}.lora_up.weight"].to(torch.float)
                    s = float(weights_sd[f"{src}.alpha"].item()) / down.shape[0]
                    out[f"{name}.dora_scale"] = _inverse_dora_rescale(
                        weights_sd[f"{src}.dora_scale"], w0_full, (up @ down) * s
                    )
                    consumed.add(f"{src}.dora_scale")
            continue

        letters = spec.component_letters
        n = len(letters)
        base = name.removesuffix(spec.fused_frag)
        comp_prefixes = [
            "diffusion_model."
            + comfy_diffusion_stem(
                _comfy_module_name(f"{base}{spec.component_frag(letter)}")
            )
            for letter in letters
        ]
        w1_native = [weights_sd.get(f"{p}.lokr_w1") for p in comp_prefixes]
        w2_native = [weights_sd.get(f"{p}.lokr_w2") for p in comp_prefixes]
        has_w2_full = all(w is not None for w in w2_native)
        w2_a_native = [weights_sd.get(f"{p}.lokr_w2_a") for p in comp_prefixes]
        has_w2_lr = all(w is not None for w in w2_a_native)
        lokr_complete = all(w is not None for w in w1_native) and (
            has_w2_full or has_w2_lr
        )
        if lokr_complete:
            a_rows = lora.lokr_w1.shape[0]
            if has_w2_full:
                c = (
                    lora.lokr_w2.shape[0]
                    if hasattr(lora, "lokr_w2")
                    else w2_native[0].shape[0]
                )
                comp_dim = (a_rows * c) // n
                all_case1 = all(w.shape[0] == c for w in w2_native)
                all_case2 = all(w.shape[0] == 1 for w in w1_native) and all(
                    w.shape[0] < c for w in w2_native
                )
                if all_case1:
                    w1_fused = torch.cat([w.to(torch.float) for w in w1_native], dim=0)
                    w2_fused = w2_native[0]
                elif all_case2:
                    w1_fused = w1_native[0].to(torch.float)
                    w2_fused = torch.cat([w.to(torch.float) for w in w2_native], dim=0)
                else:
                    raise ValueError(
                        f"{name}: native file uses per-component SVD or mixed split "
                        "cases — cannot re-fuse losslessly; keep the legacy-format "
                        "file for resume/merge"
                    )
                if w1_fused.shape[0] != a_rows or w2_fused.shape[0] != c:
                    raise ValueError(
                        f"{name}: re-fused shapes {tuple(w1_fused.shape)}/{tuple(w2_fused.shape)} "
                        f"do not match the module {a_rows}/{c}"
                    )
                out[f"{name}.lokr_w1"] = w1_fused
                out[f"{name}.lokr_w2"] = w2_fused
                out[f"{name}.alpha"] = weights_sd[f"{comp_prefixes[0]}.alpha"]
                for p in comp_prefixes:
                    for suffix in (".lokr_w1", ".lokr_w2", ".alpha"):
                        consumed.add(f"{p}{suffix}")
                if f"{comp_prefixes[0]}.dora_scale" in weights_sd:
                    delta_full = torch.kron(w1_fused, w2_fused.to(torch.float))
                    m_chunks = []
                    for i, p in enumerate(comp_prefixes):
                        sl = slice(i * comp_dim, (i + 1) * comp_dim)
                        m_chunks.append(
                            _inverse_dora_rescale(
                                weights_sd[f"{p}.dora_scale"],
                                w0_full[sl],
                                delta_full[sl],
                            )
                        )
                        consumed.add(f"{p}.dora_scale")
                    out[f"{name}.dora_scale"] = torch.cat(m_chunks, dim=0)
            else:
                c = (
                    lora.lokr_w2_a.shape[0]
                    if hasattr(lora, "lokr_w2_a")
                    else w2_a_native[0].shape[0]
                )
                comp_dim = (a_rows * c) // n
                all_case1 = all(w.shape[0] == c for w in w2_a_native)
                all_case2 = all(w.shape[0] == 1 for w in w1_native) and all(
                    w.shape[0] < c for w in w2_a_native
                )
                if all_case1:
                    w1_fused = torch.cat([w.to(torch.float) for w in w1_native], dim=0)
                    w2_a_fused = w2_a_native[0]
                elif all_case2:
                    w1_fused = w1_native[0].to(torch.float)
                    w2_a_fused = torch.cat(
                        [w.to(torch.float) for w in w2_a_native], dim=0
                    )
                else:
                    raise ValueError(
                        f"{name}: native file uses per-component SVD or mixed split "
                        "cases — cannot re-fuse losslessly; keep the legacy-format "
                        "file for resume/merge"
                    )
                if w1_fused.shape[0] != a_rows or w2_a_fused.shape[0] != c:
                    raise ValueError(
                        f"{name}: re-fused shapes {tuple(w1_fused.shape)}/{tuple(w2_a_fused.shape)} "
                        f"do not match the module {a_rows}/{c}"
                    )
                out[f"{name}.lokr_w1"] = w1_fused
                out[f"{name}.lokr_w2_a"] = w2_a_fused
                out[f"{name}.lokr_w2_b"] = weights_sd[f"{comp_prefixes[0]}.lokr_w2_b"]
                out[f"{name}.alpha"] = weights_sd[f"{comp_prefixes[0]}.alpha"]
                for p in comp_prefixes:
                    for suffix in (".lokr_w1", ".lokr_w2_a", ".lokr_w2_b", ".alpha"):
                        consumed.add(f"{p}{suffix}")
                if f"{comp_prefixes[0]}.dora_scale" in weights_sd:
                    w2 = w2_a_fused.to(torch.float) @ weights_sd[
                        f"{comp_prefixes[0]}.lokr_w2_b"
                    ].to(torch.float)
                    s = (
                        float(weights_sd[f"{comp_prefixes[0]}.alpha"].item())
                        / weights_sd[f"{comp_prefixes[0]}.lokr_w2_b"].shape[0]
                    )
                    delta_full = torch.kron(w1_fused, w2) * s
                    m_chunks = []
                    for i, p in enumerate(comp_prefixes):
                        sl = slice(i * comp_dim, (i + 1) * comp_dim)
                        m_chunks.append(
                            _inverse_dora_rescale(
                                weights_sd[f"{p}.dora_scale"],
                                w0_full[sl],
                                delta_full[sl],
                            )
                        )
                        consumed.add(f"{p}.dora_scale")
                    out[f"{name}.dora_scale"] = torch.cat(m_chunks, dim=0)
        if lokr_complete:
            continue

        # plain fused module: q/k/v (or k/v) lora_down/up components
        down_native = [weights_sd.get(f"{p}.lora_down.weight") for p in comp_prefixes]
        up_native = [weights_sd.get(f"{p}.lora_up.weight") for p in comp_prefixes]
        if not (
            all(w is not None for w in down_native)
            and all(w is not None for w in up_native)
        ):
            touched = any(
                any(
                    f"{p}{suffix}" in weights_sd
                    for suffix in (".lokr_w1", ".lora_A.weight", ".lora_down.weight")
                )
                for p in comp_prefixes
            )
            if touched:
                raise ValueError(
                    f"{name}: native file splits this fused module in a form that "
                    "cannot be re-fused losslessly (SVD components or mixed cases); "
                    "keep the legacy-format file for resume/merge"
                )
            continue  # module simply absent from the native file
        out[f"{name}.lora_down.weight"] = down_native[0].clone()
        out[f"{name}.lora_up.weight"] = torch.cat(
            [w.to(torch.float) for w in up_native], dim=0
        )
        out[f"{name}.alpha"] = weights_sd[f"{comp_prefixes[0]}.alpha"]
        for p in comp_prefixes:
            for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
                consumed.add(f"{p}{suffix}")
        if f"{comp_prefixes[0]}.dora_scale" in weights_sd:
            down = out[f"{name}.lora_down.weight"].to(torch.float)
            up = out[f"{name}.lora_up.weight"].to(torch.float)
            s = float(weights_sd[f"{comp_prefixes[0]}.alpha"].item()) / down.shape[0]
            delta_full = (up @ down) * s
            comp_dim = w0_full.shape[0] // n
            m_chunks = []
            for i, p in enumerate(comp_prefixes):
                sl = slice(i * comp_dim, (i + 1) * comp_dim)
                m_chunks.append(
                    _inverse_dora_rescale(
                        weights_sd[f"{p}.dora_scale"], w0_full[sl], delta_full[sl]
                    )
                )
                consumed.add(f"{p}.dora_scale")
            out[f"{name}.dora_scale"] = torch.cat(m_chunks, dim=0)

    leftover = [
        k for k in weights_sd if k.startswith("diffusion_model.") and k not in consumed
    ]
    if leftover:
        logger.warning(
            f"native reassembly left {len(leftover)} unconsumed keys, e.g. {leftover[:3]}"
        )
    return out


def save_network_weights(
    state_dict: Dict[str, torch.Tensor],
    *,
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
    save_variant: str,
    base_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Run the save pipeline: variant write.

    Mutates ``state_dict`` in place. ``base_weights`` (lora_name → org Linear
    weight) enables the native ComfyUI emission — see
    :func:`build_standard_state_dict`.
    """
    if metadata is not None and len(metadata) == 0:
        metadata = None

    # Variant dispatch:
    #   * hydra_moe: shared-A Hydra → *_moe.safetensors
    #   * standard: defuse qkv → *.safetensors
    # Auto-fallback: any surviving ``.lora_up_weight`` key implies a Hydra
    # payload — kept for callers that don't plumb ``save_variant`` through.
    is_hydra_variant = save_variant == "hydra_moe" or any(
        k.endswith(".lora_up_weight") for k in state_dict.keys()
    )

    if is_hydra_variant:
        hydra_file = os.path.splitext(file)[0] + "_moe.safetensors"
        hydra_sd = HydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(hydra_sd, hydra_file, metadata or {})
        logger.info(f"HydraLoRA full format saved to {hydra_file}")
        # The _moe file is the only useful artifact for HydraLoRA —
        # a uniform expert average defeats layer-local routing.
        return

    # Standard write path.
    state_dict, metadata = build_standard_state_dict(
        state_dict, dtype, metadata, base_weights=base_weights
    )

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)


def export_comfyui_sidecar(
    lora_file: str,
    dit_path: Optional[str] = None,
) -> Optional[str]:
    """Training-end convenience: write a ComfyUI-native twin of a freshly
    saved LoKr / DoKr checkpoint, so no manual export step is needed.

    Fires only when the saved file is in the LEGACY layout (``lora_unet_*``
    keys carrying ``lokr_w1``) — since the native emission landed in
    :func:`build_standard_state_dict`, the main save is already ComfyUI-native
    whenever base weights are available and this is a fallback for saves that
    fell back. Plain LoRA / DoRA saves are already ComfyUI-native after the
    standard qkv defuse + adaln relayout. ``dit_path`` supplies the base-model
    weights the rescale needs; when omitted it falls back to
    ``configs/base.toml``'s ``pretrained_model_name_or_path``. Set
    ``ANIMA_COMFYUI_EXPORT=0`` to disable.

    Returns the sidecar path, or ``None`` when skipped or failed. Every
    failure mode is logged and swallowed — the native artifact is already
    on disk and training is over; this must never raise.
    """
    import os

    if os.environ.get("ANIMA_COMFYUI_EXPORT", "").strip().lower() in (
        "0",
        "false",
        "off",
    ):
        return None

    sidecar = os.path.splitext(lora_file)[0] + "_comfyui.safetensors"
    try:
        from safetensors import safe_open

        reader = safe_open(lora_file, framework="pt", device="cpu")
        keys = reader.keys()
        if any(k.startswith("diffusion_model.") for k in keys):
            logger.info("comfyui sidecar: file is already ComfyUI-native, skipping")
            return None
        if not any(k.startswith("lora_unet_") and k.endswith(".lokr_w1") for k in keys):
            logger.info("comfyui sidecar: file has no legacy lokr_w1 modules, skipping")
            return None

        if not dit_path or not os.path.exists(dit_path):
            from scripts.export_comfyui_lora import default_dit_path

            dit_path = default_dit_path()

        from scripts.export_comfyui_lora import convert

        stats = convert(lora_file, dit_path, sidecar)
        logger.info(
            f"comfyui sidecar written: {sidecar} "
            f"({stats['tensors']} keys / {stats['modules']} modules, "
            f"median err {stats['median_err']:.5f}, worst {stats['worst_err']:.4f}, "
            f"over-2%: {stats['over_2pct']})"
        )
        return sidecar
    except Exception as exc:  # noqa: BLE001 — must never break the training end
        logger.warning(f"comfyui sidecar export failed (native file unaffected): {exc}")
        return None
