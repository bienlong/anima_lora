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
from typing import Dict, Optional

import torch

from library.log import setup_logging
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
) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    """Run the standard finalize chain WITHOUT writing a file.

    Defuse fused qkv + bake channel scaling, relay adaln keys to the ComfyUI
    layout, cast dtype. Returns the finalized ``(state_dict, metadata)``. Factored
    out of :func:`save_network_weights` so the dual-pool turbo save can finalize
    each pool to its on-disk plain-LoRA layout and concat the two exactly.
    """
    defuse_and_bake_standard(state_dict)
    metadata = _relayout_adaln_to_comfy(state_dict, metadata)
    if dtype is not None:
        for key in list(state_dict.keys()):
            state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)
    return state_dict, metadata


def save_network_weights(
    state_dict: Dict[str, torch.Tensor],
    *,
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
    save_variant: str,
) -> None:
    """Run the save pipeline: variant write.

    Mutates ``state_dict`` in place.
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
    state_dict, metadata = build_standard_state_dict(state_dict, dtype, metadata)

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

    Fires only when the saved file contains ``lokr_w1`` modules — plain
    LoRA / DoRA saves are already ComfyUI-native after the standard qkv
    defuse + adaln relayout. ``dit_path`` supplies the base-model weights
    the rescale needs; when omitted it falls back to
    ``configs/base.toml``'s ``pretrained_model_name_or_path``. Set
    ``ANIMA_COMFYUI_EXPORT=0`` to disable.

    Returns the sidecar path, or ``None`` when skipped or failed. Every
    failure mode is logged and swallowed — the native artifact is already
    on disk and training is over; this must never raise.
    """
    import os

    if os.environ.get("ANIMA_COMFYUI_EXPORT", "").strip().lower() in ("0", "false", "off"):
        return None

    sidecar = os.path.splitext(lora_file)[0] + "_comfyui.safetensors"
    try:
        from safetensors import safe_open

        reader = safe_open(lora_file, framework="pt", device="cpu")
        if not any(k.endswith(".lokr_w1") for k in reader.keys()):
            logger.info("comfyui sidecar: file has no lokr_w1 modules, skipping")
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
