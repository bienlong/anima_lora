# DoRA (Weight-Decomposed Low-Rank Adaptation, arXiv:2402.09353). The adapted
# weight W' = m ⊙ V / ‖V‖_row decomposes into a trainable per-output-channel
# magnitude vector m and the direction V = W0 + scale·(up @ down); the plain
# LoRA product trains the *direction* while m absorbs the per-channel norm
# drift that limits plain LoRA convergence. Slightly more VRAM than plain
# LoRA: the fp32 V / delta intermediates are (out × in) — they are recomputed
# in backward via torch.utils.checkpoint instead of riding the step graph,
# and DiT-block gradient checkpointing (the shipped presets) recompute-frames
# them either way.
#
# Init convention: m starts at ‖W0‖_row so W' == W0 (ΔW = 0) at step 0 — the
# same "identity start" every variant in this package honors. Checkpoints
# save the standard lora_down/up/alpha keys plus the magnitude vector as
# ``<name>.dora_scale`` — the key ComfyUI's native DoRA path consumes, so the
# saved file loads in stock ComfyUI without a converter.
#
# SPEED REALITY (measured, RTX 5060 Ti, 2.9B DiT, rank 8): plain LoRA
# ~1.5 s/step; this DoRA ~30-40 s/step. The per-module full-weight norm
# (out×in reduction + checkpoint recompute ×280 modules) is inherent to
# eager DoRA — closing the gap needs fused kernels (LyCORIS triton route).
# Use plain LoRA for day-to-day runs; DoRA when convergence quality is
# worth an overnight run.

import logging
from typing import Dict

import torch

from networks.lora_modules.lora import LoRAModule

logger = logging.getLogger(__name__)


class DoRALoRAModule(LoRAModule):
    """DoRA on the plain two-GEMM LoRA scaffold. Linear only (Anima targets).

    ``dora_scale`` is the magnitude vector from a checkpoint sniff (load
    path); when absent it is seeded from W0's row norms. Either way
    ``load_state_dict`` overwrites it with the trained values.
    """

    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
        down_init="kaiming",
        dora_scale=None,
    ):
        if org_module.__class__.__name__ != "Linear":
            raise ValueError(
                f"DoRA supports Linear targets only, got "
                f"{org_module.__class__.__name__} at {lora_name}."
            )
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            channel_scale=channel_scale,
            down_init=down_init,
        )
        if rank_dropout is not None:
            logger.warning(
                "rank_dropout is not supported on DoRA modules (%s); ignoring.",
                lora_name,
            )

        W0 = org_module.weight.data
        if dora_scale is not None:
            magnitude = dora_scale.detach().to(torch.float).clone()
        else:
            magnitude = W0.to(torch.float).norm(p=2, dim=1)
        self.register_buffer("dora_scale", magnitude)

    # -- weight math --------------------------------------------------------

    def _norm_scale_fn(self, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
        """Row-wise ``s = m / ‖W0 + scale·(up@down)‖`` → (out, 1).

        Runs in the model dtype (bf16 on GPU) with an fp32-accumulated norm —
        LyCORIS-style. The fp32 casts of the full W0 this replaces cost a
        16MB allocation per module per step (280 modules → multi-GB of pure
        cast traffic per step) and dominated the whole step time.
        """
        W0 = self.org_module_ref[0].weight
        V = W0 + self.scale * (up_w @ down_w)
        norms = V.norm(p=2, dim=1, keepdim=True, dtype=torch.float32).clamp_min(1e-12)
        return self.dora_scale.to(V.dtype).unsqueeze(1) / norms

    def _norm_scale(self, *, grad: bool) -> torch.Tensor:
        up_w = self.lora_up.weight if grad else self.lora_up.weight.detach()
        down_w = self.lora_down.weight if grad else self.lora_down.weight.detach()
        if grad and torch.is_grad_enabled():
            # Checkpoint the norm: backward recomputes the rank-r GEMM + norm
            # instead of retaining the (out × in) V intermediate. Only the
            # small (out, 1) result rides the graph — "显存略增" kept honest.
            return torch.utils.checkpoint.checkpoint(
                self._norm_scale_fn, up_w, down_w, use_reentrant=False
            )
        return self._norm_scale_fn(up_w, down_w)

    def get_weight(self, multiplier=None):
        """Return the DoRA delta (W' − W0) matching org_module.weight shape.

        Merge semantics: baking this delta into the base weight reproduces
        W' exactly (W0 + delta == W'), matching pre_calculation's add.
        """
        if multiplier is None:
            multiplier = self.multiplier
        with torch.no_grad():
            W0 = self.org_module_ref[0].weight.to(torch.float)
            V = W0 + self.scale * (
                self.lora_up.weight.to(torch.float)
                @ self.lora_down.weight.to(torch.float)
            )
            norms = V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            W_prime = (self.dora_scale.to(torch.float).unsqueeze(1) / norms) * V
            return (W_prime - W0) * multiplier

    # -- forward ------------------------------------------------------------

    def _activation_delta(self, x, org_forwarded, work, *, grad: bool):
        """DoRA output delta via the activation-side identity (no full delta
        weight, no second full GEMM):

            W'·x = s⊙(W0x) + s⊙scale·B(Ax)
                 = org(x) + (s−1)⊙org(x) + s⊙lora(x)

        ``org(x)`` is the already-computed frozen path — the only new compute
        vs plain LoRA is the small (out, 1) norm scale and elementwise multiplies.
        """
        s = self._norm_scale(grad=grad).squeeze(-1).to(work)
        x_lora = self._rebalance(x.to(work))
        lx = self._down(x_lora, work)
        lx = self._gate(lx, work)
        if self.dropout is not None:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        lx = self._up(lx.to(work), work)
        lora_out = lx * self.scale
        return (s - 1.0) * org_forwarded.to(work) + s * lora_out

    def _eval_delta(self, x, org_forwarded):
        work = org_forwarded.dtype
        with torch.no_grad():
            return self.multiplier * self._activation_delta(
                x, org_forwarded, work, grad=False
            )

    def forward(self, x):
        if not self.enabled or getattr(self, "_fused", False):
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            return org_forwarded + self._eval_delta(x, org_forwarded).to(
                org_forwarded.dtype
            )

        if self._skip_module():
            return org_forwarded

        work = org_forwarded.dtype
        delta_out = self._activation_delta(x, org_forwarded, work, grad=True)
        return org_forwarded + self.multiplier * delta_out.to(org_forwarded.dtype)

    # -- merge --------------------------------------------------------------

    def merge_to(self, sd, dtype, device):
        """Merge a per-DoRA state-dict slice into org_module.weight in-place.

        ``sd`` carries lora_down/up/alpha plus ``dora_scale`` (the saved
        magnitude); the merged weight is m ⊙ V / ‖V‖_row for multiplier=1.
        """
        with torch.no_grad():
            org_module = self.org_module_ref[0]
            weight = org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()
            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            W0 = w
            V = W0 + self.scale * (up_weight @ down_weight)
            magnitude = sd.get("dora_scale")
            if magnitude is None:
                logger.warning(
                    f"{self.lora_name}: DoRA checkpoint slice has no "
                    "dora_scale; falling back to the current row norms."
                )
                magnitude = V.norm(p=2, dim=1)
            magnitude = magnitude.to(torch.float).to(device)
            norms = V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            W_prime = (magnitude.unsqueeze(1) / norms) * V
            w += self.multiplier * (W_prime - W0)
            weight.data.copy_(w.to(dtype))


def fold_dora_scale(state_dict: Dict[str, torch.Tensor]) -> None:
    """No-op placeholder for symmetry with ``bake_inv_scale``.

    ``dora_scale`` must survive to disk verbatim (it IS the inference
    payload — ComfyUI and the in-repo loader both re-derive the norm from
    the base weight at apply time), so the standard write path needs no
    folding. Kept as a named seam so a future transform has a home.
    """
