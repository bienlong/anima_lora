# LoKr (LoRA + Kronecker product, arXiv:2309.14859). The delta weight is a
# Kronecker product kron(w1, w2): w1 is a small full matrix, w2 is either a
# second full matrix (use_w2) or itself low-rank w2_a @ w2_b. Parameter count
# stays far below a full (out x in) delta while expressive power per parameter
# is higher than a rank-r two-factor LoRA. Ported from the LyCORIS layout via
# kohya's musubi-tuner implementation (Linear only — the Anima DiT is all
# Linear; Conv2d raises).
#
# Delta math and init follow the shared convention: w2 (or w2_b) starts at
# zero so kron(w1, 0) = 0 → ΔW = 0 at init, and the trained forward is
# ``org(x) + F.linear(x, kron(w1, w2) * scale) * multiplier``.

import logging
import math
from typing import Dict, Optional, Tuple

import torch

from networks.lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
    """Return ``(m, n)`` with ``m * n == dimension`` and ``m <= n``.

    In LoKr the first value sizes the w1 "scale" axis (small) and the second
    the w2 axis (large). ``factor=-1`` picks the most balanced split
    (128 → (8, 16), 512 → (16, 32), 1024 → (32, 32)); a positive factor that
    divides the dimension pins ``m = factor`` (128 → (4, 32) for factor=4).
    Mirrors musubi-tuner/LyCORIS ``factorization`` so checkpoints round-trip.
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1: torch.Tensor, w2: torch.Tensor, scale: float) -> torch.Tensor:
    """Kronecker product scaled by ``scale``; w1 may need trailing unsqueezes."""
    if w1.dim() != w2.dim():
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


class LoKrModule(BaseLoRAModule):
    """LoKr adapter module (training + eval). Linear only.

    ``lokr_shapes`` (from the from-weights key sniff) carries the saved
    ``{w1_shape, use_w2}`` split so a checkpoint reloads with the exact
    factorization it trained with instead of re-deriving it from ``factor``.
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
        factor: int = -1,
        lokr_shapes: Optional[Dict[str, Dict]] = None,
        channel_scale=None,  # 接受但忽略——Kronecker 无 lora_down 可吸收（与 dokr.py 同）
    ):
        if org_module.__class__.__name__ != "Linear":
            raise ValueError(
                f"LoKr supports Linear targets only, got "
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
        )
        if rank_dropout is not None:
            logger.warning(
                "rank_dropout is not supported on LoKr modules (%s); ignoring.",
                lora_name,
            )

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        factor = int(factor)

        hint = (lokr_shapes or {}).get(lora_name)
        if hint is not None and "w1_shape" in hint and "use_w2" in hint:
            out_l, in_m = hint["w1_shape"]
            out_k = out_dim // out_l
            in_n = in_dim // in_m
            self.use_w2 = bool(hint["use_w2"])
        else:
            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            self.use_w2 = self.lora_dim >= max(out_k, in_n) / 2

        # w1 is always the full (small) matrix.
        self.lokr_w1 = torch.nn.Parameter(torch.empty(out_l, in_m))
        if self.use_w2:
            self.lokr_w2 = torch.nn.Parameter(torch.empty(out_k, in_n))
            if hint is None:
                logger.warning(
                    f"LoKr: lora_dim {self.lora_dim} is large for "
                    f"dim={max(in_dim, out_dim)} and factor={factor}; "
                    "w2 falls back to a full matrix (scale=1)."
                )
        else:
            self.lokr_w2_a = torch.nn.Parameter(torch.empty(out_k, self.lora_dim))
            self.lokr_w2_b = torch.nn.Parameter(torch.empty(self.lora_dim, in_n))

        if self.use_w2:
            # kron(w1, w2) with both full spans the whole weight — the
            # alpha/lora_dim scaling would be double counting; force scale=1.
            self.scale = 1.0
            self.alpha.fill_(float(self.lora_dim))
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w2)
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w2_b)

        self.org_module_ref = [org_module]
        self._fused = False
        # Kron block shapes for the block-wise activation path (no out×in
        # materialization): kron(w1, w2)[(i,k),(j,m)] = w1[i,j]·w2[k,m] and the
        # product maps x viewed as (in_m, in_n) blocks to (out_l, out_k) blocks.
        self._in_m, self._in_n = int(in_m), int(in_n)
        self._out_l, self._out_k = int(out_l), int(out_k)
        self._out_dim = int(out_dim)

    # -- block-wise kron activation path ------------------------------------

    def _kron_project(self, x: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        """``kron(w1, w2) @ x`` block-wise — two small einsums, never
        materializing the (out × in) Kronecker matrix:

            x4 = x.view(..., in_m, in_n)
            u[..., j, k] = Σ_m w2[k, m]·x4[..., j, m]   (w2 or w2_a@w2_b)
            y4[..., i, k] = Σ_j w1[i, j]·u[..., j, k]
            y = y4.view(..., out)
        """
        w1 = self.lokr_w1.to(x.dtype)
        w2 = self._w2().to(x.dtype)
        lead = x.shape[:-1]
        x4 = x.reshape(*lead, self._in_m, self._in_n)
        u = torch.einsum("...jm,km->...jk", x4, w2)
        y4 = torch.einsum("ij,...jk->...ik", w1, u)
        return y4.reshape(*lead, self._out_dim)

    def _row_norms(self) -> torch.Tensor:
        """Row norms of ``W0 + scale·kron(w1, w2)`` block-wise (fp32 accumulate,
        no full-matrix materialization): with ``c = scale·w1``,

            ‖V_row(i,k)‖² = Σ_j Σ_m (W0[i,k,j,m] + c[i,j]·w2[k,m])²
                          = Σ_j [Σ_m W0² + 2c·Σ_m(W0·w2) + c²·Σ_m w2²]
        exact — V is linear in w1/w2, so the per-element square expansion
        loses nothing (and this is the detach path: no autograd, no retention).
        """
        org_weight = self.org_module_ref[0].weight
        W04 = org_weight.to(torch.float).view(
            self._out_l, self._out_k, self._in_m, self._in_n
        )
        w1 = self.lokr_w1.to(torch.float)
        w2 = self._w2().to(torch.float)
        c = self.scale * w1
        w2_sq = (w2 * w2).sum(dim=1)  # (out_k,)
        w0_sq = (W04 * W04).sum(dim=3)  # (out_l, out_k, in_m)
        w0_dot = torch.einsum("ikjm,km->ikj", W04, w2)  # (out_l, out_k, in_m)
        row_sq = (
            w0_sq.sum(dim=2)
            + 2.0 * torch.einsum("ij,ikj->ik", c, w0_dot)
            + w2_sq.unsqueeze(0) * (c * c).sum(dim=1).unsqueeze(1)
        )
        return row_sq.clamp_min(1e-12).sqrt().reshape(-1)

    # -- delta weight -------------------------------------------------------

    def _w2(self) -> torch.Tensor:
        if self.use_w2:
            return self.lokr_w2
        return self.lokr_w2_a @ self.lokr_w2_b

    def _delta_weight_fn(self, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        """kron delta in fp32 — the caller decides graph vs no-grad."""
        return make_kron(w1.to(torch.float), w2.to(torch.float), self.scale)

    def _delta_weight(self, *, grad: bool) -> torch.Tensor:
        w1 = self.lokr_w1 if grad else self.lokr_w1.detach()
        w2 = self._w2() if grad else self._w2().detach()
        if grad and torch.is_grad_enabled():
            # Recompute in backward instead of retaining the (out x in) kron
            # intermediates on the autograd graph — same trick as the DoRA
            # module; per-step retained memory stays at the factor params.
            return torch.utils.checkpoint.checkpoint(
                self._delta_weight_fn, w1, w2, use_reentrant=False
            )
        return self._delta_weight_fn(w1, w2)

    def get_weight(self, multiplier=None):
        """Return the LoKr delta as a tensor matching org_module.weight shape."""
        if multiplier is None:
            multiplier = self.multiplier
        return self._delta_weight(grad=False) * multiplier

    # -- forward ------------------------------------------------------------

    def _eval_delta(self, x, org_forwarded):
        # Block-wise activation path: kron·x via two small einsums — same map
        # as F.linear(x, kron delta) without the (out × in) materialization.
        return self.multiplier * self.scale * self._kron_project(
            self._rebalance(x), x.dtype
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
        x_lora = self._rebalance(x.to(work))
        y = self._kron_project(x_lora, work)
        return org_forwarded + self.multiplier * self.scale * y.to(
            org_forwarded.dtype
        )

    # -- fuse / merge -------------------------------------------------------

    def fuse_weight(self):
        """Bake the LoKr delta into org_module.weight; forwards no-op after."""
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        """Subtract a previously fused LoKr delta back out of org_module."""
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False

    def merge_to(self, sd, dtype, device):
        """Merge a per-LoKr state-dict slice into org_module.weight in-place."""
        with torch.no_grad():
            org_module = self.org_module_ref[0]
            weight = org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()

            w1 = sd["lokr_w1.weight"].to(torch.float).to(device)
            if "lokr_w2.weight" in sd:
                w2 = sd["lokr_w2.weight"].to(torch.float).to(device)
            else:
                w2 = sd["lokr_w2_a.weight"].to(torch.float).to(device) @ sd[
                    "lokr_w2_b.weight"
                ].to(torch.float).to(device)

            w += self.multiplier * make_kron(w1, w2, self.scale)
            weight.data.copy_(w.to(dtype))
