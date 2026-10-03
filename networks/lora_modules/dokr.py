# DoKr — DoRA weight-decomposition applied to a LoKr direction (LyCORIS
# "dokr": algo=lokr + weight-decompose). The adapted weight is
# ``W' = m ⊙ V / ‖V‖_row`` with ``V = W0 + scale·kron(w1, w2)`` — the
# Kronecker product trains the per-row *direction* while the trainable
# magnitude vector m (one float per output channel) absorbs the norm drift.
# The weight-decompose stage is shared with DoRALoRAModule conceptually; the
# direction source differs (kron factors instead of the two-GEMM product).
#
# Init convention matches every variant here: m starts at ‖W0‖_row so
# ΔW = 0 at step 0 (kron factors alone would also zero it via w2_b = 0, but
# the magnitude seeding is what makes reloads exact). Checkpoints save the
# lokr_w1 / lokr_w2[_a/_b] keys plus ``<name>.dora_scale``; in-repo loading
# only (same constraint as plain LoKr — fused attn keys stay fused).

import logging

import torch

from networks.lora_modules.lokr import LoKrModule, make_kron

logger = logging.getLogger(__name__)


class DoKrLoRAModule(LoKrModule):
    """DoRA-decomposed LoKr (training + eval). Linear only, like LoKr.

    ``dora_scale`` is the magnitude vector from a checkpoint sniff (load
    path); when absent it is seeded from W0's row norms. Either way
    ``load_state_dict`` overwrites it with the trained values.
    """

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
        lokr_shapes=None,
        dora_scale=None,
        dora_detach_norm=False,
    ):
        if org_module.__class__.__name__ != "Linear":
            raise ValueError(
                f"DoKr supports Linear targets only, got "
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
            factor=factor,
            lokr_shapes=lokr_shapes,
        )

        W0 = org_module.weight.data
        if dora_scale is not None:
            magnitude = dora_scale.detach().to(torch.float).clone()
        else:
            magnitude = W0.to(torch.float).norm(p=2, dim=1)
        # 可训练幅度（DoRA 语义核心）——buffer 版本会让幅度永远冻结在 ‖W0‖。
        self.dora_scale = torch.nn.Parameter(magnitude)
        self.detach_norm = bool(dora_detach_norm)

    # -- weight math --------------------------------------------------------

    def _norm_scale(self) -> torch.Tensor:
        """``s = m / ‖W0 + scale·kron‖_row`` → (out,)。

        detach 分支（DoKr 变体默认）：范数块状分解（无大矩阵拼接、无图内
        保留），m 保持在图内照常训练；严格分支走旧的 materializing 路径
        （梯度精确但慢一个量级，与 DoRA 严格模式同款权衡）。

        注意 `_row_norms` 返回一维 (out,)——这里必须保持一维除法：早年
        `unsqueeze(1) / 一维 norms` 会广播成 (out, out) 方阵（每模块
        151MB 的错误形状），传进 DiT 下层就是 192GB OOM / dynamo
        fake-tensor 报错的根因。
        """
        if self.detach_norm:
            with torch.no_grad():
                norms = self._row_norms()
            return self.dora_scale / norms
        W0 = self.org_module_ref[0].weight.to(torch.float)
        V = W0 + self.scale * make_kron(
            self.lokr_w1.to(torch.float), self._w2().to(torch.float), 1.0
        )
        norms = V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return self.dora_scale.to(torch.float).unsqueeze(1) / norms

    def _dokr_delta_fn(self, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        """``W' − W0`` in fp32 from the (graph-carrying) kron factors.

        V = W0 + scale·kron(w1, w2); W' = m ⊙ V / ‖V‖_row; W0 and m are
        frozen, so the checkpoint only saves the two factor matrices.
        """
        W0 = self.org_module_ref[0].weight.to(torch.float)
        V = W0 + self.scale * make_kron(w1.to(torch.float), w2.to(torch.float), 1.0)
        norms = V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        W_prime = (self.dora_scale.to(torch.float).unsqueeze(1) / norms) * V
        return W_prime - W0

    def _dokr_delta(self, *, grad: bool) -> torch.Tensor:
        w1 = self.lokr_w1 if grad else self.lokr_w1.detach()
        w2 = self._w2() if grad else self._w2().detach()
        if grad and torch.is_grad_enabled():
            # Recompute-in-backward for the (out × in) kron + norm
            # intermediates — same memory discipline as LoKr/DoRA.
            return torch.utils.checkpoint.checkpoint(
                self._dokr_delta_fn, w1, w2, use_reentrant=False
            )
        return self._dokr_delta_fn(w1, w2)

    def get_weight(self, multiplier=None):
        """Return the DoKr delta (W' − W0) matching org_module.weight shape."""
        if multiplier is None:
            multiplier = self.multiplier
        return self._dokr_delta(grad=False) * multiplier

    # -- forward ------------------------------------------------------------

    def _eval_delta(self, x, org_forwarded):
        work = org_forwarded.dtype
        if self.detach_norm:
            with torch.no_grad():
                s = self._norm_scale().squeeze(-1).to(work)
            x_lora = self._rebalance(x.to(work))
            y = self.scale * self._kron_project(x_lora, work)
            return self.multiplier * ((s - 1.0) * org_forwarded.to(work) + s * y)
        return self.multiplier * torch.nn.functional.linear(
            self._rebalance(x), self._dokr_delta(grad=False)
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
        if self.detach_norm:
            # 激活侧恒等式 + 块状 kron：无 (out×in) 拼接、无全量 delta GEMM
            # ——速度与 DoRA detach/普通 LoRA 同级（实测见 docstring）。
            s = self._norm_scale().squeeze(-1).to(work)
            x_lora = self._rebalance(x.to(work))
            y = self.scale * self._kron_project(x_lora, work)
            return org_forwarded + self.multiplier * (
                (s - 1.0) * org_forwarded.to(work) + s * y
            ).to(org_forwarded.dtype)
        delta = self._dokr_delta(grad=True).to(work)
        x_lora = self._rebalance(x.to(work))
        return org_forwarded + self.multiplier * torch.nn.functional.linear(
            x_lora, delta
        ).to(org_forwarded.dtype)

    # -- merge --------------------------------------------------------------

    def merge_to(self, sd, dtype, device):
        """Merge a per-DoKr state-dict slice into org_module.weight in-place.

        ``sd`` carries lokr_w1 / lokr_w2[_a/_b] plus ``dora_scale``; the
        merged weight is m ⊙ V / ‖V‖_row for multiplier=1.
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

            w1 = sd["lokr_w1.weight"].to(torch.float).to(device)
            if "lokr_w2.weight" in sd:
                w2 = sd["lokr_w2.weight"].to(torch.float).to(device)
            else:
                w2 = sd["lokr_w2_a.weight"].to(torch.float).to(device) @ sd[
                    "lokr_w2_b.weight"
                ].to(torch.float).to(device)

            W0 = w
            V = W0 + self.scale * make_kron(w1, w2, 1.0)
            magnitude = sd.get("dora_scale")
            if magnitude is None:
                logger.warning(
                    f"{self.lora_name}: DoKr checkpoint slice has no "
                    "dora_scale; falling back to the current row norms."
                )
                magnitude = V.norm(p=2, dim=1)
            magnitude = magnitude.to(torch.float).to(device)
            norms = V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            W_prime = (magnitude.unsqueeze(1) / norms) * V
            w += self.multiplier * (W_prime - W0)
            weight.data.copy_(w.to(dtype))
