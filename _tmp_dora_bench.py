"""DoRA 速度拆解微基准：单个 2048x2048 模块，bf16，seq=4100，batch=1。

逐项计时 fwd+bwd：
  a) plain LoRA（对照）
  b) DoRA 现行实现（checkpoint 包裹范数）
  c) DoRA 无 checkpoint（范数在图内，保留 V）
  d) DoRA detach 范数（范数不参与梯度——牺牲一点语义换速度）
每种再测 eager 与 torch.compile 两档。×280 推算每步开销。
"""
import statistics
import time

import torch
import torch.nn as nn

torch.manual_seed(0)
dev = "cuda"
R, DIM, SEQ = 8, 2048, 4100


class LoraLike(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.base = nn.Linear(DIM, DIM, bias=False, dtype=torch.bfloat16, device=dev)
        self.down = nn.Linear(DIM, R, bias=False, dtype=torch.bfloat16, device=dev)
        self.up = nn.Linear(R, DIM, bias=False, dtype=torch.bfloat16, device=dev)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.mag = nn.Buffer(
            self.base.weight.detach().float().norm(dim=1), persistent=False
        )
        self.scale = 2.0

    def _norm_scale(self):
        V = self.base.weight + self.scale * (self.up.weight @ self.down.weight)
        n = V.norm(p=2, dim=1, keepdim=True, dtype=torch.float32).clamp_min(1e-12)
        return (self.mag.unsqueeze(1) / n).squeeze(-1)

    def forward(self, x):
        org = self.base(x)
        if self.mode == "lora":
            return org + self.scale * self.up(self.down(x))
        s = self._norm_scale()
        if self.mode == "dora_ckpt" and torch.is_grad_enabled():
            s = torch.utils.checkpoint.checkpoint(
                self._norm_scale, use_reentrant=False
            )
        if self.mode == "dora_detach":
            s = s.detach()
        lora = self.scale * self.up(self.down(x))
        return org + (s - 1.0).to(org.dtype) * org + s.to(org.dtype) * lora


def bench(fn, x, iters=30, warmup=8):
    for _ in range(warmup):
        loss = fn(x).float().pow(2).mean()
        loss.backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        loss = fn(x).float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


x = torch.randn(1, SEQ, DIM, dtype=torch.bfloat16, device=dev)
print(f"{'变体':<22}{'eager ms':>10}{'compile ms':>12}{'x280 推算':>12}")
for mode in ("lora", "dora_ckpt", "dora_nockpt", "dora_detach"):
    m = LoraLike(mode)
    eager = bench(m, x)
    cm = torch.compile(m)
    compiled = bench(cm, x)
    print(
        f"{mode:<22}{eager*1000:>9.1f} {compiled*1000:>11.1f} "
        f"compiled×280={compiled*280:.1f}s"
    )
