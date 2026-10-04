"""同条件正面对比：LyCORIS 3.3 LoKr+DoRA vs 本仓库 DoKr vs 普通 LoRA。

单个 2048→2048 Linear、bf16、seq=4104（真实桶）、batch=1，fwd+bwd 计时。
×280 模块推算每步适配器开销。
"""
import os
os.chdir(r"C:/Users/anima-user/AppData/Local/Temp")
import statistics
from lycoris.modules.lokr import LokrModule  # noqa: E402 — 必须在 CUDA 初始化前导入
import sys
import time

sys.path.insert(0, r"G:\参考训练器-0.14.0\venv\Lib\site-packages")

import torch
import torch.nn as nn

dev = "cuda"
DIM, SEQ, R = 2048, 4104, 8
x = torch.randn(1, SEQ, DIM, dtype=torch.bfloat16, device=dev)


def bench(fn, iters=20, warmup=6):
    for _ in range(warmup):
        loss = fn(x).float().pow(2).mean()
        loss.backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        loss = fn(x).float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


# ── 1) LyCORIS 4.0 LoKr + DoRA（参考丹炉的实现） ──────────────────────────
base = nn.Linear(DIM, DIM, bias=False, dtype=torch.bfloat16, device=dev)
lok = LokrModule(
    "",
    base,
    multiplier=1,
    lora_dim=R,
    alpha=16,
    weight_decompose=True,
    use_tucker=False,
    use_scalar=False,
    lokr_factor=-1,
    decompose_both=False,
)
lok.cuda()
lok.apply_to()
print(f"LyCORIS 3.3 LoKr+DoRA (factor=-1): {bench(lok):.1f} ms/模块")

# ── 2) 本仓库 DoKr（detach 范数，块状 kron） ──────────────────────────────
from networks.lora_modules.dokr import DoKrLoRAModule  # noqa: E402

org = nn.Linear(DIM, DIM, bias=False, dtype=torch.bfloat16, device=dev)
dk = DoKrLoRAModule("t", org, 1.0, R, 16, dora_detach_norm=True)
dk.org_forward = org.forward
dk.org_module = org
dk.org_module_ref = [org]
print(f"本仓库 DoKr detach (factor=-1):    {bench(dk):.1f} ms/模块")

# ── 3) 本仓库 DoKr（materializing 严格路径） ──────────────────────────────
dk2 = DoKrLoRAModule("t2", org, 1.0, R, 16, dora_detach_norm=False)
dk2.org_forward = org.forward
dk2.org_module = org
dk2.org_module_ref = [org]
print(f"本仓库 DoKr materializing:         {bench(dk2):.1f} ms/模块")

# ── 4) 普通 LoRA 对照 ─────────────────────────────────────────────────────
from networks.lora_modules.lora import LoRAModule  # noqa: E402

org2 = nn.Linear(DIM, DIM, bias=False, dtype=torch.bfloat16, device=dev)
la = LoRAModule("t3", org2, 1.0, R, 16)
la.org_forward = org2.forward
la.org_module = org2
la.org_module_ref = [org2]
print(f"本仓库普通 LoRA:                   {bench(la):.1f} ms/模块")
