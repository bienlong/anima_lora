# DoKr 速度调查：v1.17.1 → v2.0.x 的步时回归归因（2026-10-06）

## 结论（TL;DR）

「DoKr 从 1.69s/步 变 2.3s/步」由三部分叠加，**v2 升级只占其中一小部分**：

| 阶段 | 速度 | 说明 |
|---|---|---|
| 10-02，v1 旧数学（detach 范数重组引入前），factor=0 | **1.27–1.42s/步** | job 日志 `20261002-125510` 等 8 个任务 |
| 10-03 深夜，v1 detach+checkpoint 修复后（abde52f），factor=-1 | **1.66s/步** | 当日实测（记忆口径），detach 范数重组本身 +0.5s 左右 |
| 10-04，v1 最终代码，detach=true factor=0（用户现行配置） | **1.87–2.21s/步** | job 日志 `20261004-181021`(2.21) / `20261004-213253`(1.87) |
| 10-05，v2.0.0 同配置 | **2.28s/步** | 1727f0a 验证 |
| 10-06，v2.0.1 同配置（本次复测） | **2.30–2.31s/步** | 120 步 + factor=-1 对照 100 步，均 2.30 |

1. **detach 范式成本（v1 时代引入，约 +0.5s）**：`(s-1)·org + s·y` 的范数重组激活侧恒等式比旧数学贵；这是功能语义（可训练幅度范数），不是 bug。
2. **v2.0.0 升级全局回归（+0.32s，所有变体）**：纯 LoRA 1.48 → 1.80s/步（本次复测）。模型 forward、attention dispatch 逐行相同（models.py 差异全为注释），torch/triton/accelerate 版本相同，每步 token 数完全一致（同 4048–4200 带、同 20 families）。
3. **DoKr 相对回归（约 +0.1~0.3s）**：1.87–2.21 → 2.30，扣除全局部分后的余量。

## 已排除的假设（本次实测）

- **显存分配器挣扎**：新加的 `vram/*` 指标显示 allocated 4.2GB / peak 11.8GB / reserved 12.4GB 稳定，`num_alloc_retries = 0`。nvidia-smi 的 15.5GB 中约 3GB 是 CUDA 上下文 + cuBLAS/编译内核等池外开销。
- **lokr_factor 配置差异**：factor=-1 与 factor=0 实测同为 2.30s/步（对照实验）。
- **每步计算量变化**：v1/v2 job 日志的 bucket 表完全一致（864×1216: 78 张等）。
- **模型代码改动**：models.py v1→v2 差异全部为注释。
- **依赖大版本漂移**：torch 2.12 / triton 3.7 / accelerate 1.13 完全相同；diffusers 0.39→0.41.dev、transformers 5.10→5.16 为小版本。
- **GPU 降频/温度**：37°C，无 thermal 证据。

## 算子画像（v2，10 步均值）

GPU 每步 2.31s 构成：`aten::mm` 68%（cutlass bf16 GEMM，inductor 选择）、flash-attn 前后向 18%、triton fused 5%。GPU-bound（"Command Buffer Full" 仅表示 CPU 提交队列等待，非瓶颈）。残余的 ~10-20% 差异落在 inductor kernel 选择/图结构微差上，继续定位需要 Nsight Systems（本机未装），收益/成本比低，**接受现状**。

## 新增的常备工具（本次落地）

- `vram/allocated_gb / vram/reserved_gb / vram/peak_gb / vram/alloc_retries`：进 progress.jsonl 与 tensorboard（`library/training/loop.py::_log_step`）。
- `ANIMA_MEM_LOG=1`：按节奏打完整 `torch.cuda.memory_summary()`。
- `ANIMA_TORCH_PROFILE=起始步,步数`：torch.profiler 挂入 nsys 钩子缝（无 Nsight 也能看 top 算子表）。
- `--max_train_steps N`：CLI 显式给出时不再被 TOML 的 `max_train_epochs` 覆盖（train.py）。

## 基准复测数据（v2.0.1，RTX 5060 Ti 16GB，121 图 batch1）

| 配置 | s/步 | 显存 allocated/peak/reserved |
|---|---|---|
| DoKr detach factor=0（用户现行） | 2.31 | 4.2 / 11.8 / 12.4 GB |
| DoKr detach factor=-1 | 2.30 | 4.2 / 11.8 / 12.5 GB |
| LoRA（对照） | 1.80 | 4.7 / 12.4 / 12.7 GB |
