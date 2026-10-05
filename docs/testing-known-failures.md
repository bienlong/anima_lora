# 测试套件已知失败（2026-10-06 基线）

全量基线：`pytest tests/ --ignore=tests/test_daemon.py --ignore=tests/test_doc_refs.py`
当前 **12 → 2 failed**（本次清理后），剩余两项均为环境限制，**有意保留不修**：

| 测试 | 原因 | 处置 |
|---|---|---|
| `test_foveated_merge::test_merged_periphery_velocity_is_group_shared` | torch 版本相关的数值断言（`torch.equal` 精确相等），与上游 CI 的 torch 构建差异敏感。foveated.py 与上游逐字节一致 | 保留观察；换 torch 版本时复测 |
| `test_qwen_vae_2d::test_full_vae_2d_matches_3d_decode` | 需要 Qwen VAE 权重文件在本机 `models/` 下（本机模型三件套在 D:/ComfyUI 整合包） | 下载权重到 models/ 后可跑 |

## 本次已修复（曾是常驻失败）

- `test_downloads::test_anima_rows...`：base.toml 指向仓库外的绝对路径（本机 ComfyUI）时跳过该行断言，相对路径仍校验。
- `test_hang_guards::symlink_cycle`、`test_curation_walk_parity`：无 symlink 特权（Windows 无管理员/开发者模式）时跳过，附能力探测。
- `test_progress_sink::vanish_pid`：**产品 bug**——Windows 下消失的 pid 抛 `OSError winerror 87` 而非 `ProcessLookupError`，`_pid_alive` 误判"未知"。已修（progress.py）。
- `test_rocm_backend`、`test_windows_installer`：测试与 install.ps1 同步上游（rocm10 钉版 + `uv run --no-sync` 防止 ROCm 环境被重置，GH #105）。
