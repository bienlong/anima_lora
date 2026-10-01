# 分支说明（给拿到这个仓库的人）

本仓库有两个分支：

| 分支 | 内容 | 用途 |
|---|---|---|
| `main` | **修改版**：官方 v1.17.1 + 多重分辨率训练 / DoRA / LoKr / DoKr / rs-LoRA / qinglong_flux / 界面快捷下拉与高亮 | 日常使用、获取更新 |
| `upstream/v1.17.1` | **官方纯净版**（github.com/sorryhyun/anima_lora 的 release 源码包，sha256 逐文件校验过） | 对照 diff、回退原版、干净环境安装 |

tag：`mod-v1.0` = 修改版第一个完整功能集。

## 怎么用

```bash
# 拿到更新（维护者推送后）
git checkout main
git pull

# 想用官方原版
git checkout upstream/v1.17.1

# 看修改版相对官方改了什么
git diff upstream/v1.17.1 main --stat
```

没有安装过丹炉的人：先在 `upstream/v1.17.1` 分支按 README.md / install.ps1
装环境、`make download-models` 下模型，然后切回 `main` 即可（模型和 venv
不受 git 分支影响）。

## 功能入口速查（main 分支）

- **多分辨率训练**：Preprocess 页底部"多分辨率训练"区——勾档位、填每档时长
  占比（如 `1024:2, 896:1`）、点"多分辨率预处理"，再"写入变体"后直接训练。
- **适配器**：Config 页变体下拉 DoRA / rs-LoRA / LoKr；LoKr 变体内可同时勾
  use_dora（=DoKr）和 rs_lora。
- **采样**：timestep_sampling 选 qinglong_flux。
- 详细文档：`docs/methods/dora-lokr-rslora.md`、`docs/proposal/gui_ux_improvements.md`。
