# GUI 顺手度改进规划（gui UX roadmap）

来源：用户请求"优化前端，让其操作更顺手，每个功能添加快捷的选择，而不是输入名字"。
本文件记录已落地项与后续优先级。改动 GUI 前先读 `gui/CLAUDE.md` + `gui-changes` 技能。

## 已落地（2026-10-01）

1. **枚举字段快捷下拉**（`gui/widgets/fields.py::_COMBO_CHOICES`）：
   `timestep_sampling`（封闭）、`mixed_precision`（封闭）、`lr_scheduler` /
   `optimizer_type` / `discrete_flow_shift`（可编辑下拉 = 快捷 + 保留长尾输入）。
   封闭列表遇到未知已存值会追加显示，绝不静默改值。
2. **多分辨率训练区**（Preprocess 页）：档位多选 + 每档时长占比 + 步数占比实时
   预览 + 一键预处理/写入变体/恢复基础蓝图（见 `library/preprocess/multires.py`
   与 `docs/methods/` 后续补充）。
3. **★ 高亮字段**：`timestep_sampling` / `scale_weight_norms` 提升到 Basic 区
   并加强调色（用户指定的两个常调项）。

## 建议后续（按性价比排序）

1. **选项中文标签**（低成本）：`lr_scheduler`/`optimizer_type` 下拉显示
   「常数 / Cosine / Cosine 重启」式双语标签，`_read` 经 value_map 还原为
   TOML 值。参照其它训练器的呈现；i18n 走 `opt_*` 键（4 语言）。
2. **字段行内提示**（中成本）：`_fields.json` 已有全文说明；再存一份 ≤40 字
   的短提示（`_fields_short.json`），在字段下方以 dim 色常显一行，点击标签
   看长文。只对 Basic 区 ~20 个字段做，Advanced 保持现状。
3. **保存前冲突预览**（中成本）：`Save` 目前整文件写回变体；加一个 diff 预览
   （改了哪些键、从什么值到什么值），防误存。数据点：变体是机器写的，冲突
   风险主要来自多开 GUI。
4. **多分辨率预设**（低成本）：`MultiResSection` 加"常用组合"下拉
   （如 风格 1024:1,896:1 / 角色 1024:3,896:1,768:1）。
5. **训练历史卡片**（高成本）：Queue 记录产出 checkpoint 的超参快照，点击
   回填表单（`.snapshot.toml` 已有数据，纯 GUI 工作）。
6. **字段搜索**（高成本）：Advanced 展开后字段很多；顶部搜索框过滤/滚动定位。
   需要动 `_reload` 的构建顺序，注意 launch-speed 测试红线。

## 红线（动 GUI 前必读）

- Preprocess 的 knob 表改动必须重生成
  `tests/fixtures/gui_preprocess_knobs.json`（`--write`，只允许有意为之）。
- 新增 UI 字符串四语言同步（`en/cn/ja/ko`），缺失回退英文不是错误但要补。
- 全局样式表绕过 proxy style 会导致 SplitButton 标签错位——按钮着色一律走
  `theme.ACTION_COLORS` + `action_button`/`apply_variant`。
- `python -X importtime -c "import gui.app"` 不得出现 torch。
