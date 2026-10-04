"""快速开始主页签 —— 选数据集 → 字幕 → 预处理 → 训练，一条流水线。

主次感设计：本页是"第一次训练"的主入口，四个步骤卡垂直引导、每步带
状态徽章（✓ 完成 / ○ 待做 / ▶ 进行中）；完整高级参数留在原 Config /
Preprocess 页（右上角"高级模式"切换）。全部动作复用既有机器：
  * 数据集选择写回 PreprocessingTab 的 source_image_dir 并持久化；
  * 打标 + 预处理直接调 PreprocessingTab._run_te（勾选自动打标的完整链，
    与高级页同一套 daemon 快照/env，行为零差异）；
  * 训练经 gui_daemon.submit_training（overrides 内联 rank/epochs/lr）。
状态探测用 Qt-free 的 pipeline_status（与 QuickStartCard 同源）。
"""

from __future__ import annotations


import json
import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QProgressBar,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui import daemon as gui_daemon
from gui._paths import ROOT
from gui.config_io import default_lora_cache_dir, list_gui_variants, variant_metadata
from gui.i18n import t
from gui.theme import tok
from gui.widgets import action_button
from gui.widgets.fields import attach_browse
from gui.tabs.preprocess.quickstart import pipeline_status


def _action_color(key: str) -> str:
    from gui.theme import ACTION_COLORS

    return ACTION_COLORS[key]


def _dot(color: str) -> QLabel:
    lbl = QLabel("●")
    lbl.setFixedWidth(16)
    lbl.setStyleSheet(f"color:{color}; font-size:13px; border: none;")
    return lbl


class _StepCard(QWidget):
    """一张步骤卡：编号圆 + 标题 + 状态徽章顶行，内容区在下，卡间有引导线。"""

    def __init__(self, num: int, title: str, hint: str):
        super().__init__()
        self.setStyleSheet(
            f"#step {{ background: {tok('panel')}; border: 1px solid "
            f"{tok('text_dim')}; border-radius: 8px; }}"
            "QWidget { border: none; }"
        )
        self.setObjectName("step")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        head = QHBoxLayout()
        num_lbl = QLabel(str(num))
        num_lbl.setFixedSize(24, 24)
        num_lbl.setAlignment(Qt.AlignCenter)
        num_lbl.setStyleSheet(
            f"background: {tok('link')}; color: {tok('text_bright')}; border-radius: 12px;"
            "font-weight: 700; border: none;"
        )
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 15px; font-weight: 700;")
        head.addWidget(num_lbl)
        head.addWidget(title_lbl, 1)
        self.badge_dot = _dot(_action_color("warning"))
        self.badge_text = QLabel(t("qs_status_todo"))
        head.addWidget(self.badge_dot)
        head.addWidget(self.badge_text)
        v.addLayout(head)
        hint_lbl = QLabel(hint)
        hint_lbl.setStyleSheet(f"color: {tok('text_dim')};")
        hint_lbl.setWordWrap(True)
        v.addWidget(hint_lbl)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 2, 0, 0)
        v.addLayout(self.body)

    def add_layout(self, layout) -> None:
        self.body.addLayout(layout)

    def add_widget(self, w) -> None:
        self.body.addWidget(w)

    def set_status(self, ok: bool, running: bool) -> None:
        if running:
            self.badge_dot.setStyleSheet(f"color:{_action_color('info')}; font-size:13px;")
            self.badge_text.setText(t("qs_status_running"))
        elif ok:
            self.badge_dot.setStyleSheet(f"color:{_action_color('success')}; font-size:13px;")
            self.badge_text.setText(t("qs_status_done"))
        else:
            self.badge_dot.setStyleSheet(f"color:{_action_color('warning')}; font-size:13px;")
            self.badge_text.setText(t("qs_status_todo"))


class QuickStartTab(QWidget):
    """第一次训练的主入口。preprocess_tab = 主窗口已有的 PreprocessingTab。"""

    def __init__(self, preprocess_tab, tb_panel=None):
        super().__init__()
        self._pp = preprocess_tab
        self._tb_panel = tb_panel
        self._train_job_id: str | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 10)

        title = QLabel(t("qs_title"))
        title.setStyleSheet("font-size: 20px; font-weight: 800;")
        sub = QLabel(t("qs_subtitle"))
        sub.setStyleSheet(f"color: {tok('text_dim')};")
        outer.addWidget(title)
        outer.addWidget(sub)

        # ── 步骤 1：数据集文件夹 ───────────────────────────────────────
        s1 = _StepCard(1, t("qs_s1_title"), t("qs_s1_hint"))
        row1 = QHBoxLayout()
        self.src_edit = QLineEdit()
        attach_browse(self.src_edit, mode="dir")
        self.src_count = QLabel("")
        self.src_count.setStyleSheet(f"color: {tok('text_dim')};")
        row1.addWidget(self.src_edit, 1)
        row1.addWidget(self.src_count)
        s1.add_layout(row1)
        outer.addWidget(s1)
        self._s1 = s1
        self.src_edit.textChanged.connect(self._on_src_changed)

        # ── 步骤 2：字幕（自动打标）───────────────────────────────────
        s2 = _StepCard(2, t("qs_s2_title"), t("qs_s2_hint"))
        row2 = QHBoxLayout()
        self.cap_count = QLabel("")
        self.cap_count.setStyleSheet(f"color: {tok('text_dim')};")
        self.tag_btn = action_button(
            t("qs_tag_btn"),
            variant="info",
            tooltip=t("qs_tag_btn_hint"),
            on_click=self._run_tag_and_cache,
        )
        row2.addWidget(self.cap_count, 1)
        row2.addWidget(self.tag_btn)
        s2.add_layout(row2)
        outer.addWidget(s2)
        self._s2 = s2

        # ── 步骤 3：预处理缓存 ────────────────────────────────────────
        s3 = _StepCard(3, t("qs_s3_title"), t("qs_s3_hint"))
        row3 = QHBoxLayout()
        self.cache_count = QLabel("")
        self.cache_count.setStyleSheet(f"color: {tok('text_dim')};")
        self.pp_btn = action_button(
            t("qs_preprocess_btn"),
            variant="info",
            tooltip=t("qs_preprocess_btn_hint"),
            on_click=self._run_tag_and_cache,
        )
        row3.addWidget(self.cache_count, 1)
        row3.addWidget(self.pp_btn)
        s3.add_layout(row3)
        outer.addWidget(s3)
        self._s3 = s3

        # ── 步骤 4：训练 ─────────────────────────────────────────────
        s4 = _StepCard(4, t("qs_s4_title"), t("qs_s4_hint"))
        # 训练配方：一键填好三个数字（奶人模式——不要求理解 rank/lr）
        recipe_row = QHBoxLayout()
        recipe_row.addWidget(QLabel(t("qs_recipe_label")))
        for name, key in (
            (t("qs_recipe_quick"), "quick"),
            (t("qs_recipe_char"), "char"),
            (t("qs_recipe_style"), "style"),
        ):
            recipe_row.addWidget(
                action_button(
                    name,
                    variant="info",
                    tooltip=t("qs_recipe_hint"),
                    on_click=lambda _=False, k=key: self._apply_recipe(k),
                )
            )
        recipe_row.addStretch(1)
        s4.add_layout(recipe_row)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.variant_combo = QComboBox()
        for v in list_gui_variants("lora"):
            label = (variant_metadata(v) or {}).get("label") or v
            self.variant_combo.addItem(label, v)
        form.addRow(t("qs_variant"), self.variant_combo)
        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 256)
        self.rank_spin.setValue(32)
        form.addRow(t("qs_rank"), self.rank_spin)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(4)
        form.addRow(t("qs_epochs"), self.epochs_spin)
        self.lr_edit = QLineEdit("2e-5")
        form.addRow(t("qs_lr"), self.lr_edit)
        s4.add_layout(form)
        train_row = QHBoxLayout()
        self.output_edit = QLineEdit("my_lora")
        train_row.addWidget(QLabel(t("qs_output_name")))
        train_row.addWidget(self.output_edit, 1)
        self.train_btn = action_button(
            t("qs_train_btn"),
            variant="primary",
            tooltip=t("qs_train_btn_hint"),
            on_click=self._start_train,
        )
        train_row.addWidget(self.train_btn)
        s4.add_layout(train_row)

        # 实时进度块（训练中显示）
        self.progress = QProgressBar()
        self.progress.setStyleSheet(f"QProgressBar {{ border: 1px solid {tok('border')}; border-radius: 4px; text-align: center; }} QProgressBar::chunk {{ background: {tok('link')}; border-radius: 3px; }}")
        self.progress.hide()
        s4.add_widget(self.progress)
        self.prog_label = QLabel("")
        self.prog_label.setStyleSheet(f"color: {tok('text_dim')};")
        self.prog_label.hide()
        s4.add_widget(self.prog_label)

        # 完成卡（训练完成后显示）
        self.done_box = QLabel("")
        self.done_box.setStyleSheet(
            f"color: {_action_color('success')}; font-weight: 600; padding: 6px;"
            f"background: {tok('panel')}; border-radius: 6px;"
        )
        self.done_box.setWordWrap(True)
        self.done_box.hide()
        s4.add_widget(self.done_box)
        self.open_folder_btn = action_button(
            t("qs_open_folder"),
            tooltip=t("qs_open_folder_hint"),
            on_click=self._open_output_folder,
        )
        self.open_folder_btn.hide()
        s4.add_widget(self.open_folder_btn)
        outer.addWidget(s4)
        self._s4 = s4

        # ── 日志 + 高级模式 ──────────────────────────────────────────
        bottom = QHBoxLayout()
        self.adv_btn = action_button(
            t("qs_advanced"),
            tooltip=t("qs_advanced_hint"),
            on_click=self._open_advanced,
        )
        bottom.addWidget(self.adv_btn)
        bottom.addStretch(1)
        outer.addLayout(bottom)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace; font-size:11px;")
        self.log.setPlaceholderText(t("qs_log_placeholder"))
        self.log.setMaximumHeight(140)
        outer.addWidget(self.log)

        outer.addStretch(1)

        # 状态刷新：3 秒轮询磁盘 + daemon job 状态。
        self._timer = QTimer(self)
        self._timer.setInterval(3000)
        self._timer.timeout.connect(self.refresh_status)
        self._timer.start()
        self.refresh_status()

    # ── 状态 ─────────────────────────────────────────────────────────

    def _src(self) -> str:
        # QuickStart 与 Preprocess 页共享同一个源目录字段（单一事实来源）。
        return self._pp.source_dir_edit.text().strip()

    def refresh_status(self) -> None:
        src = self._src()
        cache = default_lora_cache_dir()
        st = pipeline_status(src or "", cache)
        running_pp = self._pp._is_running() if hasattr(self._pp, "_is_running") else False
        running_train = self._train_job_running()

        self._s1.set_status(st["dataset_ok"], False)
        self._s2.set_status(st["captions_ok"], running_pp and not st["captions_ok"])
        self._s3.set_status(st["cache_ok"], running_pp)
        self._s4.set_status(st["trainable"], running_train)

        self.src_count.setText(t("qs_images_count").format(n=st["n_imgs"]))
        self.cap_count.setText(
            t("qs_captions_count").format(n=st["n_captions"], miss=st["missing_captions"])
        )
        self.cache_count.setText(
            t("qs_cache_count").format(
                vae=st["n_npz"], te=st["n_te"], n=st["n_imgs"]
            )
        )
        if st["trainable"] and not running_train:
            self.train_btn.setEnabled(True)

        # 训练进度 + 完成卡
        if running_train and self._train_job_id:
            prog = self._latest_progress()
            self.progress.show()
            self.prog_label.show()
            self.done_box.hide()
            self.open_folder_btn.hide()
            if prog:
                total = prog.get("total_steps") or 0
                step = prog.get("global_step") or 0
                loss = prog.get("loss/current")
                epoch = prog.get("epoch")
                if total:
                    self.progress.setRange(0, total)
                    self.progress.setValue(min(step, total))
                avg_rate = prog.get("ts", 0) / max(step, 1)
                remain_min = max(0, (total - step)) * avg_rate / 60
                self.prog_label.setText(
                    t("qs_progress_line").format(
                        epoch=epoch or "?",
                        step=step,
                        total=total or "?",
                        loss=f"{loss:.3f}" if isinstance(loss, (int, float)) else "-",
                        remain=f"{remain_min:.0f}",
                    )
                )
        else:
            self.progress.hide()
            self.prog_label.hide()
            job_done = False
            if self._train_job_id:
                try:
                    job_done = gui_daemon.read_job_state(self._train_job_id) == "done"
                except Exception:
                    job_done = False
            if job_done:
                out_name = self.output_edit.text().strip() or "my_lora"
                self.done_box.setText(t("qs_done_text").format(name=out_name))
                self.done_box.show()
                self.open_folder_btn.show()
            else:
                self.done_box.hide()
                self.open_folder_btn.hide()

    def _train_job_running(self) -> bool:
        if not self._train_job_id:
            return False
        try:
            return gui_daemon.read_job_state(self._train_job_id) == "running"
        except Exception:
            return False

    def _on_src_changed(self, text: str) -> None:
        # 与 Preprocess 页双向同步（单一事实来源在 Preprocess 页字段）。
        pp_edit = getattr(self._pp, "source_dir_edit", None)
        if pp_edit is not None and pp_edit.text() != text:
            pp_edit.setText(text)

    # ── 动作 ─────────────────────────────────────────────────────────

    def _run_tag_and_cache(self) -> None:
        src = self._src()
        if not src:
            self.log.appendPlainText(t("qs_need_folder"))
            return
        st = pipeline_status(src, default_lora_cache_dir())
        # 缺字幕 → 勾选自动打标（完整链会先打标再缓存）。
        self._pp.caption_autotag_chk.setChecked(bool(st["missing_captions"]))
        try:
            self._pp.persist_preprocess_inputs()
        except Exception:
            pass
        self.log.appendPlainText(t("qs_running_cache"))
        self._pp._run_te()

    def _start_train(self) -> None:
        src = self._src()
        st = pipeline_status(src or "", default_lora_cache_dir())
        if not st["trainable"]:
            self.log.appendPlainText(t("qs_not_trainable"))
            return
        variant = self.variant_combo.currentData() or "lora"
        try:
            lr = float(self.lr_edit.text().strip())
        except ValueError:
            self.log.appendPlainText(t("qs_bad_lr"))
            return
        from gui.config_io import merged_gui_variant_preset

        preset = "default"
        merged, _ = merged_gui_variant_preset(variant, preset)
        output_name = "".join(
            c for c in self.output_edit.text().strip() if c not in '\\/:*?"<>|'
        ).strip() or "my_lora"
        overrides = {
            "network_dim": int(self.rank_spin.value()),
            "network_alpha": int(self.rank_spin.value()),
            "max_train_epochs": int(self.epochs_spin.value()),
            "learning_rate": lr,
            "output_name": output_name,
        }
        self.log.appendPlainText(
            t("qs_train_submitting").format(
                variant=variant, rank=overrides["network_dim"],
                epochs=overrides["max_train_epochs"], lr=lr,
            )
        )
        self._train_job_id = self._submit_job(
            lambda: gui_daemon.submit_training(
                method=variant,
                preset=preset,
                methods_subdir="gui-methods",
                config_snapshot=merged,
                overrides=overrides,
                start=True,
            ),
            on_fail=None,
        )
        if self._train_job_id:
            self.log.appendPlainText(t("qs_train_queued").format(job=self._train_job_id))
            QTimer.singleShot(2000, self.refresh_status)

    _RECIPES = {
        "quick": {"rank": 8, "alpha": 8, "epochs": 2, "lr": 1e-4},
        "char": {"rank": 16, "alpha": 16, "epochs": 8, "lr": 1e-4},
        "style": {"rank": 32, "alpha": 32, "epochs": 12, "lr": 1e-4},
    }

    def _apply_recipe(self, key: str) -> None:
        r = self._RECIPES[key]
        self.rank_spin.setValue(r["rank"])
        self.epochs_spin.setValue(r["epochs"])
        self.lr_edit.setText(f'{r["lr"]:g}')
        self.log.appendPlainText(
            t("qs_recipe_applied").format(
                rank=r["rank"], epochs=r["epochs"], lr=r["lr"]
            )
        )

    def _latest_progress(self) -> dict | None:
        """当前训练 job 的最近一条 step 记录（progress.jsonl）。"""
        if not self._train_job_id:
            return None
        p = ROOT / "output" / "daemon" / "jobs" / self._train_job_id / "progress.jsonl"
        if not p.is_file():
            return None
        last = None
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"step"' in line and '"global_step"' in line:
                        last = line
        except OSError:
            return None
        if not last:
            return None
        try:
            return json.loads(last)
        except ValueError:
            return None

    def _open_output_folder(self) -> None:
        out_dir = ROOT / "output" / "ckpt"
        os.startfile(str(out_dir))  # noqa: S606 — Windows 资源管理器

    def _open_advanced(self) -> None:
        """切到完整配置页（主窗口通过 setTabSwitcher 注入）。"""
        if callable(getattr(self, "_switch_advanced", None)):
            self._switch_advanced()

    # DaemonJobMixin 的日志出口（本页不 mix，直接写自己的 log）。
    def _submit_job(self, submit_fn, *, on_fail=None):
        try:
            resp = submit_fn()
        except Exception as e:  # noqa: BLE001
            self.log.appendPlainText(t("daemon_submit_failed", err=str(e)))
            if on_fail:
                on_fail()
            return None
        return resp.get("job_id") if isinstance(resp, dict) else None
