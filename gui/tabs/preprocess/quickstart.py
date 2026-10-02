"""快速开始 pipeline card for the Preprocess tab.

把"选数据集 → 字幕 → 预处理缓存 → 可训练"四步做成一张带状态灯的卡片，
放在 Preprocess 页最上方——主次感：卡片是主流程，下方分区是高级细调。

纯展示 + 文件系统状态探测（Qt-free 逻辑可测）；执行动作经 tab 的
``_run_te``（勾选自动打标后的完整链），不自建任务。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from gui.theme import tok
from gui.widgets import action_button

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _count(dir_path: Path, exts: set[str]) -> int:
    if not dir_path.is_dir():
        return 0
    return sum(1 for f in dir_path.rglob("*") if f.suffix.lower() in exts)


def pipeline_status(source_dir: str | Path, cache_dir: str | Path) -> dict:
    """Four-step status from disk. Pure function — unit-testable headless."""
    src = Path(source_dir)
    cache = Path(cache_dir)
    n_imgs = _count(src, IMG_EXTS)
    n_captions = _count(src, {".txt"})
    n_npz = _count(cache, {".npz"})
    n_te = _count(cache, {".safetensors"})
    missing_captions = max(0, n_imgs - n_captions)
    return {
        "n_imgs": n_imgs,
        "n_captions": n_captions,
        "missing_captions": missing_captions,
        "n_npz": n_npz,
        "n_te": n_te,
        "dataset_ok": n_imgs > 0,
        "captions_ok": n_imgs > 0 and missing_captions == 0,
        "cache_ok": n_imgs > 0 and n_npz >= n_imgs and n_te >= n_imgs,
        "trainable": (
            n_imgs > 0 and n_npz >= n_imgs and n_te >= n_imgs and missing_captions == 0
        ),
    }


class QuickStartCard(QGroupBox):
    """四步流水线状态 + 一键补齐。刷新逻辑在 tab 侧（读分区当前值）。"""

    run_all_requested = Signal()  # 勾选自动打标 + 完整预处理
    refreshed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__("快速开始 · 训练流水线", parent)
        self._status: dict = pipeline_status("", "")
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(18)
        self._step_labels: list[QLabel] = []
        steps = ("① 选数据集", "② 图片字幕", "③ 预处理缓存", "④ 可训练")
        for col, name in enumerate(steps):
            head = QLabel(name)
            head.setStyleSheet(f"color:{tok('text_dim')};")
            grid.addWidget(head, 0, col)
            val = QLabel("—")
            val.setStyleSheet("font-weight:600;")
            grid.addWidget(val, 1, col)
            self._step_labels.append(val)

        row = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color:{tok('text_dim')};")
        row.addWidget(self.summary, 1)
        self.run_all_btn = action_button(
            "一键补齐（打标 + 预处理）",
            variant="primary",
            tooltip="按需自动打标并补齐全部缓存，完成后即可回到训练页点训练。",
            on_click=lambda: self.run_all_requested.emit(),
        )
        row.addWidget(self.run_all_btn)
        grid.addLayout(row, 2, 0, 1, 4)

    def refresh(self, status: dict) -> None:
        self._status = status
        vals = [
            (status["dataset_ok"], f"{status['n_imgs']} 张图"),
            (status["captions_ok"], f"{status['n_captions']} 条"),
            (status["cache_ok"], f"VAE {status['n_npz']} · TE {status['n_te']}"),
            (status["trainable"], "就绪" if status["trainable"] else "未就绪"),
        ]
        for label, (ok, text) in zip(self._step_labels, vals):
            color = tok("success") if ok else tok("warning")
            label.setStyleSheet(f"color:{color}; font-weight:600;")
            label.setText(f"{'✓' if ok else '○'} {text}")
        miss = status["missing_captions"]
        if status["dataset_ok"] and miss:
            self.summary.setText(f"⚠ {miss} 张图缺字幕——一键补齐会自动打标。")
        elif not status["dataset_ok"]:
            self.summary.setText("先在上方“图片预处理”选择数据集文件夹。")
        elif not status["cache_ok"]:
            self.summary.setText(
                f"缓存未齐（VAE {status['n_npz']}/{status['n_imgs']}，TE {status['n_te']}/{status['n_imgs']}）。"
            )
        elif status["trainable"]:
            self.summary.setText("全部就绪——去训练页点“训练”。")
        self.run_all_btn.setVisible(not status["trainable"])
        self.refreshed.emit()
