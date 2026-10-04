"""The config-form field factory: ``_widget``/``_read`` map a TOML value to/from
an editor widget, plus the shared label/tooltip helpers built on top."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QWidget,
)

from gui.theme import tok
from gui.widgets._qt_utils import _no_wheel
from gui.widgets.sample_prompts import _SamplePromptsLauncher
from gui.widgets.target_res import _TargetResWidget

# flash4 is not supported yet (flash-attention-sm120 disabled)
_ATTN_MODES = ["flex", "flash"]

# Keys that are enum-ish in practice get a quick-pick dropdown instead of a
# free-text field (user request: "每个功能添加快捷的选择，而不是输入名字").
# Editable=True keeps the dropdown typeable for long-tail values (custom
# optimizers, exotic schedulers); closed sets are non-editable so a typo can
# never silently change semantics. A value outside a closed list is appended
# and selected rather than dropped.
_COMBO_CHOICES: dict[str, tuple[list[str], bool]] = {
    # key: (options, editable)
    "timestep_sampling": (
        [
            "sigma",
            "uniform",
            "sigmoid",
            "shift",
            "flux_shift",
            "qinglong_flux",
        ],
        False,
    ),
    "mixed_precision": (["no", "fp16", "bf16"], False),
    "lr_scheduler": (
        [
            "constant",
            "cosine",
            "cosine_with_restarts",
            "linear",
            "polynomial",
            "piecewise_constant",
            "adafactor",
        ],
        True,
    ),
    "optimizer_type": (
        [
            "AdamW",
            "AdamW8bit",
            "Lion",
            "Lion8bit",
            "Prodigy",
            "ProdigyPlusScheduleFree",
            "DAdaptAdam",
            "DAdaptLion",
            "SGDNesterov",
            "SGDNesterov8bit",
        ],
        True,
    ),
    "discrete_flow_shift": (["1.0", "2.0", "3.0", "4.0"], True),
}

# Keys that get a trailing open-icon: click → file picker → fills the field.
_FILE_BROWSE_KEYS = frozenset(
    {
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "network_weights",
    }
)


# 有中文习惯叫法的键：下拉显示本地化标签，_read 经 _value_map 还原为
# TOML 值。标签缺失的语言回退英文（t() 链），值本身永远是规范 TOML 值。
_OPTION_LABEL_KEYS = {
    "lr_scheduler": {
        "constant": "opt_sched_constant",
        "cosine": "opt_sched_cosine",
        "cosine_with_restarts": "opt_sched_cosine_restart",
        "linear": "opt_sched_linear",
        "polynomial": "opt_sched_polynomial",
        "piecewise_constant": "opt_sched_piecewise",
        "adafactor": "opt_sched_adafactor",
    },
}


def _enum_combo(key: str, v) -> QWidget:
    """Quick-pick combo for an enum-ish TOML key (see _COMBO_CHOICES)."""
    from gui.i18n import t

    options, editable = _COMBO_CHOICES[key]
    w = QComboBox()
    labels = _OPTION_LABEL_KEYS.get(key, {})
    value_map: dict[str, str] = {}
    for opt in options:
        lk = labels.get(opt)
        display = t(lk) if lk else opt
        w.addItem(display)
        value_map[display] = opt
    w.setProperty("_value_map", value_map)
    w.setEditable(editable)
    cur = "" if v is None else str(v)
    # 保存值 → 显示标签（有映射时）
    cur_display = next((d for d, val in value_map.items() if val == cur), cur)
    if w.findText(cur_display) < 0:
        if editable:
            w.setCurrentText(cur_display)
        elif cur_display:
            # Never silently change an unknown saved value.
            w.addItem(cur_display)
            w.setCurrentText(cur_display)
    else:
        w.setCurrentText(cur_display)
    return _no_wheel(w)


def _widget(v: Any, key: str = "") -> QWidget:
    if key == "target_res":
        sel = v if isinstance(v, (list, tuple)) else ([v] if v else [1024])
        return _TargetResWidget(sel)
    if key == "sample_prompts":
        return _SamplePromptsLauncher(v)
    if key == "sample_ratio":
        # Quick-pick combo; _read's QComboBox branch coerces back via the float orig.
        w = QComboBox()
        w.setEditable(True)
        w.addItems(["1.0", "0.5", "0.25", "0.1"])
        try:
            w.setCurrentText(f"{float(v):g}")
        except (TypeError, ValueError):
            w.setCurrentText(str(v))
        return _no_wheel(w)
    if key == "attn_mode":
        w = QComboBox()
        w.addItems(_ATTN_MODES)
        idx = w.findText(str(v))
        if idx >= 0:
            w.setCurrentIndex(idx)
        return _no_wheel(w)
    if key == "sample_decode_inline":
        # Tri-state "auto"/"true"/"false"; must precede the bool branch below
        # so a bool value gets this combo, not a checkbox that can't express "auto".
        w = QComboBox()
        w.addItems(["auto", "true", "false"])
        if v is None:
            cur = "auto"
        elif isinstance(v, bool):
            cur = "true" if v else "false"
        else:
            cur = str(v).strip().lower()
            if cur in ("", "none"):
                cur = "auto"
        idx = w.findText(cur)
        if idx >= 0:
            w.setCurrentIndex(idx)
        return _no_wheel(w)
    if key in _COMBO_CHOICES:
        # Quick-pick enums (before the type dispatch — a str value would
        # otherwise become a free-text QLineEdit).
        return _enum_combo(key, v)
    if isinstance(v, bool):
        w = QCheckBox()
        w.setChecked(v)
        return w
    if isinstance(v, int):
        w = QSpinBox()
        # 10k default cap guards against typos; overridden for fields that
        # legitimately exceed it or need negative values (lokr_factor = -1
        # 是"均衡分解"的约定值，必须可填).
        if key == "min_pixels":
            w.setRange(0, 100_000_000)
        elif key == "lokr_factor":
            w.setRange(-1, 10000)
        else:
            w.setRange(0, 10000)
        w.setValue(v)
        return _no_wheel(w)
    if isinstance(v, float):
        w = QLineEdit(f"{v:g}")
        return w
    if isinstance(v, list):
        w = QLineEdit(json.dumps(v))
        return w
    w = QLineEdit(str(v))
    # Quick-pick file dialogs for the model paths (user request: point-and-pick
    # instead of typing absolute paths).
    if key in _FILE_BROWSE_KEYS:
        attach_browse(w, mode="file")
    return w


def _read(w: QWidget, orig: Any = None) -> Any:
    if isinstance(w, _TargetResWidget):
        return w.value()
    if isinstance(w, _SamplePromptsLauncher):
        return w.value()
    if isinstance(w, QPlainTextEdit):
        return [
            ln.strip()
            for ln in w.toPlainText().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    if isinstance(w, QComboBox):
        txt = w.currentText()
        value_map = w.property("_value_map")
        if isinstance(value_map, dict) and txt in value_map:
            return value_map[txt]
        # Editable numeric combos round-trip as their orig type; plain string
        # combos (attn_mode, …) have string origs and fall through.
        if isinstance(orig, float):
            try:
                return float(txt)
            except ValueError:
                pass
        return txt
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    txt = w.text()
    if isinstance(orig, float):
        try:
            return float(txt)
        except ValueError:
            pass
    if isinstance(orig, list):
        try:
            return json.loads(txt)
        except (json.JSONDecodeError, ValueError):
            pass
    # Normalize pasted Windows backslashes — TOML escapes them (e.g. \U is invalid).
    if "\\" in txt:
        txt = txt.replace("\\", "/")
    return txt


class ClickableLabel(QLabel):
    """QLabel that emits `clicked` on left-click (field labels route into the
    explanation panel)."""

    clicked = Signal()

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):  # noqa: N802 — Qt event handler name
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


# Qt only word-wraps tooltips it recognizes as rich text, and that path clips
# CJK at the right edge instead of reflowing — so we wrap to plain text with
# explicit newlines ourselves, measuring in display columns (CJK glyphs = 2).
TOOLTIP_WRAP_COLS = 56
_TOOLTIP_WRAP_MIN_LEN = 80
# Cheap "is this already HTML?" probe (PySide6 doesn't expose Qt.mightBeRichText).
_LOOKS_RICH = re.compile(r"<[a-zA-Z!/]")


def _display_width(s: str) -> int:
    """Terminal-style display width: East-Asian wide/fullwidth glyphs = 2."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _wrap_plain(text: str, cols: int) -> str:
    """Greedy word-wrap to *cols* display columns; hard-breaks tokens with no
    spaces (CJK / long paths) so a single token can't overflow."""
    lines: list[str] = []
    cur = ""
    for word in text.split(" "):
        while _display_width(word) > cols:  # token alone exceeds the line
            head = ""
            for ch in word:
                if _display_width(head + ch) > cols:
                    break
                head += ch
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(head)
            word = word[len(head) :]
        if not cur:
            cur = word
        elif _display_width(cur) + 1 + _display_width(word) <= cols:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def wrap_tooltip(text: str | None) -> str | None:
    """Wrap a long, plain tooltip to multiple display-bounded lines.

    Leaves *text* unchanged if empty, short, multi-line, or rich text
    (author-formatted); otherwise wraps at :data:`TOOLTIP_WRAP_COLS`."""
    if not text or "\n" in text or len(text) <= _TOOLTIP_WRAP_MIN_LEN:
        return text
    if _LOOKS_RICH.search(text):
        return text
    return _wrap_plain(text, TOOLTIP_WRAP_COLS)


def attach_browse(edit: QLineEdit, *, mode: str, title: str = "") -> None:
    """Append an open-icon to a QLineEdit: click → file/folder dialog → fill.

    ``mode="dir"`` picks a folder (dataset source dirs), ``mode="file"`` picks
    a single file (model paths). Chosen paths are normalized to forward
    slashes so they paste into TOML safely (mirrors ``_read``'s fixup).
    """
    import os

    from PySide6.QtWidgets import QFileDialog, QStyle

    icon = edit.style().standardIcon(
        QStyle.SP_DirOpenIcon if mode == "dir" else QStyle.SP_FileDialogContentsView
    )
    action = edit.addAction(icon, QLineEdit.TrailingPosition)

    def _browse():
        cur = edit.text().strip()
        start = cur if cur and os.path.exists(cur) else ""
        if mode == "dir":
            d = QFileDialog.getExistingDirectory(edit, title, start)
        else:
            d = QFileDialog.getOpenFileName(edit, title, start)[0]
        if d:
            edit.setText(d.replace("\\", "/"))
            edit.setFocus()

    action.triggered.connect(_browse)


def make_field_label(
    text: str,
    *,
    style: str | None = None,
    tooltip: str | None = None,
    on_click=None,
) -> ClickableLabel:
    """Build a form field's :class:`ClickableLabel` (style + tooltip + click wire)."""
    lbl = ClickableLabel(text)
    if style:
        lbl.setStyleSheet(style)
    if tooltip:
        lbl.setToolTip(tooltip)
    if on_click is not None:
        lbl.clicked.connect(on_click)
    return lbl


def hint_label(text: str = "") -> QLabel:
    """A secondary/hint :class:`QLabel` in the theme's dim text color."""
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{tok('text_dim')};")
    return lbl
