"""Multi-resolution training section (多重分辨率) for the Preprocess tab.

Self-contained QGroupBox — deliberately NOT part of the knob table (it is a
structured setting: tier set + per-tier weights, not flat TOML scalars) and
persists through ``gui_settings.json`` via ``get_setting``/``set_setting``.

The tab wires the three actions:
  * run        → ``tasks.py multires --tiers … --weights …`` via ``_submit``
  * write      → inject the weighted blueprint into the active variant TOML
  * clear      → remove the inline blueprint (back to base.toml's)
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import t
from gui.theme import tok
from gui.widgets import action_button

try:
    from library.datasets.buckets import ALLOWED_TARGET_RES
except Exception:  # pragma: no cover — buckets is importable in tests, but stay safe
    ALLOWED_TARGET_RES = (512, 768, 896, 1024, 1280, 1536)

_TIERS_SETTING = "multires_tiers"
_WEIGHTS_SETTING = "multires_weights"


class MultiResSection(QGroupBox):
    """Tier checkboxes + per-tier weights + share preview."""

    changed = Signal()
    run_requested = Signal()
    write_variant_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(t("multires_section_title"), parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.setToolTip(t("multires_section_hint"))

        layout = QVBoxLayout(self)

        tier_row = QHBoxLayout()
        self._tier_group = QButtonGroup(self)
        # Multi-select tiers — QButtonGroup is exclusive by default, which
        # would make the checkboxes behave like radio buttons.
        self._tier_group.setExclusive(False)
        self._tier_checks: dict[int, QCheckBox] = {}
        for tier in ALLOWED_TARGET_RES:
            chk = QCheckBox(str(tier))
            chk.setToolTip(t("multires_tier_hint"))
            chk.toggled.connect(self._on_changed)
            self._tier_group.addButton(chk)
            self._tier_checks[tier] = chk
            tier_row.addWidget(chk)
        tier_row.addStretch(1)
        layout.addLayout(tier_row)

        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel(t("multires_weights_label")))
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("1024:2, 896:1")
        self.weights_edit.setToolTip(t("multires_weights_hint"))
        self.weights_edit.textChanged.connect(self._on_changed)
        weight_row.addWidget(self.weights_edit, 1)
        layout.addLayout(weight_row)

        self.share_label = QLabel("")
        self.share_label.setStyleSheet(f"color:{tok('text_dim')};")
        layout.addWidget(self.share_label)

        btn_row = QHBoxLayout()
        self.run_btn = action_button(
            t("multires_run"),
            variant="primary",
            tooltip=t("multires_run_hint"),
            on_click=lambda: self.run_requested.emit(),
        )
        btn_row.addWidget(self.run_btn)
        self.write_btn = action_button(
            t("multires_write_variant"),
            variant="info",
            tooltip=t("multires_write_hint"),
            on_click=lambda: self.write_variant_requested.emit(),
        )
        btn_row.addWidget(self.write_btn)
        self.clear_btn = action_button(
            t("multires_clear_variant"),
            tooltip=t("multires_clear_hint"),
            on_click=lambda: self.clear_requested.emit(),
        )
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._loading = False
        self._load_settings()
        self._update_share()

    # -- state --------------------------------------------------------------

    def selection(self) -> tuple[list[int], str]:
        tiers = sorted(t for t, c in self._tier_checks.items() if c.isChecked())
        return tiers, self.weights_edit.text().strip()

    def _on_changed(self, *_):
        if self._loading:
            return
        self._save_settings()
        self._update_share()
        self.changed.emit()

    def _load_settings(self):
        from gui import get_setting

        self._loading = True
        try:
            tiers = get_setting(_TIERS_SETTING) or []
            if isinstance(tiers, str):
                tiers = [p for p in tiers.replace("，", ",").split(",") if p.strip()]
            for tier_str in tiers:
                try:
                    chk = self._tier_checks.get(int(tier_str))
                except (TypeError, ValueError):
                    continue
                if chk is not None:
                    chk.setChecked(True)
            weights = get_setting(_WEIGHTS_SETTING)
            if isinstance(weights, str) and weights.strip():
                self.weights_edit.setText(weights)
        finally:
            self._loading = False

    def _save_settings(self):
        from gui import set_setting

        tiers, weights = self.selection()
        set_setting(_TIERS_SETTING, [str(t) for t in tiers])
        set_setting(_WEIGHTS_SETTING, weights)

    def _update_share(self):
        from library.preprocess.multires import (
            parse_tiers,
            parse_weights,
            share_summary,
            weights_to_repeats,
        )

        try:
            tiers = parse_tiers(self.selection()[0])
            repeats = weights_to_repeats(parse_weights(self.selection()[1], tiers))
        except ValueError as e:
            self.share_label.setText(f"⚠ {e}")
            return
        if not tiers:
            self.share_label.setText(t("multires_no_tiers"))
            return
        self.share_label.setText(
            f"{t('multires_share_prefix')} {share_summary(repeats)}"
        )
