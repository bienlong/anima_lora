import sys
from unittest.mock import patch

sys.stdout.reconfigure(line_buffering=True)
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication(sys.argv)
print("qt up")
from gui.tabs.config_tab import ConfigTab  # noqa: E402

ct = ConfigTab()
print("tab built")
merged = {
    "datasets": [
        {
            "subsets": [
                {"image_dir": "post_image_dataset/resized_t512"},
                {"image_dir": "post_image_dataset/resized_t768"},
            ]
        }
    ]
}
with patch.object(QMessageBox, "warning", return_value=QMessageBox.Ok) as w:
    all_empty = ct._warn_empty_dataset(merged)
print("guard all-empty ->", all_empty, "| warning shown:", w.called)
ok = ct._warn_empty_dataset({"datasets": [{"subsets": [{"image_dir": "tests"}]}]})
print("guard normal ->", ok)
with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes) as q:
    partial = ct._warn_empty_dataset(
        {
            "datasets": [
                {
                    "subsets": [
                        {"image_dir": "tests"},
                        {"image_dir": "post_image_dataset/resized_t768"},
                    ]
                }
            ]
        }
    )
print("guard partial(Yes) ->", partial, "| question shown:", q.called)
import gui.i18n.cn as cn  # noqa: E402

print(
    "i18n:",
    "train_empty_dataset" in cn.STRINGS,
    "train_partial_empty_dataset" in cn.STRINGS,
)
print("ALL_OK")
