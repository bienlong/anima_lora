"""reseed never imports the scale line's builder / recipes (they rebuild the
seed of record and stay frozen), and its table is whole."""

import ast
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

FROZEN = {"cjk_scale.builder", "cjk_scale.recipes"}


def _imports(f: Path) -> set:
    out = set()
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return out


def test_no_frozen_imports():
    for f in [HOME / "run.py", *(HOME / "reseed").glob("*.py")]:
        hit = _imports(f) & FROZEN
        assert not hit, f"{f.name} imports {sorted(hit)}"


def test_table():
    from reseed.recipes import RECIPES
    from reseed.table import TABLE

    names = [t.name for t in TABLE]
    assert len(set(names)) == len(names)
    assert abs(sum(t.share for t in TABLE) - 1.5) < 1e-9
    for t in TABLE:
        assert t.recipe in RECIPES, t.name
        lo, hi = t.band
        assert 0 <= lo < hi <= 0.9, t.name
