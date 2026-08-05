"""RFC 0001 is only enforceable if its subpackage table matches what is on disk."""

import importlib
import re
from pathlib import Path

import pytest

RFC = Path(__file__).resolve().parents[1] / "docs" / "rfcs" / "0001-package-layout.md"
SRC = Path(__file__).resolve().parents[1] / "src" / "tesserix_adk"

_ROW = re.compile(r"^\| `([a-z0-9_]+)` \| (.+?) \|$", re.MULTILINE)
_SECTION = re.compile(r"^## 2\. Subpackages$(.*?)^## 3\.", re.MULTILINE | re.DOTALL)


def documented_subpackages() -> dict[str, str]:
    section = _SECTION.search(RFC.read_text(encoding="utf-8"))
    assert section, "RFC 0001 section 2 not found"
    return dict(_ROW.findall(section.group(1)))


def actual_subpackages() -> set[str]:
    return {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}


def test_rfc_table_and_filesystem_agree() -> None:
    documented = set(documented_subpackages())
    assert documented, "RFC 0001 subpackage table did not parse"
    assert documented == actual_subpackages()


@pytest.mark.parametrize("name", sorted(documented_subpackages()))
def test_each_subpackage_imports_and_states_its_remit(name: str) -> None:
    module = importlib.import_module(f"tesserix_adk.{name}")
    assert module.__doc__, f"{name} has no remit docstring"
    assert module.__doc__.strip() == documented_subpackages()[name]
