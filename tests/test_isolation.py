"""Phase 5: enforcement of the pipeline isolation constraint.

The architectural rule of this project is that `src/ingestion/` and `src/query/`
never import each other. This module is what makes that rule real rather than
aspirational -- it inspects imports structurally, then proves each package runs
with the other one absent.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path("src")
INGESTION = SRC / "ingestion"
QUERY = SRC / "query"


def _imported_modules(path: Path) -> set[str]:
    """Every module name imported by a file, via AST rather than text search.

    Substring matching would trip over the docstrings in these packages, which
    mention each other deliberately.
    """
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: ..query -> query
                modules.add(node.module or "")
            elif node.module:
                modules.add(node.module)

    return modules


def _python_files(package: Path) -> list[Path]:
    return sorted(p for p in package.glob("*.py"))


@pytest.mark.parametrize("path", _python_files(INGESTION), ids=lambda p: p.name)
def test_ingestion_never_imports_query(path):
    offending = {m for m in _imported_modules(path) if "query" in m.split(".")}
    assert not offending, f"{path} imports {offending}"


@pytest.mark.parametrize("path", _python_files(QUERY), ids=lambda p: p.name)
def test_query_never_imports_ingestion(path):
    offending = {m for m in _imported_modules(path) if "ingestion" in m.split(".")}
    assert not offending, f"{path} imports {offending}"


def _import_without(tmp_path: Path, drop: str, module: str) -> subprocess.CompletedProcess:
    """Copy src/ minus one pipeline, then import the other in a subprocess."""
    staged = tmp_path / "src"
    shutil.copytree(SRC, staged, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.rmtree(staged / drop)

    return subprocess.run(
        [sys.executable, "-c", f"import {module}; print('ok')"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_query_imports_with_ingestion_deleted(tmp_path):
    """The strongest available proof: the query side does not need ingestion."""
    result = _import_without(tmp_path, drop="ingestion", module="src.query")
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_ingestion_imports_with_query_deleted(tmp_path):
    result = _import_without(tmp_path, drop="query", module="src.ingestion")
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_shared_modules_hold_no_pipeline_logic():
    """config.py and schema.py are declarations; logic there recouples the two."""
    for name in ("config.py", "schema.py"):
        imports = _imported_modules(SRC / name)
        assert not any(
            "ingestion" in m or "query" in m or "chroma" in m or "langchain" in m
            for m in imports
        ), f"{name} imports pipeline machinery: {imports}"
