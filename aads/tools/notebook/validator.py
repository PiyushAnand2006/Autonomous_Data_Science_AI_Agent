"""
AADS Notebook Validator Tool — programmatically executes Jupyter Notebooks top-to-bottom
in an isolated namespace to verify syntax, execution correctness, and reproducibility.
"""

from __future__ import annotations

import ast
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from aads.core.logging import get_logger
from aads.core.schemas import NotebookValidationResult

logger = get_logger(__name__)


def validate_and_execute_notebook(
    notebook_path: str | Path,
    working_dir: str | Path | None = None,
) -> NotebookValidationResult:
    """Execute all code cells of a Jupyter Notebook in sequence to validate execution.

    Args:
        notebook_path: Absolute or relative path to the .ipynb file.
        working_dir: Directory to set as cwd during execution (e.g. 05_Notebook/).

    Returns:
        NotebookValidationResult indicating success, executed cell count, duration, and errors.
    """
    nb_path = Path(notebook_path).resolve()
    if not nb_path.exists():
        return NotebookValidationResult(
            success=False,
            executed_cells=0,
            total_cells=0,
            duration_seconds=0.0,
            errors=[f"Notebook file not found: {nb_path}"],
        )

    try:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as e:
        return NotebookValidationResult(
            success=False,
            executed_cells=0,
            total_cells=0,
            duration_seconds=0.0,
            errors=[f"Invalid notebook JSON format: {e}"],
        )

    cells = data.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    total_code_cells = len(code_cells)

    # Set working directory to notebook's parent folder so relative paths resolve
    target_cwd = Path(working_dir).resolve() if working_dir else nb_path.parent

    # Execution namespace
    exec_globals: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(nb_path),
    }

    start_time = time.perf_counter()
    executed_count = 0
    errors: list[str] = []

    import os
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    orig_cwd = _PROJECT_ROOT if _PROJECT_ROOT.exists() else Path.cwd()

    try:
        os.chdir(target_cwd)
        for idx, cell in enumerate(code_cells):
            source_lines = cell.get("source", [])
            source_code = "".join(source_lines) if isinstance(source_lines, list) else str(source_lines)

            # Skip empty cells
            if not source_code.strip():
                continue

            # Check AST syntax before execution
            try:
                tree = ast.parse(source_code)
            except SyntaxError as se:
                err_msg = f"SyntaxError in cell {idx + 1}: {se.msg} (line {se.lineno})"
                errors.append(err_msg)
                logger.warning("notebook_cell_syntax_error", cell_idx=idx + 1, error=err_msg)
                break

            # Execute cell
            try:
                exec(compile(tree, filename=f"<cell_{idx+1}>", mode="exec"), exec_globals)
                executed_count += 1
            except Exception as cell_err:
                err_msg = f"RuntimeError in cell {idx + 1}: {type(cell_err).__name__}: {str(cell_err)}"
                errors.append(err_msg)
                logger.warning("notebook_cell_runtime_error", cell_idx=idx + 1, error=err_msg)
                break

    except Exception as outer_err:
        errors.append(f"Execution setup failed: {outer_err}")
    finally:
        os.chdir(orig_cwd)

    duration = round(time.perf_counter() - start_time, 4)
    success = (len(errors) == 0) and (executed_count == total_code_cells or total_code_cells == 0)

    result = NotebookValidationResult(
        success=success,
        executed_cells=executed_count,
        total_cells=total_code_cells,
        duration_seconds=duration,
        errors=errors,
        validated_at=datetime.utcnow(),
    )

    logger.info(
        "notebook_validation_finished",
        success=success,
        executed_cells=executed_count,
        total_cells=total_code_cells,
        duration=duration,
    )
    return result
