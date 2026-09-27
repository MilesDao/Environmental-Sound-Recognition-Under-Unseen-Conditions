import ast
import shlex
from pathlib import Path

import nbformat

from esr.cli import parse_args

NB = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_run.ipynb"


def _code_cells():
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _commands():
    """Every `!python scripts/<cmd>.py ...` line, with notebook variables filled in."""
    for cell in _code_cells():
        for line in cell.replace("\\\n", " ").splitlines():
            line = line.strip()
            if line.startswith("!python scripts/"):
                line = line.replace("$EXPERIMENT", "effnet_robust").replace("$DATA_PATH", "/data")
                script, *args = shlex.split(line[1:])[1:]
                yield Path(script).stem, args


def test_runner_clones_repo_and_uses_all_three_scripts():
    src = "\n".join(_code_cells())
    assert "git clone" in src and "REPO_URL" in src
    assert {name for name, _ in _commands()} == {"train", "evaluate", "compare"}


def test_every_script_call_parses_with_the_real_cli():
    calls = list(_commands())
    assert len(calls) >= 5
    for name, args in calls:
        parse_args([name, *args])


def test_python_parts_of_cells_are_valid():
    for cell in _code_cells():
        lines = [ln for ln in cell.replace("\\\n", " ").splitlines() if not ln.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(lines))
