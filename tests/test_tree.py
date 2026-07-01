import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKS = REPO_ROOT / "tests" / "tree_checks.py"


def run_check(name):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, str(CHECKS), name],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_train_tree_returns_booster():
    run_check("returns_booster")


def test_train_tree_predicts_probabilities():
    run_check("predicts_probabilities")


def test_train_tree_learns_separable_pattern():
    run_check("learns_separable_pattern")
