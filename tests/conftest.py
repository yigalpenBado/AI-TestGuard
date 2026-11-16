import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXPLORATION = Path(__file__).resolve().parent.parent / "notebooks" / "exploration.py"


def load_functions(path):
    """Charge uniquement les imports, constantes et fonctions de exploration.py,
    sans exécuter l'exploration (lecture du CSV, graphiques)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef)
    tree.body = [n for n in tree.body if isinstance(n, keep)
                 or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("TARGET", "FEATURES"))]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


expl = load_functions(EXPLORATION)
FEATURES, TARGET = expl["FEATURES"], expl["TARGET"]
validate_data, preprocess_data = expl["validate_data"], expl["preprocess_data"]


@pytest.fixture
def valid_df():
    """Petit jeu de données synthétique conforme au contrat (graine fixe)."""
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df["Time"] = np.arange(n, dtype=float)
    df["Amount"] = rng.uniform(0, 500, n)
    df[TARGET] = [0] * 18 + [1] * 2
    return df
