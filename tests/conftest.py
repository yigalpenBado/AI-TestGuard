import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, TARGET

# Jeu de données synthétique conforme au contrat et du SEED 2.
@pytest.fixture
def valid_df():
    """Petit jeu de données synthétique conforme au contrat (graine fixe)."""
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df["Time"] = np.arange(n, dtype=float)
    df["Amount"] = rng.uniform(0, 500, n)
    df[TARGET] = [0] * 18 + [1] * 2
    print("\n=" * 80)
    print("\nJeu de données synthétique conforme au contrat : \n")
    print("\nJeu de données synthétique conforme au contrat :", df.head(5), "...\n")
    print("\n=" * 80)
    return df

#Jeu de données synthétique déséquilibré (10 % de fraudes) où V14 sépare les classes.
# afin de tester le modèle sur un jeu de données déséquilibré.
# et verifier que le modèle peut détecter les fraudes dans ce cas.
@pytest.fixture
def train_data():
    """Jeu synthétique déséquilibré (10 % de fraudes) où V14 sépare les classes."""
    rng = np.random.default_rng(0)
    n, n_fraud = 500, 50
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X["Time"] = rng.uniform(0, 172800, n)
    X["Amount"] = rng.uniform(0, 500, n)
    y = pd.Series([0] * (n - n_fraud) + [1] * n_fraud)
    X.loc[y == 1, "V14"] -= 4
    return X, y
