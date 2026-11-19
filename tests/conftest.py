"""Données de test partagées par tests/test_preprocessing.py et tests/test_integration.py.

pytest charge automatiquement les fixtures de ce fichier dans tous les fichiers de tests.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, TARGET


# =============================================================================
# Génération des données de test (jeux synthétiques, graines fixes)
# =============================================================================
@pytest.fixture
def valid_df():
    """Petit jeu conforme au contrat : 20 transactions dont 2 fraudes (graine 42)."""
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    # Time croissant à partir de 0
    df["Time"] = np.arange(n, dtype=float)
    # Amount : valeurs aléatoires entre 0 et 500
    df["Amount"] = rng.uniform(0, 500, n)
    # Cible : 18 transactions normales (0) et 2 fraudes (1)
    df[TARGET] = [0] * 18 + [1] * 2
    return df


@pytest.fixture(scope="module")
def train_data():
    """Jeu déséquilibré (10 % de fraudes) où V14 sépare les classes (graine 0).

    Permet de vérifier que le modèle détecte les fraudes sur des données déséquilibrées.
    Partagé par les tests unitaires et les tests d'intégration (scope="module" :
    généré une fois par fichier ; les tests ne le modifient pas).
    """
    rng = np.random.default_rng(0)
    n, n_fraud = 500, 50
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X["Time"] = rng.uniform(0, 172800, n)
    X["Amount"] = rng.uniform(0, 500, n)
    y = pd.Series([0] * (n - n_fraud) + [1] * n_fraud)
    X.loc[y == 1, "V14"] -= 4  # les fraudes ont un V14 plus faible
    return X, y
