"""Tests de robustesse (Partie V) : comportement sur des données imparfaites.

Comportement attendu par scénario :
- REJET (ValueError)  : données manquantes, infinies, non numériques, schéma invalide ;
- TRAITEMENT + ALERTE : Amount extrême, distribution différente de l'entraînement ;
- dans tous les cas traités, les probabilités restent valides (contrat de sortie, 5.2).
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from app.prediction import predict_fraud
from app.preprocessing import FEATURES, check_input, split_data
from app.training import train_model


# Modèle entraîné et jeux issus de train_data (tests/conftest.py), créés une fois pour le fichier
@pytest.fixture(scope="module")
def splits(train_data):
    return split_data(*train_data)


@pytest.fixture(scope="module")
def X_train(splits):
    return splits[0]


@pytest.fixture(scope="module")
def X_test(splits):
    return splits[2]


@pytest.fixture(scope="module")
def model(splits):
    X_train, _, _, y_train, _, _ = splits
    return train_model(X_train, y_train, "random_forest")


def assert_valid_probabilities(probabilities):
    """Contrat de sortie (exemple 5.2)."""
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


# =============================================================================
# Scénarios REJETÉS : données manquantes, infinies, non numériques, schéma invalide
# =============================================================================
def test_invalid_inputs_are_rejected(model, X_test):
    nan = X_test.copy()
    nan.loc[nan.index[0], "V1"] = np.nan
    inf = X_test.copy()
    inf.loc[inf.index[0], "Amount"] = np.inf
    text = X_test.astype({"Amount": str})
    no_column = X_test.drop(columns="V14")

    for X, message in [(nan, "Valeurs manquantes"), (inf, "infinies"),
                       (text, "non numériques"), (no_column, "Colonnes manquantes")]:
        with pytest.raises(ValueError, match=message):
            predict_fraud(model, X)


# =============================================================================
# Scénarios TRAITÉS + ALERTE : Amount extrême, distribution différente
# =============================================================================
def test_unusual_inputs_raise_alerts(X_train, X_test):
    # Données normales : aucune alerte
    assert check_input(X_train, X_test) == []

    # Un Amount extrême : une seule alerte, « hors plage » (pas de fausse alerte de dérive)
    extreme = X_test.copy()
    extreme.loc[extreme.index[0], "Amount"] = 1e9
    alerts = check_input(X_train, extreme)
    assert len(alerts) == 1 and "Amount hors plage" in alerts[0]

    # V14 décalé de 3 écarts-types : alerte de dérive sur V14
    shifted = X_test.copy()
    shifted["V14"] += 3 * X_train["V14"].std()
    alerts = check_input(X_train, shifted)
    assert len(alerts) == 1 and "Dérive possible sur V14" in alerts[0]

    # Même décalage sur un petit lot (< 100 lignes) : pas de jugement de dérive
    assert check_input(X_train, shifted.head(10)) == []


def test_unusual_inputs_keep_valid_probabilities(model, X_test, X_train):
    extreme = X_test.assign(Amount=1e9)
    shifted = X_test.assign(V14=X_test["V14"] + 3 * X_train["V14"].std())

    for X in (X_test, extreme, shifted):
        _, probabilities = predict_fraud(model, X)
        assert_valid_probabilities(probabilities)


# =============================================================================
# Hypothesis : contrat de sortie sur des entrées valides générées automatiquement
# =============================================================================
@settings(max_examples=50, deadline=None)
@given(values=arrays(np.float64, (5, len(FEATURES)),
                     elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)))
def test_any_valid_input_gives_valid_probabilities(model, values):
    X = pd.DataFrame(values, columns=FEATURES)
    X["Time"] = X["Time"].abs()      # Time et Amount doivent rester positifs (contrat)
    X["Amount"] = X["Amount"].abs()

    _, probabilities = predict_fraud(model, X)

    assert_valid_probabilities(probabilities)
