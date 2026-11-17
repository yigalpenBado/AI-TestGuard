"""Tests unitaires de validate_data et preprocess_data (plan de test : T01, T02, T03, T08, T12)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, TARGET, preprocess_data, split_data, validate_data

DATASET = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"


class TestValidateData:
    """Acceptation des données conformes, rejet explicite des autres."""

    def test_donnees_valides_acceptees(self, valid_df):  # T01, T02
        assert validate_data(valid_df) is valid_df                       # toutes les colonnes requises
        assert len(validate_data(valid_df.head(1))) == 1                 # une seule observation
        assert validate_data(valid_df.drop(columns=TARGET), require_target=False) is not None  # prédiction sans cible

    @pytest.mark.parametrize("transform, message", [
        (lambda df: df.drop(columns="Time"), "Colonnes manquantes"),             # T01 colonne absente
        (lambda df: df.drop(columns=TARGET), "Colonnes manquantes"),             # T01 cible absente
        (lambda df: pd.concat([df, df[["Amount"]]], axis=1), "dupliquées"),  # T01 colonne dupliquée
        (lambda df: df.astype({"Amount": str}), "non numériques"),        # T02 type incorrect
        # Une seule valeur fautive (ligne 0), le reste du jeu étant correct
        (lambda df: df.assign(V1=df["V1"].where(df.index != 0, np.nan)), "Valeurs manquantes"),  # T03
        (lambda df: df.assign(Amount=df["Amount"].where(df.index != 0, np.inf)), "infinies"),    # T03
        (lambda df: df.assign(Class=df[TARGET].where(df.index != 0, 2)), "0 et 1"),             # cible hors {0, 1}
        (lambda df: df.assign(Amount=-1.0), "négatifs"),                  # montant négatif
        (lambda df: df.iloc[0:0], "non vide"),                            # données vides
    ])
    def test_donnees_invalides_rejetees(self, valid_df, transform, message):
        with pytest.raises(ValueError, match=message):
            validate_data(transform(valid_df))


class TestPreprocessData:
    """Préparation des variables et corrections documentées."""

    def test_dimensions_coherentes(self, valid_df):
        X, y = preprocess_data(valid_df)
        assert X.shape == (len(valid_df), len(FEATURES))
        assert len(X) == len(y)
        assert list(X.columns) == FEATURES

    def test_sortie_finie(self, valid_df):  # T03
        X, _ = preprocess_data(valid_df)
        assert np.isfinite(X.to_numpy()).all()

    def test_types_sortie(self, valid_df):  # T02
        X, y = preprocess_data(valid_df)
        assert (X.dtypes == float).all()
        assert set(y.unique()) <= {0, 1}

    def test_doublons_supprimes(self, valid_df):  # T08
        df = pd.concat([valid_df, valid_df.head(3)])
        X, y = preprocess_data(df)
        assert len(X) == len(valid_df)
        assert not X.duplicated().any()

    def test_colonne_inattendue_ignoree(self, valid_df):
        valid_df["extra"] = 1.0
        X, _ = preprocess_data(valid_df)
        assert "extra" not in X.columns

    def test_ordre_des_colonnes_retabli(self, valid_df):
        X, _ = preprocess_data(valid_df[valid_df.columns[::-1]])
        assert list(X.columns) == FEATURES

    def test_une_seule_observation(self, valid_df):
        X, y = preprocess_data(valid_df.head(1))
        assert X.shape == (1, len(FEATURES)) and len(y) == 1

    def test_sans_cible(self, valid_df):
        X, y = preprocess_data(valid_df.drop(columns=TARGET), require_target=False)
        assert y is None and len(X) == len(valid_df)

    def test_entree_non_modifiee(self, valid_df):
        copy = valid_df.copy()
        preprocess_data(pd.concat([valid_df, valid_df.head(2)]))
        pd.testing.assert_frame_equal(valid_df, copy)

    def test_donnee_invalide_rejetee(self, valid_df):  # T12
        with pytest.raises(ValueError):
            preprocess_data(valid_df.drop(columns="Amount"))


@pytest.mark.skipif(not DATASET.exists(), reason="jeu de données absent")
class TestVraiJeuDeDonnees:
    """Contrôle du contrat sur le jeu Kaggle réel."""

    def test_creditcard_conforme(self):
        df = pd.read_csv(DATASET)
        X, y = preprocess_data(df)
        assert X.shape[1] == 30
        assert len(X) == len(df) - df.duplicated().sum()
        assert y.sum() == 492 - 19  # fraudes après suppression des doublons
