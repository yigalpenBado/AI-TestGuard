"""Tests unitaires de validate_data et preprocess_data (plan de test : T01, T02, T03, T08, T12)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, TARGET, preprocess_data, split_data, validate_data

DATASET = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"


class TestValidateData:
    """Rejet explicite des données qui ne respectent pas le contrat."""

    def test_colonnes_requises_presentes(self, valid_df):  # T01
        assert validate_data(valid_df) is valid_df

    @pytest.mark.parametrize("col", ["Time", "V14", "Amount", TARGET])
    def test_colonne_absente(self, valid_df, col):  # T01
        with pytest.raises(ValueError, match="manquantes"):
            validate_data(valid_df.drop(columns=col))

    def test_colonne_dupliquee(self, valid_df):  # T01
        df = pd.concat([valid_df, valid_df[["Amount"]]], axis=1)
        with pytest.raises(ValueError, match="dupliquées"):
            validate_data(df)

    def test_valeur_manquante(self, valid_df):  # T03 / T12
        valid_df.loc[3, "V1"] = np.nan
        with pytest.raises(ValueError, match="manquantes"):
            validate_data(valid_df)

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_valeur_infinie(self, valid_df, value):  # T03
        valid_df.loc[0, "Amount"] = value
        with pytest.raises(ValueError, match="infinies"):
            validate_data(valid_df)

    def test_donnees_vides(self):
        with pytest.raises(ValueError, match="non vide"):
            validate_data(pd.DataFrame(columns=FEATURES + [TARGET]))

    def test_pas_un_dataframe(self):
        with pytest.raises(ValueError, match="DataFrame"):
            validate_data([[1, 2, 3]])

    def test_type_incorrect(self, valid_df):  # T02
        valid_df["Amount"] = valid_df["Amount"].astype(str)
        with pytest.raises(ValueError, match="non numériques"):
            validate_data(valid_df)

    @pytest.mark.parametrize("value", [2, -1])
    def test_cible_hors_0_1(self, valid_df, value):
        valid_df.loc[0, TARGET] = value
        with pytest.raises(ValueError, match="0 et 1"):
            validate_data(valid_df)

    @pytest.mark.parametrize("col", ["Time", "Amount"])
    def test_valeur_negative(self, valid_df, col):
        valid_df.loc[0, col] = -1.0
        with pytest.raises(ValueError, match="négatifs"):
            validate_data(valid_df)

    def test_une_seule_observation(self, valid_df):
        assert len(validate_data(valid_df.head(1))) == 1

    def test_sans_cible_en_prediction(self, valid_df):
        validate_data(valid_df.drop(columns=TARGET), require_target=False)


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
