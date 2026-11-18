"""Tests unitaires du pipeline de détection de fraude (Partie II, activité 2).

Fonctions testées :
- validate_data(df)                       : contrôle du schéma et des types ;
- preprocess_data(df)                     : préparation des variables ;
- split_data(X, y)                        : découpage train/val/test ;
- train_model(X, y)                       : entraînement du pipeline ;
- predict_fraud(model, X)                 : production des prédictions ;
- evaluate_model(y_true, y_pred, y_proba) : calcul des métriques.

Exigences du plan de test couvertes :
T01 colonnes présentes, T02 types, T03 valeurs finies, T04 probabilité valide,
T06 reproductibilité, T08 partitions disjointes et stratifiées, T09 scaler ajusté
sur le train, T10 métriques adaptées au déséquilibre, T12 entrée invalide rejetée.

Chaque test vérifie un comportement attendu (valeur, message d'erreur, dimension),
et non simplement l'absence d'exception.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, TARGET, preprocess_data, split_data, validate_data
from src.train import MODELS, evaluate_model, predict_fraud, train_model

DATASET = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"


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


@pytest.fixture
def train_data():
    """Jeu déséquilibré (10 % de fraudes) où V14 sépare les classes (graine 0).

    Permet de vérifier que le modèle détecte les fraudes sur des données déséquilibrées.
    """
    rng = np.random.default_rng(0)
    n, n_fraud = 500, 50
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X["Time"] = rng.uniform(0, 172800, n)
    X["Amount"] = rng.uniform(0, 500, n)
    y = pd.Series([0] * (n - n_fraud) + [1] * n_fraud)
    X.loc[y == 1, "V14"] -= 4  # les fraudes ont un V14 plus faible
    return X, y


# =============================================================================
# validate_data : rejet explicite des données qui ne respectent pas le contrat
# =============================================================================
class TestValidateData:

    def test_donnees_valides_acceptees(self, valid_df):  # T01
        assert validate_data(valid_df) is valid_df

    def test_une_seule_observation(self, valid_df):  # valeur limite
        assert len(validate_data(valid_df.head(1))) == 1

    def test_sans_cible_en_prediction(self, valid_df):
        df = valid_df.drop(columns=TARGET)
        assert validate_data(df, require_target=False) is df

    def test_colonne_absente(self, valid_df):  # T01
        df = valid_df.drop(columns="Time")
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            validate_data(df)

    def test_cible_absente(self, valid_df):  # T01
        df = valid_df.drop(columns=TARGET)
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            validate_data(df)

    def test_colonne_dupliquee(self, valid_df):  # T01
        df = pd.concat([valid_df, valid_df[["Amount"]]], axis=1)
        with pytest.raises(ValueError, match="dupliquées"):
            validate_data(df)

    def test_type_incorrect(self, valid_df):  # T02
        df = valid_df.copy()
        df["Amount"] = df["Amount"].astype(str)
        with pytest.raises(ValueError, match="non numériques"):
            validate_data(df)

    def test_valeur_manquante(self, valid_df):  # T03
        df = valid_df.copy()
        df.loc[0, "V1"] = np.nan  # une seule valeur fautive
        with pytest.raises(ValueError, match="Valeurs manquantes"):
            validate_data(df)

    def test_valeur_infinie(self, valid_df):  # T03
        df = valid_df.copy()
        df.loc[0, "Amount"] = np.inf  # une seule valeur fautive
        with pytest.raises(ValueError, match="infinies"):
            validate_data(df)

    def test_cible_hors_0_1(self, valid_df):
        df = valid_df.copy()
        df.loc[0, TARGET] = 2  # une seule valeur fautive
        with pytest.raises(ValueError, match="0 et 1"):
            validate_data(df)

    def test_montant_negatif(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "Amount"] = -0.01  # juste sous la limite
        with pytest.raises(ValueError, match="négatifs"):
            validate_data(df)

    def test_montant_et_time_nuls_acceptes(self, valid_df):  # valeur limite
        df = valid_df.copy()
        df.loc[0, ["Time", "Amount"]] = 0.0  # 0 est autorisé
        assert validate_data(df) is df

    def test_donnees_vides(self, valid_df):
        with pytest.raises(ValueError, match="non vide"):
            validate_data(valid_df.iloc[0:0])

    def test_pas_un_dataframe(self, valid_df):
        with pytest.raises(ValueError, match="DataFrame"):
            validate_data(valid_df.to_numpy())


# =============================================================================
# preprocess_data : préparation des variables et corrections documentées
# =============================================================================
class TestPreprocessData:

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
        assert len(X) == len(y) == len(valid_df)
        assert not X.duplicated().any()

    def test_colonne_inattendue_ignoree(self, valid_df):
        valid_df["extra"] = 1.0
        X, _ = preprocess_data(valid_df)
        assert "extra" not in X.columns

    def test_ordre_des_colonnes_retabli(self, valid_df):
        X, _ = preprocess_data(valid_df[valid_df.columns[::-1]])
        assert list(X.columns) == FEATURES

    def test_une_seule_observation(self, valid_df):  # valeur limite
        X, y = preprocess_data(valid_df.head(1))
        assert X.shape == (1, len(FEATURES))
        assert len(y) == 1

    def test_sans_cible(self, valid_df):
        X, y = preprocess_data(valid_df.drop(columns=TARGET), require_target=False)
        assert y is None
        assert len(X) == len(valid_df)

    def test_entree_non_modifiee(self, valid_df):
        copy = valid_df.copy()
        preprocess_data(pd.concat([valid_df, valid_df.head(2)]))
        pd.testing.assert_frame_equal(valid_df, copy)

    def test_donnee_invalide_rejetee(self, valid_df):  # T12
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            preprocess_data(valid_df.drop(columns="Amount"))


# =============================================================================
# split_data : partitions disjointes, stratifiées et reproductibles (T08)
# =============================================================================
class TestSplitData:

    def test_partitions_disjointes(self, train_data):
        X_train, X_val, X_test, *_ = split_data(*train_data)
        train, val, test = set(X_train.index), set(X_val.index), set(X_test.index)
        assert not train & val
        assert not train & test
        assert not val & test

    def test_proportions_60_20_20(self, train_data):
        X_train, X_val, X_test, *_ = split_data(*train_data)
        assert (len(X_train), len(X_val), len(X_test)) == (300, 100, 100)

    def test_stratification(self, train_data):
        *_, y_train, y_val, y_test = split_data(*train_data)
        for part in (y_train, y_val, y_test):
            assert abs(part.mean() - train_data[1].mean()) <= 0.01

    def test_reproductible(self, train_data):  # T06
        a = split_data(*train_data)[0]
        b = split_data(*train_data)[0]
        assert list(a.index) == list(b.index)


# =============================================================================
# train_model : entraînement du pipeline
# =============================================================================
class TestTrainModel:

    @pytest.mark.parametrize("name", list(MODELS))
    def test_entrainement(self, train_data, name):
        model = train_model(*train_data, name)
        assert hasattr(model, "predict_proba")

    def test_scaler_ajuste_sur_train_uniquement(self, train_data):  # T09
        X_train, _, _, y_train, *_ = split_data(*train_data)
        scaler = train_model(X_train, y_train).named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
        assert not np.allclose(scaler.mean_, train_data[0].mean().to_numpy())

    def test_modele_inconnu(self, train_data):
        with pytest.raises(ValueError, match="inconnu"):
            train_model(*train_data, "svm")

    @pytest.mark.parametrize("name", list(MODELS))
    def test_reproductible(self, train_data, name):  # T06
        X, y = train_data
        p1 = predict_fraud(train_model(X, y, name), X)[1]
        p2 = predict_fraud(train_model(X, y, name), X)[1]
        np.testing.assert_array_equal(p1, p2)


# =============================================================================
# predict_fraud : production des prédictions
# =============================================================================
class TestPredictFraud:

    @pytest.fixture
    def model(self, train_data):
        return train_model(*train_data)

    def test_probabilites_valides(self, model, train_data):  # T04
        y_pred, y_proba = predict_fraud(model, train_data[0])
        assert len(y_pred) == len(y_proba) == len(train_data[0])
        assert ((y_proba >= 0) & (y_proba <= 1)).all()
        assert set(np.unique(y_pred)) <= {0, 1}

    def test_seuils_limites(self, model, train_data):  # valeurs limites
        X = train_data[0]
        assert predict_fraud(model, X, threshold=0.0)[0].sum() == len(X)  # tout est fraude
        assert predict_fraud(model, X, threshold=1.01)[0].sum() == 0      # rien n'est fraude

    def test_seuil_plus_bas_plus_d_alertes(self, model, train_data):
        n_low = predict_fraud(model, train_data[0], threshold=0.1)[0].sum()
        n_high = predict_fraud(model, train_data[0], threshold=0.9)[0].sum()
        assert n_low >= n_high

    def test_une_seule_observation(self, model, train_data):  # valeur limite
        y_pred, y_proba = predict_fraud(model, train_data[0].head(1))
        assert len(y_pred) == len(y_proba) == 1

    def test_entree_invalide_rejetee(self, model, train_data):  # T12
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            predict_fraud(model, train_data[0].drop(columns="Amount"))

    def test_detecte_les_fraudes(self, model, train_data):
        y_pred, y_proba = predict_fraud(model, train_data[0])
        assert evaluate_model(train_data[1], y_pred, y_proba)["recall"] >= 0.8


# =============================================================================
# evaluate_model : métriques adaptées au déséquilibre (T10)
# =============================================================================
class TestEvaluateModel:

    def test_predictions_parfaites(self):
        y = np.array([0, 0, 1, 1])
        m = evaluate_model(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
        assert m["recall"] == m["precision"] == m["pr_auc"] == m["roc_auc"] == 1

    def test_matrice_de_confusion(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        m = evaluate_model(y_true, y_pred, y_pred.astype(float))
        assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 1)

    def test_accuracy_trompeuse(self):
        # Modèle qui prédit toujours « normal » : accuracy élevée mais aucune fraude détectée
        y_true = np.array([0] * 99 + [1])
        y_pred = np.zeros(100, dtype=int)
        m = evaluate_model(y_true, y_pred, np.zeros(100))
        assert m["accuracy"] == 0.99
        assert m["recall"] == 0


# =============================================================================
# Jeu de données réel (ignoré si le CSV est absent, par exemple en CI)
# =============================================================================
@pytest.mark.skipif(not DATASET.exists(), reason="jeu de données absent")
class TestVraiJeuDeDonnees:

    def test_creditcard_conforme(self):
        df = pd.read_csv(DATASET)
        X, y = preprocess_data(df)
        assert X.shape[1] == 30
        assert len(X) == len(df) - df.duplicated().sum()
        assert y.sum() == 492 - 19  # fraudes après suppression des doublons
