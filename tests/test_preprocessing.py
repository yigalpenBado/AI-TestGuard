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

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, TARGET, preprocess_data, split_data, validate_data
from src.train import (MODELS, choose_threshold, evaluate_model, load_data, load_model,
                       predict_fraud, save_model, train_model)

DATASET = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"


# =============================================================================
# validate_data : un test par cas de la consigne 2.1
# =============================================================================
class TestValidateData:

    def test_donnees_valides_acceptees(self, valid_df):  # T01, valeurs limites
        assert validate_data(valid_df) is valid_df                      # toutes les colonnes requises
        assert len(validate_data(valid_df.head(1))) == 1                # une seule observation
        sans_cible = valid_df.drop(columns=TARGET)
        assert validate_data(sans_cible, require_target=False) is sans_cible  # prédiction sans cible
        zeros = valid_df.copy()
        zeros.loc[0, ["Time", "Amount"]] = 0.0                          # limite : 0 est autorisé
        assert validate_data(zeros) is zeros

    def test_colonne_absente(self, valid_df):  # T01
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            validate_data(valid_df.drop(columns="Time"))
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            validate_data(valid_df.drop(columns=TARGET))  # cible obligatoire à l'entraînement

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

    def test_montant_negatif(self, valid_df):  # valeur limite
        df = valid_df.copy()
        df.loc[0, "Amount"] = -0.01  # juste sous la limite
        with pytest.raises(ValueError, match="négatifs"):
            validate_data(df)

    def test_donnees_vides(self, valid_df):
        with pytest.raises(ValueError, match="non vide"):
            validate_data(valid_df.iloc[0:0])


# =============================================================================
# preprocess_data : préparation des variables et corrections documentées
# =============================================================================
class TestPreprocessData:

    def test_sortie_conforme(self, valid_df):  # T02, T03, dimensions
        X, y = preprocess_data(valid_df)
        assert X.shape == (len(valid_df), len(FEATURES))
        assert len(y) == len(X)
        assert list(X.columns) == FEATURES
        assert (X.dtypes == float).all()
        assert np.isfinite(X.to_numpy()).all()
        assert set(y.unique()) <= {0, 1}

    def test_une_seule_observation(self, valid_df):  # valeur limite
        X, y = preprocess_data(valid_df.head(1))
        assert X.shape == (1, len(FEATURES))
        assert len(y) == 1

    def test_corrections_documentees(self, valid_df):  # T08
        copy = valid_df.copy()
        df = pd.concat([valid_df, valid_df.head(3)])      # 3 doublons
        df["extra"] = 1.0                                  # colonne inattendue
        df = df[df.columns[::-1]]                          # colonnes dans le désordre
        X, y = preprocess_data(df)
        assert len(X) == len(y) == len(valid_df)           # doublons supprimés
        assert list(X.columns) == FEATURES                 # colonne en trop ignorée, ordre rétabli
        pd.testing.assert_frame_equal(valid_df, copy)      # entrée non modifiée

    def test_sans_cible(self, valid_df):
        X, y = preprocess_data(valid_df.drop(columns=TARGET), require_target=False)
        assert y is None
        assert len(X) == len(valid_df)


# =============================================================================
# split_data : partitions disjointes, stratifiées et reproductibles (T08)
# =============================================================================
class TestSplitData:

    def test_partitions(self, train_data):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(*train_data)
        train, val, test = set(X_train.index), set(X_val.index), set(X_test.index)
        assert not (train & val or train & test or val & test)              # disjointes
        assert (len(X_train), len(X_val), len(X_test)) == (300, 100, 100)  # 60/20/20
        for part in (y_train, y_val, y_test):                               # stratifiées
            assert abs(part.mean() - train_data[1].mean()) <= 0.01

    def test_reproductible(self, train_data):  # T06
        assert list(split_data(*train_data)[0].index) == list(split_data(*train_data)[0].index)


# =============================================================================
# train_model : entraînement du pipeline (testé sur les 3 modèles)
# =============================================================================
class TestTrainModel:

    def test_scaler_ajuste_sur_train_uniquement(self, train_data):  # T09
        X_train, _, X_test, y_train, *_ = split_data(*train_data)
        for name in MODELS:
            scaler = train_model(X_train, y_train, name).named_steps["scaler"]
            np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
            assert not np.allclose(scaler.mean_, X_test.mean().to_numpy())

    def test_reproductible(self, train_data):  # T06
        X, y = train_data
        for name in MODELS:
            p1 = predict_fraud(train_model(X, y, name), X)[1]
            p2 = predict_fraud(train_model(X, y, name), X)[1]
            np.testing.assert_array_equal(p1, p2)

    def test_modele_inconnu(self, train_data):
        with pytest.raises(ValueError, match="inconnu"):
            train_model(*train_data, "svm")


# =============================================================================
# predict_fraud : production des prédictions
# =============================================================================
class TestPredictFraud:

    @pytest.fixture
    def model(self, train_data):
        return train_model(*train_data)

    def test_probabilites_valides(self, model, train_data):  # T04
        X = train_data[0]
        y_pred, y_proba = predict_fraud(model, X)
        assert len(y_pred) == len(y_proba) == len(X)
        assert ((y_proba >= 0) & (y_proba <= 1)).all()
        assert set(np.unique(y_pred)) <= {0, 1}
        assert len(predict_fraud(model, X.head(1))[0]) == 1  # une seule observation

    def test_effet_du_seuil(self, model, train_data):  # valeurs limites
        X = train_data[0]
        assert predict_fraud(model, X, threshold=0.0)[0].sum() == len(X)  # tout est fraude
        assert predict_fraud(model, X, threshold=1.01)[0].sum() == 0      # rien n'est fraude
        n_bas = predict_fraud(model, X, threshold=0.1)[0].sum()
        n_haut = predict_fraud(model, X, threshold=0.9)[0].sum()
        assert n_bas >= n_haut                                            # seuil bas = plus d'alertes

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

    def test_matrice_de_confusion_fpr_fnr(self):
        # 3 normales dont 1 signalée à tort ; 2 fraudes dont 1 manquée
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        m = evaluate_model(y_true, y_pred, y_pred.astype(float))
        assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 1)
        assert m["fpr"] == 1 / 3  # FP / (FP + TN)
        assert m["fnr"] == 1 / 2  # FN / (FN + TP)

    def test_predictions_parfaites(self):
        y = np.array([0, 0, 1, 1])
        m = evaluate_model(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
        assert m["recall"] == m["precision"] == m["pr_auc"] == m["roc_auc"] == 1
        assert m["fpr"] == m["fnr"] == 0

    def test_accuracy_trompeuse(self):
        # Modèle qui prédit toujours « normal » : accuracy élevée mais aucune fraude détectée
        y_true = np.array([0] * 99 + [1])
        m = evaluate_model(y_true, np.zeros(100, dtype=int), np.zeros(100))
        assert m["accuracy"] == 0.99
        assert m["recall"] == 0


# =============================================================================
# choose_threshold : seuil réglé sur la validation (T11)
# =============================================================================
class TestChooseThreshold:

    # 4 normales et 4 fraudes ; la normale à 0.6 est plus « suspecte » que la fraude à 0.4
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])

    def test_meilleure_precision_au_rappel_minimal(self):
        # Rappel 0.75 : seuil 0.7 (3 fraudes, 0 fausse alerte) plutôt que 0.6 (1 fausse alerte)
        assert choose_threshold(self.y_true, self.y_proba, min_recall=0.75) == 0.7
        # Rappel 1 : il faut descendre à 0.4 pour attraper la fraude la moins suspecte
        assert choose_threshold(self.y_true, self.y_proba, min_recall=1.0) == 0.4

    def test_aucun_seuil_possible(self):
        with pytest.raises(ValueError, match="Aucun seuil"):
            choose_threshold(self.y_true, np.zeros(8))


# =============================================================================
# load_data, save_model, load_model : chargement et sauvegarde
# =============================================================================
class TestChargementSauvegarde:

    def test_load_data(self, valid_df, tmp_path):
        valid_df.to_csv(tmp_path / "data.csv", index=False)
        assert load_data(tmp_path / "data.csv").shape == valid_df.shape
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_data(tmp_path / "absent.csv")
        pd.DataFrame(columns=FEATURES).to_csv(tmp_path / "vide.csv", index=False)
        with pytest.raises(ValueError, match="vide"):
            load_data(tmp_path / "vide.csv")

    def test_save_puis_load_model(self, train_data, tmp_path):
        save_model(train_model(*train_data), 0.4, tmp_path / "model.pkl")
        model, threshold = load_model(tmp_path / "model.pkl")
        assert threshold == 0.4
        assert hasattr(model, "predict_proba")

    def test_load_model_erreurs(self, train_data, tmp_path):
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_model(tmp_path / "absent.pkl")
        (tmp_path / "corrompu.pkl").write_text("ceci n'est pas un modèle")
        with pytest.raises(ValueError, match="illisible"):
            load_model(tmp_path / "corrompu.pkl")
        joblib.dump(train_model(*train_data), tmp_path / "sans_seuil.pkl")  # modèle seul
        with pytest.raises(ValueError, match="invalide"):
            load_model(tmp_path / "sans_seuil.pkl")


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
