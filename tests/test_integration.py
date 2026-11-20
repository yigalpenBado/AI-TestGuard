"""Tests d'intégration : le pipeline complet, de bout en bout (Partie IV).

Scénario : charger → vérifier le schéma → découper → entraîner → choisir le seuil
sur la validation → sauvegarder → recharger → prédire sur le test → métriques.

Les tests utilisent le jeu train_data (tests/conftest.py), écrit en CSV dans un dossier temporaire :
ils ne dépendent ni du vrai CSV ni du dossier models/.
"""
import numpy as np
import pytest

from app.evaluation import evaluate_model
from app.prediction import load_model, predict_fraud, sha256_of
from app.preprocessing import FEATURES, TARGET, load_data, preprocess_data, split_data
from app.training import TOLERANCE, choose_threshold, save_model, train_model


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """Dossier temporaire partagé par les tests du fichier."""
    return tmp_path_factory.mktemp("integration")


@pytest.fixture(scope="module")
def csv_path(workdir, train_data):
    """Écrit en CSV le jeu train_data des tests unitaires (500 transactions, 10 % de fraudes)."""
    X, y = train_data
    path = workdir / "creditcard.csv"
    X.assign(**{TARGET: y}).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def pipeline(csv_path, workdir):
    """Exécute le scénario complet une seule fois et renvoie les résultats intermédiaires."""
    X, y = preprocess_data(load_data(csv_path))                       # charger + schéma
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)  # découper
    model = train_model(X_train, y_train, "random_forest")            # entraîner sur le train
    _, proba_val = predict_fraud(model, X_val)
    threshold = choose_threshold(y_val, proba_val)                    # seuil sur la validation
    save_model(model, threshold, workdir / "model.pkl")              # sauvegarder
    loaded_model, loaded_threshold = load_model(workdir / "model.pkl")  # recharger
    y_pred, y_proba = predict_fraud(loaded_model, X_test, loaded_threshold)  # prédire sur le test
    metrics = evaluate_model(y_test, y_pred, y_proba)                 # rapport de métriques
    return {
        "X_train": X_train, "X_test": X_test,
        "model": model, "threshold": threshold,
        "loaded_model": loaded_model, "loaded_threshold": loaded_threshold,
        "y_pred": y_pred, "y_proba": y_proba, "metrics": metrics,
    }


class TestPipelineBoutEnBout:

    def test_scenario_complet(self, pipeline):
        # Le pipeline produit un rapport de métriques exploitable
        m = pipeline["metrics"]
        assert m["tp"] + m["fn"] + m["fp"] + m["tn"] == len(pipeline["X_test"])
        assert m["recall"] >= 0.75  # le seuil choisi sur la validation reste efficace

    def test_valeurs_finies_et_dimensions(self, pipeline):
        X_test = pipeline["X_test"]
        # Entrée du modèle : valeurs finies, colonnes du contrat
        assert np.isfinite(X_test.to_numpy()).all()
        assert list(X_test.columns) == FEATURES
        # Sortie du scaler : valeurs finies, mêmes dimensions que l'entrée
        scaled = pipeline["loaded_model"].named_steps["scaler"].transform(X_test)
        assert scaled.shape == X_test.shape
        assert np.isfinite(scaled).all()
        # Sortie du modèle : une prédiction par transaction
        assert len(pipeline["y_pred"]) == len(pipeline["y_proba"]) == len(X_test)

    def test_scaler_non_ajuste_sur_le_test(self, pipeline):
        scaler = pipeline["loaded_model"].named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, pipeline["X_train"].mean().to_numpy())
        assert not np.allclose(scaler.mean_, pipeline["X_test"].mean().to_numpy())

    def test_rechargement_reproductible(self, pipeline):
        # Le seuil est conservé et les probabilités sont identiques à la tolérance près
        assert pipeline["loaded_threshold"] == pipeline["threshold"]
        _, proba_avant = predict_fraud(pipeline["model"], pipeline["X_test"])
        np.testing.assert_allclose(pipeline["y_proba"], proba_avant, rtol=0, atol=TOLERANCE)

    def test_erreurs_explicites(self, pipeline, tmp_path):
        # Erreurs de chargement
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_data(tmp_path / "absent.csv")
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_model(tmp_path / "absent.pkl")
        (tmp_path / "corrompu.pkl").write_text("pas un modèle")
        (tmp_path / "corrompu.pkl.sha256").write_text(sha256_of(tmp_path / "corrompu.pkl"))
        with pytest.raises(ValueError, match="illisible"):
            load_model(tmp_path / "corrompu.pkl")
        # Erreur d'entrée sur le pipeline rechargé
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            predict_fraud(pipeline["loaded_model"], pipeline["X_test"].drop(columns="Amount"))
