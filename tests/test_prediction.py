"""Tests unitaires de la prédiction (app/prediction.py).

- predict_fraud : probabilités valides (T04), effet du seuil, entrée invalide rejetée (T12) ;
- save_model / load_model : sauvegarde du modèle avec son seuil, erreurs de chargement explicites.
"""
import joblib
import numpy as np
import pytest

from app.evaluation import evaluate_model
from app.prediction import load_model, predict_fraud, sha256_of
from app.training import save_model, train_model


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
# save_model / load_model : modèle sauvegardé avec son seuil
# =============================================================================
class TestLoadModel:

    def test_save_puis_load_model(self, train_data, tmp_path):
        save_model(train_model(*train_data), 0.4, tmp_path / "model.pkl")
        model, threshold = load_model(tmp_path / "model.pkl")
        assert threshold == 0.4
        assert hasattr(model, "predict_proba")

    def test_load_model_erreurs(self, train_data, tmp_path):
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_model(tmp_path / "absent.pkl")
        (tmp_path / "corrompu.pkl").write_text("ceci n'est pas un modèle")
        (tmp_path / "corrompu.pkl.sha256").write_text(sha256_of(tmp_path / "corrompu.pkl"))
        with pytest.raises(ValueError, match="illisible"):
            load_model(tmp_path / "corrompu.pkl")
        joblib.dump(train_model(*train_data), tmp_path / "sans_seuil.pkl")  # modèle seul
        (tmp_path / "sans_seuil.pkl.sha256").write_text(sha256_of(tmp_path / "sans_seuil.pkl"))
        with pytest.raises(ValueError, match="invalide"):
            load_model(tmp_path / "sans_seuil.pkl")
