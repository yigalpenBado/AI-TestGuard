"""Tests unitaires de split_data, train_model, predict_fraud et evaluate_model."""
import numpy as np
import pytest

from train import MODELS, evaluate_model, predict_fraud, split_data, train_model


class TestSplitData:  # T08
    def test_partitions_disjointes(self, train_data):
        X_train, X_val, X_test, *_ = split_data(*train_data)
        idx = [set(X_train.index), set(X_val.index), set(X_test.index)]
        assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])

    def test_proportions_60_20_20(self, train_data):
        X_train, X_val, X_test, *_ = split_data(*train_data)
        assert (len(X_train), len(X_val), len(X_test)) == (300, 100, 100)

    def test_stratification(self, train_data):
        *_, y_train, y_val, y_test = split_data(*train_data)
        for part in (y_train, y_val, y_test):
            assert abs(part.mean() - train_data[1].mean()) <= 0.01

    def test_reproductible(self, train_data):
        a, b = split_data(*train_data)[0], split_data(*train_data)[0]
        assert list(a.index) == list(b.index)


class TestTrainModel:
    @pytest.mark.parametrize("name", list(MODELS))
    def test_entrainement(self, train_data, name):
        model = train_model(*train_data, name)
        assert hasattr(model, "predict_proba")

    def test_scaler_ajuste_sur_train_uniquement(self, train_data):  # T09
        X_train, X_val, _, y_train, *_ = split_data(*train_data)
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


class TestPredictFraud:
    @pytest.fixture
    def model(self, train_data):
        return train_model(*train_data)

    def test_probabilites_valides(self, model, train_data):  # T04
        y_pred, y_proba = predict_fraud(model, train_data[0])
        assert len(y_pred) == len(y_proba) == len(train_data[0])
        assert ((y_proba >= 0) & (y_proba <= 1)).all()
        assert set(np.unique(y_pred)) <= {0, 1}

    def test_seuil(self, model, train_data):
        n_low = predict_fraud(model, train_data[0], threshold=0.1)[0].sum()
        n_high = predict_fraud(model, train_data[0], threshold=0.9)[0].sum()
        assert n_low >= n_high

    def test_une_seule_observation(self, model, train_data):
        y_pred, y_proba = predict_fraud(model, train_data[0].head(1))
        assert len(y_pred) == len(y_proba) == 1

    def test_entree_invalide_rejetee(self, model, train_data):  # T12
        with pytest.raises(ValueError):
            predict_fraud(model, train_data[0].drop(columns="Amount"))

    def test_detecte_les_fraudes(self, model, train_data):
        y_pred, _ = predict_fraud(model, train_data[0])
        assert evaluate_model(train_data[1], y_pred, y_pred)["recall"] >= 0.8


class TestEvaluateModel:  # T10
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
        assert m["accuracy"] == 0.99 and m["recall"] == 0
