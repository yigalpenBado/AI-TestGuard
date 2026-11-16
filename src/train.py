"""Entraînement, prédiction et évaluation du modèle de détection de fraude.

- T09 : le scaler fait partie du Pipeline, il est donc ajusté sur le train uniquement ;
- T05 : le modèle est choisi sur la validation ; le test ne sert qu'à l'évaluation finale.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
# Modèles de classification adaptés au déséquilibre (T10)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Métriques adaptées au déséquilibre (T10)
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, SEED, preprocess_data, split_data, validate_data

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "logistic_regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    "decision_tree": lambda: DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=SEED),
}


def train_model(X, y, model_name="logistic_regression"):
    """Entraîne le pipeline scaler + modèle. Le scaler n'est ajusté que sur X (T09)."""
    if model_name not in MODELS:
        raise ValueError(f"Modèle inconnu : {model_name}. Choix : {list(MODELS)}")
    model = Pipeline([("scaler", StandardScaler()), ("model", MODELS[model_name]())])
    return model.fit(X, y)


def predict_fraud(model, X, threshold=0.5):
    """Retourne (décisions 0/1, probabilités de fraude). Rejette une entrée invalide (T12)."""
    validate_data(X, require_target=False)
    y_proba = model.predict_proba(X[FEATURES])[:, 1]
    return (y_proba >= threshold).astype(int), y_proba


def evaluate_model(y_true, y_pred, y_proba):
    """Métriques adaptées au déséquilibre (T10) ; l'accuracy n'est donnée qu'à titre indicatif."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


if __name__ == "__main__":
    ## Chargement, prétraitement et découpage
    X, y = preprocess_data(pd.read_csv(ROOT / "data" / "raw" / "creditcard.csv"))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    ## Entraînement des deux modèles et comparaison sur la VALIDATION (T05)
    results, models = {}, {}
    for name in MODELS:
        models[name] = train_model(X_train, y_train, name)
        results[name] = evaluate_model(y_val, *predict_fraud(models[name], X_val))
        print(f"\n{name} (validation) :", {k: round(v, 4) for k, v in results[name].items()})

    ## Choix du meilleur modèle selon la PR-AUC de validation
    best = max(results, key=lambda n: results[n]["pr_auc"])
    test_metrics = evaluate_model(y_test, *predict_fraud(models[best], X_test))
    print(f"\nModèle retenu : {best}")
    print("Test :", {k: round(v, 4) for k, v in test_metrics.items()})

    ## Sauvegarde du modèle et des preuves
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(models[best], ROOT / "models" / "model.pkl")
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump({"seed": SEED, "model": best, "validation": results, "test": test_metrics}, f, indent=2)
    print("\nModèle : models/model.pkl | Métriques : reports/metrics.json")
