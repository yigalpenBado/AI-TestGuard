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

# Métriques adaptées au déséquilibre (T10)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Racine du projet ajoutée au chemin d'import : permet « python src/train.py »,
# le bouton Run de VS Code et l'exécution cellule par cellule.
import sys

ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd().parent
)
sys.path.insert(0, str(ROOT))
from src.data import FEATURES, SEED, preprocess_data, split_data, validate_data

# Nous utiliserons le SEED 42 pour la reproductibilité des résultats (T06)
# class_weight="balanced" : les fraudes (0,17 %) pèsent autant que les transactions normales
MODELS = {
    "logistic_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=SEED
                ),
            ),
        ]
    ),
    "decision_tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=5, random_state=SEED
    ),
    "random_forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=200, random_state=SEED
    ),
}

# Entrainement du modèle.
# Nous utilisons clone() pour créer une copie neuve du modèle à chaque entraînement, 
# afin d'éviter que les modèles partagent des paramètres internes (T09).
# Nous entrainon les modeles suivants : la regression logistique, l'arbre de décision et la forêt aléatoire.
def train_model(X, y, model_name="logistic_regression"):
    """Entraîne le pipeline scaler + modèle. Le scaler n'est ajusté que sur X (T09)."""
    if model_name not in MODELS:
        raise ValueError(f"Modèle inconnu : {model_name}. Choix : {list(MODELS)}")
    print(f"\nEntraînement : {model_name}")
    return clone(MODELS[model_name]).fit(
        X, y
    )  # clone : copie neuve à chaque entraînement

# Apres l'entrainement, nous pouvons prédire les fraudes sur un jeu de données donné.
# Une validation est effectuée pour vérifier que les données sont valides (T12).
# avant de retourner les prédictions, nous calculons les probabilités de fraude et appliquons un seuil pour obtenir des décisions binaires (0/1).
def predict_fraud(model, X, threshold=0.5):
    """Retourne (décisions 0/1, probabilités de fraude). Rejette une entrée invalide (T12)."""
    validate_data(X, require_target=False)
    y_proba = model.predict_proba(X[FEATURES])[:, 1]
    return (y_proba >= threshold).astype(int), y_proba

# Après l'entrainement et la prédiction, nous pouvons évaluer les performances du modèle.
def evaluate_model(y_true, y_pred, y_proba):
    """Métriques adaptées au déséquilibre (T10) ; l'accuracy n'est donnée qu'à titre indicatif."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # Taux de faux positifs : transactions normales signalées à tort comme fraudes
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # Taux de faux négatifs : fraudes réelles non détectées
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "fpr": float(fpr),
        "fnr": float(fnr),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# Le seuil de décision est choisi sur la VALIDATION, jamais sur le test (T05, T11).
def choose_threshold(y_true, y_proba, min_recall=0.75):
    """Parmi les seuils qui atteignent le rappel minimal, retient celui de meilleure précision.

    min_recall = 0.75 : exigence de rappel du plan de test (T11).
    Lève ValueError si aucun seuil n'atteint ce rappel.
    """
    best_threshold = None
    best_precision = -1

    # On teste les seuils 0.01, 0.02, ..., 0.99
    for i in range(1, 100):
        threshold = i / 100

        # Décisions obtenues avec ce seuil
        y_pred = (y_proba >= threshold).astype(int)

        # Le seuil est écarté s'il ne détecte pas assez de fraudes
        recall = recall_score(y_true, y_pred, zero_division=0)
        if recall < min_recall:
            continue

        # Parmi les seuils restants, on garde celui qui a la meilleure précision
        # (à précision égale, le plus haut, car la boucle va du plus bas au plus haut)
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision >= best_precision:
            best_precision = precision
            best_threshold = threshold

    if best_threshold is None:
        raise ValueError(f"Aucun seuil n'atteint un rappel de {min_recall}.")

    return best_threshold


# Rappel visé sur la validation : 0.85, soit une marge de 10 points au-dessus de
# l'exigence de 0.75 (T11). Justification : la validation ne contient que ~94 fraudes,
# chaque fraude manquée fait varier le rappel d'environ 1 point ; le rappel mesuré
# sur la validation peut donc baisser sensiblement sur de nouvelles données.
VAL_MIN_RECALL = 0.85


if __name__ == "__main__":
    ## Chargement, prétraitement et découpage
    X, y = preprocess_data(pd.read_csv(ROOT / "data" / "raw" / "creditcard.csv"))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    ## Entraînement des deux modèles et comparaison sur la VALIDATION (T05)
    results, models, thresholds = {}, {}, {}
    for name in MODELS:
        models[name] = train_model(X_train, y_train, name)
        # Seuil réglé sur la validation (T11), puis métriques de validation à ce seuil
        _, y_proba_val = predict_fraud(models[name], X_val)
        thresholds[name] = choose_threshold(y_val, y_proba_val, min_recall=VAL_MIN_RECALL)
        results[name] = evaluate_model(y_val, *predict_fraud(models[name], X_val, thresholds[name]))
        print(
            f"\n{name} (validation, seuil {thresholds[name]}) :",
            {k: round(v, 4) for k, v in results[name].items()},
        )

    ## Choix du meilleur modèle selon la PR-AUC de validation
    best = max(results, key=lambda n: results[n]["pr_auc"])
    ## Évaluation finale, unique, sur le test avec le seuil choisi sur la validation
    test_metrics = evaluate_model(y_test, *predict_fraud(models[best], X_test, thresholds[best]))
    print(f"\nModèle retenu : {best} (seuil {thresholds[best]})")
    print("Test :", {k: round(v, 4) for k, v in test_metrics.items()})

    ## Sauvegarde du modèle et des preuves
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(models[best], ROOT / "models" / "model.pkl")
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump(
            {"seed": SEED, "model": best, "threshold": thresholds[best],
             "thresholds": thresholds, "validation": results, "test": test_metrics},
            f,
            indent=2,
        )
    print("\nModèle : models/model.pkl | Métriques : reports/metrics.json")
