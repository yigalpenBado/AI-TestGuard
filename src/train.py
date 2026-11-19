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
# Chaque modèle est un Pipeline scaler + modèle : le scaler est ajusté sur le train
# uniquement (T09) et il est sauvegardé avec le modèle. Il ne change rien pour les arbres.
MODELS = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
    ]),
    "decision_tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=SEED)),
    ]),
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=SEED)),
    ]),
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

# =============================================================================
# Mise en oeuvre du pipeline complet : chargement, prétraitement, découpage, 
# entraînement, choix du seuil sur la validation, évaluation finale sur le test 
# et sauvegarde du modèle et des métriques.
# =============================================================================
# Chargement des données : erreurs explicites si le fichier est absent ou vide.
def load_data(path):
    """Lit le fichier CSV des transactions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Le fichier de données est vide : {path}")
    return df


# Sauvegarde du modèle avec son seuil : le seuil choisi sur la validation
# doit être réutilisé tel quel après rechargement.
def save_model(model, threshold, path):
    """Sauvegarde le modèle et son seuil dans un seul fichier (joblib)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, path)


# Rechargement du modèle sauvegardé : erreurs explicites si le fichier est absent ou invalide.
def load_model(path):
    """Recharge le modèle et son seuil. Retourne (model, threshold)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable : {path}")
    try:
        saved = joblib.load(path)
    except Exception as error:
        raise ValueError(f"Fichier modèle illisible : {path}") from error
    if not isinstance(saved, dict) or "model" not in saved or "threshold" not in saved:
        raise ValueError(f"Fichier modèle invalide (modèle ou seuil manquant) : {path}")
    return saved["model"], saved["threshold"]


# Verification sur la validation à 0.85, soit une marge de 10 points au-dessus de
# l'exigence de 0.75 (T11). Justification : la validation ne contient que ~94 fraudes,
# chaque fraude manquée fait varier le rappel d'environ 1 point ; le rappel mesuré
# sur la validation peut donc baisser sensiblement sur de nouvelles données.
VAL_MIN_RECALL = 0.85

# Tolérance numérique pour comparer les prédictions avant et après rechargement
TOLERANCE = 1e-12


# Description d'une version du modèle (Partie VI) : tout ce qu'il faut pour la comparer
# à une autre version. Sert à créer la référence (baseline) et la version courante.
def describe_version(model, model_name, threshold, val_metrics, csv_path, X, y, y_val):
    """Retourne un dictionnaire : code, modèle, bibliothèques, graine, seuil, métriques, données."""
    import hashlib
    import subprocess
    from datetime import datetime

    import numpy
    import sklearn

    # Version du code : commit git courant (« inconnu » hors dépôt git),
    # suivi de « -modifie » si du code n'est pas commité (la version n'est alors pas traçable)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
        if changes:
            commit += "-modifie"
    except Exception:
        commit = "inconnu"

    # Empreinte du fichier de données : change si une seule valeur du CSV change
    sha256 = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()

    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "code": {"commit": commit},
        "model": {"name": model_name, "params": model.named_steps["model"].get_params()},
        "libraries": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                      "numpy": numpy.__version__, "joblib": joblib.__version__},
        "seed": SEED,
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "data": {"sha256": sha256, "n_rows": len(X), "n_frauds": int(y.sum()),
                 "n_validation": len(y_val), "n_validation_frauds": int(y_val.sum())},
    }


if __name__ == "__main__":
    ## Chargement, prétraitement et découpage
    X, y = preprocess_data(load_data(ROOT / "data" / "raw" / "creditcard.csv"))
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
    print(f"\nModèle retenu : {best} (seuil {thresholds[best]})")

    ## Sauvegarde du pipeline et de son seuil, puis rechargement
    model_path = ROOT / "models" / "model.pkl"
    save_model(models[best], thresholds[best], model_path)
    loaded_model, loaded_threshold = load_model(model_path)

    ## Prédictions sur le test avec le pipeline RECHARGÉ (évaluation finale, unique)
    y_pred_test, y_proba_test = predict_fraud(loaded_model, X_test, loaded_threshold)
    test_metrics = evaluate_model(y_test, y_pred_test, y_proba_test)
    print("Test :", {k: round(v, 4) for k, v in test_metrics.items()})

    ## Surapprentissage : mêmes métriques sur le train, au même seuil, et écart train - test
    train_metrics = evaluate_model(y_train, *predict_fraud(loaded_model, X_train, loaded_threshold))
    overfitting = {
        k: round(train_metrics[k] - test_metrics[k], 4)
        for k in ("recall", "precision", "f1", "pr_auc")
    }
    print("Train :", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Écart train - test :", overfitting)

    ## Preuve de reproductibilité : mêmes probabilités avant et après rechargement
    _, y_proba_before = predict_fraud(models[best], X_test)
    max_diff = float(abs(y_proba_before - y_proba_test).max())
    reproducible = max_diff <= TOLERANCE and loaded_threshold == thresholds[best]
    print(f"Écart maximal avant / après rechargement : {max_diff} (tolérance {TOLERANCE})")

    ## Rapport de métriques et preuve de reproductibilité
    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump(
            {"seed": SEED, "model": best, "threshold": thresholds[best],
             "thresholds": thresholds, "validation": results, "test": test_metrics,
             "train": train_metrics, "ecart_train_test": overfitting},
            f,
            indent=2,
        )
    with open(ROOT / "reports" / "reproductibilite.json", "w") as f:
        json.dump(
            {"model_file": "models/model.pkl", "n_predictions": len(y_proba_test),
             "max_abs_diff": max_diff, "tolerance": TOLERANCE,
             "threshold_saved": thresholds[best], "threshold_loaded": loaded_threshold,
             "reproducible": reproducible},
            f,
            indent=2,
        )
    print("\nModèle : models/model.pkl | Rapports : reports/metrics.json, reports/reproductibilite.json")

    ## Description de la version courante (réécrite à chaque exécution)
    csv_path = ROOT / "data" / "raw" / "creditcard.csv"
    version = describe_version(loaded_model, best, loaded_threshold, results[best], csv_path, X, y, y_val)
    with open(ROOT / "reports" / "version_courante.json", "w") as f:
        json.dump(version, f, indent=2, default=str)
    print("Version courante : reports/version_courante.json")

    ## Référence (baseline) : créée UNIQUEMENT sur demande explicite, jamais écrasée.
    ## Pour la remplacer, il faut supprimer le fichier à la main : décision volontaire.
    baseline_path = ROOT / "reports" / "baseline.json"
    if "--save-baseline" in sys.argv:
        if baseline_path.exists():
            print(f"Référence NON modifiée : {baseline_path} existe déjà (supprimez-le pour la remplacer).")
        else:
            with open(baseline_path, "w") as f:
                json.dump(version, f, indent=2, default=str)
            print(f"Référence enregistrée : {baseline_path}")
