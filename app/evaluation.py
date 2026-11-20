"""Évaluation : métriques adaptées au déséquilibre et contrôle de non-régression.

- evaluate_model : rappel, précision, F1, PR-AUC, ROC-AUC, FPR, FNR, matrice de confusion (T10) ;
- describe_version / compare_to_baseline : comparaison d'une version à la référence (T14).
"""
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.preprocessing import SEED

ROOT = Path(__file__).resolve().parent.parent


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


# Description d'une version du modèle (Partie VI) : tout ce qu'il faut pour la comparer
# à une autre version. Sert à créer la référence (baseline) et la version courante.
def describe_version(model, model_name, threshold, val_metrics, csv_path, X, y, y_val):
    """Retourne un dictionnaire : code, modèle, bibliothèques, graine, seuil, métriques, données."""
    # Version du code : commit git courant (« inconnu » hors dépôt git),
    # suivi de « -modifie » si du code n'est pas commité (la version n'est alors pas traçable)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "app"], cwd=ROOT,
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


# Comparaison d'une nouvelle version à la référence (Partie VI, T14)
REGRESSION_TOLERANCE = 0.02  # baisse maximale tolérée (plan de test, T14)


def compare_to_baseline(baseline, current, tolerance=REGRESSION_TOLERANCE):
    """Compare deux versions sur la validation. Retourne (décision, raisons).

    - « rejeter »    : le rappel ou la PR-AUC baisse de plus de la tolérance ;
    - « investiguer » : la précision baisse de plus de la tolérance, ou les données ou
                        les bibliothèques ont changé (la comparaison n'est plus fiable) ;
    - « accepter »   : sinon.
    """
    rejeter = []
    investiguer = []

    # Métriques critiques : une baisse au-delà de la tolérance est une régression
    for metric in ("recall", "pr_auc"):
        baisse = baseline["validation_metrics"][metric] - current["validation_metrics"][metric]
        if baisse > tolerance:
            rejeter.append(f"{metric} en baisse de {baisse:.4f} (> {tolerance})")

    # Précision : une baisse signifie plus de fausses alertes, à analyser
    baisse = baseline["validation_metrics"]["precision"] - current["validation_metrics"]["precision"]
    if baisse > tolerance:
        investiguer.append(f"precision en baisse de {baisse:.4f} (> {tolerance})")

    # Contexte : si les données ou les bibliothèques changent, la comparaison n'est plus fiable
    if baseline["data"]["sha256"] != current["data"]["sha256"]:
        investiguer.append("jeu de données différent de celui de la référence")
    if baseline["libraries"] != current["libraries"]:
        investiguer.append("versions des bibliothèques différentes de la référence")

    if rejeter:
        return "rejeter", rejeter + investiguer
    if investiguer:
        return "investiguer", investiguer
    return "accepter", ["aucune dégradation au-delà de la tolérance"]
