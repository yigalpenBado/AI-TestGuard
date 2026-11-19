"""Tests de non-régression (Partie VI, T14).

- La règle de décision (accepter / investiguer / rejeter) est testée sur des cas simples.
- Le modèle sauvegardé est comparé à la référence figée reports/baseline.json :
  le test ÉCHOUE en cas de régression (décision « rejeter »).
"""
import copy
import json
import warnings
from pathlib import Path

import pytest

from src.data import preprocess_data, split_data
from src.train import (compare_to_baseline, describe_version, evaluate_model, load_data,
                       load_model, predict_fraud)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "raw" / "creditcard.csv"
BASELINE = ROOT / "reports" / "baseline.json"
MODEL = ROOT / "models" / "model.pkl"


# Référence minimale : seuls les champs utilisés par compare_to_baseline
REFERENCE = {
    "validation_metrics": {"recall": 0.86, "precision": 0.90, "pr_auc": 0.88},
    "data": {"sha256": "abc"},
    "libraries": {"scikit-learn": "1.9.1"},
}


def version(recall=0.86, precision=0.90, pr_auc=0.88, sha256="abc", sklearn="1.9.1"):
    """Nouvelle version construite à partir de la référence, avec les valeurs modifiées."""
    v = copy.deepcopy(REFERENCE)
    v["validation_metrics"].update(recall=recall, precision=precision, pr_auc=pr_auc)
    v["data"]["sha256"] = sha256
    v["libraries"]["scikit-learn"] = sklearn
    return v


# =============================================================================
# Règle de décision
# =============================================================================
def test_accepter():
    assert compare_to_baseline(REFERENCE, version())[0] == "accepter"                  # identique
    assert compare_to_baseline(REFERENCE, version(recall=0.90))[0] == "accepter"       # amélioration
    assert compare_to_baseline(REFERENCE, version(recall=0.85))[0] == "accepter"       # baisse ≤ 0,02


def test_rejeter():
    decision, reasons = compare_to_baseline(REFERENCE, version(recall=0.80))           # baisse de 0,06
    assert decision == "rejeter" and "recall" in reasons[0]
    decision, reasons = compare_to_baseline(REFERENCE, version(pr_auc=0.80))
    assert decision == "rejeter" and "pr_auc" in reasons[0]


def test_investiguer():
    decision, reasons = compare_to_baseline(REFERENCE, version(precision=0.80))        # plus de fausses alertes
    assert decision == "investiguer" and "precision" in reasons[0]
    decision, reasons = compare_to_baseline(REFERENCE, version(sha256="autre"))        # autres données
    assert decision == "investiguer" and "données" in reasons[0]
    decision, reasons = compare_to_baseline(REFERENCE, version(sklearn="1.10.0"))      # autres bibliothèques
    assert decision == "investiguer" and "bibliothèques" in reasons[0]


# =============================================================================
# Modèle sauvegardé comparé à la référence figée (ignoré si les fichiers sont absents, en CI)
# =============================================================================
@pytest.mark.skipif(not (DATASET.exists() and BASELINE.exists() and MODEL.exists()),
                    reason="jeu de données, référence ou modèle absent")
def test_modele_actuel_sans_regression():
    baseline = json.loads(BASELINE.read_text())
    model, threshold = load_model(MODEL)

    # Métriques sur le jeu de validation fixe (même découpage, même graine)
    X, y = preprocess_data(load_data(DATASET))
    _, X_val, _, _, y_val, _ = split_data(X, y)
    val_metrics = evaluate_model(y_val, *predict_fraud(model, X_val, threshold))
    current = describe_version(model, baseline["model"]["name"], threshold,
                               val_metrics, DATASET, X, y, y_val)

    decision, reasons = compare_to_baseline(baseline, current)

    if decision == "investiguer":
        warnings.warn(f"Non-régression à investiguer : {'; '.join(reasons)}")
    assert decision != "rejeter", f"Régression détectée : {'; '.join(reasons)}"
