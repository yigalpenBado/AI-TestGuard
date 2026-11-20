"""Entraînement du modèle de détection de fraude et pipeline complet.

- T09 : le scaler fait partie du Pipeline, il est donc ajusté sur le train uniquement ;
- T05 : le modèle et le seuil sont choisis sur la validation ; le test ne sert qu'à l'évaluation finale.

Lancement depuis la racine du projet : python app/training.py [--save-baseline]
"""
import json
import sys
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Racine du projet ajoutée au chemin d'import : permet « python app/training.py »
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.evaluation import REGRESSION_TOLERANCE, compare_to_baseline, describe_version, evaluate_model  # noqa: E402
from app.prediction import load_model, predict_fraud, sha256_of  # noqa: E402
from app.preprocessing import SEED, load_data, preprocess_data, split_data  # noqa: E402

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

# Sauvegarde du modèle avec son seuil : le seuil choisi sur la validation
# doit être réutilisé tel quel après rechargement.
def save_model(model, threshold, path):
    """Sauvegarde le modèle et son seuil (joblib), plus son empreinte dans <fichier>.sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, path)
    Path(str(path) + ".sha256").write_text(sha256_of(path))



# Verification sur la validation à 0.85, soit une marge de 10 points au-dessus de
# l'exigence de 0.75 (T11). Justification : la validation ne contient que ~94 fraudes,
# chaque fraude manquée fait varier le rappel d'environ 1 point ; le rappel mesuré
# sur la validation peut donc baisser sensiblement sur de nouvelles données.
VAL_MIN_RECALL = 0.85

# Tolérance numérique pour comparer les prédictions avant et après rechargement
TOLERANCE = 1e-12




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

    ## Contrôle de non-régression : comparaison de la version courante à la référence
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        decision, reasons = compare_to_baseline(baseline, version)
        comparison = {
            "baseline_commit": baseline["code"]["commit"],
            "current_commit": version["code"]["commit"],
            "tolerance": REGRESSION_TOLERANCE,
            "metrics": {
                k: {"baseline": baseline["validation_metrics"][k],
                    "current": version["validation_metrics"][k],
                    "difference": version["validation_metrics"][k] - baseline["validation_metrics"][k]}
                for k in ("recall", "precision", "f1", "pr_auc", "fpr")
            },
            "decision": decision,
            "reasons": reasons,
        }
        with open(ROOT / "reports" / "non_regression.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Non-régression : {decision.upper()} ({'; '.join(reasons)})")
