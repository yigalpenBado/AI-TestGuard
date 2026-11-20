#%%
## Comparaison d'une nouvelle version à la référence (Partie VI)
# Nouvelle version proposée : forêt aléatoire avec max_depth=10 (arbres limités en profondeur),
# pour réduire le surapprentissage mesuré sur la référence (rappel train 1,00 contre 0,75 en test).
# Le modèle en production (models/model.pkl) et la référence (reports/baseline.json)
# ne sont PAS modifiés : la nouvelle version n'est que candidate.
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from sklearn.base import clone

from app.evaluation import compare_to_baseline, describe_version, evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import load_data, preprocess_data, split_data
from app.training import MODELS, VAL_MIN_RECALL, choose_threshold

CSV = ROOT / "data" / "raw" / "creditcard.csv"
baseline = json.loads((ROOT / "reports" / "baseline.json").read_text())

#%%
## Données : même découpage et même graine que la référence
X, y = preprocess_data(load_data(CSV))
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

#%%
## Nouvelle version : même pipeline que la référence, seul max_depth change
candidate = clone(MODELS["random_forest"]).set_params(model__max_depth=10)
candidate.fit(X_train, y_train)

# Seuil choisi sur la validation, avec la même règle que la référence
_, proba_val = predict_fraud(candidate, X_val)
threshold = choose_threshold(y_val, proba_val, min_recall=VAL_MIN_RECALL)
val_metrics = evaluate_model(y_val, *predict_fraud(candidate, X_val, threshold))
train_metrics = evaluate_model(y_train, *predict_fraud(candidate, X_train, threshold))

#%%
## Comparaison à la référence (sur la validation fixe)
current = describe_version(candidate, "random_forest", threshold, val_metrics, CSV, X, y, y_val)
decision, reasons = compare_to_baseline(baseline, current)

rows = ("recall", "precision", "f1", "pr_auc", "fpr", "fp", "fn")
print(f"{'métrique':<10} {'référence':>10} {'max_depth=10':>13} {'écart':>9}")
for k in rows:
    b, c = baseline["validation_metrics"][k], val_metrics[k]
    print(f"{k:<10} {b:>10.4f} {c:>13.4f} {c - b:>+9.4f}")
print(f"\nSeuil : référence {baseline['threshold']} | nouvelle version {threshold}")
print(f"Surapprentissage (rappel train - validation) : nouvelle version "
      f"{train_metrics['recall'] - val_metrics['recall']:+.4f}")
print(f"\nDécision : {decision.upper()} — {'; '.join(reasons)}")

#%%
## Enregistrement de la comparaison (preuve pour le rapport)
with open(ROOT / "reports" / "non_regression_max_depth10.json", "w") as f:
    json.dump({
        "baseline": {"commit": baseline["code"]["commit"],
                     "max_depth": baseline["model"]["params"]["max_depth"],
                     "threshold": baseline["threshold"], "validation": baseline["validation_metrics"]},
        "candidate": {"commit": current["code"]["commit"], "max_depth": 10, "threshold": threshold,
                      "validation": val_metrics, "train": train_metrics},
        "decision": decision, "reasons": reasons,
    }, f, indent=2)
