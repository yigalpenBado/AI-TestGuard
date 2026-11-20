#%%
## Import des bibliothèques nécessaires
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)

# Racine du projet (fonctionne en script ou en mode cellules)
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import preprocess_data, split_data
from app.training import MODELS, VAL_MIN_RECALL, choose_threshold, train_model

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

#%%
## Chargement, prétraitement et découpage (mêmes règles que app/training.py)
X, y = preprocess_data(pd.read_csv(ROOT / "data" / "raw" / "creditcard.csv"))
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
print(f"Fraudes en validation : {y_val.sum()} | en test : {y_test.sum()}")

#%%
## Entraînement des modèles et choix du seuil sur la VALIDATION
models = {}
thresholds = {}
probas_val = {}
for name in MODELS:
    models[name] = train_model(X_train, y_train, name)
    _, probas_val[name] = predict_fraud(models[name], X_val)
    thresholds[name] = choose_threshold(y_val, probas_val[name], min_recall=VAL_MIN_RECALL)

#%%
## Tableau comparatif des modèles (validation)
rows = []
for name in MODELS:
    y_pred = (probas_val[name] >= thresholds[name]).astype(int)
    metrics = evaluate_model(y_val, y_pred, probas_val[name])
    metrics["seuil"] = thresholds[name]
    rows.append(pd.Series(metrics, name=name))

table = pd.DataFrame(rows)
table = table[["seuil", "recall", "precision", "f1", "pr_auc", "roc_auc",
               "fpr", "fnr", "accuracy", "tp", "fn", "fp", "tn"]]
print(table.round(4).to_string())
table.round(4).to_csv(REPORTS / "comparaison_modeles.csv")

# %% [markdown]
# Les trois modèles atteignent le rappel visé (≥ 0,85) sur la validation, mais à un coût très différent :
# - forêt aléatoire : 9 fausses alertes, précision 0,90, PR-AUC 0,88 → meilleur modèle ;
# - régression logistique : 58 fausses alertes (précision 0,58) ;
# - arbre de décision : 628 fausses alertes (précision 0,12), PR-AUC 0,10 : inutilisable.
# Les trois ont une accuracy ≥ 0,99 : elle ne permet pas de les départager.

#%%
## Matrices de confusion (validation)
fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4))
for ax, name in zip(axes, MODELS):
    y_pred = (probas_val[name] >= thresholds[name]).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred, labels=[0, 1], display_labels=["Normale", "Fraude"],
        ax=ax, colorbar=False,
    )
    ax.set_title(f"{name} (seuil {thresholds[name]})")
plt.suptitle("Matrices de confusion (validation)\nLignes = réalité, colonnes = prédiction")
plt.tight_layout()
plt.savefig(REPORTS / "matrices_confusion.png")
plt.show()
# Lecture : [TN FP]
#           [FN TP]
# TN : normale prédite normale | FP : normale signalée à tort (fausse alerte)
# FN : fraude non détectée      | TP : fraude détectée

#%%
## Rapport de classification par classe (validation)
for name in MODELS:
    y_pred = (probas_val[name] >= thresholds[name]).astype(int)
    print(f"\n===== {name} =====")
    print(classification_report(y_val, y_pred, labels=[0, 1],
                                target_names=["Normale", "Fraude"], zero_division=0))

# %% [markdown]
# Pourquoi l'accuracy est trompeuse :
# la classe « Normale » obtient 1,00 partout, et l'accuracy globale vaut 1,00 (arrondie)
# même pour l'arbre de décision, dont 88 % des alertes sont fausses.
# Avec 0,17 % de fraudes, un modèle qui prédit toujours « normale » aurait 99,83 % d'accuracy
# sans détecter une seule fraude. Seules les lignes « Fraude » (précision, rappel) sont informatives.

#%%
## Courbes ROC et précision-rappel (validation)
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
for name in MODELS:
    RocCurveDisplay.from_predictions(y_val, probas_val[name], name=name, ax=ax_roc)
    PrecisionRecallDisplay.from_predictions(y_val, probas_val[name], name=name, ax=ax_pr)
ax_roc.set_title("Courbe ROC")
ax_pr.set_title("Courbe précision-rappel")
ax_pr.axhline(y_val.mean(), linestyle="--", color="grey", label="Hasard (taux de fraude)")
ax_pr.legend()
plt.tight_layout()
plt.savefig(REPORTS / "courbes_roc_pr.png")
plt.show()

# %% [markdown]
# La courbe ROC est flatteuse : les trois modèles ont une AUC ≥ 0,93, car le taux de faux
# positifs est divisé par 56 651 transactions normales.
# La courbe précision-rappel est plus parlante pour une classe rare : la forêt aléatoire
# (AP 0,88) garde une précision > 0,9 jusqu'à un rappel d'environ 0,85, puis chute ;
# l'arbre de décision (AP 0,10) ne dépasse jamais 0,12 de précision.

#%%
## Analyse des faux positifs et faux négatifs du modèle retenu (validation)
best = max(MODELS, key=lambda n: table.loc[n, "pr_auc"])
y_pred = (probas_val[best] >= thresholds[best]).astype(int)

analyse = X_val.copy()
analyse["reel"] = y_val.to_numpy()
analyse["predit"] = y_pred
analyse["proba"] = probas_val[best]

groupes = {
    "TP (fraude détectée)": analyse[(analyse.reel == 1) & (analyse.predit == 1)],
    "FN (fraude manquée)": analyse[(analyse.reel == 1) & (analyse.predit == 0)],
    "FP (fausse alerte)": analyse[(analyse.reel == 0) & (analyse.predit == 1)],
    "TN (normale)": analyse[(analyse.reel == 0) & (analyse.predit == 0)],
}
resume = pd.DataFrame({
    nom: {
        "nombre": len(g),
        "proba moyenne": g["proba"].mean(),
        "Amount médian": g["Amount"].median(),
        "V14 moyen": g["V14"].mean(),
        "V17 moyen": g["V17"].mean(),
    }
    for nom, g in groupes.items()
}).T
print(f"Modèle retenu : {best} (seuil {thresholds[best]})\n")
print(resume.round(3).to_string())

# %% [markdown]
# Faux négatifs (13 fraudes manquées) : leur probabilité moyenne est de 0,04 et leurs V14/V17
# sont proches de ceux des transactions normales (-1,1 et 0,5 contre -8,2 et -7,8 pour les fraudes
# détectées). Ce sont des fraudes qui « ressemblent » à des transactions normales : baisser
# le seuil ne suffirait pas à les attraper sans multiplier les fausses alertes.
# Leur montant médian (30) est aussi plus élevé que celui des fraudes détectées (1,6) :
# ce sont les fraudes les plus coûteuses pour la banque.
#
# Faux positifs (9 fausses alertes) : probabilité moyenne de 0,68 et V14/V17 fortement négatifs,
# comme de vraies fraudes. Ce sont des transactions normales au profil atypique ;
# leur montant médian est faible (2,7), ce qui limite la gêne pour le client.

#%%
## Évaluation finale UNIQUE du modèle retenu sur le TEST
## (seuil choisi sur la validation, aucun réglage n'est fait ici)
y_pred_test, y_proba_test = predict_fraud(models[best], X_test, thresholds[best])
test_metrics = evaluate_model(y_test, y_pred_test, y_proba_test)
print(f"{best} (test, seuil {thresholds[best]}) :")
print({k: round(v, 4) for k, v in test_metrics.items()})

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_test, labels=[0, 1], display_labels=["Normale", "Fraude"], colorbar=False,
)
plt.title(f"{best} - test (seuil {thresholds[best]})")
plt.savefig(REPORTS / "matrice_confusion_test.png")
plt.show()

# %% [markdown]
# Sur le test : 71 fraudes détectées sur 95 (rappel 0,747), 5 fausses alertes (précision 0,93).
# Le critère de précision (≥ 0,70) est atteint ; le critère de rappel (≥ 0,75) est manqué
# d'une seule fraude. Le modèle n'est donc pas accepté en l'état (T11).
# L'écart validation (0,86) / test (0,75) s'explique par le faible nombre de fraudes (~95) :
# la mesure du rappel varie de ±4 points environ. Le seuil n'est pas réajusté ici,
# car il ne doit pas être choisi à partir du jeu de test.
