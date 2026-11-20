#%%
## Analyse des écarts de performance par segment (Partie VII) avec Fairlearn
# IMPORTANT : le jeu ne contient AUCUN attribut démographique (âge, sexe, localisation).
# Les segments ci-dessous sont OPÉRATIONNELS, construits à partir des seules variables
# interprétables (Amount, Time). Ils ne permettent pas de conclure à l'équité envers
# des groupes de clients. Aucun attribut n'est inféré à partir de V1 à V28 (anonymisées).
import sys
from pathlib import Path

import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate
from sklearn.metrics import precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from app.prediction import load_model, predict_fraud
from app.preprocessing import SEED, load_data, preprocess_data, split_data

#%%
## Modèle sauvegardé et jeu de test
model, threshold = load_model(ROOT / "models" / "model.pkl")
X, y = preprocess_data(load_data(ROOT / "data" / "raw" / "creditcard.csv"))
_, _, X_test, _, _, y_test = split_data(X, y)
y_pred, _ = predict_fraud(model, X_test, threshold)

#%%
## Segments opérationnels
# Tranche de montant : bornes simples et lisibles
montant = pd.cut(X_test["Amount"], bins=[0, 10, 100, float("inf")], right=False,
                 labels=["< 10", "10 - 100", ">= 100"])
# Période : heure relative à la première transaction (pas une heure réelle connue)
heure = (X_test["Time"] // 3600) % 24
periode = heure.map(lambda h: "nuit (0h-6h)" if h < 6 else "jour (6h-24h)")

#%%
## Métriques par segment avec intervalles de confiance à 95 % (bootstrap)
metrics = {
    "rappel": recall_score,
    "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
    "FPR": false_positive_rate,
}

rows = []
for nom, segment in [("montant", montant), ("periode", periode)]:
    mf = MetricFrame(metrics=metrics, y_true=y_test.to_numpy(), y_pred=y_pred,
                     sensitive_features=segment.to_numpy(),
                     n_boot=200, ci_quantiles=[0.025, 0.975], random_state=SEED)
    low, high = mf.by_group_ci
    # Effectifs par groupe (sans bootstrap)
    effectifs = pd.DataFrame({"groupe": segment.to_numpy(), "fraude": y_test.to_numpy()})
    effectifs = effectifs.groupby("groupe", observed=True)["fraude"].agg(["sum", "count"])
    for groupe in mf.by_group.index:
        row = {"segment": nom, "groupe": groupe,
               "fraudes": int(effectifs.loc[groupe, "sum"]),
               "transactions": int(effectifs.loc[groupe, "count"])}
        for k in ("rappel", "precision", "FPR"):
            row[k] = mf.by_group.loc[groupe, k]
            row[f"{k} IC95"] = f"[{low.loc[groupe, k]:.3f} ; {high.loc[groupe, k]:.3f}]"
        rows.append(row)

    # Écart maximal entre groupes, avec son intervalle de confiance
    d_low, d_high = mf.difference_ci()
    print(f"\n=== {nom} : écart max - min entre groupes ===")
    for k in ("rappel", "precision", "FPR"):
        print(f"{k:<10} {mf.difference()[k]:.3f}  IC95 [{d_low[k]:.3f} ; {d_high[k]:.3f}]")

table = pd.DataFrame(rows).set_index(["segment", "groupe"])
print("\n", table.round(4).to_string())
(ROOT / "reports").mkdir(exist_ok=True)
table.round(5).to_csv(ROOT / "reports" / "biais_segments.csv")

# %% [markdown]
# Interprétation (95 fraudes dans le test, IC95 par bootstrap, 200 rééchantillonnages) :
# - Montant : le rappel est nettement plus élevé pour les montants de 10 à 100 (0,95)
#   que pour les petits montants < 10 (0,68) et les gros montants >= 100 (0,69).
#   L'écart max (0,27) a un IC95 [0,16 ; 0,47] qui exclut 0 : l'écart est probablement réel.
#   Les fraudes de petit montant (tests de carte) et de gros montant (les plus coûteuses)
#   sont les moins bien détectées.
# - Période : rappel 0,85 la nuit contre 0,71 le jour ; IC95 de l'écart [0,02 ; 0,31],
#   proche de 0 : écart possible, mais incertain (27 fraudes la nuit seulement).
# - Précision et FPR : écarts faibles entre segments, très peu de fausses alertes partout.
# - Ces segments sont opérationnels : ils ne disent rien de l'équité envers des groupes
#   de clients (âge, sexe, localisation), absents du jeu de données.
