#%%
## Import des bibliothèques nécessaires
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Racine du projet (fonctionne en script ou en mode cellules)
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.data import SEED, preprocess_data, split_data
from src.train import evaluate_model, load_data, load_model, predict_fraud

#%%
## Chargement du modèle sauvegardé et des données (mêmes règles que src/train.py)
model, threshold = load_model(ROOT / "models" / "model.pkl")
X, y = preprocess_data(load_data(ROOT / "data" / "raw" / "creditcard.csv"))
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print(f"Modèle rechargé, seuil {threshold} | Test : {len(X_test)} transactions, {y_test.sum()} fraudes")
# Les perturbations sont appliquées au jeu de test pour MESURER la sensibilité ;
# aucun réglage (modèle, seuil) n'est fait à partir de ces résultats.

#%%
## Perturbations réalistes, différentes selon le type de variable (5.3)
# - V1 à V28 (composantes ACP) : bruit gaussien proportionnel à l'écart-type de chaque
#   colonne sur le train (1 %, 5 %, 10 %) : simule une imprécision de mesure ;
# - Amount : variation relative de ±5 % ou ±20 % (reste positif) : erreur de saisie, frais ;
# - Time : décalage aléatoire de ±60 secondes : horloge légèrement désynchronisée.
rng = np.random.default_rng(SEED)
V_COLS = [f"V{i}" for i in range(1, 29)]


def bruit_v(X, k):
    X = X.copy()
    X[V_COLS] += rng.normal(0, 1, X[V_COLS].shape) * (k * X_train[V_COLS].std().to_numpy())
    return X


def variation_amount(X, a):
    X = X.copy()
    X["Amount"] *= 1 + rng.uniform(-a, a, len(X))
    return X


def decalage_time(X, secondes):
    X = X.copy()
    X["Time"] += rng.uniform(-secondes, secondes, len(X))
    return X


scenarios = {
    "V1-V28 bruit 1 %": bruit_v(X_test, 0.01),
    "V1-V28 bruit 5 %": bruit_v(X_test, 0.05),
    "V1-V28 bruit 10 %": bruit_v(X_test, 0.10),
    "Amount ±5 %": variation_amount(X_test, 0.05),
    "Amount ±20 %": variation_amount(X_test, 0.20),
    "Time ±60 s": decalage_time(X_test, 60),
}

#%%
## Mesures pour chaque perturbation
_, proba_ref = predict_fraud(model, X_test)
pred_ref = (proba_ref >= threshold).astype(int)
ref = evaluate_model(y_test, pred_ref, proba_ref)
proche_seuil = np.abs(proba_ref - threshold) < 0.1  # transactions proches de la frontière

rows = []
for nom, X_p in scenarios.items():
    # Cas devenus invalides (Time ou Amount négatif) : ils seraient rejetés, on les exclut
    invalides = ((X_p["Time"] < 0) | (X_p["Amount"] < 0)).to_numpy()
    garde = ~invalides

    _, proba = predict_fraud(model, X_p[garde])
    pred = (proba >= threshold).astype(int)
    m = evaluate_model(y_test[garde], pred, proba)
    change = pred != pred_ref[garde]

    rows.append({
        "scénario": nom,
        "invalides": int(invalides.sum()),
        "Δproba moyenne": np.abs(proba - proba_ref[garde]).mean(),
        "Δproba max": np.abs(proba - proba_ref[garde]).max(),
        "changements de classe": int(change.sum()),
        "dont proches du seuil": int((change & proche_seuil[garde]).sum()),
        "rappel": m["recall"], "Δrappel": m["recall"] - ref["recall"],
        "précision": m["precision"], "Δprécision": m["precision"] - ref["precision"],
        "FPR": m["fpr"], "ΔFPR": m["fpr"] - ref["fpr"],
    })

table = pd.DataFrame(rows).set_index("scénario")
print(f"Référence (sans perturbation) : rappel {ref['recall']:.4f}, précision {ref['precision']:.4f}, FPR {ref['fpr']:.5f}\n")
print(table.round(5).to_string())
(ROOT / "reports").mkdir(exist_ok=True)
table.round(5).to_csv(ROOT / "reports" / "robustesse.csv")

# %% [markdown]
# Interprétation (95 fraudes et 56 651 transactions normales dans le test) :
# - Amount (±5 %, ±20 %) et Time (±60 s) : aucun changement de classe, Δproba max ≤ 0,05.
#   Le modèle est peu sensible à ces variables, ce qui est cohérent avec l'EDA
#   (Amount et Time sont peu discriminants).
# - V1 à V28 : la sensibilité augmente avec l'amplitude du bruit, mais reste très faible :
#   0 changement à 1 %, 1 à 5 % (une fausse alerte en plus), 2 à 10 % (une fraude perdue
#   et une fausse alerte en plus) sur 56 746 transactions, soit moins de 0,01 % de décisions modifiées.
#   La variation moyenne des probabilités reste inférieure à 0,001.
# - Δproba max atteint 0,30 avec 10 % de bruit : quelques transactions isolées sont sensibles ;
#   sur les 3 changements de classe observés, 1 seul concerne une transaction déjà proche
#   du seuil (|proba - seuil| < 0,1) : les 2 autres sont des transactions plus éloignées,
#   donc une petite instabilité locale du modèle à documenter.
# - Aucun cas devenu invalide : les perturbations choisies conservent Time et Amount positifs.
# - Exigence T13 (≥ 99 % de décisions inchangées avec un bruit de 1 %) : respectée (100 %).

# %%
