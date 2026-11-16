#%%
## Import des bibliothèques nécessaires
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chemins relatifs à la racine du projet (fonctionne en script ou en mode cellules)
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
DATA = ROOT / "data"

#%%
## Lecture du jeu de données
df = pd.read_csv(DATA / "raw" / "creditcard.csv")
## Affichage des premières lignes du jeu de données
print(df.head())

#%%
## Dimensions du jeu de données (lignes, colonnes)
print("\nDimensions :")
print(df.shape)
# %% [Markdown]
#  Nous Observons 284 807 transactions  
# 31 colonnes : Time, V1 à V28, Amount, Class

#%%
## Vérification du schéma : colonnes attendues et types (T01, T02)
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
print("\nColonnes manquantes :", set(EXPECTED_COLUMNS) - set(df.columns))
print("Colonnes inattendues :", set(df.columns) - set(EXPECTED_COLUMNS))
print("Colonnes dupliquées :", df.columns[df.columns.duplicated()].tolist())
print("\nTypes :")
print(df.dtypes.value_counts())
# Toutes les variables sont numériques (float64), la cible Class est un entier

#%%
## Valeurs manquantes et valeurs infinies (T03)
print("\nValeurs manquantes :", df.isnull().sum().sum())
print("Valeurs infinies :", np.isinf(df.select_dtypes("number")).sum().sum())
# Nous constatons qu'il n'y a ni valeur manquante ni valeur infinie

#%%
## Domaines de valeurs : la cible ne doit contenir que 0 et 1, Amount et Time positifs
print("\nValeurs de Class :", sorted(df["Class"].unique()))
print("Amount négatifs :", (df["Amount"] < 0).sum())
print("Time négatifs :", (df["Time"] < 0).sum())

#%%
## Doublons (T08) : une même transaction en train et en test fausserait l'évaluation
n_dup = df.duplicated().sum()
print(f"\nLignes dupliquées : {n_dup}")
print("Dont fraudes :", df[df.duplicated()]["Class"].sum())
# %% 
# Nous Observons 1081 doublons, dont 19 fraudes.
#  Règle documentée : les doublons seront supprimés 
# avant le découpage train/val/test

#%%
## Statistiques descriptives
print("\nStatistiques :")
print(df[["Time", "Amount", "V1", "V2", "V3"]].describe())

#%% [markdown]
# V1 à V28 sont issues d'une ACP, et  sont donc déjà centrées.
# Time et Amount sont sur des échelles très différentes : il faudra les standardiser,
# en ajustant le scaler sur le jeu d'entraînement uniquement (T09).
# Les variables V1 à V28 étant anonymisées, elles ne doivent pas être interprétées métier.

#%%
## Répartition de la cible 
## Nous observons que le jeu est fortement déséquilibré
print("\nRépartition de Class :")
print(df["Class"].value_counts())
print(f"Taux de fraude : {df['Class'].mean():.4%}")

#%% [markdown ]
# Environ 0,17 % de fraudes 
# un modèle qui prédit dans le plus grande majorité des cas « normal » aurait 99,8 % d'accuracy.
#  L'accuracy seule n'est donc pas un indicateur pertinent (T10).

#%%
## Visualisation de la répartition de la cible
sns.countplot(data=df, x="Class")
plt.yscale("log")
plt.title("Répartition des transactions (0 = normale, 1 = fraude), échelle log")
plt.savefig(DATA / "class_distribution.png")
plt.show()

#%%
## Montant des transactions selon la classe
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Class", y="Amount")
plt.yscale("symlog")
plt.title("Montant des transactions selon la classe")
plt.savefig(DATA / "amount_by_class.png")
plt.show()

#%%
## Corrélation de chaque variable avec la cible
corr = df.corr()["Class"].drop("Class").sort_values()
plt.figure(figsize=(10, 6))
corr.plot(kind="bar")
plt.title("Corrélation des variables avec Class")
plt.tight_layout()
plt.savefig(DATA / "correlation_class.png")
plt.show()
# Quelques composantes (ex. V14, V17, V12) sont les plus corrélées à la fraude.
# Les corrélations restent modérées : la relation n'est pas linéaire simple.

# %% [markdown]
# Le pretraitement concerne les cas de test sur les types T01, T02, T03, T08, T09 et T10.
# En conclusion sur la phase de prétraitement nous ponvons dire que :
# - Le schéma est conforme, nous aucune valeur manquante ni infinie, cible binaire ;
# - Nous observons des doublons qui doivent être supprimés avant le découpage ;
# - Le découpage sera stratifié train/val/test (60/20/20, random_state=42) ;
# - La standardisation de Time et Amount ajustée sur le train uniquement ;
# - Au vu de l'accuracy, nous devons utiliser des métriques adaptées au déséquilibre (rappel, précision, PR-AUC).
# %%
# Après le prétraitement, nous pouvons passer à l'exploration des données (EDA) pour mieux comprendre les relations entre les variables et la cible.

#%%
## EDA 1 — Variables les plus discriminantes
# Écart des moyennes entre fraudes et transactions normales, rapporté à l'écart-type
# (taille d'effet) : plus il est grand, plus la variable sépare bien les deux classes.
features = df.columns.drop("Class")
fraud, normal = df[df["Class"] == 1], df[df["Class"] == 0]
effect = ((fraud[features].mean() - normal[features].mean()) / df[features].std()).abs()
effect = effect.sort_values(ascending=False)
print("\nTaille d'effet par variable (top 10) :")
print(effect.head(10).round(2))
top = effect.index[:6].tolist()
# V17, V14, V12, V10, V16 et V3 séparent le mieux les deux classes.
# Time et Amount sont en bas du classement : peu discriminants seuls.

#%%
## EDA 2 — Distribution des variables les plus discriminantes selon la classe
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flat, top):
    sns.kdeplot(data=df, x=col, hue="Class", common_norm=False, fill=True, ax=ax)
    ax.set_title(col)
plt.suptitle("Distribution des variables les plus discriminantes (normalisée par classe)")
plt.tight_layout()
plt.savefig(DATA / "eda_top_features.png")
plt.show()
# common_norm=False : chaque classe est normalisée séparément, sinon les 492 fraudes
# seraient invisibles face aux 284 315 transactions normales.
# Les transactions normales sont concentrées autour de 0 ; les fraudes sont étalées
# vers des valeurs fortement négatives. Les distributions se recouvrent partiellement :
# certaines fraudes ressemblent à des transactions normales (futurs faux négatifs).

#%%
## EDA 3 — Séparation des classes dans le plan des deux meilleures variables
sample = pd.concat([fraud, normal.sample(5000, random_state=42)])
plt.figure(figsize=(7, 6))
sns.scatterplot(data=sample, x=top[0], y=top[1], hue="Class", alpha=0.5, s=12)
plt.title(f"{top[0]} vs {top[1]} (toutes les fraudes, 5 000 normales)")
plt.savefig(DATA / "eda_scatter_top2.png")
plt.show()
# La combinaison de deux variables isole déjà une grande partie des fraudes,
# mais une zone de recouvrement subsiste près de l'origine.

#%%
## EDA 4 — Montant : les fraudes portent-elles sur des montants particuliers ?
print("\nMontant selon la classe :")
print(df.groupby("Class")["Amount"].describe().round(2))
# Médiane des fraudes (9,25) plus basse que celle des normales (22) : beaucoup de petits
# montants (tests de carte), mais une moyenne plus élevée (122 contre 88).
# Montant maximal d'une fraude : 2 125,87. Amount seul ne suffit pas à détecter la fraude.

#%%
## EDA 5 — Taux de fraude selon l'heure de la journée
# Time = secondes depuis la première transaction (2 jours) : on en déduit une heure approximative.
df_hour = df.assign(hour=(df["Time"] // 3600) % 24)
rate = df_hour.groupby("hour")["Class"].mean() * 100
plt.figure(figsize=(10, 4))
rate.plot(kind="bar")
plt.ylabel("Taux de fraude (%)")
plt.title("Taux de fraude selon l'heure")
plt.tight_layout()
plt.savefig(DATA / "eda_fraud_by_hour.png")
plt.show()
print("\nHeures au taux de fraude le plus élevé :")
print(rate.sort_values(ascending=False).head(3).round(3))
# Le taux de fraude atteint 1,7 % à l'heure 2, contre 0,17 % en moyenne : la fraude est
# plus fréquente la nuit, quand il y a peu de transactions. Attention : l'heure est
# relative à la première transaction, pas une heure réelle connue.

#%%
## EDA 6 — Corrélations entre variables (multicolinéarité)
plt.figure(figsize=(9, 7))
sns.heatmap(df[top + ["Time", "Amount", "Class"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Corrélations entre les variables discriminantes, Time, Amount et Class")
plt.tight_layout()
plt.savefig(DATA / "eda_correlation_top.png")
plt.show()
# V1 à V28 sont décorrélées entre elles (ACP) : pas de multicolinéarité.
# V3 est corrélée à Time (-0,42) et Amount à certaines composantes (V3 : -0,21).
# Les corrélations linéaires avec Class restent faibles (|r| ≤ 0,33).

# %% [markdown]
# Conclusions de l'EDA :
# - quelques composantes (V17, V14, V12, V10, V16, V3) portent l'essentiel du signal ;
# - fraudes et transactions normales se recouvrent en partie : des faux négatifs
#   et des faux positifs sont inévitables, d'où l'importance du choix du seuil (T11) ;
# - relations non linéaires : un modèle à base d'arbres (Random Forest) est un bon
#   candidat, à comparer à une régression logistique ;
# - toutes les variables sont conservées (aucune redondance) ;
# - l'EDA porte sur tout le jeu de données à titre descriptif : aucun réglage du modèle
#   ni du seuil n'en est déduit, pour respecter l'indépendance du jeu de test (T05).
