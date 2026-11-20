# Données

Le jeu de données **n'est pas versionné** dans ce dépôt : le fichier fait 144 Mo, au-delà de la limite de 100 Mo par fichier de GitHub.

## Source

- Jeu **Credit Card Fraud Detection** (Machine Learning Group, Université Libre de Bruxelles), publié sur Kaggle :
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Transactions par carte bancaire de titulaires européens, septembre 2013 (deux jours).
- Conditions d'utilisation : voir la page Kaggle du jeu de données.

## Installation

1. Télécharger `creditcard.csv` depuis la page Kaggle (un compte Kaggle est nécessaire).
2. Le placer ici :

```
data/raw/creditcard.csv
```

3. Vérifier que le fichier est identique à celui utilisé pour la référence de non-régression :

```bash
sha256sum data/raw/creditcard.csv
# attendu : 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

## Contenu

| Colonne | Description |
|---|---|
| `Time` | secondes écoulées depuis la première transaction du jeu |
| `V1` à `V28` | composantes issues d'une ACP, anonymisées (sans signification métier connue) |
| `Amount` | montant de la transaction |
| `Class` | cible : 1 = fraude, 0 = transaction normale |

- 284 807 transactions, dont 492 fraudes (0,17 %).
- 1 081 lignes dupliquées, supprimées avant le découpage train / validation / test.

## Dans GitHub Actions

Le workflow `Pipeline complet` (`.github/workflows/complet.yml`) télécharge le fichier depuis Kaggle, puis vérifie son empreinte SHA-256. Il faut pour cela définir deux secrets dans le dépôt GitHub (*Settings > Secrets and variables > Actions*) :

| Secret | Valeur |
|---|---|
| `KAGGLE_USERNAME` | nom d'utilisateur Kaggle |
| `KAGGLE_KEY` | clé d'API Kaggle (*Kaggle > Settings > API > Create New Token*) |

## Sans le fichier

Les tests qui ont besoin du vrai jeu de données (`TestVraiJeuDeDonnees`, `test_modele_actuel_sans_regression`) sont **ignorés** automatiquement : c'est le cas dans GitHub Actions. Tous les autres tests utilisent des données synthétiques (`tests/conftest.py`).
