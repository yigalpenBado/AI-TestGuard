# Rapport de non-régression – Détection de fraude

## 1. Validation hors échantillon

| Exigence | Mise en œuvre |
|---|---|
| Séparer avant toute transformation apprise | Découpage d'abord ; le scaler est dans le Pipeline, ajusté sur le train uniquement (test automatisé T09). |
| Jeux d'entraînement, de validation et de test | Découpage stratifié 60/20/20 avec la graine 42 : 170 235 / 56 745 / 56 746 transactions. |
| Réglages sur les données autorisées | Le choix du modèle, le seuil de décision et la comparaison des versions se font sur la **validation**. |
| Test réservé à l'évaluation finale | Dans le pipeline (`src/train.py`), le test n'est utilisé que pour l'évaluation finale du modèle retenu, avec le seuil choisi sur la validation. |
| Comparer entraînement et test | Voir ci-dessous. |

**Surapprentissage de la référence** (forêt aléatoire, seuil 0,40) :

| Métrique | Entraînement | Test | Écart |
|---|---|---|---|
| Rappel | 1,000 | 0,747 | 0,25 |
| PR-AUC | 1,000 | 0,819 | 0,18 |
| Précision | 0,953 | 0,934 | 0,02 |

Le modèle détecte toutes les fraudes du train, mais seulement 75 % de celles du test : il a **mémorisé** les fraudes d'entraînement. La précision reste stable : le surapprentissage touche surtout la détection.

## 2. Version de référence

La référence est enregistrée dans `reports/baseline.json`. Elle n'est créée que sur demande explicite (`python src/train.py --save-baseline`) et n'est jamais écrasée automatiquement.

| Élément | Valeur |
|---|---|
| Version du code | commit `f508122` |
| Modèle | forêt aléatoire, 200 arbres, `max_depth` non limité, `class_weight="balanced"` |
| Bibliothèques | scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, joblib 1.6.0 |
| Graine | 42 |
| Seuil de décision | 0,40 (choisi sur la validation pour un rappel ≥ 0,85) |
| Données | 283 726 transactions, 473 fraudes, empreinte SHA-256 `76274b69…` |
| Validation fixe | 56 745 transactions, 94 fraudes |

## 3. Nouvelle version proposée

**Modification** : `max_depth=10`, c'est-à-dire des arbres limités à 10 niveaux. Tout le reste est identique : données, découpage, graine, pipeline, règle de choix du seuil.

**Motivation** : réduire le surapprentissage mesuré sur la référence, et obtenir un modèle plus simple.

## 4. Règle de décision (tolérance 0,02, exigence T14)

| Décision | Condition |
|---|---|
| **Rejeter** | le rappel ou la PR-AUC baisse de plus de 0,02 |
| **Investiguer** | la précision baisse de plus de 0,02, ou les données ou les bibliothèques ont changé |
| **Accepter** | sinon |

La comparaison se fait sur le jeu de **validation fixe**, jamais sur le jeu de test.

## 5. Comparaison (validation)

| Métrique | Référence | max_depth=10 | Écart | Verdict |
|---|---|---|---|---|
| Seuil choisi | 0,40 | 0,65 | – | – |
| Rappel | 0,862 | 0,851 | -0,011 | ✅ dans la tolérance |
| Précision | 0,900 | 0,860 | **-0,040** | ⚠️ au-delà de la tolérance |
| F1 | 0,880 | 0,856 | -0,025 | – |
| PR-AUC | 0,876 | 0,850 | **-0,026** | ❌ au-delà de la tolérance |
| Fausses alertes (FP) | 9 | 13 | +4 | – |
| Fraudes manquées (FN) | 13 | 14 | +1 | – |
| Écart de rappel train - validation | 0,138 | 0,145 | +0,007 | surapprentissage non réduit |

Même jeu de données (empreinte identique) et mêmes bibliothèques : la comparaison est fiable.

## 6. Décision : REJETER

**Motifs** :
- **la PR-AUC baisse de 0,026**, au-delà de la tolérance de 0,02 : la qualité globale du classement des transactions se dégrade ;
- **la précision baisse de 0,040** : 4 fausses alertes de plus sur la validation (+44 %) ;
- **le rappel baisse aussi** (-0,011, soit une fraude manquée de plus), même si cette baisse reste dans la tolérance ;
- **l'objectif de la modification n'est pas atteint** : l'écart de rappel entre l'entraînement et la validation ne diminue pas (0,145 contre 0,138). Limiter la profondeur à 10 ne réduit pas la mémorisation des fraudes.

La version de référence est conservée en production (`models/model.pkl`), et la référence n'est pas modifiée.

## 7. Limites et suites

- **Faible nombre de fraudes en validation (94)** : une fraude représente environ 1 point de rappel. Les écarts observés sont réels, mais proches de la précision de mesure ; une validation croisée donnerait une comparaison plus stable.
- **Une seule valeur testée** pour `max_depth`. D'autres réglages pourraient mieux réduire le surapprentissage, par exemple `min_samples_leaf`, ou une profondeur plus faible. Ils devront passer par la même procédure de comparaison.
- **Le contrôle automatisé** (`tests/test_non_regression.py`) compare le modèle sauvegardé à la référence et échoue en cas de décision « rejeter ».
