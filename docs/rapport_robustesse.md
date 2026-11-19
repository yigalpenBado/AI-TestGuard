# Rapport de robustesse et de sensibilité – Détection de fraude

## 1. Démarche

- **Modèle testé** : le pipeline sauvegardé (`models/model.pkl`), une forêt aléatoire avec un seuil de 0,40 choisi sur la validation.
- **Données** : jeu de test, 56 746 transactions dont 95 fraudes. Les perturbations servent uniquement à **mesurer** la sensibilité : aucun réglage du modèle ni du seuil n'en est déduit.
- **Tests automatisés** (`tests/test_robustness.py`) : rejet ou alerte pour chaque scénario, et validité des probabilités.
- **Mesure de sensibilité** (`notebooks/robustesse.py`) : effet de perturbations réalistes sur les probabilités, les décisions et les métriques.
- **Outils** : NumPy, pytest et Hypothesis, qui génère automatiquement 50 lots de transactions valides.

## 2. Tableau des scénarios

| Scénario | Comportement attendu | Mécanisme | Résultat |
|---|---|---|---|
| Données manquantes | **Rejet** | `validate_data` lève `ValueError` « Valeurs manquantes » | ✅ testé |
| Valeurs infinies | **Rejet** | `ValueError` « Valeurs infinies » | ✅ testé |
| Valeurs non numériques | **Rejet** | `ValueError` « Colonnes non numériques » | ✅ testé |
| Schéma invalide (colonne absente ou dupliquée) | **Rejet** | `ValueError` « Colonnes manquantes » ou « dupliquées » | ✅ testé |
| Amount extrêmement élevé | **Traitement + alerte** | `check_input` : alerte si Amount dépasse le maximum vu à l'entraînement | ✅ alerte émise, probabilités valides |
| Distribution différente de l'entraînement | **Traitement + alerte** | `check_input` : alerte si la médiane d'une variable s'écarte de plus de 0,5 écart-type du train (lots d'au moins 100 transactions) | ✅ alerte émise, probabilités valides |
| Perturbation limitée des variables | **Traitement** ; sensibilité mesurée | `notebooks/robustesse.py` | voir section 3 |
| Toute entrée valide (Hypothesis, valeurs jusqu'à ±10⁶) | **Traitement** | contrat de sortie : probabilités finies, entre 0 et 1 | ✅ 50 lots générés, aucun échec |

Les données invalides sont rejetées, car elles rendraient la prédiction absurde. Les données valides mais inhabituelles sont traitées, mais signalées, pour qu'un analyste puisse vérifier la décision.

## 3. Mesure de sensibilité aux perturbations

**Perturbations retenues** (différentes selon la nature des variables) :
- **V1 à V28** (composantes ACP) : bruit gaussien de 1 %, 5 % et 10 % de l'écart-type de chaque colonne sur le train. Simule une imprécision de mesure.
- **Amount** : variation relative de ±5 % et ±20 %. Simule une erreur de saisie ou des frais ; le montant reste positif.
- **Time** : décalage aléatoire de ±60 secondes. Simule une horloge légèrement désynchronisée.

**Référence sans perturbation** : rappel 0,747, précision 0,934, FPR 0,00009.

| Scénario | Δproba moyenne | Δproba max | Changements de classe | Proches du seuil | Δ rappel | Δ précision | Fausses alertes en plus |
|---|---|---|---|---|---|---|---|
| V1-V28 bruit 1 % | 0,0001 | 0,06 | 0 | 0 | 0 | 0 | 0 |
| V1-V28 bruit 5 % | 0,0004 | 0,18 | 1 | 0 | 0 | -0,012 | 1 |
| V1-V28 bruit 10 % | 0,0006 | 0,30 | 2 | 1 | -0,011 | -0,013 | 1 |
| Amount ±5 % | 0,00001 | 0,04 | 0 | 0 | 0 | 0 | 0 |
| Amount ±20 % | 0,00002 | 0,05 | 0 | 0 | 0 | 0 | 0 |
| Time ±60 s | 0 | 0,01 | 0 | 0 | 0 | 0 | 0 |

- « Proches du seuil » : changements de classe concernant une transaction dont la probabilité initiale était à moins de 0,1 du seuil.
- « Fausses alertes en plus » : 1 fausse alerte correspond à +0,00002 de FPR (sur 56 651 transactions normales).
- Cas devenus invalides : **0** dans tous les scénarios (Time et Amount restent positifs).

**Interprétation** :
- Le modèle est **peu sensible** : moins de 0,01 % des décisions changent, même avec 10 % de bruit. L'exigence T13 (au moins 99 % de décisions inchangées avec 1 % de bruit) est respectée, avec 100 %.
- **Amount et Time n'ont presque aucun effet**, ce qui est cohérent avec l'exploration des données, où ces variables étaient peu discriminantes.
- La sensibilité **augmente avec l'amplitude du bruit** sur V1 à V28 : 1 fausse alerte en plus à 5 % ; 1 fraude perdue et 1 fausse alerte en plus à 10 %.

## 4. Anomalies identifiées

| # | Anomalie | Impact |
|---|---|---|
| 1 | **Une transaction du jeu de test a un montant (25 691) supérieur au maximum vu à l'entraînement (19 657).** | Le modèle la traite hors de son domaine connu. `check_input` émet bien une alerte. |
| 2 | **Instabilité locale** : sur les 3 changements de classe observés, 2 concernent des transactions éloignées du seuil, et la probabilité d'une transaction peut varier jusqu'à 0,30 avec 10 % de bruit. | Quelques transactions isolées sont sensibles, même loin de la frontière de décision. |
| 3 | **Une alerte de dérive basée sur la moyenne était faussée par une seule valeur extrême** (détecté par les tests pendant le développement). | Corrigé : la comparaison se fait sur la médiane. |

## 5. Limites

- **Faible nombre de fraudes** (95) : une seule fraude perdue fait varier le rappel d'environ 1 point. Les Δrappel mesurés sont à la limite de ce que l'on peut détecter.
- **Perturbations simulées** : elles ne reproduisent pas les vraies erreurs ou les vraies évolutions des transactions en production.
- **Détection de dérive simple** : la comparaison de médianes ne voit pas un changement de forme de la distribution (dispersion, nouvelles combinaisons de variables). Un test statistique (par exemple Kolmogorov-Smirnov) serait plus complet.
- **Seuils d'alerte** (0,5 écart-type, 100 transactions minimum) : ils ont été fixés de façon raisonnée, mais ne sont pas calibrés sur des données réelles de production.
- **Variables anonymisées** : V1 à V28 ne permettent pas de concevoir des perturbations fondées sur le métier.

## 6. Recommandations

- Conserver le **rejet explicite** des données invalides et journaliser les **alertes** de `check_input` en production.
- Ajouter un **test statistique de dérive** et suivre son évolution dans le temps.
- Réévaluer la robustesse sur des **données récentes** et avec davantage de fraudes.
