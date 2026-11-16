# Plan de test – Modèle de détection de fraude bancaire

## 1. Périmètre du système

| Composant | Contenu | Hors périmètre |
|---|---|---|
| **Données** | `creditcard.csv` (Kaggle) : 284 807 transactions, 492 fraudes (0,17 %), colonnes `Time`, `V1`–`V28`, `Amount`, `Class` | Données bancaires actuelles réelles |
| **Prétraitement** | Chargement, contrôle qualité, suppression des doublons, découpage stratifié train/val/test (60/20/20, `random_state=42`), standardisation de `Time` et `Amount` ajustée **sur le train uniquement** | Feature engineering métier (variables anonymisées) |
| **Modèle** | Classifieur binaire scikit-learn dans un `Pipeline` (prétraitement + modèle), avec pondération des classes | Deep learning |
| **Prédiction** | Fonction `predict(df)` → probabilité de fraude + décision selon un seuil choisi **sur la validation** | – |
| **Intégration** | Pipeline complet (chargement → entraînement → évaluation → sauvegarde du modèle → prédiction) exécuté par Pytest et GitHub Actions | Déploiement en production |

## 2. Exigences et risques associés

### Exigences fonctionnelles

| ID | Exigence | Risque si non respectée |
|---|---|---|
| EF1 | Les données d'entrée respectent le schéma attendu (31 colonnes numériques, sans valeur manquante, `Class` ∈ {0,1}, `Amount` ≥ 0) | Erreurs silencieuses, modèle entraîné sur des données corrompues |
| EF2 | Les jeux train/val/test sont disjoints et stratifiés | **Fuite de données** et performance surestimée |
| EF3 | Les transformations sont ajustées sur le train uniquement | Fuite de données via le scaler |
| EF4 | Le seuil de décision est choisi sur la validation, jamais sur le test | Résultats sur le test trop optimistes |
| EF5 | Le modèle détecte suffisamment de fraudes (rappel) | **Faux négatifs** : pertes financières |
| EF6 | Le modèle limite les fausses alertes (précision) | **Faux positifs** : paiements bloqués, clients mécontents |
| EF7 | `predict()` renvoie une probabilité dans [0,1] et une décision 0/1 pour chaque ligne | Intégration cassée côté système bancaire |
| EF8 | Une entrée invalide (colonne manquante, valeur NaN) déclenche une erreur explicite | Prédictions absurdes sans alerte |

### Exigences non fonctionnelles

| ID | Exigence | Risque si non respectée |
|---|---|---|
| ENF1 | **Reproductibilité** : même code et même graine donnent les mêmes métriques | Résultats non vérifiables, audit impossible |
| ENF2 | **Robustesse** : un faible bruit sur les entrées ne change presque pas les décisions | Comportement instable en production |
| ENF3 | **Non-régression** : une nouvelle version ne dégrade pas les performances de référence | Dégradation non détectée après une modification |
| ENF4 | **Performance temps réel** : la prédiction d'une transaction prend moins de 50 ms | Incompatible avec l'autorisation de paiement |
| ENF5 | **Automatisation** : les tests s'exécutent à chaque push (CI) | Régressions découvertes trop tard |

## 3. Cas de test

| ID | Exigence | Type | Description | Résultat attendu |
|---|---|---|---|---|
| CT01 | EF1 | Unitaire | Vérifier les colonnes et les types | 31 colonnes attendues, toutes numériques |
| CT02 | EF1 | Unitaire | Rechercher les valeurs manquantes | 0 valeur manquante |
| CT03 | EF1 | Unitaire | Vérifier les domaines de valeurs | `Class` ∈ {0,1}, `Amount` ≥ 0, `Time` ≥ 0 |
| CT04 | EF2 | Unitaire | Vérifier les doublons avant le découpage | Doublons supprimés, aucun doublon restant |
| CT05 | EF2 | Unitaire | Vérifier l'intersection des index train/val/test | Intersection vide |
| CT06 | EF2 | Unitaire | Comparer le taux de fraude par partition | Écart ≤ 0,05 point de % par rapport au jeu complet |
| CT07 | EF3 | Unitaire | Comparer la moyenne du scaler avec celle du train | Égalité avec la moyenne du train (≠ moyenne globale) |
| CT08 | EF4 | Unitaire | Vérifier que le choix du seuil n'utilise que la validation | Seuil identique si le test est modifié |
| CT09 | EF5 | Intégration | Mesurer le rappel sur le test au seuil retenu | Rappel ≥ 0,75 |
| CT10 | EF6 | Intégration | Mesurer la précision sur le test au seuil retenu | Précision ≥ 0,70 |
| CT11 | EF5/EF6 | Intégration | Mesurer PR-AUC et ROC-AUC sur le test | PR-AUC ≥ 0,70 ; ROC-AUC ≥ 0,95 |
| CT12 | EF5/EF6 | Analyse | Matrice de confusion et courbe précision/rappel en fonction du seuil | FP et FN documentés, compromis justifié |
| CT13 | EF7 | Unitaire | Appeler `predict()` sur 10 lignes | 10 probabilités dans [0,1] et 10 décisions dans {0,1} |
| CT14 | EF8 | Unitaire | Appeler `predict()` sans `Amount`, puis avec un NaN | Exception levée (`ValueError`) |
| CT15 | ENF1 | Intégration | Entraîner deux fois avec `random_state=42` | Prédictions identiques, métriques égales |
| CT16 | ENF2 | Robustesse | Ajouter un bruit gaussien (σ = 1 % de l'écart-type) aux variables | ≥ 99 % des décisions inchangées |
| CT17 | ENF2 | Robustesse | Montants extrêmes (0 et 10⁶) | Aucune erreur, probabilité dans [0,1] |
| CT18 | ENF3 | Non-régression | Comparer les métriques à `baseline_metrics.json` | Baisse de PR-AUC et de rappel ≤ 0,02 |
| CT19 | ENF4 | Performance | Mesurer le temps moyen de `predict()` sur 1 transaction | < 50 ms |
| CT20 | ENF5 | CI | Push sur GitHub | Workflow vert, rapports archivés |

## 4. Outils et preuves à conserver

| Outil | Usage |
|---|---|
| Python, pandas, scikit-learn | Données, pipeline, modèle, métriques |
| Pytest (+ `pytest-cov`) | Tests unitaires, d'intégration, de robustesse et de non-régression |
| joblib | Sauvegarde du modèle entraîné |
| matplotlib | Matrice de confusion, courbe précision/rappel |
| GitHub Actions | Exécution automatique des tests à chaque push |

**Preuves conservées** (dossier `reports/` et artefacts CI) :
- `metrics.json` : métriques sur val et test, seuil retenu, graine, versions des bibliothèques ;
- `baseline_metrics.json` : référence pour la non-régression ;
- `confusion_matrix.png` et `pr_curve.png` ;
- rapport Pytest (`junit.xml`) et rapport de couverture ;
- logs du workflow GitHub Actions ;
- `requirements.txt` (versions figées) et hash du commit.

## 5. Critères d'acceptation et de rejet

**Le modèle est ACCEPTÉ si toutes les conditions suivantes sont réunies :**
1. Tous les tests de qualité de données et de fuite (CT01–CT08) passent.
2. Sur le jeu de test, au seuil choisi sur la validation : **rappel ≥ 0,75**, **précision ≥ 0,70**, **PR-AUC ≥ 0,70**, **ROC-AUC ≥ 0,95**.
3. Les tests de reproductibilité et de robustesse (CT15–CT17) passent.
4. Aucune régression supérieure à 0,02 sur la PR-AUC ou le rappel (CT18).
5. Latence < 50 ms par transaction (CT19).
6. Le pipeline CI est vert (CT20).

**Le modèle est REJETÉ si au moins une des conditions suivantes est remplie :**
- une fuite de données est détectée (CT04–CT08) ;
- un seuil de performance n'est pas atteint ;
- les résultats ne sont pas reproductibles ;
- le seuil ou le modèle a été choisi à l'aide du jeu de test ;
- l'évaluation repose uniquement sur l'accuracy.

**Limite** : ces critères qualifient le modèle sur un jeu de données pédagogique et ancien. Ils ne garantissent pas les performances sur les transactions actuelles. Une validation sur des données récentes et un suivi de la dérive en production restent nécessaires.

## 6. Matrice des exigences (T01–T07 fournies, T08–T15 ajoutées)

| ID | Exigence | Type de test | Résultat attendu | Risque maîtrisé |
|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Données | Aucune colonne indispensable manquante | Pipeline cassé ou entraîné sur des données incomplètes |
| T02 | Les données respectent les types attendus | Unitaire | Types conformes au contrat | Conversion silencieuse, calculs faux |
| T03 | Le prétraitement ne produit pas de valeurs non finies | Unitaire | Aucune valeur NaN ou infinie en sortie | Plantage ou prédictions absurdes |
| T04 | Le modèle produit une probabilité valide | Fonctionnel ML | Probabilité comprise entre 0 et 1 | Décision incohérente côté système bancaire |
| T05 | Les performances sont évaluées sur des données indépendantes | ML | Jeu de test non utilisé pour le réglage | Performance surestimée |
| T06 | Les résultats sont reproductibles | Non-régression | Écart compatible avec la tolérance définie | Résultats non vérifiables, audit impossible |
| T07 | Les tests s'exécutent sur GitHub | CI | Workflow réussi ou échec correctement signalé | Régressions découvertes trop tard |
| **T08** | Les partitions train/val/test sont disjointes, sans doublon commun, et stratifiées | Données | Intersection vide ; taux de fraude identique (± 0,05 pt) dans chaque partition | **Fuite de données** par doublons ; partition sans assez de fraudes |
| **T09** | Le scaler est ajusté uniquement sur le jeu d'entraînement | Unitaire | Moyenne du scaler = moyenne du train | **Fuite de données** via le prétraitement |
| **T10** | L'évaluation utilise des métriques adaptées au déséquilibre | ML | Rappel, précision, PR-AUC et matrice de confusion rapportés (pas l'accuracy seule) | Accuracy trompeuse (99,8 % en prédisant toujours « normal ») |
| **T11** | Le seuil de décision, choisi sur la validation, garantit un rappel et une précision minimaux sur le test | Fonctionnel ML | Rappel ≥ 0,75 et précision ≥ 0,70 | **Faux négatifs** (pertes financières) et **faux positifs** (clients bloqués) |
| **T12** | Une entrée invalide est rejetée explicitement | Unitaire | `ValueError` si colonne manquante ou valeur NaN | Prédiction fausse sans alerte |
| **T13** | Le modèle est robuste à un faible bruit sur les entrées | Robustesse | ≥ 99 % des décisions inchangées avec un bruit de 1 % de l'écart-type | Comportement instable en production |
| **T14** | Une nouvelle version ne dégrade pas les performances de référence | Non-régression | Baisse de PR-AUC et de rappel ≤ 0,02 par rapport à `baseline_metrics.json` | Dégradation non détectée après une modification |
| **T15** | La prédiction d'une transaction est assez rapide | Performance | Temps moyen < 50 ms par transaction | Incompatible avec l'autorisation de paiement en temps réel |
