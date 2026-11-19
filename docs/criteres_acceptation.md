# Critères d'acceptation métier – Modèle de détection de fraude

## 1. Coût des erreurs

Les deux types d'erreur n'ont pas le même coût pour la banque.

| Erreur | Conséquence | Coût estimé |
|---|---|---|
| **Faux négatif (FN)** : fraude non détectée | Perte financière (montant de la fraude remboursé au client), frais de litige, risque d'image | Montant moyen d'une fraude dans le jeu : **122 €** (médiane 9 €, maximum 2 126 €) + [À compléter : frais de traitement d'un litige] |
| **Faux positif (FP)** : transaction normale bloquée | Paiement refusé, appel au service client, insatisfaction, risque de perte du client | [À compléter : coût de traitement d'une alerte, ex. temps d'un conseiller] |

**Conclusion :** un faux négatif coûte en général plus cher qu'un faux positif. Le **rappel** est donc prioritaire, mais la **précision** doit rester suffisante pour ne pas saturer les équipes et gêner les clients.

[À compléter : rapport de coût FN / FP retenu, par exemple « une fraude manquée coûte environ N fois une fausse alerte ».]

## 2. Critères retenus et justification

Ces critères ont été fixés **avant** l'évaluation sur le jeu de test, à partir des exigences du plan de test (section 5, T11, T14).

| Critère | Seuil | Justification |
|---|---|---|
| **Rappel** | ≥ 0,75 | Au plus 1 fraude sur 4 non détectée. Limite les pertes financières, priorité métier (coût FN élevé). |
| **Précision** | ≥ 0,70 | Au moins 7 alertes sur 10 sont de vraies fraudes. Au-delà de 3 fausses alertes sur 10, les équipes et les clients sont trop sollicités. |
| **Non-régression** | baisse ≤ 0,02 (PR-AUC, rappel) | Une nouvelle version ne doit pas dégrader la détection. |
| **Accuracy** | non utilisée comme critère | Un modèle qui prédit toujours « normale » obtient 99,83 % d'accuracy sans détecter aucune fraude. |

Indicateurs suivis sans seuil d'acceptation : **PR-AUC** (qualité globale du classement, indépendante du seuil, utilisée pour choisir le modèle et pour la non-régression), **FPR** (part des clients légitimes bloqués) et **FNR** (part des fraudes manquées, égal à 1 − rappel).

[À compléter : justification plus fine des valeurs 0,75 et 0,70 au regard des coûts de la section 1.]

## 3. Méthode de réglage

- Le **seuil de décision** est choisi sur le jeu de **validation**, jamais sur le jeu de test (T05, T11).
- Règle : parmi les seuils qui atteignent le rappel visé, on retient celui qui a la meilleure précision (`choose_threshold`).
- **Rappel visé sur la validation : 0,85**, soit une marge de 10 points au-dessus de l'exigence de 0,75. La validation ne contient que 94 fraudes : chaque fraude manquée fait varier le rappel d'environ 1 point, et le rappel mesuré peut baisser sur de nouvelles données.
- Le modèle est choisi sur la PR-AUC de validation, puis évalué **une seule fois** sur le jeu de test.

## 4. Résultat de la qualification

Modèle retenu : **forêt aléatoire**, seuil **0,40** (choisi sur la validation).

| Critère | Seuil | Validation | Test | Verdict (test) |
|---|---|---|---|---|
| Rappel | ≥ 0,75 | 0,86 | **0,747** | ❌ non atteint (71 fraudes sur 95, il en manque une) |
| Précision | ≥ 0,70 | 0,90 | 0,93 | ✅ |
| PR-AUC | suivi | 0,88 | 0,82 | indicateur |
| FPR | suivi | 0,0002 | 0,0001 | 5 clients bloqués sur 56 651 |
| FNR | suivi | 0,14 | 0,25 | 24 fraudes manquées |

**Décision : modèle NON ACCEPTÉ en l'état**, le critère de rappel n'étant pas atteint sur le jeu de test.

Le seuil n'a pas été réajusté après observation du test : cela reviendrait à régler le modèle sur le jeu de test et rendrait l'évaluation trop optimiste.

[À compléter : commentaire sur le compromis observé, par exemple 5 fausses alertes pour 71 fraudes détectées.]

## 5. Limites et recommandations

- **Faible nombre de fraudes** (environ 95 par partition) : le rappel mesuré varie d'environ ±4 points ; l'écart validation / test (0,86 → 0,75) en est une conséquence.
- **Fraudes manquées atypiques** : elles ressemblent à des transactions normales (V14 et V17 proches de 0) et ont un montant médian plus élevé (30 € contre 1,6 € pour les fraudes détectées).
- **Données anciennes et anonymisées** (septembre 2013, variables issues d'une ACP) : les résultats ne garantissent pas les performances sur les transactions actuelles.

Recommandations :
- estimer le rappel par **validation croisée** pour obtenir une mesure plus stable ;
- collecter davantage d'exemples de fraude ;
- valider le modèle sur des données récentes et suivre la dérive en production.

[À compléter : autres recommandations.]
