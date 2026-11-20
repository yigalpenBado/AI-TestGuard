# Note d'audit des biais – Détection de fraude

## 1. Périmètre

- **Système audité** : modèle de détection de fraude par carte bancaire (forêt aléatoire, seuil 0,40), sauvegardé dans `models/model.pkl`.
- **Données** : jeu Kaggle Credit Card Fraud, transactions de titulaires de cartes européens en septembre 2013. Les variables sont `Time`, `Amount`, `V1` à `V28` (composantes ACP anonymisées) et `Class`.
- **Jeu évalué** : jeu de test, 56 746 transactions dont 95 fraudes.
- **Outil** : Fairlearn (`MetricFrame`), avec des intervalles de confiance à 95 % par bootstrap (200 rééchantillonnages).

## 2. Limite essentielle

Le jeu de données **ne contient aucun attribut démographique** : ni âge, ni sexe, ni localisation, ni nationalité du client.

**Il ne permet donc pas de conclure à l'équité du système** envers des groupes de clients. Aucune affirmation du type « le modèle traite équitablement les hommes et les femmes » ou « les jeunes et les seniors » ne peut être faite à partir de ces données.

**Aucun attribut démographique n'a été reconstitué à partir des variables V1 à V28.** Ce sont des composantes issues d'une ACP, anonymisées et sans signification connue. Leur attribuer un sens (par exemple « V3 correspond à l'âge ») serait une invention sans fondement, et pourrait créer un risque de ré-identification des clients.

## 3. Analyse réalisée : écarts par segment opérationnel

Faute d'attributs démographiques, la performance a été comparée sur des **segments opérationnels**, construits uniquement à partir des variables interprétables :
- **montant** : moins de 10, de 10 à 100, 100 et plus ;
- **période** : nuit (0h-6h) et jour (6h-24h), d'après l'heure relative à la première transaction.

Ces segments décrivent des **transactions**, pas des **clients**. Ils permettent de repérer des zones où le modèle détecte moins bien la fraude, mais **pas de mesurer l'équité**.

| Segment | Groupe | Fraudes | Rappel (IC 95 %) | Précision (IC 95 %) | FPR |
|---|---|---|---|---|---|
| Montant | < 10 | 47 | 0,68 [0,54 ; 0,80] | 0,94 [0,85 ; 1,00] | 0,0001 |
| Montant | 10 – 100 | 22 | 0,95 [0,86 ; 1,00] | 0,91 [0,79 ; 1,00] | 0,0001 |
| Montant | ≥ 100 | 26 | 0,69 [0,53 ; 0,86] | 0,95 [0,82 ; 1,00] | 0,0001 |
| Période | jour | 68 | 0,71 [0,60 ; 0,80] | 0,92 [0,86 ; 0,98] | 0,0001 |
| Période | nuit | 27 | 0,85 [0,70 ; 0,97] | 0,96 [0,88 ; 1,00] | 0,0002 |

**Écarts maximaux entre groupes** :

| Segment | Écart de rappel (IC 95 %) | Écart de précision (IC 95 %) | Écart de FPR |
|---|---|---|---|
| Montant | **0,27 [0,16 ; 0,47]** | 0,03 [0,02 ; 0,19] | ≈ 0 |
| Période | 0,15 [0,02 ; 0,31] | 0,04 [0,00 ; 0,14] | ≈ 0 |

## 4. Constats

| # | Constat | Niveau de confiance |
|---|---|---|
| 1 | **Les fraudes de petit montant (< 10) et de gros montant (≥ 100) sont moins bien détectées** : rappel de 0,68 et 0,69, contre 0,95 pour les montants intermédiaires. | **Élevé** : l'intervalle de confiance de l'écart exclut 0. |
| 2 | Les fraudes de jour semblent moins bien détectées que celles de nuit (0,71 contre 0,85). | **Faible** : l'intervalle de l'écart est proche de 0, avec seulement 27 fraudes la nuit. |
| 3 | La précision et le taux de faux positifs sont **homogènes** entre segments : les fausses alertes ne touchent pas un segment plus qu'un autre. | Élevé |

Le constat 1 a un enjeu métier : les fraudes de gros montant sont les plus coûteuses pour la banque, et les petits montants correspondent souvent à des tests de carte qui précèdent une fraude plus importante.

## 5. Limites des données

- **Pas d'attributs démographiques** : aucune conclusion sur l'équité envers des groupes de clients n'est possible.
- **Peu de fraudes par segment** (22 à 68) : les intervalles de confiance sont larges, et seuls les écarts importants sont détectables.
- **Variables anonymisées** : impossible d'expliquer pourquoi certaines fraudes sont moins bien détectées.
- **Heure relative** : `Time` compte les secondes depuis la première transaction ; la période « nuit » est une approximation.
- **Données anciennes** (2013) et limitées à deux jours : les écarts peuvent avoir changé depuis.

## 6. Données complémentaires nécessaires

Pour réaliser un véritable audit d'équité, il faudrait disposer, pour chaque transaction, d'attributs décrivant le client et le contexte :

| Donnée | Intérêt pour l'audit |
|---|---|
| Tranche d'âge du titulaire | vérifier que certains âges ne subissent pas plus de blocages |
| Sexe du titulaire | vérifier l'absence d'écart de taux de faux positifs |
| Pays ou région de résidence | détecter un traitement différent selon l'origine géographique |
| Pays de la transaction, canal (en ligne ou en magasin) | distinguer les situations de paiement |
| Type de carte, ancienneté du client | identifier des profils structurellement désavantagés |

**Cadre légal** : le jeu de données concernant des titulaires de cartes européens, le **RGPD** peut servir de base légale pour encadrer la collecte et le traitement de ces données complémentaires à des fins d'audit.

## 7. Recommandations

1. **Ne pas présenter le système comme « équitable »** : l'audit actuel ne le permet pas.
2. **Améliorer la détection des fraudes de petit et de gros montant**, par exemple en étudiant ces fraudes manquées ou en ajoutant des variables contextuelles.
3. **Collecter des attributs de groupe dans un cadre conforme au RGPD**, puis refaire l'audit avec Fairlearn par groupe de clients : rappel, précision et taux de faux positifs, avec leurs intervalles de confiance.
4. **Suivre les écarts par segment en production**, au même titre que la dérive des données.
5. **Prévoir un recours humain** pour les clients dont une transaction légitime est bloquée.
