# Rapport de performance et de sécurité opérationnelle – Détection de fraude

## 1. Protocole de mesure

| Élément | Choix |
|---|---|
| **Matériel** | Intel Core i9-13980HX, 32 cœurs logiques, 31 Go de mémoire vive, Linux |
| **Logiciel** | Python 3.12.12, scikit-learn 1.9.1 |
| **Modèle** | pipeline sauvegardé `models/model.pkl` (forêt aléatoire de 200 arbres, seuil 0,40) |
| **Données** | transactions réelles du jeu de test |
| **Méthode** | chronométrage avec `time.perf_counter`, après 50 appels d'échauffement non comptés |
| **Latence** | 1 000 répétitions d'**une** transaction, à trois niveaux (voir ci-dessous) |
| **Débit** | lots de 1, 100, 1 000 et 10 000 transactions, 5 répétitions, médiane retenue |
| **Mémoire** | mémoire du processus (RSS) mesurée avec psutil, avant et après le chargement du modèle, puis après la prédiction de 10 000 transactions |
| **Script** | `notebooks/performance.py` ; résultats dans `reports/performance.json` |

**Trois niveaux de latence**, pour distinguer la prédiction isolée de la requête API :
1. **modèle seul** : `predict_proba` sur une transaction ;
2. **prédiction isolée** : `predict_fraud`, c'est-à-dire validation des données, modèle et application du seuil ;
3. **requête API** : `POST /predict` sur l'API FastAPI (`api/main.py`), qui ajoute le traitement HTTP et la validation des entrées par Pydantic. La requête est envoyée avec le client de test de FastAPI, sans réseau.

## 2. Seuils d'acceptation et justification

Le modèle intervient pendant l'**autorisation d'un paiement par carte** : le client attend la réponse à la caisse ou sur le site marchand.

| Mesure | Seuil | Justification |
|---|---|---|
| P95 d'une prédiction isolée | < 50 ms | L'autorisation complète doit répondre en quelques centaines de millisecondes. Le score de fraude n'est qu'une étape parmi d'autres (réseau, banque émettrice), il doit en consommer une petite part. |
| P95 d'une requête API | < 100 ms | Ajoute le traitement HTTP et la validation à la prédiction, tout en restant négligeable pour le client. |
| Débit, une transaction à la fois | ≥ 100 transactions/s par instance | Le jeu de données représente environ 1,6 transaction/s en moyenne sur 48 heures. Le seuil laisse une large marge pour les pics d'activité et un volume plus important. |
| Taille du modèle | < 50 Mo | Déploiement simple dans un conteneur, chargement rapide au démarrage. |
| Mémoire | < 500 Mo | Coût d'hébergement raisonnable pour une instance. |

Le P95 est retenu plutôt que la moyenne : il garantit que 95 % des clients obtiennent une réponse dans le délai, y compris lors des ralentissements ponctuels.

## 3. Résultats

**Latence d'une transaction**

| Niveau | Moyenne | P50 | P95 | P99 |
|---|---|---|---|---|
| Modèle seul | 7,5 ms | 7,5 ms | 7,6 ms | 7,7 ms |
| Prédiction isolée | 8,9 ms | 8,8 ms | **9,0 ms** | 9,2 ms |
| Requête API | 20,5 ms | 21,1 ms | **24,6 ms** | 25,7 ms |

**Débit selon la taille des lots**

| Taille du lot | Débit |
|---|---|
| 1 | **111 transactions/s** |
| 100 | 10 394 transactions/s |
| 1 000 | 69 637 transactions/s |
| 10 000 | 172 960 transactions/s |

**Comparaison aux seuils**

| Mesure | Résultat | Seuil | Verdict |
|---|---|---|---|
| P95 d'une prédiction isolée | 9,0 ms | < 50 ms | ✅ |
| P95 d'une requête API | 24,6 ms | < 100 ms | ✅ |
| Débit, une transaction à la fois | 111 tr/s | ≥ 100 tr/s | ✅ (de justesse) |
| Taille du modèle | 6,2 Mo | < 50 Mo | ✅ |
| Mémoire du processus | 374 Mo | < 500 Mo | ✅ |

Le modèle seul occupe environ **12 Mo** en mémoire (de 152 à 164 Mo au chargement).

## 4. Sécurité opérationnelle

Les vérifications sont automatisées dans `tests/test_securite.py` (5 tests, tous réussis).

| Vérification | Mise en œuvre | OK |
|---|---|---|
| Champs manquants | l'API renvoie une erreur 422 | ✅ |
| Champs supplémentaires | l'API les refuse (erreur 422), par exemple un numéro de carte envoyé en plus | ✅ |
| Types incorrects | texte au lieu d'un nombre, valeur nulle : erreur 422 | ✅ |
| Valeurs hors contrat | montant négatif, corps vide : erreur 422 | ✅ |
| Pas de données sensibles dans les journaux | le journal ne contient que la décision (`fraude=0` ou `1`), jamais le montant ni les variables | ✅ |
| Pas de secrets dans le dépôt Git | analyse des fichiers suivis : aucun fichier `.env`, mot de passe, clé d'API, clé AWS, clé privée ou jeton GitHub | ✅ |
| Validation des fichiers chargés | le modèle n'est chargé que si son empreinte SHA-256 correspond ; un fichier modifié ou sans empreinte est refusé | ✅ |

L'API ne reçoit pas de fichiers, seulement des transactions au format JSON.

## 5. Anomalies détectées

| # | Anomalie | Impact | Action |
|---|---|---|---|
| 1 | **Faille de chargement du modèle** : `joblib.load` peut exécuter du code contenu dans le fichier, et aucun contrôle n'était fait. | Un fichier modèle piégé aurait pu compromettre le serveur. | **Corrigé** : vérification de l'empreinte SHA-256 avant tout chargement. |
| 2 | **Débit à l'unité juste au-dessus du seuil** (111 tr/s pour un minimum de 100) : chaque prédiction coûte environ 7 ms, à cause du parcours des 200 arbres. | Peu de marge en cas de pic de trafic sur une seule instance. | Prévoir plusieurs instances, ou regrouper les transactions par petits lots (le débit est environ 100 fois plus élevé par lot de 100). |
| 3 | **L'API ajoute environ 12 ms** par rapport à la prédiction isolée (HTTP, validation Pydantic, construction du tableau de données). | C'est plus que le coût du modèle lui-même. | Acceptable au regard du seuil de 100 ms ; à surveiller si le délai devient critique. |
| 4 | **Les champs supplémentaires étaient ignorés silencieusement** par la fonction de prédiction. | Une donnée inattendue, voire sensible, pouvait passer sans être signalée. | **Corrigé dans l'API** : tout champ supplémentaire est refusé. |
| 5 | **Les verdicts étaient enregistrés comme `1.0` au lieu de `true`** dans `performance.json`. | Lecture ambiguë du rapport. | **Corrigé.** |

## 6. Limites

- **Mesures sur une seule machine**, un poste puissant de développement : un serveur de production plus modeste donnerait des temps plus élevés. Les mesures sont à refaire dans l'environnement cible.
- **Requête API mesurée sans réseau** (client de test FastAPI) : la latence réelle ajoutera le temps réseau. Aucun test de charge avec plusieurs clients simultanés n'a été réalisé ; Locust ou JMeter permettraient de le faire sur une API déployée.
- **Mémoire surestimée** : les 374 Mo incluent le jeu de données complet chargé pour les mesures. L'API en production, qui ne charge que le modèle, consommerait beaucoup moins.
- **Seuils fondés sur des hypothèses** de contexte (délai d'autorisation, volume de transactions) : ils doivent être confirmés avec les équipes de production.
