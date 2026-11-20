# AI-TestGuard – Qualification d'un modèle de détection de fraude bancaire

Stratégie de test d'un modèle de Machine Learning qui détecte les transactions frauduleuses par carte bancaire : qualité des données, performances sur données déséquilibrées, intégration, robustesse, non-régression, biais, performance d'exécution, sécurité et intégration continue.

## Structure

```
app/
  preprocessing.py   chargement, validation du schéma, préparation, découpage, alertes de dérive
  training.py        modèles, entraînement, choix du seuil, sauvegarde + pipeline complet
  prediction.py      chargement sécurisé du modèle (empreinte SHA-256), prédiction
  evaluation.py      métriques, description de version, comparaison à la référence
api/main.py          API FastAPI (/, /health, /model/info, /predict)
tests/               tests pytest (unitaires, intégration, robustesse, régression, API, sécurité)
notebooks/           exploration, évaluation, robustesse, biais, performance, non-régression
data/                données (non versionnées, voir data/README.md)
models/              modèle sauvegardé et son empreinte
reports/             métriques, rapports de tests, preuves
docs/                plan de test et rapports (Word)
```

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Placer ensuite le jeu de données dans `data/raw/creditcard.csv` (voir [data/README.md](data/README.md)).

## Utilisation

| Action | Commande |
|---|---|
| Lancer les tests (avec couverture) | `pytest` |
| Vérifier le style du code | `ruff check .` |
| Entraîner le modèle (pipeline complet) | `python app/training.py` |
| Créer la référence de non-régression | `python app/training.py --save-baseline` |
| Démarrer l'API | `uvicorn api.main:app --port 8001` puis http://127.0.0.1:8001/docs |

Les rapports des tests sont écrits dans `reports/junit.xml` et `reports/coverage/index.html`.

## Démarche

- **Données** : découpage stratifié 60/20/20 (graine 42) après suppression des doublons ; le scaler est ajusté sur l'entraînement uniquement.
- **Modèles comparés** : régression logistique, arbre de décision, forêt aléatoire (`class_weight="balanced"`).
- **Seuil de décision** choisi sur la validation (rappel ≥ 0,85), jamais sur le test.
- **Modèle retenu** : forêt aléatoire, seuil 0,40.

## Résultats principaux (jeu de test)

| Métrique | Valeur | Critère |
|---|---|---|
| Rappel | 0,747 | ≥ 0,75 ❌ (une fraude manquante) |
| Précision | 0,934 | ≥ 0,70 ✅ |
| PR-AUC | 0,819 | indicateur |
| P95 de latence (API) | 24,6 ms | < 100 ms ✅ |

**Décision de qualification : modèle non accepté en l'état**, le critère de rappel n'étant pas atteint. Les limites (peu de fraudes, surapprentissage, données anonymisées de 2013, absence d'attributs démographiques) sont détaillées dans les rapports du dossier `docs/`.

## Intégration continue

GitHub Actions (`.github/workflows/tests.yml`) exécute Ruff et pytest, avec un rapport de couverture, à chaque Pull Request et à chaque envoi sur `main`.
