"""API minimale de détection de fraude.

POST /predict : reçoit UNE transaction (Time, V1 à V28, Amount) et renvoie la décision.
Validation des entrées par Pydantic (avant le modèle) :
- champ manquant, champ supplémentaire, type incorrect → erreur 422 ;
- valeur non finie (NaN, infini) ou Time / Amount négatif → erreur 422.

Lancement : uvicorn api.main:app  (depuis la racine du projet)
"""
import logging
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import ConfigDict, Field, create_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.prediction import load_model, predict_fraud
from app.preprocessing import FEATURES

# Modèle et seuil chargés une seule fois, au démarrage
model, threshold = load_model(ROOT / "models" / "model.pkl")

# Schéma d'une transaction : 30 nombres finis, Time et Amount positifs, aucun champ en plus
champs = {}
for col in FEATURES:
    if col in ("Time", "Amount"):
        champs[col] = (float, Field(ge=0, allow_inf_nan=False))
    else:
        champs[col] = (float, Field(allow_inf_nan=False))
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", strict=True),  # pas de champ en plus, pas de "12" pour 12
    **champs,
)

app = FastAPI(title="Détection de fraude")


# Point d'entrée : présentation de l'API
@app.get("/")
def root():
    return {"message": "API de détection de fraude bancaire"}


# Santé de l'API : utilisée pour vérifier que le service répond
@app.get("/health")
def health():
    return {"status": "UP"}


# Informations sur le modèle chargé (sans donnée sensible)
@app.get("/model/info")
def model_info():
    return {
        "model": type(model.named_steps["model"]).__name__,
        "threshold": threshold,
        "features": FEATURES,
    }


# Journal : on enregistre la DÉCISION, jamais le contenu de la transaction
# (montant, variables V1 à V28, Time) : ce sont des données sensibles.
logger = logging.getLogger("api.fraude")


@app.post("/predict")
def predict(transaction: Transaction):
    X = pd.DataFrame([transaction.model_dump()])[FEATURES]
    y_pred, y_proba = predict_fraud(model, X, threshold)
    logger.info("prediction effectuee : fraude=%d", int(y_pred[0]))
    return {"fraude": int(y_pred[0]), "probabilite": float(y_proba[0]), "seuil": threshold}
