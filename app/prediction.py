"""Prédiction : chargement sécurisé du modèle sauvegardé et production des décisions.

- load_model : vérifie l'empreinte SHA-256 du fichier AVANT joblib.load (Partie VIII) ;
- predict_fraud : valide l'entrée (T12), calcule les probabilités et applique le seuil.
"""
import hashlib
from pathlib import Path

import joblib

from app.preprocessing import FEATURES, validate_data


def sha256_of(path):
    """Empreinte SHA-256 d'un fichier."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Rechargement du modèle sauvegardé : erreurs explicites si le fichier est absent ou invalide.
def load_model(path):
    """Recharge le modèle et son seuil. Retourne (model, threshold)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable : {path}")
    # Sécurité : joblib.load peut exécuter du code contenu dans le fichier.
    # On vérifie donc l'empreinte AVANT de charger : un fichier modifié est refusé.
    empreinte = Path(str(path) + ".sha256")
    if not empreinte.exists():
        raise ValueError(f"Empreinte du modèle manquante : {empreinte}")
    if sha256_of(path) != empreinte.read_text().strip():
        raise ValueError(f"Intégrité du modèle non vérifiée (fichier modifié) : {path}")
    try:
        saved = joblib.load(path)
    except Exception as error:
        raise ValueError(f"Fichier modèle illisible : {path}") from error
    if not isinstance(saved, dict) or "model" not in saved or "threshold" not in saved:
        raise ValueError(f"Fichier modèle invalide (modèle ou seuil manquant) : {path}")
    return saved["model"], saved["threshold"]


# Apres l'entrainement, nous pouvons prédire les fraudes sur un jeu de données donné.
# Une validation est effectuée pour vérifier que les données sont valides (T12).
# avant de retourner les prédictions, nous calculons les probabilités de fraude
# et appliquons un seuil pour obtenir des décisions binaires (0/1).
def predict_fraud(model, X, threshold=0.5):
    """Retourne (décisions 0/1, probabilités de fraude). Rejette une entrée invalide (T12)."""
    validate_data(X, require_target=False)
    y_proba = model.predict_proba(X[FEATURES])[:, 1]
    return (y_proba >= threshold).astype(int), y_proba
