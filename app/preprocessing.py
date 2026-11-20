"""Chargement, contrôle, préparation et découpage des données (T01, T02, T03, T08, T12)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# Fonctions de contrôle et de préparation des données (testées dans tests/)
# Deux comportements :
# - On leve une exception (ValueError) si le contrat n'est pas respecté (T01, T02, T03, T12) ;
# - CORRECTION documentée : doublons supprimés (T08), colonnes en trop ignorées,
#   colonnes remises dans l'ordre attendu.
# La standardisation de Time et Amount se fera dans le Pipeline, sur le train uniquement (T09).
TARGET = "Class"

# Variables explicatives : Time, puis V1 à V28 (composantes ACP), puis Amount
FEATURES = ["Time"]
for i in range(1, 29):
    FEATURES.append("V" + str(i))
FEATURES.append("Amount")


# Chargement des données : erreurs explicites si le fichier est absent ou vide.
def load_data(path):
    """Lit le fichier CSV des transactions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Le fichier de données est vide : {path}")
    return df


def validate_data(df, require_target=True):
    """Contrôle le schéma et les types.

    Lève une exception ValueError si les données ne sont pas conformes au contrat.
    """
    # Le dataframe de doit pas etre vive, cela est un prealable à la validation du schéma et des types
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Les données doivent être un DataFrame non vide.")
    # Le dataframe doit contenir les colonnes attendues et sans doublons; et respecter les types attendus
    if df.columns.duplicated().any():
        raise ValueError(f"Colonnes dupliquées : {df.columns[df.columns.duplicated()].tolist()}")
    # Le dataframe ne doit pas contenir de valeurs manquantes ou infinies
    cols = FEATURES + [TARGET] if require_target or TARGET in df.columns else FEATURES
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Colonnes non numériques : {non_numeric}")
    if df[cols].isna().any().any():
        raise ValueError("Valeurs manquantes détectées.")
    if np.isinf(df[cols].to_numpy(dtype=float)).any():
        raise ValueError("Valeurs infinies détectées.")
    # La cible doit contenir uniquement 0 et 1
    if TARGET in cols and not df[TARGET].isin([0, 1]).all():
        raise ValueError("La cible Class ne doit contenir que 0 et 1.")
    # ÉCHEC VOLONTAIRE (preuve pour la Partie IX) : le contrôle des montants négatifs
    # est désactivé. La CI doit le détecter (test_montant_negatif) et bloquer la fusion.
    # Cette branche ne doit PAS être fusionnée.
    if (df[["Time"]] < 0).any().any():
        raise ValueError("Time et Amount ne doivent pas être négatifs.")
    return df


# Contrôle des entrées valides mais inhabituelles (Partie V) : on ne rejette pas,
# on renvoie des ALERTES. Les données invalides sont, elles, rejetées par validate_data.
DRIFT_THRESHOLD = 0.5  # écart de médiane toléré, en nombre d'écarts-types du train
MIN_ROWS_DRIFT = 100   # en dessous, la médiane d'un lot est trop instable pour juger une dérive


def check_input(X_train, X):
    """Retourne la liste des alertes sur un lot de transactions X (liste vide = rien à signaler)."""
    alerts = []

    # Alerte 1 : montants supérieurs au maximum vu à l'entraînement
    max_amount = X_train["Amount"].max()
    n_extreme = int((X["Amount"] > max_amount).sum())
    if n_extreme > 0:
        alerts.append(f"Amount hors plage d'entraînement : {n_extreme} transaction(s) > {max_amount:.2f}")

    # Alerte 2 : dérive de distribution (médiane d'une variable éloignée de celle du train).
    # La médiane est utilisée plutôt que la moyenne : une seule valeur extrême ne la déplace pas.
    if len(X) >= MIN_ROWS_DRIFT:
        for col in FEATURES:
            ecart = abs(X[col].median() - X_train[col].median()) / X_train[col].std()
            if ecart > DRIFT_THRESHOLD:
                alerts.append(f"Dérive possible sur {col} : médiane à {ecart:.2f} écart-type du train")

    return alerts


def preprocess_data(df, require_target=True):
    """Valide puis prépare les données ; retourne (X, y), y = None sans cible."""
    validate_data(df, require_target)
    if TARGET not in df.columns:
        return df[FEATURES].astype(float).reset_index(drop=True), None
    data = df.drop_duplicates()  # une transaction en double fausserait l'évaluation (T08)
    return (data[FEATURES].astype(float).reset_index(drop=True),
            data[TARGET].astype(int).reset_index(drop=True))


def split_data(X, y, seed=SEED):
    """Découpage stratifié 60/20/20 (T08). Les doublons ont été retirés par preprocess_data."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
