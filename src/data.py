"""Contrôle, préparation et découpage des données (T01, T02, T03, T08, T12)."""
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


def validate_data(df, require_target=True):
    """Contrôle le schéma et les types ; lève une exception ValueError si les donnees ne sont pas conformes au contrat."""
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
    # Time et Amount doivent être positifs
    if (df[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time et Amount ne doivent pas être négatifs.")
    return df


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
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
