"""Tests de sécurité opérationnelle (Partie VIII).

- validation des entrées reçues par l'API (champs manquants ou en trop, types, valeurs) ;
- absence de données sensibles dans les journaux ;
- intégrité du fichier modèle avant chargement ;
- absence de secrets dans le dépôt Git.
"""
import logging
import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.data import FEATURES
from src.train import load_model, save_model, train_model

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


@pytest.fixture
def transaction():
    """Transaction valide, avec un montant reconnaissable dans les journaux."""
    t = {col: 0.5 for col in FEATURES}
    t["Time"] = 1000.0
    t["Amount"] = 1234.56
    return t


# =============================================================================
# Validation des entrées de l'API
# =============================================================================
def test_api_accepte_une_transaction_valide(transaction):
    response = client.post("/predict", json=transaction)

    assert response.status_code == 200
    body = response.json()
    assert body["fraude"] in (0, 1)
    assert 0 <= body["probabilite"] <= 1


def test_api_rejette_les_entrees_invalides(transaction):
    sans_champ = {k: v for k, v in transaction.items() if k != "V14"}
    cas_invalides = {
        "champ manquant": sans_champ,
        "champ supplémentaire": {**transaction, "carte": "4970 1234 5678 9012"},
        "type incorrect (texte)": {**transaction, "Amount": "1234.56"},
        "valeur nulle": {**transaction, "V1": None},
        "montant négatif": {**transaction, "Amount": -1.0},
        "corps vide": {},
    }
    for cas, payload in cas_invalides.items():
        response = client.post("/predict", json=payload)
        assert response.status_code == 422, f"{cas} : statut {response.status_code}"


# =============================================================================
# Journaux : la décision est tracée, jamais le contenu de la transaction
# =============================================================================
def test_journaux_sans_donnees_sensibles(transaction, caplog):
    with caplog.at_level(logging.INFO, logger="api.fraude"):
        client.post("/predict", json=transaction)

    journaux = caplog.text
    assert "prediction effectuee" in journaux          # la prédiction est bien journalisée
    assert "1234.56" not in journaux                    # pas le montant
    assert "Amount" not in journaux and "V14" not in journaux  # pas les variables


# =============================================================================
# Intégrité du fichier modèle (joblib.load peut exécuter du code)
# =============================================================================
def test_modele_modifie_refuse(train_data, tmp_path):
    path = tmp_path / "model.pkl"
    save_model(train_model(*train_data), 0.4, path)
    assert load_model(path)[1] == 0.4                   # fichier intact : accepté

    path.write_bytes(path.read_bytes() + b"code ajoute")  # fichier modifié après sauvegarde
    with pytest.raises(ValueError, match="Intégrité"):
        load_model(path)

    Path(str(path) + ".sha256").unlink()                # empreinte absente
    with pytest.raises(ValueError, match="Empreinte"):
        load_model(path)


# =============================================================================
# Absence de secrets dans le dépôt Git
# =============================================================================
SECRETS = [
    r"(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*['\"][^'\"]{6,}['\"]",
    r"AKIA[0-9A-Z]{16}",                       # clé d'accès AWS
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",     # clé privée
    r"ghp_[A-Za-z0-9]{36}",                    # jeton GitHub
]


def test_aucun_secret_dans_git():
    fichiers = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.splitlines()
    assert not [f for f in fichiers if Path(f).name.startswith(".env")], "fichier .env versionné"

    trouves = []
    for f in fichiers:
        chemin = ROOT / f
        if not chemin.exists() or chemin.suffix in (".pkl", ".png", ".docx", ".csv"):
            continue  # fichiers binaires ou de données
        texte = chemin.read_text(errors="ignore")
        trouves += [f for motif in SECRETS if re.search(motif, texte, re.IGNORECASE)]
    assert trouves == [], f"secrets possibles dans : {trouves}"
