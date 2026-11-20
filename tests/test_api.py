from fastapi.testclient import TestClient

from api.main import app
from app.preprocessing import FEATURES

# Utilisation de TestClient pour tester l'API FastAPI
client = TestClient(app)


# test du point d'entrée de l'application FastAPI
def test_root():

    response = client.get("/")

    assert response.status_code == 200


# test de l'endpoint /health
# retourne un code 200 et un status "UP" si l'API est en bonne santé
def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "UP"


# test de l'endpoint /model/info
# retourne le modèle chargé, son seuil et les variables attendues
def test_model_info():

    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "RandomForestClassifier"
    assert 0 < data["threshold"] < 1
    assert data["features"] == FEATURES


# test de l'endpoint /predict pour le post d'une prédiction
def test_prediction():
# Initialisation du payload avec une transaction de test (30 variables)
    payload = {col: 0.0 for col in FEATURES}
    payload["Time"] = 1000.0
    payload["Amount"] = 25.0
# appel de l'endpoint /predict avec le payload
    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraude" in data
    assert "probabilite" in data
    assert "seuil" in data
    assert data["fraude"] in (0, 1)
    assert 0 <= data["probabilite"] <= 1
