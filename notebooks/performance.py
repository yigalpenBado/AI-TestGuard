#%%
## Mesure du coût d'exécution (Partie VIII)
# PROTOCOLE
# - Matériel : relevé automatiquement (processeur, cœurs, mémoire vive, versions).
# - Modèle : pipeline sauvegardé models/model.pkl (forêt aléatoire, 200 arbres, seuil 0,40).
# - Données : transactions réelles du jeu de test.
# - Méthode : time.perf_counter, après 50 appels d'échauffement (non comptés).
# - Latence : 1 000 répétitions d'UNE transaction, à 3 niveaux :
#     1. modèle seul (predict_proba) ;
#     2. prédiction isolée (predict_fraud = validation des données + modèle + seuil) ;
#     3. requête API (HTTP + validation Pydantic + prédiction), via le client de test FastAPI.
# - Débit : lots de 1, 100, 1 000 et 10 000 transactions, 5 répétitions, médiane retenue.
# - Mémoire : mémoire du processus (RSS, psutil) avant/après chargement du modèle et
#   pendant la prédiction d'un lot de 10 000 transactions.
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import psutil
import sklearn

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.data import preprocess_data, split_data
from src.train import load_data, load_model, predict_fraud

MODEL_PATH = ROOT / "models" / "model.pkl"
process = psutil.Process()
mo = 1024 ** 2

#%%
## Environnement de mesure
def nom_processeur():
    """Nom du processeur (Linux : /proc/cpuinfo), sinon l'architecture."""
    try:
        for ligne in Path("/proc/cpuinfo").read_text().splitlines():
            if ligne.startswith("model name"):
                return ligne.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


environnement = {
    "processeur": nom_processeur(),
    "coeurs": psutil.cpu_count(logical=True),
    "memoire_totale_mo": round(psutil.virtual_memory().total / mo),
    "systeme": platform.platform(),
    "python": platform.python_version(),
    "scikit-learn": sklearn.__version__,
}
print(environnement)

#%%
## Taille du modèle et mémoire au chargement
taille_modele_mo = os.path.getsize(MODEL_PATH) / mo
rss_avant = process.memory_info().rss / mo
model, threshold = load_model(MODEL_PATH)
rss_apres = process.memory_info().rss / mo
print(f"Modèle : {taille_modele_mo:.1f} Mo sur disque, +{rss_apres - rss_avant:.1f} Mo en mémoire")

#%%
## Données de test (chargées APRÈS la mesure mémoire du modèle)
X, y = preprocess_data(load_data(ROOT / "data" / "raw" / "creditcard.csv"))
X_test = split_data(X, y)[2]
rows = [X_test.iloc[[i]] for i in range(1050)]  # 1 050 transactions isolées (50 + 1 000)


def chronometre(fonction, entrees):
    """Temps de chaque appel en millisecondes, après 50 appels d'échauffement."""
    for x in entrees[:50]:
        fonction(x)
    temps = []
    for x in entrees[50:]:
        debut = time.perf_counter()
        fonction(x)
        temps.append((time.perf_counter() - debut) * 1000)
    return np.array(temps)


def resume(temps):
    return {"moyenne_ms": temps.mean(), "p50_ms": np.percentile(temps, 50),
            "p95_ms": np.percentile(temps, 95), "p99_ms": np.percentile(temps, 99)}


#%%
## Latence d'une transaction : modèle seul, prédiction isolée, requête API
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
payloads = [r.iloc[0].to_dict() for r in rows]

latence = {
    "modele_seul": resume(chronometre(lambda x: model.predict_proba(x), rows)),
    "prediction_isolee": resume(chronometre(lambda x: predict_fraud(model, x, threshold), rows)),
    "requete_api": resume(chronometre(lambda p: client.post("/predict", json=p), payloads)),
}
for niveau, r in latence.items():
    print(f"{niveau:<18} moyenne {r['moyenne_ms']:6.2f} ms | P50 {r['p50_ms']:6.2f} | "
          f"P95 {r['p95_ms']:6.2f} | P99 {r['p99_ms']:6.2f}")

#%%
## Débit selon la taille des lots
debit = {}
for taille in (1, 100, 1000, 10000):
    lot = X_test.iloc[:taille]
    temps = []
    for _ in range(5):
        debut = time.perf_counter()
        predict_fraud(model, lot, threshold)
        temps.append(time.perf_counter() - debut)
    debit[taille] = taille / np.median(temps)
    print(f"lot de {taille:>6} : {debit[taille]:>10.0f} transactions/s")

#%%
## Mémoire pendant la prédiction d'un lot de 10 000 transactions
predict_fraud(model, X_test.iloc[:10000], threshold)
rss_prediction = process.memory_info().rss / mo
print(f"Mémoire du processus après prédiction de 10 000 transactions : {rss_prediction:.0f} Mo")

#%%
## Comparaison aux seuils d'acceptation (justifiés dans le rapport)
SEUILS = {"p95_prediction_isolee_ms": 50, "p95_requete_api_ms": 100,
          "debit_min_tps": 100, "taille_modele_max_mo": 50, "memoire_max_mo": 500}
verdicts = {  # bool() : sinon numpy les enregistre comme 1.0 dans le JSON
    "p95_prediction_isolee_ms": latence["prediction_isolee"]["p95_ms"] < SEUILS["p95_prediction_isolee_ms"],
    "p95_requete_api_ms": latence["requete_api"]["p95_ms"] < SEUILS["p95_requete_api_ms"],
    "debit_min_tps": debit[1] >= SEUILS["debit_min_tps"],  # une transaction à la fois (cas réel)
    "taille_modele_max_mo": taille_modele_mo < SEUILS["taille_modele_max_mo"],
    "memoire_max_mo": rss_prediction < SEUILS["memoire_max_mo"],
}
verdicts = {k: bool(v) for k, v in verdicts.items()}
for k, ok in verdicts.items():
    print(f"{k:<26} seuil {SEUILS[k]:>5} : {'OK' if ok else 'NON RESPECTÉ'}")

#%%
## Enregistrement
(ROOT / "reports").mkdir(exist_ok=True)
with open(ROOT / "reports" / "performance.json", "w") as f:
    json.dump({
        "environnement": environnement,
        "protocole": {"echauffement": 50, "repetitions_latence": 1000,
                      "tailles_lots": [1, 100, 1000, 10000], "repetitions_debit": 5,
                      "methode": "time.perf_counter ; mémoire RSS via psutil"},
        "taille_modele_mo": taille_modele_mo,
        "memoire_mo": {"avant_chargement": rss_avant, "apres_chargement": rss_apres,
                       "apres_prediction_10000": rss_prediction},
        "latence": latence,
        "debit_tps": debit,
        "seuils": SEUILS,
        "verdicts": verdicts,
    }, f, indent=2, default=float)
