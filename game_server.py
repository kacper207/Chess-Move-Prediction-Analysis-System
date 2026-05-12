"""
Chess Prediction Game - Backend
================================
Uruchamianie:
    pip install fastapi uvicorn requests numpy scikit-learn xgboost
    uvicorn game_server:app --reload --port 8000

Model XGBoost musi byc w: data/models/xgb_model.pkl
Enkoder:              data/models/xgb_label_encoder.pkl
"""

import json
import pickle
import random
import numpy as np
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Chess Predictor Game")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Sciezki do modelu ──────────────────────────────────────────────────────
XGB_MODEL_PATH   = Path("src/data/models/xgb_model.pkl")
XGB_ENCODER_PATH = Path("src/data/models/xgb_label_encoder.pkl")
N_OPENING_MOVES  = 15

# ── Pula graczy do losowania parti ─────────────────────────────────────────
PLAYER_POOL = [
    "DrNykterstein", "Hikaru", "MagnusCarlsen", "Firouzja2003",
    "nihalsarin", "alireza2003", "danya003", "DanielNaroditsky",
    "RebeccaHarris", "LyonBeast", "penguingim1", "lachesisq",
    "Zhigalko_Sergei", "IMrosier", "chess24pro",
]

# ── Zaladuj model przy starcie ─────────────────────────────────────────────
model    = None
encoders = None

def load_model():
    global model, encoders
    if not XGB_MODEL_PATH.exists():
        print(f"[WARN] Model nie znaleziony: {XGB_MODEL_PATH}")
        return
    with open(XGB_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(XGB_ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    print("[OK] Model zaladowany.")

load_model()

# ── Pomocnicze ─────────────────────────────────────────────────────────────
RESULT_MAP   = {"white": 0, "black": 1, "draw": 2}
RESULT_LABEL = {v: k for k, v in RESULT_MAP.items()}


def fetch_games_for_player(player: str, n_moves_shown: int) -> dict | None:
    """
    Pobiera ostatnie partie gracza z Lichess i zwraca losową,
    która spełnia warunki długości.
    """
    try:
        resp = requests.get(
            f"https://lichess.org/api/games/user/{player}",
            params={
                "max": 50,
                "perfType": "blitz,rapid,classical",
                "opening": "true",
                "moves": "true",
                "clocks": "false",
                "evals": "false",
                "tags": "true",
                "rated": "true",
            },
            headers={"Accept": "application/x-ndjson"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] fetch_games_for_player({player}): {e}")
        return None

    lines = [l for l in resp.text.strip().split("\n") if l.strip()]
    games = []
    for line in lines:
        try:
            games.append(json.loads(line))
        except Exception:
            continue

    random.shuffle(games)

    for g in games:
        moves_str = g.get("moves", "")
        if not moves_str:
            continue
        moves_all = moves_str.strip().split()
        if len(moves_all) < n_moves_shown + 5:
            continue

        winner     = g.get("winner", None)
        status_raw = g.get("status", "")
        if winner == "white":
            actual_result = "white"
        elif winner == "black":
            actual_result = "black"
        elif status_raw in ("draw", "stalemate", "repetition", "fiftyMoves", "insufficient"):
            actual_result = "draw"
        else:
            continue

        players   = g.get("players", {})
        white_elo = players.get("white", {}).get("rating", None)
        black_elo = players.get("black", {}).get("rating", None)
        if not white_elo or not black_elo:
            continue

        opening = g.get("opening", {})
        eco     = opening.get("eco", "Unknown")
        if eco == "Unknown":
            continue

        return {
            "game_id":       g.get("id", ""),
            "white_player":  players.get("white", {}).get("user", {}).get("name", "?"),
            "black_player":  players.get("black", {}).get("user", {}).get("name", "?"),
            "white_elo":     white_elo,
            "black_elo":     black_elo,
            "opening_eco":   eco,
            "opening_name":  opening.get("name", "?"),
            "moves_all":     moves_all,
            "actual_result": actual_result,
            "total_moves":   len(moves_all),
        }

    return None


def fetch_random_game(n_moves_shown: int) -> dict:
    """
    Pobiera losową partię z Lichess.
    Losuje graczy z puli i próbuje kolejnych aż znajdzie partię.
    """
    pool = PLAYER_POOL.copy()
    random.shuffle(pool)

    tried = set()
    for player in pool[:15]:
        if player in tried:
            continue
        tried.add(player)

        result = fetch_games_for_player(player, n_moves_shown)
        if result:
            return result

    raise HTTPException(
        status_code=503,
        detail="Nie znaleziono partii. Spróbuj ponownie."
    )


def encode_for_model(moves_uci: list[str], eco: str, total_moves: int) -> np.ndarray:
    """Zamienia surowe dane partii na wektor cech identyczny jak podczas treningu."""
    categorical_cols = ["eco"] + [f"move_{i+1}" for i in range(N_OPENING_MOVES)]
    opening_moves = (moves_uci[:N_OPENING_MOVES] + ["none"] * N_OPENING_MOVES)[:N_OPENING_MOVES]
    row = {"eco": eco, "total_moves": total_moves}
    for i, m in enumerate(opening_moves):
        row[f"move_{i+1}"] = m
    encoded = []
    for col in categorical_cols:
        le  = encoders[col]
        val = str(row[col])
        if val not in le.classes_:
            val = le.classes_[0]
        encoded.append(int(le.transform([val])[0]))
    return np.array([[total_moves] + encoded], dtype=np.float32)


# ── Endpointy ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    moves_uci:   list[str]
    eco:         str
    total_moves: int


@app.get("/api/game")
def get_game(n_moves: int = 15):
    """Zwraca losową partię — tylko pierwsze n_moves ruchów i ELO graczy."""
    if n_moves < 5 or n_moves > 40:
        raise HTTPException(status_code=400, detail="n_moves musi być 5-40")

    data = fetch_random_game(n_moves)
    shown_moves = data["moves_all"][:n_moves]

    return {
        "game_id":       data["game_id"],
        "white_player":  data["white_player"],
        "black_player":  data["black_player"],
        "white_elo":     data["white_elo"],
        "black_elo":     data["black_elo"],
        "opening_eco":   data["opening_eco"],
        "opening_name":  data["opening_name"],
        "shown_moves":   shown_moves,
        "total_moves":   data["total_moves"],
        "actual_result": data["actual_result"],
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Model XGBoost przewiduje wynik na podstawie ruchow otwarcia."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model nie jest zaladowany. Sprawdz sciezke data/models/")

    X          = encode_for_model(req.moves_uci, req.eco, req.total_moves)
    pred_idx   = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0].tolist()

    return {
        "prediction": RESULT_LABEL[pred_idx],
        "confidence": {
            "white": round(pred_proba[0] * 100, 1),
            "black": round(pred_proba[1] * 100, 1),
            "draw":  round(pred_proba[2] * 100, 1),
        }
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def serve_frontend():
    return FileResponse("game.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("game_server:app", host="0.0.0.0", port=8000, reload=True)