import json
import pandas as pd
from config import RAW_DATA_PATH

def parse_games():
    with open(RAW_DATA_PATH, "r") as f:
        games_list = json.load(f)

    parsed = []
    for game in games_list:
        moves = game.get("moves", "")
        if not moves:
            continue

        parsed.append({
            "moves": moves.split(),
            "winner": game.get("winner", "draw"),
            "opening_eco": game.get("opening", {}).get("eco", "Unknown"),
            "opening_name": game.get("opening", {}).get("name", "Unknown"),
            "num_moves": len(moves.split())
        })

    df = pd.DataFrame(parsed)
    print(f"Sparsowano {len(df)} partii")

    print("Przykładowe opening_eco:", df["opening_eco"].value_counts().head(10))
    print("Przykładowe num_moves:", df["num_moves"].describe())

    return df