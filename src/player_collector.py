import berserk
import pandas as pd
from datetime import datetime
from config import PLAYER_NAME, MAX_PLAYER_GAMES, PLAYER_PERF_TYPE


def convert_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def collect_player_games():
    print(f"Downloading games for player: {PLAYER_NAME}...")

    client = berserk.Client()
    games = client.games.export_by_player(
        PLAYER_NAME,
        max=MAX_PLAYER_GAMES,
        perf_type=PLAYER_PERF_TYPE,
        as_pgn=False,
        opening=True
    )

    games_list = list(games)
    print(f"Downloaded: {len(games_list)} games")

    parsed = []
    for game in games_list:
        moves = game.get("moves", "")
        if not moves:
            continue
        parsed.append({
            "moves": moves.split(),
            "opening_eco": game.get("opening", {}).get("eco", "Unknown"),
            "opening_name": game.get("opening", {}).get("name", "Unknown"),
            "result": game.get("winner", "draw"),
            "num_moves": len(moves.split())
        })

    df = pd.DataFrame(parsed)
    print(f"Parsed: {len(df)} games")
    return df