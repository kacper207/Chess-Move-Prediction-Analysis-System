import chess.pgn
import pandas as pd
import zstandard as zstd
import io
from tqdm import tqdm
from config import PGN_FILE_PATH, MAX_PGN_GAMES, MIN_MOVES


def stream_pgn():
    games = []
    errors = 0

    if PGN_FILE_PATH.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        raw = open(PGN_FILE_PATH, "rb")
        f_stream = dctx.stream_reader(raw)
        f = io.TextIOWrapper(f_stream, encoding="utf-8", errors="ignore")
    else:
        f = open(PGN_FILE_PATH, "r", encoding="utf-8", errors="ignore")

    with tqdm(total=MAX_PGN_GAMES, desc="Loading PGN") as pbar:
        while len(games) < MAX_PGN_GAMES:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                errors += 1
                continue

            if game is None:
                print("\nEnd of file reached.")
                break

            if game.errors:
                errors += 1
                continue

            try:
                moves = [m.uci() for m in game.mainline_moves()]
            except Exception:
                errors += 1
                continue

            if len(moves) < MIN_MOVES:
                continue

            eco = game.headers.get("ECO", "Unknown")
            if eco == "Unknown":
                continue

            games.append({
                "moves": moves,
                "opening_eco": eco,
                "opening_name": game.headers.get("Opening", "Unknown"),
                "result": game.headers.get("Result", "*"),
                "num_moves": len(moves)
            })

            pbar.update(1)

    f.close()

    df = pd.DataFrame(games)
    print(f"Loaded:           {len(df):,} games")
    print(f"Skipped (errors): {errors:,}")
    print(f"Unique openings:  {df['opening_eco'].nunique()}")
    print(f"Avg moves/game:   {df['num_moves'].mean():.1f}")

    return df