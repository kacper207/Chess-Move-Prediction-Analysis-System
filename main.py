from config import DATA_SOURCE
from src.data_preprocessor import preprocess

if __name__ == "__main__":
    print(f"Data source: {DATA_SOURCE}\n")

    if DATA_SOURCE == "pgn":
        from src.pgn_collector import stream_pgn
        df = stream_pgn()

    elif DATA_SOURCE == "player":
        from src.player_collector import collect_player_games
        df = collect_player_games()

    else:
        print(f"Unknown DATA_SOURCE: '{DATA_SOURCE}'. Use 'pgn' or 'player'.")
        exit(1)

    result = preprocess(df)

    if result is not None:
        print("\nData is ready for training.")
    else:
        print("\nSomething went wrong.")