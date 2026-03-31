
DATA_SOURCE = "pgn"  # "pgn" = plik PGN, "player" = konkretny gracz z Lichess

PGN_FILE_PATH = r"C:\Users\HARDPC\PycharmProjects\chess_predictor\data\raw\lichess01.pgn"
MAX_PGN_GAMES = 1000000

PLAYER_NAME = "RoDoSfAtiHi"
MAX_PLAYER_GAMES = 100000
PLAYER_PERF_TYPE = "blitz"

MIN_MOVES = 10
SEQ_LENGTH = 20

PROCESSED_DATA_PATH = "data/processed/sequences.npz"
LABEL_ENCODER_PATH = "data/processed/label_encoder.pkl"
VOCAB_SIZE_PATH = "data/processed/vocab_size.txt"