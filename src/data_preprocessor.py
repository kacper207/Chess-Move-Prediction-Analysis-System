import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import SEQ_LENGTH, PROCESSED_DATA_PATH, LABEL_ENCODER_PATH, VOCAB_SIZE_PATH


def preprocess(df):
    print("[1/4] Cleaning data...")
    df = df[df["num_moves"] >= 10]
    df = df[df["opening_eco"] != "Unknown"]
    df = df.dropna(subset=["moves", "opening_eco"])
    df = df.reset_index(drop=True)
    print(f"  Games after cleaning: {len(df):,}")

    if df.empty:
        print("ERROR: No data after cleaning.")
        return None

    print("[2/4] Tokenizing moves...")
    all_moves = [move for moves in df["moves"] for move in moves]
    le = LabelEncoder()
    le.fit(all_moves)
    vocab_size = len(le.classes_)
    print(f"  Vocabulary size: {vocab_size:,}")

    df["moves_encoded"] = df["moves"].apply(
        lambda moves: le.transform(moves).tolist()
    )

    print(f"[3/4] Creating sequences (length: {SEQ_LENGTH})...")
    X, y = [], []

    for moves in tqdm(df["moves_encoded"], desc="  Sequences"):
        if len(moves) <= SEQ_LENGTH:
            continue
        for i in range(len(moves) - SEQ_LENGTH):
            X.append(moves[i:i + SEQ_LENGTH])
            y.append(moves[i + SEQ_LENGTH])

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    print(f"  Total sequences: {len(X):,}")

    print("[4/4] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"  Train:      {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test:       {len(X_test):,}")

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VOCAB_SIZE_PATH), exist_ok=True)

    np.savez_compressed(
        PROCESSED_DATA_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    with open(VOCAB_SIZE_PATH, "w") as f:
        f.write(str(vocab_size))

    print(f"\nSaved to:")
    print(f"  {PROCESSED_DATA_PATH}")
    print(f"  {LABEL_ENCODER_PATH}")
    print(f"  {VOCAB_SIZE_PATH}")

    return X_train, y_train, X_val, y_val, X_test, y_test, le, vocab_size