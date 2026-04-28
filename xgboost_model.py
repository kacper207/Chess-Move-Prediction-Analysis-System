import numpy as np
import pandas as pd
import pickle
import os
import chess.pgn
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
import berserk
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import PGN_FILE_PATH, MAX_PGN_GAMES, MIN_MOVES, PLAYER_NAME, MAX_PLAYER_GAMES, PLAYER_PERF_TYPE

XGB_MODEL_PATH = "data/models/xgb_model.pkl"
XGB_ENCODER_PATH = "data/models/xgb_label_encoder.pkl"
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/eda", exist_ok=True)

N_OPENING_MOVES = 15


def load_pgn_for_xgb(max_games=MAX_PGN_GAMES):
    rows = []
    errors = 0

    print(f"Loading {max_games:,} games from PGN...")
    with open(PGN_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
        with tqdm(total=max_games, desc="Reading PGN") as pbar:
            while len(rows) < max_games:
                try:
                    game = chess.pgn.read_game(f)
                except Exception:
                    errors += 1
                    continue
                if game is None:
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

                result_raw = game.headers.get("Result", "*")
                if result_raw == "1-0":
                    result = 0
                elif result_raw == "0-1":
                    result = 1
                elif result_raw == "1/2-1/2":
                    result = 2
                else:
                    continue

                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                except ValueError:
                    continue
                if white_elo == 0 or black_elo == 0:
                    continue

                eco = game.headers.get("ECO", "Unknown")
                if eco == "Unknown":
                    continue

                opening_moves = moves[:N_OPENING_MOVES]
                while len(opening_moves) < N_OPENING_MOVES:
                    opening_moves.append("none")

                rows.append({
                    "result": result,
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "elo_diff": white_elo - black_elo,
                    "total_moves": len(moves),
                    "eco": eco,
                    **{f"move_{i+1}": opening_moves[i] for i in range(N_OPENING_MOVES)}
                })
                pbar.update(1)

    print(f"Loaded: {len(rows):,} games | Skipped: {errors:,}")
    return pd.DataFrame(rows)


def load_player_games_for_xgb(player_name=PLAYER_NAME):
    print(f"Fetching games for player: {player_name}...")
    client = berserk.Client()
    games = client.games.export_by_player(
        player_name,
        max=MAX_PLAYER_GAMES,
        perf_type=PLAYER_PERF_TYPE,
        as_pgn=False,
        opening=True
    )

    rows = []
    for g in list(games):
        moves = g.get("moves", "")
        if not moves:
            continue
        move_list = moves.split()
        if len(move_list) < MIN_MOVES:
            continue

        result_raw = g.get("winner", None)
        if result_raw == "white":
            result = 0
        elif result_raw == "black":
            result = 1
        else:
            result = 2

        players = g.get("players", {})
        white_elo = players.get("white", {}).get("rating", 0) or 0
        black_elo = players.get("black", {}).get("rating", 0) or 0
        if white_elo == 0 or black_elo == 0:
            continue

        eco = g.get("opening", {}).get("eco", "Unknown")
        if eco == "Unknown":
            continue

        opening_moves = move_list[:N_OPENING_MOVES]
        while len(opening_moves) < N_OPENING_MOVES:
            opening_moves.append("none")

        rows.append({
            "result": result,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "elo_diff": white_elo - black_elo,
            "total_moves": len(move_list),
            "eco": eco,
            **{f"move_{i+1}": opening_moves[i] for i in range(N_OPENING_MOVES)}
        })

    print(f"Fetched: {len(rows)} games for {player_name}")
    return pd.DataFrame(rows)


def encode_features(df, encoders=None, fit=True):
    df = df.copy()
    categorical_cols = ["eco"] + [f"move_{i+1}" for i in range(N_OPENING_MOVES)]

    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    feature_cols = ["total_moves"] + categorical_cols
    X = df[feature_cols].values
    y = df["result"].values if "result" in df.columns else None

    return X, y, encoders


def train_xgb(df):
    X, y, encoders = encode_features(df, fit=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=10000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        num_class=3,
        objective="multi:softmax",
        tree_method="hist",
        device="cuda",
        random_state=42,
        verbosity=1,
        early_stopping_rounds=30
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["White wins", "Black wins", "Draw"],
        zero_division=0
    ))

    evaluate_model(model, X_test, y_test, label="baseline")

    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(XGB_ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    return model, encoders, X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model, X_test, y_test, label="model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[{label}] Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred,
                                target_names=["White wins", "Black wins", "Draw"],
                                zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["White wins", "Black wins", "Draw"],
                yticklabels=["White wins", "Black wins", "Draw"])
    ax.set_title(f"Confusion Matrix [{label}]")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"data/eda/confusion_matrix_{label}.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved: data/eda/confusion_matrix_{label}.png")

    importance = model.feature_importances_
    feature_names = ["total_moves", "eco"] + [f"move_{i+1}" for i in range(N_OPENING_MOVES)]
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi_df["feature"], fi_df["importance"], color="#5588cc", edgecolor="black")
    ax.set_title(f"Feature Importance [{label}]")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"data/eda/feature_importance_{label}.png", dpi=150)
    plt.close()
    print(f"Feature importance saved: data/eda/feature_importance_{label}.png")

    return acc


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=30):
    print(f"\nStarting Optuna hyperparameter optimization ({n_trials} trials)...")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        }

        model = XGBClassifier(
            **params,
            eval_metric="mlogloss",
            num_class=3,
            objective="multi:softmax",
            tree_method="hist",
            device="cuda",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20
        )

        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest accuracy (val): {study.best_value*100:.2f}%")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig("data/eda/optuna_param_importance.png", dpi=150)
    plt.close()
    print("Optuna param importance saved: data/eda/optuna_param_importance.png")

    return study.best_params


def train_optimized_xgb(X_train, y_train, X_val, y_val, X_test, y_test, best_params, encoders):
    print("\nTraining optimized model with best hyperparameters...")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        **best_params,
        eval_metric="mlogloss",
        num_class=3,
        objective="multi:softmax",
        tree_method="hist",
        device="cuda",
        random_state=42,
        verbosity=1,
        early_stopping_rounds=30
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    print("\nOptimized model evaluation:")
    evaluate_model(model, X_test, y_test, label="optimized")

    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(XGB_ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    print(f"\nOptimized model saved to {XGB_MODEL_PATH}")
    return model


def predict_player(player_name=PLAYER_NAME):
    print(f"\nPredicting results for: {player_name}")

    with open(XGB_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(XGB_ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    df_player = load_player_games_for_xgb(player_name)
    if df_player.empty:
        print("No games found.")
        return

    X, y_true, _ = encode_features(df_player, encoders=encoders, fit=False)
    y_pred = model.predict(X)

    print(f"\nPredictions for {player_name} ({len(df_player)} games):")
    print(f"Accuracy vs actual results: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=["White wins", "Black wins", "Draw"],
                                zero_division=0))

    label_map = {0: "White wins", 1: "Black wins", 2: "Draw"}
    unique, counts = np.unique(y_pred, return_counts=True)
    print("Predicted outcome distribution:")
    for u, c in zip(unique, counts):
        print(f"  {label_map[u]}: {c} ({c/len(y_pred)*100:.1f}%)")


def predict_only(player_name):
    print(f"Loading model from {XGB_MODEL_PATH}...")
    with open(XGB_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(XGB_ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    print("Model loaded successfully.")
    predict_player(player_name)


if __name__ == "__main__":
    MODE = "train_and_optimize"

    if MODE == "train_and_optimize":
        df_pgn = load_pgn_for_xgb(max_games=1_000_000)
        model, encoders, X_train, y_train, X_val, y_val, X_test, y_test = train_xgb(df_pgn)
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=30)
        train_optimized_xgb(X_train, y_train, X_val, y_val, X_test, y_test, best_params, encoders)
        predict_player(PLAYER_NAME)

    elif MODE == "train":
        df_pgn = load_pgn_for_xgb(max_games=1_000_000)
        model, encoders, X_train, y_train, X_val, y_val, X_test, y_test = train_xgb(df_pgn)
        predict_player(PLAYER_NAME)

    elif MODE == "predict":
        predict_only("Aqua_Blazing")