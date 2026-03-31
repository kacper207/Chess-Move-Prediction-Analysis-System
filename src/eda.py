import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json
import os

OUTPUT_DIR = "data/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    import chess.pgn

    pgn_path = r"C:\Users\HARDPC\PycharmProjects\chess_predictor\data\raw\lichess01.pgn"
    max_games = 1000000
    rows = []

    print(f"Wczytywanie {max_games} partii z PGN...")
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while len(rows) < max_games:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
            if game is None:
                break
            if game.errors:
                continue

            try:
                moves = [m.uci() for m in game.mainline_moves()]
            except Exception:
                continue

            if len(moves) < 10:
                continue

            eco = game.headers.get("ECO", "Unknown")
            if eco == "Unknown":
                continue

            result_raw = game.headers.get("Result", "*")
            if result_raw == "1-0":
                result = "white"
            elif result_raw == "0-1":
                result = "black"
            elif result_raw == "1/2-1/2":
                result = "draw"
            else:
                continue

            try:
                white_rating = int(game.headers.get("WhiteElo", 0))
                black_rating = int(game.headers.get("BlackElo", 0))
            except ValueError:
                continue

            if white_rating == 0 or black_rating == 0:
                continue

            rows.append({
                "num_moves": len(moves),
                "result": result,
                "opening_eco": eco,
                "opening_name": game.headers.get("Opening", "Unknown"),
                "white_rating": white_rating,
                "black_rating": black_rating,
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} games for EDA")
    return df


def plot_results(df):
    counts = df["result"].value_counts()
    colors = {"white": "#f0d9b5", "black": "#b58863", "draw": "#aaaaaa"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(r, "#888") for r in counts.index], edgecolor="black")
    ax.set_title("Rozklad wynikow partii", fontsize=14)
    ax.set_xlabel("Wynik")
    ax.set_ylabel("Liczba partii")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_results.png", dpi=150)
    plt.close()
    print("Wyniki partii:")
    print(counts.to_string())
    print(f"Win rate biale: {counts.get('white',0)/len(df)*100:.1f}%")
    print(f"Win rate czarne: {counts.get('black',0)/len(df)*100:.1f}%")
    print(f"Remisy: {counts.get('draw',0)/len(df)*100:.1f}%")
    print()


def plot_moves_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["num_moves"], bins=40, color="#5588cc", edgecolor="black", alpha=0.8)
    ax.axvline(df["num_moves"].mean(), color="red", linestyle="--",
               label=f"Srednia: {df['num_moves'].mean():.1f}")
    ax.axvline(df["num_moves"].median(), color="orange", linestyle="--",
               label=f"Mediana: {df['num_moves'].median():.1f}")
    ax.set_title("Rozklad liczby ruchow na partie", fontsize=14)
    ax.set_xlabel("Liczba ruchow")
    ax.set_ylabel("Liczba partii")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_moves_distribution.png", dpi=150)
    plt.close()
    print("Statystyki liczby ruchow:")
    print(df["num_moves"].describe().round(2).to_string())
    print()


def plot_top_openings(df, top_n=15):
    top = df["opening_eco"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1], color="#5588cc", edgecolor="black")
    ax.set_title(f"Top {top_n} najczestszych openingow (ECO)", fontsize=14)
    ax.set_xlabel("Liczba partii")
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_top_openings.png", dpi=150)
    plt.close()
    print(f"Top {top_n} openingow:")
    print(top.to_string())
    print()


def plot_opening_winrate(df, top_n=10):
    top_ecos = df["opening_eco"].value_counts().head(top_n).index
    subset = df[df["opening_eco"].isin(top_ecos)]
    winrate = subset.groupby("opening_eco")["result"].apply(
        lambda x: (x == "white").sum() / len(x) * 100
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(winrate.index, winrate.values, color="#88bb55", edgecolor="black")
    ax.axvline(50, color="red", linestyle="--", label="50%")
    ax.set_title("Win rate bialych per opening (top 10)", fontsize=14)
    ax.set_xlabel("Win rate bialych [%]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_opening_winrate.png", dpi=150)
    plt.close()
    print("Win rate bialych per opening:")
    print(winrate.round(1).to_string())
    print()


def plot_rating_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["white_rating"].dropna(), bins=30, alpha=0.6,
            color="#f0d9b5", edgecolor="black", label="Biale")
    ax.hist(df["black_rating"].dropna(), bins=30, alpha=0.6,
            color="#b58863", edgecolor="black", label="Czarne")
    ax.set_title("Rozklad rankingu ELO graczy", fontsize=14)
    ax.set_xlabel("ELO")
    ax.set_ylabel("Liczba partii")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_elo_distribution.png", dpi=150)
    plt.close()
    print("ELO statystyki:")
    print(f"  Biale - srednia: {df['white_rating'].mean():.0f}, std: {df['white_rating'].std():.0f}")
    print(f"  Czarne - srednia: {df['black_rating'].mean():.0f}, std: {df['black_rating'].std():.0f}")
    print()


def logistic_regression(df):
    print("=== REGRESJA LOGISTYCZNA ===")
    print("Cel: przewidywanie wyniku partii na podstawie openingu i ELO")
    print()

    le_eco = LabelEncoder()
    df["eco_encoded"] = le_eco.fit_transform(df["opening_eco"])

    df_clean = df.dropna(subset=["white_rating", "black_rating"])
    df_clean = df_clean[df_clean["result"] != "draw"]

    X = df_clean[["eco_encoded", "num_moves", "white_rating", "black_rating"]].values
    y = (df_clean["result"] == "white").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Dokladnosc modelu: {acc*100:.2f}%")
    print()
    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred,
                                target_names=["Wygrana czarnych", "Wygrana bialych"]))

    features = ["ECO opening", "Liczba ruchow", "ELO bialych", "ELO czarnych"]
    coefs = model.coef_[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#cc4444" if c < 0 else "#44aa44" for c in coefs]
    ax.bar(features, coefs, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Wspolczynniki regresji", fontsize=14)
    ax.set_ylabel("Wartosc wspolczynnika")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_logistic_regression.png", dpi=150)
    plt.close()

    print("Wspolczynniki regresji:")
    for f, c in zip(features, coefs):
        print(f"  {f}: {c:.4f}")
    print()


def plot_boxplot(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot liczby ruchow per wynik
    groups = [df[df["result"] == r]["num_moves"].values for r in ["white", "black", "draw"]]
    bp = axes[0].boxplot(groups, labels=["Wygrana bialych", "Wygrana czarnych", "Remis"],
                         patch_artist=True, notch=False)
    colors_box = ["#f0d9b5", "#b58863", "#aaaaaa"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
    axes[0].set_title("Rozklad liczby ruchow per wynik (ramka wasy)", fontsize=13)
    axes[0].set_ylabel("Liczba ruchow")
    axes[0].set_xlabel("Wynik partii")

    # Boxplot ELO per wynik
    white_elo_per_result = [df[df["result"] == r]["white_rating"].values for r in ["white", "black", "draw"]]
    bp2 = axes[1].boxplot(white_elo_per_result, labels=["Wygrana bialych", "Wygrana czarnych", "Remis"],
                          patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors_box):
        patch.set_facecolor(color)
    axes[1].set_title("Rozklad ELO bialych per wynik (ramka wasy)", fontsize=13)
    axes[1].set_ylabel("ELO")
    axes[1].set_xlabel("Wynik partii")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/7_boxplots.png", dpi=150)
    plt.close()
    print("Boxploty zapisane.")
    print()


def plot_scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Rozrzut ELO bialych vs liczba ruchow
    colors_map = {"white": "#f0d9b5", "black": "#b58863", "draw": "#aaaaaa"}
    for result, group in df.groupby("result"):
        axes[0].scatter(group["white_rating"], group["num_moves"],
                        alpha=0.15, s=8, label=result,
                        color=colors_map.get(result, "#888"))
    axes[0].set_title("Rozrzut: ELO bialych vs liczba ruchow", fontsize=13)
    axes[0].set_xlabel("ELO bialych")
    axes[0].set_ylabel("Liczba ruchow")
    axes[0].legend(title="Wynik")

    # Rozrzut ELO bialych vs ELO czarnych
    for result, group in df.groupby("result"):
        axes[1].scatter(group["white_rating"], group["black_rating"],
                        alpha=0.15, s=8, label=result,
                        color=colors_map.get(result, "#888"))
    axes[1].plot([800, 3000], [800, 3000], "r--", linewidth=1, label="ELO rowne")
    axes[1].set_title("Rozrzut: ELO bialych vs ELO czarnych", fontsize=13)
    axes[1].set_xlabel("ELO bialych")
    axes[1].set_ylabel("ELO czarnych")
    axes[1].legend(title="Wynik")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/8_scatter.png", dpi=150)
    plt.close()
    print("Wykresy rozrzutu zapisane.")
    print()


def plusy_minusy(df):
    print("=== PLUSY I MINUSY DANYCH ===")
    print()
    print("PLUSY:")
    print(f"  + Duza liczba partii: {len(df):,}")
    print(f"  + Wysoka roznorodnosc openingow: {df['opening_eco'].nunique()} unikalnych ECO")
    print(f"  + Zbilansowane ELO: biale srednia {df['white_rating'].mean():.0f} vs czarne {df['black_rating'].mean():.0f}")
    print(f"  + Niski procent remisow ({df[df['result']=='draw'].shape[0]/len(df)*100:.1f}%) - dane mniej niejednoznaczne")
    print()
    print("MINUSY:")

    imbalance = abs(df[df["result"]=="white"].shape[0] - df[df["result"]=="black"].shape[0])
    print(f"  - Niebalans klas: roznica wygrane biale vs czarne = {imbalance} partii")

    q1 = df["num_moves"].quantile(0.25)
    q3 = df["num_moves"].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df["num_moves"] < q1 - 1.5*iqr) | (df["num_moves"] > q3 + 1.5*iqr)]
    print(f"  - Outliery w liczbie ruchow: {len(outliers)} partii poza 1.5*IQR (min={df['num_moves'].min()}, max={df['num_moves'].max()})")

    rare_openings = df["opening_eco"].value_counts()
    rare = rare_openings[rare_openings < 10]
    print(f"  - Rzadkie openingi (<10 przykladow): {len(rare)} ECO codes - mala reprezentacja")

    print(f"  - Brak danych o czasie na zegarze i fazie gry (blitz/rapid/classical)")
    print()


if __name__ == "__main__":
    df = load_data()
    plot_results(df)
    plot_moves_distribution(df)
    plot_top_openings(df)
    plot_opening_winrate(df)
    plot_rating_distribution(df)
    logistic_regression(df)
    plot_boxplot(df)
    plot_scatter(df)
    plusy_minusy(df)
    print(f"Wszystkie wykresy zapisane w: {OUTPUT_DIR}/")