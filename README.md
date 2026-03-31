## What it does

- Ingests large-scale PGN game data from database.lichess.org (tested on 1,000,000+ games)
- Fetches individual player game history via the Lichess API (berserk)
- Performs exploratory data analysis: result distributions, opening frequency, ELO distributions, move count statistics
- Trains an LSTM model to predict the next move in a game based on sequences of previous moves
- Classifies chess openings using logistic regression / XGBoost based on ECO codes

## How it works

Games are streamed directly from compressed .pgn.zst files without loading everything into RAM. Moves are encoded using UCI notation and tokenized into a vocabulary of ~1,956 unique moves. Sequences of 20 moves are used as input to the LSTM, which predicts the most likely next move. The system supports two data sources — bulk PGN files for training a global model, and the Lichess API for analyzing a specific player.

## Key findings from EDA

- White wins 49.7% of games, black 46.5%, draws only 3.8%
- Average game length is 67 moves (median 63), with a right-skewed distribution
- Draw games are significantly longer (median ~110 moves) than decisive games
- ELO rating is the strongest predictor of game outcome
- Opening choice (ECO code) has minimal impact on outcome at amateur level
- Logistic regression baseline accuracy: 55.68% — confirming the need for sequence-based models

## Tech stack

- **python-chess** — PGN parsing, move validation, ECO codes
- **PyTorch** — LSTM model training
- **scikit-learn** — tokenization, logistic regression, evaluation metrics
- **pandas / numpy** — data processing
- **berserk** — Lichess API client
- **zstandard** — streaming decompression of .pgn.zst files
- **matplotlib** — data visualization

## Data

Monthly PGN dumps from [database.lichess.org](https://database.lichess.org). Files are read in streaming mode — no full decompression required.
