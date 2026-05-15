# CISC 483 — NBA Game Win-Probability Model

Predicts the probability that the home team beats the away team using
historical NBA data, then compares fair odds against sportsbook lines
(DraftKings / FanDuel / Kalshi).

**Team:** Donald Lafferty, John Wisniewski, Tito Olubakinde

---

## What it does

| Step | File | What it produces |
|------|------|------------------|
| 1 | `01_fetch_data.py`     | `data/raw_games.csv` — every NBA regular-season game from 2019-20 through 2024-25, pulled live from `nba_api`. |
| 2 | `02_build_features.py` | `data/features.csv` — one row per game with engineered features (rolling win %, net rating, rest days, season win %), built with a strict no-leakage rule. |
| 3 | `03_train_models.py`   | Trains **two** models: Logistic Regression and a PyTorch Neural Network. Saves them to `models/`, and saves `training_curve.png` for the NN. Trains on all seasons EXCEPT 2024-25. |
| 4 | `04_evaluate.py`       | Prints log loss / Brier / AUC / accuracy for both models + naive baseline on the 2024-25 hold-out, and writes `results.csv`. |
| 5 | `05_predict.py`        | Predicts any matchup from the command line, with either model: `python 05_predict.py --home BOS --away LAL --model neural_net` |

Plus one shared module:

* `nn_model.py` — defines the neural network architecture (used by training, evaluation, and prediction).

---

## How to run (do this in order)

### 1. Install Python 3.10+ and create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the pipeline, top to bottom

```bash
python 01_fetch_data.py       # ~2-3 minutes (rate-limited)
python 02_build_features.py   # ~30 seconds
python 03_train_models.py     # ~1 minute — trains LR + Neural Net
python 04_evaluate.py         # instant — these are the numbers for your slides
```

After step 3, you'll have:
* `models/logistic_regression.joblib`
* `models/neural_net.pt`
* `training_curve.png` ← drop this straight into your slides

### 3. Predict a single matchup with either model

```bash
python 05_predict.py --home BOS --away LAL                              # Neural Net (default)
python 05_predict.py --home DEN --away GSW --model logistic_regression
python 05_predict.py --home BOS --away LAL --model neural_net
```

### 4. Open the presentation

Open `presentation.html` in any browser. That's your demo. The training curve is embedded inline.

---

## The two models

| Model | What it is | "Iterations" knob |
|-------|------------|--------------------|
| **Logistic Regression** | Linear baseline, interpretable coefficients | `max_iter` (LBFGS solver until convergence) |
| **Neural Network** | PyTorch MLP: 12 → 32 → 16 → 1, ReLU + Dropout(0.2) | `N_EPOCHS=80`, early stopping patience=12 |

The neural network uses real **epochs** (full passes through the training data) with:
* Binary cross-entropy loss (= log loss, our primary metric)
* Adam optimizer, learning rate 1e-3, batch size 64
* Mini-batch SGD on an 85/15 train/validation split
* Early stopping on validation loss
* Best weights restored before saving

---

## Troubleshooting

* **`nba_api` times out** — the NBA site rate-limits. Re-run `01_fetch_data.py`; if it keeps failing, bump the `time.sleep(1.0)` to `time.sleep(3.0)` in that file.
* **`ModuleNotFoundError: torch`** — you forgot the venv. Run `source venv/bin/activate` and `pip install -r requirements.txt` again.
* **`FileNotFoundError: data/raw_games.csv`** — you skipped step 1.
* **Neural net loss is `nan`** — almost always a feature scaling issue. The script standardizes features automatically; if you modify it, make sure that still happens.

---

## Files in this repo

```
sports_prediction/
├── README.md              ← this file
├── requirements.txt
├── nn_model.py            ← neural network architecture (shared module)
├── 01_fetch_data.py       ← data ingestion
├── 02_build_features.py   ← feature engineering
├── 03_train_models.py     ← train LR + Neural Network
├── 04_evaluate.py         ← evaluation & comparison vs baseline
├── 05_predict.py          ← predict a single matchup
├── presentation.html      ← presentation website (training curve embedded)
├── training_curve.png     ← created when you run step 3
├── results.csv            ← created when you run step 4
├── data/                  ← created when you run step 1
│   ├── raw_games.csv
│   └── features.csv
└── models/                ← created when you run step 3
    ├── logistic_regression.joblib
    └── neural_net.pt
```
