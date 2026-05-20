# CISC 483 — NBA Game Win-Probability Model

Predicts the probability that the home team beats the away team using
historical NBA data, then compares fair odds against sportsbook lines
(DraftKings / FanDuel / Kalshi).

**Team:** Donald Lafferty, John Wisniewski, Tito Olubakinde

---

## What it does

| Step | File | What it produces |
|------|------|------------------|
| 1 | `01_fetch_data.py`     | `data/raw_games.csv`  every NBA regular-season game from 2019-20 through 2024-25, pulled live from `nba_api`. |
| 2 | `02_build_features.py` | `data/features.csv`  one row per game with engineered features (rolling win %, net rating, rest days, season win %), built with a strict no-leakage rule. |
| 3 | `03_train_models.py`   | Trains **two** models: Logistic Regression and a PyTorch Neural Network. Saves them to `models/`, and saves `training_curve.png` for the NN. Trains on all seasons EXCEPT 2024-25. |
| 4 | `04_evaluate.py`       | Prints log loss / Brier / AUC / accuracy for both models + naive baseline on the 2024-25 hold-out, and writes `results.csv`. |
| 5 | `05_predict.py`        | Predicts any matchup from the command line, with either model: `python 05_predict.py --home BOS --away LAL --model neural_net` |

Plus one shared module:

* `nn_model.py` — defines the neural network architecture (used by training, evaluation, and prediction).

---

## How to run 
### 1. Install Python

```bash
python3 -m venv venv
source venv/bin/activate         
pip install -r requirements.txt
```

###pipeline

```bash
python 01_fetch_data.py       
python 02_build_features.py  
python 03_train_models.py     
python 04_evaluate.py          
```

* `models/logistic_regression.joblib`
* `models/neural_net.pt`
* `training_curve.png`

### 3. Predict a single matchup with either model

```bash
python 05_predict.py --home BOS --away LAL                             
python 05_predict.py --home DEN --away GSW --model logistic_regression
python 05_predict.py --home BOS --away LAL --model neural_net
```

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
