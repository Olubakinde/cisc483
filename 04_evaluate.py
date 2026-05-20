"""
04_evaluate.py
--------------
Score both trained models on the held-out test season and compare them
to a naive baseline.

Output: prints results, writes results.csv
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss,
                              log_loss, roc_auc_score)

from nn_model import load_bundle, predict_proba as nn_predict_proba

FEATURE_COLS = [
    "home_win_pct_10", "away_win_pct_10",
    "home_net_rating_10", "away_net_rating_10",
    "home_season_wp", "away_season_wp",
    "home_rest_days", "away_rest_days",
    "diff_win_pct_10", "diff_net_rating_10",
    "diff_season_wp", "diff_rest_days",
]
TARGET = "HOME_WIN"
TEST_SEASON = "2024-25"


def metrics(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "log_loss":  log_loss(y_true, y_prob),
        "brier":     brier_score_loss(y_true, y_prob),
        "auc_roc":   roc_auc_score(y_true, y_prob),
        "accuracy":  accuracy_score(y_true, y_pred),
    }


def report(name: str, m: dict) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for k, v in m.items():
        print(f"  {k:<10s} {v:.4f}")


def main() -> None:
    df = pd.read_csv("data/features.csv", parse_dates=["GAME_DATE"])
    test = df[df["SEASON"] == TEST_SEASON]
    X_test_df = test[FEATURE_COLS]            
    X_test_np = X_test_df.values             
    y_test = test[TARGET].values
    print(f"Evaluating on {len(test):,} games from {TEST_SEASON}")

    # ----- Baseline: naive home-win rate -----
    p_home = df[df["SEASON"] != TEST_SEASON][TARGET].mean()
    naive_prob = np.full(len(y_test), p_home)
    m_naive = metrics(y_test, naive_prob)
    report(f"NAIVE  (always p={p_home:.3f})", m_naive)

    # ----- Logistic Regression -----
    lr = joblib.load("models/logistic_regression.joblib")
    lr_prob = lr.predict_proba(X_test_df)[:, 1]
    m_lr = metrics(y_test, lr_prob)
    report("LOGISTIC REGRESSION", m_lr)

    # ----- Neural Network -----
    nn_model, mean, scale = load_bundle("models/neural_net.pt")
    nn_prob = nn_predict_proba(nn_model, X_test_np, mean, scale)
    m_nn = metrics(y_test, nn_prob)
    report("NEURAL NETWORK", m_nn)

    print("\nSuccess criterion (from your proposal):")
    print("  beat the naive baseline on log loss AND Brier score.\n")

    summary = pd.DataFrame([
        {"model": "naive",                **m_naive},
        {"model": "logistic_regression",  **m_lr},
        {"model": "neural_net",           **m_nn},
    ])
    summary.to_csv("results.csv", index=False)
    print("Wrote results.csv (use these numbers in your slides).")


if __name__ == "__main__":
    main()
