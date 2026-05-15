"""
03_train_models.py
------------------
Train TWO models on the feature table:
    1. Logistic Regression (baseline, interpretable)
    2. Neural Network       (PyTorch MLP, trained with epochs + early stopping)

We hold out the most recent season as the test set (TEST_SEASON below).
This is a TIME-BASED split, the only correct way to evaluate a forecasting
model -- you should never train on future and test on past.

The neural network also saves a training-curve plot to `training_curve.png`,
which you can drop straight into the presentation.

Run:   python 03_train_models.py
Output:
    models/logistic_regression.joblib
    models/neural_net.pt
    training_curve.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nn_model import WinProbNet, save_bundle

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

# ---- Neural network hyperparameters ----
N_EPOCHS    = 80
BATCH_SIZE  = 64
LEARNING_RATE = 1e-3
PATIENCE    = 12      # early-stopping patience (epochs with no val improvement)
HIDDEN_1    = 32
HIDDEN_2    = 16
SEED        = 42


def train_classical(X_train, y_train) -> None:
    print("\n[1/2] Logistic Regression")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
    ])
    lr.fit(X_train, y_train)
    joblib.dump(lr, "models/logistic_regression.joblib")
    print("      saved -> models/logistic_regression.joblib")


def train_neural_net(X_train_df, y_train_series) -> None:
    """Train an MLP with a proper epoch loop and early stopping."""
    print("\n[2/2] Neural Network (PyTorch MLP)")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Carve a small validation set out of training (to monitor + early stop)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_df.values, y_train_series.values,
        test_size=0.15, random_state=SEED, stratify=y_train_series.values,
    )

    # Standardize features (NN needs this; saved alongside the model)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # Tensors / DataLoader
    X_tr_t  = torch.FloatTensor(X_tr_s)
    y_tr_t  = torch.FloatTensor(y_tr).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_s)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=BATCH_SIZE, shuffle=True)

    # Build model
    n_features = X_tr.shape[1]
    model = WinProbNet(n_features=n_features, h1=HIDDEN_1, h2=HIDDEN_2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # BCEWithLogitsLoss = numerically stable sigmoid + binary cross-entropy.
    # Binary cross-entropy IS log loss -- the exact metric in our evaluation.
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    patience_left = PATIENCE

    print(f"      epochs={N_EPOCHS} batch={BATCH_SIZE} lr={LEARNING_RATE} "
          f"arch={n_features}->{HIDDEN_1}->{HIDDEN_2}->1")
    print(f"      train={len(X_tr_t)} val={len(X_val_t)}")
    print("      epoch | train_loss | val_loss")
    print("      ----- | ---------- | --------")

    for epoch in range(1, N_EPOCHS + 1):
        # --- Training pass ---
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(X_tr_t)

        # --- Validation pass ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early-stopping bookkeeping (save the best weights)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1

        # Print every 5 epochs (plus the first and last)
        if epoch == 1 or epoch % 5 == 0 or epoch == N_EPOCHS:
            print(f"      {epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f}")

        if patience_left <= 0:
            print(f"      early stop at epoch {epoch}  (best val={best_val:.4f})")
            break

    # Restore the best weights we saw
    if best_state is not None:
        model.load_state_dict(best_state)

    save_bundle(
        model=model,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        arch={"n_features": n_features, "h1": HIDDEN_1, "h2": HIDDEN_2},
        path="models/neural_net.pt",
    )
    print("      saved -> models/neural_net.pt")

    _plot_training_curve(train_losses, val_losses)


def _plot_training_curve(train_losses, val_losses) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    fig.patch.set_facecolor("#FBF8F0")
    ax.set_facecolor("#FBF8F0")

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#14213D", lw=2.2, label="Train loss")
    ax.plot(epochs, val_losses,   color="#C84B19", lw=2.2, label="Validation loss")
    best_epoch = int(np.argmin(val_losses)) + 1
    ax.axvline(best_epoch, color="#8B8474", ls="--", lw=1, alpha=0.7)
    ax.text(best_epoch + 0.5, max(val_losses),
            f"  best val @ epoch {best_epoch}",
            fontsize=10, color="#8B8474", va="top")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Binary cross-entropy (log loss)", fontsize=12)
    ax.set_title("Neural network training curve",
                 fontsize=14, fontweight="bold", color="#14213D", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#C9BFA9")
    ax.spines["left"].set_color("#C9BFA9")
    ax.grid(True, alpha=0.3, color="#C9BFA9")
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=140, facecolor="#FBF8F0")
    plt.close()
    print("      saved -> training_curve.png (use this in your slides)")


def main() -> None:
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/features.csv", parse_dates=["GAME_DATE"])
    train = df[df["SEASON"] != TEST_SEASON]
    test  = df[df["SEASON"] == TEST_SEASON]
    print(f"Train: {len(train):,} games  (everything except {TEST_SEASON})")
    print(f"Test : {len(test):,} games  ({TEST_SEASON} held out)")

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    train_classical(X_train, y_train)
    train_neural_net(X_train, y_train)

    print("\nAll models trained. Next step:  python 04_evaluate.py")


if __name__ == "__main__":
    main()
