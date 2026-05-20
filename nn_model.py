"""
nn_model.py
-----------
Shared neural network architecture used by training, evaluation, and prediction.
We have ~6000 training games and only 12 features. Large networks
would overfit immediately. The point of including a neural net is to show the
classical-vs-deep comparison, not to oversell the model.
"""

import json
import numpy as np
import torch
import torch.nn as nn


class WinProbNet(nn.Module):
    """Small MLP that outputs a single logit per game."""

    def __init__(self, n_features: int, h1: int = 32, h2: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_bundle(model: WinProbNet, scaler_mean: np.ndarray,
                scaler_scale: np.ndarray, arch: dict, path: str) -> None:
    """Save model weights + the StandardScaler params together."""
    torch.save({
        "state_dict": model.state_dict(),
        "arch": arch,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
    }, path)


def load_bundle(path: str = "models/neural_net.pt"):
    """Load model + scaler back. Returns (model, mean, scale)."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    arch = bundle["arch"]
    model = WinProbNet(n_features=arch["n_features"],
                       h1=arch["h1"], h2=arch["h2"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    mean  = np.array(bundle["scaler_mean"])
    scale = np.array(bundle["scaler_scale"])
    return model, mean, scale


def predict_proba(model: WinProbNet, X: np.ndarray,
                  mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Run a numpy feature matrix through the model -> P(home_win)."""
    X_scaled = (X - mean) / scale
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_scaled))
        probs = torch.sigmoid(logits).squeeze(-1).numpy()
    return probs
