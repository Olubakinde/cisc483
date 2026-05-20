"""
Predict home win probability for a single matchup.

Example:
    python 05_predict.py --home BOS --away LAL
    python 05_predict.py --home DEN --away GSW --model logistic_regression
"""

import argparse
import joblib
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "home_win_pct_10", "away_win_pct_10",
    "home_net_rating_10", "away_net_rating_10",
    "home_season_wp", "away_season_wp",
    "home_rest_days", "away_rest_days",
    "diff_win_pct_10", "diff_net_rating_10",
    "diff_season_wp", "diff_rest_days",
]


def latest_stats(df, team):
    as_home = df[df["HOME_TEAM_ABBREVIATION"] == team]
    as_away = df[df["AWAY_TEAM_ABBREVIATION"] == team]
    if as_home.empty and as_away.empty:
        raise SystemExit(f"No data found for team '{team}'")

    candidates = pd.concat([
        as_home.sort_values("GAME_DATE").tail(1),
        as_away.sort_values("GAME_DATE").tail(1),
    ])
    row = candidates.sort_values("GAME_DATE").iloc[-1]

    if row["HOME_TEAM_ABBREVIATION"] == team:
        return {
            "win_pct_10":    row["home_win_pct_10"],
            "net_rating_10": row["home_net_rating_10"],
            "season_wp":     row["home_season_wp"],
        }
    return {
        "win_pct_10":    row["away_win_pct_10"],
        "net_rating_10": row["away_net_rating_10"],
        "season_wp":     row["away_season_wp"],
    }


def predict_with_model(model_name, X):
    if model_name == "neural_net":
        from nn_model import load_bundle, predict_proba as nn_predict_proba
        model, mean, scale = load_bundle("models/neural_net.pt")
        return float(nn_predict_proba(model, X, mean, scale)[0])
    model = joblib.load(f"models/{model_name}.joblib")
    return float(model.predict_proba(X)[0, 1])


def fair_moneyline(p):
    if p >= 0.5:
        return f"-{int(round(100 * p / (1 - p)))}"
    return f"+{int(round(100 * (1 - p) / p))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="Home team abbreviation")
    ap.add_argument("--away", required=True, help="Away team abbreviation")
    ap.add_argument("--model", default="neural_net",
                    choices=["logistic_regression", "neural_net"])
    ap.add_argument("--home_rest", type=int, default=2)
    ap.add_argument("--away_rest", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv("data/features.csv", parse_dates=["GAME_DATE"])
    home = latest_stats(df, args.home.upper())
    away = latest_stats(df, args.away.upper())

    row = {
        "home_win_pct_10":    home["win_pct_10"],
        "away_win_pct_10":    away["win_pct_10"],
        "home_net_rating_10": home["net_rating_10"],
        "away_net_rating_10": away["net_rating_10"],
        "home_season_wp":     home["season_wp"],
        "away_season_wp":     away["season_wp"],
        "home_rest_days":     args.home_rest,
        "away_rest_days":     args.away_rest,
    }
    row["diff_win_pct_10"]    = row["home_win_pct_10"]    - row["away_win_pct_10"]
    row["diff_net_rating_10"] = row["home_net_rating_10"] - row["away_net_rating_10"]
    row["diff_season_wp"]     = row["home_season_wp"]     - row["away_season_wp"]
    row["diff_rest_days"]     = row["home_rest_days"]     - row["away_rest_days"]

    X_df = pd.DataFrame([row])[FEATURE_COLS]
    if args.model == "neural_net":
        prob_home = predict_with_model(args.model, X_df.values)
    else:
        prob_home = predict_with_model(args.model, X_df)
    prob_away = 1.0 - prob_home

    print(f"\nMatchup: {args.home.upper()} (home) vs {args.away.upper()} (away)")
    print(f"Model: {args.model}")
    print(f"P({args.home.upper()} wins): {prob_home:.3f}  fair line {fair_moneyline(prob_home)}")
    print(f"P({args.away.upper()} wins): {prob_away:.3f}  fair line {fair_moneyline(prob_away)}\n")


if __name__ == "__main__":
    main()
