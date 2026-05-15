"""
02_build_features.py
--------------------
Turn raw game logs into a model-ready feature table.

For every NBA game we compute features for BOTH teams using ONLY data from
games that happened BEFORE that game (no leakage):

    * Win % over the last 10 games
    * Points scored / allowed per game over last 10
    * Net rating (PF - PA) over last 10
    * Season-to-date win %
    * Days of rest since last game

Then we make "difference" features (home - away), which are usually the most
predictive single columns.

Run:   python 02_build_features.py
Output: data/features.csv
"""

import pandas as pd
import numpy as np

ROLLING_WINDOW = 10


def main() -> None:
    print("Loading data/raw_games.csv ...")
    raw = pd.read_csv("data/raw_games.csv", parse_dates=["GAME_DATE"])

    # nba_api gives one row per team per game. Split into home / away.
    # MATCHUP looks like "BOS vs. LAL" (BOS is home) or "LAL @ BOS" (LAL is away).
    raw["IS_HOME"] = ~raw["MATCHUP"].str.contains("@")

    keep = ["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID",
            "TEAM_ABBREVIATION", "PTS", "WL"]

    home = (raw[raw["IS_HOME"]][keep]
            .rename(columns=lambda c: f"HOME_{c}" if c not in
                    ("GAME_ID", "GAME_DATE", "SEASON") else c))
    away = (raw[~raw["IS_HOME"]][keep]
            .rename(columns=lambda c: f"AWAY_{c}" if c not in
                    ("GAME_ID", "GAME_DATE", "SEASON") else c))

    games = pd.merge(home, away, on=["GAME_ID", "GAME_DATE", "SEASON"])
    games["HOME_WIN"] = (games["HOME_WL"] == "W").astype(int)
    games = games.sort_values("GAME_DATE").reset_index(drop=True)
    print(f"Reconstructed {len(games):,} unique games")

    # ------------------------------------------------------------------
    # Build per-team rolling stats. Each game contributes 2 team-rows so a
    # team's rolling history is correct regardless of home/away.
    # ------------------------------------------------------------------
    rows = []
    for _, g in games.iterrows():
        rows.append({"GAME_ID": g["GAME_ID"], "GAME_DATE": g["GAME_DATE"],
                     "SEASON": g["SEASON"], "TEAM_ID": g["HOME_TEAM_ID"],
                     "PTS_FOR": g["HOME_PTS"], "PTS_AGAINST": g["AWAY_PTS"],
                     "WIN": g["HOME_WIN"]})
        rows.append({"GAME_ID": g["GAME_ID"], "GAME_DATE": g["GAME_DATE"],
                     "SEASON": g["SEASON"], "TEAM_ID": g["AWAY_TEAM_ID"],
                     "PTS_FOR": g["AWAY_PTS"], "PTS_AGAINST": g["HOME_PTS"],
                     "WIN": 1 - g["HOME_WIN"]})

    team_df = (pd.DataFrame(rows)
               .sort_values(["TEAM_ID", "GAME_DATE"])
               .reset_index(drop=True))

    # NOTE: .shift(1) BEFORE rolling -> guarantees current game is excluded.
    def rolling_mean(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()

    grp = team_df.groupby("TEAM_ID", group_keys=False)
    team_df["win_pct_10"]      = grp["WIN"].transform(rolling_mean)
    team_df["pts_for_10"]      = grp["PTS_FOR"].transform(rolling_mean)
    team_df["pts_against_10"]  = grp["PTS_AGAINST"].transform(rolling_mean)
    team_df["net_rating_10"]   = team_df["pts_for_10"] - team_df["pts_against_10"]
    team_df["rest_days"]       = grp["GAME_DATE"].diff().dt.days

    # Season-to-date win % (also shifted to exclude current game)
    team_df["season_wp"] = (team_df
                            .groupby(["TEAM_ID", "SEASON"])["WIN"]
                            .transform(lambda x: x.shift(1)
                                       .expanding(min_periods=5).mean()))

    # Merge team-level features back onto games as home_* / away_*
    feat_cols_team = ["win_pct_10", "net_rating_10", "season_wp", "rest_days"]
    home_feats = team_df.rename(columns={"TEAM_ID": "HOME_TEAM_ID",
                                          **{c: f"home_{c}" for c in feat_cols_team}})
    home_feats = home_feats[["GAME_ID", "HOME_TEAM_ID"] +
                            [f"home_{c}" for c in feat_cols_team]]
    away_feats = team_df.rename(columns={"TEAM_ID": "AWAY_TEAM_ID",
                                          **{c: f"away_{c}" for c in feat_cols_team}})
    away_feats = away_feats[["GAME_ID", "AWAY_TEAM_ID"] +
                            [f"away_{c}" for c in feat_cols_team]]

    out = (games
           .merge(home_feats, on=["GAME_ID", "HOME_TEAM_ID"])
           .merge(away_feats, on=["GAME_ID", "AWAY_TEAM_ID"]))

    # Difference features (usually most predictive)
    out["diff_win_pct_10"]    = out["home_win_pct_10"]    - out["away_win_pct_10"]
    out["diff_net_rating_10"] = out["home_net_rating_10"] - out["away_net_rating_10"]
    out["diff_season_wp"]     = out["home_season_wp"]     - out["away_season_wp"]
    out["diff_rest_days"]     = out["home_rest_days"]     - out["away_rest_days"]

    feature_cols = [
        "home_win_pct_10", "away_win_pct_10",
        "home_net_rating_10", "away_net_rating_10",
        "home_season_wp", "away_season_wp",
        "home_rest_days", "away_rest_days",
        "diff_win_pct_10", "diff_net_rating_10",
        "diff_season_wp", "diff_rest_days",
    ]
    before = len(out)
    out = out.dropna(subset=feature_cols)
    print(f"Dropped {before - len(out):,} early-season games with missing rolling stats")

    out.to_csv("data/features.csv", index=False)
    print(f"\nWrote {len(out):,} games with {len(feature_cols)} features to data/features.csv")
    print("Next step:  python 03_train_models.py")


if __name__ == "__main__":
    main()
