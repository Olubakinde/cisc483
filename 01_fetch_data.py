"""
01_fetch_data.py
----------------
Fetch NBA regular-season games from the free nba_api package and save
them as a raw CSV.

Run this ONCE (or whenever you want to refresh the data):
    python 01_fetch_data.py

Output: data/raw_games.csv
"""

import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# Seasons to pull. Add or remove as you like.
SEASONS = [
    "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25",
]


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch all regular-season games for one season."""
    print(f"  -> {season} ...", end=" ", flush=True)
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",  # NBA
    )
    df = finder.get_data_frames()[0]
    df["SEASON"] = season
    print(f"{len(df)} team-rows")
    return df


def main() -> None:
    os.makedirs("data", exist_ok=True)
    frames = []
    print("Fetching NBA seasons from nba_api...")
    for season in SEASONS:
        try:
            frames.append(fetch_season(season))
            # nba.com rate limits aggressively; small sleep between calls
            time.sleep(1.0)
        except Exception as e:
            print(f"  !! failed for {season}: {e}")

    if not frames:
        raise SystemExit("No data fetched. Check internet / nba_api install.")

    combined = pd.concat(frames, ignore_index=True)
    out = "data/raw_games.csv"
    combined.to_csv(out, index=False)
    print(f"\nWrote {len(combined):,} rows to {out}")
    print("Next step:  python 02_build_features.py")


if __name__ == "__main__":
    main()
