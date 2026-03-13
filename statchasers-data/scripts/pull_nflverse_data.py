"""
pull_nflverse_data.py

Fetches play-by-play, player stats, and PFR advanced stats from nflverse
via nflreadpy.  Stores all data as Parquet files for efficient downstream
processing.

nflreadpy returns Polars DataFrames, so all filtering and I/O uses Polars.

Output: data/raw/nflverse_play_by_play.parquet
        data/raw/nflverse_player_stats.parquet
        data/raw/pfr_pass_advstats.parquet   (QB pressure / pocket data)
        data/raw/pfr_rush_advstats.parquet   (RB broken tackles / YAC)
        data/raw/pfr_rec_advstats.parquet    (WR/TE drops / broken tackles)
"""

import os
import sys

try:
    import nflreadpy as nfl
except ImportError:
    print("ERROR: nflreadpy is not installed. Run: pip install nflreadpy", file=sys.stderr)
    sys.exit(1)

try:
    import polars as pl
except ImportError:
    print("ERROR: polars is not installed. Run: pip install polars", file=sys.stderr)
    sys.exit(1)

OUTPUT_DIR          = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PBP_OUTPUT_PATH     = os.path.join(OUTPUT_DIR, "nflverse_play_by_play.parquet")
STATS_OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "nflverse_player_stats.parquet")
PFR_PASS_PATH       = os.path.join(OUTPUT_DIR, "pfr_pass_advstats.parquet")
PFR_RUSH_PATH       = os.path.join(OUTPUT_DIR, "pfr_rush_advstats.parquet")
PFR_REC_PATH        = os.path.join(OUTPUT_DIR, "pfr_rec_advstats.parquet")

# Columns to retain from play-by-play to reduce file size
PBP_COLUMNS = [
    "play_id", "game_id", "season", "week", "season_type",
    "posteam", "defteam", "play_type",
    "yards_gained", "air_yards", "yards_after_catch",
    "epa", "success", "touchdown",
    "pass_attempt", "rush_attempt", "complete_pass", "incomplete_pass", "sack", "interception",
    "passer_player_id", "passer_player_name",
    "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name",
    "pass_location", "route",
    "goal_to_go", "yardline_100",
    "third_down_converted", "fourth_down_converted",
    "qb_epa", "xyac_epa", "xreception_prob",
]

STATS_COLUMNS = [
    "player_id", "player_name", "player_display_name",
    "position", "recent_team",
    "season", "week", "season_type",
    "receiving_routes_run",
    "offense_snaps", "snap_counts_offense",
]

SEASONS = [2024, 2025]


def fetch_play_by_play(seasons: list[int]) -> pl.DataFrame:
    """Load play-by-play data for given seasons from nflverse."""
    all_frames = []
    for season in seasons:
        print(f"Loading play-by-play data for {season} season...")
        try:
            pbp = nfl.load_pbp(seasons=[season])
            # Keep only columns that exist in this dataset
            available_cols = [c for c in PBP_COLUMNS if c in pbp.columns]
            pbp = pbp.select(available_cols)
            # Filter to regular season and playoffs only
            if "season_type" in pbp.columns:
                pbp = pbp.filter(pl.col("season_type").is_in(["REG", "POST"]))
            all_frames.append(pbp)
            print(f"  Loaded {pbp.height:,} plays for {season}.")
        except Exception as e:
            print(f"  WARNING: Could not load {season} PBP data: {e}", file=sys.stderr)

    if not all_frames:
        print("ERROR: No play-by-play data could be loaded.", file=sys.stderr)
        sys.exit(1)

    combined = pl.concat(all_frames, how="diagonal_relaxed")
    print(f"Total plays loaded: {combined.height:,}")
    return combined


def fetch_player_stats(seasons: list[int]) -> pl.DataFrame | None:
    """Load weekly player stats from nflverse for snap count and route data."""
    all_frames = []
    for season in seasons:
        print(f"Loading player stats for {season} season...")
        try:
            stats = nfl.load_player_stats(seasons=[season])
            available_cols = [c for c in STATS_COLUMNS if c in stats.columns]
            stats = stats.select(available_cols)
            all_frames.append(stats)
            print(f"  Loaded player stats for {season} ({stats.height:,} rows, {len(available_cols)} cols).")
        except Exception as e:
            print(f"  WARNING: Could not load {season} player stats: {e}", file=sys.stderr)

    if not all_frames:
        print("WARNING: No player stats loaded. Snap data will be approximated.", file=sys.stderr)
        return None

    combined = pl.concat(all_frames, how="diagonal_relaxed")
    return combined


def save_parquet(df: pl.DataFrame, path: str, label: str) -> None:
    """Save Polars DataFrame to Parquet format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_parquet(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"Saved {label} to {path} ({size_mb:.1f} MB, {df.height:,} rows)")


def fetch_pfr_advstats(seasons: list[int]) -> dict[str, pl.DataFrame]:
    """
    Load PFR (Pro Football Reference) advanced stats for pass, rush, and rec.
    Returns a dict keyed by stat_type.  Missing or errored types get an empty
    DataFrame so downstream code can safely handle absence.
    """
    results: dict[str, pl.DataFrame] = {}
    for stat_type in ("pass", "rush", "rec"):
        print(f"Loading PFR {stat_type} advstats for {seasons}...")
        try:
            df = nfl.load_pfr_advstats(seasons, stat_type=stat_type)
            # Filter to REG + POST only (drop preseason if present)
            if "game_type" in df.columns:
                df = df.filter(pl.col("game_type").is_in(["REG", "POST"]))
            results[stat_type] = df
            print(f"  Loaded {df.height:,} rows ({stat_type}).")
        except Exception as e:
            print(f"  WARNING: Could not load PFR {stat_type} advstats: {e}",
                  file=sys.stderr)
            results[stat_type] = pl.DataFrame()
    return results


def main():
    pbp_df = fetch_play_by_play(SEASONS)
    save_parquet(pbp_df, PBP_OUTPUT_PATH, "play-by-play data")

    stats_df = fetch_player_stats(SEASONS)
    if stats_df is not None:
        save_parquet(stats_df, STATS_OUTPUT_PATH, "player stats")

    pfr = fetch_pfr_advstats(SEASONS)
    pfr_paths = {
        "pass": PFR_PASS_PATH,
        "rush": PFR_RUSH_PATH,
        "rec":  PFR_REC_PATH,
    }
    for stat_type, df in pfr.items():
        if not df.is_empty():
            save_parquet(df, pfr_paths[stat_type], f"PFR {stat_type} advstats")

    print("nflverse data pull complete.")


if __name__ == "__main__":
    main()
