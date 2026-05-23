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

OUTPUT_DIR              = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PBP_OUTPUT_PATH         = os.path.join(OUTPUT_DIR, "nflverse_play_by_play.parquet")
STATS_OUTPUT_PATH       = os.path.join(OUTPUT_DIR, "nflverse_player_stats.parquet")
STATS_SEASON_PATH       = os.path.join(OUTPUT_DIR, "nflverse_player_stats_season.parquet")
PFR_PASS_PATH           = os.path.join(OUTPUT_DIR, "pfr_pass_advstats.parquet")
PFR_PASS_SEASON_PATH    = os.path.join(OUTPUT_DIR, "pfr_pass_advstats_season.parquet")
PFR_RUSH_PATH           = os.path.join(OUTPUT_DIR, "pfr_rush_advstats.parquet")
PFR_REC_PATH            = os.path.join(OUTPUT_DIR, "pfr_rec_advstats.parquet")
NGS_PASSING_PATH        = os.path.join(OUTPUT_DIR, "ngs_passing.parquet")
PLAYERS_OUTPUT_PATH     = os.path.join(OUTPUT_DIR, "nflverse_players.parquet")
PARTICIPATION_OUTPUT    = os.path.join(OUTPUT_DIR, "nflverse_participation.parquet")
SNAP_COUNTS_OUTPUT      = os.path.join(OUTPUT_DIR, "nflverse_snap_counts.parquet")

# Columns to retain from play-by-play to reduce file size
PBP_COLUMNS = [
    "play_id", "game_id", "season", "week", "season_type",
    "posteam", "defteam", "play_type",
    "yards_gained", "air_yards", "yards_after_catch",
    "epa", "success", "touchdown",
    # Touchdown disambiguation — distinguish offensive credited TDs from
    # defensive return TDs (pick-6 / fumble-six).  Without these the only TD
    # column is the generic `touchdown` which inflates passer / receiver /
    # rusher credited TDs when defenders score on the same play.
    "pass_touchdown", "rush_touchdown",
    "td_player_id", "td_player_name", "td_team",
    "pass_attempt", "rush_attempt", "complete_pass", "incomplete_pass", "sack", "interception",
    # Filterable non-attempt plays: 2-pt conversions, kneeldowns, spikes —
    # all of these would otherwise leak into RZ-attempts / pass-attempts
    # counts and diverge from official NFL box-score numbers.
    "two_point_attempt", "qb_kneel", "qb_spike",
    "passer_player_id", "passer_player_name",
    "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name",
    "pass_location", "route",
    "goal_to_go", "yardline_100",
    "third_down_converted", "fourth_down_converted",
    "qb_epa", "xyac_epa", "xreception_prob",
    # QB scheme/role columns — used for play_action_rate and scramble_rate
    "play_action", "qb_scramble",
]

STATS_COLUMNS = [
    "player_id", "player_name", "player_display_name",
    "position", "recent_team",
    "season", "week", "season_type",
    "receiving_routes_run",
    "offense_snaps", "snap_counts_offense",
]

# Season-level (summary_level="reg") player stats retained for the QB Advanced
# Stats builder — raw counting stats, fantasy points, fumbles, sack data.
STATS_SEASON_COLUMNS = [
    "player_id", "player_name", "player_display_name",
    "position", "recent_team", "season", "games",
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "passing_air_yards", "passing_cpoe",
    "sacks_suffered", "sack_yards_lost", "sack_fumbles", "sack_fumbles_lost",
    "carries", "rushing_yards", "rushing_tds",
    "rushing_fumbles", "rushing_fumbles_lost",
    "fantasy_points", "fantasy_points_ppr",
]

# NextGen Stats passing columns — source for avg_time_to_throw.
NGS_PASSING_COLUMNS = [
    "season", "season_type", "week",
    "player_gsis_id", "player_display_name", "player_position", "team_abbr",
    "avg_time_to_throw", "avg_completed_air_yards", "avg_intended_air_yards",
    "aggressiveness", "attempts",
]

SEASONS = [2023, 2024, 2025]


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


def fetch_player_stats_season(seasons: list[int]) -> pl.DataFrame | None:
    """Load regular-season season-total player stats (summary_level='reg').

    Source for the QB Advanced Stats builder: raw counting stats, fantasy
    points, fumbles and sack data already aggregated to one row per
    (player, season).
    """
    all_frames = []
    for season in seasons:
        print(f"Loading season-total player stats for {season} season...")
        try:
            stats = nfl.load_player_stats(seasons=[season], summary_level="reg")
            available_cols = [c for c in STATS_SEASON_COLUMNS if c in stats.columns]
            stats = stats.select(available_cols)
            all_frames.append(stats)
            print(f"  Loaded {stats.height:,} season-total rows for {season} "
                  f"({len(available_cols)} cols).")
        except Exception as e:
            print(f"  WARNING: Could not load {season} season-total stats: {e}",
                  file=sys.stderr)

    if not all_frames:
        print("WARNING: No season-total player stats loaded.", file=sys.stderr)
        return None
    return pl.concat(all_frames, how="diagonal_relaxed")


def fetch_pfr_pass_season(seasons: list[int]) -> pl.DataFrame | None:
    """Load seasonal PFR advanced passing stats (summary_level='season').

    The seasonal table carries pocket_time, pressure_pct, scrambles and the
    blitz/hurry/hit/pressure counts that the weekly PFR advstats table omits.
    """
    print(f"Loading seasonal PFR pass advstats for {seasons}...")
    try:
        df = nfl.load_pfr_advstats(seasons, stat_type="pass", summary_level="season")
        print(f"  Loaded {df.height:,} seasonal PFR passing rows.")
        return df
    except Exception as e:
        print(f"  WARNING: Could not load seasonal PFR pass advstats: {e}",
              file=sys.stderr)
        return None


def fetch_ngs_passing(seasons: list[int]) -> pl.DataFrame | None:
    """Load NextGen Stats passing and keep only the season rows (week == 0).

    Source for avg_time_to_throw in the QB Advanced Stats builder.
    """
    all_frames = []
    for season in seasons:
        print(f"Loading NextGen passing stats for {season} season...")
        try:
            ngs = nfl.load_nextgen_stats(seasons=[season], stat_type="passing")
            if "week" in ngs.columns:
                ngs = ngs.filter(pl.col("week") == 0)        # season aggregate row
            if "season_type" in ngs.columns:
                ngs = ngs.filter(pl.col("season_type") == "REG")
            available = [c for c in NGS_PASSING_COLUMNS if c in ngs.columns]
            ngs = ngs.select(available)
            all_frames.append(ngs)
            print(f"  Loaded {ngs.height:,} NGS passing season rows for {season}.")
        except Exception as e:
            print(f"  WARNING: Could not load {season} NGS passing: {e}", file=sys.stderr)

    if not all_frames:
        print("WARNING: No NGS passing data loaded.", file=sys.stderr)
        return None
    return pl.concat(all_frames, how="diagonal_relaxed")


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


def fetch_participation(seasons: list[int]) -> pl.DataFrame | None:
    """
    Load play-level participation data (who is on the field per play).
    Used downstream to compute per-player routes run.
    Retains only the columns needed: game_id, play_id, season, offense_players, offense_positions.
    """
    all_frames = []
    for season in seasons:
        print(f"Loading participation data for {season} season...")
        try:
            part = nfl.load_participation(seasons=[season])
            keep = ["nflverse_game_id", "play_id", "offense_players", "offense_positions"]
            available = [c for c in keep if c in part.columns]
            part = part.select(available)
            part = part.with_columns(pl.lit(season).alias("season"))
            all_frames.append(part)
            print(f"  Loaded {part.height:,} plays for {season}.")
        except Exception as e:
            print(f"  WARNING: Could not load {season} participation data: {e}", file=sys.stderr)

    if not all_frames:
        print("WARNING: No participation data loaded.", file=sys.stderr)
        return None

    return pl.concat(all_frames, how="diagonal_relaxed")


def fetch_snap_counts(seasons: list[int]) -> pl.DataFrame | None:
    """
    Load per-game snap counts (offense_pct, offense_snaps) from nflverse.
    Used downstream as the primary snap_pct source for WR player overview.
    Covers all seasons and all players; avoids player_metrics name-mismatch gaps.
    """
    all_frames = []
    for season in seasons:
        print(f"Loading snap counts for {season} season...")
        try:
            snaps = nfl.load_snap_counts(seasons=[season])
            snaps = snaps.filter(pl.col("game_type") == "REG")
            keep = ["season", "week", "player", "pfr_player_id", "position",
                    "team", "game_id", "offense_pct"]
            available = [c for c in keep if c in snaps.columns]
            snaps = snaps.select(available)
            all_frames.append(snaps)
            print(f"  Loaded {snaps.height:,} snap rows for {season}.")
        except Exception as e:
            print(f"  WARNING: Could not load {season} snap counts: {e}", file=sys.stderr)

    if not all_frames:
        print("WARNING: No snap count data loaded.", file=sys.stderr)
        return None

    return pl.concat(all_frames, how="diagonal_relaxed")


def fetch_players() -> pl.DataFrame | None:
    """
    Load the nflverse player registry (all historical + current players).
    Retains only the fields needed for experience-tier classification.
    Returns None on failure so the pipeline can continue without it.
    """
    print("Loading nflverse player registry...")
    try:
        players = nfl.load_players()
        keep = ["gsis_id", "pfr_id", "display_name", "short_name", "position",
                "birth_date", "years_of_experience", "rookie_season",
                "last_season", "latest_team", "status"]
        available = [c for c in keep if c in players.columns]
        players = players.select(available)
        skill = players.filter(
            pl.col("position").is_in(["QB", "RB", "WR", "TE", "FB"])
        )
        print(f"  Loaded {skill.height:,} skill-position player records.")
        return skill
    except Exception as e:
        print(f"  WARNING: Could not load player registry: {e}", file=sys.stderr)
        return None


def main():
    pbp_df = fetch_play_by_play(SEASONS)
    save_parquet(pbp_df, PBP_OUTPUT_PATH, "play-by-play data")

    stats_df = fetch_player_stats(SEASONS)
    if stats_df is not None:
        save_parquet(stats_df, STATS_OUTPUT_PATH, "player stats")

    stats_season_df = fetch_player_stats_season(SEASONS)
    if stats_season_df is not None:
        save_parquet(stats_season_df, STATS_SEASON_PATH, "season-total player stats")

    pfr = fetch_pfr_advstats(SEASONS)
    pfr_paths = {
        "pass": PFR_PASS_PATH,
        "rush": PFR_RUSH_PATH,
        "rec":  PFR_REC_PATH,
    }
    for stat_type, df in pfr.items():
        if not df.is_empty():
            save_parquet(df, pfr_paths[stat_type], f"PFR {stat_type} advstats")

    pfr_pass_season_df = fetch_pfr_pass_season(SEASONS)
    if pfr_pass_season_df is not None and not pfr_pass_season_df.is_empty():
        save_parquet(pfr_pass_season_df, PFR_PASS_SEASON_PATH, "seasonal PFR pass advstats")

    ngs_passing_df = fetch_ngs_passing(SEASONS)
    if ngs_passing_df is not None and not ngs_passing_df.is_empty():
        save_parquet(ngs_passing_df, NGS_PASSING_PATH, "NGS passing")

    players_df = fetch_players()
    if players_df is not None:
        save_parquet(players_df, PLAYERS_OUTPUT_PATH, "player registry")

    part_df = fetch_participation(SEASONS)
    if part_df is not None:
        save_parquet(part_df, PARTICIPATION_OUTPUT, "participation data")

    snap_df = fetch_snap_counts(SEASONS)
    if snap_df is not None:
        save_parquet(snap_df, SNAP_COUNTS_OUTPUT, "snap counts")

    print("nflverse data pull complete.")


if __name__ == "__main__":
    main()
