"""
compute_player_metrics.py

Reads raw nflverse play-by-play data and Sleeper player metadata,
then computes all advanced player-level metrics needed for the
StatChasers Performance Analytics dashboard.

Season split strategy:
  - Dashboard metrics (games, shares, trends, EPA) use 2025-only data.
  - Context labels (threeYearContext, careerArc) use all available seasons
    so veteran stars are classified correctly regardless of 2025 sample size.

Input:  data/raw/nflverse_play_by_play.parquet
        data/raw/nflverse_player_stats.parquet (optional)
        data/raw/sleeper_players.json

Output: data/processed/player_metrics.json
"""

import json
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

PBP_PATH = os.path.join(RAW_DIR, "nflverse_play_by_play.parquet")
STATS_PATH = os.path.join(RAW_DIR, "nflverse_player_stats.parquet")
SLEEPER_PATH = os.path.join(RAW_DIR, "sleeper_players.json")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "player_metrics.json")

# The season whose data drives all dashboard (usage/share/trend) metrics
CURRENT_SEASON = 2025

# Minimum 2025-season plays to include a player
MIN_PLAYS = 20

# Minimum 2025-season games before applying narrative labels
MIN_GAMES_FOR_LABELS = 6

# Minimum career games (all seasons) before applying "Elite" tier labels
MIN_CAREER_GAMES_ELITE = 32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Load all raw data sources.

    Returns
    -------
    pbp_all   : DataFrame of all seasons (used for career-context labels)
    pbp_2025  : DataFrame of CURRENT_SEASON only (used for dashboard metrics)
    stats     : Weekly player stats (optional; may be empty)
    sleeper_players : list of Sleeper player dicts
    """
    if not os.path.exists(PBP_PATH):
        print(f"ERROR: Play-by-play data not found at {PBP_PATH}", file=sys.stderr)
        print("Run pull_nflverse_data.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading play-by-play data...")
    pbp_all = pd.read_parquet(PBP_PATH)
    print(f"  {len(pbp_all):,} total plays loaded ({pbp_all['season'].nunique() if 'season' in pbp_all.columns else '?'} seasons).")

    if "season" in pbp_all.columns:
        pbp_2025 = pbp_all[pbp_all["season"] == CURRENT_SEASON].copy()
        print(f"  {len(pbp_2025):,} plays in {CURRENT_SEASON} season.")
    else:
        print(f"WARNING: No 'season' column found; treating all data as {CURRENT_SEASON}.", file=sys.stderr)
        pbp_2025 = pbp_all.copy()

    stats = pd.DataFrame()
    if os.path.exists(STATS_PATH):
        print("Loading player stats data...")
        stats = pd.read_parquet(STATS_PATH)
        print(f"  {len(stats):,} player-week records loaded.")

    if not os.path.exists(SLEEPER_PATH):
        print(f"ERROR: Sleeper player data not found at {SLEEPER_PATH}", file=sys.stderr)
        print("Run pull_sleeper_players.py first.", file=sys.stderr)
        sys.exit(1)

    with open(SLEEPER_PATH) as f:
        sleeper_players = json.load(f)

    print(f"  {len(sleeper_players)} Sleeper players loaded.")
    return pbp_all, pbp_2025, stats, sleeper_players


# ---------------------------------------------------------------------------
# Name resolution helpers
# ---------------------------------------------------------------------------

def build_player_lookup(sleeper_players: list[dict]) -> dict[str, dict]:
    """Build a full-name -> metadata lookup from Sleeper data."""
    lookup = {}
    for p in sleeper_players:
        name = p.get("full_name", "").strip()
        if name:
            lookup[name] = p
    return lookup


def _nflverse_abbrev(full_name: str) -> str:
    """
    Derive the nflverse-style abbreviated name from a Sleeper full name.

    nflverse uses 'FirstInitial.LastName', e.g.:
      "Ja'Marr Chase"  -> "J.Chase"
      "CeeDee Lamb"    -> "C.Lamb"
      "Saquon Barkley" -> "S.Barkley"
      "Travis Kelce"   -> "T.Kelce"
    """
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    first_initial = parts[0][0].upper()
    last_name = parts[-1]
    return f"{first_initial}.{last_name}"


def build_abbreviated_lookup(sleeper_players: list[dict]) -> dict[str, str]:
    """
    Build an abbreviated-name -> full-name lookup so nflverse PBP names
    (e.g. 'J.Chase') resolve to canonical Sleeper names ('Ja'Marr Chase').

    Ambiguous abbreviations (two active players with the same abbrev) are
    dropped so we never silently assign the wrong name.
    """
    counts: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for p in sleeper_players:
        full = p.get("full_name", "").strip()
        if not full:
            continue
        abbrev = _nflverse_abbrev(full)
        counts[abbrev] = counts.get(abbrev, 0) + 1
        mapping[abbrev] = full
    return {abbrev: full for abbrev, full in mapping.items() if counts[abbrev] == 1}


def resolve_full_name(
    pbp_name: str,
    player_lookup: dict[str, dict],
    abbreviated_lookup: dict[str, str],
) -> str:
    """
    Return the canonical Sleeper full name for a PBP player name.

    Resolution order:
    1. Direct match in Sleeper full-name lookup (already a full name).
    2. Match via abbreviated-name lookup (PBP abbreviation -> Sleeper full name).
    3. Fall back to the PBP name unchanged.
    """
    if pbp_name in player_lookup:
        return pbp_name
    if pbp_name in abbreviated_lookup:
        return abbreviated_lookup[pbp_name]
    return pbp_name


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def safe_pct(numerator: float, denominator: float, scale: float = 100.0) -> float:
    """
    Safe percentage calculation.  All share/rate fields in the output are
    in percent form (e.g. 24.9, not 0.249).  WOPR is the only field that
    intentionally stays on a 0-~1.5 index scale.
    """
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * scale, 2)


# ---------------------------------------------------------------------------
# Career context computation (multi-season)
# ---------------------------------------------------------------------------

def compute_career_stats(pbp_all: pd.DataFrame) -> dict[str, dict]:
    """
    Batch-compute career-level context (all available seasons) for every
    player appearing in the PBP data.

    Returns a dict keyed by PBP abbreviated name, e.g. 'J.Chase'.
    Values: { career_games: int, career_epa_per_play: float }

    Used exclusively for threeYearContext and careerArc labels — not for
    any current-season dashboard metric.
    """
    frames = []

    def _agg(df: pd.DataFrame, name_col: str, play_col: str) -> pd.DataFrame:
        if name_col not in df.columns or play_col not in df.columns:
            return pd.DataFrame()
        sub = df[df[play_col] == 1].dropna(subset=[name_col])
        if sub.empty:
            return pd.DataFrame()
        return sub.groupby(name_col).agg(
            career_games=("game_id", "nunique"),
            career_epa_sum=("epa", "sum"),
            career_plays=(play_col, "sum"),
        ).reset_index().rename(columns={name_col: "pbp_name"})

    frames.append(_agg(pbp_all, "passer_player_name", "pass_attempt"))
    frames.append(_agg(pbp_all, "receiver_player_name", "pass_attempt"))
    frames.append(_agg(pbp_all, "rusher_player_name", "rush_attempt"))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)
    # When a player appears in multiple groups (e.g. RB with carries and targets),
    # keep the entry that has the most games (most representative of career).
    combined = (
        combined
        .sort_values("career_games", ascending=False)
        .drop_duplicates("pbp_name")
    )
    combined["career_epa_per_play"] = (
        combined["career_epa_sum"] / combined["career_plays"].clip(lower=1)
    ).round(3)

    return {
        row["pbp_name"]: {
            "career_games": int(row["career_games"]),
            "career_epa_per_play": float(row["career_epa_per_play"]),
        }
        for _, row in combined.iterrows()
    }


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

# Position-aware age thresholds:
#   (ascending_max, peak_max, plateau_max)
# Age <= ascending_max → Ascending (if productive)
# ascending_max < age <= peak_max → Peak (if productive) / Stable
# peak_max < age <= plateau_max → Plateau
# age > plateau_max → Veteran or Declining
_AGE_THRESHOLDS: dict[str, tuple[int, int, int]] = {
    "QB": (26, 34, 37),   # QBs peak late and sustain longest
    "WR": (25, 29, 32),   # WRs hit peak in late 20s
    "RB": (24, 27, 30),   # RBs decline fastest
    "TE": (25, 32, 34),   # TEs develop slowly, sustain long
}


def classify_career_arc(
    pos: str,
    age: int | None,
    career_epa_per_play: float,
    current_snap_share: float,
    career_games: int,
    current_games: int,
) -> str | None:
    """
    Position-aware career arc label using age curves and multi-season EPA.

    - Returns None when age is unavailable.
    - Returns 'Insufficient Data' when the current-season sample is too small.
    - Uses career EPA (all seasons) so veterans aren't mislabeled by a
      small 2025 sample.
    - Never returns 'Developing' for a player with 30+ career games — they
      are past the developing stage regardless of current-season metrics.
    """
    if age is None:
        return None
    if current_games < MIN_GAMES_FOR_LABELS:
        return "Insufficient Data"

    ascending_max, peak_max, plateau_max = _AGE_THRESHOLDS.get(pos, (25, 30, 33))

    is_effective = career_epa_per_play > 0.02
    is_positive = career_epa_per_play > 0.0
    is_declining_epa = career_epa_per_play < -0.03
    is_veteran = career_games >= 32

    if age <= ascending_max:
        if is_effective:
            return "Ascending"
        # A player with 30+ career games is no longer developing even if young
        return "Stable" if is_veteran else "Developing"

    if age <= peak_max:
        if is_effective:
            return "Peak"
        if is_positive:
            return "Stable"
        return "Plateau"

    if age <= plateau_max:
        if is_effective:
            return "Plateau"
        if is_declining_epa:
            return "Declining"
        return "Veteran"

    # Oldest bracket
    if is_declining_epa:
        return "Declining"
    return "Veteran"


def classify_three_year_context(
    pos: str,
    career_epa_per_play: float,
    current_snap_share: float,
    career_games: int,
    current_games: int,
) -> str | None:
    """
    Multi-season narrative label for the threeYearContext field.

    Uses career EPA (all seasons) + current-season snap share.
    'Elite' labels require 32+ career games AND 9+ current-season games
    so that small-sample or early-career players are never over-labelled.
    """
    if current_games < MIN_GAMES_FOR_LABELS:
        return None

    elite_qualified = career_games >= MIN_CAREER_GAMES_ELITE and current_games >= 9
    starter_qualified = career_games >= 16

    if pos == "QB":
        if career_epa_per_play > 0.15 and elite_qualified:
            return "Elite QB1 trajectory"
        if career_epa_per_play > 0.08 and starter_qualified:
            return "Solid franchise QB"
        if career_epa_per_play > 0.0:
            return "Serviceable starter"
        return "Backup / streaming option"

    if pos == "WR":
        if career_epa_per_play > 0.12 and current_snap_share > 75 and elite_qualified:
            return "Elite WR1 trajectory"
        if career_epa_per_play > 0.06 and current_snap_share > 60 and starter_qualified:
            return "WR2 with WR1 upside"
        if current_snap_share > 50:
            return "Rotational / emerging role"
        return "Depth / situational role"

    if pos == "RB":
        if current_snap_share > 60 and career_epa_per_play > 0.0 and elite_qualified:
            return "Workhorse back"
        if current_snap_share > 40 and starter_qualified:
            return "Timeshare back with value"
        return "Change-of-pace / specialty role"

    if pos == "TE":
        if career_epa_per_play > 0.10 and current_snap_share > 65 and elite_qualified:
            return "Elite TE1 trajectory"
        if career_epa_per_play > 0.0 and current_snap_share > 50 and starter_qualified:
            return "Receiving TE with upside"
        return "Blocking / depth TE"

    return None


def classify_sustainability(
    success_rate: float,
    usage_volatility: float,
    role_stability: float,
    games: int,
) -> str | None:
    """
    Evaluate production sustainability from 2025-season data.
    Returns None when the sample is too small.
    """
    if games < MIN_GAMES_FOR_LABELS:
        return None
    if success_rate > 52 and usage_volatility < 8 and role_stability > 7:
        return "Sustainable"
    if success_rate > 45 and usage_volatility < 12:
        return "Mostly Sustainable"
    if usage_volatility > 15:
        return "High Variance"
    return "Regression Risk"


# ---------------------------------------------------------------------------
# Per-position metric aggregation (2025 season only)
# ---------------------------------------------------------------------------

def compute_passing_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-QB metrics from 2025 pass plays.

    Sacks are excluded from attempt counts so the numbers match official
    NFL/ESPN box-score statistics (which record sacks separately).
    nflverse marks sacks as pass_attempt==1, so without this filter every
    QB's attempt total would be inflated by their sack count.

    Games are counted from all dropbacks (including sacks) so a game is
    credited even if the QB was sacked on every play.
    """
    all_dropbacks = pbp[pbp["pass_attempt"] == 1].copy()
    if all_dropbacks.empty:
        return pd.DataFrame()

    # Official pass attempts = dropbacks minus sacks
    if "sack" in all_dropbacks.columns:
        official = all_dropbacks[all_dropbacks["sack"] != 1]
    else:
        print("WARNING: 'sack' column not found in PBP; attempt counts will include sacks.", file=sys.stderr)
        official = all_dropbacks

    grouped = official.groupby("passer_player_name").agg(
        attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        total_epa=("epa", "sum"),
        successful_plays=("success", "sum"),
        touchdowns=("touchdown", "sum"),
        air_yards_total=("air_yards", lambda x: x.dropna().sum()),
        deep_targets=("air_yards", lambda x: (x.dropna() >= 20).sum()),
        explosive_plays=("yards_gained", lambda x: (x >= 20).sum()),
        weeks=("week", "nunique"),
    ).reset_index()
    grouped.rename(columns={"passer_player_name": "player_name"}, inplace=True)

    # games counted from all dropbacks so sack-only games still register
    games_all = (
        all_dropbacks.groupby("passer_player_name")["game_id"]
        .nunique()
        .rename("games")
        .reset_index()
        .rename(columns={"passer_player_name": "player_name"})
    )
    grouped = grouped.merge(games_all, on="player_name", how="left")
    grouped["games"] = grouped["games"].fillna(0).astype(int)

    team_last = (
        all_dropbacks.groupby("passer_player_name")["posteam"]
        .last()
        .reset_index()
        .rename(columns={"passer_player_name": "player_name", "posteam": "team"})
    )
    grouped = grouped.merge(team_last, on="player_name", how="left")
    grouped["team"] = grouped["team"].fillna("UNK")
    grouped["pos"] = "QB"
    return grouped


def compute_receiving_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-receiver metrics from 2025 pass plays."""
    targets = pbp[pbp["pass_attempt"] == 1].copy()
    if targets.empty:
        return pd.DataFrame()

    # Team-level denominators (2025 only) for share calculations
    team_air_yards = (
        targets.groupby(["posteam", "game_id"])["air_yards"]
        .sum()
        .reset_index()
        .groupby("posteam")["air_yards"]
        .mean()
        .rename("team_air_yards_per_game")
    )

    team_targets = (
        targets.groupby(["posteam", "game_id"])
        .size()
        .reset_index(name="count")
        .groupby("posteam")["count"]
        .mean()
        .rename("team_targets_per_game")
    )

    rec_plays = targets[targets["receiver_player_name"].notna()].copy()

    grouped = rec_plays.groupby("receiver_player_name").agg(
        attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        total_epa=("epa", "sum"),
        successful_plays=("success", "sum"),
        touchdowns=("touchdown", "sum"),
        air_yards_total=("air_yards", lambda x: x.dropna().sum()),
        deep_targets=("air_yards", lambda x: (x.dropna() >= 20).sum()),
        explosive_plays=("yards_gained", lambda x: (x >= 20).sum()),
        games=("game_id", "nunique"),
        weeks=("week", "nunique"),
        yards_gained_total=("yards_gained", "sum"),
        yac_total=("yards_after_catch", lambda x: x.dropna().sum()),
    ).reset_index()

    grouped.rename(columns={"receiver_player_name": "player_name"}, inplace=True)
    grouped["pos"] = "WR"
    grouped["team"] = rec_plays.groupby("receiver_player_name")["posteam"].last().values

    grouped = grouped.merge(
        team_air_yards.reset_index(), left_on="team", right_on="posteam", how="left"
    ).drop(columns=["posteam"], errors="ignore")

    grouped = grouped.merge(
        team_targets.reset_index(), left_on="team", right_on="posteam", how="left"
    ).drop(columns=["posteam"], errors="ignore")

    return grouped


def compute_rushing_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-rusher metrics from 2025 rush plays."""
    rush_plays = pbp[pbp["rush_attempt"] == 1].copy()
    if rush_plays.empty:
        return pd.DataFrame()

    if "yardline_100" in rush_plays.columns:
        rush_plays["goal_line_carry"] = rush_plays["yardline_100"].apply(
            lambda x: 1 if pd.notna(x) and x <= 5 else 0
        )
    else:
        rush_plays["goal_line_carry"] = 0

    grouped = rush_plays.groupby("rusher_player_name").agg(
        attempts=("rush_attempt", "sum"),
        total_epa=("epa", "sum"),
        successful_plays=("success", "sum"),
        touchdowns=("touchdown", "sum"),
        explosive_plays=("yards_gained", lambda x: (x >= 15).sum()),
        breakaway_runs=("yards_gained", lambda x: (x >= 15).sum()),
        goal_line_carries=("goal_line_carry", "sum"),
        games=("game_id", "nunique"),
        weeks=("week", "nunique"),
        yards_gained_total=("yards_gained", "sum"),
    ).reset_index()

    grouped.rename(columns={"rusher_player_name": "player_name"}, inplace=True)
    grouped["pos"] = "RB"
    grouped["team"] = rush_plays.groupby("rusher_player_name")["posteam"].last().values
    return grouped


# ---------------------------------------------------------------------------
# Route participation (from weekly player stats, 2025 season only)
# ---------------------------------------------------------------------------

def compute_route_participation(
    stats: pd.DataFrame,
    season: int,
) -> dict[str, float | None]:
    """
    Compute route participation rate per player from the nflverse weekly
    player stats dataset.

    Route participation = routes_run / offensive_snaps × 100

    Returns a dict keyed by player_display_name (full name) with float
    values in percent form (e.g. 84.3), or None where snaps are zero.

    If the required columns are missing from the stats DataFrame, logs a
    warning and returns an empty dict so the pipeline never crashes.
    """
    if stats.empty:
        print("WARNING: Player stats DataFrame is empty; routeParticipation will be null.", file=sys.stderr)
        return {}

    route_col = "receiving_routes_run"
    if route_col not in stats.columns:
        print(f"WARNING: '{route_col}' column not found in player stats; routeParticipation will be null.", file=sys.stderr)
        return {}

    snap_col = None
    for candidate in ("offense_snaps", "snap_counts_offense"):
        if candidate in stats.columns:
            snap_col = candidate
            break
    if snap_col is None:
        print("WARNING: No offensive snap column found in player stats; routeParticipation will be null.", file=sys.stderr)
        return {}

    name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    if name_col not in stats.columns:
        print("WARNING: No player name column found in player stats; routeParticipation will be null.", file=sys.stderr)
        return {}

    filtered = stats.copy()
    if "season" in filtered.columns:
        filtered = filtered[filtered["season"] == season]
    if "season_type" in filtered.columns:
        filtered = filtered[filtered["season_type"].isin(["REG", "POST"])]

    filtered = filtered.dropna(subset=[name_col])

    grouped = filtered.groupby(name_col).agg(
        total_routes=(route_col, "sum"),
        total_snaps=(snap_col, "sum"),
    ).reset_index()

    result: dict[str, float | None] = {}
    for _, row in grouped.iterrows():
        name = row[name_col]
        snaps = float(row["total_snaps"])
        routes = float(row["total_routes"])
        if snaps > 0:
            result[name] = round((routes / snaps) * 100, 1)
        else:
            result[name] = None

    print(f"  Route participation computed for {len(result)} players from {name_col}.")
    return result


# ---------------------------------------------------------------------------
# Weekly trend computation (2025 season only)
# ---------------------------------------------------------------------------

def compute_weekly_trends(pbp_2025: pd.DataFrame, player_name: str) -> dict:
    """
    Compute rolling weekly usage trends from 2025 data only.
    All trend values are floats in percent form (e.g. +7.4 means +7.4%).
    Fields that cannot be computed are returned as None.
    """
    mask = pd.Series(False, index=pbp_2025.index)
    for col in ["passer_player_name", "receiver_player_name", "rusher_player_name"]:
        if col in pbp_2025.columns:
            mask |= (pbp_2025[col] == player_name)

    player_plays = pbp_2025[mask]

    if player_plays.empty or "week" not in player_plays.columns:
        return {
            "rollingSnapTrend": None,
            "rollingTargetTrend": None,
            "usageVolatility": None,
            "roleStability": None,
            "routeGrowth": None,
        }

    weekly_plays = player_plays.groupby("week").size().reset_index(name="plays")

    if len(weekly_plays) < 4:
        std_val = float(weekly_plays["plays"].std()) if len(weekly_plays) > 1 else 0.0
        return {
            "rollingSnapTrend": None,
            "rollingTargetTrend": None,
            "usageVolatility": round(std_val, 2),
            "roleStability": None,
            "routeGrowth": None,
        }

    plays_arr = weekly_plays["plays"].values
    n = len(plays_arr)
    last4 = float(plays_arr[max(n - 4, 0):].mean())
    prior4 = float(plays_arr[max(n - 8, 0):max(n - 4, 0)].mean()) if n > 4 else last4

    # Percent change between the two 4-week windows, already in percent form
    snap_trend_pct = round(((last4 - prior4) / (prior4 + 1e-9)) * 100, 2)
    volatility = round(float(np.std(plays_arr)), 2)
    mean_plays = float(np.mean(plays_arr))
    cv = volatility / (mean_plays + 1e-9)
    role_stability = round(max(1.0, min(10.0, 10.0 - cv * 10)), 1)

    return {
        "rollingSnapTrend": snap_trend_pct,
        "rollingTargetTrend": round(snap_trend_pct * 0.8, 2),
        "usageVolatility": volatility,
        "roleStability": role_stability,
        "routeGrowth": round(snap_trend_pct * 0.6, 2),
    }


# ---------------------------------------------------------------------------
# Main metric assembly
# ---------------------------------------------------------------------------

def compute_metrics(
    pbp_all: pd.DataFrame,
    pbp_2025: pd.DataFrame,
    stats: pd.DataFrame,
    sleeper_players: list[dict],
) -> list[dict]:
    """
    Assemble per-player metric objects.

    Dashboard metrics (games, shares, trends) → pbp_2025 only.
    Context labels (careerArc, threeYearContext) → career_stats from pbp_all.
    Route participation → player stats for CURRENT_SEASON.
    """
    player_lookup = build_player_lookup(sleeper_players)
    abbreviated_lookup = build_abbreviated_lookup(sleeper_players)

    print("Computing career context from all seasons...")
    career_stats = compute_career_stats(pbp_all)

    print("Computing route participation from player stats...")
    route_participation = compute_route_participation(stats, CURRENT_SEASON)

    print(f"Computing 2025-season passing metrics...")
    passing = compute_passing_metrics(pbp_2025)

    print(f"Computing 2025-season receiving metrics...")
    receiving = compute_receiving_metrics(pbp_2025)

    print(f"Computing 2025-season rushing metrics...")
    rushing = compute_rushing_metrics(pbp_2025)

    # Team rush attempts from 2025 only (for RB snap share denominator)
    team_rush_attempts = (
        pbp_2025[pbp_2025["rush_attempt"] == 1].groupby("posteam")["rush_attempt"].sum()
    )

    # Red zone context from 2025 only
    rz_col = "yardline_100" in pbp_2025.columns
    if rz_col:
        rz_pbp = pbp_2025[pbp_2025["yardline_100"] <= 20]
        total_rz = len(rz_pbp)
    else:
        rz_pbp = pd.DataFrame()
        total_rz = 0

    all_players: list[dict] = []

    # -----------------------------------------------------------------------
    # Pass catchers (WR / TE / RB receiving role)
    # -----------------------------------------------------------------------
    for _, row in receiving.iterrows():
        name = row["player_name"]  # PBP abbreviated name
        if pd.isna(name):
            continue

        attempts = int(row.get("attempts", 0))
        if attempts < MIN_PLAYS:
            continue

        full_name = resolve_full_name(name, player_lookup, abbreviated_lookup)
        sleeper_info = player_lookup.get(full_name, {})
        team = str(row.get("team", sleeper_info.get("team", "UNK")))
        pos = sleeper_info.get("position", row.get("pos", "WR"))
        age = sleeper_info.get("age")
        games = int(row.get("games", 1))

        # Career context (all seasons)
        ctx = career_stats.get(name, {"career_games": games, "career_epa_per_play": 0.0})
        career_games = ctx["career_games"]
        career_epa_per_play = ctx["career_epa_per_play"]

        # 2025-season efficiency
        total_epa = float(row.get("total_epa", 0))
        successful = float(row.get("successful_plays", 0))
        air_yards = float(row.get("air_yards_total", 0))
        deep_tgts = int(row.get("deep_targets", 0))
        explosive = int(row.get("explosive_plays", 0))
        touchdowns = int(row.get("touchdowns", 0))
        yards_total = float(row.get("yards_gained_total", 0))

        epa_per_play = round(total_epa / attempts, 3) if attempts else 0.0
        success_rate = safe_pct(successful, attempts)
        explosive_rate = safe_pct(explosive, attempts)
        deep_target_rate = safe_pct(deep_tgts, attempts)

        # 2025-season share metrics (all in percent form via safe_pct)
        team_air_yds = float(row.get("team_air_yards_per_game", 0) or 0) * games
        team_tgts = float(row.get("team_targets_per_game", 0) or 0) * games
        air_yards_share = safe_pct(air_yards, team_air_yds) if team_air_yds > 0 else None
        target_share = safe_pct(attempts, team_tgts) if team_tgts > 0 else None

        # WOPR stays on 0-~1.5 index scale (not percent)
        if target_share is not None and air_yards_share is not None:
            wopr = round(1.5 * (target_share / 100) + 0.7 * (air_yards_share / 100), 3)
        else:
            wopr = None

        # Approximate snap share from target share (percent scale)
        snap_share = min(99.9, round((target_share or 0) * 2.8, 1))
        # snapTrend: percent-point change from the target-implied snap baseline
        snap_trend_val = round((snap_share - 70) * 0.3, 2)

        # Red zone utilization (2025 only, percent scale)
        if rz_col and total_rz > 0 and "receiver_player_name" in rz_pbp.columns:
            rz_targets = int((rz_pbp["receiver_player_name"] == name).sum())
            red_zone_util = safe_pct(rz_targets, total_rz)
        else:
            red_zone_util = None

        expected_tds = round(attempts * 0.05, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        actual_fp = yards_total * 0.1 + touchdowns * 6
        fpoe = round(actual_fp * 0.18, 2)

        # Trends from 2025 data only
        trends = compute_weekly_trends(pbp_2025, name)
        usage_vol = trends["usageVolatility"] or 0.0
        role_stab = trends["roleStability"] or 5.0

        # Context labels use career data
        career_arc = classify_career_arc(pos, age, career_epa_per_play, snap_share, career_games, games)
        three_year = classify_three_year_context(pos, career_epa_per_play, snap_share, career_games, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        all_players.append({
            "player": full_name,
            "team": team,
            "pos": pos,
            "age": age,
            "games": games,
            "snapShare": snap_share,
            "targetShare": round(target_share, 1) if target_share is not None else None,
            "airYardsShare": round(air_yards_share, 1) if air_yards_share is not None else None,
            "wopr": wopr,
            "redZoneUtil": round(red_zone_util, 1) if red_zone_util is not None else None,
            "goalLineCarries": 0,
            "snapTrend": snap_trend_val,
            "routeParticipation": route_participation.get(full_name, route_participation.get(name)),
            "threeYearContext": three_year,
            "careerArc": career_arc,
            "epaPlay": epa_per_play,
            "successRate": success_rate,
            "tdOverExpected": td_over_expected,
            "fpoe": fpoe,
            "explosivePlayRate": explosive_rate,
            "breakawayRunRate": 0.0,
            "deepTargetRate": round(deep_target_rate, 1),
            "sustainability": sustainability,
            "rollingSnapTrend": trends["rollingSnapTrend"],
            "rollingTargetTrend": trends["rollingTargetTrend"],
            "usageVolatility": trends["usageVolatility"],
            "roleStability": trends["roleStability"],
            "routeGrowth": trends["routeGrowth"],
        })

    # -----------------------------------------------------------------------
    # Pure rushers (RB / FB)
    # -----------------------------------------------------------------------
    for _, row in rushing.iterrows():
        name = row["player_name"]
        if pd.isna(name):
            continue

        full_name = resolve_full_name(name, player_lookup, abbreviated_lookup)
        if any(p["player"] == full_name for p in all_players):
            continue

        sleeper_info = player_lookup.get(full_name, {})
        # QBs who also rush are processed in the QB loop with their full
        # passing stats — if we let them through here their entire stat card
        # would be built from rushing plays only (e.g. 22 carries instead of
        # 212 pass attempts), producing wildly wrong EPA and success rate.
        if sleeper_info.get("position") == "QB":
            continue

        attempts = int(row.get("attempts", 0))
        if attempts < MIN_PLAYS:
            continue

        team = str(row.get("team", sleeper_info.get("team", "UNK")))
        pos = sleeper_info.get("position", "RB")
        age = sleeper_info.get("age")
        games = int(row.get("games", 1))

        ctx = career_stats.get(name, {"career_games": games, "career_epa_per_play": 0.0})
        career_games = ctx["career_games"]
        career_epa_per_play = ctx["career_epa_per_play"]

        total_epa = float(row.get("total_epa", 0))
        successful = float(row.get("successful_plays", 0))
        explosive = int(row.get("explosive_plays", 0))
        breakaway = int(row.get("breakaway_runs", 0))
        gl_carries = int(row.get("goal_line_carries", 0))
        touchdowns = int(row.get("touchdowns", 0))
        yards_total = float(row.get("yards_gained_total", 0))

        epa_per_play = round(total_epa / attempts, 3) if attempts else 0.0
        success_rate = safe_pct(successful, attempts)
        explosive_rate = safe_pct(explosive, attempts)
        breakaway_rate = safe_pct(breakaway, attempts)

        # Snap share from 2025 team rush attempts (percent scale)
        team_rushes = float(team_rush_attempts.get(team, attempts))
        snap_share = min(99.9, round(safe_pct(attempts, team_rushes) * 1.5, 1))
        snap_trend_val = round((snap_share - 50) * 0.2, 2)

        expected_tds = round(attempts * 0.04, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        actual_fp = yards_total * 0.1 + touchdowns * 6
        fpoe = round(actual_fp * 0.18, 2)

        trends = compute_weekly_trends(pbp_2025, name)
        usage_vol = trends["usageVolatility"] or 0.0
        role_stab = trends["roleStability"] or 5.0

        career_arc = classify_career_arc(pos, age, career_epa_per_play, snap_share, career_games, games)
        three_year = classify_three_year_context(pos, career_epa_per_play, snap_share, career_games, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        all_players.append({
            "player": full_name,
            "team": team,
            "pos": pos,
            "age": age,
            "games": games,
            "snapShare": snap_share,
            # Receiver opportunity metrics are null for pure rushers
            "targetShare": None,
            "airYardsShare": None,
            "wopr": None,
            "redZoneUtil": None,
            "goalLineCarries": gl_carries,
            "snapTrend": snap_trend_val,
            "routeParticipation": None,
            "threeYearContext": three_year,
            "careerArc": career_arc,
            "epaPlay": epa_per_play,
            "successRate": success_rate,
            "tdOverExpected": td_over_expected,
            "fpoe": fpoe,
            "explosivePlayRate": explosive_rate,
            "breakawayRunRate": round(breakaway_rate, 1),
            "deepTargetRate": None,
            "sustainability": sustainability,
            "rollingSnapTrend": trends["rollingSnapTrend"],
            "rollingTargetTrend": None,   # not applicable for pure rushers
            "usageVolatility": trends["usageVolatility"],
            "roleStability": trends["roleStability"],
            "routeGrowth": None,          # not applicable for pure rushers
        })

    # -----------------------------------------------------------------------
    # Quarterbacks
    # -----------------------------------------------------------------------
    for _, row in passing.iterrows():
        name = row["player_name"]
        if pd.isna(name):
            continue

        full_name = resolve_full_name(name, player_lookup, abbreviated_lookup)
        if any(p["player"] == full_name for p in all_players):
            continue

        attempts = int(row.get("attempts", 0))
        if attempts < MIN_PLAYS:
            continue

        sleeper_info = player_lookup.get(full_name, {})
        team = str(row.get("team", sleeper_info.get("team", "UNK")))
        pos = "QB"
        age = sleeper_info.get("age")
        games = int(row.get("games", 1))

        ctx = career_stats.get(name, {"career_games": games, "career_epa_per_play": 0.0})
        career_games = ctx["career_games"]
        career_epa_per_play = ctx["career_epa_per_play"]

        total_epa = float(row.get("total_epa", 0))
        successful = float(row.get("successful_plays", 0))
        explosive = int(row.get("explosive_plays", 0))
        deep_tgts = int(row.get("deep_targets", 0))
        touchdowns = int(row.get("touchdowns", 0))

        epa_per_play = round(total_epa / attempts, 3) if attempts else 0.0
        success_rate = safe_pct(successful, attempts)
        explosive_rate = safe_pct(explosive, attempts)
        # deepTargetRate for QBs = their deep throw rate (meaningful, in percent form)
        deep_target_rate = safe_pct(deep_tgts, attempts)

        expected_tds = round(attempts * 0.055, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        snap_share = 95.0   # QBs are on the field every offensive snap
        snap_trend_val = 0.0

        trends = compute_weekly_trends(pbp_2025, name)
        usage_vol = trends["usageVolatility"] or 0.0
        role_stab = trends["roleStability"] or 9.0

        career_arc = classify_career_arc(pos, age, career_epa_per_play, snap_share, career_games, games)
        three_year = classify_three_year_context(pos, career_epa_per_play, snap_share, career_games, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        all_players.append({
            "player": full_name,
            "team": team,
            "pos": pos,
            "age": age,
            "games": games,
            "snapShare": snap_share,
            # Receiver opportunity metrics are not applicable for QBs
            "targetShare": None,
            "airYardsShare": None,
            "wopr": None,
            "redZoneUtil": None,
            "goalLineCarries": 0,
            "snapTrend": snap_trend_val,
            "routeParticipation": None,
            "threeYearContext": three_year,
            "careerArc": career_arc,
            "epaPlay": epa_per_play,
            "successRate": success_rate,
            "tdOverExpected": td_over_expected,
            "fpoe": round(epa_per_play * attempts * 0.3, 2),
            "explosivePlayRate": explosive_rate,
            "breakawayRunRate": None,     # not applicable for QBs
            "deepTargetRate": round(deep_target_rate, 1),
            "sustainability": sustainability,
            "rollingSnapTrend": trends["rollingSnapTrend"],
            "rollingTargetTrend": None,   # not applicable for QBs
            "usageVolatility": trends["usageVolatility"],
            "roleStability": trends["roleStability"],
            "routeGrowth": None,          # not applicable for QBs
        })

    print(f"Total players processed: {len(all_players)}")
    return all_players


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_metrics(players: list[dict]) -> None:
    """Save computed metrics to JSON."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(players, f, indent=2)
    print(f"Saved metrics for {len(players)} players to {OUTPUT_PATH}")


def main():
    pbp_all, pbp_2025, stats, sleeper_players = load_data()
    players = compute_metrics(pbp_all, pbp_2025, stats, sleeper_players)
    save_metrics(players)
    print("Metric computation complete.")


if __name__ == "__main__":
    main()
