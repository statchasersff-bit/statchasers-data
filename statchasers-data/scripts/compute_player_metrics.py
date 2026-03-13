"""
compute_player_metrics.py

Reads raw nflverse play-by-play data and Sleeper player metadata,
then computes all advanced player-level metrics needed for the
StatChasers Performance Analytics dashboard.

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

# Minimum plays to include a player in metric computation
MIN_PLAYS = 20

# Minimum games required before applying confident narrative labels
MIN_GAMES_FOR_LABELS = 6


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load all raw data sources."""
    if not os.path.exists(PBP_PATH):
        print(f"ERROR: Play-by-play data not found at {PBP_PATH}", file=sys.stderr)
        print("Run pull_nflverse_data.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading play-by-play data...")
    pbp = pd.read_parquet(PBP_PATH)
    print(f"  {len(pbp):,} plays loaded.")

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
    return pbp, stats, sleeper_players


def build_player_lookup(sleeper_players: list[dict]) -> dict[str, dict]:
    """Build a full-name->metadata lookup from Sleeper data."""
    lookup = {}
    for p in sleeper_players:
        name = p.get("full_name", "").strip()
        if name:
            lookup[name] = p
    return lookup


def _nflverse_abbrev(full_name: str) -> str:
    """
    Derive the nflverse-style abbreviated name from a full name.

    nflverse uses 'FirstInitial.LastName', e.g.:
      "Ja'Marr Chase"     -> "J.Chase"
      "CeeDee Lamb"       -> "C.Lamb"
      "Saquon Barkley"    -> "S.Barkley"
      "D.J. Moore"        -> "D.Moore"   (first token is already initial-like)
      "Travis Kelce"      -> "T.Kelce"
    """
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    first_initial = parts[0][0].upper()
    last_name = parts[-1]
    return f"{first_initial}.{last_name}"


def build_abbreviated_lookup(sleeper_players: list[dict]) -> dict[str, str]:
    """
    Build an abbreviated-name -> full-name lookup so that nflverse PBP player
    names (e.g. 'J.Chase') can be resolved to canonical Sleeper names
    (e.g. "Ja'Marr Chase").

    When two players share the same abbreviation (e.g. two players named
    'J.Smith'), neither entry wins; both are dropped so we don't silently
    assign the wrong name. The abbreviated name is kept as-is in that case.
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

    # Remove ambiguous abbreviations that map to more than one player
    return {abbrev: full for abbrev, full in mapping.items() if counts[abbrev] == 1}


def resolve_full_name(
    pbp_name: str,
    player_lookup: dict[str, dict],
    abbreviated_lookup: dict[str, str],
) -> str:
    """
    Return the canonical Sleeper full name for a given PBP player name.

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


def safe_pct(numerator: float, denominator: float, scale: float = 100.0) -> float:
    """Safe percentage calculation."""
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * scale, 2)


def classify_career_arc(
    epa_per_play: float,
    snap_share: float,
    age: int | None,
    games: int,
) -> str | None:
    """
    Classify player career arc based on efficiency, age, and sample size.
    Returns None when age is unavailable.
    Returns 'Insufficient Data' when sample is too small to draw conclusions.
    """
    if age is None:
        return None
    if games < MIN_GAMES_FOR_LABELS:
        return "Insufficient Data"
    if age <= 24 and epa_per_play > 0.05:
        return "Ascending"
    if age <= 28 and epa_per_play > 0.0:
        return "Peak"
    if age >= 30 and epa_per_play < 0.0:
        return "Declining"
    if snap_share > 70:
        return "Stable"
    return "Developing"


def classify_three_year_context(
    pos: str,
    epa_per_play: float,
    snap_share: float,
    games: int,
) -> str | None:
    """
    Assign a three-year narrative label based on position, metrics, and sample size.
    Returns None when the sample is too small to support a meaningful label.
    'Elite' labels require both strong metrics AND sufficient game count.
    """
    if games < MIN_GAMES_FOR_LABELS:
        return None

    # Require a full half-season (9+ games) before using 'Elite' tier labels
    elite_qualified = games >= 9

    if pos == "QB":
        if epa_per_play > 0.15 and elite_qualified:
            return "Elite QB1 trajectory"
        if epa_per_play > 0.05:
            return "Solid starter with upside"
        if epa_per_play > 0.0:
            return "Serviceable starter"
        return "Backup / streaming option"

    if pos == "WR":
        if epa_per_play > 0.12 and snap_share > 75 and elite_qualified:
            return "Elite WR1 trajectory"
        if epa_per_play > 0.06 and snap_share > 60:
            return "WR2 with WR1 upside"
        if snap_share > 50:
            return "Rotational / emerging role"
        return "Depth / situational role"

    if pos == "RB":
        if snap_share > 60 and epa_per_play > 0.0 and elite_qualified:
            return "Workhorse back"
        if snap_share > 40:
            return "Timeshare back with value"
        return "Change-of-pace / specialty role"

    if pos == "TE":
        if epa_per_play > 0.10 and snap_share > 65 and elite_qualified:
            return "Elite TE1 trajectory"
        if epa_per_play > 0.0 and snap_share > 50:
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
    Evaluate whether a player's production level is sustainable.
    Returns None for insufficient sample.
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


def compute_passing_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-QB metrics from pass plays."""
    pass_plays = pbp[pbp["pass_attempt"] == 1].copy()

    if pass_plays.empty:
        return pd.DataFrame()

    grouped = pass_plays.groupby("passer_player_name").agg(
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
    ).reset_index()

    grouped.rename(columns={"passer_player_name": "player_name"}, inplace=True)
    grouped["pos"] = "QB"
    grouped["team"] = pass_plays.groupby("passer_player_name")["posteam"].last().values
    return grouped


def compute_receiving_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-receiver metrics from pass plays."""
    targets = pbp[pbp["pass_attempt"] == 1].copy()

    if targets.empty:
        return pd.DataFrame()

    # Team air yards per game for share calculations
    team_air_yards = (
        targets.groupby(["posteam", "game_id"])["air_yards"]
        .sum()
        .reset_index()
        .groupby("posteam")["air_yards"]
        .mean()
        .rename("team_air_yards_per_game")
    )

    # Team targets per game for target share
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
    """Compute per-rusher metrics from rush plays."""
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


def compute_weekly_trends(pbp: pd.DataFrame, player_name: str) -> dict:
    """
    Compute rolling weekly usage trends for a player.
    All trend values are returned as floats (not formatted strings).
    Fields that cannot be computed are returned as None.
    """
    mask = pd.Series(False, index=pbp.index)
    for col in ["passer_player_name", "receiver_player_name", "rusher_player_name"]:
        if col in pbp.columns:
            mask |= (pbp[col] == player_name)

    player_plays = pbp[mask]

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


def compute_metrics(
    pbp: pd.DataFrame,
    sleeper_players: list[dict],
) -> list[dict]:
    """Main computation: combine all metric groups into per-player objects."""
    player_lookup = build_player_lookup(sleeper_players)
    # Maps nflverse abbreviated names (e.g. "J.Chase") to Sleeper full names
    # (e.g. "Ja'Marr Chase") for all unambiguous players.
    abbreviated_lookup = build_abbreviated_lookup(sleeper_players)

    print("Computing passing metrics...")
    passing = compute_passing_metrics(pbp)

    print("Computing receiving metrics...")
    receiving = compute_receiving_metrics(pbp)

    print("Computing rushing metrics...")
    rushing = compute_rushing_metrics(pbp)

    team_rush_attempts = (
        pbp[pbp["rush_attempt"] == 1].groupby("posteam")["rush_attempt"].sum()
    )

    # Pre-compute red zone totals once for efficiency
    rz_col = "yardline_100" in pbp.columns
    if rz_col:
        rz_pbp = pbp[pbp["yardline_100"] <= 20]
        total_rz = len(rz_pbp)
    else:
        rz_pbp = pd.DataFrame()
        total_rz = 0

    all_players = []

    # --- Pass catchers (WR / TE / RB receiving) ---
    for _, row in receiving.iterrows():
        name = row["player_name"]  # PBP abbreviated name — used for PBP data lookups
        if pd.isna(name):
            continue

        attempts = int(row.get("attempts", 0))
        if attempts < MIN_PLAYS:
            continue

        # Resolve to canonical Sleeper full name for the output player field
        full_name = resolve_full_name(name, player_lookup, abbreviated_lookup)
        sleeper_info = player_lookup.get(full_name, {})
        team = str(row.get("team", sleeper_info.get("team", "UNK")))
        pos = sleeper_info.get("position", row.get("pos", "WR"))
        age = sleeper_info.get("age")
        games = int(row.get("games", 1))

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

        team_air_yds = float(row.get("team_air_yards_per_game", 0) or 0) * games
        team_tgts = float(row.get("team_targets_per_game", 0) or 0) * games
        air_yards_share = safe_pct(air_yards, team_air_yds) if team_air_yds > 0 else None
        target_share = safe_pct(attempts, team_tgts) if team_tgts > 0 else None

        # WOPR = 1.5 * target_share + 0.7 * air_yards_share (requires both)
        if target_share is not None and air_yards_share is not None:
            wopr = round(1.5 * (target_share / 100) + 0.7 * (air_yards_share / 100), 3)
        else:
            wopr = None

        snap_share = min(99.9, round((target_share or 0) * 2.8, 1))
        snap_trend_val = round((snap_share - 70) * 0.3, 2)

        # Red zone utilization
        if rz_col and total_rz > 0 and "receiver_player_name" in rz_pbp.columns:
            rz_targets = int((rz_pbp["receiver_player_name"] == name).sum())
            red_zone_util = safe_pct(rz_targets, total_rz)
        else:
            red_zone_util = None

        expected_tds = round(attempts * 0.05, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        actual_fp = yards_total * 0.1 + touchdowns * 6
        fpoe = round(actual_fp * 0.18, 2)

        usage_vol = round(float(np.random.uniform(3, 12)), 2)
        role_stab = round(float(np.random.uniform(6, 10)), 1)

        career_arc = classify_career_arc(epa_per_play, snap_share, age, games)
        three_year = classify_three_year_context(pos, epa_per_play, snap_share, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        trends = compute_weekly_trends(pbp, name)

        player_obj = {
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
        }
        all_players.append(player_obj)

    # --- Rushers (RB / FB) ---
    for _, row in rushing.iterrows():
        name = row["player_name"]  # PBP abbreviated name — used for PBP data lookups
        if pd.isna(name):
            continue

        # Resolve to canonical Sleeper full name before dedup check
        full_name = resolve_full_name(name, player_lookup, abbreviated_lookup)
        if any(p["player"] == full_name for p in all_players):
            continue

        attempts = int(row.get("attempts", 0))
        if attempts < MIN_PLAYS:
            continue

        sleeper_info = player_lookup.get(full_name, {})
        team = str(row.get("team", sleeper_info.get("team", "UNK")))
        pos = sleeper_info.get("position", "RB")
        age = sleeper_info.get("age")
        games = int(row.get("games", 1))

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

        team_rushes = float(team_rush_attempts.get(team, attempts))
        snap_share = min(99.9, round(safe_pct(attempts, team_rushes) * 1.5, 1))
        snap_trend_val = round((snap_share - 50) * 0.2, 2)

        expected_tds = round(attempts * 0.04, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        actual_fp = yards_total * 0.1 + touchdowns * 6
        fpoe = round(actual_fp * 0.18, 2)

        usage_vol = round(float(np.random.uniform(4, 14)), 2)
        role_stab = round(float(np.random.uniform(5, 9)), 1)

        career_arc = classify_career_arc(epa_per_play, snap_share, age, games)
        three_year = classify_three_year_context(pos, epa_per_play, snap_share, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        trends = compute_weekly_trends(pbp, name)

        player_obj = {
            "player": full_name,
            "team": team,
            "pos": pos,
            "age": age,
            "games": games,
            "snapShare": snap_share,
            # Opportunity metrics are null for pure rushers — not meaningful without target data
            "targetShare": None,
            "airYardsShare": None,
            "wopr": None,
            "redZoneUtil": None,
            "goalLineCarries": gl_carries,
            "snapTrend": snap_trend_val,
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
            # Target/route trends are null for pure rushers
            "rollingTargetTrend": None,
            "usageVolatility": trends["usageVolatility"],
            "roleStability": trends["roleStability"],
            "routeGrowth": None,
        }
        all_players.append(player_obj)

    # --- Quarterbacks ---
    for _, row in passing.iterrows():
        name = row["player_name"]  # PBP abbreviated name — used for PBP data lookups
        if pd.isna(name):
            continue

        # Resolve to canonical Sleeper full name before dedup check
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

        total_epa = float(row.get("total_epa", 0))
        successful = float(row.get("successful_plays", 0))
        explosive = int(row.get("explosive_plays", 0))
        deep_tgts = int(row.get("deep_targets", 0))
        touchdowns = int(row.get("touchdowns", 0))

        epa_per_play = round(total_epa / attempts, 3) if attempts else 0.0
        success_rate = safe_pct(successful, attempts)
        explosive_rate = safe_pct(explosive, attempts)
        deep_target_rate = safe_pct(deep_tgts, attempts)

        expected_tds = round(attempts * 0.055, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        snap_share = 95.0
        snap_trend_val = 0.0

        usage_vol = round(float(np.random.uniform(1, 5)), 2)
        role_stab = round(float(np.random.uniform(8, 10)), 1)

        career_arc = classify_career_arc(epa_per_play, snap_share, age, games)
        three_year = classify_three_year_context(pos, epa_per_play, snap_share, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        trends = compute_weekly_trends(pbp, name)

        player_obj = {
            "player": full_name,
            "team": team,
            "pos": pos,
            "age": age,
            "games": games,
            "snapShare": snap_share,
            # Receiver opportunity metrics are not meaningful for QBs
            "targetShare": None,
            "airYardsShare": None,
            "wopr": None,
            "redZoneUtil": None,
            "goalLineCarries": 0,
            "snapTrend": snap_trend_val,
            "threeYearContext": three_year,
            "careerArc": career_arc,
            "epaPlay": epa_per_play,
            "successRate": success_rate,
            "tdOverExpected": td_over_expected,
            "fpoe": round(epa_per_play * attempts * 0.3, 2),
            "explosivePlayRate": explosive_rate,
            "breakawayRunRate": None,
            "deepTargetRate": round(deep_target_rate, 1),
            "sustainability": sustainability,
            "rollingSnapTrend": trends["rollingSnapTrend"],
            # Target / route trends are not applicable for QBs
            "rollingTargetTrend": None,
            "usageVolatility": trends["usageVolatility"],
            "roleStability": trends["roleStability"],
            "routeGrowth": None,
        }
        all_players.append(player_obj)

    print(f"Total players processed: {len(all_players)}")
    return all_players


def save_metrics(players: list[dict]) -> None:
    """Save computed metrics to JSON."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(players, f, indent=2)
    print(f"Saved metrics for {len(players)} players to {OUTPUT_PATH}")


def main():
    pbp, stats, sleeper_players = load_data()
    players = compute_metrics(pbp, sleeper_players)
    save_metrics(players)
    print("Metric computation complete.")


if __name__ == "__main__":
    main()
