"""
build_performance_analytics.py

Reads computed player metrics and assembles the final output JSON files
consumed by the StatChasers Performance Analytics frontend dashboard.

Input:  data/processed/player_metrics.json

Output: output/performance_analytics_latest.json
        output/performance_analytics_2025.json
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

METRICS_PATH = os.path.join(PROCESSED_DIR, "player_metrics.json")
LATEST_OUTPUT = os.path.join(OUTPUT_DIR, "performance_analytics_latest.json")
SEASON_OUTPUT = os.path.join(OUTPUT_DIR, "performance_analytics_2025.json")

CURRENT_SEASON = 2025

# Fields required to be non-null before a player is exported
REQUIRED_FIELDS = ["player", "team", "pos", "games"]

# Minimum games to appear in the output
MIN_GAMES = 2

# Valid positions for export
VALID_POSITIONS = {"QB", "RB", "WR", "TE"}

# String sentinel values that should be coerced to null in output
NULL_SENTINELS = {"N/A", "n/a", "Unknown", "unknown", ""}

# Numeric fields that must always be float | null — never a string
NUMERIC_FIELDS = [
    "snapShare", "targetShare", "airYardsShare", "wopr",
    "redZoneUtil", "routeParticipation",
    "epaPlay", "successRate", "tdOverExpected",
    "fpoe", "explosivePlayRate", "breakawayRunRate", "deepTargetRate",
    "usageVolatility", "roleStability",
    "snapTrend", "rollingSnapTrend", "rollingTargetTrend", "routeGrowth",
    # QB-specific advanced and efficiency fields
    "dropbacksPerGame", "rushAttPerGame", "deepAttemptRate",
    "ypa", "airYardsPerAtt",
    "playActionRate", "scrambleRate", "designedRushRate",
    # QB composite scores
    "relativeEfficiency", "efficiencyScore", "opportunityScore",
    "usageScore", "playerScore", "seasonAvgEpa",
]

# Integer fields
INTEGER_FIELDS = [
    "games", "goalLineCarries",
    "rzAtt", "rushAtt", "rushYds", "rushTd",
    "careerGames",
]

# String label fields — null is allowed, but string sentinels must be cleaned
LABEL_FIELDS = ["threeYearContext", "careerArc", "sustainability", "experienceTier"]


def load_metrics() -> list[dict]:
    """Load computed player metrics."""
    if not os.path.exists(METRICS_PATH):
        print(f"ERROR: Metrics file not found at {METRICS_PATH}", file=sys.stderr)
        print("Run compute_player_metrics.py first.", file=sys.stderr)
        sys.exit(1)

    with open(METRICS_PATH) as f:
        players = json.load(f)

    print(f"Loaded {len(players)} player records.")
    return players


def validate_player(player: dict) -> bool:
    """Check that a player record has all required fields and meets minimum thresholds."""
    for field in REQUIRED_FIELDS:
        if field not in player or player[field] is None:
            return False
    if player.get("pos") not in VALID_POSITIONS:
        return False
    try:
        if int(player["games"]) < MIN_GAMES:
            return False
    except (ValueError, TypeError):
        return False
    return True


def coerce_numeric(value, field: str) -> float | None:
    """
    Coerce a value to float, or None if it cannot be converted or is a sentinel.
    Also returns None for NaN and Inf values which are invalid in JSON.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value in NULL_SENTINELS:
            return None
        # Strip trailing % signs from any legacy string trend values
        stripped = value.strip().rstrip("%")
        try:
            result = float(stripped)
        except ValueError:
            return None
    else:
        try:
            result = float(value)
        except (ValueError, TypeError):
            return None

    # Reject non-finite floats — they are invalid in JSON
    if not math.isfinite(result):
        return None

    return round(result, 3)


def coerce_label(value) -> str | None:
    """Coerce a label field to string | None, mapping sentinel strings to null."""
    if value is None:
        return None
    if isinstance(value, str) and value in NULL_SENTINELS:
        return None
    return value if isinstance(value, str) else None


def clean_player(player: dict) -> dict:
    """
    Normalize a player record for frontend-safe output:
    - Numeric fields → float | null (never string, never NaN/Inf)
    - Integer fields → int | null
    - Label fields → string | null (sentinel strings → null)
    - age → int | null (null is valid; frontend should render as em dash)
    - Position-inappropriate nulls are preserved as null (not coerced to 0.0)
    """
    cleaned = dict(player)

    # Age: keep null as null; coerce to int if present
    age = cleaned.get("age")
    if age is None:
        cleaned["age"] = None
    else:
        try:
            cleaned["age"] = int(age)
        except (ValueError, TypeError):
            cleaned["age"] = None

    # Numeric fields
    for field in NUMERIC_FIELDS:
        cleaned[field] = coerce_numeric(cleaned.get(field), field)

    # Integer fields
    for field in INTEGER_FIELDS:
        val = cleaned.get(field)
        if val is None:
            cleaned[field] = None
        else:
            try:
                cleaned[field] = int(val)
            except (ValueError, TypeError):
                cleaned[field] = None

    # Label fields
    for field in LABEL_FIELDS:
        cleaned[field] = coerce_label(cleaned.get(field))

    # Ensure player / team / pos are clean strings
    cleaned["player"] = str(cleaned.get("player", "")).strip() or None
    cleaned["team"] = str(cleaned.get("team", "")).strip() or None
    cleaned["pos"] = str(cleaned.get("pos", "")).strip() or None

    return cleaned


def _sort_key(player: dict):
    """
    Position-aware sort key for consistent, intentional ordering:
      1. Position group: QB → WR → RB → TE
      2. Within each group, rank by the most relevant opportunity/efficiency signal:
         - QB:    EPA/play (higher = better)
         - WR/TE: WOPR (higher = better), then target share
         - RB:    snap share (higher = better), then EPA/play
    """
    pos_order = {"QB": 0, "WR": 1, "RB": 2, "TE": 3}
    pos = player.get("pos", "")
    pos_rank = pos_order.get(pos, 99)

    epa = player.get("epaPlay") or 0.0
    wopr = player.get("wopr") or 0.0
    target_share = player.get("targetShare") or 0.0
    snap_share = player.get("snapShare") or 0.0

    if pos == "QB":
        secondary = -epa
    elif pos in ("WR", "TE"):
        secondary = (-wopr, -target_share)
    elif pos == "RB":
        secondary = (-snap_share, -epa)
    else:
        secondary = (-epa,)

    # Flatten secondary into a tuple for consistent comparison
    if not isinstance(secondary, tuple):
        secondary = (secondary,)

    return (pos_rank,) + secondary


def sort_players(players: list[dict]) -> list[dict]:
    """Sort players by position group then by strongest usage/efficiency profile."""
    return sorted(players, key=_sort_key)


def build_output(players: list[dict]) -> dict:
    """Wrap the player array with a frontend-ready metadata envelope."""
    now = datetime.now(timezone.utc)
    iso_now = now.isoformat()

    # Pull the season-average EPA baseline from any QB record that has it.
    # This value is used by the frontend to annotate the relativeEfficiency tooltip.
    qb_season_avg_epa: float | None = None
    for p in players:
        if p.get("pos") == "QB" and p.get("seasonAvgEpa") is not None:
            try:
                qb_season_avg_epa = float(p["seasonAvgEpa"])
            except (ValueError, TypeError):
                pass
            break

    return {
        "meta": {
            "season": CURRENT_SEASON,
            "playerCount": len(players),
            "updatedAt": iso_now,
            "generatedAt": iso_now,
            "source": "StatChasers Data Pipeline",
            "qbSeasonAvgEpa": qb_season_avg_epa,
        },
        "players": players,
    }


def write_output(data: dict, path: str) -> None:
    """Write compact JSON output to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(path) / 1024
    print(f"Wrote {path} ({size_kb:.1f} KB, {data['meta']['playerCount']} players)")


def main():
    raw_players = load_metrics()

    valid_players = []
    skipped = 0
    for player in raw_players:
        if validate_player(player):
            valid_players.append(clean_player(player))
        else:
            skipped += 1

    print(f"Valid players: {len(valid_players)} | Skipped: {skipped}")

    sorted_players = sort_players(valid_players)
    output = build_output(sorted_players)

    write_output(output, LATEST_OUTPUT)
    write_output(output, SEASON_OUTPUT)

    print("Performance analytics build complete.")
    print("\nFrontend can fetch from:")
    print("  https://raw.githubusercontent.com/<your-org>/statchasers-data/main/output/performance_analytics_latest.json")


if __name__ == "__main__":
    main()
