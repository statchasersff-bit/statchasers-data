"""
build_performance_analytics.py

Reads computed player metrics and assembles the final output JSON files
consumed by the StatChasers Performance Analytics frontend dashboard.

Input:  data/processed/player_metrics.json

Output: output/performance_analytics_latest.json
        output/performance_analytics_2025.json
"""

import json
import os
import sys
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

METRICS_PATH = os.path.join(PROCESSED_DIR, "player_metrics.json")
LATEST_OUTPUT = os.path.join(OUTPUT_DIR, "performance_analytics_latest.json")
SEASON_OUTPUT = os.path.join(OUTPUT_DIR, "performance_analytics_2025.json")

# Current season identifier
CURRENT_SEASON = 2025

# Fields that must be present and non-null for a player to be exported
REQUIRED_FIELDS = ["player", "team", "pos", "games"]

# Minimum games to appear in the output
MIN_GAMES = 2

# Positions to include in the export
VALID_POSITIONS = {"QB", "RB", "WR", "TE"}


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
    """Check that a player record has all required fields."""
    for field in REQUIRED_FIELDS:
        if field not in player or player[field] is None:
            return False
    if player.get("pos") not in VALID_POSITIONS:
        return False
    if int(player.get("games", 0)) < MIN_GAMES:
        return False
    return True


def clean_player(player: dict) -> dict:
    """Ensure all expected fields are present with correct types."""
    defaults = {
        "player": "Unknown",
        "team": "UNK",
        "pos": "UNK",
        "age": None,
        "games": 0,
        "snapShare": 0.0,
        "targetShare": 0.0,
        "airYardsShare": 0.0,
        "wopr": 0.0,
        "redZoneUtil": 0.0,
        "goalLineCarries": 0,
        "snapTrend": "+0.0%",
        "threeYearContext": "Unknown",
        "careerArc": "Unknown",
        "epaPlay": 0.0,
        "successRate": 0.0,
        "tdOverExpected": 0.0,
        "fpoe": 0.0,
        "explosivePlayRate": 0.0,
        "breakawayRunRate": 0.0,
        "deepTargetRate": 0.0,
        "sustainability": "Unknown",
        "rollingSnapTrend": "+0.0%",
        "rollingTargetTrend": "+0.0%",
        "usageVolatility": 0.0,
        "roleStability": 0.0,
        "routeGrowth": "+0.0%",
    }
    cleaned = {**defaults, **player}

    # Ensure numeric types
    for field in ["snapShare", "targetShare", "airYardsShare", "wopr",
                  "redZoneUtil", "epaPlay", "successRate", "tdOverExpected",
                  "fpoe", "explosivePlayRate", "breakawayRunRate",
                  "deepTargetRate", "usageVolatility", "roleStability"]:
        try:
            cleaned[field] = round(float(cleaned[field] or 0), 3)
        except (ValueError, TypeError):
            cleaned[field] = 0.0

    for field in ["games", "goalLineCarries"]:
        try:
            cleaned[field] = int(cleaned[field] or 0)
        except (ValueError, TypeError):
            cleaned[field] = 0

    return cleaned


def sort_players(players: list[dict]) -> list[dict]:
    """Sort players by position priority, then by EPA/play descending."""
    pos_order = {"QB": 0, "WR": 1, "RB": 2, "TE": 3}
    return sorted(
        players,
        key=lambda p: (pos_order.get(p.get("pos", ""), 99), -(p.get("epaPlay") or 0))
    )


def build_output(players: list[dict]) -> dict:
    """Wrap player array with metadata envelope."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "meta": {
            "generated_at": now,
            "season": CURRENT_SEASON,
            "player_count": len(players),
            "source": "StatChasers Data Pipeline",
        },
        "players": players,
    }


def write_output(data: dict, path: str) -> None:
    """Write JSON output to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))  # Compact JSON for web delivery
    size_kb = os.path.getsize(path) / 1024
    print(f"Wrote {path} ({size_kb:.1f} KB, {data['meta']['player_count']} players)")


def main():
    raw_players = load_metrics()

    # Validate and clean each record
    valid_players = []
    skipped = 0
    for player in raw_players:
        if validate_player(player):
            valid_players.append(clean_player(player))
        else:
            skipped += 1

    print(f"Valid players: {len(valid_players)} | Skipped: {skipped}")

    # Sort for consistent ordering
    sorted_players = sort_players(valid_players)

    # Build output payload
    output = build_output(sorted_players)

    # Write both output files
    write_output(output, LATEST_OUTPUT)
    write_output(output, SEASON_OUTPUT)

    print("Performance analytics build complete.")
    print(f"\nFrontend can fetch from:")
    print(f"  https://raw.githubusercontent.com/<your-org>/statchasers-data/main/output/performance_analytics_latest.json")


if __name__ == "__main__":
    main()
