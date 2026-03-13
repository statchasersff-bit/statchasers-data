"""
pull_sleeper_players.py

Fetches player metadata from the Sleeper API and saves it as raw JSON.
Endpoint: https://api.sleeper.app/v1/players/nfl

Output: data/raw/sleeper_players.json
"""

import json
import os
import sys
import requests
from datetime import date
from dateutil.relativedelta import relativedelta

SLEEPER_API_URL = "https://api.sleeper.app/v1/players/nfl"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sleeper_players.json")

RELEVANT_POSITIONS = {"QB", "RB", "WR", "TE", "K"}


def compute_age(birth_date_str: str) -> int | None:
    """Compute current age from a birth date string (YYYY-MM-DD)."""
    if not birth_date_str:
        return None
    try:
        birth_date = date.fromisoformat(birth_date_str)
        today = date.today()
        return relativedelta(today, birth_date).years
    except (ValueError, TypeError):
        return None


def fetch_sleeper_players() -> dict:
    """Fetch all NFL players from Sleeper API."""
    print("Fetching player data from Sleeper API...")
    try:
        response = requests.get(SLEEPER_API_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("ERROR: Request to Sleeper API timed out.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error from Sleeper API: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to reach Sleeper API: {e}", file=sys.stderr)
        sys.exit(1)


def process_players(raw_players: dict) -> list[dict]:
    """Extract and clean relevant fields from raw Sleeper player data."""
    players = []
    for player_id, player_data in raw_players.items():
        position = player_data.get("position")

        # Only include skill position players
        if position not in RELEVANT_POSITIONS:
            continue

        full_name = player_data.get("full_name") or (
            f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip()
        )
        if not full_name.strip():
            continue

        birth_date = player_data.get("birth_date")
        age = compute_age(birth_date)
        team = player_data.get("team")

        players.append({
            "player_id": player_id,
            "full_name": full_name,
            "team": team,
            "position": position,
            "birth_date": birth_date,
            "age": age,
        })

    active = sum(1 for p in players if p["team"])
    print(f"Processed {len(players)} skill position players ({active} with active team).")
    return players


def save_players(players: list[dict]) -> None:
    """Save processed player list to JSON file."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(players, f, indent=2)
    print(f"Saved player data to {OUTPUT_PATH}")


def main():
    raw_players = fetch_sleeper_players()
    players = process_players(raw_players)
    save_players(players)
    print("Sleeper player pull complete.")


if __name__ == "__main__":
    main()
