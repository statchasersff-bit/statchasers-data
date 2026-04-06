"""
build_qb_trends.py
──────────────────
Builds the QB Role & Usage Trends dataset for the StatChasers frontend.

For each qualifying QB, computes full-sample usage stats and a
recent-vs-prior delta to show whether dropback or rush-attempt volume
is trending up or down.

  recent window  = last 4 games in the sample (chronological)
  prior window   = all earlier games before those last 4
  delta          = recent_per_game − prior_per_game
                   (null when fewer than 6 total games)

Data sources (pre-pulled by pull_nflverse_data.py / pull_sleeper_players.py):
  data/raw/nflverse_play_by_play.parquet
  data/raw/sleeper_players.json
  data/processed/player_metrics.json

Output:
  output/qb_trends.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
PBP_PATH      = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
SLEEPER_PATH  = ROOT / "data" / "raw"       / "sleeper_players.json"
METRICS_PATH  = ROOT / "data" / "processed" / "player_metrics.json"
OUTPUT_PATH   = ROOT / "output"             / "qb_trends.json"

PIPELINE_YEAR   = 2025
TREND_SEASONS   = [2025]         # 2025 season only
MIN_DB_TOTAL    = 10             # minimum total dropbacks to qualify
MIN_GAMES_DELTA = 6              # minimum total games before delta is non-null
RECENT_WINDOW   = 4              # "last N games" = recent bucket


# ---------------------------------------------------------------------------
# Name resolution (mirrors compute_player_metrics.py logic)
# ---------------------------------------------------------------------------

def _abbrev(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _build_abbreviated_lookup(sleeper_players: list[dict]) -> dict[str, str]:
    counts: dict[str, int]  = {}
    mapping: dict[str, str] = {}
    for p in sleeper_players:
        full = p.get("full_name", "").strip()
        if not full:
            continue
        ab = _abbrev(full)
        counts[ab]  = counts.get(ab, 0) + 1
        mapping[ab] = full
    return {ab: fn for ab, fn in mapping.items() if counts[ab] == 1}


def _build_team_disambig(sleeper_players: list[dict]) -> dict[str, dict]:
    counts: dict[str, int] = {}
    by_team: dict[str, dict] = {}
    for p in sleeper_players:
        full = p.get("full_name", "").strip()
        if not full:
            continue
        ab = _abbrev(full)
        counts[ab] = counts.get(ab, 0) + 1
        team = p.get("team")
        slot = by_team.setdefault(ab, {})
        if team in slot:
            # Multiple players share the same team slot (often both team=None).
            # Promote to a list so _resolve_qb_name can apply position filtering.
            existing = slot[team]
            if isinstance(existing, list):
                existing.append(full)
            else:
                slot[team] = [existing, full]
        else:
            slot[team] = full
    return {ab: teams for ab, teams in by_team.items() if counts[ab] > 1}


def _resolve_qb_name(
    pbp_name: str,
    player_lookup: dict[str, dict],
    abbreviated_lookup: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    pbp_team: str,
) -> str:
    """
    Resolve abbreviated PBP name to canonical Sleeper full name for QBs.

    Resolution order:
    1. Direct match in Sleeper full-name lookup.
    2. Unambiguous abbreviation lookup.
    3. Team-based disambiguation with position filter (QB only):
       a. Exact team match.
       b. QB-position filter → if one match, use it; if multiple QBs,
          prefer the one on an active roster (team != None).
    4. Return PBP name unchanged.
    """
    if pbp_name in player_lookup:
        return pbp_name
    if pbp_name in abbreviated_lookup:
        return abbreviated_lookup[pbp_name]
    if pbp_name in team_disambig:
        teams = team_disambig[pbp_name]
        # Exact team match — only when slot holds a single unambiguous name.
        if pbp_team and pbp_team in teams:
            slot = teams[pbp_team]
            if isinstance(slot, str):
                return slot
            # slot is a list — fall through to QB position filter below
        # Expand list-valued slots (e.g. {None: ["Aaron Rodgers", "Amari Rodgers"]})
        # into flat (team, full_name) pairs so position filtering works correctly.
        candidates: list[tuple] = []
        for t, val in teams.items():
            if isinstance(val, list):
                candidates.extend((t, n) for n in val)
            else:
                candidates.append((t, val))
        qb_matches = [
            (t, n) for t, n in candidates
            if player_lookup.get(n, {}).get("position") == "QB"
        ]
        if len(qb_matches) == 1:
            return qb_matches[0][1]
        if len(qb_matches) > 1:
            with_team = [(t, n) for t, n in qb_matches if t is not None]
            if with_team:
                return with_team[0][1]
            return qb_matches[0][1]
    return pbp_name


# ---------------------------------------------------------------------------
# Core trend computation
# ---------------------------------------------------------------------------

def _game_order(grp: pd.DataFrame) -> list[str]:
    """Return unique game_ids for a player group in chronological order."""
    cols = [c for c in ("season", "week") if c in grp.columns]
    if cols:
        return (
            grp[["game_id"] + cols]
            .drop_duplicates("game_id")
            .sort_values(cols)["game_id"]
            .tolist()
        )
    return list(grp["game_id"].unique())


def compute_qb_trends(
    pbp: pd.DataFrame,
    sleeper_players: list[dict],
    player_metrics: list[dict],
) -> list[dict]:
    """
    For every qualifying QB in the PBP, compute usage stats and deltas.
    Returns a list of row dicts ready for the output payload.
    """
    player_lookup = {p["full_name"]: p for p in sleeper_players}
    abbreviated_lookup = _build_abbreviated_lookup(sleeper_players)
    team_disambig      = _build_team_disambig(sleeper_players)
    metrics_by_player  = {p["player"]: p for p in player_metrics}

    has_sack = "sack" in pbp.columns

    all_dbs = pbp[pbp["pass_attempt"] == 1].copy()
    if all_dbs.empty:
        return []

    rush_plays = pbp[pbp["rush_attempt"] == 1].copy() if not pbp.empty else pd.DataFrame()

    seen: set[str] = set()
    rows: list[dict] = []

    for pbp_name, grp in all_dbs.groupby("passer_player_name"):
        official_att = len(grp[grp["sack"] != 1]) if has_sack else len(grp)
        if official_att < MIN_DB_TOTAL:
            continue

        pbp_team = str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns else ""
        full_name = _resolve_qb_name(
            str(pbp_name), player_lookup, abbreviated_lookup,
            team_disambig, pbp_team,
        )

        if full_name in seen:
            continue
        seen.add(full_name)

        info = metrics_by_player.get(full_name, {})
        if info.get("pos", "QB") != "QB":
            continue

        team = info.get("team") or pbp_team

        games = _game_order(grp)
        total_games = len(games)

        total_dbs = len(grp)

        if not rush_plays.empty and "rusher_player_name" in rush_plays.columns:
            # Filter by BOTH name AND team to avoid collisions with same-abbreviation
            # non-QB players (e.g. "B.Allen" matches both Brandon Allen QB/TEN and
            # Breelton Allen RB/NYJ — without the team filter all 19 RB rushes were
            # credited to the QB).
            qb_rush = rush_plays[
                (rush_plays["rusher_player_name"] == pbp_name) &
                (rush_plays["posteam"] == pbp_team)
            ]
        else:
            qb_rush = pd.DataFrame()

        total_rush_att = len(qb_rush)
        total_rush_td  = (
            int(qb_rush["touchdown"].sum())
            if not qb_rush.empty and "touchdown" in qb_rush.columns
            else 0
        )

        dropbacks_per_game = round(total_dbs      / total_games, 1) if total_games else 0.0
        rush_att_per_game  = round(total_rush_att / total_games, 1) if total_games else 0.0

        delta_dbs_pg  : float | None = None
        delta_rush_pg : float | None = None
        recent_n      : int   | None = None
        prior_n       : int   | None = None

        if total_games >= MIN_GAMES_DELTA:
            recent_ids = set(games[-RECENT_WINDOW:])
            prior_ids  = set(games[:-RECENT_WINDOW])
            recent_n   = len(recent_ids)
            prior_n    = len(prior_ids)

            r_dbs = grp[grp["game_id"].isin(recent_ids)]
            p_dbs = grp[grp["game_id"].isin(prior_ids)]
            r_dbs_pg = len(r_dbs) / recent_n if recent_n else 0.0
            p_dbs_pg = len(p_dbs) / prior_n  if prior_n  else 0.0
            delta_dbs_pg = round(r_dbs_pg - p_dbs_pg, 1)

            if not qb_rush.empty and "game_id" in qb_rush.columns:
                r_rush = qb_rush[qb_rush["game_id"].isin(recent_ids)]
                p_rush = qb_rush[qb_rush["game_id"].isin(prior_ids)]
                r_rush_pg = len(r_rush) / recent_n if recent_n else 0.0
                p_rush_pg = len(p_rush) / prior_n  if prior_n  else 0.0
                delta_rush_pg = round(r_rush_pg - p_rush_pg, 1)
            else:
                delta_rush_pg = 0.0

        rows.append({
            "player":                full_name,
            "team":                  team,
            "age":                   info.get("age"),
            "games":                 total_games,
            "dropbacksPerGame":      dropbacks_per_game,
            "deltaDropbacksPerGame": delta_dbs_pg,
            "rushAttPerGame":        rush_att_per_game,
            "deltaRushAttPerGame":   delta_rush_pg,
            "rushTd":                total_rush_td,
            "recentWindowGames":     recent_n,
            "priorWindowGames":      prior_n,
        })

    rows.sort(key=lambda r: r.get("dropbacksPerGame") or 0, reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Output payload builder
# ---------------------------------------------------------------------------

COLUMNS = [
    {"key": "player",                "label": "Player",   "type": "string", "group": "identity",      "defaultVisible": True},
    {"key": "team",                  "label": "Team",     "type": "string", "group": "identity",      "defaultVisible": True},
    {"key": "age",                   "label": "Age",      "type": "number", "group": "identity",      "defaultVisible": True},
    {"key": "games",                 "label": "GP",       "type": "number", "group": "identity",      "defaultVisible": True},
    {"key": "dropbacksPerGame",      "label": "DB/Gm",    "type": "number", "group": "passing_usage", "defaultVisible": True},
    {"key": "deltaDropbacksPerGame", "label": "Δ DB/Gm",  "type": "number", "group": "passing_usage", "defaultVisible": True},
    {"key": "rushAttPerGame",        "label": "Rush/Gm",  "type": "number", "group": "rushing_usage", "defaultVisible": True},
    {"key": "deltaRushAttPerGame",   "label": "Δ Rush/Gm","type": "number", "group": "rushing_usage", "defaultVisible": True},
    {"key": "rushTd",                "label": "Rush TD",  "type": "number", "group": "rushing_usage", "defaultVisible": True},
]


def build_payload(rows: list[dict], pbp: pd.DataFrame) -> dict[str, Any]:
    seasons = sorted(pbp["season"].dropna().unique().astype(int).tolist()) if "season" in pbp.columns else TREND_SEASONS
    if len(seasons) >= 2:
        sample_window = f"{seasons[0]}\u2013{seasons[-1]}"
    else:
        sample_window = str(seasons[0]) if seasons else str(PIPELINE_YEAR)

    return {
        "position":     "QB",
        "sampleLabel":  "Rolling Multi-Season",
        "sampleWindow": sample_window,
        "pipelineYear": PIPELINE_YEAR,
        "columns":      COLUMNS,
        "rows":         rows,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not PBP_PATH.exists():
        print(f"ERROR: {PBP_PATH} not found. Run pull_nflverse_data.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading play-by-play data...")
    pbp_full = pd.read_parquet(PBP_PATH)
    pbp_full = pbp_full[pbp_full["season_type"] == "REG"].copy()  # regular season only
    if "season" in pbp_full.columns:
        pbp = pbp_full[pbp_full["season"].isin(TREND_SEASONS)].copy()
    else:
        pbp = pbp_full.copy()
    print(f"  {len(pbp):,} plays across {TREND_SEASONS} seasons.")

    if not SLEEPER_PATH.exists():
        print(f"ERROR: {SLEEPER_PATH} not found. Run pull_sleeper_players.py first.", file=sys.stderr)
        sys.exit(1)

    with open(SLEEPER_PATH) as f:
        sleeper_players: list[dict] = json.load(f)
    print(f"  {len(sleeper_players):,} Sleeper players loaded.")

    if not METRICS_PATH.exists():
        print(f"ERROR: {METRICS_PATH} not found. Run compute_player_metrics.py first.", file=sys.stderr)
        sys.exit(1)

    with open(METRICS_PATH) as f:
        player_metrics: list[dict] = json.load(f)
    print(f"  {len(player_metrics):,} player metric records loaded.")

    print("Computing QB usage trends...")
    rows = compute_qb_trends(pbp, sleeper_players, player_metrics)
    print(f"  {len(rows)} QBs qualified.")

    payload = build_payload(rows, pbp)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.1f} KB, {len(rows)} QBs)")
    print("QB trends build complete.")


if __name__ == "__main__":
    main()
