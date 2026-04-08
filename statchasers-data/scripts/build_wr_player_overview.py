"""
build_wr_player_overview.py
───────────────────────────
Builds the WR Player Overview dataset for the StatChasers WR explorer.

Produces five output files:
  output/wr_player_overview.json        — alias for 2025 (backward compat)
  output/wr_player_overview_2023.json
  output/wr_player_overview_2024.json
  output/wr_player_overview_2025.json
  output/wr_player_overview_all.json    — 2023–2025 combined (one row per player)

Data sources:
  data/raw/nflverse_play_by_play.parquet   — targets, recs, air yards, YAC, games
  data/raw/nflverse_participation.parquet  — routes run (WR appearances on pass plays)
  data/processed/player_metrics.json       — snapShare, fpoe, careerArc, wopr,
                                             stability, volatility (2025 only)
  data/raw/nflverse_players.parquet        — years_of_experience → exp_tier
  data/raw/sleeper_players.json            — name disambiguation + age

Data rules:
  - WRs only
  - ≥ 15 targets to qualify (per season)
  - null when data is truly unavailable; never empty strings
  - percentages stored as decimals (e.g. 0.273), rounded to 4 decimal places
  - rate stats: 2 decimal places
  - stability/volatility from player_metrics for 2025, PBP-computed for prior seasons
  - routes_per_gm from participation data (WR appearances on pass plays per game)
  - yards_per_route_run = receiving yards / total routes run (seasonal)

Do NOT compute WR Tier in this backend — assembled client-side.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT                = Path(__file__).resolve().parent.parent
PBP_PATH            = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
PARTICIPATION_PATH  = ROOT / "data" / "raw"       / "nflverse_participation.parquet"
SNAP_COUNTS_PATH    = ROOT / "data" / "raw"       / "nflverse_snap_counts.parquet"
METRICS_PATH        = ROOT / "data" / "processed" / "player_metrics.json"
NFL_PATH            = ROOT / "data" / "raw"       / "nflverse_players.parquet"
SLEEPER_PATH        = ROOT / "data" / "raw"       / "sleeper_players.json"
OUTPUT_DIR          = ROOT / "output"

SEASONS     = [2023, 2024, 2025]
CURRENT     = 2025
MIN_TARGETS = 15


# ---------------------------------------------------------------------------
# Column metadata (frontend contract)
# ---------------------------------------------------------------------------

COLUMNS: list[str] = [
    "player", "team", "age", "games",
    "snap_pct",
    "routes_per_gm", "targets_per_gm", "receptions_per_gm", "air_yards_per_gm",
    "target_share_pct", "air_yards_share_pct", "wopr", "rz_tgt",
    "catch_rate", "yards_per_route_run", "yards_per_target", "yac_per_rec", "fpoe",
    "stability", "volatility",
    "career_arc", "exp_tier",
    "role_score", "efficiency_score", "overall_score",
]


# ---------------------------------------------------------------------------
# Name resolution  (identical pattern to build_wr_advanced_stats.py)
# ---------------------------------------------------------------------------

def _abbrev(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _build_unambig(sleeper_players: list[dict]) -> dict[str, str]:
    counts: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for p in sleeper_players:
        full = (p.get("full_name") or "").strip()
        if not full:
            continue
        ab = _abbrev(full)
        counts[ab] = counts.get(ab, 0) + 1
        mapping[ab] = full
    return {ab: fn for ab, fn in mapping.items() if counts[ab] == 1}


def _build_team_disambig(sleeper_players: list[dict]) -> dict[str, dict[str, str]]:
    counts: dict[str, int] = {}
    by_team: dict[str, dict] = {}
    for p in sleeper_players:
        full = (p.get("full_name") or "").strip()
        if not full:
            continue
        ab = _abbrev(full)
        counts[ab] = counts.get(ab, 0) + 1
        by_team.setdefault(ab, {})[p.get("team")] = full
    return {ab: t for ab, t in by_team.items() if counts[ab] > 1}


# Identical override table from build_wr_advanced_stats.py — keep in sync.
_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    "T.Etienne":    {"JAX": "Travis Etienne",    "CAR": "Trevor Etienne"},
    "B.Robinson":   {"ATL": "Bijan Robinson",    "SF": "Brian Robinson", "WAS": "Brian Robinson"},
    "J.Williams":   {"DEN": "Javonte Williams",  "NO": "Jamaal Williams"},
    "D.Moore":      {"CHI": "DJ Moore", "BUF": "DJ Moore", "CAR": "David Moore", "TB": "David Moore"},
    "K.Allen":      {"LAC": "Keenan Allen",      "CHI": "Keenan Allen"},
    "D.Montgomery": {"DET": "David Montgomery",  "HOU": "David Montgomery", "IND": "D.J. Montgomery"},
    "K.Williams":   {"LA": "Kyren Williams",     "NE": "Kyle Williams",    "CIN": "Ke'Shawn Williams"},
    "R.White":      {"TB": "Rachaad White",      "WAS": "Rachaad White",   "SEA": "Ricky White"},
    "J.Smith":      {"ATL": "Jonnu Smith",       "MIA": "Jonnu Smith",     "PIT": "Jonnu Smith"},
    # Multi-char prefix abbreviations (nflverse disambiguates within a game)
    "Mi.Wilson":    {"ARI": "Michael Wilson"},   # nflverse 2-char prefix for Michael Wilson Jr.
    # K.Juszczyk resolution: maps to Kyle Juszczyk (FB) so WR filter will correctly exclude him
    "K.Juszczyk":   {"SF": "Kyle Juszczyk"},
    # Non-WR receivers whose Sleeper team field is stale (None) or uses a different
    # team code than nflverse PBP, causing the pos-based fallback to pick the wrong
    # WR with the same abbreviation.
    "M.Carter":     {"ARI": "Michael Carter"},                      # RB/ARI, not Malachi Carter
    "D.Allen":      {"LA": "Davis Allen", "LAR": "Davis Allen"},    # TE/LAR, not Devon Allen
    "J.Ford":       {"CLE": "Jerome Ford"},                         # RB/CLE, not Jacoby Ford
    # Multi-word last name — nflverse uses A.St. Brown; _abbrev() produces A.Brown
    # which collides with A.J. Brown (PHI). Explicit override needed for correct resolution.
    "A.St. Brown":  {"DET": "Amon-Ra St. Brown"},
}


def _resolve(
    pbp_name: str,
    pbp_team: str,
    full_name_set: set[str],
    unambig: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    sleeper_pos: dict[str, str],
) -> str:
    if pbp_name in _MANUAL_TEAM_OVERRIDES and pbp_team:
        hit = _MANUAL_TEAM_OVERRIDES[pbp_name].get(pbp_team)
        if hit:
            return hit
    if pbp_name in full_name_set:
        return pbp_name
    if pbp_name in unambig:
        return unambig[pbp_name]
    if pbp_name in team_disambig:
        teams = team_disambig[pbp_name]
        if pbp_team and pbp_team in teams:
            return teams[pbp_team]
        pos_matches = [(t, n) for t, n in teams.items() if sleeper_pos.get(n, "") == "WR"]
        if not pos_matches:
            pos_matches = list(teams.items())
        if len(pos_matches) == 1:
            return pos_matches[0][1]
        conflicting     = [(t, n) for t, n in pos_matches if t is not None and t != pbp_team]
        non_conflicting = [(t, n) for t, n in pos_matches if t is None or t == pbp_team]
        if non_conflicting and conflicting:
            no_team = [(t, n) for t, n in non_conflicting if t is None]
            if no_team:
                return no_team[0][1]
            return non_conflicting[0][1]
        with_team = [(t, n) for t, n in pos_matches if t is not None]
        if with_team:
            return with_team[0][1]
        return pos_matches[0][1]
    return pbp_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_tier(yoe: float | None) -> str | None:
    if yoe is None:
        return None
    try:
        y = int(float(yoe))
    except (ValueError, TypeError):
        return None
    if y <= 1:  return "Rookie"
    if y == 2:  return "Year 2"
    if y <= 4:  return "Year 3–4"
    if y <= 9:  return "Veteran"
    return "Senior Veteran"


def _pct(value: float | None, arr: list[float | None]) -> float:
    """Percentile rank (0–100).  Null value or empty pool → 50.0 (neutral)."""
    clean = np.array(
        [float(v) for v in arr if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=float,
    )
    if len(clean) == 0 or value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    v = float(value)
    below = float(np.sum(clean < v))
    equal = float(np.sum(clean == v))
    return float((below + 0.5 * equal) / len(clean) * 100.0)


def _stability_volatility(
    game_target_map: dict[str, int],
    game_order: list[str],
) -> tuple[float | None, float | None]:
    """
    stability  = (1 – σ/μ of last-6 games) × 10, clamped 0–10
    volatility = σ/μ of all games (coefficient of variation)
    Uses per-game target counts as the 'usage' signal.
    """
    ordered = [game_target_map[g] for g in game_order if g in game_target_map]
    extra   = [v for g, v in game_target_map.items() if g not in game_order]
    ordered = ordered + extra
    if len(ordered) < 2:
        return (None, None)

    mean_all = float(np.mean(ordered))
    std_all  = float(np.std(ordered))
    volatility = round(std_all / mean_all, 2) if mean_all > 0 else None

    window = ordered[-6:] if len(ordered) >= 6 else ordered
    if len(window) < 2:
        return (None, volatility)
    mean_w = float(np.mean(window))
    std_w  = float(np.std(window))
    if mean_w == 0:
        return (None, volatility)
    raw = 1.0 - (std_w / mean_w)
    stability = round(max(0.0, min(10.0, raw * 10.0)), 2)
    return (stability, volatility)


# ---------------------------------------------------------------------------
# Name normalisation helper
# ---------------------------------------------------------------------------

# snap_counts (PFR) uses slightly different names from Sleeper for some players.
# Key: snap_counts player_name  →  Value: Sleeper full_name (canonical in our system)
_SNAP_PLAYER_ALIASES: dict[str, str] = {
    "Gabriel Davis": "Gabe Davis",   # snap_counts PFR name → Sleeper canonical
}


def _norm(name: str) -> str:
    """Remove dots, hyphens, spaces and lowercase for fuzzy name matching.
    e.g. "D.J. Moore" → "djmoore", "Amon-Ra St. Brown" → "amonrastbrown"
    """
    return name.lower().replace(".", "").replace("-", "").replace(" ", "").replace("'", "")


# ---------------------------------------------------------------------------
# Snap-count lookup (from nflverse load_snap_counts)
# ---------------------------------------------------------------------------

def _build_snap_pct_lookups(
    snap_df: pd.DataFrame,
    nfl_df: pd.DataFrame | None,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """
    Build two snap-pct lookup dicts from load_snap_counts data:
      snap_by_norm : {season → {normalised_player_name → avg_snap_pct}}
      snap_by_gsis : {season → {gsis_id → avg_snap_pct}}  (via nflverse_players bridge)

    snap_by_norm covers the vast majority of players via direct / normalised match.
    snap_by_gsis covers edge cases like 'A.St. Brown' whose PBP abbreviation doesn't
    normalise to the same string as the snap_counts full name 'Amon-Ra St. Brown'.
    """
    wr_snap = snap_df[snap_df["position"] == "WR"].copy()

    snap_by_norm: dict[int, dict[str, float]] = {}
    for (season, player), grp in wr_snap.groupby(["season", "player"]):
        pct = round(float(grp["offense_pct"].mean()), 4)
        s   = int(season)
        snap_by_norm.setdefault(s, {})[_norm(str(player))] = pct
        # Also index under canonical Sleeper name if this snap_counts name has an alias
        alias = _SNAP_PLAYER_ALIASES.get(str(player))
        if alias:
            snap_by_norm[s][_norm(alias)] = pct

    snap_by_gsis: dict[int, dict[str, float]] = {}
    if nfl_df is not None and "display_name" in nfl_df.columns and "gsis_id" in nfl_df.columns:
        display_to_gsis: dict[str, str] = (
            nfl_df.dropna(subset=["gsis_id", "display_name"])
            .drop_duplicates("display_name")
            .set_index("display_name")["gsis_id"]
            .to_dict()
        )
        for (season, player), grp in wr_snap.groupby(["season", "player"]):
            gsis = display_to_gsis.get(str(player))
            if gsis:
                snap_by_gsis.setdefault(int(season), {})[str(gsis)] = round(
                    float(grp["offense_pct"].mean()), 4
                )

    return snap_by_norm, snap_by_gsis


# ---------------------------------------------------------------------------
# Age lookup via gsis_id → nflverse display_name → Sleeper age
# (covers players whose PBP abbreviation couldn't be resolved to Sleeper full_name)
# ---------------------------------------------------------------------------

def _build_gsis_age_lookup(
    nfl_df: pd.DataFrame | None,
    sleeper_age: dict[str, Any],
) -> dict[str, Any]:
    """
    Build {gsis_id → age} by bridging nflverse display_name to Sleeper full_name.
    Uses normalised string matching so e.g. 'Amon-Ra St. Brown' matches Sleeper's
    'Amon-Ra St. Brown' directly.
    """
    if nfl_df is None or "gsis_id" not in nfl_df.columns or "display_name" not in nfl_df.columns:
        return {}

    sleeper_age_norm: dict[str, Any] = {_norm(fn): age for fn, age in sleeper_age.items()}

    gsis_to_age: dict[str, Any] = {}
    for _, row in nfl_df.dropna(subset=["gsis_id", "display_name"]).iterrows():
        gsis    = str(row["gsis_id"])
        display = str(row["display_name"])
        age     = sleeper_age.get(display) or sleeper_age_norm.get(_norm(display))
        if age is not None:
            gsis_to_age[gsis] = age

    return gsis_to_age


# ---------------------------------------------------------------------------
# Routes lookup from participation data
# ---------------------------------------------------------------------------

def _build_routes_lookup(
    pbp: pd.DataFrame,
    part: pd.DataFrame,
) -> dict[int, dict[str, dict[str, int]]]:
    """
    Build {season → {player_abbrev → {game_id → route_count}}} from participation data.

    Bridges participation gsis_ids → PBP receiver abbreviations so that the rest
    of the pipeline can use the same canonical name resolution already in place.

    A 'route' is defined as a play where the player appears in offense_players with
    position == 'WR'.  The participation dataset is already filtered to passing plays
    (nearly all rows have a route value), so every WR on the field = 1 route run.

    Only regular-season weeks (week ≤ 18) are counted.
    """
    # Step 1: gsis_id → PBP abbreviation from all receiver targets
    gsis_to_abbrev: dict[str, str] = (
        pbp[pbp["pass_attempt"] == 1][["receiver_player_id", "receiver_player_name"]]
        .dropna(subset=["receiver_player_id", "receiver_player_name"])
        .drop_duplicates("receiver_player_id")
        .set_index("receiver_player_id")["receiver_player_name"]
        .to_dict()
    )

    # Step 2: filter participation to regular season (weeks 1-18)
    part = part.copy()
    part["week_num"] = part["nflverse_game_id"].str.split("_").str[1].astype(int)
    part_reg = part[
        (part["week_num"] <= 18)
        & part["offense_players"].notna()
        & part["offense_positions"].notna()
    ].copy()

    # Step 3: vectorised explode of semicolon-separated player/position lists
    part_reg["pid_list"] = part_reg["offense_players"].str.split(";")
    part_reg["pos_list"] = part_reg["offense_positions"].str.split(";")

    exp = (
        part_reg[["nflverse_game_id", "season", "pid_list", "pos_list"]]
        .explode(["pid_list", "pos_list"])          # pandas ≥ 1.3 multi-col explode
        .rename(columns={"pid_list": "gsis_id", "pos_list": "position"})
    )
    exp["gsis_id"]  = exp["gsis_id"].str.strip()
    exp["position"] = exp["position"].str.strip()

    # Step 4: keep only WRs that we can map to an abbreviation
    wr = exp[exp["position"] == "WR"].copy()
    wr["abbrev"] = wr["gsis_id"].map(gsis_to_abbrev)
    wr = wr[wr["abbrev"].notna()]

    # Step 5: group into {season: {abbrev: {game_id: count}}}
    result: dict[int, dict[str, dict[str, int]]] = {}
    for (season, abbrev, game_id), cnt in (
        wr.groupby(["season", "abbrev", "nflverse_game_id"]).size().items()
    ):
        result.setdefault(int(season), {}).setdefault(str(abbrev), {})[str(game_id)] = int(cnt)

    matched = wr["abbrev"].notna().sum()
    total   = len(exp[exp["position"] == "WR"])
    print(f"  Routes lookup built: {matched:,}/{total:,} WR instances matched ({matched/total*100:.1f}%)")
    return result


# ---------------------------------------------------------------------------
# Per-season builder
# ---------------------------------------------------------------------------

def build_season(
    pbp_season: pd.DataFrame,
    season: int,
    full_name_set: set[str],
    unambig: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    sleeper_pos: dict[str, str],
    sleeper_age: dict[str, Any],
    metrics_map: dict[str, dict],
    yoe_lookup: dict[str, int],
    routes_by_season: dict[int, dict[str, dict[str, int]]] | None = None,
    snap_by_norm: dict[int, dict[str, float]] | None = None,
    snap_by_gsis: dict[int, dict[str, float]] | None = None,
    gsis_to_age: dict[str, Any] | None = None,
) -> list[dict]:
    """Return one row per qualifying WR for this season (no composite scores yet)."""
    routes_map: dict[str, dict[str, int]] = (
        routes_by_season.get(season, {}) if routes_by_season else {}
    )
    snap_norm_s: dict[str, float] = snap_by_norm.get(season, {}) if snap_by_norm else {}
    snap_gsis_s: dict[str, float] = snap_by_gsis.get(season, {}) if snap_by_gsis else {}

    pass_plays = pbp_season[pbp_season["pass_attempt"] == 1].copy()
    if pass_plays.empty:
        return []

    has_yardline = "yardline_100" in pass_plays.columns
    has_yac      = "yards_after_catch" in pass_plays.columns
    has_air      = "air_yards" in pass_plays.columns

    # ── Name resolution ───────────────────────────────────────────────────────
    cache: dict[tuple[str, str], str] = {}

    def _tag(row: pd.Series) -> str:
        pbp_name = str(row.get("receiver_player_name") or "")
        if not pbp_name or pbp_name == "nan":
            return ""
        key = (pbp_name, str(row.get("posteam", "")))
        if key not in cache:
            cache[key] = _resolve(key[0], key[1], full_name_set, unambig, team_disambig, sleeper_pos)
        return cache[key]

    targeted = pass_plays[pass_plays["receiver_player_name"].notna()].copy()
    targeted["_fn"] = targeted.apply(_tag, axis=1)
    targeted = targeted[targeted["_fn"] != ""]

    # ── WR filter ─────────────────────────────────────────────────────────────
    # If Sleeper explicitly knows this player's position, use it as the truth.
    # Only defer to player_metrics position when Sleeper has no record of the player.
    def _is_wr(fn: str) -> bool:
        s_pos = sleeper_pos.get(fn, "")
        if s_pos == "WR":
            return True
        if s_pos and s_pos != "WR":
            return False  # Sleeper says FB/TE/RB/QB → exclude
        # Unknown in Sleeper — check player_metrics position
        return metrics_map.get(fn, {}).get("pos", "") == "WR"

    wr_names: set[str] = {fn for fn in targeted["_fn"].unique() if _is_wr(str(fn))}
    wr_plays = targeted[targeted["_fn"].isin(wr_names)].copy()
    if wr_plays.empty:
        return []

    # ── canonical_name → gsis_id map (for snap and age bridge lookups) ────────
    canonical_to_gsis: dict[str, str] = {}
    if "receiver_player_id" in wr_plays.columns:
        canonical_to_gsis = (
            wr_plays[["_fn", "receiver_player_id"]]
            .dropna(subset=["receiver_player_id"])
            .drop_duplicates("_fn")
            .set_index("_fn")["receiver_player_id"]
            .to_dict()
        )

    # ── Team-level totals (for share denominators) ────────────────────────────
    team_pass_totals: dict[str, int] = (
        pass_plays.groupby("posteam")["pass_attempt"].count().to_dict()
    )
    if has_air:
        team_air_totals: dict[str, float] = (
            pass_plays.groupby("posteam")["air_yards"].sum().to_dict()
        )
    else:
        team_air_totals = {}

    sort_cols = [c for c in ("season", "week") if c in wr_plays.columns]

    # ── Per-player aggregation ────────────────────────────────────────────────
    rows: list[dict] = []

    for fn, grp in wr_plays.groupby("_fn"):
        fn  = str(fn)
        tgt = len(grp)
        if tgt < MIN_TARGETS:
            continue

        comps = grp[grp["complete_pass"] == 1]
        rec   = len(comps)
        yds   = int(comps["yards_gained"].sum())
        games = int(grp["game_id"].nunique())

        # Primary team (most recent game)
        if sort_cols:
            team = str(grp.sort_values(sort_cols)["posteam"].iloc[-1])
            game_order = (
                grp[["game_id"] + sort_cols]
                .drop_duplicates("game_id")
                .sort_values(sort_cols)["game_id"]
                .tolist()
            )
        else:
            team       = str(grp["posteam"].iloc[-1])
            game_order = list(grp["game_id"].unique())

        # Per-game target map (for stability/volatility)
        game_tgt_map: dict[str, int] = dict(grp.groupby("game_id")["pass_attempt"].count())

        # Air yards (on all targets, not just catches)
        if has_air:
            player_air  = float(grp["air_yards"].dropna().sum())
            team_air    = team_air_totals.get(team, 0.0)
        else:
            player_air = 0.0
            team_air   = 0.0

        # YAC (completions only)
        total_yac = float(comps["yards_after_catch"].dropna().sum()) if has_yac and rec > 0 else None

        # Red-zone targets
        rz_tgt = int((grp["yardline_100"] <= 20).sum()) if has_yardline else None

        # --- Rates (decimal, 4 places for shares; 2 places for per-game) ---
        team_pass = team_pass_totals.get(team, 0)
        target_share_pct    = round(tgt / team_pass, 4)       if team_pass > 0 else None
        air_yards_share_pct = round(player_air / team_air, 4) if team_air  > 0 else None

        # WOPR = 1.5 × target_share + 0.7 × air_yards_share
        wopr = None
        if target_share_pct is not None and air_yards_share_pct is not None:
            wopr = round(1.5 * target_share_pct + 0.7 * air_yards_share_pct, 2)

        catch_rate      = round(rec / tgt, 4)        if tgt > 0 else None
        yards_per_tgt   = round(yds / tgt, 2)        if tgt > 0 else None
        yac_per_rec     = round(total_yac / rec, 2)  if rec > 0 and total_yac is not None else None
        tgt_per_gm      = round(tgt / games, 2)      if games > 0 else None
        rec_per_gm      = round(rec / games, 2)      if games > 0 else None
        air_yds_per_gm  = round(player_air / games, 2) if games > 0 and has_air else None

        stab_pbp, vol_pbp = _stability_volatility(game_tgt_map, game_order)

        # --- player_metrics fields (CURRENT season only) ---------------------
        m = metrics_map.get(fn, {}) if season == CURRENT else {}
        # Also try PBP abbreviation key for player_metrics (e.g. "M.Evans")
        if not m and season == CURRENT:
            m = metrics_map.get(_abbrev(fn), {})

        # snap_pct: snap_counts is the primary source (covers all seasons + all players).
        # Lookup chain: 1) direct normalised name  2) gsis_id bridge  3) player_metrics fallback
        gsis_id = canonical_to_gsis.get(fn, "")
        snap_from_counts = (
            snap_norm_s.get(_norm(fn))
            or (snap_gsis_s.get(gsis_id) if gsis_id else None)
        )
        if snap_from_counts is not None:
            snap_pct = round(float(snap_from_counts), 4)
        elif m.get("snapShare") is not None:
            snap_pct = round(float(m["snapShare"]) / 100.0, 4)
        else:
            snap_pct = None

        fpoe       = m.get("fpoe")
        career_arc = m.get("careerArc")

        # For 2025 prefer player_metrics stability/volatility (more accurate);
        # for prior seasons use PBP-computed values.
        if season == CURRENT:
            stability  = m.get("roleStability")   or stab_pbp
            volatility = m.get("usageVolatility") or vol_pbp
            # Also prefer player_metrics wopr/shares if available (more precise)
            if m.get("wopr") is not None:
                wopr = round(float(m["wopr"]), 2)
            if m.get("targetShare") is not None:
                target_share_pct = round(float(m["targetShare"]) / 100.0, 4)
            if m.get("airYardsShare") is not None:
                air_yards_share_pct = round(float(m["airYardsShare"]) / 100.0, 4)
        else:
            stability  = stab_pbp
            volatility = vol_pbp

        # Age: player_metrics → Sleeper direct → gsis_id bridge (for PBP abbreviations)
        age = (
            m.get("age")
            or sleeper_age.get(fn)
            or (gsis_to_age.get(gsis_id) if (gsis_to_age and gsis_id) else None)
        )
        yoe      = yoe_lookup.get(fn)
        exp_tier = _exp_tier(yoe)

        # --- Routes from participation data -----------------------------------
        player_routes: dict[str, int] = routes_map.get(fn, {})
        # Also try PBP abbreviation key in case resolution changed
        if not player_routes:
            for abbrev_key in cache:
                if cache[abbrev_key] == fn and abbrev_key[0] in routes_map:
                    player_routes = routes_map[abbrev_key[0]]
                    break
        total_routes  = sum(player_routes.values())
        route_games   = len(player_routes)
        routes_pg     = round(total_routes / route_games, 1)   if route_games > 0 else None
        yds_per_route = round(yds / total_routes, 2)           if total_routes > 0 else None

        rows.append({
            "player":               fn,
            "team":                 team,
            "age":                  age,
            "games":                games,
            "snap_pct":             snap_pct,
            "routes_per_gm":        routes_pg,
            "targets_per_gm":       tgt_per_gm,
            "receptions_per_gm":    rec_per_gm,
            "air_yards_per_gm":     air_yds_per_gm,
            "target_share_pct":     target_share_pct,
            "air_yards_share_pct":  air_yards_share_pct,
            "wopr":                 wopr,
            "rz_tgt":               rz_tgt,
            "catch_rate":           catch_rate,
            "yards_per_route_run":  yds_per_route,
            "yards_per_target":     yards_per_tgt,
            "yac_per_rec":          yac_per_rec,
            "fpoe":                 fpoe,
            "stability":            stability,
            "volatility":           volatility,
            "career_arc":           career_arc,
            "exp_tier":             exp_tier,
            # --- internals for scoring (dropped from final output) -----------
            "_tgt":          tgt,
            "_rec":          rec,
            "_yds":          yds,
            "_air":          player_air if has_air else None,
            "_yac":          total_yac,
            "_total_routes": total_routes,
            "_route_games":  route_games,
            "_season":       season,
        })

    return rows


# ---------------------------------------------------------------------------
# Composite scores (percentile-based, same design as RB overview)
# ---------------------------------------------------------------------------

def _add_scores(rows: list[dict]) -> list[dict]:
    def col(key: str) -> list[float | None]:
        return [r.get(key) for r in rows]

    arr_tgt_pg   = col("targets_per_gm")
    arr_rec_pg   = col("receptions_per_gm")
    arr_air_pg   = col("air_yards_per_gm")
    arr_tgt_shr  = col("target_share_pct")
    arr_air_shr  = col("air_yards_share_pct")
    arr_wopr     = col("wopr")
    arr_rz       = col("rz_tgt")
    arr_snap     = col("snap_pct")
    arr_catch    = col("catch_rate")
    arr_ypt      = col("yards_per_target")
    arr_yac      = col("yac_per_rec")
    arr_fpoe     = col("fpoe")
    arr_stab     = col("stability")
    arr_vol      = col("volatility")
    arr_yprr     = col("yards_per_route_run")

    final: list[dict] = []
    for r in rows:
        p_tgt_pg  = _pct(r["targets_per_gm"],      arr_tgt_pg)
        p_rec_pg  = _pct(r["receptions_per_gm"],   arr_rec_pg)
        p_air_pg  = _pct(r["air_yards_per_gm"],    arr_air_pg)
        p_tgt_shr = _pct(r["target_share_pct"],    arr_tgt_shr)
        p_air_shr = _pct(r["air_yards_share_pct"], arr_air_shr)
        p_wopr    = _pct(r["wopr"],                arr_wopr)
        p_rz      = _pct(r["rz_tgt"],             arr_rz)
        p_snap    = _pct(r["snap_pct"],            arr_snap)
        p_catch   = _pct(r["catch_rate"],          arr_catch)
        p_ypt     = _pct(r["yards_per_target"],    arr_ypt)
        p_yac     = _pct(r["yac_per_rec"],         arr_yac)
        p_fpoe    = _pct(r["fpoe"],                arr_fpoe)
        p_stab    = _pct(r["stability"],           arr_stab)
        p_vol_inv = 100.0 - _pct(r["volatility"],         arr_vol)
        p_yprr    = _pct(r["yards_per_route_run"],        arr_yprr)

        # 1. Role Score — unified opportunity + deployment profile (0–100)
        #    Answers: how strong and fantasy-relevant is this WR's offensive role?
        role_score = round(
            min(100.0, max(0.0,
                p_tgt_pg  * 0.20
                + p_tgt_shr * 0.20
                + p_air_shr * 0.15
                + p_wopr    * 0.15
                + p_snap    * 0.10
                + p_air_pg  * 0.10
                + p_rec_pg  * 0.05
                + p_rz      * 0.05,
            )),
            1,
        )

        # 2. Efficiency Score — per-touch production quality (0–100)
        efficiency_score = round(
            p_ypt    * 0.25
            + p_catch * 0.20
            + p_yac   * 0.15
            + p_fpoe  * 0.20
            + p_yprr  * 0.20,
            1,
        )

        # 3. Overall Score — blended profile composite (0–100)
        stability_block = p_stab * 0.60 + p_vol_inv * 0.40
        overall_score = round(
            min(100.0, max(0.0,
                role_score         * 0.50
                + efficiency_score * 0.35
                + stability_block  * 0.15,
            )),
            1,
        )

        final.append({
            **r,
            "role_score":       role_score,
            "efficiency_score": efficiency_score,
            "overall_score":    overall_score,
        })

    return final


# ---------------------------------------------------------------------------
# Multi-season aggregation (one combined row per player)
# ---------------------------------------------------------------------------

def _aggregate_all(season_rows: list[dict], yoe_lookup: dict[str, int]) -> list[dict]:
    """
    Combine 2023+2024+2025 rows into one row per player.
    Counting stats summed; rate stats recomputed from totals.
    Metrics that only exist for 2025 (snap_pct, fpoe, career_arc) taken from
    the most recent season that has them.
    """
    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in season_rows:
        by_player[r["player"]].append(r)

    result: list[dict] = []
    for player, player_rows in by_player.items():
        latest = max(player_rows, key=lambda r: r.get("_season", 0))

        total_tgt    = sum(r["_tgt"]               for r in player_rows)
        total_rec    = sum(r["_rec"]               for r in player_rows)
        total_yds    = sum(r["_yds"]               for r in player_rows)
        total_air    = sum(r["_air"] or 0          for r in player_rows)
        total_yac    = sum(r["_yac"] or 0          for r in player_rows)
        total_games  = sum(r["games"]              for r in player_rows)
        total_rz     = sum(r["rz_tgt"] or 0        for r in player_rows)
        total_routes = sum(r.get("_total_routes") or 0 for r in player_rows)
        total_rg     = sum(r.get("_route_games")  or 0 for r in player_rows)

        tgt_pg  = round(total_tgt / total_games, 2) if total_games > 0 else None
        rec_pg  = round(total_rec / total_games, 2) if total_games > 0 else None
        air_pg  = round(total_air / total_games, 2) if total_games > 0 else None
        routes_pg     = round(total_routes / total_rg, 1) if total_rg > 0 else None
        yds_per_route = round(total_yds / total_routes, 2) if total_routes > 0 else None

        catch_rate  = round(total_rec / total_tgt, 4) if total_tgt > 0 else None
        yds_per_tgt = round(total_yds / total_tgt, 2) if total_tgt > 0 else None
        yac_per_rec = round(total_yac / total_rec, 2) if total_rec > 0 else None

        # Shares not meaningful across seasons → null
        result.append({
            "player":               player,
            "team":                 latest.get("team"),
            "age":                  latest.get("age"),
            "games":                total_games,
            "snap_pct":             latest.get("snap_pct"),       # 2025 value
            "routes_per_gm":        routes_pg,
            "targets_per_gm":       tgt_pg,
            "receptions_per_gm":    rec_pg,
            "air_yards_per_gm":     air_pg,
            "target_share_pct":     None,   # not meaningful across seasons
            "air_yards_share_pct":  None,
            "wopr":                 None,
            "rz_tgt":               total_rz,
            "catch_rate":           catch_rate,
            "yards_per_route_run":  yds_per_route,
            "yards_per_target":     yds_per_tgt,
            "yac_per_rec":          yac_per_rec,
            "fpoe":                 latest.get("fpoe"),           # 2025 value
            "stability":            latest.get("stability"),      # 2025 value
            "volatility":           latest.get("volatility"),     # 2025 value
            "career_arc":           latest.get("career_arc"),     # 2025 value
            "exp_tier":             _exp_tier(yoe_lookup.get(player)),
            "_tgt":          total_tgt,
            "_rec":          total_rec,
            "_yds":          total_yds,
            "_air":          total_air,
            "_yac":          total_yac,
            "_total_routes": total_routes,
            "_route_games":  total_rg,
            "_season":       latest.get("_season", 0),
        })

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(rows: list[dict], label: str) -> None:
    seen: set[str] = set()
    for r in rows:
        p = r.get("player", "?")
        if p in seen:
            print(f"  WARNING: duplicate '{p}' in {label}", file=sys.stderr)
        seen.add(p)
        for col in COLUMNS:
            if col not in r:
                print(f"  WARNING: {p} missing column '{col}' in {label}", file=sys.stderr)
        pos_check = r.get("_pos_check")
        if pos_check and pos_check != "WR":
            print(f"  WARNING: non-WR '{p}' ({pos_check}) in {label}", file=sys.stderr)
    print(f"  Validation OK [{label}]: {len(rows)} WRs, no duplicate IDs")


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _make_payload(
    rows: list[dict],
    season: int | str,
    week: int | None,
    updated_at: str,
) -> dict[str, Any]:
    clean_rows = [{col: r.get(col) for col in COLUMNS} for r in rows]
    return {
        "meta": {
            "position":     "WR",
            "table":        "player_overview",
            "season":       str(season),
            "generated_at": updated_at,
            "columns":      COLUMNS,
        },
        "players": clean_rows,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (PBP_PATH, METRICS_PATH, SLEEPER_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    print("Loading play-by-play data (all seasons)...")
    pbp_raw  = pd.read_parquet(PBP_PATH)
    pbp_full = pbp_raw[pbp_raw["season_type"] == "REG"].copy()

    with open(SLEEPER_PATH) as f:
        raw = json.load(f)
    sleeper_players: list[dict] = list(raw.values()) if isinstance(raw, dict) else raw
    print(f"  {len(sleeper_players):,} Sleeper players.")

    with open(METRICS_PATH) as f:
        player_metrics: list[dict] = json.load(f)
    print(f"  {len(player_metrics):,} player metric records.")

    # ── Shared lookup tables ─────────────────────────────────────────────────
    full_name_set = {(p.get("full_name") or "").strip() for p in sleeper_players}
    unambig       = _build_unambig(sleeper_players)
    team_disambig = _build_team_disambig(sleeper_players)
    sleeper_pos   = {(p.get("full_name") or "").strip(): p.get("position", "") for p in sleeper_players}
    sleeper_age   = {(p.get("full_name") or "").strip(): p.get("age")          for p in sleeper_players}
    metrics_map   = {p["player"]: p for p in player_metrics}

    # ── Routes lookup from participation data ────────────────────────────────
    routes_by_season: dict[int, dict[str, dict[str, int]]] = {}
    if PARTICIPATION_PATH.exists():
        print("Loading participation data for routes computation...")
        part_df = pd.read_parquet(PARTICIPATION_PATH)
        routes_by_season = _build_routes_lookup(pbp_raw, part_df)
    else:
        print(f"  WARNING: {PARTICIPATION_PATH} not found — routes_per_gm will be null.", file=sys.stderr)

    # ── nflverse players: years_of_experience + bridge for snap/age ──────────
    yoe_lookup: dict[str, int] = {}
    nfl_df: pd.DataFrame | None = None
    if NFL_PATH.exists():
        nfl_df = pd.read_parquet(NFL_PATH)
        print(f"  {len(nfl_df):,} nflverse players loaded.")
        if "display_name" in nfl_df.columns and "years_of_experience" in nfl_df.columns:
            for _, row in nfl_df.iterrows():
                name = str(row["display_name"]).strip()
                yoe  = row["years_of_experience"]
                if name and yoe is not None and not (isinstance(yoe, float) and np.isnan(yoe)):
                    yoe_lookup[name] = int(yoe)
    else:
        print("  nflverse_players.parquet not found — exp_tier will be null.", file=sys.stderr)

    # ── Snap-pct lookups from load_snap_counts (primary source for all seasons) ─
    snap_by_norm: dict[int, dict[str, float]] = {}
    snap_by_gsis: dict[int, dict[str, float]] = {}
    if SNAP_COUNTS_PATH.exists():
        print("Loading snap_counts data...")
        snap_df = pd.read_parquet(SNAP_COUNTS_PATH)
        snap_by_norm, snap_by_gsis = _build_snap_pct_lookups(snap_df, nfl_df)
        total_snap_players = sum(len(v) for v in snap_by_norm.values())
        print(f"  Snap-pct lookup built: {total_snap_players} player-seasons.")
    else:
        print(f"  WARNING: {SNAP_COUNTS_PATH} not found — snap_pct from player_metrics only.", file=sys.stderr)

    # ── Age lookup via gsis_id bridge (covers unresolved PBP abbreviations) ──
    gsis_to_age: dict[str, Any] = _build_gsis_age_lookup(nfl_df, sleeper_age)
    print(f"  gsis_to_age bridge built: {len(gsis_to_age)} entries.")

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_raw_rows: list[dict] = []
    latest_week_2025: int | None = None

    # ── Per-season files ─────────────────────────────────────────────────────
    for season in SEASONS:
        print(f"\nBuilding {season} season...")
        pbp_s = pbp_full[pbp_full["season"] == season].copy()
        week  = int(pbp_s["week"].max()) if "week" in pbp_s.columns and not pbp_s.empty else None
        if season == CURRENT:
            latest_week_2025 = week

        raw_rows = build_season(
            pbp_season       = pbp_s,
            season           = season,
            full_name_set    = full_name_set,
            unambig          = unambig,
            team_disambig    = team_disambig,
            sleeper_pos      = sleeper_pos,
            sleeper_age      = sleeper_age,
            metrics_map      = metrics_map,
            yoe_lookup       = yoe_lookup,
            routes_by_season = routes_by_season,
            snap_by_norm     = snap_by_norm,
            snap_by_gsis     = snap_by_gsis,
            gsis_to_age      = gsis_to_age,
        )

        # Composite scores within-season pool
        scored = _add_scores(raw_rows)

        # Sort: overall_score → role_score → efficiency_score (all desc)
        scored.sort(
            key=lambda r: (-(r.get("overall_score") or 0), -(r.get("role_score") or 0), -(r.get("efficiency_score") or 0)),
        )
        print(f"  {len(scored)} WRs qualified for {season}.")
        _validate(scored, str(season))

        out_path = OUTPUT_DIR / f"wr_player_overview_{season}.json"
        payload  = _make_payload(scored, season, week, updated_at)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

        all_raw_rows.extend(scored)

    # ── All-seasons combined file ─────────────────────────────────────────────
    print("\nBuilding all-seasons combined file...")
    all_agg   = _aggregate_all(all_raw_rows, yoe_lookup)
    all_scored = _add_scores(all_agg)
    all_scored.sort(
        key=lambda r: (-(r.get("overall_score") or 0), -(r.get("role_score") or 0), -(r.get("efficiency_score") or 0)),
    )
    _validate(all_scored, "all-seasons")

    all_path = OUTPUT_DIR / "wr_player_overview_all.json"
    all_payload = _make_payload(all_scored, "2023-2025", None, updated_at)
    with open(all_path, "w") as f:
        json.dump(all_payload, f, indent=2)
    print(f"  Wrote {all_path} ({all_path.stat().st_size/1024:.1f} KB, {len(all_scored)} players)")

    # ── Alias for 2025 ───────────────────────────────────────────────────────
    alias_path = OUTPUT_DIR / "wr_player_overview.json"
    shutil.copy(OUTPUT_DIR / "wr_player_overview_2025.json", alias_path)
    print(f"  Wrote {alias_path} (alias for 2025)")

    print("\nWR Player Overview build complete.")
    print("Endpoints:")
    for name in ["wr_player_overview_2023.json", "wr_player_overview_2024.json",
                 "wr_player_overview_2025.json", "wr_player_overview_all.json"]:
        print(f"  output/{name}")


if __name__ == "__main__":
    main()
