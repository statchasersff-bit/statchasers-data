#!/usr/bin/env python3
"""
build_wr_trends.py

Build wr_trends_2025.json — WR Role & Usage Trends dataset.

Per-player metrics (2025 REG season, WRs only):
  Current season-to-date:
    snap_pct, routes_per_gm, targets_per_gm,
    air_yards_per_gm, adot, rz_tgt, end_zone_tgt

  Recent vs prior 4-game deltas:
    delta_snap_pct, delta_routes_per_gm, delta_targets_per_gm,
    delta_air_yards_per_gm, delta_adot

  Composite / label:
    usage_score (0-100 percentile), role_trend (Rising Fast / … / Insufficient Data)

Output JSON shape:
  { "meta": {...}, "players": [{WrTrendsRow}, ...] }
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT               = Path(__file__).resolve().parent.parent
SCRIPTS_DIR        = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from _id_resolution import build_canonical_id_lookup, filter_canonical_id  # noqa: E402

PBP_PATH           = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
PARTICIPATION_PATH = ROOT / "data" / "raw"       / "nflverse_participation.parquet"
SNAP_PATH          = ROOT / "data" / "raw"       / "nflverse_snap_counts.parquet"
NFL_PATH           = ROOT / "data" / "raw"       / "nflverse_players.parquet"
SLEEPER_PATH       = ROOT / "data" / "raw"       / "sleeper_players.json"
OUTPUT_DIR         = ROOT / "output"

SEASON  = 2025
# Minimum games to include a WR in the dataset
MIN_GAMES = 1
# Windows for delta computation
RECENT_WINDOW = 4
PRIOR_WINDOW  = 4

META_COLUMNS = [
    "player", "team", "games",
    "snap_pct", "delta_snap_pct",
    "routes_per_gm", "delta_routes_per_gm",
    "targets_per_gm", "delta_targets_per_gm",
    "air_yards_per_gm", "delta_air_yards_per_gm",
    "adot", "delta_adot",
    "rz_tgt", "end_zone_tgt",
    "receptions_per_gm",
    "usage_score", "role_score", "role_trend",
]

# ---------------------------------------------------------------------------
# Manual team overrides (keep in sync with other WR scripts)
# ---------------------------------------------------------------------------
_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    "T.Etienne":    {"JAX": "Travis Etienne",    "CAR": "Trevor Etienne"},
    "B.Robinson":   {"ATL": "Bijan Robinson",    "SF": "Brian Robinson", "WAS": "Brian Robinson"},
    # J.Williams: multiple players share this abbreviation; map by team.
    # Javonte Williams moved to DAL in 2025; Sleeper still shows stale team.
    "J.Williams":   {"DEN": "Javonte Williams", "DAL": "Javonte Williams", "NO": "Jamaal Williams"},
    "D.Moore":      {"CHI": "DJ Moore", "BUF": "DJ Moore", "CAR": "David Moore", "TB": "David Moore"},
    "K.Allen":      {"LAC": "Keenan Allen",      "CHI": "Keenan Allen"},
    "D.Montgomery": {"DET": "David Montgomery",  "HOU": "David Montgomery", "IND": "D.J. Montgomery"},
    "K.Williams":   {"LA": "Kyren Williams",     "NE": "Kyle Williams",    "CIN": "Ke'Shawn Williams"},
    "R.White":      {"TB": "Rachaad White",      "WAS": "Rachaad White",   "SEA": "Ricky White"},
    "J.Smith":      {"ATL": "Jonnu Smith",       "MIA": "Jonnu Smith",     "PIT": "Jonnu Smith"},
    "Mi.Wilson":    {"ARI": "Michael Wilson"},
    "K.Juszczyk":   {"SF": "Kyle Juszczyk"},
    # Multi-word last name — nflverse uses A.St. Brown, _abbrev() produces A.Brown
    "A.St. Brown":  {"DET": "Amon-Ra St. Brown"},
    # Non-WR receivers whose Sleeper team field is stale (None) or uses a different
    # team code than nflverse PBP, causing the pos-based fallback to pick the wrong
    # WR with the same abbreviation.
    "M.Carter":     {"ARI": "Michael Carter"},                      # RB/ARI, not Malachi Carter
    "D.Allen":      {"LA": "Davis Allen", "LAR": "Davis Allen"},    # TE/LAR, not Devon Allen
    "J.Ford":       {"CLE": "Jerome Ford"},                         # RB/CLE, not Jacoby Ford
    # Ji.Horn: nflverse uses 2-char prefix for Jimmy Horn (CAR). "J.Horn" is ambiguous.
    "Ji.Horn":      {"CAR": "Jimmy Horn"},
    # K.Walker: Sleeper has two "Kenneth Walker" players — one is a WR (team=None).
    # The RB K.Walker+SEA (Kenneth Walker III) must not resolve to the WR identity.
    # Mapping to "Kenneth Walker III" ensures he falls outside sleeper_wr_set.
    "K.Walker":     {"SEA": "Kenneth Walker III"},
}

# PFR snap-counts name → Sleeper canonical name aliases
_SNAP_PLAYER_ALIASES: dict[str, str] = {
    "Gabriel Davis":       "Gabe Davis",
    "Michael Pittman Jr.": "Michael Pittman",
    "Marvin Harrison Jr.": "Marvin Harrison",
    "Calvin Austin III":   "Calvin Austin",
    "Josh Palmer":         "Joshua Palmer",
}


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

def _abbrev(full_name: str) -> str:
    """'CeeDee Lamb' → 'C.Lamb'  (nflverse PBP abbreviation format)."""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _ns(name: str) -> str:
    """Snap-norm: strip all punctuation and spaces, lowercase.
    'D.J. Moore' → 'djmoore', 'Amon-Ra St. Brown' → 'amonrastbrown'
    """
    return re.sub(r"[.\-\s']", "", name).lower()


def _build_unambig(sleeper_players: list[dict]) -> dict[str, str]:
    counts: dict[str, int]  = {}
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
    counts: dict[str, int]   = {}
    by_team: dict[str, dict] = {}
    for p in sleeper_players:
        full = (p.get("full_name") or "").strip()
        if not full:
            continue
        ab = _abbrev(full)
        counts[ab] = counts.get(ab, 0) + 1
        by_team.setdefault(ab, {})[p.get("team")] = full
    return {ab: t for ab, t in by_team.items() if counts[ab] > 1}


def _resolve(
    pbp_name: str,
    pbp_team: str,
    full_name_set: set[str],
    unambig: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    sleeper_pos: dict[str, str],
) -> str | None:
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
    return None


# ---------------------------------------------------------------------------
# Per-game routes lookup  {gsis_id → {game_id → route_count}}
# ---------------------------------------------------------------------------

def _build_per_game_routes(season: int) -> dict[str, dict[str, int]]:
    """Explode participation parquet → {gsis_id → {game_id → WR route count}}.

    Participation rows are inner-joined to pbp on (game_id, play_id) and
    filtered to play_type == 'pass' so we count routes only on pass plays.
    """
    part = pd.read_parquet(
        PARTICIPATION_PATH,
        columns=["nflverse_game_id", "play_id", "season",
                 "offense_players", "offense_positions"],
    )
    part = part[
        (part["season"] == season) &
        part["offense_players"].notna() &
        part["offense_positions"].notna()
    ].copy()

    # Filter to regular-season weeks via game_id ("2025_03_BUF_NE" → week 3)
    part["week_num"] = part["nflverse_game_id"].str.split("_").str[1].astype(int)
    part = part[part["week_num"] <= 18]

    pbp_pass = pd.read_parquet(
        PBP_PATH,
        columns=["game_id", "play_id", "season", "season_type", "play_type"],
    )
    pbp_pass = pbp_pass[
        (pbp_pass["season"] == season) &
        (pbp_pass["season_type"] == "REG") &
        (pbp_pass["play_type"] == "pass")
    ][["game_id", "play_id"]]
    part = part.merge(
        pbp_pass,
        left_on=["nflverse_game_id", "play_id"],
        right_on=["game_id", "play_id"],
        how="inner",
    )

    part["pid_list"] = part["offense_players"].str.split(";")
    part["pos_list"] = part["offense_positions"].str.split(";")
    exp = (
        part[["nflverse_game_id", "pid_list", "pos_list"]]
        .explode(["pid_list", "pos_list"])
        .rename(columns={"pid_list": "gsis_id", "pos_list": "position"})
    )
    exp["gsis_id"]  = exp["gsis_id"].str.strip()
    exp["position"] = exp["position"].str.strip()

    wr = exp[exp["position"] == "WR"].copy()
    wr["game_id"] = wr["nflverse_game_id"]

    out: dict[str, dict[str, int]] = {}
    for (gsis_id, game_id), cnt in wr.groupby(["gsis_id", "game_id"]).size().items():
        out.setdefault(gsis_id, {})[game_id] = int(cnt)
    return out


# ---------------------------------------------------------------------------
# Per-game snap-pct lookup  {canonical → {game_id → snap_pct}}
# ---------------------------------------------------------------------------

def _build_per_game_snaps(
    season: int,
    canonical_wr_set: set[str],
) -> dict[str, dict[str, float]]:
    """Build {canonical WR name → {game_id → offense_pct}} from snap_counts."""
    snaps = pd.read_parquet(SNAP_PATH)
    snaps = snaps[
        (snaps["season"] == season) &
        (snaps["week"] <= 18) &
        (snaps["position"] == "WR")
    ].copy()

    # Build norm index: {_ns(canonical) → canonical}
    ns_to_canonical: dict[str, str] = {_ns(c): c for c in canonical_wr_set}

    # Build per-game lookup
    out: dict[str, dict[str, float]] = {}
    for _, row in snaps.iterrows():
        pfr_name  = str(row["player"])
        game_id   = str(row["game_id"])
        pct       = row["offense_pct"]
        if pd.isna(pct):
            continue

        # Apply alias then normalize to find canonical
        aliased = _SNAP_PLAYER_ALIASES.get(pfr_name, pfr_name)
        canonical = (
            ns_to_canonical.get(_ns(aliased))
            or ns_to_canonical.get(_ns(pfr_name))
        )
        if not canonical:
            continue
        out.setdefault(canonical, {})[game_id] = float(pct)

    return out


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def _week_of(game_id: str) -> int:
    """Extract week number from nflverse game_id, e.g. '2025_03_BUF_NE' → 3."""
    try:
        return int(game_id.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _window_avg(game_ids: list[str], data: dict[str, float]) -> float | None:
    """Average of data values for the given game_ids, ignoring missing."""
    vals = [data[g] for g in game_ids if g in data]
    return round(float(sum(vals) / len(vals)), 4) if vals else None


def _window_sum(game_ids: list[str], data: dict[str, int]) -> int:
    return sum(data.get(g, 0) for g in game_ids)


def _delta(recent: float | None, prior: float | None) -> float | None:
    if recent is None or prior is None:
        return None
    return round(recent - prior, 4)


# ---------------------------------------------------------------------------
# NaN / Inf cleanup for JSON serialisation
# ---------------------------------------------------------------------------

def _clean_rows(rows: list[dict]) -> list[dict]:
    """Replace float NaN/Inf with None so json.dump produces valid JSON."""
    cleaned = []
    for row in rows:
        cleaned.append({
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in row.items()
        })
    return cleaned


# Usage score (0-100 percentile composite)
# ---------------------------------------------------------------------------

_USAGE_WEIGHTS = [
    ("targets_per_gm",    0.35),
    ("snap_pct",          0.30),
    ("air_yards_per_gm",  0.20),
    ("receptions_per_gm", 0.15),
]


def _add_usage_scores(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    score = pd.Series(0.0, index=df.index)
    for col, w in _USAGE_WEIGHTS:
        if col not in df.columns:
            continue
        ranks = df[col].rank(pct=True, na_option="bottom")
        score += ranks * w
    df["usage_score"] = (score * 100).round(1)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Role trend label
# ---------------------------------------------------------------------------

def _role_trend(
    games: int,
    dtgt: float | None,
    droutes: float | None,
    dair: float | None,
    dadot: float | None,
    dsnap: float | None,
) -> str:
    needed = RECENT_WINDOW + PRIOR_WINDOW
    if games < needed or dtgt is None:
        return "Insufficient Data"

    pos_moves = sum(1 for v in (droutes, dair, dsnap) if v is not None and v > 0)
    neg_moves = sum(1 for v in (droutes, dair, dsnap) if v is not None and v < 0)

    if dtgt >= 1.5 and pos_moves >= 2:
        return "Rising Fast"
    if dtgt >= 0.5 and pos_moves >= 1:
        return "Trending Up"
    if dtgt <= -1.5 and neg_moves >= 2:
        return "Falling Fast"
    if dtgt <= -0.5 and neg_moves >= 1:
        return "Trending Down"
    # Boom/bust: air yards or aDOT rising but targets not improving
    if dair is not None and dair > 2 and dtgt < 0.5:
        return "Boom/Bust Role"
    if dadot is not None and dadot > 1.5 and dtgt < 0.5:
        return "Boom/Bust Role"
    return "Stable"


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build() -> list[dict[str, Any]]:
    print(f"[{SEASON}] Loading play-by-play …")
    pbp_cols = [
        "season", "season_type", "week", "game_id", "posteam",
        "receiver_player_name", "receiver_player_id",
        "complete_pass", "pass_attempt", "two_point_attempt",
        "air_yards", "yardline_100", "epa",
    ]
    pbp = pd.read_parquet(PBP_PATH, columns=pbp_cols)
    pbp = pbp[
        (pbp["season"] == SEASON) &
        (pbp["season_type"] == "REG") &
        (pbp["week"] <= 18) &
        (pbp["two_point_attempt"].fillna(0) != 1)
    ].copy()

    tgts = pbp[
        (pbp["pass_attempt"] == 1) &
        pbp["receiver_player_name"].notna()
    ].copy()

    print(f"[{SEASON}] Loading Sleeper players …")
    with open(SLEEPER_PATH) as f:
        raw = json.load(f)
    sleeper_players = list(raw.values()) if isinstance(raw, dict) else raw

    sleeper_wr_set   = {p["full_name"] for p in sleeper_players
                        if p.get("position") == "WR" and p.get("full_name")}
    sleeper_full_set = {p["full_name"] for p in sleeper_players
                        if p.get("full_name")}
    sleeper_pos: dict[str, str] = {
        p["full_name"]: p.get("position", "")
        for p in sleeper_players if p.get("full_name")
    }

    unambig       = _build_unambig(sleeper_players)
    team_disambig = _build_team_disambig(sleeper_players)

    # Resolve each (abbrev, team) pair in PBP
    pairs = tgts[["receiver_player_name", "posteam"]].drop_duplicates().values.tolist()
    pair_to_canonical: dict[tuple[str, str], str | None] = {}
    for aname, team in pairs:
        pair_to_canonical[(aname, team)] = _resolve(
            aname, team, sleeper_full_set, unambig, team_disambig, sleeper_pos
        )

    tgts["canonical"] = tgts.apply(
        lambda r: pair_to_canonical.get((r["receiver_player_name"], r["posteam"])),
        axis=1,
    )

    # WR filter via Sleeper position
    tgts = tgts[tgts["canonical"].apply(
        lambda c: c in sleeper_wr_set if c else False
    )].copy()

    # Drop name-collision plays via canonical gsis_id
    nfl_df = pd.read_parquet(NFL_PATH, columns=["gsis_id", "display_name", "position"])
    canonical_id_lookup = build_canonical_id_lookup(nfl_df, position="WR")
    tgts = filter_canonical_id(tgts, "canonical", "receiver_player_id", canonical_id_lookup)

    print(f"[{SEASON}] Building per-game routes …")
    per_game_routes = _build_per_game_routes(SEASON)

    print(f"[{SEASON}] Building per-game snaps …")
    per_game_snaps = _build_per_game_snaps(SEASON, sleeper_wr_set)

    # Per-game PBP stats per canonical WR
    # Mark red-zone (≤20 yards from goal line) and end-zone (≤10 yards)
    tgts["is_rz"] = tgts["yardline_100"] <= 20
    tgts["is_ez"] = tgts["yardline_100"] <= 10

    print(f"[{SEASON}] Aggregating per-game stats …")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for canonical, grp in tgts.groupby("canonical"):
        if canonical in seen:
            continue
        seen.add(canonical)

        # Team — most common posteam
        team = grp["posteam"].mode().iloc[0]

        # Games-with-target — narrow definition; matches stat explorer /
        # overview / efficiency / advanced_stats so the dashboard's GP column
        # is internally consistent across tabs.
        target_game_ids = set(grp["game_id"].unique())
        games = len(target_game_ids)
        if games < MIN_GAMES:
            continue

        # Trend-line window: broader — include route-only games so a player
        # who was on the field but not targeted still shows up as a 0-target
        # entry on the trend, not a gap.
        gsis_id = (
            grp["receiver_player_id"].dropna().mode().iloc[0]
            if grp["receiver_player_id"].notna().any() else None
        )
        all_game_ids = sorted(target_game_ids, key=_week_of)
        if gsis_id and gsis_id in per_game_routes:
            route_game_ids = set(per_game_routes[gsis_id].keys())
            all_game_ids = sorted(
                target_game_ids | route_game_ids,
                key=_week_of,
            )

        # ---- Per-game dicts ----
        game_targets:   dict[str, int]   = {}
        game_air_yards: dict[str, float] = {}
        game_rz_tgt:    dict[str, int]   = {}
        game_ez_tgt:    dict[str, int]   = {}
        game_recs:      dict[str, int]   = {}

        for game_id, g in grp.groupby("game_id"):
            game_targets[game_id]   = len(g)
            ays = g["air_yards"].dropna().sum()
            game_air_yards[game_id] = float(ays) if not np.isnan(ays) else 0.0
            game_rz_tgt[game_id]    = int(g["is_rz"].sum())
            game_ez_tgt[game_id]    = int(g["is_ez"].sum())
            game_recs[game_id]      = int(g["complete_pass"].sum())

        game_routes: dict[str, int] = (
            per_game_routes.get(gsis_id, {}) if gsis_id else {}
        )
        game_snaps: dict[str, float] = per_game_snaps.get(canonical, {})

        # ---- Full-season aggregates ----
        total_targets   = sum(game_targets.values())
        total_air_yards = sum(game_air_yards.values())
        total_rz        = sum(game_rz_tgt.values())
        total_ez        = sum(game_ez_tgt.values())
        total_routes    = sum(game_routes.values())
        total_recs      = sum(game_recs.values())
        snap_vals       = [game_snaps.get(g) for g in all_game_ids if g in game_snaps]

        targets_per_gm   = round(total_targets / games, 2)  if games > 0 else None
        air_yards_per_gm = round(total_air_yards / games, 2) if games > 0 else None
        routes_per_gm    = round(total_routes / games, 2)    if total_routes > 0 and games > 0 else None
        receptions_per_gm = round(total_recs / games, 2)    if games > 0 else None
        snap_pct         = round(float(sum(snap_vals) / len(snap_vals)), 4) if snap_vals else None

        # ADOT = total target air yards / total targets (season)
        adot_vals = grp["air_yards"].dropna()
        adot = round(float(adot_vals.sum() / len(grp)), 2) if len(grp) > 0 else None

        # ---- Recent vs prior deltas ----
        recent_games = all_game_ids[-RECENT_WINDOW:]   if len(all_game_ids) >= RECENT_WINDOW else []
        prior_games  = all_game_ids[-(RECENT_WINDOW + PRIOR_WINDOW):-RECENT_WINDOW] \
                       if len(all_game_ids) >= RECENT_WINDOW + PRIOR_WINDOW \
                       else all_game_ids[:-RECENT_WINDOW] if len(all_game_ids) >= RECENT_WINDOW + 1 \
                       else []

        def _avg_per_game(game_ids: list[str], data: dict) -> float | None:
            vals = [data[g] for g in game_ids if g in data]
            if not vals:
                return None
            return round(float(sum(vals) / len(game_ids)), 4)

        def _adot_window(game_ids: list[str]) -> float | None:
            sub = grp[grp["game_id"].isin(game_ids)]
            ay  = sub["air_yards"].dropna()
            return round(float(ay.sum() / len(sub)), 2) if len(sub) > 0 else None

        # Window averages per game (not per appearance)
        r_tgt   = _avg_per_game(recent_games, game_targets)
        p_tgt   = _avg_per_game(prior_games,  game_targets)
        r_air   = _avg_per_game(recent_games, game_air_yards)
        p_air   = _avg_per_game(prior_games,  game_air_yards)
        r_rte   = _avg_per_game(recent_games, game_routes)
        p_rte   = _avg_per_game(prior_games,  game_routes)
        r_snap  = _window_avg(recent_games, game_snaps)
        p_snap  = _window_avg(prior_games,  game_snaps)
        r_adot  = _adot_window(recent_games)
        p_adot  = _adot_window(prior_games)

        has_both_windows = bool(recent_games) and bool(prior_games)

        delta_tgt   = _delta(r_tgt,  p_tgt)  if has_both_windows else None
        delta_air   = _delta(r_air,  p_air)  if has_both_windows else None
        delta_rte   = _delta(r_rte,  p_rte)  if has_both_windows else None
        delta_snap  = _delta(r_snap, p_snap) if has_both_windows else None
        delta_adot  = _delta(r_adot, p_adot) if has_both_windows else None

        trend = _role_trend(games, delta_tgt, delta_rte, delta_air, delta_adot, delta_snap)

        rows.append({
            "player":  canonical,
            "team":    team,
            "games":   games,
            "snap_pct":              snap_pct,
            "delta_snap_pct":        round(delta_snap, 4) if delta_snap is not None else None,
            "routes_per_gm":         routes_per_gm,
            "delta_routes_per_gm":   round(delta_rte, 2) if delta_rte is not None else None,
            "targets_per_gm":        targets_per_gm,
            "delta_targets_per_gm":  round(delta_tgt, 2) if delta_tgt is not None else None,
            "air_yards_per_gm":      air_yards_per_gm,
            "delta_air_yards_per_gm": round(delta_air, 2) if delta_air is not None else None,
            "adot":               adot,
            "delta_adot":         round(delta_adot, 2) if delta_adot is not None else None,
            "rz_tgt":             total_rz,
            "end_zone_tgt":       total_ez,
            "receptions_per_gm":  receptions_per_gm,
            "usage_score":        None,   # filled below
            "role_score":         None,   # patched from overview
            "role_trend":         trend,
        })

    print(f"[{SEASON}] Built {len(rows)} WR rows — adding usage scores …")
    rows = _add_usage_scores(rows)

    # Sort: usage_score desc, delta_targets_per_gm desc, delta_air_yards_per_gm desc
    rows.sort(
        key=lambda r: (
            -(r["usage_score"] or 0),
            -(r["delta_targets_per_gm"] or 0),
            -(r["delta_air_yards_per_gm"] or 0),
        )
    )

    # Validation
    assert all(r["player"] for r in rows), "Empty player name found"
    assert len({r["player"] for r in rows}) == len(rows), "Duplicate players found"
    for r in rows:
        for col in META_COLUMNS:
            assert col in r, f"Missing column: {col}"

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    players = build()

    # ── Patch usage_score from player overview (single source of truth) ──────
    overview_path = OUTPUT_DIR / f"wr_player_overview_{SEASON}.json"
    if overview_path.exists():
        with open(overview_path) as _f:
            _ov = json.load(_f)
        _ov_rows = _ov.get("players", [])
        # Two-pass lookup: exact name, then (team, last_name) fallback
        _ov_exact_usage: dict[str, float | None] = {
            p["player"]: p.get("usage_score") for p in _ov_rows
        }
        _ov_exact_role: dict[str, float | None] = {
            p["player"]: p.get("role_score") for p in _ov_rows
        }
        _ov_team_last_usage: dict[tuple, float | None] = {
            (p["team"], p["player"].split()[-1]): p.get("usage_score")
            for p in _ov_rows
        }
        _ov_team_last_role: dict[tuple, float | None] = {
            (p["team"], p["player"].split()[-1]): p.get("role_score")
            for p in _ov_rows
        }
        patched = 0
        role_patched = 0
        for p in players:
            key = (p.get("team"), p["player"].split()[-1])
            # usage_score: only overwrite the locally-computed value when the
            # overview has an explicit non-null entry (overview currently does
            # not compute usage_score, so guarding prevents it from wiping the
            # rank-based score calculated by _add_usage_scores above).
            ov_usage = _ov_exact_usage.get(p["player"])
            if ov_usage is None:
                ov_usage = _ov_team_last_usage.get(key)
            if ov_usage is not None:
                p["usage_score"] = ov_usage
                patched += 1
            if p["player"] in _ov_exact_role:
                p["role_score"] = _ov_exact_role[p["player"]]
                role_patched += 1
            elif key in _ov_team_last_role:
                p["role_score"] = _ov_team_last_role[key]
                role_patched += 1
        players.sort(key=lambda p: p.get("role_score") or 0.0, reverse=True)
        print(f"[{SEASON}] Patched usage_score from player overview for {patched} WRs")
        print(f"[{SEASON}] Patched role_score from player overview for {role_patched} WRs")
    else:
        print(f"[{SEASON}] WARN: {overview_path} not found — using analytics-computed usage_score")

    # ── NaN → None before serialisation ──────────────────────────────────────
    players = _clean_rows(players)

    out = {
        "meta": {
            "position": "WR",
            "table": "role_usage_trends",
            "season": str(SEASON),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "columns": META_COLUMNS,
        },
        "players": players,
    }

    # Validate meta columns match exactly
    assert out["meta"]["columns"] == META_COLUMNS

    out_path = OUTPUT_DIR / f"wr_trends_{SEASON}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{SEASON}] Wrote {len(players)} rows → {out_path}")


if __name__ == "__main__":
    main()
