#!/usr/bin/env python3
"""
build_wr_efficiency_analytics.py

Build wr_efficiency_analytics_{season}.json for the WR Efficiency Analytics tab.

Metrics produced per WR (15+ targets, REG season):
  Core    : epa_per_target, success_rate, yards_per_target, catch_rate, fpoe
  Routes  : yards_per_route_run, targets_per_route_run
  Air yards: air_yards_per_target, air_yards_share_pct
  Explosive: explosive_play_rate, explosive_rec_20_plus, explosive_rec_40_plus,
             longest_reception
  Contact : yac_per_rec, ybc_per_rec, broken_tackles, btkl_per_rec
  Composite: wr_efficiency_score (0-100, percentile-weighted)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT               = Path(__file__).resolve().parent.parent
PBP_PATH           = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
PARTICIPATION_PATH = ROOT / "data" / "raw"       / "nflverse_participation.parquet"
METRICS_PATH       = ROOT / "data" / "processed" / "player_metrics.json"
NFL_PATH           = ROOT / "data" / "raw"       / "nflverse_players.parquet"
SLEEPER_PATH       = ROOT / "data" / "raw"       / "sleeper_players.json"
PFR_REC_PATH       = ROOT / "data" / "raw"       / "pfr_rec_advstats.parquet"
OUTPUT_DIR         = ROOT / "output"

SEASONS     = [2025]
MIN_TARGETS = 15

COLUMNS = [
    "player", "team", "games", "targets",
    "epa_per_target", "success_rate", "yards_per_target", "catch_rate", "fpoe",
    "yards_per_route_run", "targets_per_route_run", "air_yards_per_target",
    "air_yards_share_pct",
    "explosive_play_rate", "explosive_rec_20_plus", "explosive_rec_40_plus",
    "longest_reception",
    "yac_per_rec", "ybc_per_rec", "broken_tackles", "btkl_per_rec",
    "wr_efficiency_score",
]

# ---------------------------------------------------------------------------
# Name helpers — kept in sync with build_wr_player_overview.py
# ---------------------------------------------------------------------------

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
    "Mi.Wilson":    {"ARI": "Michael Wilson"},
    "K.Juszczyk":   {"SF": "Kyle Juszczyk"},
    # nflverse abbreviates "Amon-Ra St. Brown" as "A.St. Brown" (multi-word last name).
    # _abbrev("Amon-Ra St. Brown") produces "A.Brown" which collides with A.J. Brown,
    # so we handle this manually.
    "A.St. Brown":  {"DET": "Amon-Ra St. Brown"},
    # Non-WR receivers whose Sleeper team field is stale (None) or uses a different
    # team code than nflverse PBP, causing the pos-based fallback to pick the wrong
    # WR with the same abbreviation.  Explicit team→canonical locks the correct player
    # so the WR position filter can then exclude them cleanly.
    "M.Carter":     {"ARI": "Michael Carter"},                      # RB/ARI, not Malachi Carter
    "D.Allen":      {"LA": "Davis Allen", "LAR": "Davis Allen"},    # TE/LAR, not Devon Allen
    "J.Ford":       {"CLE": "Jerome Ford"},                         # RB/CLE, not Jacoby Ford
}

# PFR sometimes uses a different given name than Sleeper/PBP (nickname vs full).
# Keys are _norm(canonical_name), values are _norm(pfr_player_name).
_CANONICAL_TO_PFR_NORM: dict[str, str] = {
    "joshua palmer": "josh palmer",
}


def _norm(name: str) -> str:
    """Lowercase, strip non-alpha, collapse whitespace — for PFR name matching."""
    if not name:
        return ""
    n = re.sub(r"[^a-z\s]", "", name.lower())
    return re.sub(r"\s+", " ", n).strip()


def _abbrev(full_name: str) -> str:
    """'CeeDee Lamb' → 'C.Lamb'  (matches nflverse PBP abbreviation format)."""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _build_unambig(sleeper_players: list[dict]) -> dict[str, str]:
    """abbrev → full_name when unambiguous across ALL positions."""
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


def _build_team_disambig(
    sleeper_players: list[dict],
) -> dict[str, dict[str, str]]:
    """abbrev → {team → full_name} for ambiguous abbreviations."""
    counts: dict[str, int]         = {}
    by_team: dict[str, dict]       = {}
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
    """Resolve PBP abbreviation → canonical Sleeper full name."""
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
# Routes lookup (participation parquet → {gsis_id: {season: routes}})
# ---------------------------------------------------------------------------

def _build_routes_lookup(season: int) -> dict[str, int]:
    """Return {gsis_id → total routes run} for one season, REG weeks ≤ 18.

    Participation parquet schema:
        nflverse_game_id, play_id, season,
        offense_players  (semicolon-delimited gsis_ids),
        offense_positions (semicolon-delimited positions),
        offense_names    (semicolon-delimited names)
    Week is parsed from game_id (e.g. '2025_03_BUF_NE' → week 3).
    A WR appearance on any pass play counts as one route run.
    """
    part = pd.read_parquet(
        PARTICIPATION_PATH,
        columns=["nflverse_game_id", "season",
                 "offense_players", "offense_positions"],
    )
    part = part[
        (part["season"] == season) &
        part["offense_players"].notna() &
        part["offense_positions"].notna()
    ].copy()

    # Parse week number from game_id string ("2025_03_BUF_NE" → 3)
    part["week_num"] = (
        part["nflverse_game_id"].str.split("_").str[1].astype(int)
    )
    part = part[part["week_num"] <= 18]

    # Explode semicolon-separated player/position lists
    part["pid_list"] = part["offense_players"].str.split(";")
    part["pos_list"] = part["offense_positions"].str.split(";")
    exp = (
        part[["pid_list", "pos_list"]]
        .explode(["pid_list", "pos_list"])
        .rename(columns={"pid_list": "gsis_id", "pos_list": "position"})
    )
    exp["gsis_id"]  = exp["gsis_id"].str.strip()
    exp["position"] = exp["position"].str.strip()

    # Count WR appearances (each appearance = 1 route)
    wr = exp[exp["position"] == "WR"]
    return wr.groupby("gsis_id").size().to_dict()


# ---------------------------------------------------------------------------
# PFR broken-tackles lookup
# ---------------------------------------------------------------------------

def _norm_pfr(name: str) -> str:
    """Normalise PFR full name for fuzzy matching."""
    return _norm(name)


_BTKL_NORM_SUFFIXES = frozenset(["jr", "sr", "ii", "iii", "iv", "v"])


def _strip_btkl_suffixes(norm: str) -> str:
    """'marvin harrison jr' → 'marvin harrison', 'calvin austin iii' → 'calvin austin'."""
    parts = norm.split()
    while parts and parts[-1] in _BTKL_NORM_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _build_btkl_lookup(season: int) -> dict[str, float]:
    """Return {norm_full_name → total receiving_broken_tackles} for season.

    The dict is augmented with suffix-stripped aliases so that Sleeper canonical
    names without generational suffixes (e.g. 'Marvin Harrison') match PFR names
    that include them (e.g. 'Marvin Harrison Jr.').
    """
    pfr = pd.read_parquet(PFR_REC_PATH)
    pfr = pfr[pfr["season"] == season].copy()
    pfr["_nname"] = pfr["pfr_player_name"].fillna("").apply(_norm_pfr)
    agg: dict[str, float] = (
        pfr.groupby("_nname")["receiving_broken_tackles"]
        .sum()
        .to_dict()
    )
    # Add suffix-stripped aliases (only when the stripped key doesn't already exist).
    stripped_aliases = {
        _strip_btkl_suffixes(k): v
        for k, v in agg.items()
        if _strip_btkl_suffixes(k) != k and _strip_btkl_suffixes(k) not in agg
    }
    agg.update(stripped_aliases)
    return agg


# ---------------------------------------------------------------------------
# fpoe lookup from player_metrics.json
# ---------------------------------------------------------------------------

def _build_fpoe_lookup() -> tuple[dict[str, float], dict[str, float]]:
    """Return (raw_name_lookup, norm_name_lookup) dicts from player_metrics.json.

    player_metrics has no season field and uses a mix of Sleeper full names
    (e.g. 'CeeDee Lamb') and PBP-style abbreviations (e.g. 'A.St. Brown').
    We build both a direct and a normalized index so both styles can be matched.
    """
    with open(METRICS_PATH) as f:
        raw = json.load(f)
    data = raw if isinstance(raw, list) else raw.get("players", [])
    by_name: dict[str, float] = {}
    by_norm: dict[str, float] = {}
    for p in data:
        name = (p.get("player") or "").strip()
        val  = p.get("fpoe")
        if not name or val is None:
            continue
        fval = float(val)
        by_name[name]        = fval
        by_norm[_norm(name)] = fval
    return by_name, by_norm


# ---------------------------------------------------------------------------
# Team air-yards totals (denominator for air_yards_share_pct)
# ---------------------------------------------------------------------------

def _build_team_air_yards(pbp: pd.DataFrame) -> dict[str, float]:
    """Sum all target air_yards per team from the full PBP."""
    tgt = pbp[pbp["pass_attempt"] == 1].copy()
    return tgt.groupby("posteam")["air_yards"].sum().to_dict()


# ---------------------------------------------------------------------------
# WR efficiency score (0–100 composite from percentile ranks)
# ---------------------------------------------------------------------------

_SCORE_COMPONENTS = [
    ("epa_per_target",   0.25),
    ("yards_per_route_run", 0.25),
    ("success_rate",     0.20),
    ("yards_per_target", 0.15),
    ("explosive_play_rate", 0.10),
    ("btkl_per_rec",     0.05),
]


def _add_efficiency_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add wr_efficiency_score column (0–100) using percentile ranks."""
    df = df.copy()
    score = pd.Series(0.0, index=df.index)
    for col, weight in _SCORE_COMPONENTS:
        if col not in df.columns:
            continue
        # pct=True gives rank in [0,1]; NaN rows get NaN rank → fill with 0
        ranks = df[col].rank(pct=True, na_option="bottom")
        score += ranks * weight
    df["wr_efficiency_score"] = (score * 100).round(1)
    return df


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_season(season: int) -> list[dict[str, Any]]:
    print(f"[{season}] Loading play-by-play …")
    pbp_cols = [
        "season", "season_type", "week", "game_id", "posteam",
        "receiver_player_name", "receiver_player_id",
        "complete_pass", "pass_attempt",
        "yards_gained", "air_yards", "yards_after_catch", "epa",
    ]
    pbp = pd.read_parquet(PBP_PATH, columns=pbp_cols)
    pbp = pbp[
        (pbp["season"] == season) &
        (pbp["season_type"] == "REG") &
        (pbp["week"] <= 18)
    ].copy()

    # Only target plays (pass_attempt with a receiver)
    tgts = pbp[
        (pbp["pass_attempt"] == 1) &
        pbp["receiver_player_name"].notna()
    ].copy()

    print(f"[{season}] Loading Sleeper players …")
    with open(SLEEPER_PATH) as f:
        raw = json.load(f)
    sleeper_players = list(raw.values()) if isinstance(raw, dict) else raw
    # WR-only position set from Sleeper (authoritative)
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

    print(f"[{season}] Loading routes, fpoe, broken-tackles …")
    routes_by_gsis              = _build_routes_lookup(season)
    btkl_by_name                = _build_btkl_lookup(season)
    fpoe_by_name, fpoe_by_norm  = _build_fpoe_lookup()
    team_air_yards              = _build_team_air_yards(tgts)

    # nflverse player registry: gsis_id → full_name
    nfl = pd.read_parquet(NFL_PATH, columns=["gsis_id", "display_name"])
    nfl = nfl.dropna(subset=["gsis_id", "display_name"])
    gsis_to_nfl_name: dict[str, str] = dict(
        zip(nfl["gsis_id"], nfl["display_name"])
    )

    # Resolve abbrev → canonical for each unique (abbrev, team) pair in PBP.
    # Key is (abbrev, team) so ambiguous names are handled per-team.
    abbrev_team_pairs = (
        tgts[["receiver_player_name", "posteam"]]
        .drop_duplicates()
        .values.tolist()
    )
    pair_to_canonical: dict[tuple[str, str], str | None] = {}
    for aname, team in abbrev_team_pairs:
        pair_to_canonical[(aname, team)] = _resolve(
            aname, team, sleeper_full_set, unambig, team_disambig, sleeper_pos
        )

    tgts["canonical"] = tgts.apply(
        lambda r: pair_to_canonical.get(
            (r["receiver_player_name"], r["posteam"])
        ),
        axis=1,
    )

    # Filter to WRs only via Sleeper position (drop FB/TE/RB if Sleeper knows them)
    def _is_wr(canonical: str | None) -> bool:
        if not canonical:
            return False
        if canonical in sleeper_wr_set:
            return True
        # If name is in Sleeper but NOT a WR, exclude
        if canonical in sleeper_full_set:
            return False
        return False  # Unknown → exclude conservatively

    tgts = tgts[tgts["canonical"].apply(_is_wr)].copy()

    # Deduplicate receiver_player_id → use most common canonical per gsis_id
    # (ensures routes join works correctly)
    gsis_to_canonical: dict[str, str] = {}
    for gsis, grp in tgts.groupby("receiver_player_id"):
        if pd.isna(gsis):
            continue
        gsis_to_canonical[gsis] = grp["canonical"].mode().iloc[0]

    # -----------------------------------------------------------------------
    # Aggregate per canonical player
    # -----------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for canonical, grp in tgts.groupby("canonical"):
        if canonical in seen:
            continue
        seen.add(canonical)

        # Team — use most common posteam
        team = grp["posteam"].mode().iloc[0]

        # Games
        games = grp["game_id"].nunique()

        # Targets
        targets = len(grp)
        if targets < MIN_TARGETS:
            continue

        # Completions sub-frame
        comps = grp[grp["complete_pass"] == 1]
        rec   = len(comps)

        # ---- Core metrics ----
        epa_valid = grp["epa"].dropna()
        epa_per_target = (
            round(float(epa_valid.sum() / len(epa_valid)), 3)
            if len(epa_valid) > 0 else None
        )
        success_rate = (
            round(float((epa_valid > 0).sum() / len(epa_valid)), 4)
            if len(epa_valid) > 0 else None
        )
        yards_per_target = (
            round(float(comps["yards_gained"].sum() / targets), 2)
            if targets > 0 else None
        )
        catch_rate = round(float(rec / targets), 4) if targets > 0 else None

        # fpoe — try canonical → _abbrev(canonical) → normalized fallback
        def _get_fpoe(name: str) -> float | None:
            for key in (name, _abbrev(name), _norm(name)):
                d = fpoe_by_norm if key == _norm(name) else fpoe_by_name
                v = d.get(key)
                if v is not None:
                    return v
            return None

        fpoe_val = _get_fpoe(canonical)
        fpoe = round(float(fpoe_val), 2) if fpoe_val is not None else None

        # ---- Routes ----
        # Get gsis_id for this player (most frequent in group)
        gsis_id = (
            grp["receiver_player_id"]
            .dropna()
            .mode()
            .iloc[0]
            if grp["receiver_player_id"].notna().any()
            else None
        )
        routes = routes_by_gsis.get(gsis_id, 0) if gsis_id else 0

        yards_per_route_run = (
            round(float(comps["yards_gained"].sum() / routes), 3)
            if routes > 0 else None
        )
        targets_per_route_run = (
            round(float(targets / routes), 3)
            if routes > 0 else None
        )

        # ---- Air yards ----
        air_total = grp["air_yards"].sum()
        air_yards_per_target = (
            round(float(air_total / targets), 2)
            if targets > 0 and not np.isnan(air_total) else None
        )
        team_total_air = team_air_yards.get(team, 0)
        air_yards_share_pct = (
            round(float(air_total / team_total_air), 4)
            if team_total_air > 0 and not np.isnan(air_total) else None
        )

        # ---- Explosive plays (completions ≥ threshold) ----
        exp_15 = (comps["yards_gained"] >= 15).sum()
        exp_20 = int((comps["yards_gained"] >= 20).sum())
        exp_40 = int((comps["yards_gained"] >= 40).sum())
        explosive_play_rate = (
            round(float(exp_15 / rec), 4) if rec > 0 else None
        )
        longest = (
            int(comps["yards_gained"].max())
            if rec > 0 else None
        )

        # ---- YAC / YBC ----
        yac_total = comps["yards_after_catch"].sum()
        yac_per_rec = (
            round(float(yac_total / rec), 2)
            if rec > 0 and not np.isnan(yac_total) else None
        )
        # YBC = yards before catch = yards_gained − yards_after_catch (per completion)
        comps_valid = comps.dropna(subset=["yards_gained", "yards_after_catch"])
        ybc_total = (comps_valid["yards_gained"] - comps_valid["yards_after_catch"]).sum()
        ybc_per_rec = (
            round(float(ybc_total / rec), 2)
            if rec > 0 and len(comps_valid) > 0 else None
        )

        # ---- Broken tackles (PFR) ----
        norm_name = _norm(canonical)
        btkl_raw  = btkl_by_name.get(norm_name)
        # Nickname alias fallback (e.g. "joshua palmer" → "josh palmer").
        # Only try when the direct lookup truly missed (None), not when it returned 0.
        if btkl_raw is None:
            alias_norm = _CANONICAL_TO_PFR_NORM.get(norm_name, "")
            if alias_norm:
                btkl_raw = btkl_by_name.get(alias_norm)
        broken_tackles = int(btkl_raw) if btkl_raw is not None and not np.isnan(btkl_raw) else None
        btkl_per_rec   = (
            round(float(broken_tackles / rec), 3)
            if broken_tackles is not None and rec > 0 else None
        )

        rows.append({
            "player":  canonical,
            "team":    team,
            "games":   games,
            "targets": targets,
            # Core
            "epa_per_target":   epa_per_target,
            "success_rate":     success_rate,
            "yards_per_target": yards_per_target,
            "catch_rate":       catch_rate,
            "fpoe":             fpoe,
            # Routes
            "yards_per_route_run":   yards_per_route_run,
            "targets_per_route_run": targets_per_route_run,
            # Air yards
            "air_yards_per_target": air_yards_per_target,
            "air_yards_share_pct":  air_yards_share_pct,
            # Explosive
            "explosive_play_rate":  explosive_play_rate,
            "explosive_rec_20_plus": exp_20,
            "explosive_rec_40_plus": exp_40,
            "longest_reception":    longest,
            # Contact
            "yac_per_rec":    yac_per_rec,
            "ybc_per_rec":    ybc_per_rec,
            "broken_tackles": broken_tackles,
            "btkl_per_rec":   btkl_per_rec,
            # Efficiency score added below
            "wr_efficiency_score": None,
        })

    print(f"[{season}] Built {len(rows)} qualifying WRs before scoring …")

    # -----------------------------------------------------------------------
    # Add WR efficiency score
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df = _add_efficiency_scores(df)

    # Sort: efficiency score desc, then yards_per_route_run desc
    df = df.sort_values(
        ["wr_efficiency_score", "yards_per_route_run"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    # Ensure correct column order; fill any missing columns with None
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]

    # Convert NaN → None for JSON serialisation
    return json.loads(df.to_json(orient="records"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        print(f"\n=== Season {season} ===")
        players = build_season(season)

        out_path = OUTPUT_DIR / f"wr_efficiency_analytics_{season}.json"
        with open(out_path, "w") as f:
            json.dump(players, f, indent=2)
        print(f"[{season}] Wrote {len(players)} rows → {out_path}")

        # Also write a latest alias
        if season == max(SEASONS):
            alias = OUTPUT_DIR / "wr_efficiency_analytics.json"
            with open(alias, "w") as f:
                json.dump(players, f, indent=2)
            print(f"[{season}] Alias → {alias}")

    print("\nDone.")


if __name__ == "__main__":
    main()
