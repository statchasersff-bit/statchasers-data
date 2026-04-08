"""
build_rb_player_overview.py
───────────────────────────
Builds the RB Player Overview dataset for the StatChasers Research tab.

This table answers: "What type of RB is this, how strong is the role,
and how fantasy-friendly is the profile overall?"

Data rules:
  - RBs only (position == "RB")
  - One row per player
  - 2025 season
  - Minimum 3 games AND 15 rush attempts to qualify
  - null for missing values, never empty strings
  - Counting stats: integers
  - Rate stats: 2 decimals
  - Percentages (display-facing): 1 decimal

Data sources:
  data/raw/nflverse_play_by_play.parquet  — all counting + snap stats
  data/processed/player_metrics.json      — route%, FPOE, career arc, age
  data/raw/nflverse_players.parquet       — years_of_experience → exp_tier
  data/raw/sleeper_players.json           — name disambiguation + age

Output:
  output/rb_player_overview.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
PBP_PATH     = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
METRICS_PATH = ROOT / "data" / "processed" / "player_metrics.json"
NFL_PATH     = ROOT / "data" / "raw"       / "nflverse_players.parquet"
SLEEPER_PATH = ROOT / "data" / "raw"       / "sleeper_players.json"
OUTPUT_PATH  = ROOT / "output"             / "rb_player_overview.json"

SEASON       = 2025
MIN_GAMES    = 3
MIN_RUSH_ATT = 15  # include all meaningful RBs

# Hardcoded team overrides for ambiguous same-position abbreviations where
# Sleeper's team field is stale relative to the 2025 PBP season.
_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    "T.Etienne": {"JAX": "Travis Etienne", "CAR": "Trevor Etienne"},
    "B.Robinson": {"ATL": "Bijan Robinson", "SF": "Brian Robinson", "WAS": "Brian Robinson"},
    "J.Williams": {"DEN": "Javonte Williams", "NO": "Jamaal Williams"},
}

COLUMNS: list[dict] = [
    {"key": "rank",              "label": "#",              "type": "number", "group": "Identity",               "defaultVisible": True},
    {"key": "player",            "label": "Player",         "type": "string", "group": "Identity",               "defaultVisible": True},
    {"key": "team",              "label": "Team",           "type": "string", "group": "Identity",               "defaultVisible": True},
    {"key": "age",               "label": "Age",            "type": "number", "group": "Identity",               "defaultVisible": True},
    {"key": "games",             "label": "GP",             "type": "number", "group": "Identity",               "defaultVisible": True},
    {"key": "snap_pct",          "label": "Snap %",         "type": "number", "group": "Identity",               "defaultVisible": True},
    {"key": "touches_per_gm",    "label": "Touches / Gm",  "type": "number", "group": "Role & Usage",           "defaultVisible": True},
    {"key": "rush_att_per_gm",   "label": "Rush / Gm",     "type": "number", "group": "Role & Usage",           "defaultVisible": True},
    {"key": "targets_per_gm",    "label": "Targets / Gm",  "type": "number", "group": "Role & Usage",           "defaultVisible": True},
    {"key": "route_pct",         "label": "Route %",        "type": "number", "group": "Role & Usage",           "defaultVisible": False},
    {"key": "rz_rush_att",       "label": "RZ Att",         "type": "number", "group": "Opportunity",            "defaultVisible": True},
    {"key": "goal_line_att",     "label": "GL Att",         "type": "number", "group": "Opportunity",            "defaultVisible": True},
    {"key": "target_share_pct",  "label": "Tgt Share %",   "type": "number", "group": "Opportunity",            "defaultVisible": True},
    {"key": "receptions",        "label": "REC",            "type": "number", "group": "Receiving + Efficiency", "defaultVisible": True},
    {"key": "receiving_yds",     "label": "REC YDS",        "type": "number", "group": "Receiving + Efficiency", "defaultVisible": True},
    {"key": "yards_per_touch",   "label": "Yds / Touch",   "type": "number", "group": "Receiving + Efficiency", "defaultVisible": True},
    {"key": "explosive_run_pct", "label": "Explosive %",   "type": "number", "group": "Receiving + Efficiency", "defaultVisible": True},
    {"key": "breakaway_run_pct", "label": "Breakaway %",   "type": "number", "group": "Receiving + Efficiency", "defaultVisible": False},
    {"key": "fpoe",              "label": "FPOE",           "type": "number", "group": "Receiving + Efficiency", "defaultVisible": True},
    {"key": "stability",         "label": "Stability (/10)","type": "number", "group": "Stability",              "defaultVisible": True},
    {"key": "volatility",        "label": "Volatility",     "type": "number", "group": "Stability",              "defaultVisible": True},
    {"key": "career_arc",        "label": "Career Arc",     "type": "string", "group": "Career Context",         "defaultVisible": True},
    {"key": "exp_tier",          "label": "Exp. Tier",      "type": "string", "group": "Career Context",         "defaultVisible": True},
    {"key": "opp_score",         "label": "Opp Score",      "type": "number", "group": "Composite Scores",       "defaultVisible": True},
    {"key": "usage_score",       "label": "Usage Score",    "type": "number", "group": "Composite Scores",       "defaultVisible": True},
    {"key": "player_score",      "label": "Player Score",   "type": "number", "group": "Composite Scores",       "defaultVisible": True},
    {"key": "rb_tier_score",     "label": "Tier Score",     "type": "number", "group": "Composite Scores",       "defaultVisible": True},
    {"key": "rb_tier",           "label": "Tier",           "type": "string", "group": "Composite Scores",       "defaultVisible": True},
]


# ---------------------------------------------------------------------------
# Name resolution
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


def _resolve(
    pbp_name: str,
    pbp_team: str,
    full_name_set: set[str],
    unambig: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    metrics_pos: dict[str, str],
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
        pos_matches = [
            (t, n) for t, n in teams.items()
            if sleeper_pos.get(n, "") == "RB" or metrics_pos.get(n, "") == "RB"
        ]
        if not pos_matches:
            pos_matches = list(teams.items())
        if len(pos_matches) == 1:
            return pos_matches[0][1]
        no_team = [(t, n) for t, n in pos_matches if t is None]
        if no_team:
            return no_team[0][1]
        with_team = [(t, n) for t, n in pos_matches if t is not None]
        if with_team:
            return with_team[0][1]
        return pos_matches[0][1]
    return pbp_name


# ---------------------------------------------------------------------------
# Label helpers
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


def _rb_tier_label(score: float) -> str:
    if score >= 95: return "Fantasy Alpha"
    if score >= 90: return "Elite Workhorse"
    if score >= 80: return "Bellcow RB"
    if score >= 70: return "Lead Back"
    if score >= 60: return "Committee RB"
    if score >= 50: return "Flex Option"
    return "Handcuff / Depth"


# ---------------------------------------------------------------------------
# Percentile (0-100).  kind="rank": (below + 0.5*equal) / n * 100
# null value or empty pool → 50.0 (neutral)
# ---------------------------------------------------------------------------

def _pct(value: float | None, arr: list[float | None]) -> float:
    clean = np.array(
        [float(v) for v in arr if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=float,
    )
    if len(clean) == 0:
        return 50.0
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    v = float(value)
    n = len(clean)
    below = float(np.sum(clean < v))
    equal = float(np.sum(clean == v))
    return float((below + 0.5 * equal) / n * 100.0)


# ---------------------------------------------------------------------------
# Stability & volatility from per-game touch sequence
# ---------------------------------------------------------------------------

def _stability_volatility(
    game_touch_map: dict[str, int],
    game_order: list[str],
) -> tuple[float | None, float | None]:
    """
    stability  = 1 – σ(last-6-game touches) / μ(last-6-game touches), ×10, clamped 0-10
    volatility = σ(all-game touches) / μ(all-game touches)  [coefficient of variation]
    """
    ordered = [game_touch_map[g] for g in game_order if g in game_touch_map]
    # Fill in games that appear only in rec map (receiver but no rush that game)
    extra = [v for g, v in game_touch_map.items() if g not in game_order]
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
# Main builder
# ---------------------------------------------------------------------------

def build(
    pbp: pd.DataFrame,
    player_metrics: list[dict],
    nfl_players: pd.DataFrame,
    sleeper_players: list[dict],
) -> list[dict]:

    # ── Setup lookup tables ──────────────────────────────────────────────────
    full_name_set  = {(p.get("full_name") or "").strip() for p in sleeper_players}
    unambig        = _build_unambig(sleeper_players)
    team_disambig  = _build_team_disambig(sleeper_players)
    sleeper_pos    = {(p.get("full_name") or "").strip(): (p.get("position") or "") for p in sleeper_players}
    sleeper_age    = {(p.get("full_name") or "").strip(): p.get("age") for p in sleeper_players}
    metrics_pos    = {p["player"]: p.get("pos", "") for p in player_metrics}
    metrics_map    = {p["player"]: p for p in player_metrics}
    rb_metric_names: set[str] = {p["player"] for p in player_metrics if p.get("pos") == "RB"}
    print(f"  {len(rb_metric_names)} RBs in player_metrics.")

    # ── nflverse years_of_experience lookup ─────────────────────────────────
    yoe_lookup: dict[str, int] = {}
    if not nfl_players.empty and "display_name" in nfl_players.columns:
        yoe_col = "years_of_experience" if "years_of_experience" in nfl_players.columns else None
        if yoe_col:
            for _, row in nfl_players.iterrows():
                name = str(row["display_name"]).strip()
                yoe  = row[yoe_col]
                if name and yoe is not None and not (isinstance(yoe, float) and np.isnan(yoe)):
                    yoe_lookup[name] = int(yoe)

    # ── Split PBP by play type ───────────────────────────────────────────────
    rush   = pbp[pbp["rush_attempt"] == 1].copy()
    passes = pbp[pbp["pass_attempt"] == 1].copy()

    # ── Team-level totals ────────────────────────────────────────────────────
    # Total offensive plays per team (for snap_pct approximation)
    off_plays = pbp[
        (pbp["rush_attempt"] == 1) | (pbp["pass_attempt"] == 1)
    ].groupby("posteam")["play_id"].count().to_dict()

    # Total targets per team (for target_share_pct)
    team_targets_tot: dict[str, int] = (
        passes[passes["receiver_player_name"].notna()]
        .groupby("posteam")["pass_attempt"].count()
        .to_dict()
    )

    # ── Name resolution cache ────────────────────────────────────────────────
    name_cache: dict[tuple[str, str], str] = {}

    def _tag(pbp_name: str, pbp_team: str) -> str:
        key = (pbp_name, pbp_team)
        if key not in name_cache:
            name_cache[key] = _resolve(
                str(pbp_name), str(pbp_team),
                full_name_set, unambig, team_disambig,
                metrics_pos, sleeper_pos,
            )
        return name_cache[key]

    # Tag plays
    rush["_fn"] = rush.apply(
        lambda r: _tag(str(r["rusher_player_name"] or ""), str(r.get("posteam", ""))), axis=1
    )
    passes["_fn"] = passes.apply(
        lambda r: _tag(str(r.get("receiver_player_name") or ""), str(r.get("posteam", ""))), axis=1
    )

    # ── Collect qualifying RB names ──────────────────────────────────────────
    # Primary: player_metrics RBs
    # Supplement: RBs from PBP with ≥ MIN_RUSH_ATT carries who are in Sleeper as RB
    rb_pbp_names: set[str] = set()
    for fn, grp in rush.groupby("_fn"):
        if str(fn) in rb_metric_names:
            rb_pbp_names.add(str(fn))
            continue
        if sleeper_pos.get(str(fn), "") == "RB" and len(grp) >= MIN_RUSH_ATT:
            rb_pbp_names.add(str(fn))

    print(f"  {len(rb_pbp_names)} total qualifying RBs (player_metrics + PBP supplement).")

    has_yardline = "yardline_100" in pbp.columns
    sort_cols    = [c for c in ("season", "week") if c in rush.columns]

    # ── Per-player rush stats ────────────────────────────────────────────────
    rush_stats: dict[str, dict] = {}
    rb_rush = rush[rush["_fn"].isin(rb_pbp_names)]
    for fn, grp in rb_rush.groupby("_fn"):
        fn = str(fn)
        rush_att = len(grp)
        rush_yds = int(grp["yards_gained"].sum())
        games    = int(grp["game_id"].nunique())

        # Chronological game order
        if sort_cols:
            game_order = (
                grp[["game_id"] + sort_cols]
                .drop_duplicates("game_id")
                .sort_values(sort_cols)["game_id"]
                .tolist()
            )
        else:
            game_order = list(grp["game_id"].unique())

        # Primary team (last game's team, for snap_pct lookup)
        primary_team = str(grp.sort_values(sort_cols)["posteam"].iloc[-1]) if sort_cols else ""

        rush_stats[fn] = {
            "rush_att":      rush_att,
            "rush_yds":      rush_yds,
            "games":         games,
            "primary_team":  primary_team,
            "rz_rush_att":   int((grp["yardline_100"] <= 20).sum()) if has_yardline else None,
            "goal_line_att": int((grp["yardline_100"] <= 5).sum())  if has_yardline else None,
            "explosive10":   int((grp["yards_gained"] >= 10).sum()),
            "explosive15":   int((grp["yards_gained"] >= 15).sum()),
            "game_order":    game_order,
            "game_rush_map": dict(grp.groupby("game_id")["rush_attempt"].count()),
        }

    # ── Per-player snap_pct from PBP ────────────────────────────────────────
    # snap_pct ≈ (rush plays + targeted pass plays) / team total offensive plays × 100
    # — this is a play-participation rate, not true "on-field snap" count, but
    #   it is the most consistent metric available without dedicated snap-count data.
    rb_pass_tgt = passes[
        passes["_fn"].isin(rb_pbp_names) & passes["_fn"].ne("")
    ]
    player_pass_plays: dict[str, int] = dict(
        rb_pass_tgt.groupby("_fn")["pass_attempt"].count()
    )
    snap_pct_map: dict[str, float] = {}
    for fn, rs in rush_stats.items():
        team = rs["primary_team"]
        total_team = off_plays.get(team, 0)
        if total_team > 0:
            plays_in = rs["rush_att"] + player_pass_plays.get(fn, 0)
            snap_pct_map[fn] = round(plays_in / total_team * 100, 1)

    # ── Per-player receiving stats ───────────────────────────────────────────
    rec_stats: dict[str, dict] = {}
    rb_pass = passes[passes["_fn"].isin(rb_pbp_names) & passes["_fn"].ne("")]
    for fn, grp in rb_pass.groupby("_fn"):
        fn   = str(fn)
        tgt  = len(grp)
        comps = grp[grp["complete_pass"] == 1]
        rec  = len(comps)
        rec_yds = int(comps["yards_gained"].sum())

        pbp_team = str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns else ""
        team_tgt = team_targets_tot.get(pbp_team, 0)
        tgt_share = round(tgt / team_tgt * 100, 1) if team_tgt > 0 else None

        rec_stats[fn] = {
            "targets":          tgt,
            "receptions":       rec,
            "receiving_yds":    rec_yds,
            "target_share_pct": tgt_share,
            "game_rec_map":     dict(comps.groupby("game_id")["complete_pass"].count()),
        }

    # ── Assemble raw rows ────────────────────────────────────────────────────
    raw_rows: list[dict] = []
    for fn in rb_pbp_names:
        rs = rush_stats.get(fn, {})
        rc = rec_stats.get(fn, {})

        rush_att  = rs.get("rush_att", 0)
        games     = rs.get("games", 0)

        if games < MIN_GAMES or rush_att < MIN_RUSH_ATT:
            continue

        rush_yds     = rs.get("rush_yds", 0)
        receptions   = rc.get("receptions", 0)
        receiving_yds = rc.get("receiving_yds", 0)
        targets      = rc.get("targets", 0)
        touches      = rush_att + receptions

        touches_pg   = round(touches       / games, 2) if games else None
        rush_att_pg  = round(rush_att      / games, 2) if games else None
        targets_pg   = round(targets       / games, 2) if games else None
        ypt          = round((rush_yds + receiving_yds) / touches, 2) if touches > 0 else None

        exp10_pct    = round(rs.get("explosive10", 0) / rush_att * 100, 1) if rush_att > 0 else None
        exp15_pct    = round(rs.get("explosive15", 0) / rush_att * 100, 1) if rush_att > 0 else None

        # Stability / volatility
        game_rush_map  = rs.get("game_rush_map", {})
        game_rec_map   = rc.get("game_rec_map", {})
        game_order     = rs.get("game_order", [])
        game_touch_map = {
            g: game_rush_map.get(g, 0) + game_rec_map.get(g, 0)
            for g in set(game_rush_map) | set(game_rec_map)
        }
        stability, volatility = _stability_volatility(game_touch_map, game_order)

        # Snap pct from PBP (preferred) — no fallback to player_metrics approximation
        snap_pct = snap_pct_map.get(fn)

        # Fields from player_metrics (None for supplement-only players)
        m = metrics_map.get(fn, {})
        route_pct  = m.get("routeParticipation")
        fpoe       = m.get("fpoe")
        career_arc = m.get("careerArc")
        age        = m.get("age") or sleeper_age.get(fn)

        # Target share: prefer PBP, fall back to player_metrics
        tgt_share = rc.get("target_share_pct")
        if tgt_share is None:
            tgt_share = m.get("targetShare")

        # Goal-line att: prefer PBP, fall back to player_metrics
        goal_line_att = rs.get("goal_line_att")
        if goal_line_att is None:
            goal_line_att = m.get("goalLineCarries")

        # Team: PBP primary_team is ground truth; player_metrics.team is stale.
        team = rs.get("primary_team") or m.get("team") or ""

        # Exp tier from nflverse years_of_experience
        yoe = yoe_lookup.get(fn)
        exp_tier_label = _exp_tier(yoe)

        raw_rows.append({
            "player":           fn,
            "team":             team,
            "age":              age,
            "games":            games,
            "snap_pct":         snap_pct,
            "touches_per_gm":   touches_pg,
            "rush_att_per_gm":  rush_att_pg,
            "targets_per_gm":   targets_pg,
            "route_pct":        route_pct,
            "rz_rush_att":      rs.get("rz_rush_att"),
            "goal_line_att":    goal_line_att,
            "target_share_pct": tgt_share,
            "receptions":       receptions,
            "receiving_yds":    receiving_yds,
            "yards_per_touch":  ypt,
            "explosive_run_pct":exp10_pct,
            "breakaway_run_pct":exp15_pct,
            "fpoe":             fpoe,
            "stability":        stability,
            "volatility":       volatility,
            "career_arc":       career_arc,
            "exp_tier":         exp_tier_label,
        })

    # ── Composite scores ─────────────────────────────────────────────────────

    def _col(key: str) -> list[float | None]:
        return [r.get(key) for r in raw_rows]

    arr_touches  = _col("touches_per_gm")
    arr_snap     = _col("snap_pct")
    arr_rz       = _col("rz_rush_att")
    arr_gl       = _col("goal_line_att")
    arr_tgtshare = _col("target_share_pct")
    arr_tgtspg   = _col("targets_per_gm")
    arr_route    = _col("route_pct")
    arr_rush_pg  = _col("rush_att_per_gm")
    arr_exp10    = _col("explosive_run_pct")
    arr_ypts     = _col("yards_per_touch")
    arr_break    = _col("breakaway_run_pct")
    arr_fpoe     = _col("fpoe")
    arr_stab     = _col("stability")

    final_rows: list[dict] = []
    for r in raw_rows:
        p_touches  = _pct(r["touches_per_gm"],   arr_touches)
        p_snap     = _pct(r["snap_pct"],          arr_snap)
        p_rz       = _pct(r["rz_rush_att"],       arr_rz)
        p_gl       = _pct(r["goal_line_att"],     arr_gl)
        p_tgtshare = _pct(r["target_share_pct"],  arr_tgtshare)
        p_tgtspg   = _pct(r["targets_per_gm"],    arr_tgtspg)
        p_route    = _pct(r["route_pct"],         arr_route)
        p_rush_pg  = _pct(r["rush_att_per_gm"],   arr_rush_pg)
        p_exp10    = _pct(r["explosive_run_pct"], arr_exp10)
        p_ypts     = _pct(r["yards_per_touch"],   arr_ypts)
        p_break    = _pct(r["breakaway_run_pct"], arr_break)
        p_fpoe     = _pct(r["fpoe"],              arr_fpoe)
        p_stab     = _pct(r["stability"],         arr_stab)

        # 1. Opportunity Score
        opp_score = round(
            p_touches  * 0.40
            + p_snap   * 0.20
            + p_rz     * 0.20
            + p_gl     * 0.10
            + p_tgtshare * 0.10,
            1,
        )

        # 2. Usage Score (all role/deployment signals — no efficiency metrics)
        usage_score = round(
            p_tgtspg    * 0.35
            + p_route   * 0.25
            + p_rush_pg * 0.25
            + p_tgtshare * 0.15,
            1,
        )

        # 3. Efficiency Score (internal, also used in player_score)
        eff_score = (
            p_ypts  * 0.35
            + p_break * 0.25
            + p_fpoe  * 0.40
        )

        # 4. Player Score
        player_score = round(
            opp_score   * 0.50
            + usage_score * 0.25
            + eff_score   * 0.25,
            1,
        )

        # 5. RB Tier Score
        receiving_sub = (
            p_tgtspg   * 0.50
            + p_route  * 0.30
            + p_tgtshare * 0.20
        )
        rb_tier_score = round(
            opp_score       * 0.50
            + receiving_sub * 0.20
            + eff_score     * 0.20
            + p_stab        * 0.10,
            1,
        )

        final_rows.append({
            **r,
            "opp_score":    opp_score,
            "usage_score":  usage_score,
            "player_score": player_score,
            "rb_tier_score":rb_tier_score,
            "rb_tier":      _rb_tier_label(rb_tier_score),
        })

    # ── Sort + rank ──────────────────────────────────────────────────────────
    final_rows.sort(key=lambda r: (r.get("rb_tier_score") or 0.0), reverse=True)
    ordered_keys = [c["key"] for c in COLUMNS]
    result: list[dict] = []
    for i, r in enumerate(final_rows, start=1):
        r["rank"] = i
        result.append({k: r.get(k) for k in ordered_keys})

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (PBP_PATH, METRICS_PATH, SLEEPER_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    print("Loading play-by-play data...")
    pbp_full = pd.read_parquet(PBP_PATH)
    pbp = pbp_full[(pbp_full["season"] == SEASON) & (pbp_full["season_type"] == "REG")].copy()  # regular season only
    print(f"  {len(pbp):,} plays for {SEASON}.")

    with open(METRICS_PATH) as f:
        player_metrics: list[dict] = json.load(f)
    print(f"  {len(player_metrics)} player metric records.")

    with open(SLEEPER_PATH) as f:
        sleeper_players: list[dict] = json.load(f)
    print(f"  {len(sleeper_players)} Sleeper players.")

    nfl_players = pd.DataFrame()
    if NFL_PATH.exists():
        nfl_players = pd.read_parquet(NFL_PATH)
        print(f"  {len(nfl_players)} nflverse players (for exp_tier).")
    else:
        print("  nflverse_players.parquet not found — exp_tier will be null.", file=sys.stderr)

    print("Building RB Player Overview...")
    rows = build(pbp, player_metrics, nfl_players, sleeper_players)
    print(f"  {len(rows)} RBs qualified.")

    latest_week = int(pbp["week"].max()) if "week" in pbp.columns else None

    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season":     SEASON,
        "week":       latest_week,
        "table":      "rb_player_overview",
        "columns":    COLUMNS,
        "rows":       rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.1f} KB, {len(rows)} rows)")
    print("RB Player Overview build complete.")


if __name__ == "__main__":
    main()
