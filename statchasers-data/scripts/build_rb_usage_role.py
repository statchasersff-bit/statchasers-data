"""
build_rb_usage_role.py
──────────────────────
Builds the RB Usage & Role dataset for the StatChasers Research tab.

Answers: Which RBs are gaining role, usage, and fantasy-relevant opportunity?

Emphasis:
  - Recent role growth (snap%, touch rate, target rate)
  - Passing-game involvement (targets, route % proxy)
  - Scoring opportunity (RZ touches, GL attempts)
  - Composite signal (Usage & Role Score) that surfaces risers

Minimal overlap with:
  rb_player_overview      → season-long totals, tier labels, composite profile
  rb_efficiency_analytics → per-play efficiency (EPA, success %, YAC, BTKL)
  rb_advanced_stats       → raw volume totals

Data sources:
  data/raw/nflverse_play_by_play.parquet   — all per-play stats
  data/raw/sleeper_players.json            — name disambiguation + position

Delta windows (recent vs baseline):
  ≥ 8 games → last 4 vs prior 4
  ≥ 6 games → last 3 vs prior 3
  ≥ 4 games → last 3 vs season average
  < 4 games → null deltas, role_trend = "Limited Sample"

Data rules:
  - RBs only, one row per player, 2025 season
  - Minimum 3 games AND 15 season touches to appear in output
  - Percentile pool for usage_role_score: same minimum
  - null for missing values, never empty strings
  - Percentages: 2 decimals
  - Per-game rates: 2 decimals
  - Counting stats: integers
  - usage_role_score: 2 decimals (0–100)
  - Deltas: signed, 2 decimals

Output:
  output/rb_usage_role.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
PBP_PATH     = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
SLEEPER_PATH = ROOT / "data" / "raw"       / "sleeper_players.json"
OUTPUT_PATH  = ROOT / "output"             / "rb_usage_role.json"

SEASON       = 2025
MIN_GAMES    = 3
MIN_TOUCHES  = 15     # rush_att + receptions, season total

COLUMNS: list[dict] = [
    {"key": "player",               "label": "Player",            "type": "string", "group": "Identity"},
    {"key": "team",                 "label": "Team",              "type": "string", "group": "Identity"},
    {"key": "games",                "label": "GP",                "type": "number", "group": "Identity"},
    {"key": "snap_pct",             "label": "Snap %",            "type": "number", "group": "Role"},
    {"key": "delta_snap_pct",       "label": "Δ Snap %",          "type": "number", "group": "Role"},
    {"key": "touches_per_gm",       "label": "Touches / Gm",      "type": "number", "group": "Usage"},
    {"key": "delta_touches_per_gm", "label": "Δ Touches / Gm",    "type": "number", "group": "Usage"},
    {"key": "rush_att_per_gm",      "label": "Rush / Gm",         "type": "number", "group": "Usage"},
    {"key": "delta_rush_att_per_gm","label": "Δ Rush / Gm",       "type": "number", "group": "Usage"},
    {"key": "targets_per_gm",       "label": "Targets / Gm",      "type": "number", "group": "Receiving Role"},
    {"key": "delta_targets_per_gm", "label": "Δ Targets / Gm",    "type": "number", "group": "Receiving Role"},
    {"key": "route_pct",            "label": "Route %",           "type": "number", "group": "Receiving Role"},
    {"key": "delta_route_pct",      "label": "Δ Route %",         "type": "number", "group": "Receiving Role"},
    {"key": "rz_touches",           "label": "RZ Touches",        "type": "number", "group": "Scoring Role"},
    {"key": "goal_line_att",        "label": "GL Att",            "type": "number", "group": "Scoring Role"},
    {"key": "usage_role_score",     "label": "Usage & Role Score", "type": "number", "group": "Composite"},
    {"key": "role_trend",           "label": "Role Trend",        "type": "string", "group": "Composite"},
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _abbrev(full_name: str) -> str:
    """'Saquon Barkley' → 'S.Barkley'"""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return parts[0][0] + "." + " ".join(parts[1:])


def _build_name_lookups(sleeper_players: list[dict]) -> tuple[set[str], dict[str, str], dict[str, dict[str, str]]]:
    """
    Returns:
      full_name_set  — all known full names
      unambig_full   — abbrev → full name (unique abbrevs only)
      team_disambig  — abbrev → {team: full_name} (for disambiguation by team)
    """
    full_name_set: set[str] = set()
    abbrev_map: dict[str, list[str]] = {}

    for p in sleeper_players:
        full = (p.get("full_name") or "").strip()
        if not full:
            continue
        full_name_set.add(full)
        ab = _abbrev(full)
        abbrev_map.setdefault(ab, []).append(full)

    unambig_full = {ab: fns[0] for ab, fns in abbrev_map.items() if len(fns) == 1}

    # team_disambig: for ambiguous abbreviations, map {team: full_name}.
    # Players with team=None in Sleeper are stored under the key None —
    # they serve as the fallback for any PBP team not already claimed by a
    # known-team player (e.g. Brian Robinson team=None is the fallback for
    # any team that isn't ATL where Bijan Robinson is explicitly registered).
    team_disambig: dict[str, dict] = {}
    for p in sleeper_players:
        full = (p.get("full_name") or "").strip()
        if not full:
            continue
        raw_team = p.get("team")           # preserve Python None
        ab = _abbrev(full)
        if len(abbrev_map.get(ab, [])) > 1:
            team_disambig.setdefault(ab, {})[raw_team] = full

    return full_name_set, unambig_full, team_disambig


def _resolve(
    pbp_name: str,
    team: str,
    full_name_set: set[str],
    unambig_full: dict[str, str],
    team_disambig: dict[str, dict],
) -> str | None:
    if not pbp_name or pbp_name == "nan":
        return None
    if pbp_name in full_name_set:
        return pbp_name
    if pbp_name in team_disambig:
        teams = team_disambig[pbp_name]
        # Exact team match — highest confidence
        if team in teams:
            return teams[team]
        # No exact match: return the player with no known team (if any).
        # Do NOT fall back to a player with a different known team — that
        # would incorrectly assign ATL-registered Bijan Robinson to SF plays.
        if None in teams:
            return teams[None]
        return None
    if pbp_name in unambig_full:
        return unambig_full[pbp_name]
    return None


def _pct(value: float, arr: np.ndarray) -> float:
    """Rank percentile: (below + 0.5*equal) / n * 100. Returns 50.0 for null."""
    if value is None or np.isnan(value):
        return 50.0
    below = np.sum(arr < value)
    equal = np.sum(arr == value)
    return float((below + 0.5 * equal) / len(arr) * 100)


def _r2(v: float | None) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 2)


# ── Per-game window logic ──────────────────────────────────────────────────────

def _compute_delta(recent: list[dict], baseline: list[dict] | None, key: str) -> float | None:
    """
    Compute mean(recent[key]) - mean(baseline[key]).
    If baseline is None, uses a pre-computed season average passed via a sentinel dict.
    Returns None if data is missing.
    """
    r_vals = [g[key] for g in recent if g.get(key) is not None]
    if not r_vals:
        return None
    if baseline is None:
        return None
    b_vals = [g[key] for g in baseline if g.get(key) is not None]
    if not b_vals:
        return None
    return float(np.mean(r_vals)) - float(np.mean(b_vals))


def _window_split(game_rows: list[dict], season_avgs: dict[str, float]):
    """
    Returns (recent_games, baseline_games_or_None, tier_label).
    Tier 1: last 4 vs prior 4 (n >= 8)
    Tier 2: last 3 vs prior 3 (n >= 6)
    Tier 3: last 3 vs season_avgs proxy (n >= 4)
    Tier 4: null (n < 4)
    """
    n = len(game_rows)
    if n >= 8:
        return game_rows[-4:], game_rows[-8:-4], "4v4"
    if n >= 6:
        return game_rows[-3:], game_rows[-6:-3], "3v3"
    if n >= 4:
        # baseline = one-element list with season averages
        return game_rows[-3:], [season_avgs], "recent_vs_avg"
    return None, None, "limited"


# ── Main build ─────────────────────────────────────────────────────────────────

def build(pbp_path: Path, sleeper_path: Path) -> list[dict]:
    print("Loading play-by-play data...")
    pbp = pd.read_parquet(pbp_path)
    pbp25 = pbp[pbp["season"] == SEASON].copy()
    print(f"  {len(pbp25):,} plays for {SEASON}.")

    print("Loading Sleeper players...")
    with open(sleeper_path) as f:
        raw_sleeper = json.load(f)
    sleeper_players = list(raw_sleeper.values()) if isinstance(raw_sleeper, dict) else raw_sleeper
    sleeper_players = [p for p in sleeper_players if isinstance(p, dict)]
    rb_names: set[str] = {
        (p.get("full_name") or "").strip()
        for p in sleeper_players
        if (p.get("position") or "") == "RB" and (p.get("full_name") or "").strip()
    }
    print(f"  {len(sleeper_players):,} Sleeper players, {len(rb_names):,} RBs.")

    full_name_set, unambig_full, team_disambig = _build_name_lookups(sleeper_players)

    # ── Per-game team-level stats ──────────────────────────────────────────────
    # off plays: rush OR pass attempt (for snap_pct denominator)
    # pass plays: pass_attempt == 1 (proxy for dropbacks = route_pct denominator)
    off = pbp25[(pbp25["rush_attempt"] == 1) | (pbp25["pass_attempt"] == 1)].copy()
    team_game_off   = off.groupby(["posteam", "game_id", "week"])["play_id"].count().rename("team_off_plays")
    team_game_pass  = (
        pbp25[pbp25["pass_attempt"] == 1]
        .groupby(["posteam", "game_id", "week"])["play_id"]
        .count()
        .rename("team_pass_plays")
    )
    team_game = pd.concat([team_game_off, team_game_pass], axis=1).reset_index()
    team_game["team_pass_plays"] = team_game["team_pass_plays"].fillna(0)

    # ── Rush plays ─────────────────────────────────────────────────────────────
    rush = pbp25[pbp25["rush_attempt"] == 1].copy()

    name_cache: dict[tuple[str, str], str | None] = {}

    def resolve_rusher(row):
        key = (str(row["rusher_player_name"]), str(row.get("posteam", "")))
        if key not in name_cache:
            name_cache[key] = _resolve(key[0], key[1], full_name_set, unambig_full, team_disambig)
        return name_cache[key]

    rush["full_name"] = rush.apply(resolve_rusher, axis=1)
    rush = rush[rush["full_name"].notna()].copy()

    rb_abbrevs: set[str] = {ab for (ab, _), fn in name_cache.items() if fn in rb_names}

    # ── Pass plays (targeted) ──────────────────────────────────────────────────
    passes = pbp25[
        (pbp25["pass_attempt"] == 1)
        & pbp25["receiver_player_name"].notna()
        & pbp25["receiver_player_name"].isin(rb_abbrevs)
    ].copy()

    rec_cache: dict[tuple[str, str], str | None] = {}

    def resolve_receiver(row):
        key = (str(row.get("receiver_player_name", "")), str(row.get("posteam", "")))
        if key not in rec_cache:
            rec_cache[key] = _resolve(key[0], key[1], full_name_set, unambig_full, team_disambig)
        return rec_cache[key]

    passes["full_name"] = passes.apply(resolve_receiver, axis=1)
    passes = passes[passes["full_name"].notna() & passes["full_name"].isin(rb_names)].copy()

    # completion flag
    passes["is_complete"] = (passes["complete_pass"] == 1).astype(int)

    # ── Per-game rush stats ────────────────────────────────────────────────────
    rush["rz_rush"] = (rush["yardline_100"] <= 20).astype(int)
    rush["gl_att"]  = (rush["yardline_100"] <= 5).astype(int)

    rush_game = (
        rush.groupby(["full_name", "posteam", "game_id", "week"])
        .agg(
            rush_att   = ("rush_attempt", "sum"),
            rz_rush    = ("rz_rush",      "sum"),
            gl_att     = ("gl_att",       "sum"),
        )
        .reset_index()
        .rename(columns={"full_name": "player", "posteam": "team"})
    )

    # ── Per-game pass stats ────────────────────────────────────────────────────
    passes["rz_tgt"] = (passes["yardline_100"] <= 20).astype(int)

    pass_game = (
        passes.groupby(["full_name", "posteam", "game_id", "week"])
        .agg(
            targets    = ("pass_attempt", "sum"),
            receptions = ("is_complete",  "sum"),
            rz_tgt     = ("rz_tgt",       "sum"),
        )
        .reset_index()
        .rename(columns={"full_name": "player", "posteam": "team"})
    )

    # ── Merge into per-game per-player table ───────────────────────────────────
    # Full outer join so RBs with only rush or only pass plays are included
    all_games = pd.merge(
        rush_game, pass_game,
        on=["player", "team", "game_id", "week"],
        how="outer",
    ).fillna(0)

    all_games = pd.merge(
        all_games,
        team_game,
        left_on=["team", "game_id", "week"],
        right_on=["posteam", "game_id", "week"],
        how="left",
    )

    # per-game derived metrics
    all_games["touches"]       = all_games["rush_att"] + all_games["receptions"]
    all_games["snap_proxy"]    = all_games["rush_att"] + all_games["targets"]  # for snap_pct
    all_games["rz_touches"]    = all_games["rz_rush"] + all_games["rz_tgt"]
    all_games["snap_pct_gm"]   = np.where(
        all_games["team_off_plays"] > 0,
        all_games["snap_proxy"] / all_games["team_off_plays"] * 100,
        np.nan,
    )
    all_games["route_pct_gm"]  = np.where(
        all_games["team_pass_plays"] > 0,
        all_games["targets"] / all_games["team_pass_plays"] * 100,
        np.nan,
    )

    all_games = all_games.sort_values(["player", "week"]).reset_index(drop=True)

    # ── Build rows ─────────────────────────────────────────────────────────────
    print("Building RB Usage & Role rows...")
    rows: list[dict[str, Any]] = []

    # Group by (player, team) so that name-resolution collisions (e.g. two
    # different "B.Robinson" players, one on ATL and one on SF) are kept
    # separate. After building per-team stats, deduplicate to one row per
    # player name by retaining only the team with the most games.
    player_team_rows: dict[str, list[dict]] = {}

    for (player, team), grp in all_games.groupby(["player", "team"]):
        if player not in rb_names:
            continue

        grp = grp.sort_values("week").reset_index(drop=True)
        games = int(len(grp))

        # Season totals
        season_rush_att    = int(grp["rush_att"].sum())
        season_receptions  = int(grp["receptions"].sum())
        season_touches     = int(grp["touches"].sum())
        season_targets     = int(grp["targets"].sum())
        season_rz_touches  = int(grp["rz_touches"].sum())
        season_gl_att      = int(grp["gl_att"].sum())

        if games < MIN_GAMES or season_touches < MIN_TOUCHES:
            continue

        # Season-level rates
        season_snap_pct = _r2(grp["snap_pct_gm"].mean())
        season_route_pct = _r2(grp["route_pct_gm"].mean())
        touches_per_gm  = _r2(season_touches / games)
        rush_att_per_gm = _r2(season_rush_att / games)
        targets_per_gm  = _r2(season_targets  / games)

        # Per-game records for window splits
        game_records = grp[["week", "rush_att", "touches", "targets", "snap_pct_gm", "route_pct_gm"]].to_dict("records")

        # Season averages (for Tier 3 baseline)
        season_avgs_dict = {
            "rush_att":    float(grp["rush_att"].mean()),
            "touches":     float(grp["touches"].mean()),
            "targets":     float(grp["targets"].mean()),
            "snap_pct_gm": float(grp["snap_pct_gm"].mean()),
            "route_pct_gm":float(grp["route_pct_gm"].mean()),
        }

        recent, baseline, tier = _window_split(game_records, season_avgs_dict)

        if tier == "limited":
            delta_snap        = None
            delta_touches     = None
            delta_rush_att    = None
            delta_targets     = None
            delta_route       = None
        else:
            delta_snap     = _r2(_compute_delta(recent, baseline, "snap_pct_gm"))
            delta_touches  = _r2(_compute_delta(recent, baseline, "touches"))
            delta_rush_att = _r2(_compute_delta(recent, baseline, "rush_att"))
            delta_targets  = _r2(_compute_delta(recent, baseline, "targets"))
            delta_route    = _r2(_compute_delta(recent, baseline, "route_pct_gm"))

        row = {
            "player":               player,
            "team":                 str(team),
            "games":                games,
            "snap_pct":             season_snap_pct,
            "delta_snap_pct":       delta_snap,
            "touches_per_gm":       touches_per_gm,
            "delta_touches_per_gm": delta_touches,
            "rush_att_per_gm":      rush_att_per_gm,
            "delta_rush_att_per_gm":delta_rush_att,
            "targets_per_gm":       targets_per_gm,
            "delta_targets_per_gm": delta_targets,
            "route_pct":            season_route_pct,
            "delta_route_pct":      delta_route,
            "rz_touches":           season_rz_touches,
            "goal_line_att":        season_gl_att,
            "_tier":                tier,
        }
        player_team_rows.setdefault(player, []).append(row)

    # Deduplicate: for players who appear on multiple teams (real trade or name
    # collision), retain only the team entry with the most games.
    rows: list[dict[str, Any]] = []
    for player, candidates in player_team_rows.items():
        best = max(candidates, key=lambda r: r["games"])
        rows.append(best)

    print(f"  {len(rows)} RBs before scoring.")

    # ── Usage & Role Score ─────────────────────────────────────────────────────
    # Percentile pool: all rows in `rows` (already MIN_GAMES/MIN_TOUCHES filtered)
    # For delta fields, only rows with non-null deltas participate in percentile;
    # null-delta rows receive 50.0 (neutral) for that component.

    def _pool(key: str) -> np.ndarray:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return np.array(vals, dtype=float) if vals else np.array([0.0])

    pool_delta_snap    = _pool("delta_snap_pct")
    pool_delta_touches = _pool("delta_touches_per_gm")
    pool_delta_rush    = _pool("delta_rush_att_per_gm")
    pool_delta_targets = _pool("delta_targets_per_gm")
    pool_delta_route   = _pool("delta_route_pct")
    pool_rz            = _pool("rz_touches")
    pool_gl            = _pool("goal_line_att")

    for r in rows:
        p_snap    = _pct(r["delta_snap_pct"],       pool_delta_snap)
        p_touches = _pct(r["delta_touches_per_gm"], pool_delta_touches)
        p_rush    = _pct(r["delta_rush_att_per_gm"],pool_delta_rush)
        p_targets = _pct(r["delta_targets_per_gm"], pool_delta_targets)
        p_route   = _pct(r["delta_route_pct"],      pool_delta_route)
        p_rz      = _pct(r["rz_touches"],           pool_rz)
        p_gl      = _pct(r["goal_line_att"],         pool_gl)

        r["usage_role_score"] = round(
            p_snap    * 0.22
            + p_touches * 0.22
            + p_rush    * 0.12
            + p_targets * 0.18
            + p_route   * 0.12
            + p_rz      * 0.08
            + p_gl      * 0.06,
            2,
        )

    # ── Role Trend label ───────────────────────────────────────────────────────
    def _role_trend(r: dict) -> str:
        tier  = r["_tier"]
        ds    = r["delta_snap_pct"]
        dt    = r["delta_touches_per_gm"]
        dtgt  = r["delta_targets_per_gm"]
        score = r["usage_role_score"]

        if tier == "limited":
            return "Limited Sample"

        # Falling Fast: snap drop ≤ -8 AND touch drop ≤ -3
        if ds is not None and dt is not None and ds <= -8 and dt <= -3:
            return "Falling Fast"
        # Trending Down: snap drop ≤ -4 OR touch drop ≤ -2
        if (ds is not None and ds <= -4) or (dt is not None and dt <= -2):
            return "Trending Down"
        # Rising Fast: snap gain ≥ 8 AND (touch gain ≥ 2 OR target gain ≥ 1)
        if (ds is not None and ds >= 8
                and ((dt is not None and dt >= 2) or (dtgt is not None and dtgt >= 1))):
            return "Rising Fast"
        # Trending Up: snap gain ≥ 4 AND score ≥ 65
        if ds is not None and ds >= 4 and score >= 65:
            return "Trending Up"
        # Default
        return "Stable"

    for r in rows:
        r["role_trend"] = _role_trend(r)

    # ── Sort and clean internal fields ─────────────────────────────────────────
    rows.sort(key=lambda r: r["usage_role_score"], reverse=True)
    for r in rows:
        r.pop("_tier", None)

    return rows


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    rows = build(PBP_PATH, SLEEPER_PATH)

    output = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season":     SEASON,
        "week":       None,
        "table":      "rb_usage_role",
        "columns":    COLUMNS,
        "rows":       rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.1f} KB, {len(rows)} rows)")
    print("RB Usage & Role build complete.")


if __name__ == "__main__":
    main()
