"""
build_rb_advanced_stats.py
──────────────────────────
Builds the RB Advanced Stats datasets for the StatChasers frontend.

Produces five output files (mirrors the QB/WR/TE advanced builders):
  output/rb_advanced_stats_2023.json   — 2023 season only
  output/rb_advanced_stats_2024.json   — 2024 season only
  output/rb_advanced_stats_2025.json   — 2025 season only
  output/rb_advanced_stats_all.json    — 2023 + 2024 + 2025 combined (one row per player)
  output/rb_advanced_stats.json        — alias for 2025 (backward compat)

A volume / efficiency / elusiveness leaderboard covering every RB with at least
one regular-season touch (carry or target).  Columns include the shared identity
fields, rushing & receiving counting stats, usage (snap %, routes, red-zone /
goal-line / end-zone opportunities), efficiency (EPA/play, explosive %, breakaway
%, YPRR), and PFR contact/elusiveness metrics (YBC/att, YAC/att, broken tackles).

Definitions match the canonical RB efficiency tab so a RB reports consistent
values across the platform:
  - explosive run  = rush of >= 10 yards
  - breakaway run  = rush of >= 15 yards
  - YBC/att, YAC/att, broken tackles come from PFR rush advanced stats
  - tacklesForLoss / negative-yardage rushes are derived from PBP (PFR has no TFL
    field for rushers); the two name pairs carry identical values.

Data sources (pre-pulled by pull_nflverse_data.py):
  data/raw/nflverse_player_stats_season.parquet  — counting stats, receiving,
                                                    target share, fantasy pts, fumbles
  data/raw/pfr_rush_advstats.parquet             — YBC, YAC, broken tackles
  data/raw/nflverse_play_by_play.parquet         — EPA, explosive/breakaway, buckets,
                                                    longest, negative yards, RZ/EZ/GL
  data/raw/nflverse_participation.parquet        — routes (pass-play participation)
  data/raw/nflverse_players.parquet              — birth_date (age), pfr_id↔gsis map
  data/raw/sleeper_players.json                  — Sleeper playerId resolution
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from schema_config import PlayerIdResolver

ROOT             = Path(__file__).resolve().parent.parent
STATS_SEASON     = ROOT / "data" / "raw" / "nflverse_player_stats_season.parquet"
PFR_RUSH         = ROOT / "data" / "raw" / "pfr_rush_advstats.parquet"
PBP_PATH         = ROOT / "data" / "raw" / "nflverse_play_by_play.parquet"
SNAP_COUNTS      = ROOT / "data" / "raw" / "nflverse_snap_counts.parquet"
PARTICIPATION    = ROOT / "data" / "raw" / "nflverse_participation.parquet"
PLAYERS_PATH     = ROOT / "data" / "raw" / "nflverse_players.parquet"
OUTPUT_DIR       = ROOT / "output"

SEASONS = [2023, 2024, 2025]

# ---------------------------------------------------------------------------
# Column definitions (order + labels for the frontend table)
# ---------------------------------------------------------------------------

COLUMNS: list[dict] = [
    {"key": "rank",                         "label": "Rank",          "type": "number",  "defaultVisible": True},
    {"key": "playerId",                     "label": "Player ID",     "type": "string",  "defaultVisible": False},
    {"key": "playerName",                   "label": "Player",        "type": "string",  "defaultVisible": True},
    {"key": "position",                     "label": "Pos",           "type": "string",  "defaultVisible": False},
    {"key": "team",                         "label": "Team",          "type": "string",  "defaultVisible": True},
    {"key": "age",                          "label": "Age",           "type": "number",  "defaultVisible": False},
    {"key": "season",                       "label": "Season",        "type": "number",  "defaultVisible": False},
    {"key": "games",                        "label": "G",             "type": "number",  "defaultVisible": True},
    {"key": "snapPct",                      "label": "Snap %",        "type": "decimal", "defaultVisible": True},
    {"key": "routes",                       "label": "Routes",        "type": "number",  "defaultVisible": True},
    {"key": "rushAttempts",                 "label": "Rush ATT",      "type": "number",  "defaultVisible": True},
    {"key": "redZoneOpportunities",         "label": "RZ Opp",        "type": "number",  "defaultVisible": True},
    {"key": "goalLineCarries",              "label": "GL ATT",        "type": "number",  "defaultVisible": True},
    {"key": "targets",                      "label": "TGT",           "type": "number",  "defaultVisible": True},
    {"key": "receptions",                   "label": "REC",           "type": "number",  "defaultVisible": True},
    {"key": "redZoneTargets",               "label": "RZ TGT",        "type": "number",  "defaultVisible": True},
    {"key": "endZoneTargets",               "label": "EZ TGT",        "type": "number",  "defaultVisible": True},
    {"key": "rushYards",                    "label": "Rush YDS",      "type": "number",  "defaultVisible": True},
    {"key": "yardsPerCarry",                "label": "Y/C",           "type": "decimal", "defaultVisible": True},
    {"key": "rushTouchdowns",               "label": "Rush TDs",      "type": "number",  "defaultVisible": True},
    {"key": "targetSharePct",               "label": "TGT Share %",   "type": "decimal", "defaultVisible": True},
    {"key": "receivingYards",               "label": "Rec YDS",       "type": "number",  "defaultVisible": True},
    {"key": "yardsPerReception",            "label": "Rec Y/R",       "type": "decimal", "defaultVisible": True},
    {"key": "yardsPerRouteRun",             "label": "YPRR",          "type": "decimal", "defaultVisible": True},
    {"key": "receivingTouchdowns",          "label": "Rec TDs",       "type": "number",  "defaultVisible": True},
    {"key": "epaPerPlay",                   "label": "EPA/Play",      "type": "decimal", "defaultVisible": True},
    {"key": "fantasyPoints",                "label": "FPTS",          "type": "decimal", "defaultVisible": True},
    {"key": "rushes10Plus",                 "label": "10+ YDS",       "type": "number",  "defaultVisible": False},
    {"key": "rushes20Plus",                 "label": "20+ YDS",       "type": "number",  "defaultVisible": True},
    {"key": "rushes30Plus",                 "label": "30+ YDS",       "type": "number",  "defaultVisible": False},
    {"key": "rushes40Plus",                 "label": "40+ YDS",       "type": "number",  "defaultVisible": False},
    {"key": "rushes50Plus",                 "label": "50+ YDS",       "type": "number",  "defaultVisible": False},
    {"key": "explosiveRunPct",              "label": "Explosive %",   "type": "decimal", "defaultVisible": True},
    {"key": "breakawayRunPct",              "label": "Breakaway %",   "type": "decimal", "defaultVisible": True},
    {"key": "longestRush",                  "label": "LNG",           "type": "number",  "defaultVisible": True},
    {"key": "longestRushTouchdown",         "label": "LNG TD",        "type": "number",  "defaultVisible": True},
    {"key": "brokenTackles",                "label": "BRKTKL",        "type": "number",  "defaultVisible": True},
    {"key": "rushAttemptsPerBrokenTackle",  "label": "Att/BRKTKL",    "type": "decimal", "defaultVisible": True},
    {"key": "yardsAfterContactPerAttempt",  "label": "YAC/Att",       "type": "decimal", "defaultVisible": True},
    {"key": "yardsBeforeContactPerAttempt", "label": "YBC/Att",       "type": "decimal", "defaultVisible": True},
    {"key": "receivingYardsAfterCatch",     "label": "Rec YAC",       "type": "number",  "defaultVisible": True},
    {"key": "tackleEludedRate",             "label": "Tkl Eluded %",  "type": "decimal", "defaultVisible": True},
    {"key": "fumbles",                      "label": "FUM",           "type": "number",  "defaultVisible": True},
    {"key": "tacklesForLoss",               "label": "TFL",           "type": "number",  "defaultVisible": True},
    {"key": "tacklesForLossYards",          "label": "TFL YDS",       "type": "number",  "defaultVisible": True},
    {"key": "rushAttForNegativeYards",      "label": "Neg Rush ATT",  "type": "number",  "defaultVisible": False},
]

# Counting fields summed when collapsing across seasons for the *_all file.
_SUM_FIELDS = [
    "games", "routes", "rushAttempts", "redZoneOpportunities", "goalLineCarries",
    "targets", "receptions", "endZoneTargets", "rushYards", "rushTouchdowns",
    "receivingYards", "receivingTouchdowns", "receivingYardsAfterCatch",
    "fantasyPoints", "brokenTackles", "fumbles",
    "rushes10Plus", "rushes20Plus", "rushes30Plus", "rushes40Plus", "rushes50Plus",
    "redZoneTargets",
    "tacklesForLoss", "tacklesForLossYards", "rushAttForNegativeYards",
    "_carriesPbp", "_explosive10", "_breakaway15",
    "_epaSum", "_epaPlays", "_ybcTotal", "_yacTotal", "_pfrCarries",
    "_snapPctXgames",
]


def _round(v: Any, n: int) -> Any:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(f, n)


def _int(v: Any) -> Any:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return int(round(f))


def _age_at_season(birth_date: Any, season: int) -> int | None:
    if not birth_date or (isinstance(birth_date, float) and birth_date != birth_date):
        return None
    try:
        b = datetime.strptime(str(birth_date)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    ref = date(season, 9, 1)
    return ref.year - b.year - ((ref.month, ref.day) < (b.month, b.day))


# ---------------------------------------------------------------------------
# Play-by-play aggregation (per gsis): rush + receiving derived stats
# ---------------------------------------------------------------------------

def _aggregate_pbp(pbp_season: pd.DataFrame) -> tuple[dict[str, dict], dict[str, dict], dict[str, int]]:
    """Return (rush_by_gsis, tgt_by_gsis, team_off_plays) for one season."""
    rush = pbp_season[pbp_season["rush_attempt"] == 1]
    rush = rush[rush["rusher_player_id"].notna()]
    passes = pbp_season[pbp_season["pass_attempt"] == 1]
    targeted = passes[passes["receiver_player_id"].notna()]

    has_yl = "yardline_100" in pbp_season.columns

    rush_by_gsis: dict[str, dict] = {}
    has_td = "touchdown" in rush.columns
    for pid, grp in rush.groupby("rusher_player_id"):
        yds = grp["yards_gained"]
        neg = grp[yds < 0]
        longest = int(yds.max()) if len(yds) else None
        # Longest rushing TD = the yardage of this player's longest rush that
        # scored a touchdown (null if he had no rushing TDs).
        longest_td = None
        if has_td:
            td_rushes = grp[grp["touchdown"] == 1]["yards_gained"]
            longest_td = int(td_rushes.max()) if len(td_rushes) else None
        rush_by_gsis[str(pid)] = {
            "_carriesPbp":  int(len(grp)),
            "rushYardsPbp": int(yds.sum()),
            "_explosive10": int((yds >= 10).sum()),
            "_breakaway15": int((yds >= 15).sum()),
            "rushes10Plus": int((yds >= 10).sum()),
            "rushes20Plus": int((yds >= 20).sum()),
            "rushes30Plus": int((yds >= 30).sum()),
            "rushes40Plus": int((yds >= 40).sum()),
            "rushes50Plus": int((yds >= 50).sum()),
            "longestRush":  longest,
            "longestRushTd": longest_td,
            "negCount":     int(len(neg)),
            "negYards":     int(abs(neg["yards_gained"].sum())) if len(neg) else 0,
            "goalLine":     int((grp["yardline_100"] <= 5).sum()) if has_yl else None,
            "rzCarries":    int((grp["yardline_100"] <= 20).sum()) if has_yl else 0,
            "epaRush":      float(grp["epa"].dropna().sum()),
        }

    tgt_by_gsis: dict[str, dict] = {}
    for pid, grp in targeted.groupby("receiver_player_id"):
        tgt_by_gsis[str(pid)] = {
            "targetsPbp":   int(len(grp)),
            "rzTargets":    int((grp["yardline_100"] <= 20).sum()) if has_yl else 0,
            "endZoneTgts":  int((grp["yardline_100"] <= 10).sum()) if has_yl else 0,
            "epaTarget":    float(grp["epa"].dropna().sum()),
        }

    off = pbp_season[(pbp_season["rush_attempt"] == 1) | (pbp_season["pass_attempt"] == 1)]
    team_off_plays = off.groupby("posteam")["play_id"].count().to_dict()
    return rush_by_gsis, tgt_by_gsis, team_off_plays


# ---------------------------------------------------------------------------
# PFR rush aggregation (per gsis via pfr_id map)
# ---------------------------------------------------------------------------

def _aggregate_pfr(pfr_season: pd.DataFrame, pfr_to_gsis: dict[str, str]) -> dict[str, dict]:
    agg = (
        pfr_season.groupby("pfr_player_id")
        .agg(
            ybc=("rushing_yards_before_contact", "sum"),
            yac=("rushing_yards_after_contact", "sum"),
            carries=("carries", "sum"),
            broken_tackles=("rushing_broken_tackles", "sum"),
        )
        .to_dict("index")
    )
    out: dict[str, dict] = {}
    for pfr_id, vals in agg.items():
        g = pfr_to_gsis.get(str(pfr_id))
        if g:
            out[g] = vals
    return out


# ---------------------------------------------------------------------------
# Routes from participation (per gsis): count of REG pass plays the RB was on
# ---------------------------------------------------------------------------

def _build_routes(part: pd.DataFrame, pbp_pass_keys: pd.DataFrame, season: int) -> dict[str, int]:
    p = part[(part["season"] == season)
             & part["offense_players"].notna()
             & part["offense_positions"].notna()].copy()
    if p.empty:
        return {}
    p = p.merge(
        pbp_pass_keys, left_on=["nflverse_game_id", "play_id"],
        right_on=["game_id", "play_id"], how="inner",
    )
    if p.empty:
        return {}
    p["pid_list"] = p["offense_players"].str.split(";")
    p["pos_list"] = p["offense_positions"].str.split(";")
    exp = p[["pid_list", "pos_list"]].explode(["pid_list", "pos_list"])
    exp["gsis_id"] = exp["pid_list"].str.strip()
    exp["position"] = exp["pos_list"].str.strip()
    rb = exp[exp["position"] == "RB"]
    return rb.groupby("gsis_id").size().astype(int).to_dict()


# ---------------------------------------------------------------------------
# True snap share from snap counts (per gsis): season mean of offense_pct
# ---------------------------------------------------------------------------

def _build_snap_pct(sc_season: pd.DataFrame, pfr_to_gsis: dict[str, str]) -> dict[str, float]:
    if sc_season.empty:
        return {}
    means = sc_season.groupby("pfr_player_id")["offense_pct"].mean().to_dict()
    out: dict[str, float] = {}
    for pfr_id, pct in means.items():
        g = pfr_to_gsis.get(str(pfr_id))
        if g and pct is not None and pct == pct:
            out[g] = round(float(pct) * 100, 1)
    return out


# ---------------------------------------------------------------------------
# Per-season builder
# ---------------------------------------------------------------------------

def build_season(
    season: int,
    stats: pd.DataFrame,
    rush_by_gsis: dict[str, dict],
    tgt_by_gsis: dict[str, dict],
    team_off_plays: dict[str, int],
    pfr_by_gsis: dict[str, dict],
    routes_by_gsis: dict[str, int],
    snap_by_gsis: dict[str, float],
    birth_by_gsis: dict[str, str],
    resolver: PlayerIdResolver,
) -> list[dict]:
    rb = stats[(stats["position"] == "RB") & (stats["season"] == season)].copy()

    rows: list[dict] = []
    for _, s in rb.iterrows():
        gsis = str(s["player_id"])
        carries     = _int(s.get("carries")) or 0
        targets     = _int(s.get("targets")) or 0
        if carries == 0 and targets == 0:
            continue

        name = s.get("player_display_name") or s.get("player_name")
        team = s.get("recent_team")

        pbp_r = rush_by_gsis.get(gsis, {})
        pbp_t = tgt_by_gsis.get(gsis, {})
        pfr   = pfr_by_gsis.get(gsis, {})

        rush_yds   = _int(s.get("rushing_yards"))
        receptions = _int(s.get("receptions"))
        rec_yds    = _int(s.get("receiving_yards"))

        # Usage
        carries_pbp = pbp_r.get("_carriesPbp") or 0
        targets_pbp = pbp_t.get("targetsPbp") or 0
        # True snap share from snap counts (season mean of per-game offense_pct).
        snap_pct = snap_by_gsis.get(gsis)
        routes   = routes_by_gsis.get(gsis)
        rz_opp   = (pbp_r.get("rzCarries") or 0) + (pbp_t.get("rzTargets") or 0)

        # Efficiency (rates over official carries; counts/EPA from PBP)
        ypc = _round(rush_yds / carries, 2) if rush_yds is not None and carries else None
        explosive_pct = _round(pbp_r.get("_explosive10", 0) / carries * 100, 2) if carries else None
        breakaway_pct = _round(pbp_r.get("_breakaway15", 0) / carries * 100, 2) if carries else None
        epa_plays = carries_pbp + targets_pbp
        epa_sum   = (pbp_r.get("epaRush", 0.0)) + (pbp_t.get("epaTarget", 0.0))
        epa_play  = _round(epa_sum / epa_plays, 3) if epa_plays else None
        yprr = _round(rec_yds / routes, 2) if rec_yds is not None and routes else None
        ypr  = _round(rec_yds / receptions, 2) if rec_yds is not None and receptions else None

        # PFR contact / elusiveness
        ybc_total = pfr.get("ybc")
        yac_total = pfr.get("yac")
        pfr_carr  = pfr.get("carries") or 0
        bt        = _int(pfr.get("broken_tackles"))
        ybc_att = _round(float(ybc_total) / pfr_carr, 2) if ybc_total is not None and pfr_carr else None
        yac_att = _round(float(yac_total) / pfr_carr, 2) if yac_total is not None and pfr_carr else None
        att_per_bt = _round(carries / bt, 2) if bt and carries else None
        touches = carries + (receptions or 0)
        tkl_eluded = _round(bt / touches * 100, 1) if bt is not None and touches else None

        # Fumbles (rushing + receiving + sack)
        fum = sum(int(s.get(f) or 0) for f in ("rushing_fumbles", "receiving_fumbles", "sack_fumbles"))

        ts = s.get("target_share")
        target_share_pct = _round(float(ts) * 100, 1) if ts is not None and not (isinstance(ts, float) and ts != ts) else None

        neg_count = pbp_r.get("negCount") or 0
        neg_yards = pbp_r.get("negYards") or 0

        rows.append({
            "playerId":                     resolver.resolve(name, team, "RB"),
            "playerName":                   name,
            "position":                     "RB",
            "team":                         team,
            "age":                          _age_at_season(birth_by_gsis.get(gsis), season),
            "season":                       season,
            "games":                        _int(s.get("games")),
            "snapPct":                      snap_pct,
            "routes":                       routes,
            "rushAttempts":                 carries,
            "redZoneOpportunities":         rz_opp,
            "goalLineCarries":              pbp_r.get("goalLine"),
            "targets":                      targets,
            "receptions":                   receptions,
            "redZoneTargets":               pbp_t.get("rzTargets"),
            "endZoneTargets":               pbp_t.get("endZoneTgts"),
            "rushYards":                    rush_yds,
            "yardsPerCarry":                ypc,
            "rushTouchdowns":               _int(s.get("rushing_tds")),
            "targetSharePct":               target_share_pct,
            "receivingYards":               rec_yds,
            "yardsPerReception":            ypr,
            "yardsPerRouteRun":             yprr,
            "receivingTouchdowns":          _int(s.get("receiving_tds")),
            "epaPerPlay":                   epa_play,
            "fantasyPoints":                _round(s.get("fantasy_points"), 1),
            "rushes10Plus":                 pbp_r.get("rushes10Plus"),
            "rushes20Plus":                 pbp_r.get("rushes20Plus"),
            "rushes30Plus":                 pbp_r.get("rushes30Plus"),
            "rushes40Plus":                 pbp_r.get("rushes40Plus"),
            "rushes50Plus":                 pbp_r.get("rushes50Plus"),
            "explosiveRunPct":              explosive_pct,
            "breakawayRunPct":              breakaway_pct,
            "longestRush":                  pbp_r.get("longestRush"),
            "longestRushTouchdown":         pbp_r.get("longestRushTd"),
            "brokenTackles":                bt,
            "rushAttemptsPerBrokenTackle":  att_per_bt,
            "yardsAfterContactPerAttempt":  yac_att,
            "yardsBeforeContactPerAttempt": ybc_att,
            "receivingYardsAfterCatch":     _int(s.get("receiving_yards_after_catch")),
            "tackleEludedRate":             tkl_eluded,
            "fumbles":                      fum,
            "tacklesForLoss":               neg_count,
            "tacklesForLossYards":          neg_yards,
            "rushAttForNegativeYards":      neg_count,
            # internal carry-overs for combined *_all aggregation
            "_carriesPbp":   carries_pbp,
            "_explosive10":  pbp_r.get("_explosive10", 0),
            "_breakaway15":  pbp_r.get("_breakaway15", 0),
            "_epaSum":       epa_sum,
            "_epaPlays":     epa_plays,
            "_ybcTotal":     float(ybc_total) if ybc_total is not None else 0.0,
            "_yacTotal":     float(yac_total) if yac_total is not None else 0.0,
            "_pfrCarries":   pfr_carr,
            "_snapPctXgames": (snap_pct or 0) * (_int(s.get("games")) or 0),
        })

    _rank_and_sort(rows)
    return rows


def _rank_and_sort(rows: list[dict]) -> None:
    rows.sort(key=lambda r: (r.get("rushYards") or 0) + (r.get("receivingYards") or 0), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i


# ---------------------------------------------------------------------------
# Multi-season aggregation (one combined row per player)
# ---------------------------------------------------------------------------

def _aggregate_combined(rows_with_season: list[dict]) -> list[dict]:
    from collections import defaultdict

    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in rows_with_season:
        key = r.get("playerId") or r.get("playerName")
        by_player[key].append(r)

    result: list[dict] = []
    for _, prows in by_player.items():
        latest = max(prows, key=lambda r: r.get("season", 0))
        c: dict[str, Any] = {
            "playerId":   latest.get("playerId"),
            "playerName": latest.get("playerName"),
            "position":   "RB",
            "team":       latest.get("team"),
            "age":        latest.get("age"),
            "season":     "all",
        }
        for f in _SUM_FIELDS:
            vals = [r[f] for r in prows if r.get(f) is not None]
            c[f] = sum(vals) if vals else None

        games   = c.get("games") or 0
        carries = c.get("rushAttempts") or 0
        rush_yds = c.get("rushYards") or 0
        rec_yds  = c.get("receivingYards") or 0
        receptions = c.get("receptions") or 0
        routes  = c.get("routes") or 0
        carries_pbp = c.get("_carriesPbp") or 0
        pfr_carr = c.get("_pfrCarries") or 0
        bt = c.get("brokenTackles")
        epa_plays = c.get("_epaPlays") or 0

        c["yardsPerCarry"]      = round(rush_yds / carries, 2) if carries else None
        c["explosiveRunPct"]    = round((c.get("_explosive10") or 0) / carries_pbp * 100, 2) if carries_pbp else None
        c["breakawayRunPct"]    = round((c.get("_breakaway15") or 0) / carries_pbp * 100, 2) if carries_pbp else None
        c["epaPerPlay"]         = round((c.get("_epaSum") or 0) / epa_plays, 3) if epa_plays else None
        c["yardsPerReception"]  = round(rec_yds / receptions, 2) if receptions else None
        c["yardsPerRouteRun"]   = round(rec_yds / routes, 2) if routes else None
        c["rushAttemptsPerBrokenTackle"] = round(carries / bt, 2) if bt and carries else None
        touches = carries + receptions
        c["tackleEludedRate"]   = round(bt / touches * 100, 1) if bt is not None and touches else None
        c["yardsBeforeContactPerAttempt"] = round((c.get("_ybcTotal") or 0) / pfr_carr, 2) if pfr_carr else None
        c["yardsAfterContactPerAttempt"]  = round((c.get("_yacTotal") or 0) / pfr_carr, 2) if pfr_carr else None
        c["fantasyPoints"]      = round(c["fantasyPoints"], 1) if c.get("fantasyPoints") is not None else None
        c["snapPct"]            = round((c.get("_snapPctXgames") or 0) / games, 1) if games else None
        c["targetSharePct"]     = None  # not meaningful across seasons
        lng = [r["longestRush"] for r in prows if r.get("longestRush") is not None]
        c["longestRush"] = max(lng) if lng else None
        lng_td = [r["longestRushTouchdown"] for r in prows if r.get("longestRushTouchdown") is not None]
        c["longestRushTouchdown"] = max(lng_td) if lng_td else None

        result.append(c)

    _rank_and_sort(result)
    return result


# ---------------------------------------------------------------------------
# Payload + validation
# ---------------------------------------------------------------------------

def _make_payload(rows: list[dict], season: int | str, week: int | None, updated_at: str) -> dict:
    keys = [c["key"] for c in COLUMNS]
    ordered = [{k: r.get(k) for k in keys} for r in rows]
    return {
        "updated_at": updated_at,
        "season":     season,
        "week":       week,
        "table":      "rb_advanced_stats",
        "columns":    COLUMNS,
        "rows":       ordered,
    }


def _validate(rows: list[dict], label: str) -> None:
    seen: set[str] = set()
    for r in rows:
        pid = r.get("playerId") or r.get("playerName")
        if pid in seen:
            print(f"  WARNING: duplicate player '{r.get('playerName')}' in {label}", file=sys.stderr)
        seen.add(pid)
    no_id = [r["playerName"] for r in rows if not r.get("playerId")]
    if no_id:
        print(f"  NOTE: {len(no_id)} RB(s) without a Sleeper playerId in {label}: "
              f"{', '.join(no_id[:8])}{'…' if len(no_id) > 8 else ''}", file=sys.stderr)
    print(f"  Validation OK for {label}: {len(rows)} RBs.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (STATS_SEASON, PFR_RUSH, PBP_PATH, PLAYERS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run pull_nflverse_data.py first.", file=sys.stderr)
            sys.exit(1)

    print("Loading season-total player stats...")
    stats = pd.read_parquet(STATS_SEASON)

    print("Loading PFR rush advstats...")
    pfr = pd.read_parquet(PFR_RUSH)

    print("Loading player registry (birth_date, pfr_id↔gsis)...")
    players = pd.read_parquet(PLAYERS_PATH)
    pfr_to_gsis = {
        str(r["pfr_id"]): str(r["gsis_id"])
        for _, r in players.iterrows()
        if pd.notna(r.get("pfr_id")) and pd.notna(r.get("gsis_id"))
    }
    birth_by_gsis = {
        str(r["gsis_id"]): r["birth_date"]
        for _, r in players.iterrows()
        if pd.notna(r.get("gsis_id"))
    }

    print("Loading play-by-play (regular season)...")
    pbp = pd.read_parquet(PBP_PATH)
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    if "two_point_attempt" in pbp.columns:
        pbp = pbp[pbp["two_point_attempt"].fillna(0) != 1].copy()

    have_snaps = SNAP_COUNTS.exists()
    if have_snaps:
        print("Loading snap counts...")
        snaps = pd.read_parquet(SNAP_COUNTS)
    else:
        print("  WARN: snap counts parquet missing — snapPct will be null.", file=sys.stderr)
        snaps = None

    have_part = PARTICIPATION.exists()
    if have_part:
        print("Loading participation (routes)...")
        part = pd.read_parquet(
            PARTICIPATION,
            columns=["nflverse_game_id", "play_id", "season", "offense_players", "offense_positions"],
        )
        pass_keys = pbp[pbp["play_type"] == "pass"][["game_id", "play_id", "season"]]
    else:
        print("  WARN: participation parquet missing — routes / YPRR will be null.", file=sys.stderr)
        part, pass_keys = None, None

    resolver = PlayerIdResolver()
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for season in SEASONS:
        print(f"\nBuilding {season} season...")
        pbp_s = pbp[pbp["season"] == season]
        rush_by_gsis, tgt_by_gsis, team_off_plays = _aggregate_pbp(pbp_s)
        pfr_by_gsis = _aggregate_pfr(pfr[pfr["season"] == season], pfr_to_gsis)
        if have_part:
            routes_by_gsis = _build_routes(
                part, pass_keys[pass_keys["season"] == season][["game_id", "play_id"]], season,
            )
        else:
            routes_by_gsis = {}
        snap_by_gsis = _build_snap_pct(snaps[snaps["season"] == season], pfr_to_gsis) if have_snaps else {}
        week = int(pbp_s["week"].max()) if not pbp_s.empty else None

        rows = build_season(
            season, stats, rush_by_gsis, tgt_by_gsis, team_off_plays,
            pfr_by_gsis, routes_by_gsis, snap_by_gsis, birth_by_gsis, resolver,
        )
        _validate(rows, str(season))

        out_path = OUTPUT_DIR / f"rb_advanced_stats_{season}.json"
        with open(out_path, "w") as f:
            json.dump(_make_payload(rows, season, week, updated_at), f, indent=2)
        print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        all_rows.extend(rows)

    combined = _aggregate_combined(all_rows)
    _validate(combined, "all-seasons")
    multi_path = OUTPUT_DIR / "rb_advanced_stats_all.json"
    with open(multi_path, "w") as f:
        json.dump(_make_payload(combined, "all", None, updated_at), f, indent=2)
    print(f"\nWrote {multi_path} ({multi_path.stat().st_size / 1024:.1f} KB, {len(combined)} players combined)")

    alias_path = OUTPUT_DIR / "rb_advanced_stats.json"
    shutil.copy2(OUTPUT_DIR / "rb_advanced_stats_2025.json", alias_path)
    print(f"Wrote {alias_path} (alias for 2025)")

    print("\nRB advanced stats build complete.")


if __name__ == "__main__":
    main()
