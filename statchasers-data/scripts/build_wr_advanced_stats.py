"""
build_wr_advanced_stats.py
──────────────────────────
Builds the WR Advanced Stats datasets for the StatChasers frontend.

Produces five output files (mirrors the QB/RB advanced builders):
  output/wr_advanced_stats_2023.json   — 2023 season only
  output/wr_advanced_stats_2024.json   — 2024 season only
  output/wr_advanced_stats_2025.json   — 2025 season only
  output/wr_advanced_stats_all.json    — 2023 + 2024 + 2025 combined (one row per player)
  output/wr_advanced_stats.json        — alias for 2025 (backward compat)

A target-volume / efficiency / separation leaderboard covering every WR with at
least one regular-season target.  Columns include the shared identity fields,
target/route usage (snap %, routes, TPRR, target share, air-yards share, WOPR),
receiving production, efficiency (EPA/play, success rate, YPRR, YBC/YAC per rec),
and PFR contact metrics (broken tackles, drops, INTs when targeted).

Definitions match the canonical WR efficiency tab so a WR reports consistent
values across the platform (EPA/target, success rate, YPRR, TPRR, YBC/rec,
broken tackles from PFR).

Data sources (pre-pulled by pull_nflverse_data.py):
  data/raw/nflverse_player_stats_season.parquet  — receiving counting stats,
                                                    target/air-yards share, WOPR, fantasy pts
  data/raw/pfr_rec_advstats.parquet              — broken tackles, drops, INTs when targeted
  data/raw/nflverse_play_by_play.parquet         — EPA, success, buckets, longest, RZ/EZ
  data/raw/nflverse_participation.parquet        — routes (pass-play participation)
  data/raw/nflverse_snap_counts.parquet          — true snap share
  data/raw/nflverse_players.parquet              — birth_date (age), pfr_id↔gsis map
  data/raw/sleeper_players.json                  — Sleeper playerId resolution

NOTE: contestedCatchRate has no source in nflverse / PFR (it is a charting metric)
and is emitted as null.
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
PFR_REC          = ROOT / "data" / "raw" / "pfr_rec_advstats.parquet"
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
    {"key": "rank",                          "label": "Rank",          "type": "number",  "defaultVisible": True},
    {"key": "playerId",                      "label": "Player ID",     "type": "string",  "defaultVisible": False},
    {"key": "playerName",                    "label": "Player",        "type": "string",  "defaultVisible": True},
    {"key": "position",                      "label": "Pos",           "type": "string",  "defaultVisible": False},
    {"key": "team",                          "label": "Team",          "type": "string",  "defaultVisible": True},
    {"key": "age",                           "label": "Age",           "type": "number",  "defaultVisible": False},
    {"key": "season",                        "label": "Season",        "type": "number",  "defaultVisible": False},
    {"key": "games",                         "label": "G",             "type": "number",  "defaultVisible": True},
    {"key": "snapPct",                       "label": "Snap %",        "type": "decimal", "defaultVisible": True},
    {"key": "routes",                        "label": "Routes",        "type": "number",  "defaultVisible": True},
    {"key": "targets",                       "label": "TGT",           "type": "number",  "defaultVisible": True},
    {"key": "targetsPerRouteRun",            "label": "TPRR",          "type": "decimal", "defaultVisible": True},
    {"key": "targetSharePct",                "label": "TGT Share %",   "type": "decimal", "defaultVisible": True},
    {"key": "airYardsPerTarget",             "label": "AY/TGT",        "type": "decimal", "defaultVisible": True},
    {"key": "airYardsSharePct",              "label": "AY Share %",    "type": "decimal", "defaultVisible": True},
    {"key": "wopr",                          "label": "WOPR",          "type": "decimal", "defaultVisible": True},
    {"key": "receptions",                    "label": "REC",           "type": "number",  "defaultVisible": True},
    {"key": "redZoneTargets",                "label": "RZ TGT",        "type": "number",  "defaultVisible": True},
    {"key": "endZoneTargets",                "label": "EZ TGT",        "type": "number",  "defaultVisible": True},
    {"key": "catchPct",                      "label": "Catch %",       "type": "decimal", "defaultVisible": True},
    {"key": "receivingYards",                "label": "Rec YDS",       "type": "number",  "defaultVisible": True},
    {"key": "yardsPerReception",             "label": "Y/R",           "type": "decimal", "defaultVisible": True},
    {"key": "yardsBeforeCatchPerReception",  "label": "YBC/R",         "type": "decimal", "defaultVisible": True},
    {"key": "yardsAfterCatchPerReception",   "label": "YAC/R",         "type": "decimal", "defaultVisible": True},
    {"key": "yardsPerRouteRun",              "label": "YPRR",          "type": "decimal", "defaultVisible": True},
    {"key": "receivingTouchdowns",           "label": "Rec TDs",       "type": "number",  "defaultVisible": True},
    {"key": "epaPerPlay",                    "label": "EPA/Play",      "type": "decimal", "defaultVisible": True},
    {"key": "successRate",                   "label": "Success %",     "type": "decimal", "defaultVisible": True},
    {"key": "contestedCatchRate",            "label": "Contested %",   "type": "decimal", "defaultVisible": False},
    {"key": "fantasyPoints",                 "label": "FPTS",          "type": "decimal", "defaultVisible": True},
    {"key": "avoidedTackleRate",             "label": "Avoid Tkl %",   "type": "decimal", "defaultVisible": True},
    {"key": "brokenTackles",                 "label": "BRKTKL",        "type": "number",  "defaultVisible": True},
    {"key": "receptionsPerBrokenTackle",     "label": "REC/BRKTKL",    "type": "decimal", "defaultVisible": False},
    {"key": "drops",                         "label": "Drops",         "type": "number",  "defaultVisible": True},
    {"key": "dropPct",                       "label": "Drop %",        "type": "decimal", "defaultVisible": True},
    {"key": "interceptionsWhenTargeted",     "label": "INT (tgt)",     "type": "number",  "defaultVisible": True},
    {"key": "receptions10Plus",              "label": "10+ YDS",       "type": "number",  "defaultVisible": False},
    {"key": "receptions20Plus",              "label": "20+ YDS",       "type": "number",  "defaultVisible": True},
    {"key": "receptions30Plus",              "label": "30+ YDS",       "type": "number",  "defaultVisible": False},
    {"key": "receptions40Plus",              "label": "40+ YDS",       "type": "number",  "defaultVisible": False},
    {"key": "receptions50Plus",              "label": "50+ YDS",       "type": "number",  "defaultVisible": False},
    {"key": "longestReception",              "label": "LNG",           "type": "number",  "defaultVisible": True},
]

# Counting fields summed when collapsing across seasons for the *_all file.
_SUM_FIELDS = [
    "games", "routes", "targets", "receptions", "redZoneTargets", "endZoneTargets",
    "receivingYards", "receivingTouchdowns", "fantasyPoints",
    "brokenTackles", "drops", "interceptionsWhenTargeted",
    "receptions10Plus", "receptions20Plus", "receptions30Plus",
    "receptions40Plus", "receptions50Plus",
    "_recYds", "_recYac", "_recAirYards",
    "_epaSum", "_epaCount", "_successCount",
    "_snapPctXgames",
]


def _round(v: Any, n: int) -> Any:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
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
# Play-by-play aggregation (per receiver gsis): EPA, success, buckets, RZ/EZ
# ---------------------------------------------------------------------------

def _aggregate_pbp(pbp_season: pd.DataFrame) -> dict[str, dict]:
    targeted = pbp_season[(pbp_season["pass_attempt"] == 1)
                          & pbp_season["receiver_player_id"].notna()]
    has_yl = "yardline_100" in pbp_season.columns

    out: dict[str, dict] = {}
    for pid, grp in targeted.groupby("receiver_player_id"):
        epa = grp["epa"].dropna()
        comps = grp[grp["complete_pass"] == 1]
        cy = comps["yards_gained"]
        out[str(pid)] = {
            "targetsPbp":   int(len(grp)),
            "_epaSum":      float(epa.sum()) if len(epa) else 0.0,
            "_epaCount":    int(len(epa)),
            "_successCount": int((epa > 0).sum()) if len(epa) else 0,
            "receptions10Plus": int((cy >= 10).sum()),
            "receptions20Plus": int((cy >= 20).sum()),
            "receptions30Plus": int((cy >= 30).sum()),
            "receptions40Plus": int((cy >= 40).sum()),
            "receptions50Plus": int((cy >= 50).sum()),
            "longestReception": int(cy.max()) if len(cy) else None,
            "redZoneTargets":   int((grp["yardline_100"] <= 20).sum()) if has_yl else 0,
            "endZoneTargets":   int((grp["yardline_100"] <= 10).sum()) if has_yl else 0,
        }
    return out


def _aggregate_pfr(pfr_season: pd.DataFrame, pfr_to_gsis: dict[str, str]) -> dict[str, dict]:
    agg = (
        pfr_season.groupby("pfr_player_id")
        .agg(
            broken_tackles=("receiving_broken_tackles", "sum"),
            drops=("receiving_drop", "sum"),
            ints=("receiving_int", "sum"),
        )
        .to_dict("index")
    )
    out: dict[str, dict] = {}
    for pfr_id, vals in agg.items():
        g = pfr_to_gsis.get(str(pfr_id))
        if g:
            out[g] = vals
    return out


def _build_routes(part: pd.DataFrame, pbp_pass_keys: pd.DataFrame, season: int) -> dict[str, int]:
    p = part[(part["season"] == season)
             & part["offense_players"].notna()
             & part["offense_positions"].notna()].copy()
    if p.empty:
        return {}
    p = p.merge(pbp_pass_keys, left_on=["nflverse_game_id", "play_id"],
                right_on=["game_id", "play_id"], how="inner")
    if p.empty:
        return {}
    p["pid_list"] = p["offense_players"].str.split(";")
    p["pos_list"] = p["offense_positions"].str.split(";")
    exp = p[["pid_list", "pos_list"]].explode(["pid_list", "pos_list"])
    exp["gsis_id"] = exp["pid_list"].str.strip()
    exp["position"] = exp["pos_list"].str.strip()
    skill = exp[exp["position"].isin(["WR", "RB", "TE"])]
    return skill.groupby("gsis_id").size().astype(int).to_dict()


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
    pbp_by_gsis: dict[str, dict],
    pfr_by_gsis: dict[str, dict],
    routes_by_gsis: dict[str, int],
    snap_by_gsis: dict[str, float],
    birth_by_gsis: dict[str, str],
    resolver: PlayerIdResolver,
) -> list[dict]:
    wr = stats[(stats["position"] == "WR") & (stats["season"] == season)].copy()

    rows: list[dict] = []
    for _, s in wr.iterrows():
        gsis = str(s["player_id"])
        targets = _int(s.get("targets")) or 0
        if targets == 0:
            continue

        name = s.get("player_display_name") or s.get("player_name")
        team = s.get("recent_team")

        receptions = _int(s.get("receptions")) or 0
        rec_yds    = _int(s.get("receiving_yards"))
        rec_yac    = _int(s.get("receiving_yards_after_catch"))
        rec_air    = _int(s.get("receiving_air_yards"))

        pbp = pbp_by_gsis.get(gsis, {})
        pfr = pfr_by_gsis.get(gsis, {})
        routes = routes_by_gsis.get(gsis)

        # Usage / volume
        tprr = _round(targets / routes, 3) if routes else None
        ts   = s.get("target_share")
        ays  = s.get("air_yards_share")
        target_share_pct = _round(float(ts) * 100, 1) if ts is not None and not (isinstance(ts, float) and ts != ts) else None
        air_yards_share_pct = _round(float(ays) * 100, 1) if ays is not None and not (isinstance(ays, float) and ays != ays) else None
        air_per_tgt = _round(rec_air / targets, 2) if rec_air is not None and targets else None

        # Production / efficiency
        catch_pct = _round(receptions / targets * 100, 1) if targets else None
        ypr  = _round(rec_yds / receptions, 2) if rec_yds is not None and receptions else None
        yac_per_rec = _round(rec_yac / receptions, 2) if rec_yac is not None and receptions else None
        ybc_per_rec = _round((rec_yds - rec_yac) / receptions, 2) if rec_yds is not None and rec_yac is not None and receptions else None
        yprr = _round(rec_yds / routes, 2) if rec_yds is not None and routes else None

        epa_count = pbp.get("_epaCount") or 0
        epa_play  = _round(pbp.get("_epaSum", 0.0) / epa_count, 3) if epa_count else None
        success_rate = _round(pbp.get("_successCount", 0) / epa_count * 100, 1) if epa_count else None

        # PFR contact / drops / INTs
        bt    = _int(pfr.get("broken_tackles"))
        drops = _int(pfr.get("drops"))
        ints  = _int(pfr.get("ints"))
        avoided_rate = _round(bt / receptions * 100, 1) if bt is not None and receptions else None
        rec_per_bt   = _round(receptions / bt, 1) if bt else None
        drop_pct     = _round(drops / targets * 100, 1) if drops is not None and targets else None

        rows.append({
            "playerId":                     resolver.resolve(name, team, "WR"),
            "playerName":                   name,
            "position":                     "WR",
            "team":                         team,
            "age":                          _age_at_season(birth_by_gsis.get(gsis), season),
            "season":                       season,
            "games":                        _int(s.get("games")),
            "snapPct":                      snap_by_gsis.get(gsis),
            "routes":                       routes,
            "targets":                      targets,
            "targetsPerRouteRun":           tprr,
            "targetSharePct":               target_share_pct,
            "airYardsPerTarget":            air_per_tgt,
            "airYardsSharePct":             air_yards_share_pct,
            "wopr":                         _round(s.get("wopr"), 3),
            "receptions":                   receptions,
            "redZoneTargets":               pbp.get("redZoneTargets"),
            "endZoneTargets":               pbp.get("endZoneTargets"),
            "catchPct":                     catch_pct,
            "receivingYards":               rec_yds,
            "yardsPerReception":            ypr,
            "yardsBeforeCatchPerReception": ybc_per_rec,
            "yardsAfterCatchPerReception":  yac_per_rec,
            "yardsPerRouteRun":             yprr,
            "receivingTouchdowns":          _int(s.get("receiving_tds")),
            "epaPerPlay":                   epa_play,
            "successRate":                  success_rate,
            "contestedCatchRate":           None,   # no source in nflverse / PFR
            "fantasyPoints":                _round(s.get("fantasy_points"), 1),
            "avoidedTackleRate":            avoided_rate,
            "brokenTackles":                bt,
            "receptionsPerBrokenTackle":    rec_per_bt,
            "drops":                        drops,
            "dropPct":                      drop_pct,
            "interceptionsWhenTargeted":    ints,
            "receptions10Plus":             pbp.get("receptions10Plus"),
            "receptions20Plus":             pbp.get("receptions20Plus"),
            "receptions30Plus":             pbp.get("receptions30Plus"),
            "receptions40Plus":             pbp.get("receptions40Plus"),
            "receptions50Plus":             pbp.get("receptions50Plus"),
            "longestReception":             pbp.get("longestReception"),
            # internal carry-overs for combined *_all aggregation
            "_recYds":       rec_yds,
            "_recYac":       rec_yac,
            "_recAirYards":  rec_air,
            "_epaSum":       pbp.get("_epaSum", 0.0),
            "_epaCount":     epa_count,
            "_successCount": pbp.get("_successCount", 0),
            "_snapPctXgames": (snap_by_gsis.get(gsis) or 0) * (_int(s.get("games")) or 0),
        })

    _rank_and_sort(rows)
    return rows


def _rank_and_sort(rows: list[dict]) -> None:
    rows.sort(key=lambda r: r.get("receivingYards") or 0, reverse=True)
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
            "position":   "WR",
            "team":       latest.get("team"),
            "age":        latest.get("age"),
            "season":     "all",
        }
        for f in _SUM_FIELDS:
            vals = [r[f] for r in prows if r.get(f) is not None]
            c[f] = sum(vals) if vals else None

        games      = c.get("games") or 0
        targets    = c.get("targets") or 0
        receptions = c.get("receptions") or 0
        rec_yds    = c.get("_recYds") or 0
        rec_yac    = c.get("_recYac") or 0
        rec_air    = c.get("_recAirYards") or 0
        routes     = c.get("routes") or 0
        bt         = c.get("brokenTackles")
        drops      = c.get("drops")
        epa_count  = c.get("_epaCount") or 0

        c["receivingYards"]   = rec_yds
        c["targetsPerRouteRun"] = round(targets / routes, 3) if routes else None
        c["airYardsPerTarget"]  = round(rec_air / targets, 2) if targets else None
        c["catchPct"]         = round(receptions / targets * 100, 1) if targets else None
        c["yardsPerReception"] = round(rec_yds / receptions, 2) if receptions else None
        c["yardsAfterCatchPerReception"] = round(rec_yac / receptions, 2) if receptions else None
        c["yardsBeforeCatchPerReception"] = round((rec_yds - rec_yac) / receptions, 2) if receptions else None
        c["yardsPerRouteRun"] = round(rec_yds / routes, 2) if routes else None
        c["epaPerPlay"]       = round((c.get("_epaSum") or 0) / epa_count, 3) if epa_count else None
        c["successRate"]      = round((c.get("_successCount") or 0) / epa_count * 100, 1) if epa_count else None
        c["avoidedTackleRate"] = round(bt / receptions * 100, 1) if bt is not None and receptions else None
        c["receptionsPerBrokenTackle"] = round(receptions / bt, 1) if bt else None
        c["dropPct"]          = round(drops / targets * 100, 1) if drops is not None and targets else None
        c["fantasyPoints"]    = round(c["fantasyPoints"], 1) if c.get("fantasyPoints") is not None else None
        c["snapPct"]          = round((c.get("_snapPctXgames") or 0) / games, 1) if games else None
        c["contestedCatchRate"] = None
        c["targetSharePct"]   = None   # not meaningful across seasons
        c["airYardsSharePct"] = None
        c["wopr"]             = None
        lng = [r["longestReception"] for r in prows if r.get("longestReception") is not None]
        c["longestReception"] = max(lng) if lng else None

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
        "table":      "wr_advanced_stats",
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
        print(f"  NOTE: {len(no_id)} WR(s) without a Sleeper playerId in {label}: "
              f"{', '.join(no_id[:8])}{'…' if len(no_id) > 8 else ''}", file=sys.stderr)
    print(f"  Validation OK for {label}: {len(rows)} WRs.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (STATS_SEASON, PFR_REC, PBP_PATH, PLAYERS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run pull_nflverse_data.py first.", file=sys.stderr)
            sys.exit(1)

    print("Loading season-total player stats...")
    stats = pd.read_parquet(STATS_SEASON)

    print("Loading PFR rec advstats...")
    pfr = pd.read_parquet(PFR_REC)

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
    snaps = pd.read_parquet(SNAP_COUNTS) if have_snaps else None
    if not have_snaps:
        print("  WARN: snap counts parquet missing — snapPct will be null.", file=sys.stderr)

    have_part = PARTICIPATION.exists()
    if have_part:
        print("Loading participation (routes)...")
        part = pd.read_parquet(
            PARTICIPATION,
            columns=["nflverse_game_id", "play_id", "season", "offense_players", "offense_positions"],
        )
        pass_keys = pbp[pbp["play_type"] == "pass"][["game_id", "play_id", "season"]]
    else:
        print("  WARN: participation parquet missing — routes / YPRR / TPRR will be null.", file=sys.stderr)
        part, pass_keys = None, None

    resolver = PlayerIdResolver()
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for season in SEASONS:
        print(f"\nBuilding {season} season...")
        pbp_s = pbp[pbp["season"] == season]
        pbp_by_gsis = _aggregate_pbp(pbp_s)
        pfr_by_gsis = _aggregate_pfr(pfr[pfr["season"] == season], pfr_to_gsis)
        routes_by_gsis = (
            _build_routes(part, pass_keys[pass_keys["season"] == season][["game_id", "play_id"]], season)
            if have_part else {}
        )
        snap_by_gsis = _build_snap_pct(snaps[snaps["season"] == season], pfr_to_gsis) if have_snaps else {}
        week = int(pbp_s["week"].max()) if not pbp_s.empty else None

        rows = build_season(
            season, stats, pbp_by_gsis, pfr_by_gsis, routes_by_gsis,
            snap_by_gsis, birth_by_gsis, resolver,
        )
        _validate(rows, str(season))

        out_path = OUTPUT_DIR / f"wr_advanced_stats_{season}.json"
        with open(out_path, "w") as f:
            json.dump(_make_payload(rows, season, week, updated_at), f, indent=2)
        print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        all_rows.extend(rows)

    combined = _aggregate_combined(all_rows)
    _validate(combined, "all-seasons")
    multi_path = OUTPUT_DIR / "wr_advanced_stats_all.json"
    with open(multi_path, "w") as f:
        json.dump(_make_payload(combined, "all", None, updated_at), f, indent=2)
    print(f"\nWrote {multi_path} ({multi_path.stat().st_size / 1024:.1f} KB, {len(combined)} players combined)")

    alias_path = OUTPUT_DIR / "wr_advanced_stats.json"
    shutil.copy2(OUTPUT_DIR / "wr_advanced_stats_2025.json", alias_path)
    print(f"Wrote {alias_path} (alias for 2025)")

    print("\nWR advanced stats build complete.")


if __name__ == "__main__":
    main()
