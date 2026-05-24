"""
build_qb_advanced_stats.py
──────────────────────────
Builds the QB Advanced Stats datasets for the StatChasers frontend.

Produces five output files (mirrors the WR/RB/TE advanced builders):
  output/qb_advanced_stats_2023.json   — 2023 season only
  output/qb_advanced_stats_2024.json   — 2024 season only
  output/qb_advanced_stats_2025.json   — 2025 season only
  output/qb_advanced_stats_all.json    — 2023 + 2024 + 2025 combined (one row per player)
  output/qb_advanced_stats.json        — alias for 2025 (backward compat)

This tab is a volume / pressure leaderboard — raw passing & rushing counting
stats plus PFR pressure data (pocket time, blitzes, hurries, knockdowns,
pressure %) and the NextGen "time to throw" metric.  No modeled scores or tiers.
Every QB with at least one regular-season pass attempt qualifies.

Data sources (pre-pulled by pull_nflverse_data.py):
  data/raw/nflverse_player_stats_season.parquet  — season-total counting stats,
                                                    fantasy points, fumbles, sacks
  data/raw/pfr_pass_advstats_season.parquet      — pocket time / pressure / scrambles
  data/raw/ngs_passing.parquet                   — avg_time_to_throw (season row)
  data/raw/nflverse_play_by_play.parquet         — distance buckets, LNG, RZ, deep att%
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
PFR_PASS_SEASON  = ROOT / "data" / "raw" / "pfr_pass_advstats_season.parquet"
NGS_PASSING      = ROOT / "data" / "raw" / "ngs_passing.parquet"
PBP_PATH         = ROOT / "data" / "raw" / "nflverse_play_by_play.parquet"
PLAYERS_PATH     = ROOT / "data" / "raw" / "nflverse_players.parquet"
OUTPUT_DIR       = ROOT / "output"

SEASONS = [2023, 2024, 2025]

# ---------------------------------------------------------------------------
# Column definitions (order + labels for the frontend table)
# ---------------------------------------------------------------------------

COLUMNS: list[dict] = [
    {"key": "rank",               "label": "Rank",        "type": "number",  "defaultVisible": True},
    {"key": "playerId",           "label": "Player ID",   "type": "string",  "defaultVisible": False},
    {"key": "playerName",         "label": "Player",      "type": "string",  "defaultVisible": True},
    {"key": "position",           "label": "Pos",         "type": "string",  "defaultVisible": False},
    {"key": "team",               "label": "Team",        "type": "string",  "defaultVisible": True},
    {"key": "age",                "label": "Age",         "type": "number",  "defaultVisible": False},
    {"key": "season",             "label": "Season",      "type": "number",  "defaultVisible": False},
    {"key": "games",              "label": "G",           "type": "number",  "defaultVisible": True},
    {"key": "attempts",           "label": "ATT",         "type": "number",  "defaultVisible": True},
    {"key": "completions",        "label": "CMP",         "type": "number",  "defaultVisible": True},
    {"key": "completionPct",      "label": "CMP%",        "type": "decimal", "defaultVisible": True},
    {"key": "passingYards",       "label": "YDS",         "type": "number",  "defaultVisible": True},
    {"key": "yardsPerCatch",      "label": "Y/C",         "type": "decimal", "defaultVisible": True},
    {"key": "passingTouchdowns",  "label": "Passing TDs", "type": "number",  "defaultVisible": True},
    {"key": "interceptions",      "label": "INT",         "type": "number",  "defaultVisible": True},
    {"key": "fantasyPoints",      "label": "FPTS",        "type": "decimal", "defaultVisible": True},
    {"key": "airYards",           "label": "Air YDS",     "type": "number",  "defaultVisible": False},
    {"key": "deepAttemptPct",     "label": "Deep ATT%",   "type": "decimal", "defaultVisible": True},
    {"key": "epaPerPlay",         "label": "EPA/Play",    "type": "decimal", "defaultVisible": True},
    {"key": "successRate",        "label": "Success%",    "type": "decimal", "defaultVisible": True},
    {"key": "cpoe",               "label": "CPOE",        "type": "decimal", "defaultVisible": True},
    {"key": "passes10Plus",       "label": "10+ YDS",     "type": "number",  "defaultVisible": False},
    {"key": "passes20Plus",       "label": "20+ YDS",     "type": "number",  "defaultVisible": True},
    {"key": "passes30Plus",       "label": "30+ YDS",     "type": "number",  "defaultVisible": False},
    {"key": "passes40Plus",       "label": "40+ YDS",     "type": "number",  "defaultVisible": False},
    {"key": "passes50Plus",       "label": "50+ YDS",     "type": "number",  "defaultVisible": False},
    {"key": "longestPass",        "label": "LNG",         "type": "number",  "defaultVisible": True},
    {"key": "rushAttempts",       "label": "Rush ATT",    "type": "number",  "defaultVisible": True},
    {"key": "rushYards",          "label": "Rush YDS",    "type": "number",  "defaultVisible": True},
    {"key": "rushTouchdowns",     "label": "Rush TDs",    "type": "number",  "defaultVisible": True},
    {"key": "redZoneAttempts",    "label": "RZ ATT",      "type": "number",  "defaultVisible": True},
    {"key": "scrambleRate",       "label": "SCRM%",       "type": "decimal", "defaultVisible": False},
    {"key": "fumbles",            "label": "FUM",         "type": "number",  "defaultVisible": True},
    {"key": "sacks",              "label": "SACK",        "type": "number",  "defaultVisible": True},
    {"key": "sackYardsLost",      "label": "Sack YDS",    "type": "number",  "defaultVisible": False},
    {"key": "pocketTime",         "label": "PKT TIME",    "type": "decimal", "defaultVisible": True},
    {"key": "timeToThrow",        "label": "TT Throw",    "type": "decimal", "defaultVisible": True},
    {"key": "blitzes",            "label": "BLZ",         "type": "number",  "defaultVisible": True},
    {"key": "hurries",            "label": "HRY",         "type": "number",  "defaultVisible": True},
    {"key": "knockdowns",         "label": "KD",          "type": "number",  "defaultVisible": True},
    {"key": "pressurePct",        "label": "PRSS%",       "type": "decimal", "defaultVisible": True},
]

# Counting fields summed when collapsing across seasons for the *_all file.
_SUM_FIELDS = [
    "games", "attempts", "completions", "passingYards", "passingTouchdowns",
    "interceptions", "fantasyPoints", "airYards",
    "passes10Plus", "passes20Plus", "passes30Plus", "passes40Plus", "passes50Plus",
    "rushAttempts", "rushYards", "rushTouchdowns", "redZoneAttempts",
    "fumbles", "sacks", "sackYardsLost",
    "blitzes", "hurries", "knockdowns",
    "_deepAttempts", "_scrambles", "_pbpAttempts",
    "_officialAtt", "_epaSum", "_successPlays",
]
# Rate fields averaged across seasons weighted by pass attempts (cannot be
# reconstructed from summed totals without the underlying weighting).
_WEIGHTED_RATE_FIELDS = ["pocketTime", "timeToThrow", "pressurePct", "cpoe"]
# Weighted-rate fields rounded to 1 decimal (others default to 2).
_ONE_DP_RATE_FIELDS = {"pressurePct", "cpoe"}


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
    """Age (years) as of Sept 1 of the given season."""
    if not birth_date or (isinstance(birth_date, float) and birth_date != birth_date):
        return None
    try:
        b = datetime.strptime(str(birth_date)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    ref = date(season, 9, 1)
    return ref.year - b.year - ((ref.month, ref.day) < (b.month, b.day))


# ---------------------------------------------------------------------------
# Play-by-play aggregation (distance buckets, LNG, red zone, deep attempts)
# ---------------------------------------------------------------------------

def _aggregate_pbp(pbp_season: pd.DataFrame) -> dict[str, dict]:
    """Per-passer (gsis id) PBP-derived stats for one season."""
    passes = pbp_season[pbp_season["pass_attempt"] == 1].copy()
    passes = passes[passes["passer_player_id"].notna()]
    if passes.empty:
        return {}

    out: dict[str, dict] = {}
    for pid, grp in passes.groupby("passer_player_id"):
        comp = grp[grp["complete_pass"] == 1]
        comp_yds = comp["yards_gained"]
        air = grp["air_yards"]
        # Official attempts = dropbacks minus sacks (matches the EPA/play and
        # success-rate definition used in compute_player_metrics.py so the same
        # QB reports identical values across the platform).
        official = grp[grp["sack"] != 1] if "sack" in grp.columns else grp
        official_att = int(len(official))
        out[str(pid)] = {
            "passes10Plus":  int((comp_yds >= 10).sum()),
            "passes20Plus":  int((comp_yds >= 20).sum()),
            "passes30Plus":  int((comp_yds >= 30).sum()),
            "passes40Plus":  int((comp_yds >= 40).sum()),
            "passes50Plus":  int((comp_yds >= 50).sum()),
            "longestPass":   int(comp_yds.max()) if len(comp_yds) else None,
            "redZoneAttempts": int((grp["yardline_100"] <= 20).sum()),
            "_deepAttempts": int((air >= 20).sum()),
            "_pbpAttempts":  int(len(grp)),
            "_officialAtt":  official_att,
            "_epaSum":       float(official["epa"].dropna().sum()),
            "_successPlays": float(official["success"].dropna().sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Per-season builder
# ---------------------------------------------------------------------------

def build_season(
    season: int,
    stats: pd.DataFrame,
    pfr_by_gsis: dict[str, dict],
    ngs_by_gsis: dict[str, dict],
    pbp_by_gsis: dict[str, dict],
    birth_by_gsis: dict[str, str],
    resolver: PlayerIdResolver,
) -> list[dict]:
    qb = stats[(stats["position"] == "QB") & (stats["season"] == season)].copy()
    qb = qb[qb["attempts"].fillna(0) > 0]

    rows: list[dict] = []
    for _, s in qb.iterrows():
        gsis = str(s["player_id"])
        name = s.get("player_display_name") or s.get("player_name")
        team = s.get("recent_team")

        att = _int(s.get("attempts"))
        cmp = _int(s.get("completions"))
        yds = _int(s.get("passing_yards"))
        air_yards = _int(s.get("passing_air_yards"))

        pfr = pfr_by_gsis.get(gsis, {})
        ngs = ngs_by_gsis.get(gsis, {})
        pbp = pbp_by_gsis.get(gsis, {})

        sacks = _int(s.get("sacks_suffered"))
        scrambles = _int(pfr.get("scrambles"))
        # Scramble rate = scrambles / dropbacks (attempts + sacks + scrambles)
        dropbacks = (att or 0) + (sacks or 0) + (scrambles or 0)
        scramble_rate = _round(scrambles / dropbacks * 100, 1) if scrambles and dropbacks else None

        deep_att = pbp.get("_deepAttempts")
        pbp_att  = pbp.get("_pbpAttempts")
        deep_pct = _round(deep_att / pbp_att * 100, 1) if deep_att is not None and pbp_att else None

        # Efficiency metrics over official dropbacks (sacks excluded), matching
        # the canonical efficiency-tab definitions. CPOE is a season-level mean.
        official_att  = pbp.get("_officialAtt")
        epa_sum       = pbp.get("_epaSum")
        success_plays = pbp.get("_successPlays")
        epa_per_play  = _round(epa_sum / official_att, 3) if epa_sum is not None and official_att else None
        success_rate  = _round(success_plays / official_att * 100, 1) if success_plays is not None and official_att else None

        sack_fumbles = s.get("sack_fumbles") or 0
        rush_fumbles = s.get("rushing_fumbles") or 0
        fumbles = _int(sack_fumbles + rush_fumbles)

        rows.append({
            "playerId":           resolver.resolve(name, team, "QB"),
            "playerName":         name,
            "position":           "QB",
            "team":               team,
            "age":                _age_at_season(birth_by_gsis.get(gsis), season),
            "season":             season,
            "games":              _int(s.get("games")),
            "attempts":           att,
            "completions":        cmp,
            "completionPct":      _round(cmp / att * 100, 1) if cmp is not None and att else None,
            "passingYards":       yds,
            # NOTE: yardsPerCatch retains the yards-per-attempt calculation (yards/attempts) by request.
            "yardsPerCatch":      _round(yds / att, 2) if yds is not None and att else None,
            "passingTouchdowns":  _int(s.get("passing_tds")),
            "interceptions":      _int(s.get("passing_interceptions")),
            "fantasyPoints":      _round(s.get("fantasy_points"), 1),
            "airYards":           air_yards,
            "deepAttemptPct":     deep_pct,
            "epaPerPlay":         epa_per_play,
            "successRate":        success_rate,
            "cpoe":               _round(s.get("passing_cpoe"), 1),
            "passes10Plus":       pbp.get("passes10Plus"),
            "passes20Plus":       pbp.get("passes20Plus"),
            "passes30Plus":       pbp.get("passes30Plus"),
            "passes40Plus":       pbp.get("passes40Plus"),
            "passes50Plus":       pbp.get("passes50Plus"),
            "longestPass":        pbp.get("longestPass"),
            "rushAttempts":       _int(s.get("carries")),
            "rushYards":          _int(s.get("rushing_yards")),
            "rushTouchdowns":     _int(s.get("rushing_tds")),
            "redZoneAttempts":    pbp.get("redZoneAttempts"),
            "scrambleRate":       scramble_rate,
            "fumbles":            fumbles,
            "sacks":              sacks,
            # nflverse stores sack_yards_lost as a negative value; report the
            # positive magnitude of yards lost.
            "sackYardsLost":      abs(_int(s.get("sack_yards_lost"))) if s.get("sack_yards_lost") is not None else None,
            "pocketTime":         _round(pfr.get("pocket_time"), 2),
            "timeToThrow":        _round(ngs.get("avg_time_to_throw"), 2),
            "blitzes":            _int(pfr.get("times_blitzed")),
            "hurries":            _int(pfr.get("times_hurried")),
            "knockdowns":         _int(pfr.get("times_hit")),
            "pressurePct":        _round(pfr.get("pressure_pct"), 1),
            # internal-only carry-overs for the combined *_all aggregation
            "_deepAttempts":      deep_att,
            "_pbpAttempts":       pbp_att,
            "_scrambles":         scrambles,
            "_officialAtt":       official_att,
            "_epaSum":            epa_sum,
            "_successPlays":      success_plays,
        })

    _rank_and_sort(rows)
    return rows


def _rank_and_sort(rows: list[dict]) -> None:
    rows.sort(key=lambda r: (r.get("passingYards") or 0, r.get("attempts") or 0), reverse=True)
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
            "position":   "QB",
            "team":       latest.get("team"),
            "age":        latest.get("age"),
            "season":     "all",
        }
        for f in _SUM_FIELDS:
            vals = [r[f] for r in prows if r.get(f) is not None]
            c[f] = sum(vals) if vals else None

        for f in _WEIGHTED_RATE_FIELDS:
            num = den = 0.0
            for r in prows:
                v, w = r.get(f), r.get("attempts")
                if v is not None and w:
                    num += v * w
                    den += w
            c[f] = round(num / den, 1 if f in _ONE_DP_RATE_FIELDS else 2) if den else None

        att = c.get("attempts") or 0
        cmp = c.get("completions") or 0
        yds = c.get("passingYards") or 0
        c["completionPct"]      = round(cmp / att * 100, 1) if att else None
        c["yardsPerCatch"]      = round(yds / att, 2) if att else None
        deep = c.get("_deepAttempts")
        pbp_att = c.get("_pbpAttempts")
        c["deepAttemptPct"] = round(deep / pbp_att * 100, 1) if deep is not None and pbp_att else None
        scr = c.get("_scrambles")
        dropbacks = att + (c.get("sacks") or 0) + (scr or 0)
        c["scrambleRate"] = round(scr / dropbacks * 100, 1) if scr and dropbacks else None
        c["fantasyPoints"] = round(c["fantasyPoints"], 1) if c.get("fantasyPoints") is not None else None

        # EPA/play and success rate recomputed from summed official-dropback totals.
        off_att = c.get("_officialAtt")
        epa_sum = c.get("_epaSum")
        suc = c.get("_successPlays")
        c["epaPerPlay"]  = round(epa_sum / off_att, 3) if epa_sum is not None and off_att else None
        c["successRate"] = round(suc / off_att * 100, 1) if suc is not None and off_att else None

        lng = [r["longestPass"] for r in prows if r.get("longestPass") is not None]
        c["longestPass"] = max(lng) if lng else None

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
        "table":      "qb_advanced_stats",
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
        print(f"  NOTE: {len(no_id)} QB(s) without a Sleeper playerId in {label}: "
              f"{', '.join(no_id[:8])}{'…' if len(no_id) > 8 else ''}", file=sys.stderr)
    print(f"  Validation OK for {label}: {len(rows)} QBs.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (STATS_SEASON, PFR_PASS_SEASON, NGS_PASSING, PBP_PATH, PLAYERS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run pull_nflverse_data.py first.", file=sys.stderr)
            sys.exit(1)

    print("Loading season-total player stats...")
    stats = pd.read_parquet(STATS_SEASON)

    print("Loading seasonal PFR pass advstats...")
    pfr = pd.read_parquet(PFR_PASS_SEASON)

    print("Loading NextGen passing...")
    ngs = pd.read_parquet(NGS_PASSING)

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

    resolver = PlayerIdResolver()
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for season in SEASONS:
        print(f"\nBuilding {season} season...")

        # PFR seasonal → keyed by gsis via pfr_id map
        pfr_s = pfr[pfr["season"] == season]
        pfr_by_gsis: dict[str, dict] = {}
        for _, r in pfr_s.iterrows():
            g = pfr_to_gsis.get(str(r.get("pfr_id")))
            if g:
                pfr_by_gsis[g] = r.to_dict()

        ngs_s = ngs[ngs["season"] == season]
        ngs_by_gsis = {
            str(r["player_gsis_id"]): r.to_dict()
            for _, r in ngs_s.iterrows()
            if pd.notna(r.get("player_gsis_id"))
        }

        pbp_by_gsis = _aggregate_pbp(pbp[pbp["season"] == season])

        week = int(pbp[pbp["season"] == season]["week"].max()) if not pbp[pbp["season"] == season].empty else None

        rows = build_season(
            season, stats, pfr_by_gsis, ngs_by_gsis, pbp_by_gsis,
            birth_by_gsis, resolver,
        )
        _validate(rows, str(season))

        out_path = OUTPUT_DIR / f"qb_advanced_stats_{season}.json"
        with open(out_path, "w") as f:
            json.dump(_make_payload(rows, season, week, updated_at), f, indent=2)
        print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        all_rows.extend(rows)

    # Combined multi-season file
    combined = _aggregate_combined(all_rows)
    _validate(combined, "all-seasons")
    multi_path = OUTPUT_DIR / "qb_advanced_stats_all.json"
    with open(multi_path, "w") as f:
        json.dump(_make_payload(combined, "all", None, updated_at), f, indent=2)
    print(f"\nWrote {multi_path} ({multi_path.stat().st_size / 1024:.1f} KB, {len(combined)} players combined)")

    # Alias → 2025
    alias_path = OUTPUT_DIR / "qb_advanced_stats.json"
    shutil.copy2(OUTPUT_DIR / "qb_advanced_stats_2025.json", alias_path)
    print(f"Wrote {alias_path} (alias for 2025)")

    print("\nQB advanced stats build complete.")


if __name__ == "__main__":
    main()
