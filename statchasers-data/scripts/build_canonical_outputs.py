"""
build_canonical_outputs.py
──────────────────────────
Project legacy snake_case position JSONs into the canonical camelCase tree
at output/positions/<pos>/<tab>_<season>.json.

Tabs (per Performance Analytics frontend):
  overview, efficiency, usage, stat_explorer

Seasons:
  2023, 2024, 2025, all
  Per-season files are produced only when source data exists for that
  (position, tab, season) — no fake data.  Historical seasons that
  cannot be produced are reported as `unavailableSeasons` in the
  per-position manifest.

Identity / source-of-truth contracts
------------------------------------
- Every row carries: playerId, playerName, position, team, age, season,
  games.
- Player Overview owns: roleScore / efficiencyScore / overallScore /
  tier / careerArc / experienceTier / stabilityScore.  Other tabs that
  include these copy them from Overview.
- Stat Explorer rows are scrubbed of modeled / interpretive fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from schema_config import (
    CANONICAL_DIR,
    CURRENT_SEASON,
    FIELD_SCHEMA,
    HISTORICAL_SEASONS,
    OUTPUT_DIR,
    POSITIONS,
    SEASONS,
    STAT_EXPLORER_FORBIDDEN,
    TABS,
    PlayerIdResolver,
    canonical_path,
    derive_tier,
    make_envelope,
    write_global_manifest,
    write_json,
    write_position_manifest,
    _norm_name_base,
)


def _norm_player(name: str | None) -> str:
    """Suffix/accent-insensitive name key for matching advanced-stats rows."""
    return _norm_name_base(name) if name else ""


# ---------------------------------------------------------------------------
# Legacy file resolver
# ---------------------------------------------------------------------------

def _legacy_for(pos: str, tab: str, season: str) -> dict[str, str]:
    """Return the legacy filename candidates we'll read for one slot.

    Each value is a filename relative to OUTPUT_DIR.  The caller checks
    existence — missing files just mean "source data unavailable for
    this season" and the slot is omitted.

    Returns a dict so projection functions can request specific sources
    by role (overview vs efficiency vs advanced vs stat_explorer vs trends).
    """
    s = season  # "2023" / "2024" / "2025"
    files: dict[str, str] = {}

    # --- Player Overview ----------------------------------------------------
    if pos == "qb":
        files["overview"] = f"qb_player_overview_{s}.json"
    elif pos == "rb":
        # Current season uses the canonical unsuffixed alias; historical
        # seasons use the suffixed copies written by build_prior_seasons.
        files["overview"] = (
            "rb_player_overview.json" if s == CURRENT_SEASON
            else f"rb_player_overview_{s}.json"
        )
    elif pos in ("wr", "te"):
        files["overview"] = f"{pos}_player_overview_{s}.json"

    # --- Efficiency analytics ----------------------------------------------
    if pos == "qb":
        files["efficiency"] = f"qb_efficiency_analytics_{s}.json"
    elif pos == "rb":
        files["efficiency"] = (
            "rb_efficiency_analytics.json" if s == CURRENT_SEASON
            else f"rb_efficiency_analytics_{s}.json"
        )
    elif pos == "wr":
        files["efficiency"] = f"wr_efficiency_analytics_{s}.json"
    elif pos == "te":
        files["efficiency"] = f"te_efficiency_analytics_{s}.json"

    # --- Usage / trends ----------------------------------------------------
    if pos == "qb":
        # 2025 uses the canonical unsuffixed qb_trends.json; prior seasons
        # use the suffixed copies emitted by build_prior_seasons.py.
        files["trends"] = (
            "qb_trends.json" if s == CURRENT_SEASON
            else f"qb_trends_{s}.json"
        )
    elif pos == "rb":
        files["trends"] = (
            "rb_usage_role.json" if s == CURRENT_SEASON
            else f"rb_usage_role_{s}.json"
        )
    elif pos == "wr":
        files["trends"] = f"wr_trends_{s}.json"
    elif pos == "te":
        files["trends"] = f"te_trends_{s}.json"

    # --- Stat Explorer ------------------------------------------------------
    files["stat_explorer"] = f"stat_explorer_{pos}_{s}.json"

    # --- Advanced stats (merged into stat_explorer + efficiency) ----------
    if pos == "rb":
        files["advanced"] = f"rb_advanced_stats_{s}.json"
    elif pos in ("wr", "te"):
        files["advanced"] = f"{pos}_advanced_stats_{s}.json"

    return files


def _load(name: str) -> dict | list | None:
    if not name:
        return None
    fp = OUTPUT_DIR / name
    if not fp.exists() or not fp.is_file():
        return None
    with open(fp) as f:
        return json.load(f)


def _rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("rows", "players"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    return []


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


def _project(row_template: list[str], data: dict[str, Any]) -> dict[str, Any]:
    return {k: data.get(k) for k in row_template}


def _trend_label(delta: float | None) -> str | None:
    if delta is None:
        return None
    try:
        v = float(delta)
    except (TypeError, ValueError):
        return None
    if v >= 5:   return "Rising Fast"
    if v >= 2:   return "Rising"
    if v >= -2:  return "Stable"
    if v >= -5:  return "Declining"
    return "Falling Fast"


def _percentile_score(values: list[float | None], v: float | None) -> float | None:
    if v is None:
        return None
    clean = [float(x) for x in values if x is not None]
    if not clean:
        return None
    below = sum(1 for x in clean if x < v)
    equal = sum(1 for x in clean if x == v)
    return round((below + 0.5 * equal) / len(clean) * 100.0, 1)


def _identity(
    *,
    resolver: PlayerIdResolver,
    position: str,
    season: str,
    player: str | None,
    team: str | None,
    age: Any,
    games: Any,
) -> dict[str, Any]:
    return {
        "playerId":   resolver.resolve(player, team, position=position),
        "playerName": player,
        "position":   position.upper(),
        "team":       team,
        "age":        age,
        "season":     season,
        "games":      games,
    }


# ---------------------------------------------------------------------------
# Per-(position, tab) projection functions
# ---------------------------------------------------------------------------
# Each takes (resolver, season, legacy_files) and returns
#   (canonical rows, list of legacy source filenames)
# or (None, []) when the required source data is missing.

# ── RB ─────────────────────────────────────────────────────────────────────

def _project_rb_overview(resolver, season, lf):
    payload = _load(lf.get("overview", ""))
    if not payload:
        return None, []
    out: list[dict] = []
    for r in _rows(payload):
        out.append({
            **_identity(
                resolver=resolver, position="rb", season=season,
                player=r.get("player"), team=r.get("team"),
                age=r.get("age"), games=r.get("games"),
            ),
            "snapPct":                   r.get("snap_pct"),
            "touchesPerGame":            r.get("touches_per_gm"),
            "rushAttemptsPerGame":       r.get("rush_att_per_gm"),
            "targetsPerGame":            r.get("targets_per_gm"),
            "routesPerGame":             r.get("routes_per_gm"),
            "redZoneAttempts":           r.get("rz_rush_att"),
            "goalLineAttempts":          r.get("goal_line_att"),
            "targetSharePct":            r.get("target_share_pct"),
            "receptions":                r.get("receptions"),
            "receivingYards":            r.get("receiving_yds"),
            "yardsPerTouch":             r.get("yards_per_touch"),
            "explosiveRunPct":           r.get("explosive_run_pct"),
            "breakawayRunPct":           r.get("breakaway_run_pct"),
            "fantasyPointsOverExpected": r.get("fpoe"),
            "stabilityScore":            r.get("stability"),
            "volatilityScore":           r.get("volatility"),
            "careerArc":                 r.get("career_arc"),
            "experienceTier":            r.get("exp_tier"),
            "roleScore":                 r.get("role_score"),
            "efficiencyScore":           r.get("efficiency_score"),
            "overallScore":              r.get("overall_score"),
            "tier":                      r.get("rb_tier"),
        })
    return out, [lf["overview"]]


def _project_rb_efficiency(resolver, season, lf):
    payload = _load(lf.get("efficiency", ""))
    if not payload:
        return None, []
    overview = _load(lf.get("overview", "")) or {}
    ov_idx   = {r["player"]: r for r in _rows(overview)}
    out: list[dict] = []
    for r in _rows(payload):
        ov = ov_idx.get(r["player"], {})
        out.append({
            **_identity(
                resolver=resolver, position="rb", season=season,
                player=r.get("player"), team=r.get("team"),
                age=ov.get("age"), games=r.get("games"),
            ),
            "rushAttempts":                 r.get("rush_attempts"),
            "epaPerRush":                   r.get("epa_per_rush"),
            "successRate":                  r.get("success_rate"),
            "yardsPerCarry":                r.get("yards_per_attempt"),
            "yardsPerTouch":                r.get("yards_per_touch"),
            "fantasyPointsOverExpected":    r.get("fpoe"),
            "explosiveRunPct":              r.get("explosive_run_pct"),
            "breakawayRunPct":              r.get("breakaway_run_pct"),
            "breakawayRuns":                r.get("breakaway_runs"),
            "longestRush":                  r.get("longest_run"),
            "brokenTackles":                r.get("broken_tackles"),
            "brokenTacklesPerAttempt":      r.get("broken_tackles_per_att"),
            "yardsAfterContactPerAttempt":  r.get("yac_per_att"),
            "yardsBeforeContactPerAttempt": r.get("ybc_per_att"),
            "efficiencyScore":              ov.get("efficiency_score", r.get("efficiency_score")),
        })
    sources = [lf["efficiency"]]
    if overview:
        sources.append(lf["overview"])
    return out, sources


def _project_rb_usage(resolver, season, lf):
    payload = _load(lf.get("trends", ""))
    if not payload:
        return None, []
    overview = _load(lf.get("overview", "")) or {}
    ov_idx   = {r["player"]: r for r in _rows(overview)}
    out: list[dict] = []
    for r in _rows(payload):
        ov = ov_idx.get(r["player"], {})
        out.append({
            **_identity(
                resolver=resolver, position="rb", season=season,
                player=r.get("player"), team=r.get("team"),
                age=ov.get("age"), games=r.get("games"),
            ),
            "snapPct":                  r.get("snap_pct"),
            "deltaSnapPct":             r.get("delta_snap_pct"),
            "touchesPerGame":           r.get("touches_per_gm"),
            "deltaTouchesPerGame":      r.get("delta_touches_per_gm"),
            "rushAttemptsPerGame":      r.get("rush_att_per_gm"),
            "deltaRushAttemptsPerGame": r.get("delta_rush_att_per_gm"),
            "targetsPerGame":           r.get("targets_per_gm"),
            "deltaTargetsPerGame":      r.get("delta_targets_per_gm"),
            "routesPerGame":            ov.get("routes_per_gm"),
            "routePct":                 r.get("route_pct"),
            "deltaRoutePct":            r.get("delta_route_pct"),
            "redZoneTouches":           r.get("rz_touches"),
            "goalLineAttempts":         r.get("goal_line_att"),
            "roleTrend":                r.get("role_trend"),
            "usageScore":               r.get("usage_role_score"),
        })
    sources = [lf["trends"]]
    if overview:
        sources.append(lf["overview"])
    return out, sources


def _project_rb_stat_explorer(resolver, season, lf):
    se = _load(lf.get("stat_explorer", ""))
    if not se:
        return None, []
    adv = _load(lf.get("advanced", "")) or {}
    # rb_advanced_stats now uses camelCase keys + playerId/playerName identity.
    # Match primarily on the resolved Sleeper playerId (robust against the
    # abbreviated names stat_explorer sometimes carries, e.g. "J.Taylor"), then
    # fall back to a suffix/accent-insensitive name key.
    adv_idx_id   = {r.get("playerId"): r for r in _rows(adv) if r.get("playerId")}
    adv_idx_norm = {_norm_player(r.get("playerName")): r for r in _rows(adv) if r.get("playerName")}
    rows: list[dict] = []
    for r in _rows(se):
        ident = _identity(
            resolver=resolver, position="rb", season=season,
            player=r.get("player"), team=r.get("team"),
            age=None, games=r.get("gp"),
        )
        a = adv_idx_id.get(ident.get("playerId")) or adv_idx_norm.get(_norm_player(r["player"])) or {}
        rows.append({
            **ident,
            "rushAttempts":                 r.get("carries"),
            "rushYards":                    r.get("yds"),
            "yardsPerCarry":                r.get("ypc"),
            "yardsBeforeContactPerAttempt": r.get("ybc_per_carry") or a.get("yardsBeforeContactPerAttempt"),
            "yardsAfterContactPerAttempt":  r.get("yac_per_carry") or a.get("yardsAfterContactPerAttempt"),
            "brokenTackles":                r.get("broken_tackles") or a.get("brokenTackles"),
            "tacklesForLoss":               a.get("tacklesForLoss"),
            "tacklesForLossYards":          a.get("tacklesForLossYards"),
            "rushes10Plus":                 a.get("rushes10Plus"),
            "rushes20Plus":                 a.get("rushes20Plus"),
            "rushes30Plus":                 a.get("rushes30Plus"),
            "rushes40Plus":                 a.get("rushes40Plus"),
            "rushes50Plus":                 a.get("rushes50Plus"),
            "longestRush":                  r.get("long") or a.get("longestRush"),
            "longestRushTouchdown":         a.get("longestRushTouchdown"),
            "targets":                      r.get("tgt"),
            "receptions":                   r.get("rec"),
            "redZoneTargets":               r.get("rz_tgt"),
            "receivingYardsAfterCatch":     a.get("receivingYardsAfterCatch"),
        })
    rows.sort(key=lambda x: -(x.get("rushYards") or 0))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    sources = [lf["stat_explorer"]]
    if adv:
        sources.append(lf["advanced"])
    return rows, sources


# ── QB ─────────────────────────────────────────────────────────────────────

def _project_qb_overview(resolver, season, lf):
    payload = _load(lf.get("overview", ""))
    if not payload:
        return None, []
    eff     = _load(lf.get("efficiency", "")) or {}
    eff_idx = {r["player"]: r for r in _rows(eff)}
    out: list[dict] = []
    for r in _rows(payload):
        e = eff_idx.get(r["player"], {})
        overall = r.get("overall_score")
        out.append({
            **_identity(
                resolver=resolver, position="qb", season=season,
                player=r.get("player"), team=r.get("team"),
                age=r.get("age"), games=r.get("games"),
            ),
            "snapPct":              r.get("snap_pct"),
            "dropbacksPerGame":     r.get("dropbacks_per_gm"),
            "rushAttemptsPerGame":  r.get("rush_att_per_gm"),
            "designedRushPct":      e.get("designed_rush_rate"),
            "redZoneAttempts":      r.get("rz_att"),
            "deepAttemptPct":       r.get("deep_attempt_rate"),
            "stabilityScore":       r.get("stability"),
            "careerArc":            r.get("career_arc"),
            "experienceTier":       r.get("exp_tier"),
            "efficiencyScore":      r.get("efficiency_score"),
            "overallScore":         overall,
            "tier":                 r.get("qb_tier") or derive_tier("qb", overall),
        })
    sources = [lf["overview"]]
    if eff:
        sources.append(lf["efficiency"])
    return out, sources


def _project_qb_efficiency(resolver, season, lf):
    payload = _load(lf.get("efficiency", ""))
    if not payload:
        return None, []
    overview = _load(lf.get("overview", "")) or {}
    ov_idx   = {r["player"]: r for r in _rows(overview)}
    out: list[dict] = []
    for r in _rows(payload):
        ov = ov_idx.get(r["player"], {})
        out.append({
            **_identity(
                resolver=resolver, position="qb", season=season,
                player=r.get("player"), team=r.get("team"),
                age=ov.get("age"), games=r.get("games"),
            ),
            "attempts":                  r.get("attempts"),
            "epaPerPlay":                r.get("epa_per_play"),
            "successRate":               r.get("success_rate"),
            "yardsPerAttempt":           r.get("ypa"),
            "airYardsPerAttempt":        r.get("air_yards_per_att"),
            "fantasyPointsOverExpected": r.get("fpoe"),
            "touchdownsOverExpected":    r.get("td_over_expected"),
            "explosivePassPct":          r.get("explosive_play_rate"),
            "relativeEfficiency":        r.get("relative_efficiency"),
            "efficiencyScore":           ov.get("efficiency_score", r.get("efficiency_score")),
        })
    sources = [lf["efficiency"]]
    if overview:
        sources.append(lf["overview"])
    return out, sources


def _project_qb_usage(resolver, season, lf):
    payload = _load(lf.get("trends", ""))
    if not payload:
        return None, []
    overview = _load(lf.get("overview", "")) or {}
    ov_idx   = {r["player"]: r for r in _rows(overview)}
    raw_rows: list[dict] = []
    for r in _rows(payload):
        games        = r.get("games") or 0
        rush_td      = r.get("rushTd") or 0
        rush_td_pg   = _round(rush_td / games, 2) if games else None
        delta_drops  = r.get("deltaDropbacksPerGame")
        raw_rows.append({
            "ident": _identity(
                resolver=resolver, position="qb", season=season,
                player=r.get("player"), team=r.get("team"),
                age=r.get("age"), games=games or None,
            ),
            "dropbacksPerGame":         r.get("dropbacksPerGame"),
            "deltaDropbacksPerGame":    delta_drops,
            "rushAttemptsPerGame":      r.get("rushAttPerGame"),
            "deltaRushAttemptsPerGame": r.get("deltaRushAttPerGame"),
            "rushTouchdownsPerGame":    rush_td_pg,
            "roleTrend":                _trend_label(delta_drops),
            "_db":                      r.get("dropbacksPerGame"),
        })
    db_values = [r["_db"] for r in raw_rows]
    out: list[dict] = []
    for r in raw_rows:
        usage = _percentile_score(db_values, r.pop("_db"))
        ident = r.pop("ident")
        out.append({
            **ident,
            "dropbacksPerGame":         r["dropbacksPerGame"],
            "deltaDropbacksPerGame":    r["deltaDropbacksPerGame"],
            "rushAttemptsPerGame":      r["rushAttemptsPerGame"],
            "deltaRushAttemptsPerGame": r["deltaRushAttemptsPerGame"],
            "rushTouchdownsPerGame":    r["rushTouchdownsPerGame"],
            "roleTrend":                r["roleTrend"],
            "usageScore":               usage,
        })
    sources = [lf["trends"]]
    if overview:
        sources.append(lf["overview"])
    return out, sources


def _project_qb_stat_explorer(resolver, season, lf):
    payload = _load(lf.get("stat_explorer", ""))
    if not payload:
        return None, []
    out: list[dict] = []
    for r in _rows(payload):
        out.append({
            **_identity(
                resolver=resolver, position="qb", season=season,
                player=r.get("player"), team=r.get("team"),
                age=None, games=r.get("gp"),
            ),
            "completions":         r.get("comp"),
            "attempts":            r.get("att"),
            "completionPct":       r.get("pct"),
            "passingYards":        r.get("yds"),
            "yardsPerAttempt":     r.get("ypa"),
            "passingTouchdowns":   r.get("td"),
            "interceptions":       r.get("int"),
            "airYards":            r.get("air_yards"),
            "airYardsPerAttempt":  r.get("air_yards_per_att"),
            "passes10Plus":        r.get("pass_10_plus"),
            "passes20Plus":        r.get("pass_20_plus"),
            "passes30Plus":        r.get("pass_30_plus"),
            "passes40Plus":        r.get("pass_40_plus"),
            "passes50Plus":        r.get("pass_50_plus"),
            "rushAttempts":        r.get("rush_att"),
            "rushYards":           r.get("rush_yds"),
            "rushTouchdowns":      r.get("rush_td"),
            "sacks":               r.get("sacks"),
            "redZoneAttempts":     r.get("rz_att"),
            "passerRating":        r.get("passer_rating"),
        })
    out.sort(key=lambda x: -(x.get("passingYards") or 0))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out, [lf["stat_explorer"]]


# ── WR / TE ────────────────────────────────────────────────────────────────

def _project_wrte_overview(pos, resolver, season, lf):
    payload = _load(lf.get("overview", ""))
    if not payload:
        return None, []
    out: list[dict] = []
    for r in _rows(payload):
        overall = r.get("overall_score")
        out.append({
            **_identity(
                resolver=resolver, position=pos, season=season,
                player=r.get("player"), team=r.get("team"),
                age=r.get("age"), games=r.get("games"),
            ),
            "snapPct":                     r.get("snap_pct"),
            "routesPerGame":               r.get("routes_per_gm"),
            "targetsPerGame":              r.get("targets_per_gm"),
            "receptionsPerGame":           r.get("receptions_per_gm"),
            "airYardsPerGame":             r.get("air_yards_per_gm"),
            "targetSharePct":              r.get("target_share_pct"),
            "airYardsSharePct":            r.get("air_yards_share_pct"),
            "wopr":                        r.get("wopr"),
            "redZoneTargets":              r.get("rz_tgt"),
            "catchPct":                    r.get("catch_rate"),
            "yardsPerRouteRun":            r.get("yards_per_route_run"),
            "yardsPerTarget":              r.get("yards_per_target"),
            "yardsAfterCatchPerReception": r.get("yac_per_rec"),
            "fantasyPointsOverExpected":   r.get("fpoe"),
            "stabilityScore":              r.get("stability"),
            "careerArc":                   r.get("career_arc"),
            "experienceTier":              r.get("exp_tier"),
            "roleScore":                   r.get("role_score"),
            "efficiencyScore":             r.get("efficiency_score"),
            "overallScore":                overall,
            "tier":                        derive_tier(pos, overall),
        })
    return out, [lf["overview"]]


def _project_wrte_efficiency(pos, resolver, season, lf):
    payload = _load(lf.get("efficiency", ""))
    if not payload:
        return None, []
    overview = _load(lf.get("overview", "")) or {}
    ov_idx   = {r["player"]: r for r in _rows(overview)}
    out: list[dict] = []
    for r in _rows(payload):
        ov = ov_idx.get(r["player"], {})
        out.append({
            **_identity(
                resolver=resolver, position=pos, season=season,
                player=r.get("player"), team=r.get("team"),
                age=ov.get("age"), games=r.get("games"),
            ),
            "targets":                       r.get("targets"),
            "epaPerTarget":                  r.get("epa_per_target"),
            "successRate":                   r.get("success_rate"),
            "yardsPerTarget":                r.get("yards_per_target"),
            "catchPct":                      r.get("catch_rate"),
            "fantasyPointsOverExpected":     r.get("fpoe"),
            "yardsPerRouteRun":              r.get("yards_per_route_run"),
            "targetsPerRouteRun":            r.get("targets_per_route_run"),
            "airYardsPerTarget":             r.get("air_yards_per_target"),
            "airYardsSharePct":              r.get("air_yards_share_pct"),
            "explosivePlayPct":              r.get("explosive_play_rate"),
            "explosiveReceptions20Plus":     r.get("explosive_rec_20_plus"),
            "explosiveReceptions40Plus":     r.get("explosive_rec_40_plus"),
            "longestReception":              r.get("longest_reception"),
            "yardsAfterCatchPerReception":   r.get("yac_per_rec"),
            "yardsBeforeCatchPerReception":  r.get("ybc_per_rec"),
            "brokenTackles":                 r.get("broken_tackles"),
            "brokenTacklesPerReception":     r.get("btkl_per_rec"),
            "efficiencyScore":               ov.get("efficiency_score", r.get("efficiency_score")),
        })
    sources = [lf["efficiency"]]
    if overview:
        sources.append(lf["overview"])
    return out, sources


def _project_wrte_usage(pos, resolver, season, lf):
    payload = _load(lf.get("trends", ""))
    if not payload:
        return None, []
    overview = _load(lf.get("overview", "")) or {}
    ov_idx   = {r["player"]: r for r in _rows(overview)}
    out: list[dict] = []
    for r in _rows(payload):
        ov = ov_idx.get(r["player"], {})
        out.append({
            **_identity(
                resolver=resolver, position=pos, season=season,
                player=r.get("player"), team=r.get("team"),
                age=ov.get("age"), games=r.get("games"),
            ),
            "snapPct":                   r.get("snap_pct"),
            "deltaSnapPct":              r.get("delta_snap_pct"),
            "routesPerGame":             r.get("routes_per_gm"),
            "deltaRoutesPerGame":        r.get("delta_routes_per_gm"),
            "targetsPerGame":            r.get("targets_per_gm"),
            "deltaTargetsPerGame":       r.get("delta_targets_per_gm"),
            "airYardsPerGame":           r.get("air_yards_per_gm"),
            "deltaAirYardsPerGame":      r.get("delta_air_yards_per_gm"),
            "averageDepthOfTarget":      r.get("adot"),
            "deltaAverageDepthOfTarget": r.get("delta_adot"),
            "redZoneTargets":            r.get("rz_tgt"),
            "endZoneTargets":            r.get("end_zone_tgt"),
            "roleTrend":                 r.get("role_trend"),
            "usageScore":                r.get("usage_score"),
        })
    sources = [lf["trends"]]
    if overview:
        sources.append(lf["overview"])
    return out, sources


def _project_wrte_stat_explorer(pos, resolver, season, lf):
    se = _load(lf.get("stat_explorer", ""))
    if not se:
        return None, []
    adv = _load(lf.get("advanced", "")) or {}
    # WR advanced now uses camelCase keys + playerId/playerName identity; TE
    # advanced still uses the legacy snake-case + "player" identity.  Index both
    # ways and read each field with new-key-first / legacy-key fallback so this
    # shared projection keeps working for both positions.
    adv_idx_id   = {r.get("playerId"): r for r in _rows(adv) if r.get("playerId")}
    adv_idx_name = {(r.get("playerName") or r.get("player")): r
                    for r in _rows(adv) if (r.get("playerName") or r.get("player"))}
    adv_idx_norm = {_norm_player(r.get("playerName") or r.get("player")): r
                    for r in _rows(adv) if (r.get("playerName") or r.get("player"))}
    out: list[dict] = []
    for r in _rows(se):
        ident = _identity(
            resolver=resolver, position=pos, season=season,
            player=r.get("player"), team=r.get("team"),
            age=None, games=r.get("gp"),
        )
        a = (adv_idx_id.get(ident.get("playerId"))
             or adv_idx_name.get(r["player"])
             or adv_idx_norm.get(_norm_player(r["player"]))
             or {})
        receptions = r.get("rec") or a.get("receptions") or a.get("rec")
        ybc_per_rec = a.get("yardsBeforeCatchPerReception") or a.get("ybc_per_rec")
        ybc_total = a.get("ybc")
        if ybc_total is None and ybc_per_rec is not None and receptions:
            ybc_total = round(ybc_per_rec * receptions)
        out.append({
            **ident,
            "receptions":                   receptions,
            "receivingYards":               r.get("rec_yds") or a.get("receivingYards") or a.get("yds"),
            "yardsPerReception":            r.get("ypr") or a.get("yardsPerReception") or a.get("ypr"),
            "yardsBeforeCatch":             ybc_total,
            "yardsBeforeCatchPerReception": ybc_per_rec,
            "yardsAfterCatch":              r.get("yac") or a.get("yac"),
            "yardsAfterCatchPerReception":  r.get("yac_per_rec") or a.get("yardsAfterCatchPerReception") or a.get("yac_per_rec"),
            "brokenTackles":                r.get("broken_tackles") or a.get("brokenTackles") or a.get("brktkl"),
            "targets":                      r.get("tgt") or a.get("targets") or a.get("tgt"),
            "targetSharePct":               a.get("targetSharePct") or a.get("target_share"),
            "redZoneTargets":               r.get("rz_tgt") or a.get("redZoneTargets") or a.get("rz_tgt"),
            "receptions10Plus":             a.get("receptions10Plus") or a.get("rec_10_plus"),
            "receptions20Plus":             a.get("receptions20Plus") or a.get("rec_20_plus"),
            "receptions30Plus":             a.get("receptions30Plus") or a.get("rec_30_plus"),
            "receptions40Plus":             a.get("receptions40Plus") or a.get("rec_40_plus"),
            "receptions50Plus":             a.get("receptions50Plus") or a.get("rec_50_plus"),
            "longestReception":             r.get("long") or a.get("longestReception") or a.get("lng"),
        })
    out.sort(key=lambda x: -(x.get("receivingYards") or 0))
    for i, row in enumerate(out, 1):
        row["rank"] = i
    sources = [lf["stat_explorer"]]
    if adv:
        sources.append(lf["advanced"])
    return out, sources


ProjectFn = Callable[..., tuple[list[dict] | None, list[str]]]

PROJECTIONS: dict[tuple[str, str], ProjectFn] = {
    ("rb", "overview"):      _project_rb_overview,
    ("rb", "efficiency"):    _project_rb_efficiency,
    ("rb", "usage"):         _project_rb_usage,
    ("rb", "stat_explorer"): _project_rb_stat_explorer,
    ("qb", "overview"):      _project_qb_overview,
    ("qb", "efficiency"):    _project_qb_efficiency,
    ("qb", "usage"):         _project_qb_usage,
    ("qb", "stat_explorer"): _project_qb_stat_explorer,
    ("wr", "overview"):      lambda r, s, lf: _project_wrte_overview("wr",     r, s, lf),
    ("wr", "efficiency"):    lambda r, s, lf: _project_wrte_efficiency("wr",   r, s, lf),
    ("wr", "usage"):         lambda r, s, lf: _project_wrte_usage("wr",        r, s, lf),
    ("wr", "stat_explorer"): lambda r, s, lf: _project_wrte_stat_explorer("wr", r, s, lf),
    ("te", "overview"):      lambda r, s, lf: _project_wrte_overview("te",     r, s, lf),
    ("te", "efficiency"):    lambda r, s, lf: _project_wrte_efficiency("te",   r, s, lf),
    ("te", "usage"):         lambda r, s, lf: _project_wrte_usage("te",        r, s, lf),
    ("te", "stat_explorer"): lambda r, s, lf: _project_wrte_stat_explorer("te", r, s, lf),
}


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _scrub_stat_explorer(rows: list[dict]) -> list[dict]:
    forbidden = set(STAT_EXPLORER_FORBIDDEN)
    return [{k: v for k, v in r.items() if k not in forbidden} for r in rows]


def write_canonical_file(
    pos: str, tab: str, season: str, resolver: PlayerIdResolver,
) -> dict[str, Any]:
    projection = PROJECTIONS.get((pos, tab))
    if projection is None:
        return {"tab": tab, "season": season, "status": "no-projection", "rowCount": 0}
    lf = _legacy_for(pos, tab, season)
    rows, sources = projection(resolver, season, lf)
    if rows is None:
        return {"tab": tab, "season": season, "status": "source-missing", "rowCount": 0}
    if not rows:
        return {"tab": tab, "season": season, "status": "no-rows", "rowCount": 0}
    if tab == "stat_explorer":
        rows = _scrub_stat_explorer(rows)

    template = FIELD_SCHEMA[pos][tab]
    final_rows = [_project(template, r) for r in rows]
    envelope = make_envelope(
        position=pos, tab=tab, season=season,
        rows=final_rows, columns=list(template),
        source_files=sources,
    )
    path = canonical_path(pos, tab, season)
    write_json(path, envelope)
    return {
        "tab":      tab,
        "season":   season,
        "status":   "ok",
        "sources":  sources,
        "path":     str(path.relative_to(OUTPUT_DIR)),
        "rowCount": len(final_rows),
    }


def synthesize_all(pos: str, tab: str) -> dict[str, Any] | None:
    """Build {pos}/{tab}_all.json from whatever per-season canonical files
    exist (latest-season-per-player wins)."""
    SEASONS_NEWEST_FIRST = ["2025", "2024", "2023"]
    by_player: dict[str, dict] = {}
    chosen: list[str] = []
    for s in SEASONS_NEWEST_FIRST:
        fp = canonical_path(pos, tab, s)
        if not fp.exists():
            continue
        with open(fp) as f:
            payload = json.load(f)
        rows = payload.get("rows") or []
        chosen.append(fp.name)
        for r in rows:
            pid = r.get("playerId") or r.get("playerName")
            if not pid or pid in by_player:
                continue
            by_player[pid] = dict(r)
    if not by_player:
        return None
    rows = list(by_player.values())
    if tab == "stat_explorer":
        rows.sort(key=lambda r: -(
            r.get("receivingYards") or r.get("rushYards") or r.get("passingYards") or 0
        ))
        for i, r in enumerate(rows, 1):
            r["rank"] = i
    else:
        rows.sort(
            key=lambda r: (
                -(r.get("overallScore") or 0),
                -(r.get("efficiencyScore") or 0),
                r.get("playerName") or "",
            )
        )
    template = FIELD_SCHEMA[pos][tab]
    final_rows = [_project(template, r) for r in rows]
    envelope = make_envelope(
        position=pos, tab=tab, season="all",
        rows=final_rows, columns=list(template),
        source_files=chosen,
        extra_meta={"synthesized": True, "strategy": "latest-season-per-player"},
    )
    path = canonical_path(pos, tab, "all")
    write_json(path, envelope)
    return {
        "tab":      tab,
        "season":   "all",
        "status":   "ok-synthesized",
        "sources":  chosen,
        "path":     str(path.relative_to(OUTPUT_DIR)),
        "rowCount": len(final_rows),
    }


# ---------------------------------------------------------------------------
# Cleanup + entry point
# ---------------------------------------------------------------------------

def _clear_old_canonical_tree() -> None:
    """Drop canonical files whose name doesn't match the current spec.

    Specifically removes stale files left behind by an earlier schema
    (e.g. advanced_*.json from the 5-tab version) so the directory
    contains only files that match the current canonical structure.
    """
    if not CANONICAL_DIR.exists():
        return
    valid_names = {
        f"{tab}_{season}.json" for tab in TABS for season in SEASONS
    } | {"manifest.json"}
    for pos in POSITIONS:
        pdir = CANONICAL_DIR / pos
        if not pdir.exists():
            continue
        for fp in pdir.glob("*.json"):
            if fp.name not in valid_names:
                fp.unlink()


def main() -> None:
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    _clear_old_canonical_tree()
    resolver = PlayerIdResolver()

    counts = {"ok": 0, "synthesized": 0, "unavailable": 0}
    unavailable_per_pos: dict[str, list[tuple[str, str]]] = {p: [] for p in POSITIONS}

    for pos in POSITIONS:
        print(f"\n── {pos.upper()} ──")
        for tab in TABS:
            # Per-season writes
            for s in HISTORICAL_SEASONS + (CURRENT_SEASON,):
                res = write_canonical_file(pos, tab, s, resolver)
                if res["status"] == "ok":
                    counts["ok"] += 1
                    src = ", ".join(res.get("sources", []))
                    print(f"  {tab:14s} {s:5s} ← {src:55s} ({res['rowCount']} rows)")
                else:
                    counts["unavailable"] += 1
                    unavailable_per_pos[pos].append((tab, s))
                    print(f"  {tab:14s} {s:5s} omit ({res['status']})")

            # _all synthesis
            res_all = synthesize_all(pos, tab)
            if res_all is None:
                counts["unavailable"] += 1
                unavailable_per_pos[pos].append((tab, "all"))
                print(f"  {tab:14s} all   omit (no per-season files)")
            else:
                counts["synthesized"] += 1
                src = ", ".join(res_all.get("sources", []))
                print(f"  {tab:14s} all   ← {src:55s} ({res_all['rowCount']} rows, synthesized)")

        write_position_manifest(pos, unavailable=unavailable_per_pos[pos])

    write_global_manifest()

    print(
        f"\nCanonical build complete: "
        f"{counts['ok']} ok, {counts['synthesized']} synthesized, "
        f"{counts['unavailable']} unavailable."
    )


if __name__ == "__main__":
    main()
