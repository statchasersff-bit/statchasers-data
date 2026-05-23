"""
build_rb_efficiency_analytics.py
─────────────────────────────────
Builds the RB Efficiency Analytics dataset for the StatChasers Research tab.

Answers: How effective is this RB on a per-play and per-touch basis?

Minimal overlap with:
  RB Player Overview   → role, opportunity, summary profile
  RB Advanced Stats    → raw totals and volume stats

Data sources:
  data/raw/nflverse_play_by_play.parquet  — EPA, success, yard counts
  data/raw/pfr_rush_advstats.parquet      — broken tackles, YBC, YAC
  data/raw/sleeper_players.json           — name disambiguation
  data/processed/player_metrics.json      — FPOE

Data rules:
  - RBs only, one row per player, 2025 season
  - Minimum 15 rush_att to appear in output
  - Percentile pool for efficiency_score uses players with ≥ 25 rush_att
  - null for missing values, never empty strings
  - Rate/percentage stats: 2 decimals
  - Counting stats: integers
  - efficiency_score: 1 decimal (0-100)

efficiency_score weights (mirrors RB player overview — source of truth):
  Yds / Touch     = 30%
  Explosive Run % = 20%
  Breakaway Run % = 20%
  FPOE            = 20%
  Stability       = 10%

Note: efficiency_score values are patched from rb_player_overview.json
at build time to guarantee exact cross-tab alignment.

Output:
  output/rb_efficiency_analytics.json
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
SCRIPTS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from _id_resolution import build_canonical_id_lookup, filter_canonical_id  # noqa: E402

PBP_PATH     = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
NFL_PATH     = ROOT / "data" / "raw"       / "nflverse_players.parquet"
PFR_PATH     = ROOT / "data" / "raw"       / "pfr_rush_advstats.parquet"
SLEEPER_PATH = ROOT / "data" / "raw"       / "sleeper_players.json"
METRICS_PATH = ROOT / "data" / "processed" / "player_metrics.json"
OUTPUT_PATH  = ROOT / "output"             / "rb_efficiency_analytics.json"

SEASON               = 2025
MIN_CARRIES          = 15   # minimum to appear in output

# Hardcoded team overrides for ambiguous same-position abbreviations where
# Sleeper's team field is stale relative to the 2025 PBP season.
_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    "T.Etienne": {"JAX": "Travis Etienne", "CAR": "Trevor Etienne"},
    "B.Robinson": {"ATL": "Bijan Robinson", "SF": "Brian Robinson", "WAS": "Brian Robinson"},
    "J.Williams": {"DEN": "Javonte Williams", "NO": "Jamaal Williams"},
}
MIN_CARRIES_PCT_POOL = 25   # minimum to enter the percentile pool for runner_score

COLUMNS: list[dict] = [
    {"key": "player",                 "label": "Player",          "type": "string", "group": "Identity"},
    {"key": "team",                   "label": "Team",            "type": "string", "group": "Identity"},
    {"key": "games",                  "label": "GP",              "type": "number", "group": "Identity"},
    {"key": "rush_attempts",          "label": "ATT",             "type": "number", "group": "Identity"},
    {"key": "epa_per_rush",           "label": "EPA / Rush",      "type": "number", "group": "Efficiency"},
    {"key": "success_rate",           "label": "Success %",       "type": "number", "group": "Efficiency"},
    {"key": "yards_per_attempt",      "label": "Yds / Att",       "type": "number", "group": "Efficiency"},
    {"key": "yards_per_touch",        "label": "Yds / Touch",     "type": "number", "group": "Efficiency"},
    {"key": "fpoe",                   "label": "FPOE",            "type": "number", "group": "Efficiency"},
    {"key": "explosive_run_pct",      "label": "Explosive %",     "type": "number", "group": "Explosiveness"},
    {"key": "breakaway_run_pct",      "label": "Breakaway %",     "type": "number", "group": "Explosiveness"},
    {"key": "breakaway_runs",         "label": "Breakaway Runs",  "type": "number", "group": "Explosiveness"},
    {"key": "longest_run",            "label": "Longest Run",     "type": "number", "group": "Explosiveness"},
    {"key": "broken_tackles",         "label": "Broken Tackles",  "type": "number", "group": "Elusiveness"},
    {"key": "broken_tackles_per_att", "label": "BTKL / Att",      "type": "number", "group": "Elusiveness"},
    {"key": "yac_per_att",            "label": "YAC / Att",       "type": "number", "group": "Elusiveness"},
    {"key": "ybc_per_att",            "label": "YBC / Att",       "type": "number", "group": "Elusiveness"},
    {"key": "efficiency_score",        "label": "Efficiency Score","type": "number", "group": "Composite"},
]


# ---------------------------------------------------------------------------
# Name resolution (identical pattern to build_rb_advanced_stats.py)
# ---------------------------------------------------------------------------

def _abbrev(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _build_unambig(sleeper_players: list[dict]) -> dict[str, str]:
    counts: dict[str, int]  = {}
    mapping: dict[str, str] = {}
    for p in sleeper_players:
        full = (p.get("full_name") or "").strip()
        if not full:
            continue
        ab = _abbrev(full)
        counts[ab]  = counts.get(ab, 0) + 1
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


def _build_pfr_unambig(pfr_df: pd.DataFrame) -> dict[str, str]:
    counts: dict[str, int]  = {}
    mapping: dict[str, str] = {}
    for name in pfr_df["pfr_player_name"].dropna().unique():
        name = str(name)
        ab = _abbrev(name)
        counts[ab]  = counts.get(ab, 0) + 1
        mapping[ab] = name
    return {ab: fn for ab, fn in mapping.items() if counts[ab] == 1}


def _resolve(
    pbp_name: str,
    pbp_team: str,
    full_name_set: set[str],
    unambig: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    metrics_pos: dict[str, str],
    sleeper_pos: dict[str, str],
    pos_hint: str = "RB",
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
            if sleeper_pos.get(n, "") == pos_hint or metrics_pos.get(n, "") == pos_hint
        ]
        if not pos_matches:
            pos_matches = list(teams.items())
        if len(pos_matches) == 1:
            return pos_matches[0][1]
        conflicting    = [(t, n) for t, n in pos_matches if t is not None and t != pbp_team]
        non_conflicting = [(t, n) for t, n in pos_matches if t is None or t == pbp_team]
        if non_conflicting and conflicting:
            no_team = [(t, n) for t, n in non_conflicting if t is None]
            return no_team[0][1] if no_team else non_conflicting[0][1]
        with_team = [(t, n) for t, n in pos_matches if t is not None]
        return with_team[0][1] if with_team else pos_matches[0][1]
    return pbp_name


# ---------------------------------------------------------------------------
# Percentile helper — (below + 0.5×equal) / n × 100  (rank kind)
# null value → 50.0 (neutral)
# ---------------------------------------------------------------------------

def _pct(value: float | None, arr: list[float | None]) -> float:
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


# ---------------------------------------------------------------------------
# Stability / volatility from per-game touch sequence
# ---------------------------------------------------------------------------

def _stability_volatility(
    game_touch_map: dict[str, int],
    game_order: list[str],
) -> tuple[float | None, float | None]:
    """
    stability  = (1 – σ/μ of last-6 games) × 10, clamped 0–10
    volatility = σ/μ of all games (coefficient of variation)
    Mirrors build_rb_player_overview._stability_volatility exactly.
    """
    ordered = [game_touch_map[g] for g in game_order if g in game_touch_map]
    extra   = [v for g, v in game_touch_map.items() if g not in game_order]
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
# PFR aggregation — season totals per player name
# ---------------------------------------------------------------------------

def _aggregate_pfr(pfr: pd.DataFrame, season: int) -> dict[str, dict]:
    s = pfr[pfr["season"] == season]
    if s.empty:
        return {}
    return (
        s.groupby("pfr_player_name")
        .agg(
            ybc=("rushing_yards_before_contact", "sum"),
            yac=("rushing_yards_after_contact",  "sum"),
            carries=("carries",                  "sum"),
            broken_tackles=("rushing_broken_tackles", "sum"),
        )
        .to_dict("index")
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(
    pbp: pd.DataFrame,
    pfr: pd.DataFrame,
    sleeper_players: list[dict],
    player_metrics: list[dict],
) -> list[dict]:

    full_name_set  = {(p.get("full_name") or "").strip() for p in sleeper_players}
    unambig        = _build_unambig(sleeper_players)
    team_disambig  = _build_team_disambig(sleeper_players)
    pfr_unambig    = _build_pfr_unambig(pfr[pfr["season"] == SEASON])
    # PFR only supplements Sleeper for names NOT already flagged as ambiguous
    pfr_safe       = {ab: fn for ab, fn in pfr_unambig.items() if ab not in team_disambig}
    unambig_full   = {**pfr_safe, **unambig}  # Sleeper wins conflicts

    sleeper_pos    = {(p.get("full_name") or "").strip(): (p.get("position") or "") for p in sleeper_players}
    metrics_pos    = {p["player"]: p.get("pos", "") for p in player_metrics}
    metrics_map    = {p["player"]: p for p in player_metrics}

    pfr_agg = _aggregate_pfr(pfr, SEASON)

    # ── Isolate rush and pass plays ──────────────────────────────────────────
    rush   = pbp[pbp["rush_attempt"] == 1].copy()
    passes = pbp[pbp["pass_attempt"] == 1].copy()

    has_epa     = "epa"     in rush.columns
    has_success = "success" in rush.columns
    has_yardline = "yardline_100" in pbp.columns

    # ── Tag rush plays with resolved full names ──────────────────────────────
    name_cache: dict[tuple[str, str], str] = {}

    def _tag_rush(row: pd.Series) -> str:
        key = (str(row["rusher_player_name"]), str(row.get("posteam", "")))
        if key not in name_cache:
            name_cache[key] = _resolve(
                key[0], key[1],
                full_name_set, unambig_full, team_disambig,
                metrics_pos, sleeper_pos,
            )
        return name_cache[key]

    rush["_fn"] = rush.apply(_tag_rush, axis=1)

    # Drop name-collision plays (e.g. Brandon Allen QB on SF being merged into Braelon Allen RB).
    if NFL_PATH.exists():
        _nfl = pd.read_parquet(NFL_PATH, columns=["gsis_id", "display_name", "position"])
        _canonical_id_lookup = build_canonical_id_lookup(_nfl, position="RB")
        rush = filter_canonical_id(rush, "_fn", "rusher_player_id", _canonical_id_lookup)
    else:
        _canonical_id_lookup = {}

    # ── Identify RB full names ───────────────────────────────────────────────
    rb_names: set[str] = set()
    for fn in rush["_fn"].unique():
        fn = str(fn)
        if metrics_pos.get(fn, "") == "RB" or sleeper_pos.get(fn, "") == "RB":
            rb_names.add(fn)

    rb_abbrevs: set[str] = {ab for (ab, _), fn in name_cache.items() if fn in rb_names}

    # ── Receiving stats (for yards_per_touch) ───────────────────────────────
    # passes is already filtered to pass_attempt == 1
    rb_pass = passes[
        passes["receiver_player_name"].notna()
        & passes["receiver_player_name"].isin(rb_abbrevs)
    ].copy()

    rec_cache: dict[tuple[str, str], str] = {}

    def _tag_rec(row: pd.Series) -> str:
        key = (str(row.get("receiver_player_name", "")), str(row.get("posteam", "")))
        if key not in rec_cache:
            rec_cache[key] = _resolve(
                key[0], key[1],
                full_name_set, unambig_full, team_disambig,
                metrics_pos, sleeper_pos,
            )
        return rec_cache[key]

    rec_stats: dict[str, dict] = {}
    if not rb_pass.empty:
        rb_pass["_fn"] = rb_pass.apply(_tag_rec, axis=1)
        if _canonical_id_lookup:
            rb_pass = filter_canonical_id(rb_pass, "_fn", "receiver_player_id", _canonical_id_lookup)
        rb_pass = rb_pass[rb_pass["_fn"].isin(rb_names)]
        for fn, grp in rb_pass.groupby("_fn"):
            comps = grp[grp["complete_pass"] == 1]
            rec_stats[str(fn)] = {
                "receptions":    int(len(comps)),
                "receiving_yds": int(comps["yards_gained"].sum()),
            }

    # ── Aggregate per-player rush metrics ────────────────────────────────────
    rb_rush = rush[rush["_fn"].isin(rb_names)].copy()
    raw_rows: list[dict] = []

    for fn, grp in rb_rush.groupby("_fn"):
        fn = str(fn)
        rush_att = len(grp)
        if rush_att < MIN_CARRIES:
            continue

        rush_yds = float(grp["yards_gained"].sum())
        games    = int(grp["game_id"].nunique())
        yds      = grp["yards_gained"]

        # Per-att rates
        ypa = round(rush_yds / rush_att, 2)

        # EPA
        epa_per_rush = (
            round(float(grp["epa"].mean()), 2) if has_epa else None
        )

        # Success rate (stored as 0-100 percentage)
        success_rate = (
            round(float(grp["success"].mean()) * 100, 2) if has_success else None
        )

        # Explosive / breakaway
        exp10         = int((yds >= 10).sum())
        exp15         = int((yds >= 15).sum())
        exp10_pct     = round(exp10 / rush_att * 100, 2)
        exp15_pct     = round(exp15 / rush_att * 100, 2)
        longest_run   = int(yds.max())

        # Yards per touch
        rc = rec_stats.get(fn, {})
        receptions   = rc.get("receptions", 0)
        receiving_yds = rc.get("receiving_yds", 0)
        touches      = rush_att + receptions
        ypt          = round((rush_yds + receiving_yds) / touches, 2) if touches > 0 else None

        # Team: always use PBP posteam (ground truth).
        # player_metrics.team is stale and must not override.
        team = str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns else ""
        m = metrics_map.get(fn, {})

        # FPOE from player_metrics
        fpoe = m.get("fpoe")

        # PFR contact stats
        pf = pfr_agg.get(fn, {})
        pfr_carries    = pf.get("carries") or 0
        bt_raw         = pf.get("broken_tackles")
        yac_raw        = pf.get("yac")
        ybc_raw        = pf.get("ybc")

        broken_tackles = int(bt_raw) if bt_raw is not None and not (isinstance(bt_raw, float) and np.isnan(bt_raw)) else None
        bt_per_att     = round(broken_tackles / rush_att, 2) if broken_tackles is not None else None
        yac_per_att    = round(float(yac_raw) / pfr_carries, 2) if yac_raw is not None and pfr_carries > 0 else None
        ybc_per_att    = round(float(ybc_raw) / pfr_carries, 2) if ybc_raw is not None and pfr_carries > 0 else None

        # Per-game rush map for stability (mirrors RB Player Overview logic)
        game_rush_map = {
            str(gid): int(cnt)
            for gid, cnt in grp.groupby("game_id").size().items()
        }
        game_order_rb = sorted(game_rush_map.keys())
        stability_val, _ = _stability_volatility(game_rush_map, game_order_rb)

        raw_rows.append({
            "player":                 fn,
            "team":                   team,
            "games":                  games,
            "rush_attempts":          rush_att,
            "epa_per_rush":           epa_per_rush,
            "success_rate":           success_rate,
            "yards_per_attempt":      ypa,
            "yards_per_touch":        ypt,
            "fpoe":                   fpoe,
            "explosive_run_pct":      exp10_pct,
            "breakaway_run_pct":      exp15_pct,
            "breakaway_runs":         exp15,
            "longest_run":            longest_run,
            "broken_tackles":         broken_tackles,
            "broken_tackles_per_att": bt_per_att,
            "yac_per_att":            yac_per_att,
            "ybc_per_att":            ybc_per_att,
            "stability":              stability_val,
        })

    # ── Efficiency Score ──────────────────────────────────────────────────────
    # Mirrors build_rb_player_overview.py efficiency_score formula exactly:
    #   yards/touch 30%, explosive% 20%, breakaway% 20%, fpoe 20%, stability 10%
    # Percentile pool: only players with ≥ MIN_CARRIES_PCT_POOL rush attempts
    pool = [r for r in raw_rows if r["rush_attempts"] >= MIN_CARRIES_PCT_POOL]

    def _pool_col(key: str) -> list[float | None]:
        return [r.get(key) for r in pool]

    arr_ypts  = _pool_col("yards_per_touch")
    arr_exp10 = _pool_col("explosive_run_pct")
    arr_break = _pool_col("breakaway_run_pct")
    arr_fpoe  = _pool_col("fpoe")
    arr_stab  = _pool_col("stability")

    final_rows: list[dict] = []
    for r in raw_rows:
        efficiency_score = round(
            _pct(r.get("yards_per_touch"),    arr_ypts)  * 0.30
            + _pct(r.get("explosive_run_pct"), arr_exp10) * 0.20
            + _pct(r.get("breakaway_run_pct"), arr_break) * 0.20
            + _pct(r.get("fpoe"),              arr_fpoe)  * 0.20
            + _pct(r.get("stability"),         arr_stab)  * 0.10,
            1,
        )
        final_rows.append({**r, "efficiency_score": efficiency_score})

    # ── Sort and emit ─────────────────────────────────────────────────────────
    final_rows.sort(key=lambda r: r.get("efficiency_score") or 0.0, reverse=True)
    ordered_keys = [c["key"] for c in COLUMNS]
    return [{k: r.get(k) for k in ordered_keys} for r in final_rows]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (PBP_PATH, PFR_PATH, SLEEPER_PATH, METRICS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    print("Loading play-by-play data...")
    pbp_full = pd.read_parquet(PBP_PATH)
    pbp = pbp_full[(pbp_full["season"] == SEASON) & (pbp_full["season_type"] == "REG")].copy()  # regular season only
    if "two_point_attempt" in pbp.columns:
        pbp = pbp[pbp["two_point_attempt"].fillna(0) != 1].copy()
    print(f"  {len(pbp):,} plays for {SEASON}.")

    print("Loading PFR rush advanced stats...")
    pfr = pd.read_parquet(PFR_PATH)

    with open(SLEEPER_PATH) as f:
        sleeper_players: list[dict] = json.load(f)
    print(f"  {len(sleeper_players):,} Sleeper players.")

    with open(METRICS_PATH) as f:
        player_metrics: list[dict] = json.load(f)
    print(f"  {len(player_metrics):,} player metric records.")

    print("Building RB Efficiency Analytics...")
    rows = build(pbp, pfr, sleeper_players, player_metrics)
    print(f"  {len(rows)} RBs qualified.")

    # ── Patch efficiency_score from player overview (single source of truth) ──
    overview_path = OUTPUT_PATH.parent / "rb_player_overview.json"
    if overview_path.exists():
        with open(overview_path) as _f:
            _ov = json.load(_f)
        _ov_scores: dict[str, float | None] = {
            p["player"]: p.get("efficiency_score") for p in _ov.get("rows", [])
        }
        for r in rows:
            if r["player"] in _ov_scores:
                r["efficiency_score"] = _ov_scores[r["player"]]
        rows.sort(key=lambda r: r.get("efficiency_score") or 0.0, reverse=True)
        print(f"  Patched efficiency_score from player overview for {len(_ov_scores)} RBs")
    else:
        print(f"  WARN: {overview_path} not found — using analytics-computed efficiency_score")

    latest_week = int(pbp["week"].max()) if "week" in pbp.columns else None

    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season":     SEASON,
        "week":       latest_week,
        "table":      "rb_efficiency_analytics",
        "columns":    COLUMNS,
        "rows":       rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.1f} KB, {len(rows)} rows)")
    print("RB Efficiency Analytics build complete.")


if __name__ == "__main__":
    main()
