"""
build_rb_advanced_stats.py
──────────────────────────
Builds the RB Advanced Stats datasets for the StatChasers frontend.

Produces four output files:
  output/rb_advanced_stats_2023.json   — 2023 season only
  output/rb_advanced_stats_2024.json   — 2024 season only
  output/rb_advanced_stats_2025.json   — 2025 season only
  output/rb_advanced_stats_all.json    — 2023 + 2024 + 2025 combined
  output/rb_advanced_stats.json        — alias for 2025 (backward compat)

Data sources (pre-pulled by pull_nflverse_data.py / pull_sleeper_players.py):
  data/raw/nflverse_play_by_play.parquet  — rush/receiving play-level data
  data/raw/pfr_rush_advstats.parquet      — YBC, YAC, broken tackles (PFR)
  data/raw/sleeper_players.json           — canonical name lookup
  data/processed/player_metrics.json      — position / team context
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
PBP_PATH     = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
PFR_PATH     = ROOT / "data" / "raw"       / "pfr_rush_advstats.parquet"
SLEEPER_PATH = ROOT / "data" / "raw"       / "sleeper_players.json"
METRICS_PATH = ROOT / "data" / "processed" / "player_metrics.json"
OUTPUT_DIR   = ROOT / "output"

SEASONS     = [2023, 2024, 2025]
MIN_CARRIES = 15

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_BASE_COLUMNS: list[dict] = [
    {"key": "rank",                         "label": "Rank",     "type": "number",  "defaultVisible": True},
    {"key": "player",                        "label": "Player",   "type": "string",  "defaultVisible": True},
    {"key": "team",                          "label": "Team",     "type": "string",  "defaultVisible": True},
    {"key": "games",                         "label": "G",        "type": "number",  "defaultVisible": True},
    {"key": "rush_att",                      "label": "ATT",      "type": "number",  "defaultVisible": True},
    {"key": "rush_yds",                      "label": "YDS",      "type": "number",  "defaultVisible": True},
    {"key": "yards_per_att",                 "label": "Y/ATT",    "type": "decimal", "defaultVisible": True},
    {"key": "yards_before_contact_per_att",  "label": "YBC/ATT",  "type": "decimal", "defaultVisible": True},
    {"key": "yards_after_contact_per_att",   "label": "YAC/ATT",  "type": "decimal", "defaultVisible": True},
    {"key": "broken_tackles",                "label": "BRKTKL",   "type": "number",  "defaultVisible": True},
    {"key": "tfl",                           "label": "TFL",      "type": "number",  "defaultVisible": True},
    {"key": "tfl_yds_lost",                  "label": "TFL YDS",  "type": "number",  "defaultVisible": True},
    {"key": "runs_10_plus",                  "label": "10+ YDS",  "type": "number",  "defaultVisible": True},
    {"key": "runs_20_plus",                  "label": "20+ YDS",  "type": "number",  "defaultVisible": True},
    {"key": "runs_30_plus",                  "label": "30+ YDS",  "type": "number",  "defaultVisible": True},
    {"key": "runs_40_plus",                  "label": "40+ YDS",  "type": "number",  "defaultVisible": False},
    {"key": "runs_50_plus",                  "label": "50+ YDS",  "type": "number",  "defaultVisible": False},
    {"key": "longest_run",                   "label": "LNG",      "type": "number",  "defaultVisible": True},
    {"key": "longest_run_td",                "label": "LNG TD",   "type": "flag",    "defaultVisible": True},
    {"key": "targets",                        "label": "TGT",      "type": "number",  "defaultVisible": True},
    {"key": "receptions",                    "label": "REC",      "type": "number",  "defaultVisible": True},
    {"key": "red_zone_targets",              "label": "RZ TGT",   "type": "number",  "defaultVisible": True},
    {"key": "rec_yac",                       "label": "REC YAC",  "type": "number",  "defaultVisible": True},
]

# Per-season JSON uses the base columns (no season field needed — it's in the envelope).
COLUMNS_SEASON = _BASE_COLUMNS

# Multi-season JSON adds a Season column after Team so the frontend can filter/group.
COLUMNS_MULTI = (
    _BASE_COLUMNS[:3]                          # rank, player, team
    + [{"key": "season", "label": "Season", "type": "number", "defaultVisible": True}]
    + _BASE_COLUMNS[3:]                        # games … rec_yac
)


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def _abbrev(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _build_sleeper_abbrev_lookup(sleeper_players: list[dict]) -> dict[str, str]:
    counts: dict[str, int]  = {}
    mapping: dict[str, str] = {}
    for p in sleeper_players:
        full = p.get("full_name", "").strip()
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
        full = p.get("full_name", "").strip()
        if not full:
            continue
        ab = _abbrev(full)
        counts[ab] = counts.get(ab, 0) + 1
        by_team.setdefault(ab, {})[p.get("team")] = full
    return {ab: teams for ab, teams in by_team.items() if counts[ab] > 1}


def _build_pfr_abbrev_lookup(pfr_df: pd.DataFrame) -> dict[str, str]:
    counts: dict[str, int]  = {}
    mapping: dict[str, str] = {}
    for name in pfr_df["pfr_player_name"].dropna().unique():
        name = str(name)
        ab = _abbrev(name)
        counts[ab]  = counts.get(ab, 0) + 1
        mapping[ab] = name
    return {ab: fn for ab, fn in mapping.items() if counts[ab] == 1}


_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    "T.Etienne": {"JAX": "Travis Etienne", "CAR": "Trevor Etienne"},
    "B.Robinson": {"ATL": "Bijan Robinson", "SF": "Brian Robinson", "WAS": "Brian Robinson"},
}


def _resolve_with_team(
    pbp_name: str,
    pbp_team: str,
    full_name_set: set[str],
    unambiguous_lookup: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    metrics_by_player: dict[str, dict],
    sleeper_pos: dict[str, str],
    pos_hint: str = "RB",
) -> str:
    if pbp_name in _MANUAL_TEAM_OVERRIDES and pbp_team:
        hit = _MANUAL_TEAM_OVERRIDES[pbp_name].get(pbp_team)
        if hit:
            return hit
    if pbp_name in full_name_set:
        return pbp_name
    if pbp_name in unambiguous_lookup:
        return unambiguous_lookup[pbp_name]
    if pbp_name in team_disambig:
        teams = team_disambig[pbp_name]

        if pbp_team and pbp_team in teams:
            return teams[pbp_team]

        pos_matches = [
            (t, n) for t, n in teams.items()
            if sleeper_pos.get(n, "") == pos_hint
        ]
        if not pos_matches:
            pos_matches = [
                (t, n) for t, n in teams.items()
                if metrics_by_player.get(n, {}).get("pos", "") == pos_hint
            ]
        if not pos_matches:
            pos_matches = list(teams.items())

        if len(pos_matches) == 1:
            return pos_matches[0][1]

        conflicting    = [(t, n) for t, n in pos_matches if t is not None and t != pbp_team]
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
# PFR aggregation
# ---------------------------------------------------------------------------

def _aggregate_pfr_rush(pfr: pd.DataFrame, season: int) -> dict[str, dict]:
    s = pfr[pfr["season"] == season].copy()
    if s.empty:
        return {}
    agg = (
        s.groupby("pfr_player_name")
        .agg(
            ybc=("rushing_yards_before_contact", "sum"),
            yac=("rushing_yards_after_contact",  "sum"),
            carries=("carries",                  "sum"),
            broken_tackles=("rushing_broken_tackles", "sum"),
        )
        .to_dict("index")
    )
    return agg


# ---------------------------------------------------------------------------
# Per-season builder — returns a flat list of row dicts (no season field)
# ---------------------------------------------------------------------------

def build_season(
    pbp_season: pd.DataFrame,   # already filtered to one season
    pfr: pd.DataFrame,
    season: int,
    sleeper_players: list[dict],
    player_metrics: list[dict],
    full_name_set: set[str],
    unambig_with_pfr: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    metrics_by_player: dict[str, dict],
    sleeper_pos: dict[str, str],
) -> list[dict]:
    """Build one season's worth of RB advanced stats rows (no rank assigned yet)."""

    pfr_agg = _aggregate_pfr_rush(pfr, season)

    rush = pbp_season[pbp_season["rush_attempt"] == 1].copy()
    if rush.empty:
        return []

    has_yardline = "yardline_100" in rush.columns

    # Tag rush plays with resolved full names
    name_team_cache: dict[tuple[str, str], str] = {}

    def _tag(row: pd.Series) -> str:
        key = (str(row["rusher_player_name"]), str(row.get("posteam", "")))
        if key not in name_team_cache:
            name_team_cache[key] = _resolve_with_team(
                key[0], key[1],
                full_name_set, unambig_with_pfr, team_disambig,
                metrics_by_player, sleeper_pos,
            )
        return name_team_cache[key]

    rush["_full_name"] = rush.apply(_tag, axis=1)

    # Identify RB full names
    rb_full_names: set[str] = set()
    for full in rush["_full_name"].unique():
        is_rb = (
            metrics_by_player.get(full, {}).get("pos", "") == "RB"
            or sleeper_pos.get(full, "") == "RB"
        )
        if is_rb:
            rb_full_names.add(full)

    rb_name_team_pairs: set[tuple[str, str]] = {
        k for k, v in name_team_cache.items() if v in rb_full_names
    }
    rb_abbrev_names: set[str] = {ab for ab, _ in rb_name_team_pairs}

    # Receiving stats
    has_yac = "yards_after_catch" in pbp_season.columns

    pass_plays = pbp_season[
        (pbp_season["pass_attempt"] == 1) &
        pbp_season["receiver_player_name"].notna() &
        pbp_season["receiver_player_name"].isin(rb_abbrev_names)
    ].copy()

    rec_by_full: dict[str, dict] = {}
    if not pass_plays.empty:
        rec_name_cache: dict[tuple[str, str], str] = {}

        def _tag_rec(row: pd.Series) -> str:
            key = (str(row["receiver_player_name"]), str(row.get("posteam", "")))
            if key not in rec_name_cache:
                rec_name_cache[key] = _resolve_with_team(
                    key[0], key[1],
                    full_name_set, unambig_with_pfr, team_disambig,
                    metrics_by_player, sleeper_pos,
                )
            return rec_name_cache[key]

        pass_plays["_full_name"] = pass_plays.apply(_tag_rec, axis=1)
        pass_plays = pass_plays[pass_plays["_full_name"].isin(rb_full_names)]

        for full, grp in pass_plays.groupby("_full_name"):
            tgt    = len(grp)
            comps  = grp[grp["complete_pass"] == 1]
            rec    = len(comps)
            rz_tgt = int((grp["yardline_100"] <= 20).sum()) if has_yardline else None
            yac    = int(comps["yards_after_catch"].dropna().sum()) if has_yac else None
            rec_by_full[str(full)] = {
                "targets":          tgt,
                "receptions":       rec,
                "red_zone_targets": rz_tgt,
                "rec_yac":          yac,
            }

    # Aggregate rush stats per player
    rb_rush = rush[rush["_full_name"].isin(rb_full_names)].copy()
    rows: list[dict] = []

    for full_name, grp in rb_rush.groupby("_full_name"):
        rush_att = len(grp)
        if rush_att < MIN_CARRIES:
            continue

        rush_yds = int(grp["yards_gained"].sum())
        games    = int(grp["game_id"].nunique())
        ypa      = round(rush_yds / rush_att, 2) if rush_att else None

        team = str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns else ""
        info = metrics_by_player.get(str(full_name), {})
        if info.get("team"):
            team = info["team"]

        neg_plays    = grp[grp["yards_gained"] < 0]
        tfl          = int(len(neg_plays))
        tfl_yds_lost = int(abs(neg_plays["yards_gained"].sum()))

        yds      = grp["yards_gained"]
        runs_10  = int((yds >= 10).sum())
        runs_20  = int((yds >= 20).sum())
        runs_30  = int((yds >= 30).sum())
        runs_40  = int((yds >= 40).sum())
        runs_50  = int((yds >= 50).sum())

        max_idx        = grp["yards_gained"].idxmax()
        longest_run    = int(grp.loc[max_idx, "yards_gained"])
        td_val         = grp.loc[max_idx, "touchdown"] if "touchdown" in grp.columns else 0
        longest_run_td = 1 if td_val == 1 or td_val is True else 0

        pd_stats    = pfr_agg.get(str(full_name), {})
        pfr_carries = pd_stats.get("carries") or 0
        ybc_total   = pd_stats.get("ybc")
        yac_total   = pd_stats.get("yac")
        bt          = pd_stats.get("broken_tackles")

        ybc_per_att = (
            round(float(ybc_total) / pfr_carries, 2)
            if ybc_total is not None and pfr_carries > 0 else None
        )
        yac_per_att = (
            round(float(yac_total) / pfr_carries, 2)
            if yac_total is not None and pfr_carries > 0 else None
        )
        broken_tackles = int(bt) if bt is not None and not pd.isna(bt) else None

        rv = rec_by_full.get(str(full_name), {})

        rows.append({
            "player":                       str(full_name),
            "team":                         team,
            "games":                        games,
            "rush_att":                     rush_att,
            "rush_yds":                     rush_yds,
            "yards_per_att":                ypa,
            "yards_before_contact_per_att": ybc_per_att,
            "yards_after_contact_per_att":  yac_per_att,
            "broken_tackles":               broken_tackles,
            "tfl":                          tfl,
            "tfl_yds_lost":                 tfl_yds_lost,
            "runs_10_plus":                 runs_10,
            "runs_20_plus":                 runs_20,
            "runs_30_plus":                 runs_30,
            "runs_40_plus":                 runs_40,
            "runs_50_plus":                 runs_50,
            "longest_run":                  longest_run,
            "longest_run_td":               longest_run_td,
            "targets":                      rv.get("targets", 0),
            "receptions":                   rv.get("receptions", 0),
            "red_zone_targets":             rv.get("red_zone_targets", None),
            "rec_yac":                      rv.get("rec_yac", None),
        })

    # Sort by rush_att descending → assign within-season rank
    rows.sort(key=lambda r: r.get("rush_att") or 0, reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    return rows


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _make_payload(
    rows: list[dict],
    columns: list[dict],
    season: int | str,
    week: int | None,
    updated_at: str,
) -> dict[str, Any]:
    ordered_keys = [c["key"] for c in columns]
    ordered_rows = [{k: r.get(k) for k in ordered_keys} for r in rows]
    return {
        "updated_at": updated_at,
        "season":     season,
        "week":       week,
        "table":      "rb_advanced_stats",
        "columns":    columns,
        "rows":       ordered_rows,
    }


# ---------------------------------------------------------------------------
# Multi-season aggregation (one row per player, stats summed across seasons)
# ---------------------------------------------------------------------------

def _aggregate_combined_rb(rows: list[dict]) -> list[dict]:
    """
    Collapse per-season RB rows (each tagged with a '_season' key) into one
    combined row per player: counting stats summed, rates recalculated.
    team is taken from the most recent season row.
    """
    from collections import defaultdict

    SUM_FIELDS = [
        "games", "rush_att", "rush_yds", "broken_tackles",
        "tfl", "tfl_yds_lost",
        "runs_10_plus", "runs_20_plus", "runs_30_plus",
        "runs_40_plus", "runs_50_plus",
        "targets", "receptions", "red_zone_targets", "rec_yac",
    ]

    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_player[r["player"]].append(r)

    result: list[dict] = []
    for player, player_rows in by_player.items():
        latest = max(player_rows, key=lambda r: r.get("_season", 0))

        combined: dict = {
            "player": player,
            "team":   latest.get("team"),
        }

        for f in SUM_FIELDS:
            vals = [r[f] for r in player_rows if r.get(f) is not None]
            combined[f] = sum(vals) if vals else None

        lng_vals = [r["longest_run"] for r in player_rows if r.get("longest_run") is not None]
        combined["longest_run"]    = max(lng_vals) if lng_vals else None
        combined["longest_run_td"] = any(r.get("longest_run_td") for r in player_rows)

        rush_att = combined.get("rush_att") or 0
        rush_yds = combined.get("rush_yds") or 0
        if rush_att:
            combined["yards_per_att"] = round(rush_yds / rush_att, 2)
            ybc_num = sum(
                (r.get("yards_before_contact_per_att") or 0) * (r.get("rush_att") or 0)
                for r in player_rows
            )
            yac_num = sum(
                (r.get("yards_after_contact_per_att") or 0) * (r.get("rush_att") or 0)
                for r in player_rows
            )
            combined["yards_before_contact_per_att"] = round(ybc_num / rush_att, 2)
            combined["yards_after_contact_per_att"]  = round(yac_num / rush_att, 2)
        else:
            combined["yards_per_att"]                = None
            combined["yards_before_contact_per_att"] = None
            combined["yards_after_contact_per_att"]  = None

        result.append(combined)

    result.sort(key=lambda r: r.get("rush_att") or 0, reverse=True)
    for i, r in enumerate(result, 1):
        r["rank"] = i

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (PBP_PATH, PFR_PATH, SLEEPER_PATH, METRICS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    print("Loading play-by-play data (all seasons)...")
    pbp_full = pd.read_parquet(PBP_PATH)

    print("Loading PFR rush advanced stats...")
    pfr = pd.read_parquet(PFR_PATH)

    with open(SLEEPER_PATH) as f:
        sleeper_players: list[dict] = json.load(f)
    print(f"  {len(sleeper_players):,} Sleeper players.")

    with open(METRICS_PATH) as f:
        player_metrics: list[dict] = json.load(f)
    print(f"  {len(player_metrics):,} player metric records.")

    # Pre-build shared lookup tables (season-independent)
    full_name_set     = {p.get("full_name", "") for p in sleeper_players}
    unambig_lookup    = _build_sleeper_abbrev_lookup(sleeper_players)
    team_disambig     = _build_team_disambig(sleeper_players)
    pfr_lookup        = _build_pfr_abbrev_lookup(pfr)
    _pfr_safe         = {ab: fn for ab, fn in pfr_lookup.items() if ab not in team_disambig}
    unambig_with_pfr  = {**_pfr_safe, **unambig_lookup}
    sleeper_pos       = {p.get("full_name", ""): p.get("position", "") for p in sleeper_players}
    metrics_by_player = {p["player"]: p for p in player_metrics}

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build and write one file per season ───────────────────────────────────
    all_rows_with_season: list[dict] = []
    latest_week_2025: int | None = None

    for season in SEASONS:
        print(f"\nBuilding {season} season...")
        pbp_s = pbp_full[pbp_full["season"] == season].copy()
        week  = int(pbp_s["week"].max()) if "week" in pbp_s.columns and not pbp_s.empty else None

        if season == 2025:
            latest_week_2025 = week

        rows = build_season(
            pbp_season       = pbp_s,
            pfr              = pfr,
            season           = season,
            sleeper_players  = sleeper_players,
            player_metrics   = player_metrics,
            full_name_set    = full_name_set,
            unambig_with_pfr = unambig_with_pfr,
            team_disambig    = team_disambig,
            metrics_by_player= metrics_by_player,
            sleeper_pos      = sleeper_pos,
        )
        print(f"  {len(rows)} RBs qualified for {season}.")

        # Per-season file
        out_path = OUTPUT_DIR / f"rb_advanced_stats_{season}.json"
        payload  = _make_payload(rows, COLUMNS_SEASON, season, week, updated_at)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

        # Tag rows with season for aggregation (using private key to avoid
        # polluting the output columns)
        for r in rows:
            all_rows_with_season.append({**r, "_season": season})

    # ── Multi-season file (one combined row per player) ────────────────────────
    combined_rows = _aggregate_combined_rb(all_rows_with_season)
    multi_path = OUTPUT_DIR / "rb_advanced_stats_all.json"
    multi_payload = _make_payload(
        rows       = combined_rows,
        columns    = COLUMNS_SEASON,
        season     = "all",
        week       = None,
        updated_at = updated_at,
    )
    with open(multi_path, "w") as f:
        json.dump(multi_payload, f, indent=2)
    print(f"\nWrote {multi_path} ({multi_path.stat().st_size/1024:.1f} KB, {len(combined_rows)} players combined)")

    # ── Backward-compat alias: rb_advanced_stats.json → 2025 ─────────────────
    alias_path = OUTPUT_DIR / "rb_advanced_stats.json"
    shutil.copy2(OUTPUT_DIR / "rb_advanced_stats_2025.json", alias_path)
    print(f"Wrote {alias_path} (alias for 2025)")

    print("\nRB advanced stats build complete.")


if __name__ == "__main__":
    main()
