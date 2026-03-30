"""
build_wr_advanced_stats.py
──────────────────────────
Builds the WR Advanced Stats datasets for the StatChasers frontend.

Produces five output files:
  output/wr_advanced_stats_2023.json   — 2023 season only
  output/wr_advanced_stats_2024.json   — 2024 season only
  output/wr_advanced_stats_2025.json   — 2025 season only
  output/wr_advanced_stats_all.json    — 2023 + 2024 + 2025 combined (one row per player)
  output/wr_advanced_stats.json        — alias for 2025 (backward compat)

Data sources (pre-pulled by pull_nflverse_data.py / pull_sleeper_players.py):
  data/raw/nflverse_play_by_play.parquet  — play-level receiving / target data
  data/raw/pfr_rec_advstats.parquet       — receiving broken tackles (PFR)
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
PFR_PATH     = ROOT / "data" / "raw"       / "pfr_rec_advstats.parquet"
SLEEPER_PATH = ROOT / "data" / "raw"       / "sleeper_players.json"
METRICS_PATH = ROOT / "data" / "processed" / "player_metrics.json"
OUTPUT_DIR   = ROOT / "output"

SEASONS     = [2023, 2024, 2025]
MIN_TARGETS = 15

# ---------------------------------------------------------------------------
# Column definitions (match the spec exactly)
# ---------------------------------------------------------------------------

COLUMNS: list[dict] = [
    {"key": "rank",         "label": "Rank",        "type": "number",  "defaultVisible": True},
    {"key": "player",       "label": "Player",      "type": "string",  "defaultVisible": True},
    {"key": "team",         "label": "Team",        "type": "string",  "defaultVisible": True},
    {"key": "g",            "label": "G",           "type": "number",  "defaultVisible": True},
    {"key": "rec",          "label": "REC",         "type": "number",  "defaultVisible": True},
    {"key": "yds",          "label": "YDS",         "type": "number",  "defaultVisible": True},
    {"key": "ypr",          "label": "Y/R",         "type": "decimal", "defaultVisible": True},
    {"key": "ybc",          "label": "YBC",         "type": "number",  "defaultVisible": True},
    {"key": "ybc_per_rec",  "label": "YBC/R",       "type": "decimal", "defaultVisible": True},
    {"key": "yac",          "label": "YAC",         "type": "number",  "defaultVisible": True},
    {"key": "yac_per_rec",  "label": "YAC/R",       "type": "decimal", "defaultVisible": True},
    {"key": "brktkl",       "label": "BRKTKL",      "type": "number",  "defaultVisible": True},
    {"key": "tgt",          "label": "TGT",         "type": "number",  "defaultVisible": True},
    {"key": "target_share", "label": "TGT Share",   "type": "decimal", "defaultVisible": True},
    {"key": "rz_tgt",       "label": "RZ TGT",      "type": "number",  "defaultVisible": True},
    {"key": "rec_10_plus",  "label": "10+ YDS",     "type": "number",  "defaultVisible": True},
    {"key": "rec_20_plus",  "label": "20+ YDS",     "type": "number",  "defaultVisible": True},
    {"key": "rec_30_plus",  "label": "30+ YDS",     "type": "number",  "defaultVisible": False},
    {"key": "rec_40_plus",  "label": "40+ YDS",     "type": "number",  "defaultVisible": False},
    {"key": "rec_50_plus",  "label": "50+ YDS",     "type": "number",  "defaultVisible": False},
    {"key": "lng",          "label": "LNG",         "type": "number",  "defaultVisible": True},
]


# ---------------------------------------------------------------------------
# Name resolution (same pattern as build_rb_advanced_stats.py)
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


# WR-specific manual overrides for players whose abbreviation is ambiguous
# and whose team in the PBP is the clearest disambiguator.
#
# Root cause: when multiple Sleeper players share team=None the dict collapses
# to one entry and tiebreak logic picks the wrong player.  Overrides here are
# authoritative — PBP posteam is the source of truth.
_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    # ── Cross-position guards (RB abbreviations that appear in pass plays) ───
    "T.Etienne": {"JAX": "Travis Etienne", "CAR": "Trevor Etienne"},
    "B.Robinson": {"ATL": "Bijan Robinson", "SF": "Brian Robinson", "WAS": "Brian Robinson"},
    "J.Williams": {"DEN": "Javonte Williams", "NO": "Jamaal Williams"},
    # ── WR / name-collision fixes ────────────────────────────────────────────
    # D.Moore: CHI/BUF = DJ Moore; CAR/TB = David Moore; Denarius Moore retired
    "D.Moore": {
        "CHI": "DJ Moore",
        "BUF": "DJ Moore",
        "CAR": "David Moore",
        "TB":  "David Moore",
    },
    # K.Allen: Keenan Allen (LAC 2023, CHI 2024, LAC 2025) wins over
    # Kazmeir Allen (team=None) and Kaytron Allen (RB, team=None)
    "K.Allen": {
        "LAC": "Keenan Allen",
        "CHI": "Keenan Allen",
    },
    # D.Montgomery: DET/HOU = David Montgomery (RB → excluded by pos filter);
    # IND = D.J. Montgomery (WR, ~8 tgt — below threshold anyway)
    "D.Montgomery": {
        "DET": "David Montgomery",
        "HOU": "David Montgomery",
        "IND": "D.J. Montgomery",
    },
    # K.Williams: LA = Kyren Williams (RB → excluded by pos filter);
    # NE = Kyle Williams (WR); CIN = Ke'Shawn Williams (WR)
    "K.Williams": {
        "LA":  "Kyren Williams",
        "NE":  "Kyle Williams",
        "CIN": "Ke'Shawn Williams",
    },
}


def _resolve_with_team(
    pbp_name: str,
    pbp_team: str,
    full_name_set: set[str],
    unambiguous_lookup: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    metrics_by_player: dict[str, dict],
    sleeper_pos: dict[str, str],
    pos_hint: str = "WR",
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

        conflicting     = [(t, n) for t, n in pos_matches if t is not None and t != pbp_team]
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
# PFR broken-tackle aggregation (receiving)
# ---------------------------------------------------------------------------

def _aggregate_pfr_rec(pfr: pd.DataFrame, season: int) -> dict[str, dict]:
    s = pfr[pfr["season"] == season].copy()
    if s.empty:
        return {}
    agg = (
        s.groupby("pfr_player_name")
        .agg(brktkl=("receiving_broken_tackles", "sum"))
        .to_dict("index")
    )
    return agg


# ---------------------------------------------------------------------------
# Per-season builder
# ---------------------------------------------------------------------------

def build_season(
    pbp_season: pd.DataFrame,
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
    """Build one season's WR advanced stats rows (no rank assigned yet)."""

    pfr_agg = _aggregate_pfr_rec(pfr, season)

    pass_plays = pbp_season[pbp_season["pass_attempt"] == 1].copy()
    if pass_plays.empty:
        return []

    has_yardline = "yardline_100" in pass_plays.columns
    has_yac_col  = "yards_after_catch" in pass_plays.columns
    has_air      = "air_yards" in pass_plays.columns

    # ── Resolve receiver full names ───────────────────────────────────────────
    name_team_cache: dict[tuple[str, str], str] = {}

    def _tag(row: pd.Series) -> str:
        pbp_name = str(row.get("receiver_player_name") or "")
        if not pbp_name or pbp_name == "nan":
            return ""
        key = (pbp_name, str(row.get("posteam", "")))
        if key not in name_team_cache:
            name_team_cache[key] = _resolve_with_team(
                key[0], key[1],
                full_name_set, unambig_with_pfr, team_disambig,
                metrics_by_player, sleeper_pos,
            )
        return name_team_cache[key]

    targeted = pass_plays[pass_plays["receiver_player_name"].notna()].copy()
    targeted["_full_name"] = targeted.apply(_tag, axis=1)
    targeted = targeted[targeted["_full_name"] != ""]

    # ── Identify WR full names ────────────────────────────────────────────────
    wr_full_names: set[str] = set()
    for full in targeted["_full_name"].unique():
        full = str(full)
        pos_metrics = metrics_by_player.get(full, {}).get("pos", "")
        pos_sleeper = sleeper_pos.get(full, "")
        if pos_metrics == "WR" or pos_sleeper == "WR":
            wr_full_names.add(full)

    wr_plays = targeted[targeted["_full_name"].isin(wr_full_names)].copy()
    if wr_plays.empty:
        return []

    # ── Team total pass attempts per team (for target_share denominator) ──────
    team_pass_totals: dict[str, int] = (
        pass_plays.groupby("posteam")["pass_attempt"].count().to_dict()
    )

    # ── Aggregate per player ──────────────────────────────────────────────────
    rows: list[dict] = []

    for full_name, grp in wr_plays.groupby("_full_name"):
        full_name = str(full_name)
        tgt = len(grp)
        if tgt < MIN_TARGETS:
            continue

        comps = grp[grp["complete_pass"] == 1]
        rec   = len(comps)
        yds   = int(comps["yards_gained"].sum())
        g     = int(grp["game_id"].nunique())

        # Team: use PBP posteam (ground truth — do NOT override with player_metrics)
        team = str(grp.sort_values(["season", "week"])["posteam"].iloc[-1]) \
               if "season" in grp.columns and "week" in grp.columns \
               else str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns else ""

        # Rates
        ypr = round(yds / rec, 1) if rec > 0 else None

        # YBC = air yards on completed passes (depth of target at catch point)
        if has_air and rec > 0:
            ybc_total = int(comps["air_yards"].dropna().sum())
            ybc_per_rec = round(ybc_total / rec, 1)
        else:
            ybc_total   = None
            ybc_per_rec = None

        # YAC = yards after catch on completed passes
        if has_yac_col and rec > 0:
            yac_total = int(comps["yards_after_catch"].dropna().sum())
            yac_per_rec = round(yac_total / rec, 1)
        else:
            yac_total   = None
            yac_per_rec = None

        # Red-zone targets
        rz_tgt = int((grp["yardline_100"] <= 20).sum()) if has_yardline else None

        # Target share (decimal, 4 places)
        team_tgt_total = team_pass_totals.get(team, 0)
        target_share = round(tgt / team_tgt_total, 4) if team_tgt_total > 0 else None

        # Explosive receptions (on completions only)
        comp_yds = comps["yards_gained"]
        rec_10   = int((comp_yds >= 10).sum())
        rec_20   = int((comp_yds >= 20).sum())
        rec_30   = int((comp_yds >= 30).sum())
        rec_40   = int((comp_yds >= 40).sum())
        rec_50   = int((comp_yds >= 50).sum())
        lng      = int(comp_yds.max()) if rec > 0 else None

        # Broken tackles from PFR (matched by abbreviation → full name)
        pfr_stats = pfr_agg.get(full_name, {})
        bt = pfr_stats.get("brktkl")
        brktkl = int(bt) if bt is not None and not pd.isna(bt) else None

        rows.append({
            "player":       full_name,
            "team":         team,
            "g":            g,
            "rec":          rec,
            "yds":          yds,
            "ypr":          ypr,
            "ybc":          ybc_total,
            "ybc_per_rec":  ybc_per_rec,
            "yac":          yac_total,
            "yac_per_rec":  yac_per_rec,
            "brktkl":       brktkl,
            "tgt":          tgt,
            "target_share": target_share,
            "rz_tgt":       rz_tgt,
            "rec_10_plus":  rec_10,
            "rec_20_plus":  rec_20,
            "rec_30_plus":  rec_30,
            "rec_40_plus":  rec_40,
            "rec_50_plus":  rec_50,
            "lng":          lng,
        })

    # Default sort: yds desc → rec desc → tgt desc
    rows.sort(key=lambda r: (r.get("yds") or 0, r.get("rec") or 0, r.get("tgt") or 0), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    return rows


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _make_payload(
    rows: list[dict],
    season: int | str,
    week: int | None,
    updated_at: str,
) -> dict[str, Any]:
    ordered_keys = [c["key"] for c in COLUMNS]
    ordered_rows = [{k: r.get(k) for k in ordered_keys} for r in rows]
    return {
        "updated_at": updated_at,
        "season":     season,
        "week":       week,
        "table":      "wr_advanced_stats",
        "columns":    COLUMNS,
        "rows":       ordered_rows,
    }


# ---------------------------------------------------------------------------
# Multi-season aggregation (one combined row per player)
# ---------------------------------------------------------------------------

def _aggregate_combined_wr(rows: list[dict]) -> list[dict]:
    """
    Collapse per-season WR rows (each tagged with '_season') into one
    combined row per player: counting stats summed, rates recalculated from totals.
    team taken from most recent season. target_share set to null (not meaningful
    across seasons).
    """
    from collections import defaultdict

    SUM_FIELDS = [
        "g", "rec", "yds", "ybc", "yac", "brktkl",
        "tgt", "rz_tgt",
        "rec_10_plus", "rec_20_plus", "rec_30_plus",
        "rec_40_plus", "rec_50_plus",
    ]

    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_player[r["player"]].append(r)

    result: list[dict] = []
    for player, player_rows in by_player.items():
        latest = max(player_rows, key=lambda r: r.get("_season", 0))

        combined: dict = {
            "player":       player,
            "team":         latest.get("team"),
            "target_share": None,   # not meaningful across seasons
        }

        for f in SUM_FIELDS:
            vals = [r[f] for r in player_rows if r.get(f) is not None]
            combined[f] = sum(vals) if vals else None

        rec = combined.get("rec") or 0
        yds = combined.get("yds") or 0
        ybc = combined.get("ybc")
        yac = combined.get("yac")

        combined["ypr"]         = round(yds / rec, 1) if rec > 0 else None
        combined["ybc_per_rec"] = round(ybc / rec, 1) if rec > 0 and ybc is not None else None
        combined["yac_per_rec"] = round(yac / rec, 1) if rec > 0 and yac is not None else None

        lng_vals = [r["lng"] for r in player_rows if r.get("lng") is not None]
        combined["lng"] = max(lng_vals) if lng_vals else None

        result.append(combined)

    result.sort(key=lambda r: (r.get("yds") or 0, r.get("rec") or 0, r.get("tgt") or 0), reverse=True)
    for i, r in enumerate(result, 1):
        r["rank"] = i

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(rows: list[dict], season_label: str) -> None:
    required_keys = {c["key"] for c in COLUMNS}
    seen_players: set[str] = set()

    for r in rows:
        player = r.get("player", "")
        if player in seen_players:
            print(f"  WARNING: duplicate player '{player}' in {season_label}", file=sys.stderr)
        seen_players.add(player)

        missing = required_keys - set(r.keys())
        if missing:
            print(f"  WARNING: {player} missing keys {missing}", file=sys.stderr)

    print(f"  Validation OK for {season_label}: {len(rows)} players, no duplicate IDs")


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
    pbp_full = pbp_full[pbp_full["season_type"] == "REG"].copy()  # regular season only

    print("Loading PFR receiving advanced stats...")
    pfr = pd.read_parquet(PFR_PATH)

    with open(SLEEPER_PATH) as f:
        raw = json.load(f)
    sleeper_players: list[dict] = list(raw.values()) if isinstance(raw, dict) else raw
    print(f"  {len(sleeper_players):,} Sleeper players.")

    with open(METRICS_PATH) as f:
        player_metrics: list[dict] = json.load(f)
    print(f"  {len(player_metrics):,} player metric records.")

    # Pre-build shared lookup tables
    full_name_set     = {(p.get("full_name") or "").strip() for p in sleeper_players}
    unambig_lookup    = _build_sleeper_abbrev_lookup(sleeper_players)
    team_disambig     = _build_team_disambig(sleeper_players)
    pfr_lookup        = _build_pfr_abbrev_lookup(pfr)
    pfr_safe          = {ab: fn for ab, fn in pfr_lookup.items() if ab not in team_disambig}
    unambig_with_pfr  = {**pfr_safe, **unambig_lookup}
    sleeper_pos       = {(p.get("full_name") or "").strip(): p.get("position", "") for p in sleeper_players}
    metrics_by_player = {p["player"]: p for p in player_metrics}

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows_with_season: list[dict] = []
    latest_week_2025: int | None = None

    for season in SEASONS:
        print(f"\nBuilding {season} season...")
        pbp_s = pbp_full[pbp_full["season"] == season].copy()
        week  = int(pbp_s["week"].max()) if "week" in pbp_s.columns and not pbp_s.empty else None

        if season == 2025:
            latest_week_2025 = week

        rows = build_season(
            pbp_season        = pbp_s,
            pfr               = pfr,
            season            = season,
            sleeper_players   = sleeper_players,
            player_metrics    = player_metrics,
            full_name_set     = full_name_set,
            unambig_with_pfr  = unambig_with_pfr,
            team_disambig     = team_disambig,
            metrics_by_player = metrics_by_player,
            sleeper_pos       = sleeper_pos,
        )
        print(f"  {len(rows)} WRs qualified for {season}.")
        _validate(rows, str(season))

        out_path = OUTPUT_DIR / f"wr_advanced_stats_{season}.json"
        payload  = _make_payload(rows, season, week, updated_at)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        for r in rows:
            all_rows_with_season.append({**r, "_season": season})

    # Multi-season combined file
    combined_rows = _aggregate_combined_wr(all_rows_with_season)
    _validate(combined_rows, "all-seasons")
    multi_path    = OUTPUT_DIR / "wr_advanced_stats_all.json"
    multi_payload = _make_payload(combined_rows, "all", None, updated_at)
    with open(multi_path, "w") as f:
        json.dump(multi_payload, f, indent=2)
    print(f"\nWrote {multi_path} ({multi_path.stat().st_size / 1024:.1f} KB, {len(combined_rows)} players combined)")

    # Alias: wr_advanced_stats.json → 2025
    alias_path = OUTPUT_DIR / "wr_advanced_stats.json"
    shutil.copy2(OUTPUT_DIR / "wr_advanced_stats_2025.json", alias_path)
    print(f"Wrote {alias_path} (alias for 2025)")

    print("\nWR advanced stats build complete.")
    print(f"\nEndpoints:")
    for s in [*SEASONS, "all"]:
        slug = f"wr_advanced_stats_{s}.json"
        print(f"  output/{slug}")


if __name__ == "__main__":
    main()
