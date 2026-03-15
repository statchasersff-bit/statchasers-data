"""
build_rb_advanced_stats.py
──────────────────────────
Builds the RB Advanced Stats dataset for the StatChasers frontend.

One row per qualifying RB with rushing and receiving counting stats,
PFR-sourced contact metrics, and explosive-run breakdowns.

Data sources (pre-pulled by pull_nflverse_data.py / pull_sleeper_players.py):
  data/raw/nflverse_play_by_play.parquet  — rush/receiving play-level data
  data/raw/pfr_rush_advstats.parquet      — YBC, YAC, broken tackles (PFR)
  data/raw/sleeper_players.json           — canonical name lookup
  data/processed/player_metrics.json      — position / team context

Output:
  output/rb_advanced_stats.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
PBP_PATH     = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
PFR_PATH     = ROOT / "data" / "raw"       / "pfr_rush_advstats.parquet"
SLEEPER_PATH = ROOT / "data" / "raw"       / "sleeper_players.json"
METRICS_PATH = ROOT / "data" / "processed" / "player_metrics.json"
OUTPUT_PATH  = ROOT / "output"             / "rb_advanced_stats.json"

SEASON       = 2025
MIN_CARRIES  = 15

COLUMNS: list[dict] = [
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


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def _abbrev(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _build_sleeper_abbrev_lookup(sleeper_players: list[dict]) -> dict[str, str]:
    """Unambiguous abbrev → full_name from Sleeper (drops collisions)."""
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
    """
    For abbreviated names that collide across players, build
    { abbrev: { team: full_name } } so we can resolve by team.
    """
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
    """
    Resolve abbreviated PBP name to canonical full name using team hint.

    Resolution order:
    1. Direct Sleeper full-name match.
    2. Unambiguous abbreviation lookup.
    3. Team-based disambiguation:
       a. Exact team match.
       b. Position filter — if only one pos match, use it.
       c. Conflict check — if some candidates have a Sleeper team that
          *contradicts* pbp_team (explicit ≠ pbp) and at least one candidate
          has team=None (unsigned/recently traded), prefer the None-team player.
          This correctly handles e.g. two "B.Robinson" where one is on ATL
          in Sleeper but the PBP team is SF → prefer the unrostered Robinson.
       d. Fall back to first candidate with an explicit team.
    4. Return PBP name unchanged (fallback).
    """
    if pbp_name in full_name_set:
        return pbp_name
    if pbp_name in unambiguous_lookup:
        return unambiguous_lookup[pbp_name]
    if pbp_name in team_disambig:
        teams = team_disambig[pbp_name]

        # 1. Exact Sleeper-team match
        if pbp_team and pbp_team in teams:
            return teams[pbp_team]

        # 2. Position filter — use Sleeper position (covers unsigned players
        #    not present in player_metrics, e.g. Brian Robinson team=None).
        pos_matches = [
            (t, n) for t, n in teams.items()
            if sleeper_pos.get(n, "") == pos_hint
        ]
        if not pos_matches:
            # Fallback to metrics-based position
            pos_matches = [
                (t, n) for t, n in teams.items()
                if metrics_by_player.get(n, {}).get("pos", "") == pos_hint
            ]
        if not pos_matches:
            pos_matches = list(teams.items())

        if len(pos_matches) == 1:
            return pos_matches[0][1]

        # 3. Conflict check — separate players whose Sleeper team contradicts pbp_team
        conflicting    = [(t, n) for t, n in pos_matches if t is not None and t != pbp_team]
        non_conflicting = [(t, n) for t, n in pos_matches if t is None or t == pbp_team]

        if non_conflicting and conflicting:
            # The player with no fixed Sleeper team (or matching team) is
            # more likely to be the one playing for pbp_team.
            no_team = [(t, n) for t, n in non_conflicting if t is None]
            if no_team:
                return no_team[0][1]
            return non_conflicting[0][1]

        # 4. Fallback — prefer player with an explicit team
        with_team = [(t, n) for t, n in pos_matches if t is not None]
        if with_team:
            return with_team[0][1]
        return pos_matches[0][1]

    return pbp_name


# ---------------------------------------------------------------------------
# PFR aggregation — sum season totals per player (full name as key)
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
# Main builder
# ---------------------------------------------------------------------------

def build_rb_advanced_stats(
    pbp: pd.DataFrame,
    pfr: pd.DataFrame,
    sleeper_players: list[dict],
    player_metrics: list[dict],
) -> list[dict]:

    full_name_set     = {p.get("full_name", "") for p in sleeper_players}
    unambig_lookup    = _build_sleeper_abbrev_lookup(sleeper_players)
    team_disambig     = _build_team_disambig(sleeper_players)
    pfr_lookup        = _build_pfr_abbrev_lookup(pfr)
    # PFR supplements Sleeper for unambiguous names ONLY.
    # Strip any PFR entries whose abbreviation already appears in team_disambig —
    # those collisions are known by Sleeper and must go through team-based resolution
    # (e.g. PFR only has Bijan Robinson, so "B.Robinson" looks unambiguous there,
    # but Sleeper knows Brian Robinson Jr. also maps to "B.Robinson").
    _pfr_safe         = {ab: fn for ab, fn in pfr_lookup.items() if ab not in team_disambig}
    unambig_with_pfr  = {**_pfr_safe, **unambig_lookup}  # Sleeper wins conflicts
    # Sleeper position lookup — includes unrostered players not in player_metrics
    sleeper_pos       = {p.get("full_name", ""): p.get("position", "") for p in sleeper_players}

    metrics_by_player = {p["player"]: p for p in player_metrics}
    pfr_agg           = _aggregate_pfr_rush(pfr, SEASON)

    rush = pbp[pbp["rush_attempt"] == 1].copy()
    if rush.empty:
        return []

    has_yardline = "yardline_100" in rush.columns

    # ── Step 1: tag every rush play with a resolved full name ────────────────
    # Group by (rusher_player_name, posteam) to avoid merging different players
    # with the same abbreviation (e.g. two "B.Robinson" on different teams).
    # Then resolve each (abbrev, team) → full_name and stamp on the play rows.

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

    # ── Step 2: identify which resolved names are RBs ───────────────────────
    # Include any player whose Sleeper position is RB (covers unrostered players
    # not present in player_metrics, e.g. Brian Robinson with team=None in Sleeper).
    rb_full_names: set[str] = set()
    for full in rush["_full_name"].unique():
        is_rb = (
            metrics_by_player.get(full, {}).get("pos", "") == "RB"
            or sleeper_pos.get(full, "") == "RB"
        )
        if is_rb:
            rb_full_names.add(full)

    # Also build the set of (abbreviated name, team) pairs that map to RBs,
    # so we can efficiently filter receiving plays.
    rb_name_team_pairs: set[tuple[str, str]] = {
        k for k, v in name_team_cache.items() if v in rb_full_names
    }
    rb_abbrev_names: set[str] = {ab for ab, _ in rb_name_team_pairs}

    # ── Step 3: receiving stats ──────────────────────────────────────────────
    # Tag pass plays in the same (receiver, posteam) → full_name way.
    has_yac = "yards_after_catch" in pbp.columns

    pass_plays = pbp[
        (pbp["pass_attempt"] == 1) &
        pbp["receiver_player_name"].notna() &
        pbp["receiver_player_name"].isin(rb_abbrev_names)
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
            tgt   = len(grp)
            comps = grp[grp["complete_pass"] == 1]
            rec   = len(comps)
            rz_tgt = int((grp["yardline_100"] <= 20).sum()) if has_yardline else None
            yac    = int(comps["yards_after_catch"].dropna().sum()) if has_yac else None
            rec_by_full[str(full)] = {
                "targets":          tgt,
                "receptions":       rec,
                "red_zone_targets": rz_tgt,
                "rec_yac":          yac,
            }

    # ── Step 4: aggregate rush stats per canonical full name ─────────────────
    rb_rush = rush[rush["_full_name"].isin(rb_full_names)].copy()

    rows: list[dict] = []
    for full_name, grp in rb_rush.groupby("_full_name"):
        rush_att = len(grp)
        if rush_att < MIN_CARRIES:
            continue

        rush_yds = int(grp["yards_gained"].sum())
        games    = int(grp["game_id"].nunique())
        ypa      = round(rush_yds / rush_att, 2) if rush_att else None

        # Use most-recent posteam as the player's current team
        team = str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns else ""
        info = metrics_by_player.get(str(full_name), {})
        if info.get("team"):
            team = info["team"]

        # TFL: negative yards (proxy)
        neg_plays    = grp[grp["yards_gained"] < 0]
        tfl          = int(len(neg_plays))
        tfl_yds_lost = int(abs(neg_plays["yards_gained"].sum()))

        # Explosive runs
        yds = grp["yards_gained"]
        runs_10 = int((yds >= 10).sum())
        runs_20 = int((yds >= 20).sum())
        runs_30 = int((yds >= 30).sum())
        runs_40 = int((yds >= 40).sum())
        runs_50 = int((yds >= 50).sum())

        # Longest run + TD flag
        max_idx     = grp["yards_gained"].idxmax()
        longest_run = int(grp.loc[max_idx, "yards_gained"])
        td_val      = grp.loc[max_idx, "touchdown"] if "touchdown" in grp.columns else 0
        longest_run_td = 1 if td_val == 1 or td_val is True else 0

        # PFR contact stats (matched by full name)
        pd_stats    = pfr_agg.get(str(full_name), {})
        pfr_carries = pd_stats.get("carries") or 0
        ybc_total   = pd_stats.get("ybc")
        yac_total   = pd_stats.get("yac")
        bt          = pd_stats.get("broken_tackles")

        ybc_per_att = (
            round(float(ybc_total) / pfr_carries, 2)
            if ybc_total is not None and pfr_carries > 0
            else None
        )
        yac_per_att = (
            round(float(yac_total) / pfr_carries, 2)
            if yac_total is not None and pfr_carries > 0
            else None
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

    # Sort by rush_att descending → assign rank
    rows.sort(key=lambda r: r.get("rush_att") or 0, reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    # Re-order keys to match spec
    ordered_keys = [c["key"] for c in COLUMNS]
    rows = [{k: r.get(k) for k in ordered_keys} for r in rows]

    return rows


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
    pbp = pbp_full[pbp_full["season"] == SEASON].copy()
    print(f"  {len(pbp):,} plays for {SEASON} season.")

    print("Loading PFR rush advanced stats...")
    pfr = pd.read_parquet(PFR_PATH)

    with open(SLEEPER_PATH) as f:
        sleeper_players: list[dict] = json.load(f)
    print(f"  {len(sleeper_players):,} Sleeper players.")

    with open(METRICS_PATH) as f:
        player_metrics: list[dict] = json.load(f)
    print(f"  {len(player_metrics):,} player metric records.")

    print("Building RB advanced stats...")
    rows = build_rb_advanced_stats(pbp, pfr, sleeper_players, player_metrics)
    print(f"  {len(rows)} RBs qualified.")

    latest_week = int(pbp["week"].max()) if "week" in pbp.columns else None

    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season":     SEASON,
        "week":       latest_week,
        "table":      "rb_advanced_stats",
        "columns":    COLUMNS,
        "rows":       rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.1f} KB, {len(rows)} RBs)")
    print("RB advanced stats build complete.")


if __name__ == "__main__":
    main()
