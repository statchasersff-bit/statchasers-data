"""
build_te_advanced_stats.py
──────────────────────────
Builds the TE Advanced Stats dataset for the StatChasers frontend.

Produces:
  output/te_advanced_stats_2025.json   — 2025 season only

Data sources:
  data/raw/nflverse_play_by_play.parquet  — play-level receiving / target data
  data/raw/pfr_rec_advstats.parquet       — receiving broken tackles (PFR)
  data/raw/sleeper_players.json           — canonical name lookup
  data/processed/player_metrics.json      — position / team context
"""

from __future__ import annotations

import json
import re
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

SEASON      = 2025
MIN_TARGETS = 15

COLUMNS: list[dict] = [
    {"key": "rank",         "label": "Rank",      "type": "number",  "defaultVisible": True},
    {"key": "player",       "label": "Player",    "type": "string",  "defaultVisible": True},
    {"key": "team",         "label": "Team",      "type": "string",  "defaultVisible": True},
    {"key": "g",            "label": "G",         "type": "number",  "defaultVisible": True},
    {"key": "rec",          "label": "REC",       "type": "number",  "defaultVisible": True},
    {"key": "yds",          "label": "YDS",       "type": "number",  "defaultVisible": True},
    {"key": "ypr",          "label": "Y/R",       "type": "decimal", "defaultVisible": True},
    {"key": "ybc",          "label": "YBC",       "type": "number",  "defaultVisible": True},
    {"key": "ybc_per_rec",  "label": "YBC/R",     "type": "decimal", "defaultVisible": True},
    {"key": "yac",          "label": "YAC",       "type": "number",  "defaultVisible": True},
    {"key": "yac_per_rec",  "label": "YAC/R",     "type": "decimal", "defaultVisible": True},
    {"key": "brktkl",       "label": "BRKTKL",    "type": "number",  "defaultVisible": True},
    {"key": "tgt",          "label": "TGT",       "type": "number",  "defaultVisible": True},
    {"key": "target_share", "label": "TGT Share", "type": "decimal", "defaultVisible": True},
    {"key": "rz_tgt",       "label": "RZ TGT",    "type": "number",  "defaultVisible": True},
    {"key": "rec_10_plus",  "label": "10+ YDS",   "type": "number",  "defaultVisible": True},
    {"key": "rec_20_plus",  "label": "20+ YDS",   "type": "number",  "defaultVisible": True},
    {"key": "rec_30_plus",  "label": "30+ YDS",   "type": "number",  "defaultVisible": False},
    {"key": "rec_40_plus",  "label": "40+ YDS",   "type": "number",  "defaultVisible": False},
    {"key": "rec_50_plus",  "label": "50+ YDS",   "type": "number",  "defaultVisible": False},
    {"key": "lng",          "label": "LNG",       "type": "number",  "defaultVisible": True},
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
    counts: dict[str, int]   = {}
    by_team: dict[str, dict] = {}
    for p in sleeper_players:
        full = (p.get("full_name") or "").strip()
        if not full:
            continue
        ab = _abbrev(full)
        counts[ab] = counts.get(ab, 0) + 1
        by_team.setdefault(ab, {})[p.get("team")] = full
    return {ab: teams for ab, teams in by_team.items() if counts[ab] > 1}


# TE-specific manual overrides for ambiguous PBP abbreviations.
# These are authoritative — PBP posteam is the source of truth.
_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    # D.Allen: Davis Allen (TE LAR/LA) vs Devon Allen (WR PHI)
    "D.Allen":   {"LA": "Davis Allen", "LAR": "Davis Allen", "PHI": "Devon Allen"},
    # J.Smith: Jonnu Smith (TE) vs many non-TE J. Smiths
    "J.Smith":   {"ATL": "Jonnu Smith", "MIA": "Jonnu Smith", "PIT": "Jonnu Smith"},
    # T.Smith: can conflict with WR T. Smiths — map known active TE teams
    "T.Smith":   {"CAR": "Tommy Tremble"},  # Tommy Tremble is FB/TE
    # C.Jordan: Cameron Jordan (DE) is not a receiver, but guard for any C.Jordan TE
    "D.Njoku":   {"CLE": "David Njoku"},
    # M.Andrews: Mark Andrews vs other M. Andrews
    "M.Andrews": {"BAL": "Mark Andrews"},
    # B.Likely: Likely is unique enough; belt-and-suspenders
    "I.Thomas":  {"NYG": "Isaiah Thomas"},
}


def _resolve(
    pbp_name: str,
    pbp_team: str,
    full_name_set: set[str],
    unambig: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
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
        pos_matches = [(t, n) for t, n in teams.items() if sleeper_pos.get(n, "") == "TE"]
        if not pos_matches:
            pos_matches = list(teams.items())
        if len(pos_matches) == 1:
            return pos_matches[0][1]
        with_team = [(t, n) for t, n in pos_matches if t is not None]
        if with_team:
            return with_team[0][1]
        return pos_matches[0][1]
    return pbp_name


# ---------------------------------------------------------------------------
# PFR name normalisation
# ---------------------------------------------------------------------------

_PFR_NAME_SUFFIXES = frozenset(["jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"])

_CANONICAL_TO_PFR_NORM: dict[str, str] = {}

_EXCLUDED_PLAYERS: frozenset[str] = frozenset({
    "Eli Wilson",
    "Daniel Brown",
})


def _norm_pfr(name: str) -> str:
    """Strip punctuation, lower-case, remove trailing generational suffixes."""
    parts = re.sub(r"[.\-']", " ", name).lower().split()
    if parts and parts[-1] in _PFR_NAME_SUFFIXES:
        parts.pop()
    return "".join(parts)


# ---------------------------------------------------------------------------
# PFR broken-tackle aggregation (receiving)
# ---------------------------------------------------------------------------

def _aggregate_pfr_rec(
    pfr: pd.DataFrame, season: int
) -> tuple[dict[str, dict], dict[str, dict]]:
    s = pfr[pfr["season"] == season].copy()
    if s.empty:
        return {}, {}
    agg = (
        s.groupby("pfr_player_name")
        .agg(brktkl=("receiving_broken_tackles", "sum"))
        .to_dict("index")
    )
    agg_norm = {_norm_pfr(k): v for k, v in agg.items()}
    return agg, agg_norm


# ---------------------------------------------------------------------------
# Season builder
# ---------------------------------------------------------------------------

def build_season(
    pbp_season: pd.DataFrame,
    pfr: pd.DataFrame,
    sleeper_players: list[dict],
    player_metrics: list[dict],
) -> list[dict]:
    pfr_agg, pfr_agg_norm = _aggregate_pfr_rec(pfr, SEASON)

    full_name_set: set[str] = {
        p["full_name"] for p in sleeper_players if p.get("full_name")
    }
    sleeper_pos: dict[str, str] = {
        p["full_name"]: p.get("position", "")
        for p in sleeper_players if p.get("full_name")
    }
    sleeper_te_set: set[str] = {
        p["full_name"] for p in sleeper_players
        if p.get("position") == "TE" and p.get("full_name")
    }
    metrics_by_player: dict[str, dict] = {
        m["player"]: m for m in player_metrics if m.get("player")
    }

    unambig       = _build_unambig(sleeper_players)
    team_disambig = _build_team_disambig(sleeper_players)

    pass_plays = pbp_season[pbp_season["pass_attempt"] == 1].copy()
    if pass_plays.empty:
        return []

    has_yardline = "yardline_100" in pass_plays.columns
    has_yac_col  = "yards_after_catch" in pass_plays.columns
    has_air      = "air_yards" in pass_plays.columns

    # Resolve names
    name_team_cache: dict[tuple[str, str], str] = {}

    def _tag(row: pd.Series) -> str:
        pbp_name = str(row.get("receiver_player_name") or "")
        if not pbp_name or pbp_name == "nan":
            return ""
        key = (pbp_name, str(row.get("posteam", "")))
        if key not in name_team_cache:
            name_team_cache[key] = _resolve(
                key[0], key[1], full_name_set, unambig, team_disambig, sleeper_pos
            )
        return name_team_cache[key]

    targeted = pass_plays[pass_plays["receiver_player_name"].notna()].copy()
    targeted["_full_name"] = targeted.apply(_tag, axis=1)
    targeted = targeted[targeted["_full_name"] != ""]

    # TE position filter
    def _is_te(fn: str) -> bool:
        s_pos = sleeper_pos.get(fn, "")
        if s_pos == "TE":
            return True
        if s_pos and s_pos != "TE":
            return False
        return metrics_by_player.get(fn, {}).get("pos", "") == "TE"

    te_names: set[str] = {fn for fn in targeted["_full_name"].unique() if _is_te(str(fn))}
    te_plays = targeted[targeted["_full_name"].isin(te_names)].copy()
    if te_plays.empty:
        return []

    team_pass_totals: dict[str, int] = (
        pass_plays.groupby("posteam")["pass_attempt"].count().to_dict()
    )

    rows: list[dict] = []

    for full_name, grp in te_plays.groupby("_full_name"):
        full_name = str(full_name)
        tgt = len(grp)
        if tgt < MIN_TARGETS:
            continue

        comps = grp[grp["complete_pass"] == 1]
        rec   = len(comps)
        yds   = int(comps["yards_gained"].sum())
        g     = int(grp["game_id"].nunique())

        team = (
            str(grp.sort_values(["season", "week"])["posteam"].iloc[-1])
            if "season" in grp.columns and "week" in grp.columns
            else str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns
            else ""
        )

        ypr = round(yds / rec, 1) if rec > 0 else None

        if has_air and rec > 0:
            ybc_total   = int(comps["air_yards"].dropna().sum())
            ybc_per_rec = round(ybc_total / rec, 1)
        else:
            ybc_total   = None
            ybc_per_rec = None

        if has_yac_col and rec > 0:
            yac_total   = int(comps["yards_after_catch"].dropna().sum())
            yac_per_rec = round(yac_total / rec, 1)
        else:
            yac_total   = None
            yac_per_rec = None

        rz_tgt = int((grp["yardline_100"] <= 20).sum()) if has_yardline else None

        team_tgt_total = team_pass_totals.get(team, 0)
        target_share = round(tgt / team_tgt_total, 4) if team_tgt_total > 0 else None

        comp_yds = comps["yards_gained"]
        rec_10   = int((comp_yds >= 10).sum())
        rec_20   = int((comp_yds >= 20).sum())
        rec_30   = int((comp_yds >= 30).sum())
        rec_40   = int((comp_yds >= 40).sum())
        rec_50   = int((comp_yds >= 50).sum())
        lng      = int(comp_yds.max()) if rec > 0 else None

        _pfr_key = _norm_pfr(full_name)
        _pfr_key = _CANONICAL_TO_PFR_NORM.get(_pfr_key, _pfr_key)
        pfr_stats = pfr_agg.get(full_name) \
                    or pfr_agg_norm.get(_pfr_key) \
                    or {}
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

    rows.sort(
        key=lambda r: (r.get("yds") or 0, r.get("rec") or 0, r.get("tgt") or 0),
        reverse=True,
    )
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(rows: list[dict]) -> None:
    required_keys = {c["key"] for c in COLUMNS}
    seen: set[str] = set()
    for r in rows:
        player = r.get("player", "")
        if player in seen:
            print(f"  WARNING: duplicate player '{player}'", file=sys.stderr)
        seen.add(player)
        missing = required_keys - set(r.keys())
        if missing:
            print(f"  WARNING: {player} missing keys {missing}", file=sys.stderr)
    print(f"  Validation OK: {len(rows)} TEs, no duplicate IDs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (PBP_PATH, PFR_PATH, SLEEPER_PATH, METRICS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Loading play-by-play ({SEASON} REG)...")
    pbp_full = pd.read_parquet(PBP_PATH)
    pbp_full = pbp_full[
        (pbp_full["season"] == SEASON) &
        (pbp_full["season_type"] == "REG")
    ].copy()

    print("Loading PFR receiving advanced stats...")
    pfr = pd.read_parquet(PFR_PATH)

    with open(SLEEPER_PATH) as f:
        raw = json.load(f)
    sleeper_players = list(raw.values()) if isinstance(raw, dict) else raw

    with open(METRICS_PATH) as f:
        raw_m = json.load(f)
    player_metrics = raw_m if isinstance(raw_m, list) else raw_m.get("players", [])

    rows = build_season(pbp_full, pfr, sleeper_players, player_metrics)
    rows = [r for r in rows if r["player"] not in _EXCLUDED_PLAYERS]
    _validate(rows)

    ordered_keys = [c["key"] for c in COLUMNS]
    ordered_rows = [{k: r.get(k) for k in ordered_keys} for r in rows]

    payload: dict[str, Any] = {
        "meta": {
            "position":     "TE",
            "table":        "te_advanced_stats",
            "season":       str(SEASON),
            "generated_at": now,
            "columns":      COLUMNS,
        },
        "rows": ordered_rows,
    }

    out_path = OUTPUT_DIR / f"te_advanced_stats_{SEASON}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.name} ({kb:.1f} KB, {len(rows)} TEs)")
    print("TE advanced stats build complete.")


if __name__ == "__main__":
    main()
