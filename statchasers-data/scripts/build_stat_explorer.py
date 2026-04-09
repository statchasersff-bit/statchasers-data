"""
build_stat_explorer.py
──────────────────────
Builds the Stat Explorer dataset for the StatChasers frontend.

Stat Explorer is the *raw-stat layer* — transparent counting stats and
near-raw rates with no modeled labels, career-arc classifications, or
interpretation fields.  Advanced Metrics is the intelligence layer.
These two outputs must remain strictly separate.

Positions supported: QB, RB, WR, TE.
All four share the same response envelope; only the field maps differ.

Data sources (all pre-pulled by pull_nflverse_data.py):
  data/raw/nflverse_play_by_play.parquet   — raw PBP counting stats
  data/raw/pfr_pass_advstats.parquet       — QB pressure / pocket (PFR)
  data/raw/pfr_rush_advstats.parquet       — RB broken tackles / YAC (PFR)
  data/raw/pfr_rec_advstats.parquet        — WR/TE drops / broken tackles (PFR)
  data/raw/sleeper_players.json            — canonical full-name lookup
  data/processed/player_metrics.json       — team / position context

Outputs (per-season):
  output/stat_explorer_qb_2023.json
  output/stat_explorer_qb_2024.json
  output/stat_explorer_qb_2025.json
  output/stat_explorer_rb_{year}.json
  output/stat_explorer_wr_{year}.json
  output/stat_explorer_te_{year}.json

Outputs (rolling multi-season):
  output/stat_explorer_qb.json
  output/stat_explorer_rb.json
  output/stat_explorer_wr.json
  output/stat_explorer_te.json

Union file:
  output/stat_explorer_latest.json  — { seasons: {2023: {QB,RB,WR,TE}, ...}, rolling: {QB,RB,WR,TE} }
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

ROOT             = Path(__file__).resolve().parent.parent
PBP_PATH         = ROOT / "data" / "raw" / "nflverse_play_by_play.parquet"
PFR_PASS_PATH    = ROOT / "data" / "raw" / "pfr_pass_advstats.parquet"
PFR_RUSH_PATH    = ROOT / "data" / "raw" / "pfr_rush_advstats.parquet"
PFR_REC_PATH     = ROOT / "data" / "raw" / "pfr_rec_advstats.parquet"
SLEEPER_PATH     = ROOT / "data" / "raw" / "sleeper_players.json"
METRICS_PATH     = ROOT / "data" / "processed" / "player_metrics.json"
OUTPUT_DIR       = ROOT / "output"

PIPELINE_YEAR    = 2025
SEASONS          = [2023, 2024, 2025]
MIN_QB_ATT       = 10
MIN_RB_CARRIES   = 15
MIN_WR_TGT       = 15
MIN_TE_TGT       = 10


# ---------------------------------------------------------------------------
# NFL passer rating (official formula)
# ---------------------------------------------------------------------------

def _passer_rating(comp: int, att: int, yds: int, td: int, ints: int) -> float | None:
    if att <= 0:
        return None
    a = min(max((comp / att - 0.3) / 0.2,    0.0), 2.375)
    b = min(max((yds  / att - 3.0) / 4.0,    0.0), 2.375)
    c = min(max((td   / att) / 0.05,          0.0), 2.375)
    d = min(max(2.375 - (ints / att) / 0.095, 0.0), 2.375)
    return round((a + b + c + d) / 6.0 * 100.0, 1)


# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

class FieldDef(NamedTuple):
    key:             str
    label:           str
    type:            str   # string | number | decimal | percent
    group:           str   # games | passing | deep | misc | rushing | pressure | contact | receiving | air | explosive | redZone
    default_visible: bool
    description:     str


# ── QB ──────────────────────────────────────────────────────────────────────
QB_FIELDS: list[FieldDef] = [
    # identity
    FieldDef("player",            "PLAYER",   "string",  "games",    True,  "Player full name"),
    FieldDef("team",              "TEAM",     "string",  "games",    True,  "Current team"),
    FieldDef("position",          "POS",      "string",  "games",    False, "Position"),
    FieldDef("gp",                "GP",       "number",  "games",    True,  "Games played"),
    # passing
    FieldDef("comp",              "COMP",     "number",  "passing",  True,  "Completions"),
    FieldDef("att",               "ATT",      "number",  "passing",  True,  "Pass attempts (excludes sacks)"),
    FieldDef("pct",               "PCT",      "percent", "passing",  True,  "Completion percentage"),
    FieldDef("yds",               "YDS",      "number",  "passing",  True,  "Passing yards"),
    FieldDef("ypa",               "Y/A",      "decimal", "passing",  True,  "Yards per attempt"),
    FieldDef("td",                "TD",       "number",  "passing",  True,  "Passing touchdowns"),
    FieldDef("int",               "INT",      "number",  "passing",  True,  "Interceptions"),
    FieldDef("air_yards",         "AIR YDS",  "number",  "passing",  False, "Total air yards on all attempts"),
    FieldDef("air_yards_per_att", "AIR/A",    "decimal", "passing",  False, "Air yards per attempt"),
    # deep ball
    FieldDef("pass_10_plus",      "10+",      "number",  "deep",     False, "Completions of 10+ yards"),
    FieldDef("pass_20_plus",      "20+",      "number",  "deep",     False, "Completions of 20+ yards"),
    FieldDef("pass_30_plus",      "30+",      "number",  "deep",     False, "Completions of 30+ yards"),
    FieldDef("pass_40_plus",      "40+",      "number",  "deep",     False, "Completions of 40+ yards"),
    FieldDef("pass_50_plus",      "50+",      "number",  "deep",     False, "Completions of 50+ yards"),
    # misc
    FieldDef("sacks",             "SACK",     "number",  "misc",     True,  "Times sacked"),
    FieldDef("passer_rating",     "RTG",      "decimal", "misc",     True,  "NFL passer rating"),
    FieldDef("rz_att",            "RZ ATT",   "number",  "misc",     False, "Pass attempts from inside the 20"),
    # rushing
    FieldDef("rush_att",          "RUSH ATT", "number",  "rushing",  False, "Rushing attempts"),
    FieldDef("rush_yds",          "RUSH YDS", "number",  "rushing",  False, "Rushing yards"),
    FieldDef("rush_td",           "RUSH TD",  "number",  "rushing",  False, "Rushing touchdowns"),
    # pressure (PFR)
    FieldDef("blitzes",           "BLITZ",    "number",  "pressure", False, "Times blitzed (PFR)"),
    FieldDef("hurries",           "HURRY",    "number",  "pressure", False, "Times hurried (PFR)"),
    FieldDef("knockdowns",        "HITS",     "number",  "pressure", False, "Times hit / knocked down (PFR)"),
    FieldDef("pressures",         "PRESS",    "number",  "pressure", False, "Total times pressured (PFR)"),
    FieldDef("poor_throws",       "BAD",      "number",  "pressure", False, "Bad / poor throws (PFR)"),
    FieldDef("drops",             "DROP",     "number",  "pressure", False, "Dropped passes by receivers (PFR)"),
    FieldDef("pocket_time",       "PKT T",    "decimal", "pressure", False, "Average time in pocket — NGS data (not yet available)"),
]

# ── RB ──────────────────────────────────────────────────────────────────────
RB_FIELDS: list[FieldDef] = [
    FieldDef("player",            "PLAYER",  "string",  "core",      True,  "Player full name"),
    FieldDef("team",              "TEAM",    "string",  "core",      True,  "Current team"),
    FieldDef("position",          "POS",     "string",  "core",      False, "Position"),
    FieldDef("gp",                "GP",      "number",  "core",      True,  "Games played"),
    FieldDef("carries",           "CAR",     "number",  "core",      True,  "Rush attempts"),
    FieldDef("yds",               "YDS",     "number",  "core",      True,  "Rushing yards"),
    FieldDef("ypc",               "YPC",     "decimal", "core",      True,  "Yards per carry"),
    FieldDef("td",                "TD",      "number",  "core",      True,  "Rushing touchdowns"),
    FieldDef("long",              "LNG",     "number",  "core",      False, "Longest rush"),
    FieldDef("explosive_runs",    "20+",     "number",  "explosive", False, "Runs of 15+ yards"),
    FieldDef("rz_carries",        "RZ CAR",  "number",  "redZone",   False, "Carries inside the 20"),
    FieldDef("broken_tackles",     "BRK TKL", "number",  "contact",   False, "Rushing broken tackles (PFR)"),
    FieldDef("yds_before_contact", "YBC",     "number",  "contact",   False, "Rushing yards before contact (PFR)"),
    FieldDef("ybc_per_carry",      "YBC/C",   "decimal", "contact",   False, "Yards before contact per carry (PFR)"),
    FieldDef("yds_after_contact",  "YAC",     "number",  "contact",   False, "Rushing yards after contact (PFR)"),
    FieldDef("yac_per_carry",      "YAC/C",   "decimal", "contact",   False, "Yards after contact per carry (PFR)"),
    FieldDef("tgt",               "TGT",     "number",  "receiving", False, "Receiving targets"),
    FieldDef("rec",               "REC",     "number",  "receiving", False, "Receptions"),
    FieldDef("rec_yds",           "R YDS",   "number",  "receiving", False, "Receiving yards"),
    FieldDef("ypr",               "YPR",     "decimal", "receiving", False, "Yards per reception"),
    FieldDef("rec_td",            "R TD",    "number",  "receiving", False, "Receiving touchdowns"),
    FieldDef("rz_tgt",            "RZ TGT",  "number",  "redZone",   False, "Receiving targets inside the 20"),
]

# ── WR ──────────────────────────────────────────────────────────────────────
WR_FIELDS: list[FieldDef] = [
    FieldDef("player",            "PLAYER",  "string",  "core",      True,  "Player full name"),
    FieldDef("team",              "TEAM",    "string",  "core",      True,  "Current team"),
    FieldDef("position",          "POS",     "string",  "core",      False, "Position"),
    FieldDef("gp",                "GP",      "number",  "core",      True,  "Games played"),
    FieldDef("tgt",               "TGT",     "number",  "core",      True,  "Targets"),
    FieldDef("rec",               "REC",     "number",  "core",      True,  "Receptions"),
    FieldDef("rec_yds",           "YDS",     "number",  "core",      True,  "Receiving yards"),
    FieldDef("ypr",               "YPR",     "decimal", "core",      True,  "Yards per reception"),
    FieldDef("td",                "TD",      "number",  "core",      True,  "Receiving touchdowns"),
    FieldDef("long",              "LNG",     "number",  "core",      False, "Longest reception"),
    FieldDef("air_yards",         "AIR YDS", "number",  "air",       False, "Total air yards on all targets"),
    FieldDef("air_per_tgt",       "AIR/T",   "decimal", "air",       False, "Air yards per target"),
    FieldDef("yac",               "YAC",     "number",  "air",       False, "Yards after catch"),
    FieldDef("yac_per_rec",       "YAC/R",   "decimal", "air",       False, "Yards after catch per reception"),
    FieldDef("explosive_recs",    "20+",     "number",  "explosive", False, "Receptions of 20+ yards"),
    FieldDef("rz_tgt",            "RZ TGT",  "number",  "redZone",   False, "Targets inside the 20"),
    FieldDef("rz_rec",            "RZ REC",  "number",  "redZone",   False, "Receptions inside the 20"),
    FieldDef("drops",             "DROP",    "number",  "contact",   False, "Dropped targets (PFR)"),
    FieldDef("broken_tackles",    "BRK TKL", "number",  "contact",   False, "Broken tackles after catch (PFR)"),
]

# ── TE ── same shape as WR, add inline note via description ─────────────────
TE_FIELDS: list[FieldDef] = [
    FieldDef("player",            "PLAYER",  "string",  "core",      True,  "Player full name"),
    FieldDef("team",              "TEAM",    "string",  "core",      True,  "Current team"),
    FieldDef("position",          "POS",     "string",  "core",      False, "Position"),
    FieldDef("gp",                "GP",      "number",  "core",      True,  "Games played"),
    FieldDef("tgt",               "TGT",     "number",  "core",      True,  "Targets"),
    FieldDef("rec",               "REC",     "number",  "core",      True,  "Receptions"),
    FieldDef("rec_yds",           "YDS",     "number",  "core",      True,  "Receiving yards"),
    FieldDef("ypr",               "YPR",     "decimal", "core",      True,  "Yards per reception"),
    FieldDef("td",                "TD",      "number",  "core",      True,  "Receiving touchdowns"),
    FieldDef("long",              "LNG",     "number",  "core",      False, "Longest reception"),
    FieldDef("air_yards",         "AIR YDS", "number",  "air",       False, "Total air yards on all targets"),
    FieldDef("air_per_tgt",       "AIR/T",   "decimal", "air",       False, "Air yards per target"),
    FieldDef("yac",               "YAC",     "number",  "air",       False, "Yards after catch"),
    FieldDef("yac_per_rec",       "YAC/R",   "decimal", "air",       False, "Yards after catch per reception"),
    FieldDef("explosive_recs",    "20+",     "number",  "explosive", False, "Receptions of 20+ yards"),
    FieldDef("rz_tgt",            "RZ TGT",  "number",  "redZone",   False, "Targets inside the 20"),
    FieldDef("rz_rec",            "RZ REC",  "number",  "redZone",   False, "Receptions inside the 20"),
    FieldDef("drops",             "DROP",    "number",  "contact",   False, "Dropped targets (PFR)"),
    FieldDef("broken_tackles",    "BRK TKL", "number",  "contact",   False, "Broken tackles after catch (PFR)"),
]

_FIELD_MAPS: dict[str, list[FieldDef]] = {
    "QB": QB_FIELDS,
    "RB": RB_FIELDS,
    "WR": WR_FIELDS,
    "TE": TE_FIELDS,
}


def get_field_map(position: str) -> list[FieldDef]:
    if position not in _FIELD_MAPS:
        raise ValueError(f"No field map for '{position}'. Available: {list(_FIELD_MAPS)}")
    return _FIELD_MAPS[position]


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def _nflverse_abbrev(full_name: str) -> str:
    """'Josh Allen' → 'J.Allen',  'Ja\'Marr Chase' → 'J.Chase'"""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _build_sleeper_abbrev_lookup(sleeper_players: list[dict]) -> dict[str, str]:
    """abbrev → full_name from Sleeper roster, dropping ambiguous abbreviations."""
    counts: dict[str, int]  = {}
    mapping: dict[str, str] = {}
    for p in sleeper_players:
        full = p.get("full_name", "").strip()
        if not full:
            continue
        ab = _nflverse_abbrev(full)
        counts[ab]  = counts.get(ab, 0) + 1
        mapping[ab] = full
    return {ab: fn for ab, fn in mapping.items() if counts[ab] == 1}


def _build_pfr_abbrev_lookup(pfr_df: pd.DataFrame | None) -> dict[str, str]:
    """
    Build supplementary abbrev → full_name from PFR player names.
    PFR uses full canonical names (e.g. "Aaron Rodgers") and is pulled
    per-position, so abbreviation collisions are rare within a position.
    """
    if pfr_df is None or pfr_df.empty:
        return {}
    counts: dict[str, int]  = {}
    mapping: dict[str, str] = {}
    col = "pfr_player_name"
    if col not in pfr_df.columns:
        return {}
    for name in pfr_df[col].dropna().unique():
        name = str(name)
        ab = _nflverse_abbrev(name)
        counts[ab]  = counts.get(ab, 0) + 1
        mapping[ab] = name
    return {ab: fn for ab, fn in mapping.items() if counts[ab] == 1}


# Generational suffixes PFR appends that Sleeper / PBP canonical names omit.
_PFR_SUFFIXES = frozenset(["jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"])


def _norm_pfr(name: str) -> str:
    """Normalise a player name for PFR → canonical matching.

    Strips punctuation (dots, hyphens, apostrophes), lowercases, and drops
    trailing generational suffixes so that:
      'Kenneth Walker III' → 'kennethwalker'
      'Tyrone Tracy Jr.'   → 'tyronetracy'
      'Brian Robinson Jr.' → 'brianrobinson'
      'Ty.Johnson'         → 'tyjohnson'  (PBP 2-char dot prefix → PFR full name)
    """
    clean = re.sub(r"[.\-']", " ", name).lower()
    parts = clean.split()
    while parts and parts[-1] in _PFR_SUFFIXES:
        parts.pop()
    return "".join(parts)


_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    "T.Etienne": {"JAX": "Travis Etienne", "CAR": "Trevor Etienne"},
    "B.Robinson": {"ATL": "Bijan Robinson", "SF": "Brian Robinson", "WAS": "Brian Robinson"},
    # J.Williams: multiple players share this abbreviation; map by team
    "J.Williams": {"DEN": "Javonte Williams", "DAL": "Javonte Williams", "NO": "Jamaal Williams"},
    # "K.Walker" is ambiguous in Sleeper (multiple players) so the unambiguous
    # abbrev lookup misses it and the name stays abbreviated.
    "K.Walker":  {"SEA": "Kenneth Walker"},
    # K.Williams: ambiguous (Kyren, Kyle, Ke'Shawn Williams etc.) — map by team
    "K.Williams": {"LA": "Kyren Williams", "LAR": "Kyren Williams",
                   "NE": "Kyle Williams", "CIN": "Ke'Shawn Williams"},
    # J.Wright: ambiguous — Jaylen Wright (MIA) vs others
    "J.Wright":  {"MIA": "Jaylen Wright"},
    # K.Johnson: ambiguous — Kaleb Johnson (PIT) vs others
    "K.Johnson": {"PIT": "Kaleb Johnson"},
    # D.Moore: DJ Moore (CHI) vs David Moore (CAR/TB) etc.
    "D.Moore":   {"CHI": "DJ Moore", "BUF": "DJ Moore",
                  "CAR": "David Moore", "TB":  "David Moore"},
    # J.Ford: Jacoby Ford (WR, CLE receiver) vs Jerome Ford (RB, CLE rusher)
    # In RB stat explorer context (rushing plays), J.Ford+CLE = Jerome Ford
    "J.Ford":    {"CLE": "Jerome Ford"},
    # M.Carter: Malachi Carter (WR, ARI receiver) vs Michael Carter (RB, ARI rusher)
    # In RB stat explorer context (rushing plays), M.Carter+ARI = Michael Carter
    "M.Carter":  {"ARI": "Michael Carter"},
    # Co.Heyward: 2-char first-name prefix for Connor Heyward (PIT)
    "Co.Heyward": {"PIT": "Connor Heyward"},
    # Ji.Horn: 2-char prefix for Jimmy Horn (CAR)
    "Ji.Horn":   {"CAR": "Jimmy Horn"},
}


def _resolve_name(
    pbp_name: str,
    full_name_set: set[str],
    combined_lookup: dict[str, str],
    pbp_team: str = "",
) -> str:
    """
    Resolve a PBP abbreviated name to a canonical full name.
    Priority: manual team override → direct Sleeper full-name match → combined Sleeper+PFR abbrev lookup.
    Falls back to the PBP name unchanged (should be rare with PFR supplement).
    """
    if pbp_name in _MANUAL_TEAM_OVERRIDES and pbp_team:
        hit = _MANUAL_TEAM_OVERRIDES[pbp_name].get(pbp_team)
        if hit:
            return hit
    if pbp_name in full_name_set:
        return pbp_name
    return combined_lookup.get(pbp_name, pbp_name)


# ---------------------------------------------------------------------------
# Coverage and column builders
# ---------------------------------------------------------------------------

def compute_field_coverage(rows: list[dict], keys: list[str]) -> dict[str, float]:
    """Fraction of rows (0–1) with a non-null, non-empty value for each key."""
    n = len(rows)
    if n == 0:
        return {k: 0.0 for k in keys}
    return {
        key: round(
            sum(1 for r in rows if r.get(key) is not None and r.get(key) != "") / n,
            3,
        )
        for key in keys
    }


def build_stat_explorer_columns(
    position: str,
    coverage: dict[str, float],
) -> list[dict]:
    """
    Emit column metadata.  Fields with < 1 % coverage are omitted so the
    frontend never renders a dead column (e.g. pocket_time until NGS data
    is available, or any future PFR field added before data arrives).
    """
    columns = []
    for f in get_field_map(position):
        cov = coverage.get(f.key, 0.0)
        if f.type != "string" and cov < 0.01:
            continue
        columns.append({
            "key":            f.key,
            "label":          f.label,
            "type":           f.type,
            "group":          f.group,
            "defaultVisible": f.default_visible,
            "description":    f.description,
            "coverage":       cov,
        })
    return columns


# ---------------------------------------------------------------------------
# PFR aggregation helpers
# ---------------------------------------------------------------------------

def _load_pfr(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"WARNING: Could not load {path.name}: {e}", file=sys.stderr)
        return None


def _filter_pfr_by_season(pfr: pd.DataFrame | None, season: int) -> pd.DataFrame | None:
    """Return rows for a single season, or the full frame if no season column exists."""
    if pfr is None or pfr.empty:
        return pfr
    if "season" not in pfr.columns:
        return pfr
    filtered = pfr[pfr["season"] == season].reset_index(drop=True)
    return filtered if not filtered.empty else None


def _aggregate_pfr_pass(pfr: pd.DataFrame | None) -> dict[str, dict]:
    """
    Group PFR pass advstats by player.  Returns:
      { full_name: { blitzes, hurries, knockdowns, pressures, poor_throws, drops } }
    """
    if pfr is None or pfr.empty:
        return {}
    cols = {
        "blitzes":    "times_blitzed",
        "hurries":    "times_hurried",
        "knockdowns": "times_hit",
        "pressures":  "times_pressured",
        "poor_throws":"passing_bad_throws",
        "drops":      "passing_drops",
    }
    agg_spec = {
        out_key: (src_col, "sum")
        for out_key, src_col in cols.items()
        if src_col in pfr.columns
    }
    if not agg_spec:
        return {}
    result = pfr.groupby("pfr_player_name").agg(**agg_spec).reset_index()
    return {
        row["pfr_player_name"]: {k: int(row[k]) for k in agg_spec}
        for _, row in result.iterrows()
    }


def _aggregate_pfr_rush(pfr: pd.DataFrame | None) -> dict[str, dict]:
    """
    Group PFR rush advstats by player.  Returns:
      { full_name: { broken_tackles, yds_after_contact } }
    """
    if pfr is None or pfr.empty:
        return {}
    cols = {
        "broken_tackles":     "rushing_broken_tackles",
        "yds_before_contact": "rushing_yards_before_contact",
        "yds_after_contact":  "rushing_yards_after_contact",
    }
    agg_spec = {
        out_key: (src_col, "sum")
        for out_key, src_col in cols.items()
        if src_col in pfr.columns
    }
    if not agg_spec:
        return {}
    result = pfr.groupby("pfr_player_name").agg(**agg_spec).reset_index()
    return {
        row["pfr_player_name"]: {k: float(row[k]) for k in agg_spec}
        for _, row in result.iterrows()
    }


def _aggregate_pfr_rec(pfr: pd.DataFrame | None) -> dict[str, dict]:
    """
    Group PFR rec advstats by player.  Returns:
      { full_name: { drops, broken_tackles } }
    """
    if pfr is None or pfr.empty:
        return {}
    cols = {
        "drops":          "receiving_drop",
        "broken_tackles": "receiving_broken_tackles",
    }
    agg_spec = {
        out_key: (src_col, "sum")
        for out_key, src_col in cols.items()
        if src_col in pfr.columns
    }
    if not agg_spec:
        return {}
    result = pfr.groupby("pfr_player_name").agg(**agg_spec).reset_index()
    return {
        row["pfr_player_name"]: {k: int(row[k]) for k in agg_spec}
        for _, row in result.iterrows()
    }


# ---------------------------------------------------------------------------
# PBP stat computation
# ---------------------------------------------------------------------------

def compute_qb_raw_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling multi-season QB passing AND rushing stats.
    Sacks excluded from pass attempt counts to match official NFL/ESPN box scores.
    QB rush stats are drawn from rush_attempt plays where rusher_player_name
    matches the passer_player_name (same abbreviated-name convention in PBP).
    INT requires the 'interception' PBP column; treated as 0 if absent.
    """
    all_dropbacks = pbp[pbp["pass_attempt"] == 1].copy()
    if all_dropbacks.empty:
        return pd.DataFrame()

    has_sack     = "sack"         in all_dropbacks.columns
    has_int      = "interception" in all_dropbacks.columns
    has_yardline = "yardline_100" in all_dropbacks.columns

    official = (
        all_dropbacks[all_dropbacks["sack"] != 1].copy()
        if has_sack else all_dropbacks.copy()
    )
    completions = official[official["complete_pass"] == 1].copy()

    # Pre-aggregate QB rushing from the full PBP (rush_attempt plays).
    # Matches on the same abbreviated name used for passer_player_name.
    rush_plays = pbp[pbp["rush_attempt"] == 1]
    qb_rush_agg: dict[str, dict] = {}
    if not rush_plays.empty and "rusher_player_name" in rush_plays.columns:
        qb_rush_agg = (
            rush_plays.groupby("rusher_player_name")
            .agg(
                rush_att=("rush_attempt", "sum"),
                rush_yds=("yards_gained",  "sum"),
                rush_td= ("touchdown",     "sum"),
            )
            .to_dict("index")
        )

    rows = []
    for name, grp in official.groupby("passer_player_name"):
        att  = len(grp)
        comp = int(grp["complete_pass"].sum())
        yds  = int(grp["yards_gained"].sum())
        td   = int(grp["touchdown"].sum())
        ints = int(grp["interception"].sum()) if has_int else 0

        db_grp = all_dropbacks[all_dropbacks["passer_player_name"] == name]
        sacks  = int(db_grp["sack"].sum()) if has_sack else 0
        gp     = int(db_grp["game_id"].nunique())

        air_yds = round(float(grp["air_yards"].dropna().sum()), 1)
        rz_att  = int((grp["yardline_100"].dropna() <= 20).sum()) if has_yardline else None

        comp_yds = completions.loc[
            completions["passer_player_name"] == name, "yards_gained"
        ]
        team = str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns else ""

        pct         = round(comp / att * 100, 1) if att > 0 else 0.0
        ypa         = round(yds  / att,        2) if att > 0 else 0.0
        air_per_att = round(air_yds / att,     2) if att > 0 else 0.0

        # QB rushing stats
        rd       = qb_rush_agg.get(name, {})
        rush_att = int(rd.get("rush_att", 0))
        rush_yds = int(rd.get("rush_yds", 0))
        rush_td  = int(rd.get("rush_td",  0))

        rows.append({
            "player_name":       name,
            "team_pbp":          team,
            "gp":                gp,
            "comp":              comp,
            "att":               att,
            "pct":               pct,
            "yds":               yds,
            "ypa":               ypa,
            "td":                td,
            "int":               ints,
            "sacks":             sacks,
            "air_yards":         air_yds,
            "air_yards_per_att": air_per_att,
            "pass_10_plus":      int((comp_yds >= 10).sum()),
            "pass_20_plus":      int((comp_yds >= 20).sum()),
            "pass_30_plus":      int((comp_yds >= 30).sum()),
            "pass_40_plus":      int((comp_yds >= 40).sum()),
            "pass_50_plus":      int((comp_yds >= 50).sum()),
            "rz_att":            rz_att,
            "passer_rating":     _passer_rating(comp, att, yds, td, ints),
            "rush_att":          rush_att,
            "rush_yds":          rush_yds,
            "rush_td":           rush_td,
        })

    return pd.DataFrame(rows)


def compute_rushing_raw_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling multi-season rushing stats for all rushers.
    Receiving stats for the same player are joined in the dataset builder.
    """
    rush = pbp[pbp["rush_attempt"] == 1].copy()
    if rush.empty:
        return pd.DataFrame()

    has_yardline = "yardline_100" in rush.columns

    rows = []
    # Group by (name, team) so players who share an abbreviated name but play
    # for different teams (e.g. B.Robinson on ATL vs SF) get separate rows.
    for (name, team_pbp), grp in rush.groupby(["rusher_player_name", "posteam"]):
        carries = len(grp)
        yds     = int(grp["yards_gained"].sum())
        td      = int(grp["touchdown"].sum())
        gp      = int(grp["game_id"].nunique())
        long_r  = int(grp["yards_gained"].max())
        expl    = int((grp["yards_gained"] >= 15).sum())
        rz_car  = int((grp["yardline_100"] <= 20).sum()) if has_yardline else None

        rows.append({
            "player_name": name,
            "team_pbp":    str(team_pbp),
            "gp":          gp,
            "carries":     carries,
            "yds":         yds,
            "ypc":         round(yds / carries, 2) if carries > 0 else 0.0,
            "td":          td,
            "long":        long_r,
            "explosive_runs": expl,
            "rz_carries":  rz_car,
        })

    return pd.DataFrame(rows)


def compute_receiving_raw_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling multi-season receiving stats for all receivers.
    Position filtering happens in the dataset builder.
    """
    pass_plays = pbp[pbp["pass_attempt"] == 1].copy()
    if "sack" in pass_plays.columns:
        pass_plays = pass_plays[pass_plays["sack"] != 1]

    if pass_plays.empty:
        return pd.DataFrame()

    has_yardline = "yardline_100" in pass_plays.columns
    has_yac      = "yards_after_catch" in pass_plays.columns

    rows = []
    # Group by (name, team) so players who share an abbreviated name but play
    # for different teams get separate rows.
    for (name, team_pbp), grp in pass_plays.groupby(["receiver_player_name", "posteam"]):
        tgt     = len(grp)
        rec     = int(grp["complete_pass"].sum())
        rec_yds = int(grp["yards_gained"].sum())
        td      = int(grp["touchdown"].sum())
        gp      = int(grp["game_id"].nunique())
        air_yds = round(float(grp["air_yards"].dropna().sum()), 1)
        long_r  = int(grp.loc[grp["complete_pass"] == 1, "yards_gained"].max()
                      if rec > 0 else 0)
        expl    = int((grp.loc[grp["complete_pass"] == 1, "yards_gained"] >= 20).sum())
        rz_tgt  = int((grp["yardline_100"] <= 20).sum()) if has_yardline else None
        rz_rec  = int((grp.loc[grp["complete_pass"] == 1, "yardline_100"] <= 20).sum()) if has_yardline else None
        yac     = round(float(grp["yards_after_catch"].dropna().sum()), 1) if has_yac else None

        rows.append({
            "player_name":  name,
            "team_pbp":     str(team_pbp),
            "gp":           gp,
            "tgt":          tgt,
            "rec":          rec,
            "rec_yds":      rec_yds,
            "ypr":          round(rec_yds / rec, 2) if rec > 0 else 0.0,
            "td":           td,
            "long":         long_r,
            "air_yards":    air_yds,
            "air_per_tgt":  round(air_yds / tgt, 2) if tgt > 0 else 0.0,
            "yac":          yac,
            "yac_per_rec":  round(yac / rec, 2) if (yac and rec > 0) else None,
            "explosive_recs": expl,
            "rz_tgt":       rz_tgt,
            "rz_rec":       rz_rec,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Row sanitizers — shape the final frontend row per position
# ---------------------------------------------------------------------------

def _sanitize_qb(raw: dict, full_name: str, team: str,
                  pfr_stats: dict) -> dict[str, Any]:
    return {
        "player":             full_name,
        "team":               team,
        "position":           "QB",
        "gp":                 raw.get("gp"),
        "comp":               raw.get("comp"),
        "att":                raw.get("att"),
        "pct":                raw.get("pct"),
        "yds":                raw.get("yds"),
        "ypa":                raw.get("ypa"),
        "td":                 raw.get("td"),
        "int":                raw.get("int"),
        "air_yards":          raw.get("air_yards"),
        "air_yards_per_att":  raw.get("air_yards_per_att"),
        "pass_10_plus":       raw.get("pass_10_plus"),
        "pass_20_plus":       raw.get("pass_20_plus"),
        "pass_30_plus":       raw.get("pass_30_plus"),
        "pass_40_plus":       raw.get("pass_40_plus"),
        "pass_50_plus":       raw.get("pass_50_plus"),
        "sacks":              raw.get("sacks"),
        "passer_rating":      raw.get("passer_rating"),
        "rz_att":             raw.get("rz_att"),
        "rush_att":           raw.get("rush_att", 0),
        "rush_yds":           raw.get("rush_yds", 0),
        "rush_td":            raw.get("rush_td",  0),
        "blitzes":            pfr_stats.get("blitzes"),
        "hurries":            pfr_stats.get("hurries"),
        "knockdowns":         pfr_stats.get("knockdowns"),
        "pressures":          pfr_stats.get("pressures"),
        "poor_throws":        pfr_stats.get("poor_throws"),
        "drops":              pfr_stats.get("drops"),
        "pocket_time":        None,   # NGS only — not yet available
    }


def _sanitize_rb(raw_rush: dict, raw_rec: dict | None,
                  full_name: str, team: str, pfr_stats: dict) -> dict[str, Any]:
    pfr_ybc  = pfr_stats.get("yds_before_contact")
    pfr_yac  = pfr_stats.get("yds_after_contact")
    carries  = raw_rush.get("carries") or 0
    return {
        "player":             full_name,
        "team":               team,
        "position":           "RB",
        "gp":                 raw_rush.get("gp"),
        "carries":            carries,
        "yds":                raw_rush.get("yds"),
        "ypc":                raw_rush.get("ypc"),
        "td":                 raw_rush.get("td"),
        "long":               raw_rush.get("long"),
        "explosive_runs":     raw_rush.get("explosive_runs"),
        "rz_carries":         raw_rush.get("rz_carries"),
        "broken_tackles":     pfr_stats.get("broken_tackles"),
        "yds_before_contact": round(pfr_ybc, 1) if pfr_ybc is not None else None,
        "ybc_per_carry":      round(pfr_ybc / carries, 2) if (pfr_ybc is not None and carries > 0) else None,
        "yds_after_contact":  round(pfr_yac, 1) if pfr_yac is not None else None,
        "yac_per_carry":      round(pfr_yac / carries, 2) if (pfr_yac is not None and carries > 0) else None,
        "tgt":               raw_rec.get("tgt") if raw_rec else None,
        "rec":               raw_rec.get("rec") if raw_rec else None,
        "rec_yds":           raw_rec.get("rec_yds") if raw_rec else None,
        "ypr":               raw_rec.get("ypr") if raw_rec else None,
        "rec_td":            raw_rec.get("td") if raw_rec else None,
        "rz_tgt":            raw_rec.get("rz_tgt") if raw_rec else None,
    }


def _sanitize_receiver(raw: dict, full_name: str, team: str,
                         position: str, pfr_stats: dict) -> dict[str, Any]:
    return {
        "player":          full_name,
        "team":            team,
        "position":        position,
        "gp":              raw.get("gp"),
        "tgt":             raw.get("tgt"),
        "rec":             raw.get("rec"),
        "rec_yds":         raw.get("rec_yds"),
        "ypr":             raw.get("ypr"),
        "td":              raw.get("td"),
        "long":            raw.get("long"),
        "air_yards":       raw.get("air_yards"),
        "air_per_tgt":     raw.get("air_per_tgt"),
        "yac":             raw.get("yac"),
        "yac_per_rec":     raw.get("yac_per_rec"),
        "explosive_recs":  raw.get("explosive_recs"),
        "rz_tgt":          raw.get("rz_tgt"),
        "rz_rec":          raw.get("rz_rec"),
        "drops":           pfr_stats.get("drops"),
        "broken_tackles":  pfr_stats.get("broken_tackles"),
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_stat_explorer_dataset(
    position: str,
    pbp: pd.DataFrame,
    sleeper_players: list[dict],
    player_metrics: list[dict],
    pfr_pass: pd.DataFrame | None = None,
    pfr_rush: pd.DataFrame | None = None,
    pfr_rec:  pd.DataFrame | None = None,
) -> dict:
    """
    Orchestrate raw-stat computation, name resolution, PFR merge, coverage
    checks, and return the full Stat Explorer response envelope.
    """
    # ── name resolution lookups ─────────────────────────────────────────────
    full_name_set   = {p.get("full_name", "") for p in sleeper_players}
    sleeper_lookup  = _build_sleeper_abbrev_lookup(sleeper_players)

    # PFR provides full names — supplement Sleeper to fix abbreviated QBs/RBs/WRs/TEs
    pfr_for_pos = {"QB": pfr_pass, "RB": pfr_rush, "WR": pfr_rec, "TE": pfr_rec}
    pfr_lookup  = _build_pfr_abbrev_lookup(pfr_for_pos.get(position))
    combined_lookup = {**sleeper_lookup, **pfr_lookup}  # PFR wins conflicts

    # ── analytics-pipeline context (team, confirmed position) ───────────────
    metrics_by_player = {p["player"]: p for p in player_metrics}

    # ── PFR aggregated stats per player (full names as keys) ────────────────
    if position == "QB":
        pfr_agg = _aggregate_pfr_pass(pfr_pass)
    elif position == "RB":
        pfr_agg = _aggregate_pfr_rush(pfr_rush)
    else:
        pfr_agg = _aggregate_pfr_rec(pfr_rec)
    # Normalised fallback: strips suffixes (Jr., III) and punctuation (Ty.Johnson)
    # so canonical Sleeper names match PFR names that include suffixes.
    pfr_agg_norm: dict[str, dict] = {_norm_pfr(k): v for k, v in pfr_agg.items()}

    # ── compute PBP raw stats ────────────────────────────────────────────────
    if position == "QB":
        raw_df   = compute_qb_raw_stats(pbp)
        min_thr  = MIN_QB_ATT
        thr_col  = "att"
    elif position == "RB":
        rush_df  = compute_rushing_raw_stats(pbp)
        rec_df   = compute_receiving_raw_stats(pbp)
        raw_df   = rush_df
        min_thr  = MIN_RB_CARRIES
        thr_col  = "carries"
    else:
        raw_df   = compute_receiving_raw_stats(pbp)
        min_thr  = MIN_WR_TGT if position == "WR" else MIN_TE_TGT
        thr_col  = "tgt"

    if raw_df.empty:
        return {}

    raw_df = raw_df[raw_df[thr_col] >= min_thr].reset_index(drop=True)

    # ── sample window metadata ───────────────────────────────────────────────
    seasons = sorted(pbp["season"].dropna().unique().astype(int).tolist())
    if len(seasons) >= 2:
        sample_label  = "Rolling Multi-Season"
        sample_window = f"{seasons[0]}\u2013{seasons[-1]}"
    elif seasons:
        sample_label  = f"{seasons[0]} Season"
        sample_window = str(seasons[0])
    else:
        sample_label  = "Unknown"
        sample_window = ""

    # ── build rows ───────────────────────────────────────────────────────────
    rows: list[dict] = []
    for _, raw in raw_df.iterrows():
        pbp_name = raw["player_name"]
        full_name = _resolve_name(pbp_name, full_name_set, combined_lookup, raw.get("team_pbp", ""))

        info  = metrics_by_player.get(full_name, {})
        pos   = info.get("pos", position)

        # Position filter — skip players whose Sleeper position doesn't match
        # (e.g., exclude QBs who appear in rushing stats)
        if pos and pos != position:
            continue

        # Prefer PBP posteam (seasonal ground truth) over player_metrics.team
        # which reflects current roster and is stale for historical seasons.
        team = raw.get("team_pbp", "") or info.get("team", "")

        pfr_stats = pfr_agg.get(full_name) \
                    or pfr_agg_norm.get(_norm_pfr(full_name)) \
                    or {}

        if position == "QB":
            row = _sanitize_qb(raw.to_dict(), full_name, team, pfr_stats)
        elif position == "RB":
            rec_row = (
                rec_df[rec_df["player_name"] == pbp_name]
                .iloc[0].to_dict()
                if not rec_df.empty and (rec_df["player_name"] == pbp_name).any()
                else None
            )
            row = _sanitize_rb(raw.to_dict(), rec_row, full_name, team, pfr_stats)
        else:
            row = _sanitize_receiver(raw.to_dict(), full_name, team, position, pfr_stats)

        rows.append(row)

    # ── sort + coverage + columns ────────────────────────────────────────────
    sort_key = {"QB": "att", "RB": "carries"}.get(position, "tgt")
    rows.sort(key=lambda r: r.get(sort_key) or 0, reverse=True)

    all_keys = [f.key for f in get_field_map(position)]
    coverage = compute_field_coverage(rows, all_keys)
    columns  = build_stat_explorer_columns(position, coverage)

    return {
        "position":     position,
        "sampleLabel":  sample_label,
        "sampleWindow": sample_window,
        "pipelineYear": PIPELINE_YEAR,
        "generatedAt":  datetime.now(timezone.utc).isoformat(),
        "playerCount":  len(rows),
        "columns":      columns,
        "rows":         rows,
    }


# ---------------------------------------------------------------------------
# Multi-season aggregation (one row per player, stats summed across seasons)
# ---------------------------------------------------------------------------

def _aggregate_rolling_rows(rows: list[dict], pos: str) -> list[dict]:
    """
    Collapse per-season rows (each tagged with a 'season' key) into one
    combined row per player, summing counting stats and recalculating rates.
    Identity fields (team, position) are taken from the most recent season.
    """
    from collections import defaultdict

    SUM_FIELDS: dict[str, list[str]] = {
        "QB": [
            "gp", "comp", "att", "yds", "td", "int", "air_yards",
            "pass_10_plus", "pass_20_plus", "pass_30_plus", "pass_40_plus", "pass_50_plus",
            "sacks", "rz_att", "rush_att", "rush_yds", "rush_td",
            "blitzes", "hurries", "knockdowns", "pressures", "poor_throws", "drops",
        ],
        "RB": [
            "gp", "carries", "yds", "td", "explosive_runs", "rz_carries",
            "broken_tackles", "yds_after_contact", "tgt", "rec", "rec_yds", "rec_td", "rz_tgt",
        ],
        "WR": [
            "gp", "tgt", "rec", "rec_yds", "td", "air_yards", "yac",
            "explosive_recs", "rz_tgt", "rz_rec", "drops", "broken_tackles",
        ],
        "TE": [
            "gp", "tgt", "rec", "rec_yds", "td", "air_yards", "yac",
            "explosive_recs", "rz_tgt", "rz_rec", "drops", "broken_tackles",
        ],
    }
    MAX_FIELDS: dict[str, list[str]] = {
        "QB": [],
        "RB": ["long"],
        "WR": ["long"],
        "TE": ["long"],
    }

    sum_fields = SUM_FIELDS.get(pos, [])
    max_fields = MAX_FIELDS.get(pos, [])

    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_player[r["player"]].append(r)

    result: list[dict] = []
    for player, player_rows in by_player.items():
        latest = max(player_rows, key=lambda r: r.get("season", 0))

        combined: dict = {
            "player":   player,
            "team":     latest.get("team"),
            "position": latest.get("position"),
        }

        for f in sum_fields:
            vals = [r[f] for r in player_rows if r.get(f) is not None]
            combined[f] = sum(vals) if vals else None

        for f in max_fields:
            vals = [r[f] for r in player_rows if r.get(f) is not None]
            combined[f] = max(vals) if vals else None

        if pos == "QB":
            att  = combined.get("att") or 0
            comp = combined.get("comp") or 0
            yds  = combined.get("yds") or 0
            td   = combined.get("td") or 0
            int_ = combined.get("int") or 0
            air  = combined.get("air_yards") or 0
            combined["pct"]              = round(comp / att * 100, 1) if att else None
            combined["ypa"]              = round(yds  / att, 2)       if att else None
            combined["air_yards_per_att"]= round(air  / att, 1)       if att else None
            if att:
                a = max(0.0, min(2.375, ((comp / att) - 0.3) * 5))
                b = max(0.0, min(2.375, ((yds  / att) - 3)   * 0.25))
                c = max(0.0, min(2.375, (td   / att)          * 20))
                d = max(0.0, min(2.375, 2.375 - (int_ / att)  * 25))
                combined["passer_rating"] = round((a + b + c + d) / 6 * 100, 1)
            else:
                combined["passer_rating"] = None
            pt_num = sum((r.get("pocket_time") or 0) * (r.get("att") or 0) for r in player_rows)
            combined["pocket_time"] = round(pt_num / att, 2) if att else None

        elif pos == "RB":
            carries  = combined.get("carries") or 0
            yds      = combined.get("yds") or 0
            yac      = combined.get("yds_after_contact") or 0
            rec      = combined.get("rec") or 0
            rec_yds  = combined.get("rec_yds") or 0
            combined["ypc"]         = round(yds     / carries, 2) if carries else None
            combined["yac_per_carry"]= round(yac     / carries, 2) if carries else None
            combined["ypr"]         = round(rec_yds / rec,     2) if rec     else None

        elif pos in ("WR", "TE"):
            tgt     = combined.get("tgt") or 0
            rec     = combined.get("rec") or 0
            rec_yds = combined.get("rec_yds") or 0
            air     = combined.get("air_yards") or 0
            yac     = combined.get("yac") or 0
            combined["ypr"]        = round(rec_yds / rec, 2) if rec else None
            combined["air_per_tgt"]= round(air / tgt, 1)    if tgt else None
            combined["yac_per_rec"]= round(yac / rec, 2)    if rec else None

        result.append(combined)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path, label in [
        (PBP_PATH,     "PBP parquet"),
        (SLEEPER_PATH, "Sleeper players"),
        (METRICS_PATH, "player_metrics.json"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    print("Loading PBP data...")
    pbp = pd.read_parquet(PBP_PATH)
    pbp = pbp[pbp["season_type"] == "REG"].copy()  # regular season only

    print("Loading Sleeper players...")
    with open(SLEEPER_PATH) as f:
        sleeper_players = json.load(f)

    print("Loading player metrics...")
    with open(METRICS_PATH) as f:
        player_metrics = json.load(f)

    # PFR — all optional; missing files trigger a warning, not a crash
    pfr_pass = _load_pfr(PFR_PASS_PATH)
    pfr_rush = _load_pfr(PFR_RUSH_PATH)
    pfr_rec  = _load_pfr(PFR_REC_PATH)
    if pfr_pass is None:
        print("WARNING: PFR pass advstats not found — pressure fields will be null.",
              file=sys.stderr)
    if pfr_rush is None:
        print("WARNING: PFR rush advstats not found — contact fields will be null.",
              file=sys.stderr)
    if pfr_rec is None:
        print("WARNING: PFR rec advstats not found — drops/BT fields will be null.",
              file=sys.stderr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pos_slugs = [("QB", "qb"), ("RB", "rb"), ("WR", "wr"), ("TE", "te")]

    all_by_season: dict[str, dict[str, dict]] = {}
    all_rolling:   dict[str, dict]            = {}

    # ── per-season files ─────────────────────────────────────────────────────
    available_seasons = sorted(pbp["season"].dropna().unique().astype(int).tolist())
    for season in available_seasons:
        pbp_s      = pbp[pbp["season"] == season].reset_index(drop=True)
        pfr_pass_s = _filter_pfr_by_season(pfr_pass, season)
        pfr_rush_s = _filter_pfr_by_season(pfr_rush, season)
        pfr_rec_s  = _filter_pfr_by_season(pfr_rec,  season)

        if pbp_s.empty:
            print(f"WARNING: No PBP data for {season} — skipping.", file=sys.stderr)
            continue

        print(f"\nBuilding {season} season...")
        all_by_season[str(season)] = {}

        for pos, slug in pos_slugs:
            dataset = build_stat_explorer_dataset(
                pos, pbp_s, sleeper_players, player_metrics,
                pfr_pass=pfr_pass_s, pfr_rush=pfr_rush_s, pfr_rec=pfr_rec_s,
            )
            if not dataset:
                print(f"  WARNING: No {pos} data for {season}.")
                continue

            filename = f"stat_explorer_{slug}_{season}.json"
            out_path = OUTPUT_DIR / filename
            with open(out_path, "w") as fh:
                json.dump(dataset, fh, separators=(",", ":"))
            kb = out_path.stat().st_size / 1024
            print(f"  Wrote {filename}  ({kb:.1f} KB, {dataset['playerCount']} players)")
            all_by_season[str(season)][pos] = dataset

    # ── rolling multi-season files ────────────────────────────────────────────
    # Collect per-season rows then aggregate into ONE row per player (counting
    # stats summed, rate stats recalculated from the combined totals).
    print("\nBuilding rolling multi-season data...")
    for pos, slug in pos_slugs:
        season_rows: list[dict] = []
        for season in available_seasons:
            season_data = all_by_season.get(str(season), {}).get(pos, {})
            for row in season_data.get("rows", []):
                season_rows.append({**row, "season": season})

        if not season_rows:
            print(f"  WARNING: No rolling {pos} data.")
            continue

        # Collapse to one row per player across all seasons
        combined_rows = _aggregate_rolling_rows(season_rows, pos)

        # Sort by primary volume stat
        sort_key = {"QB": "att", "RB": "carries"}.get(pos, "tgt")
        combined_rows.sort(key=lambda r: r.get(sort_key) or 0, reverse=True)

        # Columns: use per-season column spec (no season column needed)
        sample_cols: list[dict] = []
        for s in available_seasons:
            candidate = all_by_season.get(str(s), {}).get(pos, {}).get("columns", [])
            if candidate:
                sample_cols = candidate
                break

        seasons_present = sorted({r["season"] for r in season_rows})
        sample_window = (
            f"{seasons_present[0]}\u2013{seasons_present[-1]}"
            if len(seasons_present) >= 2 else str(seasons_present[0])
        )

        dataset = {
            "position":     pos,
            "sampleLabel":  "Multi-Season (Combined)",
            "sampleWindow": sample_window,
            "pipelineYear": PIPELINE_YEAR,
            "generatedAt":  datetime.now(timezone.utc).isoformat(),
            "playerCount":  len(combined_rows),
            "columns":      sample_cols,
            "rows":         combined_rows,
        }

        filename = f"stat_explorer_{slug}.json"
        out_path = OUTPUT_DIR / filename
        with open(out_path, "w") as fh:
            json.dump(dataset, fh, separators=(",", ":"))
        kb = out_path.stat().st_size / 1024
        print(f"  Wrote {filename}  ({kb:.1f} KB, {len(combined_rows)} players across {seasons_present})")
        all_rolling[pos] = dataset

    # ── union file ─────────────────────────────────────────────────────────────
    union = {"seasons": all_by_season, "rolling": all_rolling}
    union_path = OUTPUT_DIR / "stat_explorer_latest.json"
    with open(union_path, "w") as fh:
        json.dump(union, fh, separators=(",", ":"))
    kb = union_path.stat().st_size / 1024
    print(f"\n  Wrote stat_explorer_latest.json  ({kb:.1f} KB)")

    print("\nStat Explorer build complete.")
    base = "https://raw.githubusercontent.com/<your-org>/statchasers-data/main/output"
    print("\nPer-season endpoints (example):")
    for season in available_seasons:
        print(f"  {base}/stat_explorer_qb_{season}.json")
    print(f"\n  {base}/stat_explorer_latest.json  (all seasons + rolling)")


if __name__ == "__main__":
    main()
