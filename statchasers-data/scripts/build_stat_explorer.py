"""
build_stat_explorer.py
──────────────────────
Builds the Stat Explorer dataset for the StatChasers frontend.

Stat Explorer is the *raw-stat layer* — transparent counting stats and
near-raw rates with no modeled labels, career-arc classifications, or
interpretation fields.  Advanced Metrics is the intelligence layer.
These two outputs must remain strictly separate.

Currently supports QB.  RB / WR / TE can be added by implementing
their FIELD_MAP entry and compute function; the response envelope and
coverage logic are shared.

Reads from:
  data/raw/nflverse_play_by_play.parquet   — all downloaded seasons (PBP)
  data/raw/sleeper_players.json            — canonical full-name lookup
  data/processed/player_metrics.json       — team / player-context lookup

Outputs:
  output/stat_explorer_qb.json             — QB-only dataset
  output/stat_explorer_latest.json         — all positions keyed by pos
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
PBP_PATH     = ROOT / "data" / "raw" / "nflverse_play_by_play.parquet"
SLEEPER_PATH = ROOT / "data" / "raw" / "sleeper_players.json"
METRICS_PATH = ROOT / "data" / "processed" / "player_metrics.json"
OUTPUT_DIR   = ROOT / "output"

PIPELINE_YEAR = 2025
MIN_QB_ATT    = 10  # minimum official pass attempts to include a QB


# ---------------------------------------------------------------------------
# NFL passer rating (official formula)
# ---------------------------------------------------------------------------

def _passer_rating(comp: int, att: int, yds: int, td: int, ints: int) -> float | None:
    if att <= 0:
        return None
    a = min(max((comp / att - 0.3) / 0.2,     0.0), 2.375)
    b = min(max((yds  / att - 3.0) / 4.0,     0.0), 2.375)
    c = min(max((td   / att) / 0.05,           0.0), 2.375)
    d = min(max(2.375 - (ints / att) / 0.095,  0.0), 2.375)
    return round((a + b + c + d) / 6.0 * 100.0, 1)


# ---------------------------------------------------------------------------
# Name-resolution helpers  (mirrors compute_player_metrics.py)
# ---------------------------------------------------------------------------

def _nflverse_abbrev(full_name: str) -> str:
    """'Josh Allen' -> 'J.Allen',  'Ja\'Marr Chase' -> 'J.Chase'"""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0].upper()}.{parts[-1]}"


def _build_abbrev_lookup(sleeper_players: list[dict]) -> dict[str, str]:
    """
    abbrev -> full_name, dropping ambiguous abbreviations so we never
    silently assign the wrong player.
    """
    counts: dict[str, int]  = {}
    mapping: dict[str, str] = {}
    for p in sleeper_players:
        full = p.get("full_name", "").strip()
        if not full:
            continue
        abbrev = _nflverse_abbrev(full)
        counts[abbrev]  = counts.get(abbrev, 0) + 1
        mapping[abbrev] = full
    return {ab: fn for ab, fn in mapping.items() if counts[ab] == 1}


def _resolve_full_name(
    pbp_name: str,
    full_name_set: set[str],
    abbrev_lookup: dict[str, str],
) -> str:
    if pbp_name in full_name_set:
        return pbp_name
    return abbrev_lookup.get(pbp_name, pbp_name)


# ---------------------------------------------------------------------------
# Field maps  { position -> [(backend_key, display_label, value_type)] }
#
# value_type hints for the frontend renderer:
#   "string"   text
#   "number"   integer count  (displayed as-is)
#   "decimal"  float with 1-2 decimal places
#   "percent"  float displayed with a % suffix
# ---------------------------------------------------------------------------

QB_FIELDS: list[tuple[str, str, str]] = [
    ("player",            "PLAYER",  "string"),
    ("team",              "TEAM",    "string"),
    ("position",          "POS",     "string"),
    ("gp",                "GP",      "number"),
    ("comp",              "COMP",    "number"),
    ("att",               "ATT",     "number"),
    ("pct",               "PCT",     "percent"),
    ("yds",               "YDS",     "number"),
    ("ypa",               "Y/A",     "decimal"),
    ("td",                "TD",      "number"),
    ("int",               "INT",     "number"),
    ("sacks",             "SACK",    "number"),
    ("air_yards",         "AIR YDS", "number"),
    ("air_yards_per_att", "AIR/A",   "decimal"),
    ("pass_10_plus",      "10+",     "number"),
    ("pass_20_plus",      "20+",     "number"),
    ("pass_30_plus",      "30+",     "number"),
    ("pass_40_plus",      "40+",     "number"),
    ("pass_50_plus",      "50+",     "number"),
    ("rz_att",            "RZ ATT",  "number"),
    ("passer_rating",     "RTG",     "decimal"),
]

_FIELD_MAPS: dict[str, list[tuple[str, str, str]]] = {
    "QB": QB_FIELDS,
    # "RB": RB_FIELDS,   ← add here when ready
    # "WR": WR_FIELDS,
    # "TE": TE_FIELDS,
}


def get_field_map(position: str) -> list[tuple[str, str, str]]:
    if position not in _FIELD_MAPS:
        raise ValueError(f"No field map for position '{position}'. "
                         f"Available: {list(_FIELD_MAPS)}")
    return _FIELD_MAPS[position]


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------

def compute_field_coverage(rows: list[dict], keys: list[str]) -> dict[str, float]:
    """
    Return the fraction of rows (0.0–1.0) where each key has a non-null,
    non-empty value.  Used to detect dead columns before sending to frontend.
    """
    n = len(rows)
    if n == 0:
        return {k: 0.0 for k in keys}
    result = {}
    for key in keys:
        present = sum(
            1 for r in rows
            if r.get(key) is not None and r.get(key) != ""
        )
        result[key] = round(present / n, 3)
    return result


def build_stat_explorer_columns(
    position: str,
    coverage: dict[str, float],
) -> list[dict]:
    """
    Build the columns metadata array.

    Fields whose coverage is below 1 % of rows are omitted so the
    frontend never renders an empty column (e.g. pocket_time if we
    add it to the field map before the data source is available).
    """
    columns = []
    for key, label, vtype in get_field_map(position):
        cov = coverage.get(key, 0.0)
        if vtype != "string" and cov < 0.01:
            continue
        columns.append({
            "key":      key,
            "label":    label,
            "type":     vtype,
            "coverage": cov,
        })
    return columns


# ---------------------------------------------------------------------------
# QB raw-stat computation
# ---------------------------------------------------------------------------

def compute_qb_raw_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling multi-season counting stats for each QB.

    Official pass attempts exclude sacks so totals match ESPN/NFL box scores.
    Sacks are counted separately and appended back.
    Games are credited from all dropbacks (including sacks).
    INT requires the 'interception' column; if absent it is treated as 0.
    """
    all_dropbacks = pbp[pbp["pass_attempt"] == 1].copy()
    if all_dropbacks.empty:
        return pd.DataFrame()

    has_sack      = "sack"         in all_dropbacks.columns
    has_int       = "interception" in all_dropbacks.columns
    has_yardline  = "yardline_100" in all_dropbacks.columns

    if has_sack:
        official = all_dropbacks[all_dropbacks["sack"] != 1].copy()
    else:
        print("WARNING: 'sack' column absent; attempts will include sacks.", file=sys.stderr)
        official = all_dropbacks.copy()

    completions = official[official["complete_pass"] == 1].copy()

    rows = []
    for name, grp in official.groupby("passer_player_name"):
        att  = len(grp)
        comp = int(grp["complete_pass"].sum())
        yds  = int(grp["yards_gained"].sum())
        td   = int(grp["touchdown"].sum())
        ints = int(grp["interception"].sum()) if has_int else 0

        # sacks from all dropbacks (not just official attempts)
        db_grp = all_dropbacks[all_dropbacks["passer_player_name"] == name]
        sacks  = int(db_grp["sack"].sum()) if has_sack else 0

        # games from all dropbacks so a sack-only game still counts
        gp = int(db_grp["game_id"].nunique())

        air_yds = round(float(grp["air_yards"].dropna().sum()), 1)

        rz_att = (
            int((grp["yardline_100"] <= 20).sum())
            if has_yardline else None
        )

        # completions of X+ yards  (only complete passes count)
        comp_yds = completions.loc[
            completions["passer_player_name"] == name, "yards_gained"
        ]
        pass_10p = int((comp_yds >= 10).sum())
        pass_20p = int((comp_yds >= 20).sum())
        pass_30p = int((comp_yds >= 30).sum())
        pass_40p = int((comp_yds >= 40).sum())
        pass_50p = int((comp_yds >= 50).sum())

        pct         = round(comp / att * 100, 1) if att > 0 else 0.0
        ypa         = round(yds  / att,        2) if att > 0 else 0.0
        air_per_att = round(air_yds / att,     2) if att > 0 else 0.0
        rating      = _passer_rating(comp, att, yds, td, ints)

        # most-recent team (last game in the dataset)
        team = str(grp["posteam"].iloc[-1]) if "posteam" in grp.columns else ""

        rows.append({
            "player_name":      name,
            "team_pbp":         team,
            "gp":               gp,
            "comp":             comp,
            "att":              att,
            "pct":              pct,
            "yds":              yds,
            "ypa":              ypa,
            "td":               td,
            "int":              ints,
            "sacks":            sacks,
            "air_yards":        air_yds,
            "air_yards_per_att": air_per_att,
            "pass_10_plus":     pass_10p,
            "pass_20_plus":     pass_20p,
            "pass_30_plus":     pass_30p,
            "pass_40_plus":     pass_40p,
            "pass_50_plus":     pass_50p,
            "rz_att":           rz_att,
            "passer_rating":    rating,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Row sanitizer — one per position
# ---------------------------------------------------------------------------

def sanitize_qb_row(
    raw: dict[str, Any],
    full_name: str,
    team: str,
) -> dict[str, Any]:
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
        "sacks":              raw.get("sacks"),
        "air_yards":          raw.get("air_yards"),
        "air_yards_per_att":  raw.get("air_yards_per_att"),
        "pass_10_plus":       raw.get("pass_10_plus"),
        "pass_20_plus":       raw.get("pass_20_plus"),
        "pass_30_plus":       raw.get("pass_30_plus"),
        "pass_40_plus":       raw.get("pass_40_plus"),
        "pass_50_plus":       raw.get("pass_50_plus"),
        "rz_att":             raw.get("rz_att"),
        "passer_rating":      raw.get("passer_rating"),
    }


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_stat_explorer_dataset(
    position: str,
    pbp: pd.DataFrame,
    sleeper_players: list[dict],
    player_metrics: list[dict],
) -> dict:
    """
    Orchestrate raw-stat computation, apply field-coverage checks, and
    return the full Stat Explorer response envelope for one position.
    """
    full_name_set  = {p.get("full_name", "") for p in sleeper_players}
    abbrev_lookup  = _build_abbrev_lookup(sleeper_players)

    # Analytics-pipeline lookup: full_name -> {team, pos, ...}
    # Used to prefer the canonical full name and Sleeper team abbreviation
    # over the PBP-abbreviated name and PBP team code.
    metrics_lookup: dict[str, dict] = {
        p["player"]: p
        for p in player_metrics
        if p.get("pos") == position
    }

    if position == "QB":
        raw_df  = compute_qb_raw_stats(pbp)
        min_att = MIN_QB_ATT
    else:
        raise ValueError(f"Position '{position}' not yet implemented.")

    if raw_df.empty:
        return {}

    # Filter to minimum attempt threshold
    raw_df = raw_df[raw_df["att"] >= min_att].reset_index(drop=True)

    # Detect sample window from PBP seasons
    seasons = sorted(pbp["season"].dropna().unique().astype(int).tolist())
    if len(seasons) >= 2:
        sample_label  = "Rolling Multi-Season"
        sample_window = f"{seasons[0]}\u2013{seasons[-1]}"   # en-dash
    elif len(seasons) == 1:
        sample_label  = f"{seasons[0]} Season"
        sample_window = str(seasons[0])
    else:
        sample_label  = "Unknown"
        sample_window = ""

    rows: list[dict] = []
    for _, raw_row in raw_df.iterrows():
        pbp_name = raw_row["player_name"]
        pbp_team = raw_row["team_pbp"]

        # Resolve to Sleeper canonical full name
        full_name = _resolve_full_name(pbp_name, full_name_set, abbrev_lookup)

        # Prefer Sleeper/analytics team over PBP team code (same values but
        # Sleeper sometimes has more up-to-date transaction data mid-season)
        info = metrics_lookup.get(full_name, {})
        team = info.get("team") or pbp_team

        if position == "QB":
            row = sanitize_qb_row(raw_row.to_dict(), full_name, team)
        else:
            continue

        rows.append(row)

    # Sort by attempts descending — most active players first
    rows.sort(key=lambda r: r.get("att") or 0, reverse=True)

    # Coverage + column metadata
    all_keys = [k for k, _, _ in get_field_map(position)]
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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path, label in [(PBP_PATH, "PBP parquet"), (SLEEPER_PATH, "Sleeper players"),
                        (METRICS_PATH, "player_metrics.json")]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    print("Loading PBP data...")
    pbp = pd.read_parquet(PBP_PATH)

    print("Loading Sleeper players...")
    with open(SLEEPER_PATH) as f:
        sleeper_players = json.load(f)

    print("Loading player metrics...")
    with open(METRICS_PATH) as f:
        player_metrics = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_positions: dict[str, dict] = {}

    # ── QB ─────────────────────────────────────────────────────────────────
    print("Building QB Stat Explorer...")
    qb_dataset = build_stat_explorer_dataset("QB", pbp, sleeper_players, player_metrics)
    qb_path = OUTPUT_DIR / "stat_explorer_qb.json"
    with open(qb_path, "w") as f:
        json.dump(qb_dataset, f, separators=(",", ":"))
    print(f"  Wrote {qb_path.name}  "
          f"({qb_path.stat().st_size / 1024:.1f} KB, "
          f"{qb_dataset['playerCount']} QBs)")
    all_positions["QB"] = qb_dataset

    # ── add RB / WR / TE here in future sprints ────────────────────────────

    # Union file — all positions in one payload
    union_path = OUTPUT_DIR / "stat_explorer_latest.json"
    with open(union_path, "w") as f:
        json.dump(all_positions, f, separators=(",", ":"))
    print(f"  Wrote {union_path.name}  "
          f"({union_path.stat().st_size / 1024:.1f} KB)")

    print("\nStat Explorer build complete.")
    print("\nFrontend endpoints:")
    base = "https://raw.githubusercontent.com/<your-org>/statchasers-data/main/output"
    print(f"  {base}/stat_explorer_qb.json")
    print(f"  {base}/stat_explorer_latest.json")


if __name__ == "__main__":
    main()
