"""
build_te_player_overview.py
───────────────────────────
Builds the TE Player Overview dataset for the StatChasers TE explorer.

Produces:
  output/te_player_overview_2025.json   — 2025 season only

Data sources:
  data/raw/nflverse_play_by_play.parquet   — targets, recs, air yards, YAC, games
  data/raw/nflverse_participation.parquet  — routes run (TE appearances on pass plays)
  data/processed/player_metrics.json       — snapShare, fpoe, careerArc, wopr,
                                             stability, volatility (2025 only)
  data/raw/nflverse_players.parquet        — years_of_experience → exp_tier
  data/raw/sleeper_players.json            — name disambiguation + age
  data/raw/nflverse_snap_counts.parquet    — per-game snap percentages

Data rules:
  - TEs only
  - ≥ 15 targets to qualify
  - null when data is truly unavailable
  - percentages as decimals (4 dp), rate stats 2 dp
  - Do NOT compute TE Tier in backend — client-side only
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import math

import numpy as np
import pandas as pd

ROOT               = Path(__file__).resolve().parent.parent
PBP_PATH           = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
PARTICIPATION_PATH = ROOT / "data" / "raw"       / "nflverse_participation.parquet"
SNAP_COUNTS_PATH   = ROOT / "data" / "raw"       / "nflverse_snap_counts.parquet"
METRICS_PATH       = ROOT / "data" / "processed" / "player_metrics.json"
NFL_PATH           = ROOT / "data" / "raw"       / "nflverse_players.parquet"
SLEEPER_PATH       = ROOT / "data" / "raw"       / "sleeper_players.json"
OUTPUT_DIR         = ROOT / "output"

SEASON      = 2025
MIN_TARGETS = 8

_EXCLUDED_PLAYERS: frozenset[str] = frozenset({
    "Eli Wilson",
    "Daniel Brown",
})

COLUMNS: list[str] = [
    "player", "team", "age", "games",
    "snap_pct",
    "routes_per_gm", "targets_per_gm", "receptions_per_gm", "air_yards_per_gm",
    "target_share_pct", "air_yards_share_pct", "wopr", "rz_tgt",
    "catch_rate", "yards_per_route_run", "yards_per_target", "yac_per_rec", "fpoe",
    "stability", "volatility",
    "career_arc", "exp_tier",
    "role_score", "efficiency_score", "overall_score",
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
        counts[ab] = counts.get(ab, 0) + 1
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
    return {ab: t for ab, t in by_team.items() if counts[ab] > 1}


_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    "D.Allen":   {"LA": "Davis Allen", "LAR": "Davis Allen", "PHI": "Devon Allen"},
    "J.Smith":   {"ATL": "Jonnu Smith", "MIA": "Jonnu Smith", "PIT": "Jonnu Smith"},
    "M.Andrews": {"BAL": "Mark Andrews"},
    "D.Njoku":   {"CLE": "David Njoku"},
    "I.Thomas":  {"NYG": "Isaiah Thomas"},
    # M.Evans: Mitchell Evans (TE, CAR) vs Mike Evans (WR, TB).
    # Sleeper lists Mike Evans as team=SF (stale); without the TB entry the
    # TB plays fall through to the TE positional fallback → Mitchell Evans.
    "M.Evans":   {"CAR": "Mitchell Evans", "TB": "Mike Evans"},
    # T.Johnson: Theo Johnson (TE, NYG) vs Tyler Johnson (WR, NYJ),
    # Tez Johnson (WR, TB) and Ty Johnson (RB, BUF).
    # Only NYG was defined before; other-team plays fell through to Theo Johnson.
    "T.Johnson": {"NYG": "Theo Johnson", "NYJ": "Tyler Johnson",
                  "TB": "Tez Johnson",   "BUF": "Ty Johnson"},
}

_SNAP_PLAYER_ALIASES: dict[str, str] = {
    # snap_counts (PFR) full name → Sleeper canonical name
    "Chigoziem Okonkwo": "Chig Okonkwo",
    "Oronde Gadsden II": "Oronde Gadsden",
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
        non_conflicting = [(t, n) for t, n in pos_matches if t is None or t == pbp_team]
        if non_conflicting:
            return non_conflicting[0][1]
        with_team = [(t, n) for t, n in pos_matches if t is not None]
        if with_team:
            return with_team[0][1]
        return pos_matches[0][1]
    return pbp_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_tier(yoe: float | None) -> str | None:
    if yoe is None:
        return None
    try:
        y = int(float(yoe))
    except (ValueError, TypeError):
        return None
    if y <= 1:  return "Rookie"
    if y == 2:  return "Year 2"
    if y <= 4:  return "Year 3–4"
    if y <= 9:  return "Veteran"
    return "Senior Veteran"


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


def _stability_volatility(
    game_target_map: dict[str, int],
    game_order: list[str],
) -> tuple[float | None, float | None]:
    ordered = [game_target_map[g] for g in game_order if g in game_target_map]
    extra   = [v for g, v in game_target_map.items() if g not in game_order]
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


def _norm(name: str) -> str:
    return name.lower().replace(".", "").replace("-", "").replace(" ", "").replace("'", "")


# ---------------------------------------------------------------------------
# Routes lookup (participation → TE appearances on pass plays)
# ---------------------------------------------------------------------------

def _build_routes_lookup(
    pbp: pd.DataFrame,
    part: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Return {player_abbrev → {game_id → route_count}}."""
    gsis_to_abbrev: dict[str, str] = (
        pbp[pbp["pass_attempt"] == 1][["receiver_player_id", "receiver_player_name"]]
        .dropna(subset=["receiver_player_id", "receiver_player_name"])
        .drop_duplicates("receiver_player_id")
        .set_index("receiver_player_id")["receiver_player_name"]
        .to_dict()
    )

    part = part.copy()
    part["week_num"] = part["nflverse_game_id"].str.split("_").str[1].astype(int)
    part_reg = part[
        (part["week_num"] <= 18)
        & part["offense_players"].notna()
        & part["offense_positions"].notna()
    ].copy()

    part_reg["pid_list"] = part_reg["offense_players"].str.split(";")
    part_reg["pos_list"] = part_reg["offense_positions"].str.split(";")

    exp = (
        part_reg[["nflverse_game_id", "season", "pid_list", "pos_list"]]
        .explode(["pid_list", "pos_list"])
        .rename(columns={"pid_list": "gsis_id", "pos_list": "position"})
    )
    exp["gsis_id"]  = exp["gsis_id"].str.strip()
    exp["position"] = exp["position"].str.strip()

    te = exp[exp["position"] == "TE"].copy()
    te["abbrev"] = te["gsis_id"].map(gsis_to_abbrev)
    te = te[te["abbrev"].notna()]

    result: dict[str, dict[str, int]] = {}
    for (abbrev, game_id), cnt in (
        te.groupby(["abbrev", "nflverse_game_id"]).size().items()
    ):
        result.setdefault(str(abbrev), {})[str(game_id)] = int(cnt)

    matched = te["abbrev"].notna().sum()
    total   = len(exp[exp["position"] == "TE"])
    pct_str = f"{matched/total*100:.1f}%" if total > 0 else "n/a"
    print(f"  Routes lookup built: {matched:,}/{total:,} TE instances matched ({pct_str})")
    return result


# ---------------------------------------------------------------------------
# Snap-count lookup
# ---------------------------------------------------------------------------

def _build_snap_pct_lookup(
    snap_df: pd.DataFrame,
    te_set: set[str],
) -> dict[str, float]:
    """Return {_norm(canonical_name) → avg_snap_pct} for TEs."""
    # Include QB-designated players whose Sleeper position is TE (e.g. Taysom Hill).
    te_snaps = snap_df[snap_df["position"].isin(["TE", "QB"])].copy()
    ns_to_canonical: dict[str, str] = {_norm(c): c for c in te_set}

    by_name: dict[str, list[float]] = defaultdict(list)
    for _, row in te_snaps.iterrows():
        pfr_name = str(row["player"])
        pct      = row.get("offense_pct")
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            continue
        aliased   = _SNAP_PLAYER_ALIASES.get(pfr_name, pfr_name)
        canonical = (
            ns_to_canonical.get(_norm(aliased))
            or ns_to_canonical.get(_norm(pfr_name))
        )
        if canonical:
            by_name[canonical].append(float(pct))

    return {
        _norm(name): round(float(np.mean(vals)), 4)
        for name, vals in by_name.items()
        if vals
    }


def _clean_rows(rows: list[dict]) -> list[dict]:
    """Replace float NaN/Inf with None so json.dump produces valid JSON."""
    cleaned = []
    for row in rows:
        cleaned.append({
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in row.items()
        })
    return cleaned


# ---------------------------------------------------------------------------
# Per-season builder
# ---------------------------------------------------------------------------

def build_season(
    pbp_season: pd.DataFrame,
    sleeper_players: list[dict],
    metrics_map: dict[str, dict],
    yoe_lookup: dict[str, int],
    sleeper_age: dict[str, Any],
    routes_map: dict[str, dict[str, int]],
    snap_by_norm: dict[str, float],
) -> list[dict]:
    full_name_set: set[str] = {p["full_name"] for p in sleeper_players if p.get("full_name")}
    sleeper_pos: dict[str, str] = {
        p["full_name"]: p.get("position", "")
        for p in sleeper_players if p.get("full_name")
    }
    sleeper_te_set: set[str] = {
        p["full_name"] for p in sleeper_players
        if p.get("position") == "TE" and p.get("full_name")
    }

    unambig       = _build_unambig(sleeper_players)
    team_disambig = _build_team_disambig(sleeper_players)

    pass_plays = pbp_season[pbp_season["pass_attempt"] == 1].copy()
    if pass_plays.empty:
        return []

    has_yardline = "yardline_100" in pass_plays.columns
    has_yac      = "yards_after_catch" in pass_plays.columns
    has_air      = "air_yards" in pass_plays.columns

    cache: dict[tuple[str, str], str] = {}

    def _tag(row: pd.Series) -> str:
        pbp_name = str(row.get("receiver_player_name") or "")
        if not pbp_name or pbp_name == "nan":
            return ""
        key = (pbp_name, str(row.get("posteam", "")))
        if key not in cache:
            cache[key] = _resolve(key[0], key[1], full_name_set, unambig, team_disambig, sleeper_pos)
        return cache[key]

    targeted = pass_plays[pass_plays["receiver_player_name"].notna()].copy()
    targeted["_fn"] = targeted.apply(_tag, axis=1)
    targeted = targeted[targeted["_fn"] != ""]

    def _is_te(fn: str) -> bool:
        s_pos = sleeper_pos.get(fn, "")
        if s_pos == "TE":
            return True
        if s_pos and s_pos != "TE":
            return False
        return metrics_map.get(fn, {}).get("pos", "") == "TE"

    te_names: set[str] = {fn for fn in targeted["_fn"].unique() if _is_te(str(fn))}
    te_plays = targeted[targeted["_fn"].isin(te_names)].copy()
    if te_plays.empty:
        return []

    canonical_to_gsis: dict[str, str] = {}
    if "receiver_player_id" in te_plays.columns:
        canonical_to_gsis = (
            te_plays[["_fn", "receiver_player_id"]]
            .dropna(subset=["receiver_player_id"])
            .drop_duplicates("_fn")
            .set_index("_fn")["receiver_player_id"]
            .to_dict()
        )

    team_pass_totals: dict[str, int] = (
        pass_plays.groupby("posteam")["pass_attempt"].count().to_dict()
    )
    if has_air:
        team_air_totals: dict[str, float] = (
            pass_plays.groupby("posteam")["air_yards"].sum().to_dict()
        )
    else:
        team_air_totals = {}

    sort_cols = [c for c in ("season", "week") if c in te_plays.columns]
    rows: list[dict] = []

    for fn, grp in te_plays.groupby("_fn"):
        fn  = str(fn)
        tgt = len(grp)
        if tgt < MIN_TARGETS:
            continue

        comps = grp[grp["complete_pass"] == 1]
        rec   = len(comps)
        yds   = int(comps["yards_gained"].sum())
        games = int(grp["game_id"].nunique())

        if sort_cols:
            team = str(grp.sort_values(sort_cols)["posteam"].iloc[-1])
            game_order = (
                grp[["game_id"] + sort_cols]
                .drop_duplicates("game_id")
                .sort_values(sort_cols)["game_id"]
                .tolist()
            )
        else:
            team       = str(grp["posteam"].iloc[-1])
            game_order = list(grp["game_id"].unique())

        game_tgt_map: dict[str, int] = dict(
            grp.groupby("game_id")["pass_attempt"].count()
        )

        if has_air:
            player_air = float(grp["air_yards"].dropna().sum())
            team_air   = team_air_totals.get(team, 0.0)
        else:
            player_air = 0.0
            team_air   = 0.0

        total_yac = float(comps["yards_after_catch"].dropna().sum()) if has_yac and rec > 0 else None
        rz_tgt    = int((grp["yardline_100"] <= 20).sum()) if has_yardline else None

        team_pass = team_pass_totals.get(team, 0)
        target_share_pct    = round(tgt / team_pass, 4)       if team_pass > 0 else None
        air_yards_share_pct = round(player_air / team_air, 4) if team_air  > 0 else None

        wopr = None
        if target_share_pct is not None and air_yards_share_pct is not None:
            wopr = round(1.5 * target_share_pct + 0.7 * air_yards_share_pct, 2)

        catch_rate    = round(rec / tgt, 4)       if tgt > 0 else None
        yards_per_tgt = round(yds / tgt, 2)       if tgt > 0 else None
        yac_per_rec   = round(total_yac / rec, 2) if rec > 0 and total_yac is not None else None
        tgt_per_gm    = round(tgt / games, 2)     if games > 0 else None
        rec_per_gm    = round(rec / games, 2)     if games > 0 else None
        air_per_gm    = round(player_air / games, 2) if games > 0 else None

        # Routes
        pbp_abbrev   = grp["receiver_player_name"].mode().iloc[0] if len(grp) > 0 else ""
        route_map    = routes_map.get(pbp_abbrev, {})
        total_routes = sum(route_map.values())
        routes_per_gm       = round(total_routes / games, 2) if total_routes > 0 and games > 0 else None
        yards_per_route_run = round(yds / total_routes, 2)   if total_routes > 0 else None

        # Snap %
        snap_pct = snap_by_norm.get(_norm(fn)) if snap_by_norm else None

        # Stability / volatility
        stability, volatility = _stability_volatility(game_tgt_map, game_order)

        # Metrics from player_metrics (2025 only)
        m = metrics_map.get(fn, {})
        career_arc = m.get("careerArc") or m.get("career_arc")
        fpoe_val   = m.get("fpoe")
        fpoe       = round(float(fpoe_val), 2) if fpoe_val is not None else None

        # Age (from Sleeper)
        age_raw = sleeper_age.get(fn)
        age = int(age_raw) if age_raw is not None else None

        # Experience tier (from nflverse)
        yoe     = yoe_lookup.get(fn)
        exp_tier = _exp_tier(yoe)

        rows.append({
            "player":             fn,
            "team":               team,
            "age":                age,
            "games":              games,
            "snap_pct":           snap_pct,
            "routes_per_gm":      routes_per_gm,
            "targets_per_gm":     tgt_per_gm,
            "receptions_per_gm":  rec_per_gm,
            "air_yards_per_gm":   air_per_gm,
            "target_share_pct":   target_share_pct,
            "air_yards_share_pct": air_yards_share_pct,
            "wopr":               wopr,
            "rz_tgt":             rz_tgt,
            "catch_rate":         catch_rate,
            "yards_per_route_run": yards_per_route_run,
            "yards_per_target":   yards_per_tgt,
            "yac_per_rec":        yac_per_rec,
            "fpoe":               fpoe,
            "stability":          stability,
            "volatility":         volatility,
            "career_arc":         career_arc,
            "exp_tier":           exp_tier,
            # Scores added below
            "opp_score":       None,
            "usage_score":     None,
            "efficiency_score": None,
            "overall_score":   None,
        })

    return rows


# ---------------------------------------------------------------------------
# Composite score computation
# ---------------------------------------------------------------------------

def _add_composite_scores(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)

    def _pct_rank(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(50.0, index=df.index)
        return df[col].rank(pct=True, na_option="bottom") * 100

    # Role score: unified opportunity + deployment profile
    #   Answers: how strong and fantasy-relevant is this TE's offensive role?
    role = (
        _pct_rank("target_share_pct")     * 0.20
        + _pct_rank("targets_per_gm")     * 0.15
        + _pct_rank("routes_per_gm")      * 0.15
        + _pct_rank("air_yards_share_pct") * 0.10
        + _pct_rank("wopr")               * 0.10
        + _pct_rank("air_yards_per_gm")   * 0.10
        + _pct_rank("snap_pct")           * 0.10
        + _pct_rank("rz_tgt")             * 0.10
    )
    df["role_score"] = role.clip(0, 100).round(1)

    # Efficiency score: per-touch production quality (unchanged formula)
    player = (
        _pct_rank("yards_per_route_run") * 0.25
        + _pct_rank("catch_rate")        * 0.20
        + _pct_rank("yards_per_target")  * 0.20
        + _pct_rank("yac_per_rec")       * 0.15
        + _pct_rank("fpoe")              * 0.10
        + _pct_rank("stability")         * 0.10
    )
    df["efficiency_score"] = player.round(1)

    # Overall score: role 50% + efficiency 35% + stability/reliability 15%
    stability_block = (
        _pct_rank("stability")             * 0.60
        + (100 - _pct_rank("volatility"))  * 0.40
    )
    overall = (
        df["role_score"]       * 0.50
        + df["efficiency_score"] * 0.35
        + stability_block        * 0.15
    ).clip(0, 100).round(1)
    df["overall_score"] = overall

    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (PBP_PATH, PARTICIPATION_PATH, SNAP_COUNTS_PATH, METRICS_PATH, NFL_PATH, SLEEPER_PATH):
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

    print("Loading participation data...")
    part = pd.read_parquet(
        PARTICIPATION_PATH,
        columns=["nflverse_game_id", "season", "offense_players", "offense_positions"],
    )
    part = part[part["season"] == SEASON].copy()

    print("Loading snap counts...")
    snap_df = pd.read_parquet(SNAP_COUNTS_PATH)
    snap_df = snap_df[snap_df["season"] == SEASON].copy()

    print("Loading Sleeper players...")
    with open(SLEEPER_PATH) as f:
        raw = json.load(f)
    sleeper_players = list(raw.values()) if isinstance(raw, dict) else raw

    sleeper_te_set: set[str] = {
        p["full_name"] for p in sleeper_players
        if p.get("position") == "TE" and p.get("full_name")
    }
    sleeper_age: dict[str, Any] = {
        p["full_name"]: p.get("age")
        for p in sleeper_players if p.get("full_name") and p.get("age") is not None
    }

    print("Loading player metrics...")
    with open(METRICS_PATH) as f:
        raw_m = json.load(f)
    metrics_list = raw_m if isinstance(raw_m, list) else raw_m.get("players", [])
    metrics_map: dict[str, dict] = {m["player"]: m for m in metrics_list if m.get("player")}

    print("Loading nflverse players (experience)...")
    nfl_df = pd.read_parquet(NFL_PATH)
    yoe_lookup: dict[str, int] = {}
    if "display_name" in nfl_df.columns and "years_of_experience" in nfl_df.columns:
        for _, row in nfl_df.dropna(subset=["display_name"]).iterrows():
            yoe = row.get("years_of_experience")
            if yoe is not None:
                try:
                    yoe_lookup[str(row["display_name"])] = int(float(yoe))
                except (ValueError, TypeError):
                    pass

    print("Building routes lookup...")
    routes_map = _build_routes_lookup(pbp_full, part)

    print("Building snap % lookup...")
    snap_by_norm = _build_snap_pct_lookup(snap_df, sleeper_te_set)

    print(f"Building {SEASON} TE overview...")
    rows = build_season(
        pbp_full, sleeper_players, metrics_map, yoe_lookup,
        sleeper_age, routes_map, snap_by_norm,
    )

    rows = _add_composite_scores(rows)

    # Sort: overall_score → role_score → efficiency_score (all desc)
    rows.sort(
        key=lambda r: (
            -(r["overall_score"] or 0),
            -(r["role_score"] or 0),
            -(r["efficiency_score"] or 0),
        )
    )

    rows = [r for r in rows if r["player"] not in _EXCLUDED_PLAYERS]

    # Validation
    assert all(r["player"] for r in rows), "Empty player name"
    assert len({r["player"] for r in rows}) == len(rows), "Duplicate players"
    for r in rows:
        missing = set(COLUMNS) - set(r.keys())
        assert not missing, f"Missing columns for {r['player']}: {missing}"
        assert r.get("role_score") is not None, f"Null role_score: {r['player']}"
        assert r.get("efficiency_score") is not None, f"Null efficiency_score: {r['player']}"
        assert r.get("overall_score") is not None, f"Null overall_score: {r['player']}"
    print(f"Validation OK: {len(rows)} TEs, no duplicates")

    ordered_rows = [{c: r.get(c) for c in COLUMNS} for r in rows]

    payload: dict[str, Any] = {
        "meta": {
            "position":     "TE",
            "table":        "te_player_overview",
            "season":       str(SEASON),
            "generated_at": now,
            "columns":      COLUMNS,
        },
        "players": _clean_rows(ordered_rows),
    }

    out_path = OUTPUT_DIR / f"te_player_overview_{SEASON}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.name} ({kb:.1f} KB, {len(rows)} TEs)")
    print("TE player overview build complete.")


if __name__ == "__main__":
    main()
