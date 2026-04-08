#!/usr/bin/env python3
"""
build_te_efficiency_analytics.py

Build te_efficiency_analytics_2025.json for the TE Efficiency Analytics tab.

Metrics per TE (15+ targets, REG season, 2025):
  Core    : epa_per_target, success_rate, yards_per_target, catch_rate, fpoe
  Routes  : yards_per_route_run, targets_per_route_run
  Air yards: air_yards_per_target, air_yards_share_pct
  Explosive: explosive_play_rate, explosive_rec_20_plus, explosive_rec_40_plus,
             longest_reception
  Contact : yac_per_rec, ybc_per_rec, broken_tackles, btkl_per_rec
  Composite: efficiency_score (0-100, percentile-weighted)

efficiency_score weights (mirrors TE player overview — source of truth):
  YPRR           = 25%
  Catch Rate     = 20%
  Yds / TGT      = 20%
  YAC / Rec      = 15%
  FPOE           = 10%
  Stability      = 10%

Note: efficiency_score values are patched from te_player_overview_{season}.json
at build time to guarantee exact cross-tab alignment.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import math

import numpy as np
import pandas as pd

ROOT               = Path(__file__).resolve().parent.parent
PBP_PATH           = ROOT / "data" / "raw"       / "nflverse_play_by_play.parquet"
PARTICIPATION_PATH = ROOT / "data" / "raw"       / "nflverse_participation.parquet"
METRICS_PATH       = ROOT / "data" / "processed" / "player_metrics.json"
NFL_PATH           = ROOT / "data" / "raw"       / "nflverse_players.parquet"
SLEEPER_PATH       = ROOT / "data" / "raw"       / "sleeper_players.json"
PFR_REC_PATH       = ROOT / "data" / "raw"       / "pfr_rec_advstats.parquet"
OUTPUT_DIR         = ROOT / "output"

SEASON      = 2025
MIN_TARGETS = 15

COLUMNS = [
    "player", "team", "games", "targets",
    "epa_per_target", "success_rate", "yards_per_target", "catch_rate", "fpoe",
    "yards_per_route_run", "targets_per_route_run", "air_yards_per_target",
    "air_yards_share_pct",
    "explosive_play_rate", "explosive_rec_20_plus", "explosive_rec_40_plus",
    "longest_reception",
    "yac_per_rec", "ybc_per_rec", "broken_tackles", "btkl_per_rec",
    "stability",
    "efficiency_score",
]

# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    "D.Allen":   {"LA": "Davis Allen", "LAR": "Davis Allen", "PHI": "Devon Allen"},
    "J.Smith":   {"ATL": "Jonnu Smith", "MIA": "Jonnu Smith", "PIT": "Jonnu Smith"},
    "M.Andrews": {"BAL": "Mark Andrews"},
    "D.Njoku":   {"CLE": "David Njoku"},
    "I.Thomas":  {"NYG": "Isaiah Thomas"},
    # M.Evans: Mitchell Evans (TE, CAR) vs Mike Evans (WR, TB)
    "M.Evans":   {"CAR": "Mitchell Evans"},
    # T.Johnson: Theo Johnson (TE, NYG) vs Ty Johnson (RB)
    "T.Johnson": {"NYG": "Theo Johnson"},
}

_CANONICAL_TO_PFR_NORM: dict[str, str] = {}

_EXCLUDED_PLAYERS: frozenset[str] = frozenset({
    "Eli Wilson",
    "Daniel Brown",
})


def _norm(name: str) -> str:
    if not name:
        return ""
    n = re.sub(r"[^a-z\s]", "", name.lower())
    return re.sub(r"\s+", " ", n).strip()


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


def _resolve(
    pbp_name: str,
    pbp_team: str,
    full_name_set: set[str],
    unambig: dict[str, str],
    team_disambig: dict[str, dict[str, str]],
    sleeper_pos: dict[str, str],
) -> str | None:
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
        conflicting     = [(t, n) for t, n in pos_matches if t is not None and t != pbp_team]
        non_conflicting = [(t, n) for t, n in pos_matches if t is None or t == pbp_team]
        if non_conflicting and conflicting:
            no_team = [(t, n) for t, n in non_conflicting if t is None]
            if no_team:
                return no_team[0][1]
    return None


# ---------------------------------------------------------------------------
# Routes lookup (participation → TE appearances on pass plays)
# ---------------------------------------------------------------------------

def _build_routes_lookup(season: int) -> dict[str, int]:
    """Return {gsis_id → total routes run} for TEs in one season."""
    part = pd.read_parquet(
        PARTICIPATION_PATH,
        columns=["nflverse_game_id", "season",
                 "offense_players", "offense_positions"],
    )
    part = part[
        (part["season"] == season) &
        part["offense_players"].notna() &
        part["offense_positions"].notna()
    ].copy()

    part["week_num"] = part["nflverse_game_id"].str.split("_").str[1].astype(int)
    part = part[part["week_num"] <= 18]

    part["pid_list"] = part["offense_players"].str.split(";")
    part["pos_list"] = part["offense_positions"].str.split(";")
    exp = (
        part[["pid_list", "pos_list"]]
        .explode(["pid_list", "pos_list"])
        .rename(columns={"pid_list": "gsis_id", "pos_list": "position"})
    )
    exp["gsis_id"]  = exp["gsis_id"].str.strip()
    exp["position"] = exp["position"].str.strip()

    te = exp[exp["position"] == "TE"]
    return te.groupby("gsis_id").size().to_dict()


# ---------------------------------------------------------------------------
# PFR broken-tackles lookup
# ---------------------------------------------------------------------------

_BTKL_NORM_SUFFIXES = frozenset(["jr", "sr", "ii", "iii", "iv", "v"])


def _strip_suffixes(norm: str) -> str:
    parts = norm.split()
    while parts and parts[-1] in _BTKL_NORM_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _build_btkl_lookup(season: int) -> dict[str, float]:
    pfr = pd.read_parquet(PFR_REC_PATH)
    pfr = pfr[pfr["season"] == season].copy()
    pfr["_nname"] = pfr["pfr_player_name"].fillna("").apply(_norm)
    agg: dict[str, float] = (
        pfr.groupby("_nname")["receiving_broken_tackles"].sum().to_dict()
    )
    stripped = {
        _strip_suffixes(k): v
        for k, v in agg.items()
        if _strip_suffixes(k) != k and _strip_suffixes(k) not in agg
    }
    agg.update(stripped)
    return agg


# ---------------------------------------------------------------------------
# fpoe lookup
# ---------------------------------------------------------------------------

def _build_fpoe_lookup() -> tuple[dict[str, float], dict[str, float]]:
    with open(METRICS_PATH) as f:
        raw = json.load(f)
    data = raw if isinstance(raw, list) else raw.get("players", [])
    by_name: dict[str, float] = {}
    by_norm: dict[str, float] = {}
    for p in data:
        name = (p.get("player") or "").strip()
        val  = p.get("fpoe")
        if not name or val is None:
            continue
        fval = float(val)
        by_name[name]        = fval
        by_norm[_norm(name)] = fval
    return by_name, by_norm


# ---------------------------------------------------------------------------
# Team air-yards totals
# ---------------------------------------------------------------------------

def _build_team_air_yards(pbp: pd.DataFrame) -> dict[str, float]:
    tgt = pbp[pbp["pass_attempt"] == 1].copy()
    return tgt.groupby("posteam")["air_yards"].sum().to_dict()


# ---------------------------------------------------------------------------
# TE efficiency score
# ---------------------------------------------------------------------------

_SCORE_COMPONENTS = [
    ("yards_per_route_run", 0.25),
    ("catch_rate",          0.20),
    ("yards_per_target",    0.20),
    ("yac_per_rec",         0.15),
    ("fpoe",                0.10),
    ("stability",           0.10),
]


def _stability_volatility(
    game_target_map: dict[str, int],
    game_order: list[str],
) -> tuple[float | None, float | None]:
    """
    stability  = (1 – σ/μ of last-6 games) × 10, clamped 0–10
    volatility = σ/μ of all games (coefficient of variation)
    Mirrors build_te_player_overview._stability_volatility exactly.
    """
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


def _clean_rows(rows: list[dict]) -> list[dict]:
    """Replace float NaN/Inf with None so json.dump produces valid JSON."""
    cleaned = []
    for row in rows:
        cleaned.append({
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in row.items()
        })
    return cleaned


def _add_efficiency_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add efficiency_score column (0–100) using percentile ranks.
    Mirrors the TE Player Overview efficiency_score formula exactly.
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)
    for col, weight in _SCORE_COMPONENTS:
        if col not in df.columns:
            continue
        ranks = df[col].rank(pct=True, na_option="bottom")
        score += ranks * weight
    df["efficiency_score"] = (score * 100).round(1)
    return df


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_season(season: int) -> list[dict[str, Any]]:
    print(f"[{season}] Loading play-by-play …")
    pbp_cols = [
        "season", "season_type", "week", "game_id", "posteam",
        "receiver_player_name", "receiver_player_id",
        "complete_pass", "pass_attempt",
        "yards_gained", "air_yards", "yards_after_catch", "epa",
    ]
    pbp = pd.read_parquet(PBP_PATH, columns=pbp_cols)
    pbp = pbp[
        (pbp["season"] == season) &
        (pbp["season_type"] == "REG") &
        (pbp["week"] <= 18)
    ].copy()

    tgts = pbp[
        (pbp["pass_attempt"] == 1) &
        pbp["receiver_player_name"].notna()
    ].copy()

    print(f"[{season}] Loading Sleeper players …")
    with open(SLEEPER_PATH) as f:
        raw = json.load(f)
    sleeper_players = list(raw.values()) if isinstance(raw, dict) else raw

    sleeper_te_set   = {p["full_name"] for p in sleeper_players
                        if p.get("position") == "TE" and p.get("full_name")}
    sleeper_full_set = {p["full_name"] for p in sleeper_players
                        if p.get("full_name")}
    sleeper_pos: dict[str, str] = {
        p["full_name"]: p.get("position", "")
        for p in sleeper_players if p.get("full_name")
    }

    unambig       = _build_unambig(sleeper_players)
    team_disambig = _build_team_disambig(sleeper_players)

    print(f"[{season}] Loading routes, fpoe, broken-tackles …")
    routes_by_gsis             = _build_routes_lookup(season)
    btkl_by_name               = _build_btkl_lookup(season)
    fpoe_by_name, fpoe_by_norm = _build_fpoe_lookup()
    team_air_yards             = _build_team_air_yards(tgts)

    nfl = pd.read_parquet(NFL_PATH, columns=["gsis_id", "display_name"])
    nfl = nfl.dropna(subset=["gsis_id", "display_name"])

    pairs = (
        tgts[["receiver_player_name", "posteam"]]
        .drop_duplicates()
        .values.tolist()
    )
    pair_to_canonical: dict[tuple[str, str], str | None] = {}
    for aname, team in pairs:
        pair_to_canonical[(aname, team)] = _resolve(
            aname, team, sleeper_full_set, unambig, team_disambig, sleeper_pos
        )

    tgts["canonical"] = tgts.apply(
        lambda r: pair_to_canonical.get((r["receiver_player_name"], r["posteam"])),
        axis=1,
    )

    def _is_te(canonical: str | None) -> bool:
        if not canonical:
            return False
        if canonical in sleeper_te_set:
            return True
        if canonical in sleeper_full_set:
            return False
        return False

    tgts = tgts[tgts["canonical"].apply(_is_te)].copy()

    gsis_to_canonical: dict[str, str] = {}
    for gsis, grp in tgts.groupby("receiver_player_id"):
        if pd.isna(gsis):
            continue
        gsis_to_canonical[gsis] = grp["canonical"].mode().iloc[0]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for canonical, grp in tgts.groupby("canonical"):
        if canonical in seen:
            continue
        seen.add(canonical)

        team    = grp["posteam"].mode().iloc[0]
        games   = grp["game_id"].nunique()
        targets = len(grp)
        if targets < MIN_TARGETS:
            continue

        comps = grp[grp["complete_pass"] == 1]
        rec   = len(comps)

        epa_valid = grp["epa"].dropna()
        epa_per_target = (
            round(float(epa_valid.sum() / len(epa_valid)), 3)
            if len(epa_valid) > 0 else None
        )
        success_rate = (
            round(float((epa_valid > 0).sum() / len(epa_valid)), 4)
            if len(epa_valid) > 0 else None
        )
        yards_per_target = (
            round(float(comps["yards_gained"].sum() / targets), 2)
            if targets > 0 else None
        )
        catch_rate = round(float(rec / targets), 4) if targets > 0 else None

        def _get_fpoe(name: str) -> float | None:
            for key in (name, _abbrev(name), _norm(name)):
                d = fpoe_by_norm if key == _norm(name) else fpoe_by_name
                v = d.get(key)
                if v is not None:
                    return v
            return None

        fpoe_val = _get_fpoe(canonical)
        fpoe = round(float(fpoe_val), 2) if fpoe_val is not None else None

        gsis_id = (
            grp["receiver_player_id"].dropna().mode().iloc[0]
            if grp["receiver_player_id"].notna().any() else None
        )
        routes = routes_by_gsis.get(gsis_id, 0) if gsis_id else 0

        yards_per_route_run = (
            round(float(comps["yards_gained"].sum() / routes), 3)
            if routes > 0 else None
        )
        targets_per_route_run = (
            round(float(targets / routes), 3)
            if routes > 0 else None
        )

        air_total = grp["air_yards"].sum()
        air_yards_per_target = (
            round(float(air_total / targets), 2)
            if targets > 0 and not np.isnan(air_total) else None
        )
        team_total_air = team_air_yards.get(team, 0)
        air_yards_share_pct = (
            round(float(air_total / team_total_air), 4)
            if team_total_air > 0 and not np.isnan(air_total) else None
        )

        exp_15 = (comps["yards_gained"] >= 15).sum()
        exp_20 = int((comps["yards_gained"] >= 20).sum())
        exp_40 = int((comps["yards_gained"] >= 40).sum())
        explosive_play_rate = (
            round(float(exp_15 / rec), 4) if rec > 0 else None
        )
        longest = int(comps["yards_gained"].max()) if rec > 0 else None

        yac_total = comps["yards_after_catch"].sum()
        yac_per_rec = (
            round(float(yac_total / rec), 2)
            if rec > 0 and not np.isnan(yac_total) else None
        )
        comps_valid = comps.dropna(subset=["yards_gained", "yards_after_catch"])
        ybc_total = (comps_valid["yards_gained"] - comps_valid["yards_after_catch"]).sum()
        ybc_per_rec = (
            round(float(ybc_total / rec), 2)
            if rec > 0 and len(comps_valid) > 0 else None
        )

        norm_name = _norm(canonical)
        btkl_raw  = btkl_by_name.get(norm_name)
        if btkl_raw is None:
            alias_norm = _CANONICAL_TO_PFR_NORM.get(norm_name, "")
            if alias_norm:
                btkl_raw = btkl_by_name.get(alias_norm)
        broken_tackles = (
            int(btkl_raw) if btkl_raw is not None and not np.isnan(btkl_raw) else None
        )
        btkl_per_rec = (
            round(float(broken_tackles / rec), 3)
            if broken_tackles is not None and rec > 0 else None
        )

        # Per-game target map for stability (mirrors TE Player Overview logic)
        game_tgt_map = {
            str(gid): int(cnt)
            for gid, cnt in grp.groupby("game_id").size().items()
        }
        game_order = sorted(game_tgt_map.keys())
        stability, _ = _stability_volatility(game_tgt_map, game_order)

        rows.append({
            "player":  canonical,
            "team":    team,
            "games":   games,
            "targets": targets,
            "epa_per_target":        epa_per_target,
            "success_rate":          success_rate,
            "yards_per_target":      yards_per_target,
            "catch_rate":            catch_rate,
            "fpoe":                  fpoe,
            "yards_per_route_run":   yards_per_route_run,
            "targets_per_route_run": targets_per_route_run,
            "air_yards_per_target":  air_yards_per_target,
            "air_yards_share_pct":   air_yards_share_pct,
            "explosive_play_rate":   explosive_play_rate,
            "explosive_rec_20_plus": exp_20,
            "explosive_rec_40_plus": exp_40,
            "longest_reception":     longest,
            "yac_per_rec":           yac_per_rec,
            "ybc_per_rec":           ybc_per_rec,
            "broken_tackles":        broken_tackles,
            "btkl_per_rec":          btkl_per_rec,
            "stability":             stability,
            "efficiency_score":      None,
        })

    print(f"[{season}] Built {len(rows)} qualifying TEs — scoring …")

    df = pd.DataFrame(rows)
    df = _add_efficiency_scores(df)

    df = df.sort_values(
        ["efficiency_score", "yards_per_route_run"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (PBP_PATH, PARTICIPATION_PATH, METRICS_PATH, NFL_PATH, SLEEPER_PATH, PFR_REC_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    rows = build_season(SEASON)
    rows = [r for r in rows if r["player"] not in _EXCLUDED_PLAYERS]

    # ── Patch efficiency_score from player overview (single source of truth) ──
    overview_path = OUTPUT_DIR / f"te_player_overview_{SEASON}.json"
    if overview_path.exists():
        with open(overview_path) as _f:
            _ov = json.load(_f)
        _ov_scores: dict[str, float | None] = {
            p["player"]: p.get("efficiency_score") for p in _ov.get("players", [])
        }
        for r in rows:
            if r["player"] in _ov_scores:
                r["efficiency_score"] = _ov_scores[r["player"]]
        rows.sort(key=lambda r: r.get("efficiency_score") or 0.0, reverse=True)
        print(f"Patched efficiency_score from player overview for {len(_ov_scores)} TEs")
    else:
        print(f"WARN: {overview_path} not found — using analytics-computed efficiency_score")

    assert len({r["player"] for r in rows}) == len(rows), "Duplicate players found"
    for r in rows:
        for col in COLUMNS:
            assert col in r, f"Missing column: {col} for {r.get('player')}"
    print(f"Validation OK: {len(rows)} TEs, all columns present, no duplicates")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ordered_rows = [{c: r.get(c) for c in COLUMNS} for r in rows]

    payload: dict[str, Any] = {
        "meta": {
            "position":     "TE",
            "table":        "te_efficiency_analytics",
            "season":       str(SEASON),
            "generated_at": now,
            "columns":      COLUMNS,
        },
        "players": _clean_rows(ordered_rows),
    }

    out_path = OUTPUT_DIR / f"te_efficiency_analytics_{SEASON}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.name} ({kb:.1f} KB, {len(rows)} TEs)")
    print("TE efficiency analytics build complete.")


if __name__ == "__main__":
    main()
