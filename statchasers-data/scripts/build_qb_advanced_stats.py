"""
build_qb_advanced_stats.py
──────────────────────────
Builds the QB Advanced Stats dataset for the StatChasers QB explorer.

Output:
  output/qb_advanced_stats_2025.json
  output/qb_advanced_stats.json   — alias for 2025

Source:
  output/stat_explorer_qb_2025.json         (raw counting + pressure)
  output/qb_player_overview_2025.json       (universe filter)

This tab is volume / pressure focused — raw counting stats plus PFR
pressure data (blitzes, hurries, knockdowns, pressures, poor throws,
pocket time, etc.).  No modeled scores or tiers.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

STAT_EXPLORER     = OUTPUT_DIR / "stat_explorer_qb_2025.json"
QB_OVERVIEW_PATH  = OUTPUT_DIR / "qb_player_overview_2025.json"

SEASON = 2025

COLUMNS: list[str] = [
    "rank", "player", "team", "games",
    "completions", "attempts", "completion_pct",
    "yards", "ypa", "air_yards", "air_yards_per_att",
    "touchdowns", "interceptions",
    "pass_10_plus", "pass_20_plus", "pass_30_plus", "pass_40_plus", "pass_50_plus",
    "sacks",
    "rush_attempts", "rush_yards", "rush_touchdowns",
    "rz_att",
    "blitzes", "hurries", "knockdowns", "pressures",
    "poor_throws", "drops", "pocket_time",
]


def _round(v: Any, n: int = 2) -> Any:
    if v is None:
        return None
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    if np.isnan(f):
        return None
    return round(f, n)


def main() -> None:
    for p in (STAT_EXPLORER, QB_OVERVIEW_PATH):
        if not p.exists():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    with open(STAT_EXPLORER) as f:
        se = json.load(f)
    with open(QB_OVERVIEW_PATH) as f:
        ov = json.load(f)

    ov_players: set[str] = {r["player"] for r in ov.get("rows", [])}

    rows: list[dict] = []
    for s in se.get("rows", []):
        if s["player"] not in ov_players:
            continue
        comp = s.get("comp")
        att  = s.get("att")
        completion_pct = _round((comp / att * 100), 1) if comp and att else None

        rows.append({
            "player":              s["player"],
            "team":                s.get("team"),
            "games":               s.get("gp"),
            "completions":         comp,
            "attempts":            att,
            "completion_pct":      completion_pct,
            "yards":               s.get("yds"),
            "ypa":                 _round(s.get("ypa"), 2),
            "air_yards":           s.get("air_yards"),
            "air_yards_per_att":   _round(s.get("air_yards_per_att"), 2),
            "touchdowns":          s.get("td"),
            "interceptions":       s.get("int"),
            "pass_10_plus":        s.get("pass_10_plus"),
            "pass_20_plus":        s.get("pass_20_plus"),
            "pass_30_plus":        s.get("pass_30_plus"),
            "pass_40_plus":        s.get("pass_40_plus"),
            "pass_50_plus":        s.get("pass_50_plus"),
            "sacks":               s.get("sacks"),
            "rush_attempts":       s.get("rush_att"),
            "rush_yards":          s.get("rush_yds"),
            "rush_touchdowns":     s.get("rush_td"),
            "rz_att":              s.get("rz_att"),
            "blitzes":             s.get("blitzes"),
            "hurries":             s.get("hurries"),
            "knockdowns":          s.get("knockdowns"),
            "pressures":           s.get("pressures"),
            "poor_throws":         s.get("poor_throws"),
            "drops":               s.get("drops"),
            "pocket_time":         _round(s.get("pocket_time"), 2),
        })

    rows.sort(key=lambda r: -(r.get("yards") or 0))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    final = [{c: r.get(c) for c in COLUMNS} for r in rows]
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season":     SEASON,
        "week":       None,
        "table":      "qb_advanced_stats",
        "columns":    COLUMNS,
        "rows":       final,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"qb_advanced_stats_{SEASON}.json"
    alias    = OUTPUT_DIR / "qb_advanced_stats.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    shutil.copy(out_path, alias)
    print(f"Wrote {out_path} ({len(final)} QBs).")
    print(f"Wrote {alias} (alias).")


if __name__ == "__main__":
    main()
