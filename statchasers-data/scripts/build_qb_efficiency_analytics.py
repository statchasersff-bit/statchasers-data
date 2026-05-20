"""
build_qb_efficiency_analytics.py
─────────────────────────────────
Builds the QB Efficiency Analytics dataset for the StatChasers Research tab.

Answers: How effective is this QB on a per-dropback and per-attempt basis?

Output:
  output/qb_efficiency_analytics_2025.json
  output/qb_efficiency_analytics.json     — alias for 2025

Source:
  output/performance_analytics_latest.json  (intelligence layer)
  output/stat_explorer_qb_2025.json         (counting + pressure)
  output/qb_player_overview_2025.json       (efficiency_score source of truth)

Score alignment
---------------
`efficiency_score` is *copied verbatim* from qb_player_overview, never
recomputed.  This guarantees cross-tab alignment with the Overview tab.
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

PERF_PATH         = OUTPUT_DIR / "performance_analytics_latest.json"
STAT_EXPLORER     = OUTPUT_DIR / "stat_explorer_qb_2025.json"
QB_OVERVIEW_PATH  = OUTPUT_DIR / "qb_player_overview_2025.json"

SEASON = 2025

COLUMNS: list[str] = [
    "player", "team", "games",
    "attempts", "completions", "completion_pct",
    "epa_per_play", "success_rate", "ypa", "air_yards_per_att",
    "passer_rating", "fpoe", "td_over_expected",
    "explosive_play_rate", "deep_attempt_rate", "deep_target_rate",
    "play_action_rate", "scramble_rate", "designed_rush_rate",
    "relative_efficiency", "season_avg_epa",
    "efficiency_score",
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
    for p in (PERF_PATH, STAT_EXPLORER, QB_OVERVIEW_PATH):
        if not p.exists():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    with open(PERF_PATH) as f:
        perf = json.load(f)
    with open(STAT_EXPLORER) as f:
        se = json.load(f)
    with open(QB_OVERVIEW_PATH) as f:
        ov = json.load(f)

    qbs = [p for p in perf["players"] if p.get("pos") == "QB"]

    se_index: dict[str, dict] = {r["player"]: r for r in se.get("rows", [])}
    ov_eff_score: dict[str, float] = {
        r["player"]: r.get("efficiency_score") for r in ov.get("rows", [])
    }
    ov_players: set[str] = set(ov_eff_score.keys())

    rows: list[dict] = []
    for q in qbs:
        # Only include QBs that qualified for the Overview tab — keeps
        # the two tabs aligned on universe and threshold.
        if q["player"] not in ov_players:
            continue

        s = se_index.get(q["player"], {})
        comp = s.get("comp")
        att  = s.get("att")
        completion_pct = _round((comp / att * 100), 1) if comp and att else None

        rows.append({
            "player":              q["player"],
            "team":                q.get("team"),
            "games":               q.get("games"),
            "attempts":            att,
            "completions":         comp,
            "completion_pct":      completion_pct,
            "epa_per_play":        _round(q.get("epaPlay"), 3),
            "success_rate":        _round(q.get("successRate"), 2),
            "ypa":                 _round(q.get("ypa"), 2),
            "air_yards_per_att":   _round(q.get("airYardsPerAtt"), 2),
            "passer_rating":       _round(s.get("passer_rating"), 1),
            "fpoe":                _round(q.get("fpoe"), 2),
            "td_over_expected":    _round(q.get("tdOverExpected"), 2),
            "explosive_play_rate": _round(q.get("explosivePlayRate"), 2),
            "deep_attempt_rate":   _round(q.get("deepAttemptRate"), 2),
            "deep_target_rate":    _round(q.get("deepTargetRate"), 2),
            "play_action_rate":    _round(q.get("playActionRate"), 2),
            "scramble_rate":       _round(q.get("scrambleRate"), 2),
            "designed_rush_rate":  _round(q.get("designedRushRate"), 2),
            "relative_efficiency": _round(q.get("relativeEfficiency"), 3),
            "season_avg_epa":      _round(q.get("seasonAvgEpa"), 3),
            "efficiency_score":    ov_eff_score.get(q["player"]),
        })

    rows.sort(key=lambda r: -(r.get("efficiency_score") or 0))

    final = [{c: r.get(c) for c in COLUMNS} for r in rows]
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season":     SEASON,
        "week":       None,
        "table":      "qb_efficiency_analytics",
        "columns":    COLUMNS,
        "rows":       final,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"qb_efficiency_analytics_{SEASON}.json"
    alias    = OUTPUT_DIR / "qb_efficiency_analytics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    shutil.copy(out_path, alias)
    print(f"Wrote {out_path} ({len(final)} QBs).")
    print(f"Wrote {alias} (alias).")


if __name__ == "__main__":
    main()
