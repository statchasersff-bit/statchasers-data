"""
build_qb_player_overview.py
───────────────────────────
Builds the QB Player Overview dataset for the StatChasers QB explorer.

Produces:
  output/qb_player_overview_2025.json   — 2025 season
  output/qb_player_overview.json        — alias for 2025

Source: derived from the already-computed
  output/performance_analytics_latest.json   (player_metrics intelligence layer)
  output/stat_explorer_qb_2025.json          (raw counting + pressure data)

This script is intentionally lightweight — all QB-specific metrics already
live in player_metrics / performance_analytics; we just project the QB rows
into the standard Player Overview shape and assign composite scores.

Composite scoring (QB-tuned)
----------------------------
  role_score        — dropbacks/gm, rush att/gm, snap share, RZ att
  efficiency_score  — EPA/play, success rate, YPA, FPOE, TD over expected, passer rating
  overall_score     — role 0.45, efficiency 0.40, stability 0.15
  qb_tier           — derived from overall_score

QB Tier thresholds (overall_score → tier):
  ≥ 80  → "Elite"
  ≥ 65  → "QB1"
  ≥ 50  → "Mid-Tier"
  ≥ 35  → "Backup-Caliber"
  else  → "Bench"

This is the **source of truth** for QB role_score / efficiency_score /
overall_score / qb_tier / career_arc / exp_tier / stability.
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

SEASON      = 2025
MIN_DROPBACKS_PER_GAME = 10           # filter clipboard-only QBs

COLUMNS: list[str] = [
    "rank", "player", "team", "age", "games",
    "snap_pct",
    "dropbacks_per_gm", "rush_att_per_gm", "rz_att",
    "ypa", "completion_pct", "passer_rating",
    "epa_per_play", "success_rate", "fpoe", "td_over_expected",
    "deep_attempt_rate", "play_action_rate", "scramble_rate",
    "stability", "volatility",
    "career_arc", "exp_tier",
    "role_score", "efficiency_score", "overall_score",
    "qb_tier",
]


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


def _qb_tier(overall: float | None) -> str | None:
    if overall is None:
        return None
    if overall >= 80: return "Elite"
    if overall >= 65: return "QB1"
    if overall >= 50: return "Mid-Tier"
    if overall >= 35: return "Backup-Caliber"
    return "Bench"


def main() -> None:
    for p in (PERF_PATH, STAT_EXPLORER):
        if not p.exists():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    with open(PERF_PATH) as f:
        perf = json.load(f)
    with open(STAT_EXPLORER) as f:
        stat_explorer = json.load(f)

    qbs = [p for p in perf["players"] if p.get("pos") == "QB"]
    print(f"Loaded {len(qbs)} QBs from performance_analytics.")

    # Build (player, team) → stat_explorer row for raw counting stats
    se_index: dict[tuple[str, str], dict] = {
        (r["player"], r["team"]): r for r in stat_explorer.get("rows", [])
    }
    # Also try player-only lookup as fallback
    se_player_index: dict[str, dict] = {r["player"]: r for r in stat_explorer.get("rows", [])}

    raw_rows: list[dict] = []
    for q in qbs:
        # Apply min volume filter using dropbacks/game
        if (q.get("dropbacksPerGame") or 0) < MIN_DROPBACKS_PER_GAME:
            continue

        se = se_index.get((q["player"], q.get("team", "")), se_player_index.get(q["player"], {}))

        # Compute completion_pct
        comp = se.get("comp")
        att  = se.get("att")
        completion_pct = _round((comp / att * 100), 1) if comp and att else None

        raw_rows.append({
            "player":             q["player"],
            "team":               q.get("team"),
            "age":                q.get("age"),
            "games":              q.get("games"),
            "snap_pct":           _round((q.get("snapShare") or 0) / 100.0, 4) if q.get("snapShare") is not None else None,
            "dropbacks_per_gm":   _round(q.get("dropbacksPerGame"), 2),
            "rush_att_per_gm":    _round(q.get("rushAttPerGame"), 2),
            "rz_att":             q.get("rzAtt"),
            "ypa":                _round(q.get("ypa"), 2),
            "completion_pct":     completion_pct,
            "passer_rating":      _round(se.get("passer_rating"), 1),
            "epa_per_play":       _round(q.get("epaPlay"), 3),
            "success_rate":       _round(q.get("successRate"), 2),
            "fpoe":               _round(q.get("fpoe"), 2),
            "td_over_expected":   _round(q.get("tdOverExpected"), 2),
            "deep_attempt_rate":  _round(q.get("deepAttemptRate"), 2),
            "play_action_rate":   _round(q.get("playActionRate"), 2),
            "scramble_rate":      _round(q.get("scrambleRate"), 2),
            "stability":          _round(q.get("roleStability"), 2),
            "volatility":         _round(q.get("usageVolatility"), 2),
            "career_arc":         q.get("careerArc"),
            "exp_tier":           q.get("experienceTier"),
        })

    if not raw_rows:
        print("ERROR: no QBs qualified.", file=sys.stderr)
        sys.exit(1)

    # ── Composite scores via percentile pools ──────────────────────────────
    def col(key: str) -> list:
        return [r.get(key) for r in raw_rows]

    arr_db_pg   = col("dropbacks_per_gm")
    arr_ru_pg   = col("rush_att_per_gm")
    arr_snap    = col("snap_pct")
    arr_rz      = col("rz_att")
    arr_ypa     = col("ypa")
    arr_epa     = col("epa_per_play")
    arr_succ    = col("success_rate")
    arr_fpoe    = col("fpoe")
    arr_tdoe    = col("td_over_expected")
    arr_rating  = col("passer_rating")
    arr_stab    = col("stability")
    arr_vol     = col("volatility")

    for r in raw_rows:
        p_db   = _pct(r["dropbacks_per_gm"], arr_db_pg)
        p_ru   = _pct(r["rush_att_per_gm"],  arr_ru_pg)
        p_snap = _pct(r["snap_pct"],         arr_snap)
        p_rz   = _pct(r["rz_att"],           arr_rz)
        p_ypa  = _pct(r["ypa"],              arr_ypa)
        p_epa  = _pct(r["epa_per_play"],     arr_epa)
        p_succ = _pct(r["success_rate"],     arr_succ)
        p_fpoe = _pct(r["fpoe"],             arr_fpoe)
        p_tdoe = _pct(r["td_over_expected"], arr_tdoe)
        p_rate = _pct(r["passer_rating"],    arr_rating)
        p_stab = _pct(r["stability"],        arr_stab)
        p_vinv = 100.0 - _pct(r["volatility"], arr_vol)

        role_score = round(
            min(100.0, max(0.0,
                p_db   * 0.45
                + p_snap * 0.20
                + p_ru   * 0.15
                + p_rz   * 0.20,
            )),
            1,
        )
        efficiency_score = round(
            min(100.0, max(0.0,
                p_epa  * 0.25
                + p_succ * 0.20
                + p_ypa  * 0.15
                + p_fpoe * 0.15
                + p_tdoe * 0.10
                + p_rate * 0.15,
            )),
            1,
        )
        stab_block = p_stab * 0.60 + p_vinv * 0.40
        overall_score = round(
            min(100.0, max(0.0,
                role_score      * 0.45
                + efficiency_score * 0.40
                + stab_block       * 0.15,
            )),
            1,
        )
        r["role_score"]       = role_score
        r["efficiency_score"] = efficiency_score
        r["overall_score"]    = overall_score
        r["qb_tier"]          = _qb_tier(overall_score)

    raw_rows.sort(key=lambda r: -r["overall_score"])
    for i, r in enumerate(raw_rows, 1):
        r["rank"] = i

    final = [{c: r.get(c) for c in COLUMNS} for r in raw_rows]
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season":     SEASON,
        "week":       None,
        "table":      "qb_player_overview",
        "columns":    COLUMNS,
        "rows":       final,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path  = OUTPUT_DIR / f"qb_player_overview_{SEASON}.json"
    alias     = OUTPUT_DIR / "qb_player_overview.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    shutil.copy(out_path, alias)

    print(f"Wrote {out_path} ({len(final)} QBs).")
    print(f"Wrote {alias} (alias).")


if __name__ == "__main__":
    main()
