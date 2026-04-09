#!/usr/bin/env python3
"""
validate_score_alignment.py

Verifies that efficiency_score and usage_score values in the analytics/trends
JSON outputs exactly match the player overview JSON (source of truth).

Usage:
  python scripts/validate_score_alignment.py

Exit code 0 = all checks pass.
Exit code 1 = one or more checks failed.

Name-alias map:
  WR analytics uses Sleeper full names (e.g. "Amon-Ra St. Brown") while
  WR overview uses nflverse abbreviations (e.g. "A.St. Brown").  Matching
  falls back to (team, last_name) when exact names disagree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "output"
SEASON = 2025

failures: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(fname: str) -> list[dict]:
    path = OUTPUT / fname
    if not path.exists():
        failures.append(f"MISSING FILE: {path}")
        return []
    data = json.load(open(path))
    if isinstance(data, list):
        return data
    return data.get("players") or data.get("rows", [])


def _overview_lookup(ov_rows: list[dict], field: str) -> tuple[dict, dict]:
    """Return (exact_name → value, (team, last_name) → value) dicts."""
    exact = {r["player"]: r.get(field) for r in ov_rows}
    team_last = {
        (r["team"], r["player"].split()[-1]): r.get(field) for r in ov_rows
    }
    return exact, team_last


def check_scores(
    label: str,
    analytic_rows: list[dict],
    overview_rows: list[dict],
    field: str,
) -> None:
    exact, team_last = _overview_lookup(overview_rows, field)
    mismatches: list[tuple] = []
    for r in analytic_rows:
        name = r["player"]
        actual = r.get(field)
        expected = exact.get(name)
        if expected is None:
            key = (r.get("team"), name.split()[-1])
            expected = team_last.get(key)
        if expected is not None and actual != expected:
            mismatches.append((name, actual, expected))

    if mismatches:
        failures.append(
            f"{label} — {len(mismatches)} {field} mismatches "
            f"(e.g. {mismatches[0][0]}: analytic={mismatches[0][1]} "
            f"overview={mismatches[0][2]})"
        )
    else:
        print(f"  OK  {label}: 0 {field} mismatches")


def check_top10(
    label: str,
    analytic_rows: list[dict],
    overview_rows: list[dict],
    field: str,
) -> None:
    """Check that the top-10 set (by name) is the same in both outputs.
    Uses score-based cutoff to handle ties at the boundary correctly."""
    def _top_set(rows: list[dict]) -> set[str]:
        ranked = sorted(rows, key=lambda r: -(r.get(field) or 0))
        if len(ranked) < 10:
            return {r["player"].split()[-1] for r in ranked}
        cutoff = ranked[9].get(field) or 0
        return {r["player"].split()[-1] for r in ranked if (r.get(field) or 0) >= cutoff}

    a_top  = _top_set(analytic_rows)
    ov_top = _top_set(overview_rows)
    diff = a_top.symmetric_difference(ov_top)
    if diff:
        failures.append(
            f"{label} top-10 {field} mismatch — diff last-names: {sorted(diff)}"
        )
    else:
        print(f"  OK  {label}: top-10 {field} match")


def check_no_nan(label: str, rows: list[dict]) -> None:
    import math
    nans = [
        (r.get("player", "?"), k)
        for r in rows
        for k, v in r.items()
        if isinstance(v, float) and math.isnan(v)
    ]
    if nans:
        failures.append(f"{label}: {len(nans)} NaN values (e.g. {nans[0]})")
    else:
        print(f"  OK  {label}: 0 NaN values")


# ---------------------------------------------------------------------------
# Run checks
# ---------------------------------------------------------------------------

print("=== StatChasers score alignment validation ===\n")

te_eff  = load(f"te_efficiency_analytics_{SEASON}.json")
wr_eff  = load(f"wr_efficiency_analytics_{SEASON}.json")
rb_eff  = load("rb_efficiency_analytics.json")
te_trnd = load(f"te_trends_{SEASON}.json")
wr_trnd = load(f"wr_trends_{SEASON}.json")

te_ov   = load(f"te_player_overview_{SEASON}.json")
wr_ov   = load(f"wr_player_overview_{SEASON}.json")
rb_ov   = load("rb_player_overview.json")

print("--- NaN checks ---")
check_no_nan("TE efficiency analytics", te_eff)
check_no_nan("WR efficiency analytics", wr_eff)
check_no_nan("RB efficiency analytics", rb_eff)
check_no_nan("TE trends",               te_trnd)
check_no_nan("WR trends",               wr_trnd)

print("\n--- efficiency_score exact match ---")
check_scores("TE efficiency analytics", te_eff,  te_ov, "efficiency_score")
check_scores("WR efficiency analytics", wr_eff,  wr_ov, "efficiency_score")
check_scores("RB efficiency analytics", rb_eff,  rb_ov, "efficiency_score")

print("\n--- role_score exact match (trends vs overview) ---")
check_scores("TE trends", te_trnd, te_ov, "role_score")
check_scores("WR trends", wr_trnd, wr_ov, "role_score")

print("\n--- top-10 ranking alignment ---")
check_top10("TE efficiency analytics", te_eff,  te_ov,  "efficiency_score")
check_top10("WR efficiency analytics", wr_eff,  wr_ov,  "efficiency_score")
check_top10("RB efficiency analytics", rb_eff,  rb_ov,  "efficiency_score")
check_top10("TE trends",               te_trnd, te_ov,  "role_score")
check_top10("WR trends",               wr_trnd, wr_ov,  "role_score")

print()
if failures:
    print(f"FAILED — {len(failures)} issue(s):")
    for f in failures:
        print(f"  ✗ {f}")
    sys.exit(1)
else:
    print("All checks PASSED.")
    sys.exit(0)
