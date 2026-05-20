"""
validate_position_outputs.py
─────────────────────────────
Validate the canonical output tree at output/positions/ and the related
legacy aliases.

Checks
------
1. Every position has all 4 tab files for season=2025
   (overview, efficiency, usage, stat_explorer).
2. Every file has at least one row.
3. Every row has the 7 shared identity fields populated (playerName is
   required; playerId is required when resolvable from Sleeper, warned
   otherwise).
4. Every overview row has overallScore AND tier.
5. Every efficiency row has efficiencyScore.
6. Every usage row has usageScore where possible (warned, not errored,
   when the source data does not support a usageScore).
7. Stat Explorer rows contain none of schema_config.STAT_EXPLORER_FORBIDDEN.
8. No snake_case keys appear in canonical rows.
9. All legacy aliases listed in schema_config.LEGACY_ALIASES still exist.
10. Global + per-position manifests exist.

Exit 0 = pass, exit 1 = errors.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from schema_config import (
    CANONICAL_DIR,
    CURRENT_SEASON,
    FIELD_SCHEMA,
    HISTORICAL_SEASONS,
    IDENTITY_FIELDS,
    LEGACY_ALIASES,
    OUTPUT_DIR,
    POSITIONS,
    SEASONS,
    STAT_EXPLORER_FORBIDDEN,
    TABS,
    TAB_REQUIRED_SCORE,
    canonical_path,
    position_dir,
)


_SNAKE_RE = re.compile(r"[a-z0-9]_[a-z0-9]")


def _is_camel(key: str) -> bool:
    if key.startswith("_"):
        return False
    return _SNAKE_RE.search(key) is None


def _load(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    # ── 10. Manifests exist ────────────────────────────────────────────
    gm = CANONICAL_DIR / "manifest.json"
    if not gm.exists():
        errors.append(f"Missing global manifest: {gm.relative_to(OUTPUT_DIR)}")
    for pos in POSITIONS:
        pm = position_dir(pos) / "manifest.json"
        if not pm.exists():
            errors.append(f"Missing position manifest: {pm.relative_to(OUTPUT_DIR)}")

    # ── 9. Legacy aliases ──────────────────────────────────────────────
    for name in LEGACY_ALIASES:
        if not (OUTPUT_DIR / name).exists():
            errors.append(f"Legacy alias missing: {name}")

    # ── 1. Required CURRENT_SEASON canonical files (errors if missing) ──
    canonical_payloads: dict[tuple[str, str, str], dict] = {}
    for pos in POSITIONS:
        # Load per-position manifest so we know which omissions are intentional
        pm_path = position_dir(pos) / "manifest.json"
        unavailable: dict[str, list[str]] = {}
        if pm_path.exists():
            try:
                unavailable = (json.load(open(pm_path)).get("unavailableSeasons") or {})
            except Exception:
                unavailable = {}

        for tab in TABS:
            for season in (CURRENT_SEASON, *HISTORICAL_SEASONS, "all"):
                fp = canonical_path(pos, tab, season)
                if not fp.exists():
                    is_current = season == CURRENT_SEASON
                    is_intentional_omit = season in unavailable.get(tab, [])
                    msg = (
                        f"[{pos.upper()} / {tab} / {season}] canonical file "
                        f"not produced ({fp.relative_to(OUTPUT_DIR)})"
                    )
                    if is_current and not is_intentional_omit:
                        # 2025 must exist unless explicitly marked unavailable
                        errors.append(msg + " — required for current season")
                    else:
                        warnings.append(msg)
                    continue
                try:
                    payload = _load(fp)
                except Exception as e:
                    errors.append(f"{fp.name}: failed to parse ({e})")
                    continue
                if not isinstance(payload, dict) or "rows" not in payload or "meta" not in payload:
                    errors.append(f"{fp.name}: missing meta/rows envelope")
                    continue
                canonical_payloads[(pos, tab, season)] = payload

                # 2. Files have rows
                if not payload["rows"]:
                    errors.append(f"{fp.name}: rows[] is empty")

    # ── 3. Identity-field check ────────────────────────────────────────
    for (pos, tab, season), payload in canonical_payloads.items():
        missing_pid = 0
        for row in payload["rows"]:
            for ident in IDENTITY_FIELDS:
                if ident not in row:
                    errors.append(
                        f"[{pos.upper()} / {tab} / {season}] row missing "
                        f"identity field {ident!r}: player={row.get('playerName')!r}"
                    )
                    break
            if not row.get("playerName"):
                errors.append(
                    f"[{pos.upper()} / {tab} / {season}] empty playerName"
                )
            if not row.get("playerId"):
                missing_pid += 1
        if missing_pid:
            warnings.append(
                f"[{pos.upper()} / {tab} / {season}] {missing_pid} row(s) "
                f"with no playerId (Sleeper resolver miss)"
            )

    # ── 4-6. Tab-specific required scores ─────────────────────────────
    for (pos, tab, season), payload in canonical_payloads.items():
        required = TAB_REQUIRED_SCORE.get(tab)
        if not required:
            continue
        for field in required:
            missing = sum(
                1 for r in payload["rows"]
                if r.get(field) is None
            )
            if missing == 0:
                continue
            if tab == "usage":
                warnings.append(
                    f"[{pos.upper()} / usage / {season}] {missing} row(s) "
                    f"with null {field}"
                )
            else:
                errors.append(
                    f"[{pos.upper()} / {tab} / {season}] {missing} row(s) "
                    f"missing required {field}"
                )

    # ── 7. Stat Explorer must be raw ──────────────────────────────────
    forbidden = set(STAT_EXPLORER_FORBIDDEN)
    for (pos, tab, season), payload in canonical_payloads.items():
        if tab != "stat_explorer":
            continue
        leaked: set[str] = set()
        for row in payload["rows"]:
            leaked |= forbidden & set(row.keys())
        if leaked:
            errors.append(
                f"[{pos.upper()} / stat_explorer / {season}] modeled fields "
                f"leaked into raw layer: {sorted(leaked)}"
            )

    # ── 8. camelCase enforcement ──────────────────────────────────────
    for (pos, tab, season), payload in canonical_payloads.items():
        bad: set[str] = set()
        for row in payload["rows"]:
            for k in row.keys():
                if not _is_camel(k):
                    bad.add(k)
        if bad:
            errors.append(
                f"[{pos.upper()} / {tab} / {season}] non-camelCase row "
                f"keys: {sorted(bad)}"
            )

        # Also confirm row columns match declared schema exactly
        template = FIELD_SCHEMA[pos][tab]
        for row in payload["rows"]:
            extra   = set(row.keys()) - set(template)
            missing = set(template) - set(row.keys())
            if extra:
                errors.append(
                    f"[{pos.upper()} / {tab} / {season}] row has extra "
                    f"fields: {sorted(extra)}"
                )
                break
            if missing:
                errors.append(
                    f"[{pos.upper()} / {tab} / {season}] row missing schema "
                    f"fields: {sorted(missing)}"
                )
                break

    # ── Report ─────────────────────────────────────────────────────────
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        print(f"\nValidator FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1

    print(
        f"\nValidator PASSED: {len(canonical_payloads)} canonical files OK"
        f" ({len(warnings)} warning(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
