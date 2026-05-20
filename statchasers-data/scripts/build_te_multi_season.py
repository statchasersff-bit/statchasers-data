"""
build_te_multi_season.py
────────────────────────
Re-runs the TE overview / advanced / efficiency builders for prior seasons
(2023, 2024) so the TE position has the same multi-season coverage as WR.

The existing build_te_*.py scripts already accept the season at runtime via
their module-level `SEASON` constant — we simply patch it, run main(), then
move on to the next season.

  python scripts/build_te_multi_season.py            # default: [2023, 2024]
  python scripts/build_te_multi_season.py 2022 2023 2024

The 2025 outputs are NOT touched by this script — the canonical 2025 build
remains owned by the original builders.  Run those separately:

  python scripts/build_te_player_overview.py
  python scripts/build_te_advanced_stats.py
  python scripts/build_te_efficiency_analytics.py

Notes
-----
- player_metrics.json is 2025-only, so fpoe / careerArc / stability are
  null for 2023 and 2024 (acceptable — the data source isn't available).
- The score-alignment guarantee (efficiency_score copied from overview) is
  preserved per season because each season is built end-to-end.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
BUILDERS = [
    "build_te_player_overview",      # writes te_player_overview_{SEASON}.json
    "build_te_advanced_stats",       # writes te_advanced_stats_{SEASON}.json
    "build_te_efficiency_analytics", # writes te_efficiency_analytics_{SEASON}.json
]


def _load(builder_name: str):
    spec = importlib.util.spec_from_file_location(
        builder_name, SCRIPTS_DIR / f"{builder_name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not locate {builder_name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_for(season: int) -> None:
    print(f"\n══════════════════════════════════════════════════")
    print(f" TE multi-season build — season {season}")
    print(f"══════════════════════════════════════════════════")
    for name in BUILDERS:
        print(f"\n─── {name} [{season}] ───")
        mod = _load(name)
        mod.SEASON = season
        try:
            mod.main()
        except SystemExit as e:
            if e.code not in (0, None):
                print(
                    f"  WARN: {name} exited with code {e.code} for season {season}",
                    file=sys.stderr,
                )
        except Exception as e:
            # Per-season failures shouldn't kill the whole sweep — log and continue.
            print(
                f"  WARN: {name} failed for season {season}: {e}",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if argv:
        try:
            seasons = [int(s) for s in argv]
        except ValueError:
            print("Usage: build_te_multi_season.py [SEASON ...]", file=sys.stderr)
            sys.exit(2)
    else:
        seasons = [2023, 2024]

    for s in seasons:
        run_for(s)

    print("\nTE multi-season build complete.")


if __name__ == "__main__":
    main()
