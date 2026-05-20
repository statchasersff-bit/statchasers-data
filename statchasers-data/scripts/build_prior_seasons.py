"""
build_prior_seasons.py
──────────────────────
Re-runs the position builders for prior seasons (default: 2023, 2024)
so the canonical output tree has the same coverage as 2025.

Each prior-season run patches the module-level SEASON / SEASONS constant
of the legacy builder via importlib, runs main(), then restores any
state that other builders depend on.

  python scripts/build_prior_seasons.py                  # default: [2023, 2024]
  python scripts/build_prior_seasons.py 2022 2023 2024

Coverage produced (per season)
------------------------------
  WR efficiency, WR trends                          (from PBP/participation)
  TE overview, advanced, efficiency, trends         (from PBP/participation)
  RB overview, efficiency, usage_role               (from PBP — unsuffixed
                                                    outputs are renamed to
                                                    season-suffixed afterwards)
  QB overview, efficiency, advanced                 (require a per-season
                                                    perf_analytics_latest.json,
                                                    rebuilt below)

QB prerequisites
----------------
The QB builders depend on `performance_analytics_latest.json` (produced by
`build_performance_analytics.py` from `player_metrics.json`).  Both files
are 2025-focused.  For each prior season we:

  1. Patch compute_player_metrics.CURRENT_SEASON → run         (overwrites
     data/processed/player_metrics.json with prior-season dashboard data)
  2. Patch build_performance_analytics.CURRENT_SEASON →
     output paths → run                                         (overwrites
     output/performance_analytics_latest.json with prior-season payload)
  3. Run the 3 QB builders against the now-prior-season state

After all prior seasons are processed we restore the 2025-state by
re-running compute_player_metrics, build_performance_analytics, and the
RB/QB builders with their original constants.

Notes
-----
- Per-season failures are logged and don't abort the sweep.
- WR efficiency rewrites a season-less alias (wr_efficiency_analytics.json)
  on every run; the alias is restored to 2025 at the end.
- RB builders write unsuffixed filenames (rb_player_overview.json etc.);
  the wrapper renames them to suffixed names after each prior-season run
  and restores the unsuffixed 2025 aliases at the end.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT        = SCRIPTS_DIR.parent
OUTPUT_DIR  = ROOT / "output"
PROCESSED   = ROOT / "data" / "processed"


# ── PBP-only WR/TE/RB builders (cheap to run per season) ──────────────────

WR_TE_BUILDERS: list[tuple[str, str, str]] = [
    ("build_wr_efficiency_analytics", "SEASONS", "list"),
    ("build_wr_trends",               "SEASON",  "scalar"),
    ("build_te_player_overview",      "SEASON",  "scalar"),
    ("build_te_advanced_stats",       "SEASON",  "scalar"),
    ("build_te_efficiency_analytics", "SEASON",  "scalar"),
    ("build_te_trends",               "SEASON",  "scalar"),
]

# RB builders write unsuffixed outputs — the wrapper renames them after.
RB_BUILDERS: list[tuple[str, str, str, str]] = [
    # (module, attribute, kind, source_filename)
    ("build_rb_player_overview",      "SEASON", "scalar", "rb_player_overview.json"),
    ("build_rb_efficiency_analytics", "SEASON", "scalar", "rb_efficiency_analytics.json"),
    ("build_rb_usage_role",           "SEASON", "scalar", "rb_usage_role.json"),
]

# QB builders read perf_analytics_latest.json + per-season stat_explorer_qb.
# Each carries module-level path constants we patch.
QB_BUILDERS: list[str] = [
    "build_qb_player_overview",
    "build_qb_efficiency_analytics",
    "build_qb_advanced_stats",
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


def _safe_run(label: str, fn) -> None:
    try:
        fn()
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"  WARN: {label} exited with code {e.code}", file=sys.stderr)
    except Exception as e:
        print(f"  WARN: {label} failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Per-season runners
# ---------------------------------------------------------------------------

def run_wr_te(season: int) -> None:
    for name, attr, kind in WR_TE_BUILDERS:
        print(f"\n─── {name} [{season}] ───")
        mod = _load(name)
        if kind == "scalar":
            setattr(mod, attr, season)
        else:
            setattr(mod, attr, [season])
        _safe_run(f"{name}[{season}]", mod.main)


def run_rb(season: int) -> None:
    """Run RB builders for a season then rename their unsuffixed outputs
    to season-suffixed filenames.  Builders share state via the unsuffixed
    overview output (efficiency/usage_role read it for score patching),
    so all three must run before renaming.
    """
    for name, attr, kind, _src in RB_BUILDERS:
        print(f"\n─── {name} [{season}] ───")
        mod = _load(name)
        setattr(mod, attr, season)
        _safe_run(f"{name}[{season}]", mod.main)

    # Move unsuffixed outputs to suffixed
    for _, _, _, src in RB_BUILDERS:
        src_path = OUTPUT_DIR / src
        if not src_path.exists():
            continue
        dst_name = src.replace(".json", f"_{season}.json")
        dst_path = OUTPUT_DIR / dst_name
        shutil.copy(src_path, dst_path)
        print(f"  copied {src} → {dst_name}")


def run_perf_pipeline(season: int) -> None:
    """Rebuild player_metrics.json + perf_analytics_latest.json for `season`.

    These overwrite the 2025-state files temporarily — the wrapper restores
    them at the end of the run by repeating this call with season=2025.
    """
    print(f"\n══ perf-pipeline backfill — season {season} ══")

    print(f"\n─── compute_player_metrics [{season}] ───")
    mod = _load("compute_player_metrics")
    mod.CURRENT_SEASON = season
    _safe_run(f"compute_player_metrics[{season}]", mod.main)

    print(f"\n─── build_performance_analytics [{season}] ───")
    mod = _load("build_performance_analytics")
    mod.CURRENT_SEASON = season
    # Point SEASON_OUTPUT to the per-season filename; LATEST_OUTPUT keeps
    # the rolling name because QB builders read perf_analytics_latest.json.
    import os as _os
    mod.SEASON_OUTPUT = _os.path.join(str(OUTPUT_DIR), f"performance_analytics_{season}.json")
    _safe_run(f"build_performance_analytics[{season}]", mod.main)


def run_qb(season: int) -> None:
    """Run the 3 derived QB builders + QB trends with patched SEASON +
    per-season stat_explorer/overview paths.  Assumes
    perf_analytics_latest.json already reflects this season
    (call run_perf_pipeline first).
    """
    for name in QB_BUILDERS:
        print(f"\n─── {name} [{season}] ───")
        mod = _load(name)
        mod.SEASON = season

        # Repoint module-level path constants where present
        stat_explorer_name = f"stat_explorer_qb_{season}.json"
        if hasattr(mod, "STAT_EXPLORER"):
            mod.STAT_EXPLORER = OUTPUT_DIR / stat_explorer_name
        if hasattr(mod, "QB_OVERVIEW_PATH"):
            mod.QB_OVERVIEW_PATH = OUTPUT_DIR / f"qb_player_overview_{season}.json"

        _safe_run(f"{name}[{season}]", mod.main)

    # QB trends — patch TREND_SEASONS + OUTPUT_PATH for season-suffixed write
    print(f"\n─── build_qb_trends [{season}] ───")
    mod = _load("build_qb_trends")
    mod.TREND_SEASONS = [season]
    mod.PIPELINE_YEAR = season
    # Season-suffix the output for historical; 2025 keeps its unsuffixed name
    if season != 2025:
        mod.OUTPUT_PATH = OUTPUT_DIR / f"qb_trends_{season}.json"
    _safe_run(f"build_qb_trends[{season}]", mod.main)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if argv:
        try:
            seasons = [int(s) for s in argv]
        except ValueError:
            print("Usage: build_prior_seasons.py [SEASON ...]", file=sys.stderr)
            sys.exit(2)
    else:
        seasons = [2023, 2024]

    # ── For each prior season ──────────────────────────────────────────────
    for s in seasons:
        print(f"\n══════════════════════════════════════════════════")
        print(f" Prior-season build — {s}")
        print(f"══════════════════════════════════════════════════")
        # Cheap PBP-derived builders first
        run_wr_te(s)
        run_rb(s)
        # Then the QB pipeline (needs perf_analytics rebuild)
        run_perf_pipeline(s)
        run_qb(s)

    # ── Restore 2025 canonical state ──────────────────────────────────────
    print(f"\n══ Restoring 2025 canonical state ══")
    run_perf_pipeline(2025)

    # Restore RB unsuffixed aliases (rb_player_overview.json etc.) by
    # re-running the RB builders with SEASON=2025.  Skip the rename step
    # so they keep their canonical unsuffixed names.
    for name, attr, kind, _src in RB_BUILDERS:
        print(f"\n─── {name} [2025 restore] ───")
        mod = _load(name)
        setattr(mod, attr, 2025)
        _safe_run(f"{name}[2025]", mod.main)
        # also drop a season-suffixed copy so consumers can read either
        src_path = OUTPUT_DIR / _src
        dst_name = _src.replace(".json", "_2025.json")
        if src_path.exists():
            shutil.copy(src_path, OUTPUT_DIR / dst_name)

    # Restore WR efficiency alias (it writes wr_efficiency_analytics.json
    # to the last season run; we want 2025 in the alias)
    print(f"\n─── build_wr_efficiency_analytics [2025 restore] ───")
    mod = _load("build_wr_efficiency_analytics")
    mod.SEASONS = [2025]
    _safe_run("build_wr_efficiency_analytics[2025]", mod.main)

    # Restore QB 2025 builds
    print(f"\n─── QB [2025 restore] ───")
    run_qb(2025)

    print("\nPrior-season build complete.")


if __name__ == "__main__":
    main()
