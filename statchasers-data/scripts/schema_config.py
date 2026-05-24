"""
schema_config.py
────────────────
Single source of truth for the canonical StatChasers output structure.

Canonical tree
--------------
output/positions/<position>/<tab>_<season>.json

  position ∈ {"qb", "rb", "wr", "te"}
  tab      ∈ {"overview", "efficiency", "usage", "stat_explorer"}
  season   ∈ {"2025", "all"}   (per-season + cross-season union)

Conventions
-----------
- Every row uses **camelCase** keys.
- Every row contains the shared identity fields:
    playerId, playerName, position, team, age, season, games
- Overview is the *source of truth* for: roleScore, efficiencyScore,
  overallScore, tier, careerArc, experienceTier (a.k.a. expTier),
  and stabilityScore.  Other tabs MAY include these only by copying
  from Overview at build time.
- Stat Explorer files contain raw counting / near-raw rate stats only.
  Modeled fields (scores, tiers, archetypes, careerArc, roleTrend,
  experienceTier, stability) MUST NOT appear in stat_explorer rows.
- Legacy snake_case filenames (e.g. wr_player_overview_2025.json) are
  preserved as aliases so existing consumers keep working.

Source-of-truth file
--------------------
Player Overview for each (position, season) — written by the legacy
builders.  Canonical Overview projects from there; canonical Efficiency
and canonical Usage *copy* the score fields verbatim where applicable.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).resolve().parent.parent
OUTPUT_DIR    = ROOT / "output"
CANONICAL_DIR = OUTPUT_DIR / "positions"
SLEEPER_PATH  = ROOT / "data" / "raw" / "sleeper_players.json"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

POSITIONS:        tuple[str, ...] = ("qb", "rb", "wr", "te")
TABS:             tuple[str, ...] = ("overview", "efficiency", "usage", "stat_explorer")
SEASONS:          tuple[str, ...] = ("2023", "2024", "2025", "all")
HISTORICAL_SEASONS: tuple[str, ...] = ("2023", "2024")
CURRENT_SEASON:   str             = "2025"

SCHEMA_VERSION = "1.0.0"

# Identity fields that must appear on every canonical row.
IDENTITY_FIELDS: tuple[str, ...] = (
    "playerId", "playerName", "position", "team", "age", "season", "games",
)

# Tabs whose envelopes require a specific score column.
TAB_REQUIRED_SCORE: dict[str, tuple[str, ...]] = {
    "overview":   ("overallScore", "tier"),
    "efficiency": ("efficiencyScore",),
    "usage":      ("usageScore",),
}

# Stat Explorer must never carry these modeled / interpretive fields.
STAT_EXPLORER_FORBIDDEN: tuple[str, ...] = (
    "roleScore", "efficiencyScore", "overallScore", "usageScore",
    "tier", "careerArc", "experienceTier", "expTier",
    "stabilityScore", "stability", "volatilityScore", "volatility",
    "fantasyPointsOverExpected", "fpoe", "roleTrend",
    "sustainability", "threeYearContext", "relativeEfficiency",
)


# ---------------------------------------------------------------------------
# Per-(position, tab) canonical field schemas
# ---------------------------------------------------------------------------
# Each list defines the exact, ordered set of fields emitted in the canonical
# file's `rows[]`.  Identity fields lead; tab-specific fields follow.

_RB_OVERVIEW = [
    *IDENTITY_FIELDS,
    "snapPct",
    "touchesPerGame",
    "rushAttemptsPerGame",
    "targetsPerGame",
    "routesPerGame",
    "redZoneAttempts",
    "goalLineAttempts",
    "targetSharePct",
    "receptions",
    "receivingYards",
    "yardsPerTouch",
    "explosiveRunPct",
    "breakawayRunPct",
    "fantasyPointsOverExpected",
    "stabilityScore",
    "volatilityScore",
    "careerArc",
    "experienceTier",
    "roleScore",
    "efficiencyScore",
    "overallScore",
    "tier",
]

_QB_OVERVIEW = [
    *IDENTITY_FIELDS,
    "snapPct",
    "dropbacksPerGame",
    "rushAttemptsPerGame",
    "designedRushPct",
    "redZoneAttempts",
    "deepAttemptPct",
    "stabilityScore",
    "careerArc",
    "experienceTier",
    "efficiencyScore",
    "overallScore",
    "tier",
]

_WR_OVERVIEW = [
    *IDENTITY_FIELDS,
    "snapPct",
    "routesPerGame",
    "targetsPerGame",
    "receptionsPerGame",
    "airYardsPerGame",
    "targetSharePct",
    "airYardsSharePct",
    "wopr",
    "redZoneTargets",
    "catchPct",
    "yardsPerRouteRun",
    "yardsPerTarget",
    "yardsAfterCatchPerReception",
    "fantasyPointsOverExpected",
    "stabilityScore",
    "careerArc",
    "experienceTier",
    "roleScore",
    "efficiencyScore",
    "overallScore",
    "tier",
]

_TE_OVERVIEW = list(_WR_OVERVIEW)

_RB_EFFICIENCY = [
    *IDENTITY_FIELDS,
    "rushAttempts",
    "epaPerRush",
    "successRate",
    "yardsPerCarry",
    "yardsPerTouch",
    "fantasyPointsOverExpected",
    "explosiveRunPct",
    "breakawayRunPct",
    "breakawayRuns",
    "longestRush",
    "brokenTackles",
    "brokenTacklesPerAttempt",
    "yardsAfterContactPerAttempt",
    "yardsBeforeContactPerAttempt",
    "efficiencyScore",
]

_QB_EFFICIENCY = [
    *IDENTITY_FIELDS,
    "attempts",
    "epaPerPlay",
    "successRate",
    "yardsPerAttempt",
    "airYardsPerAttempt",
    "fantasyPointsOverExpected",
    "touchdownsOverExpected",
    "explosivePassPct",
    "relativeEfficiency",
    "efficiencyScore",
]

_WR_EFFICIENCY = [
    *IDENTITY_FIELDS,
    "targets",
    "epaPerTarget",
    "successRate",
    "yardsPerTarget",
    "catchPct",
    "fantasyPointsOverExpected",
    "yardsPerRouteRun",
    "targetsPerRouteRun",
    "airYardsPerTarget",
    "airYardsSharePct",
    "explosivePlayPct",
    "explosiveReceptions20Plus",
    "explosiveReceptions40Plus",
    "longestReception",
    "yardsAfterCatchPerReception",
    "yardsBeforeCatchPerReception",
    "brokenTackles",
    "brokenTacklesPerReception",
    "efficiencyScore",
]
_TE_EFFICIENCY = list(_WR_EFFICIENCY)

_RB_USAGE = [
    *IDENTITY_FIELDS,
    "snapPct",
    "deltaSnapPct",
    "touchesPerGame",
    "deltaTouchesPerGame",
    "rushAttemptsPerGame",
    "deltaRushAttemptsPerGame",
    "targetsPerGame",
    "deltaTargetsPerGame",
    "routesPerGame",
    "routePct",
    "deltaRoutePct",
    "redZoneTouches",
    "goalLineAttempts",
    "roleTrend",
    "usageScore",
]

_QB_USAGE = [
    *IDENTITY_FIELDS,
    "dropbacksPerGame",
    "deltaDropbacksPerGame",
    "rushAttemptsPerGame",
    "deltaRushAttemptsPerGame",
    "rushTouchdownsPerGame",
    "roleTrend",
    "usageScore",
]

_WR_USAGE = [
    *IDENTITY_FIELDS,
    "snapPct",
    "deltaSnapPct",
    "routesPerGame",
    "deltaRoutesPerGame",
    "targetsPerGame",
    "deltaTargetsPerGame",
    "airYardsPerGame",
    "deltaAirYardsPerGame",
    "averageDepthOfTarget",
    "deltaAverageDepthOfTarget",
    "redZoneTargets",
    "endZoneTargets",
    "roleTrend",
    "usageScore",
]
_TE_USAGE = list(_WR_USAGE)

_RB_STAT_EXPLORER = [
    "rank",
    *IDENTITY_FIELDS,
    "rushAttempts",
    "rushYards",
    "yardsPerCarry",
    "yardsBeforeContactPerAttempt",
    "yardsAfterContactPerAttempt",
    "brokenTackles",
    "tacklesForLoss",
    "tacklesForLossYards",
    "rushes10Plus",
    "rushes20Plus",
    "rushes30Plus",
    "rushes40Plus",
    "rushes50Plus",
    "longestRush",
    "longestRushTouchdown",
    "targets",
    "receptions",
    "redZoneTargets",
    "receivingYardsAfterCatch",
]

_WR_STAT_EXPLORER = [
    "rank",
    *IDENTITY_FIELDS,
    "receptions",
    "receivingYards",
    "yardsPerReception",
    "yardsBeforeCatch",
    "yardsBeforeCatchPerReception",
    "yardsAfterCatch",
    "yardsAfterCatchPerReception",
    "brokenTackles",
    "targets",
    "targetSharePct",
    "redZoneTargets",
    "receptions10Plus",
    "receptions20Plus",
    "receptions30Plus",
    "receptions40Plus",
    "receptions50Plus",
    "longestReception",
]
_TE_STAT_EXPLORER = list(_WR_STAT_EXPLORER)

_QB_STAT_EXPLORER = [
    "rank",
    *IDENTITY_FIELDS,
    "completions",
    "attempts",
    "completionPct",
    "passingYards",
    "yardsPerAttempt",
    "passingTouchdowns",
    "interceptions",
    "airYards",
    "airYardsPerAttempt",
    "passes10Plus",
    "passes20Plus",
    "passes30Plus",
    "passes40Plus",
    "passes50Plus",
    "rushAttempts",
    "rushYards",
    "rushTouchdowns",
    "sacks",
    "redZoneAttempts",
    "passerRating",
]


FIELD_SCHEMA: dict[str, dict[str, list[str]]] = {
    "qb": {
        "overview":      _QB_OVERVIEW,
        "efficiency":    _QB_EFFICIENCY,
        "usage":         _QB_USAGE,
        "stat_explorer": _QB_STAT_EXPLORER,
    },
    "rb": {
        "overview":      _RB_OVERVIEW,
        "efficiency":    _RB_EFFICIENCY,
        "usage":         _RB_USAGE,
        "stat_explorer": _RB_STAT_EXPLORER,
    },
    "wr": {
        "overview":      _WR_OVERVIEW,
        "efficiency":    _WR_EFFICIENCY,
        "usage":         _WR_USAGE,
        "stat_explorer": _WR_STAT_EXPLORER,
    },
    "te": {
        "overview":      _TE_OVERVIEW,
        "efficiency":    _TE_EFFICIENCY,
        "usage":         _TE_USAGE,
        "stat_explorer": _TE_STAT_EXPLORER,
    },
}


# ---------------------------------------------------------------------------
# Tier labels (derived from overallScore)
# ---------------------------------------------------------------------------
# Per the spec, every overview row carries a `tier`.  RB tiers come from the
# legacy rb_player_overview.rb_tier value verbatim.  QB/WR/TE tier is
# derived here from overallScore using position-appropriate labels.

_TIER_BANDS: list[tuple[float, str, str, str]] = [
    # (threshold, qb_label, wr_label, te_label)
    (85.0, "Elite",          "Elite",  "Elite"),
    (70.0, "QB1",            "WR1",    "TE1"),
    (55.0, "Mid-Tier",       "WR2",    "TE2"),
    (40.0, "Backup-Caliber", "WR3",    "TE3"),
]


def derive_tier(position: str, overall_score: float | None) -> str | None:
    if overall_score is None:
        return None
    try:
        v = float(overall_score)
    except (TypeError, ValueError):
        return None
    pos = position.lower()
    idx = {"qb": 1, "wr": 2, "te": 3}.get(pos)
    if idx is None:
        return None
    for thresh, *labels in _TIER_BANDS:
        if v >= thresh:
            return labels[idx - 1]
    return "Bench"


# ---------------------------------------------------------------------------
# Legacy aliases that must still be produced by the standalone builders.
# Validator confirms each one exists on disk.
# ---------------------------------------------------------------------------

LEGACY_ALIASES: tuple[str, ...] = (
    "rb_player_overview.json",
    "rb_efficiency_analytics.json",
    "rb_usage_role.json",
    "rb_advanced_stats_2025.json",
    "rb_advanced_stats_all.json",
    "wr_player_overview_2025.json",
    "wr_efficiency_analytics_2025.json",
    "wr_trends_2025.json",
    "wr_advanced_stats_2025.json",
    "wr_advanced_stats_all.json",
    "te_player_overview_2025.json",
    "te_efficiency_analytics_2025.json",
    "te_trends_2025.json",
    "te_advanced_stats_2025.json",
    "qb_trends.json",
    "performance_analytics_latest.json",
    "stat_explorer_qb.json",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_SNAKE_RE = re.compile(r"_([a-z0-9])")


def snake_to_camel(s: str) -> str:
    return _SNAKE_RE.sub(lambda m: m.group(1).upper(), s)


def canonical_path(position: str, tab: str, season: str) -> Path:
    pos    = position.lower()
    tab    = tab.lower()
    season = str(season)
    if pos not in POSITIONS:
        raise ValueError(f"unknown position {pos!r}")
    if tab not in TABS:
        raise ValueError(f"unknown tab {tab!r}")
    if season not in SEASONS:
        raise ValueError(f"unknown season {season!r}")
    return CANONICAL_DIR / pos / f"{tab}_{season}.json"


def position_dir(position: str) -> Path:
    return CANONICAL_DIR / position.lower()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def make_envelope(
    *,
    position: str,
    tab: str,
    season: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    source_files: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "position":      position.upper(),
        "tab":           tab,
        "season":        season,
        "rowCount":      len(rows),
        "generatedAt":   now_iso(),
    }
    if source_files:
        meta["sourceFiles"] = source_files
    if extra_meta:
        meta.update(extra_meta)
    return {
        "meta":    meta,
        "columns": columns,
        "rows":    rows,
    }


# ---------------------------------------------------------------------------
# Sleeper playerId resolver
# ---------------------------------------------------------------------------

_NAME_ALIASES: dict[str, str] = {
    # snap_counts / PFR / nflverse spellings → Sleeper canonical full_name
    "Chigoziem Okonkwo": "Chig Okonkwo",
    "Gabriel Davis":     "Gabe Davis",
    "Oronde Gadsden II": "Oronde Gadsden",
    "Kyle Juszczyk":     "Kyle Juszczyk",
    "Michael Penix Jr.": "Michael Penix",
    "Joe Milton III":    "Joe Milton",
    "Josh Palmer":       "Joshua Palmer",
    "Scott Miller":      "Scotty Miller",
    "Mike Strachan":     "Michael Strachan",
    "Mike Woods":        "Michael Woods",
}


_GEN_SUFFIXES = frozenset(["jr", "sr", "ii", "iii", "iv", "v"])


def _strip_accents(s: str) -> str:
    """Remove diacritics so 'Estimé' matches Sleeper's 'Estime'."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _norm_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/whitespace for fuzzy name matching."""
    name = _strip_accents(name)
    return name.lower().replace(".", "").replace("-", "").replace(" ", "").replace("'", "")


def _norm_name_base(name: str) -> str:
    """Like _norm_name but also drops a trailing generational suffix
    (Jr./Sr./II–V) so nflverse 'Kenneth Walker III' matches Sleeper
    'Kenneth Walker'."""
    parts = _strip_accents(name).split()
    if parts and parts[-1].lower().rstrip(".") in _GEN_SUFFIXES:
        parts.pop()
    return _norm_name(" ".join(parts))


class PlayerIdResolver:
    """Maps (playerName, team) → Sleeper player_id (string).

    Resolution order:
      1. alias rewrite (e.g. "Chigoziem Okonkwo" → "Chig Okonkwo")
      2. exact full_name match (prefer same team)
      3. normalized full_name match (strip punct/whitespace)
      4. PBP-style abbreviation "X.LastName" / "Xx.LastName" + team
         (e.g. "M.Evans" + TB → Mike Evans)
      5. (team, last_name) tie-break
      6. None — ID-less row (callers may still emit the row)
    """

    _ABBREV_RE = re.compile(r"^([A-Za-z]{1,3})\.\s*(.+)$")

    def __init__(self, sleeper_path: Path = SLEEPER_PATH) -> None:
        with open(sleeper_path) as f:
            raw = json.load(f)
        players = list(raw.values()) if isinstance(raw, dict) else raw

        self._by_name: dict[str, list[dict]] = {}
        self._by_norm: dict[str, list[dict]] = {}
        # suffix-stripped + accent-stripped normalized name → records
        self._by_norm_base: dict[str, list[dict]] = {}
        # ((first_initial_lower, last_name_lower, team) → player_id)
        self._by_initial_last_team: dict[tuple[str, str, str], str] = {}
        # ((first_initial_lower, last_name_lower) → list of (player_id, team, full_name))
        self._by_initial_last: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        # (team, last_name_lower) → first unambiguous player_id
        self._by_team_last: dict[tuple[str, str], str] = {}

        for p in players:
            name = (p.get("full_name") or "").strip()
            pid  = p.get("player_id")
            team = p.get("team") or ""
            position = p.get("position") or ""
            if not name:
                continue
            self._by_name.setdefault(name, []).append(p)
            self._by_norm.setdefault(_norm_name(name), []).append(p)
            self._by_norm_base.setdefault(_norm_name_base(name), []).append(p)

            parts = name.split()
            if pid and parts:
                first_initial = parts[0][0].lower()
                last          = parts[-1].lower()
                if team:
                    self._by_initial_last_team[(first_initial, last, team)] = pid
                self._by_initial_last.setdefault(
                    (first_initial, last), []
                ).append((pid, team, name, position))
                if team:
                    self._by_team_last.setdefault((team, last), pid)

    def resolve(
        self,
        player_name: str | None,
        team: str | None,
        position: str | None = None,
    ) -> str | None:
        if not player_name:
            return None
        name = (player_name or "").strip()
        if not name:
            return None
        name = _NAME_ALIASES.get(name, name)
        team = team or None
        pos  = (position or "").upper() or None

        def _filter_pos(records: list[dict]) -> list[dict]:
            if not pos:
                return records
            same = [r for r in records if (r.get("position") or "").upper() == pos]
            return same if same else records

        # 2. exact full_name match
        hits = _filter_pos(self._by_name.get(name) or [])
        if hits:
            if team:
                for p in hits:
                    if p.get("team") == team:
                        return p.get("player_id")
            return hits[0].get("player_id")

        # 3. normalized full_name
        nhits = _filter_pos(self._by_norm.get(_norm_name(name)) or [])
        if nhits:
            if team:
                for p in nhits:
                    if p.get("team") == team:
                        return p.get("player_id")
            return nhits[0].get("player_id")

        # 3b. suffix-stripped normalized match (e.g. nflverse "Kenneth Walker III"
        #     → Sleeper "Kenneth Walker"; also covers accent differences).
        base = _norm_name_base(name)
        bhits = _filter_pos((self._by_norm_base.get(base) or []) + (self._by_norm.get(base) or []))
        if bhits:
            if team:
                for p in bhits:
                    if p.get("team") == team:
                        return p.get("player_id")
            # position filter already applied; prefer a single position match
            return bhits[0].get("player_id")

        # 4. PBP-style abbreviation "M.Evans" / "Mi.Wilson"
        m = self._ABBREV_RE.match(name)
        if m:
            prefix  = m.group(1).lower()
            last    = m.group(2).strip().lower()
            initial = prefix[0]

            def _filter_pos_tuples(cands):
                if not pos:
                    return cands
                same = [c for c in cands if (c[3] or "").upper() == pos]
                return same if same else cands

            if team:
                pid = self._by_initial_last_team.get((initial, last, team))
                if pid:
                    return pid
            candidates = _filter_pos_tuples(self._by_initial_last.get((initial, last), []))
            if len(prefix) > 1:
                prefer = [
                    c for c in candidates
                    if c[2].split()[0][:len(prefix)].lower() == prefix
                ]
                if prefer:
                    if team:
                        for pid, t, *_ in prefer:
                            if t == team:
                                return pid
                    return prefer[0][0]
            if candidates:
                if team:
                    for pid, t, *_ in candidates:
                        if t == team:
                            return pid
                # Position-filtered candidates without team match: take first
                if len(candidates) == 1 or pos:
                    return candidates[0][0]

        # 5. team + last name fallback
        if team:
            parts = name.split()
            if parts:
                pid = self._by_team_last.get((team, parts[-1].lower()))
                if pid:
                    return pid
        return None


# ---------------------------------------------------------------------------
# Manifest writers
# ---------------------------------------------------------------------------

def discover_canonical_files(position: str) -> list[dict[str, Any]]:
    pos = position.lower()
    out: list[dict[str, Any]] = []
    pdir = position_dir(pos)
    if not pdir.exists():
        return out
    for tab in TABS:
        for season in SEASONS:
            fp = canonical_path(pos, tab, season)
            if not fp.exists():
                continue
            try:
                with open(fp) as fh:
                    payload = json.load(fh)
                row_count = payload.get("meta", {}).get("rowCount")
                if row_count is None:
                    row_count = len(payload.get("rows") or [])
            except Exception:
                row_count = None
            out.append({
                "tab":       tab,
                "season":    season,
                "filename":  fp.name,
                "path":      str(fp.relative_to(OUTPUT_DIR)),
                "rowCount":  row_count,
                "bytes":     fp.stat().st_size,
            })
    return out


def _unavailable_by_tab(
    unavailable: list[tuple[str, str]] | None,
) -> dict[str, list[str]]:
    """Group [(tab, season), …] into {tab: [season, …]} for manifest output."""
    out: dict[str, list[str]] = {}
    for tab, season in (unavailable or []):
        out.setdefault(tab, []).append(season)
    for tab in out:
        out[tab].sort()
    return out


def write_position_manifest(
    position: str,
    unavailable: list[tuple[str, str]] | None = None,
) -> Path:
    pos = position.lower()
    files = discover_canonical_files(pos)
    tabs_present: dict[str, list[str]] = {}
    for f in files:
        tabs_present.setdefault(f["tab"], []).append(f["season"])
    for tab in tabs_present:
        tabs_present[tab].sort()
    payload = {
        "schemaVersion":       SCHEMA_VERSION,
        "position":            pos.upper(),
        "generatedAt":         now_iso(),
        "tabs":                tabs_present,
        "seasons":             sorted({f["season"] for f in files}),
        "unavailableSeasons":  _unavailable_by_tab(unavailable),
        "files":               files,
    }
    path = position_dir(pos) / "manifest.json"
    write_json(path, payload)
    return path


def write_global_manifest() -> Path:
    positions: dict[str, Any] = {}
    for pos in POSITIONS:
        files = discover_canonical_files(pos)
        tabs_present: dict[str, list[str]] = {}
        for f in files:
            tabs_present.setdefault(f["tab"], []).append(f["season"])
        for tab in tabs_present:
            tabs_present[tab].sort()

        # Read per-position manifest (if any) to surface unavailableSeasons
        unavail: dict[str, list[str]] = {}
        pm_path = position_dir(pos) / "manifest.json"
        if pm_path.exists():
            try:
                with open(pm_path) as fh:
                    pm = json.load(fh)
                unavail = pm.get("unavailableSeasons") or {}
            except Exception:
                unavail = {}

        positions[pos] = {
            "tabs":               tabs_present,
            "unavailableSeasons": unavail,
        }
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt":   now_iso(),
        "positions":     positions,
    }
    path = CANONICAL_DIR / "manifest.json"
    write_json(path, payload)
    return path
