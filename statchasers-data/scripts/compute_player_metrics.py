"""
compute_player_metrics.py

Reads raw nflverse play-by-play data and Sleeper player metadata,
then computes all advanced player-level metrics needed for the
StatChasers Performance Analytics dashboard.

Season split strategy:
  - Dashboard metrics (games, shares, trends, EPA) use 2025-only data.
  - Context labels (threeYearContext, careerArc) use all available seasons
    so veteran stars are classified correctly regardless of 2025 sample size.

Input:  data/raw/nflverse_play_by_play.parquet
        data/raw/nflverse_player_stats.parquet (optional)
        data/raw/sleeper_players.json

Output: data/processed/player_metrics.json
"""

import json
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

PBP_PATH     = os.path.join(RAW_DIR, "nflverse_play_by_play.parquet")
STATS_PATH   = os.path.join(RAW_DIR, "nflverse_player_stats.parquet")
SLEEPER_PATH = os.path.join(RAW_DIR, "sleeper_players.json")
NFL_PLAYERS_PATH = os.path.join(RAW_DIR, "nflverse_players.parquet")
OUTPUT_PATH  = os.path.join(PROCESSED_DIR, "player_metrics.json")

# The season whose data drives all dashboard (usage/share/trend) metrics
CURRENT_SEASON = 2025

# Minimum 2025-season plays to include a player
# Aligned with WR/TE overview MIN_TARGETS=15 so FPOE is available for
# all players that appear in the analytics outputs
MIN_PLAYS = 15

# Hardcoded team overrides for cases where Sleeper team data is stale and
# pos_hint alone cannot disambiguate two same-position players.
# Key: PBP abbreviated name  →  {pbp_team: canonical full name}
_MANUAL_TEAM_OVERRIDES: dict[str, dict[str, str]] = {
    # Travis Etienne plays JAX in 2025 PBP; Sleeper still lists him as NO.
    # Both Etienne players are RBs so pos_hint cannot distinguish them.
    "T.Etienne": {"JAX": "Travis Etienne", "CAR": "Trevor Etienne"},
    # Brian Robinson moved to SF; Sleeper still lists him as None (was WAS).
    # Both Robinson players are RBs so pos_hint cannot distinguish them.
    "B.Robinson": {"ATL": "Bijan Robinson", "SF": "Brian Robinson", "WAS": "Brian Robinson"},
    # Javonte Williams played DEN 2023-2024 then DAL 2025 (DAL already in disambig).
    # Jamaal Williams played NO 2023 (retired, team=None in Sleeper).
    # Both fall through to Jarveon Williams (None slot) without this override.
    "J.Williams": {"DEN": "Javonte Williams", "NO": "Jamaal Williams"},
    # Four distinct T.Johnson players in 2025 PBP — none in Sleeper by gsis_id.
    # Without overrides the ambiguous T.Johnson abbreviation maps to Tony Johnson (K).
    # BUF entry covers Ty Johnson (RB) who can appear as a receiver in screen plays.
    "T.Johnson": {"NYG": "Theo Johnson", "NYJ": "Tyler Johnson",
                  "TB": "Tez Johnson",   "BUF": "Ty Johnson"},
    # nflverse uses multi-word abbreviation A.St. Brown; _abbrev() produces A.Brown
    # which collides with A.J. Brown. Store under full Sleeper name so analytics
    # scripts that resolve via Sleeper (Amon-Ra St. Brown) can find the FPOE.
    "A.St. Brown": {"DET": "Amon-Ra St. Brown"},
    # Mike Evans (WR, TB): Sleeper lists him as team=SF (stale, moved to TB).
    # Without this override M.Evans/TB is stored under the PBP abbreviation
    # rather than the canonical full name.
    "M.Evans": {"CAR": "Mitchell Evans", "TB": "Mike Evans"},
    # Keenan Allen (WR): Sleeper team=None (was released by LAC, now FA/retired).
    # "K.Allen" is ambiguous (Kaytron Allen RB, Kyle Allen QB) so abbreviated
    # lookup skips it; team_disambig fails because no Sleeper team matches LAC.
    "K.Allen": {"LAC": "Keenan Allen"},
    # DJ Moore (WR): Sleeper full_name="DJ Moore" team=BUF.
    # "D.Moore" is ambiguous (Denarius Moore, David Moore); when PBP team is CHI
    # or CAR the disambig fails because DJ Moore's Sleeper team is now BUF.
    "D.Moore": {"CHI": "DJ Moore", "CAR": "DJ Moore", "BUF": "DJ Moore"},
    # DeAndre Hopkins (WR): Sleeper team=None. "D.Hopkins" collides with
    # Dustin Hopkins (K, also team=None) — position hint not applied in receiving loop.
    "D.Hopkins": {"BAL": "DeAndre Hopkins", "TEN": "DeAndre Hopkins"},
    # Gabe Davis (WR): Sleeper team=None. "G.Davis" is ambiguous (Geremy Davis).
    "G.Davis": {"BUF": "Gabe Davis", "JAX": "Gabe Davis"},
}

# Minimum 2025-season games before applying narrative labels
MIN_GAMES_FOR_LABELS = 6

# Minimum career games (all seasons) before applying "Elite" tier labels
MIN_CAREER_GAMES_ELITE = 32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    """
    Load all raw data sources.

    Returns
    -------
    pbp_all         : DataFrame of all seasons (used for career-context labels)
    pbp_2025        : DataFrame of CURRENT_SEASON only (used for dashboard metrics)
    stats           : Weekly player stats (optional; may be empty)
    sleeper_players : list of Sleeper player dicts
    nfl_players     : nflverse player registry with years_of_experience
    """
    if not os.path.exists(PBP_PATH):
        print(f"ERROR: Play-by-play data not found at {PBP_PATH}", file=sys.stderr)
        print("Run pull_nflverse_data.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading play-by-play data...")
    pbp_all = pd.read_parquet(PBP_PATH)
    print(f"  {len(pbp_all):,} total plays loaded ({pbp_all['season'].nunique() if 'season' in pbp_all.columns else '?'} seasons).")

    if "season" in pbp_all.columns:
        pbp_2025 = pbp_all[pbp_all["season"] == CURRENT_SEASON].copy()
        print(f"  {len(pbp_2025):,} plays in {CURRENT_SEASON} season.")
    else:
        print(f"WARNING: No 'season' column found; treating all data as {CURRENT_SEASON}.", file=sys.stderr)
        pbp_2025 = pbp_all.copy()

    stats = pd.DataFrame()
    if os.path.exists(STATS_PATH):
        print("Loading player stats data...")
        stats = pd.read_parquet(STATS_PATH)
        print(f"  {len(stats):,} player-week records loaded.")

    if not os.path.exists(SLEEPER_PATH):
        print(f"ERROR: Sleeper player data not found at {SLEEPER_PATH}", file=sys.stderr)
        print("Run pull_sleeper_players.py first.", file=sys.stderr)
        sys.exit(1)

    with open(SLEEPER_PATH) as f:
        sleeper_players = json.load(f)

    print(f"  {len(sleeper_players)} Sleeper players loaded.")

    nfl_players = pd.DataFrame()
    if os.path.exists(NFL_PLAYERS_PATH):
        nfl_players = pd.read_parquet(NFL_PLAYERS_PATH)
        print(f"  {len(nfl_players):,} nflverse player registry records loaded.")
    else:
        print("  NOTE: nflverse_players.parquet not found — experience tiers will use game-count estimate.")

    return pbp_all, pbp_2025, stats, sleeper_players, nfl_players


# ---------------------------------------------------------------------------
# Name resolution helpers
# ---------------------------------------------------------------------------

def build_player_lookup(sleeper_players: list[dict]) -> dict[str, dict]:
    """Build a full-name -> metadata lookup from Sleeper data."""
    lookup = {}
    for p in sleeper_players:
        name = p.get("full_name", "").strip()
        if name:
            lookup[name] = p
    return lookup


def _nflverse_abbrev(full_name: str) -> str:
    """
    Derive the nflverse-style abbreviated name from a Sleeper full name.

    nflverse uses 'FirstInitial.LastName', e.g.:
      "Ja'Marr Chase"  -> "J.Chase"
      "CeeDee Lamb"    -> "C.Lamb"
      "Saquon Barkley" -> "S.Barkley"
      "Travis Kelce"   -> "T.Kelce"
    """
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    first_initial = parts[0][0].upper()
    last_name = parts[-1]
    return f"{first_initial}.{last_name}"


def build_abbreviated_lookup(sleeper_players: list[dict]) -> dict[str, str]:
    """
    Build an abbreviated-name -> full-name lookup so nflverse PBP names
    (e.g. 'J.Chase') resolve to canonical Sleeper names ('Ja'Marr Chase').

    Unambiguous abbreviations only (exactly one Sleeper player maps to that
    abbrev).  Ambiguous ones are handled by build_team_disambiguation().
    """
    counts: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for p in sleeper_players:
        full = p.get("full_name", "").strip()
        if not full:
            continue
        abbrev = _nflverse_abbrev(full)
        counts[abbrev] = counts.get(abbrev, 0) + 1
        mapping[abbrev] = full
    return {abbrev: full for abbrev, full in mapping.items() if counts[abbrev] == 1}


def build_team_disambiguation(
    sleeper_players: list[dict],
) -> dict[str, dict[str, str]]:
    """
    For abbreviated names that map to more than one Sleeper player, build a
    secondary lookup keyed by (abbrev -> {team -> full_name}).

    Used by resolve_full_name when the primary lookup is ambiguous so that
    "R.Wilson on PIT" → "Roman Wilson" and "R.Wilson on NYG" → "Russell Wilson"
    even though both share the "R.Wilson" abbreviation.

    Players without a team in Sleeper (cut / retired) are stored under key
    None so they can still be matched if no team-specific match is found.
    """
    counts: dict[str, int] = {}
    by_team: dict[str, dict] = {}
    for p in sleeper_players:
        full = p.get("full_name", "").strip()
        if not full:
            continue
        abbrev = _nflverse_abbrev(full)
        counts[abbrev] = counts.get(abbrev, 0) + 1
        team = p.get("team")
        slot = by_team.setdefault(abbrev, {})
        if team in slot:
            # Multiple players share the same team slot (often both team=None).
            # Promote to a list so resolve_full_name can apply pos_hint filtering.
            existing = slot[team]
            if isinstance(existing, list):
                existing.append(full)
            else:
                slot[team] = [existing, full]
        else:
            slot[team] = full
    return {abbrev: teams for abbrev, teams in by_team.items() if counts[abbrev] > 1}


def build_exp_lookup(nfl_players: pd.DataFrame) -> dict[str, int]:
    """
    Build a display_name → years_of_experience lookup from the nflverse
    player registry.

    The nflverse `display_name` field (e.g. "Russell Wilson") closely matches
    Sleeper's `full_name`, making it the best join key without a shared player ID.

    Returns an empty dict when the registry is unavailable.
    """
    if nfl_players.empty or "display_name" not in nfl_players.columns:
        return {}
    if "years_of_experience" not in nfl_players.columns:
        return {}
    lookup: dict[str, int] = {}
    for _, row in nfl_players.iterrows():
        name = row.get("display_name")
        yoe = row.get("years_of_experience")
        if name and pd.notna(yoe):
            lookup[str(name).strip()] = int(yoe)
    return lookup


def resolve_full_name(
    pbp_name: str,
    player_lookup: dict[str, dict],
    abbreviated_lookup: dict[str, str],
    team_disambig: dict[str, dict[str, str]] | None = None,
    pbp_team: str | None = None,
    pos_hint: str | None = None,
) -> str:
    """
    Return the canonical Sleeper full name for a PBP player name.

    Resolution order:
    1. Direct match in Sleeper full-name lookup (already a full name).
    2. Unambiguous abbreviated-name lookup.
    3. Team-based disambiguation for ambiguous abbreviations:
       a. Exact team match (e.g. "R.Wilson" + "PIT" → "Roman Wilson").
       b. Position filter (pos_hint provided): keep only candidates whose
          Sleeper position matches pos_hint.
            • Exactly one match → return it.
            • Multiple matches (e.g. two QBs) → prefer the one on an active
              NFL roster (team ≠ None) over a teamless/retired player.
       c. No position hint: prefer the candidate on an active roster (non-None
          team), then fall back to the teamless entry as a last resort.
    4. Return the PBP name unchanged.

    The pos_hint tiebreaker prevents Taulia Tagovailoa (teamless, QB) from
    being chosen over Tua Tagovailoa (ATL, QB) just because Tua's Sleeper team
    (post-season move) no longer matches his 2025 PBP team (MIA).
    """
    # Manual override: checked first so stale Sleeper team data never wins.
    if pbp_name in _MANUAL_TEAM_OVERRIDES and pbp_team:
        hit = _MANUAL_TEAM_OVERRIDES[pbp_name].get(pbp_team)
        if hit:
            return hit
    if pbp_name in player_lookup:
        return pbp_name
    if pbp_name in abbreviated_lookup:
        return abbreviated_lookup[pbp_name]
    if team_disambig and pbp_name in team_disambig:
        teams = team_disambig[pbp_name]           # {team | None: full_name | [full_name, ...]}
        # (a) Exact team match (only when slot is unambiguous)
        if pbp_team and pbp_team in teams:
            slot = teams[pbp_team]
            if isinstance(slot, str):
                return slot
            # slot is a list — fall through to pos_hint disambiguation below
        # Expand list-valued slots into flat (team, full_name) pairs so that
        # pos_hint filtering can distinguish e.g. Aaron Rodgers (QB, None)
        # from Amari Rodgers (WR, None) even though both have team=None.
        candidates: list[tuple] = []
        for t, val in teams.items():
            if isinstance(val, list):
                candidates.extend((t, n) for n in val)
            else:
                candidates.append((t, val))
        # (b) Position-filtered disambiguation
        if pos_hint:
            pos_matches = [
                (team, name) for team, name in candidates
                if player_lookup.get(name, {}).get("position") == pos_hint
            ]
            if len(pos_matches) == 1:
                return pos_matches[0][1]
            if len(pos_matches) > 1:
                # Multiple same-position candidates: active roster > teamless
                with_team = [(t, n) for t, n in pos_matches if t is not None]
                if with_team:
                    return with_team[0][1]
                return pos_matches[0][1]
        # (c) No position hint — prefer active roster, fall back to teamless
        with_team = [(t, n) for t, n in candidates if t is not None]
        if len(with_team) == 1:
            return with_team[0][1]
        # Fall back to the first teamless candidate (teams[None] may be a list)
        teamless = [n for t, n in candidates if t is None]
        if teamless:
            return teamless[0]
    return pbp_name


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def safe_pct(numerator: float, denominator: float, scale: float = 100.0) -> float:
    """
    Safe percentage calculation.  All share/rate fields in the output are
    in percent form (e.g. 24.9, not 0.249).  WOPR is the only field that
    intentionally stays on a 0-~1.5 index scale.
    """
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * scale, 2)


# ---------------------------------------------------------------------------
# Career context computation (multi-season)
# ---------------------------------------------------------------------------

def compute_career_stats(pbp_all: pd.DataFrame) -> dict[str, dict]:
    """
    Batch-compute career-level context (all available seasons) for every
    player appearing in the PBP data.

    Returns a dict keyed by PBP abbreviated name, e.g. 'J.Chase'.
    Values: { career_games: int, career_epa_per_play: float }

    Used exclusively for threeYearContext and careerArc labels — not for
    any current-season dashboard metric.
    """
    frames = []

    def _agg(df: pd.DataFrame, name_col: str, play_col: str) -> pd.DataFrame:
        if name_col not in df.columns or play_col not in df.columns:
            return pd.DataFrame()
        sub = df[df[play_col] == 1].dropna(subset=[name_col])
        if sub.empty:
            return pd.DataFrame()
        return sub.groupby(name_col).agg(
            career_games=("game_id", "nunique"),
            career_epa_sum=("epa", "sum"),
            career_plays=(play_col, "sum"),
        ).reset_index().rename(columns={name_col: "pbp_name"})

    frames.append(_agg(pbp_all, "passer_player_name", "pass_attempt"))
    frames.append(_agg(pbp_all, "receiver_player_name", "pass_attempt"))
    frames.append(_agg(pbp_all, "rusher_player_name", "rush_attempt"))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)
    # When a player appears in multiple groups (e.g. RB with carries and targets),
    # keep the entry that has the most games (most representative of career).
    combined = (
        combined
        .sort_values("career_games", ascending=False)
        .drop_duplicates("pbp_name")
    )
    combined["career_epa_per_play"] = (
        combined["career_epa_sum"] / combined["career_plays"].clip(lower=1)
    ).round(3)

    return {
        row["pbp_name"]: {
            "career_games": int(row["career_games"]),
            "career_epa_per_play": float(row["career_epa_per_play"]),
        }
        for _, row in combined.iterrows()
    }


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

# Position-aware age thresholds:
#   (ascending_max, peak_max, plateau_max)
# Age <= ascending_max → Ascending (if productive)
# ascending_max < age <= peak_max → Peak (if productive) / Stable
# peak_max < age <= plateau_max → Plateau
# age > plateau_max → Veteran or Declining
_AGE_THRESHOLDS: dict[str, tuple[int, int, int]] = {
    "QB": (26, 34, 37),   # QBs peak late and sustain longest
    "WR": (25, 29, 32),   # WRs hit peak in late 20s
    "RB": (24, 27, 30),   # RBs decline fastest
    "TE": (25, 32, 34),   # TEs develop slowly, sustain long
}


def classify_career_arc(
    pos: str,
    age: int | None,
    career_epa_per_play: float,
    current_snap_share: float,
    career_games: int,
    current_games: int,
) -> str | None:
    """
    Position-aware career arc label using age curves and multi-season EPA.

    - Returns None when age is unavailable.
    - Returns 'Insufficient Data' when the current-season sample is too small.
    - Uses career EPA (all seasons) so veterans aren't mislabeled by a
      small 2025 sample.
    - Never returns 'Developing' for a player with 30+ career games — they
      are past the developing stage regardless of current-season metrics.
    """
    if age is None:
        return None
    if current_games < MIN_GAMES_FOR_LABELS:
        return "Insufficient Data"

    ascending_max, peak_max, plateau_max = _AGE_THRESHOLDS.get(pos, (25, 30, 33))

    is_effective = career_epa_per_play > 0.02
    is_positive = career_epa_per_play > 0.0
    is_declining_epa = career_epa_per_play < -0.03
    is_veteran = career_games >= 32

    if age <= ascending_max:
        if is_effective:
            return "Ascending"
        # A player with 30+ career games is no longer developing even if young
        return "Stable" if is_veteran else "Developing"

    if age <= peak_max:
        if is_effective:
            return "Peak"
        if is_positive:
            return "Stable"
        return "Plateau"

    if age <= plateau_max:
        if is_effective:
            return "Plateau"
        if is_declining_epa:
            return "Declining"
        return "Veteran"

    # Oldest bracket
    if is_declining_epa:
        return "Declining"
    return "Veteran"


def classify_three_year_context(
    pos: str,
    career_epa_per_play: float,
    current_snap_share: float,
    career_games: int,
    current_games: int,
) -> str | None:
    """
    Multi-season narrative label for the threeYearContext field.

    Uses career EPA (all seasons) + current-season snap share.
    'Elite' labels require 32+ career games AND 9+ current-season games
    so that small-sample or early-career players are never over-labelled.
    """
    if current_games < MIN_GAMES_FOR_LABELS:
        return None

    elite_qualified = career_games >= MIN_CAREER_GAMES_ELITE and current_games >= 9
    starter_qualified = career_games >= 16

    if pos == "QB":
        if career_epa_per_play > 0.15 and elite_qualified:
            return "Elite QB1 trajectory"
        if career_epa_per_play > 0.08 and starter_qualified:
            return "Solid franchise QB"
        if career_epa_per_play > 0.0:
            return "Serviceable starter"
        return "Backup / streaming option"

    if pos == "WR":
        if career_epa_per_play > 0.12 and current_snap_share > 75 and elite_qualified:
            return "Elite WR1 trajectory"
        if career_epa_per_play > 0.06 and current_snap_share > 60 and starter_qualified:
            return "WR2 with WR1 upside"
        if current_snap_share > 50:
            return "Rotational / emerging role"
        return "Depth / situational role"

    if pos == "RB":
        if current_snap_share > 60 and career_epa_per_play > 0.0 and elite_qualified:
            return "Workhorse back"
        if current_snap_share > 40 and starter_qualified:
            return "Timeshare back with value"
        return "Change-of-pace / specialty role"

    if pos == "TE":
        if career_epa_per_play > 0.10 and current_snap_share > 65 and elite_qualified:
            return "Elite TE1 trajectory"
        if career_epa_per_play > 0.0 and current_snap_share > 50 and starter_qualified:
            return "Receiving TE with upside"
        return "Blocking / depth TE"

    return None


def classify_sustainability(
    success_rate: float,
    usage_volatility: float,
    role_stability: float,
    games: int,
) -> str | None:
    """
    Evaluate production sustainability from 2025-season data.
    Returns None when the sample is too small.
    """
    if games < MIN_GAMES_FOR_LABELS:
        return None
    if success_rate > 52 and usage_volatility < 8 and role_stability > 7:
        return "Sustainable"
    if success_rate > 45 and usage_volatility < 12:
        return "Mostly Sustainable"
    if usage_volatility > 15:
        return "High Variance"
    return "Regression Risk"


# ---------------------------------------------------------------------------
# Experience tier
# ---------------------------------------------------------------------------

def classify_experience_tier(years_of_experience: int) -> str:
    """
    Deterministic career-longevity tier based on NFL seasons played.

    Boundaries align with standard NFL contract phases:
      1       → Rookie       (on rookie contract, first year)
      2–4     → Early Career  (years 2-4, developing starter)
      5–9     → Established  (prime contract window)
      ≥ 10    → Veteran      (decade of experience)

    Input is seasons of NFL experience (1 = rookie year), sourced from the
    nflverse player registry (years_of_experience field).  Callers should
    prefer that source; the game-count estimator is a fallback only.
    """
    if years_of_experience <= 1:
        return "Rookie"
    if years_of_experience <= 4:
        return "Early Career"
    if years_of_experience <= 9:
        return "Established"
    return "Veteran"


def years_exp_from_games(career_games: int) -> int:
    """
    Estimate seasons of experience from game count when the nflverse
    player registry is unavailable.  Uses 17-game seasons.
    """
    return max(1, round(career_games / 17))


# ---------------------------------------------------------------------------
# Per-position metric aggregation (2025 season only)
# ---------------------------------------------------------------------------

def compute_passing_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-QB metrics from 2025 pass plays.

    Sacks are excluded from attempt counts so the numbers match official
    NFL/ESPN box-score statistics (which record sacks separately).
    nflverse marks sacks as pass_attempt==1, so without this filter every
    QB's attempt total would be inflated by their sack count.

    Games are counted from all dropbacks (including sacks) so a game is
    credited even if the QB was sacked on every play.
    """
    all_dropbacks = pbp[pbp["pass_attempt"] == 1].copy()
    if all_dropbacks.empty:
        return pd.DataFrame()

    # Official pass attempts = dropbacks minus sacks
    if "sack" in all_dropbacks.columns:
        official = all_dropbacks[all_dropbacks["sack"] != 1]
    else:
        print("WARNING: 'sack' column not found in PBP; attempt counts will include sacks.", file=sys.stderr)
        official = all_dropbacks

    grouped = official.groupby("passer_player_name").agg(
        attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        yards_total=("yards_gained", "sum"),
        total_epa=("epa", "sum"),
        successful_plays=("success", "sum"),
        touchdowns=("touchdown", "sum"),
        air_yards_total=("air_yards", lambda x: x.dropna().sum()),
        deep_targets=("air_yards", lambda x: (x.dropna() >= 20).sum()),
        explosive_plays=("yards_gained", lambda x: (x >= 20).sum()),
        weeks=("week", "nunique"),
        rz_attempts=("yardline_100", lambda x: (x.dropna() <= 20).sum()),
    ).reset_index()
    grouped.rename(columns={"passer_player_name": "player_name"}, inplace=True)

    # games counted from all dropbacks so sack-only games still register
    games_all = (
        all_dropbacks.groupby("passer_player_name")["game_id"]
        .nunique()
        .rename("games")
        .reset_index()
        .rename(columns={"passer_player_name": "player_name"})
    )
    grouped = grouped.merge(games_all, on="player_name", how="left")
    grouped["games"] = grouped["games"].fillna(0).astype(int)

    team_last = (
        all_dropbacks.groupby("passer_player_name")["posteam"]
        .last()
        .reset_index()
        .rename(columns={"passer_player_name": "player_name", "posteam": "team"})
    )
    grouped = grouped.merge(team_last, on="player_name", how="left")
    grouped["team"] = grouped["team"].fillna("UNK")
    grouped["pos"] = "QB"
    return grouped


def compute_receiving_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-receiver metrics from 2025 pass plays."""
    targets = pbp[pbp["pass_attempt"] == 1].copy()
    if targets.empty:
        return pd.DataFrame()

    # Team-level denominators (2025 only) for share calculations
    team_air_yards = (
        targets.groupby(["posteam", "game_id"])["air_yards"]
        .sum()
        .reset_index()
        .groupby("posteam")["air_yards"]
        .mean()
        .rename("team_air_yards_per_game")
    )

    team_targets = (
        targets.groupby(["posteam", "game_id"])
        .size()
        .reset_index(name="count")
        .groupby("posteam")["count"]
        .mean()
        .rename("team_targets_per_game")
    )

    rec_plays = targets[targets["receiver_player_name"].notna()].copy()

    # Group by (name, team) so players who share an abbreviated name but play
    # for different teams (e.g. B.Robinson on ATL vs SF) get separate rows.
    grouped = rec_plays.groupby(["receiver_player_name", "posteam"]).agg(
        attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        total_epa=("epa", "sum"),
        successful_plays=("success", "sum"),
        touchdowns=("touchdown", "sum"),
        air_yards_total=("air_yards", lambda x: x.dropna().sum()),
        deep_targets=("air_yards", lambda x: (x.dropna() >= 20).sum()),
        explosive_plays=("yards_gained", lambda x: (x >= 20).sum()),
        games=("game_id", "nunique"),
        weeks=("week", "nunique"),
        yards_gained_total=("yards_gained", "sum"),
        yac_total=("yards_after_catch", lambda x: x.dropna().sum()),
    ).reset_index()

    grouped.rename(columns={"receiver_player_name": "player_name", "posteam": "team"}, inplace=True)
    grouped["pos"] = "WR"

    grouped = grouped.merge(
        team_air_yards.reset_index(), left_on="team", right_on="posteam", how="left"
    ).drop(columns=["posteam"], errors="ignore")

    grouped = grouped.merge(
        team_targets.reset_index(), left_on="team", right_on="posteam", how="left"
    ).drop(columns=["posteam"], errors="ignore")

    return grouped


def compute_rushing_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-rusher metrics from 2025 rush plays."""
    rush_plays = pbp[pbp["rush_attempt"] == 1].copy()
    if rush_plays.empty:
        return pd.DataFrame()

    if "yardline_100" in rush_plays.columns:
        rush_plays["goal_line_carry"] = rush_plays["yardline_100"].apply(
            lambda x: 1 if pd.notna(x) and x <= 5 else 0
        )
    else:
        rush_plays["goal_line_carry"] = 0

    # Group by (name, team) so players who share an abbreviated name but play
    # for different teams (e.g. B.Robinson on ATL vs SF) get separate rows.
    grouped = rush_plays.groupby(["rusher_player_name", "posteam"]).agg(
        attempts=("rush_attempt", "sum"),
        total_epa=("epa", "sum"),
        successful_plays=("success", "sum"),
        touchdowns=("touchdown", "sum"),
        explosive_plays=("yards_gained", lambda x: (x >= 15).sum()),
        breakaway_runs=("yards_gained", lambda x: (x >= 15).sum()),
        goal_line_carries=("goal_line_carry", "sum"),
        games=("game_id", "nunique"),
        weeks=("week", "nunique"),
        yards_gained_total=("yards_gained", "sum"),
    ).reset_index()

    grouped.rename(columns={"rusher_player_name": "player_name", "posteam": "team"}, inplace=True)
    grouped["pos"] = "RB"
    return grouped


# ---------------------------------------------------------------------------
# Route participation (from weekly player stats, 2025 season only)
# ---------------------------------------------------------------------------

def compute_route_participation(
    stats: pd.DataFrame,
    season: int,
) -> dict[str, float | None]:
    """
    Compute route participation rate per player from the nflverse weekly
    player stats dataset.

    Route participation = routes_run / offensive_snaps × 100

    Returns a dict keyed by player_display_name (full name) with float
    values in percent form (e.g. 84.3), or None where snaps are zero.

    If the required columns are missing from the stats DataFrame, logs a
    warning and returns an empty dict so the pipeline never crashes.
    """
    if stats.empty:
        print("WARNING: Player stats DataFrame is empty; routeParticipation will be null.", file=sys.stderr)
        return {}

    route_col = "receiving_routes_run"
    if route_col not in stats.columns:
        print(f"WARNING: '{route_col}' column not found in player stats; routeParticipation will be null.", file=sys.stderr)
        return {}

    snap_col = None
    for candidate in ("offense_snaps", "snap_counts_offense"):
        if candidate in stats.columns:
            snap_col = candidate
            break
    if snap_col is None:
        print("WARNING: No offensive snap column found in player stats; routeParticipation will be null.", file=sys.stderr)
        return {}

    name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    if name_col not in stats.columns:
        print("WARNING: No player name column found in player stats; routeParticipation will be null.", file=sys.stderr)
        return {}

    filtered = stats.copy()
    if "season" in filtered.columns:
        filtered = filtered[filtered["season"] == season]
    if "season_type" in filtered.columns:
        filtered = filtered[filtered["season_type"].isin(["REG", "POST"])]

    filtered = filtered.dropna(subset=[name_col])

    grouped = filtered.groupby(name_col).agg(
        total_routes=(route_col, "sum"),
        total_snaps=(snap_col, "sum"),
    ).reset_index()

    result: dict[str, float | None] = {}
    for _, row in grouped.iterrows():
        name = row[name_col]
        snaps = float(row["total_snaps"])
        routes = float(row["total_routes"])
        if snaps > 0:
            result[name] = round((routes / snaps) * 100, 1)
        else:
            result[name] = None

    print(f"  Route participation computed for {len(result)} players from {name_col}.")
    return result


# ---------------------------------------------------------------------------
# Weekly trend computation (2025 season only)
# ---------------------------------------------------------------------------

def compute_weekly_trends(pbp_2025: pd.DataFrame, player_name: str) -> dict:
    """
    Compute rolling weekly usage trends from 2025 data only.
    All trend values are floats in percent form (e.g. +7.4 means +7.4%).
    Fields that cannot be computed are returned as None.
    """
    mask = pd.Series(False, index=pbp_2025.index)
    for col in ["passer_player_name", "receiver_player_name", "rusher_player_name"]:
        if col in pbp_2025.columns:
            mask |= (pbp_2025[col] == player_name)

    player_plays = pbp_2025[mask]

    if player_plays.empty or "week" not in player_plays.columns:
        return {
            "rollingSnapTrend": None,
            "rollingTargetTrend": None,
            "usageVolatility": None,
            "roleStability": None,
            "routeGrowth": None,
        }

    weekly_plays = player_plays.groupby("week").size().reset_index(name="plays")

    if len(weekly_plays) < 4:
        std_val = float(weekly_plays["plays"].std()) if len(weekly_plays) > 1 else 0.0
        return {
            "rollingSnapTrend": None,
            "rollingTargetTrend": None,
            "usageVolatility": round(std_val, 2),
            "roleStability": None,
            "routeGrowth": None,
        }

    plays_arr = weekly_plays["plays"].values
    n = len(plays_arr)
    last4 = float(plays_arr[max(n - 4, 0):].mean())
    prior4 = float(plays_arr[max(n - 8, 0):max(n - 4, 0)].mean()) if n > 4 else last4

    # Percent change between the two 4-week windows, already in percent form
    snap_trend_pct = round(((last4 - prior4) / (prior4 + 1e-9)) * 100, 2)
    volatility = round(float(np.std(plays_arr)), 2)
    mean_plays = float(np.mean(plays_arr))
    cv = volatility / (mean_plays + 1e-9)
    role_stability = round(max(1.0, min(10.0, 10.0 - cv * 10)), 1)

    return {
        "rollingSnapTrend": snap_trend_pct,
        "rollingTargetTrend": round(snap_trend_pct * 0.8, 2),
        "usageVolatility": volatility,
        "roleStability": role_stability,
        "routeGrowth": round(snap_trend_pct * 0.6, 2),
    }


# ---------------------------------------------------------------------------
# Main metric assembly
# ---------------------------------------------------------------------------

def compute_metrics(
    pbp_all: pd.DataFrame,
    pbp_2025: pd.DataFrame,
    stats: pd.DataFrame,
    sleeper_players: list[dict],
    nfl_players: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Assemble per-player metric objects.

    Dashboard metrics (games, shares, trends) → pbp_2025 only.
    Context labels (careerArc, threeYearContext) → career_stats from pbp_all.
    Route participation → player stats for CURRENT_SEASON.
    """
    if nfl_players is None:
        nfl_players = pd.DataFrame()

    player_lookup = build_player_lookup(sleeper_players)
    abbreviated_lookup = build_abbreviated_lookup(sleeper_players)
    team_disambig = build_team_disambiguation(sleeper_players)
    exp_lookup = build_exp_lookup(nfl_players)

    print("Computing career context from all seasons...")
    career_stats = compute_career_stats(pbp_all)

    print("Computing route participation from player stats...")
    route_participation = compute_route_participation(stats, CURRENT_SEASON)

    print(f"Computing 2025-season passing metrics...")
    passing = compute_passing_metrics(pbp_2025)

    print(f"Computing 2025-season receiving metrics...")
    receiving = compute_receiving_metrics(pbp_2025)

    print(f"Computing 2025-season rushing metrics...")
    rushing = compute_rushing_metrics(pbp_2025)

    # Team rush attempts from 2025 only (for RB snap share denominator)
    team_rush_attempts = (
        pbp_2025[pbp_2025["rush_attempt"] == 1].groupby("posteam")["rush_attempt"].sum()
    )

    # Team dropbacks from 2025 only (all pass_attempt==1, including sacks)
    # Used as the denominator for QB snap share so backups get realistic values
    # instead of the old hardcoded 95%.
    team_dropbacks = (
        pbp_2025[pbp_2025["pass_attempt"] == 1].groupby("posteam")["pass_attempt"].sum()
    )

    # Pre-aggregate QB dropbacks (including sacks) as the numerator
    qb_dropbacks_map: dict[str, int] = {
        str(k): int(v)
        for k, v in pbp_2025[pbp_2025["pass_attempt"] == 1]
        .groupby("passer_player_name")["pass_attempt"]
        .sum()
        .items()
    }

    # Pre-aggregate QB rushing stats (rush_attempt plays where the rusher is a QB).
    # Uses the same abbreviated-name convention as passer_player_name so lookups match.
    _qb_rush_plays = pbp_2025[pbp_2025["rush_attempt"] == 1]
    qb_rush_agg: dict[str, dict] = {}
    if not _qb_rush_plays.empty and "rusher_player_name" in _qb_rush_plays.columns:
        qb_rush_agg = (
            _qb_rush_plays.groupby("rusher_player_name")
            .agg(
                rush_att=("rush_attempt", "sum"),
                rush_yds=("yards_gained",  "sum"),
                rush_td= ("touchdown",     "sum"),
            )
            .to_dict("index")
        )

    # play_action_rate — official pass plays where play_action == 1.
    # Column only present after pull script is re-run with updated PBP_COLUMNS.
    qb_play_action_map: dict[str, int] = {}
    if "play_action" in pbp_2025.columns:
        _pa_source = pbp_2025[
            (pbp_2025["pass_attempt"] == 1) &
            (pbp_2025["sack"].fillna(0) != 1) &
            (pbp_2025["play_action"].fillna(0) == 1)
        ]
        qb_play_action_map = (
            _pa_source.groupby("passer_player_name")["play_action"]
            .count()
            .to_dict()
        )

    # scramble_rate — plays flagged qb_scramble == 1 by rusher name.
    # Scrambles are dropbacks where the QB ran rather than threw; they appear in
    # the PBP as rush plays (rush_attempt==1, qb_scramble==1).
    # designed_rush_rate = total_qb_rush - scrambles, exposed as a separate field.
    # Column only present after pull script is re-run with updated PBP_COLUMNS.
    qb_scramble_map: dict[str, int] = {}
    if "qb_scramble" in pbp_2025.columns:
        _sc_source = pbp_2025[pbp_2025["qb_scramble"].fillna(0) == 1]
        if not _sc_source.empty and "rusher_player_name" in _sc_source.columns:
            qb_scramble_map = (
                _sc_source.groupby("rusher_player_name")["qb_scramble"]
                .count()
                .to_dict()
            )

    # Red zone context from 2025 only
    rz_col = "yardline_100" in pbp_2025.columns
    if rz_col:
        rz_pbp = pbp_2025[pbp_2025["yardline_100"] <= 20]
        total_rz = len(rz_pbp)
    else:
        rz_pbp = pd.DataFrame()
        total_rz = 0

    all_players: list[dict] = []

    # -----------------------------------------------------------------------
    # Pass catchers (WR / TE / RB receiving role)
    # -----------------------------------------------------------------------
    for _, row in receiving.iterrows():
        name = row["player_name"]  # PBP abbreviated name
        if pd.isna(name):
            continue

        attempts = int(row.get("attempts", 0))
        if attempts < MIN_PLAYS:
            continue

        full_name = resolve_full_name(
            name, player_lookup, abbreviated_lookup,
            team_disambig=team_disambig, pbp_team=str(row.get("team", "")),
        )
        sleeper_info = player_lookup.get(full_name, {})
        sleeper_pos = sleeper_info.get("position", "")
        if sleeper_pos == "QB":
            continue
        team = str(row.get("team", sleeper_info.get("team", "UNK")))
        pos = sleeper_info.get("position", row.get("pos", "WR"))
        age = sleeper_info.get("age")
        games = int(row.get("games", 1))

        # Career context (all seasons)
        ctx = career_stats.get(name, {"career_games": games, "career_epa_per_play": 0.0})
        career_games = ctx["career_games"]
        career_epa_per_play = ctx["career_epa_per_play"]

        # 2025-season efficiency
        total_epa = float(row.get("total_epa", 0))
        successful = float(row.get("successful_plays", 0))
        air_yards = float(row.get("air_yards_total", 0))
        deep_tgts = int(row.get("deep_targets", 0))
        explosive = int(row.get("explosive_plays", 0))
        touchdowns = int(row.get("touchdowns", 0))
        yards_total = float(row.get("yards_gained_total", 0))

        epa_per_play = round(total_epa / attempts, 3) if attempts else 0.0
        success_rate = safe_pct(successful, attempts)
        explosive_rate = safe_pct(explosive, attempts)
        deep_target_rate = safe_pct(deep_tgts, attempts)

        # 2025-season share metrics (all in percent form via safe_pct)
        team_air_yds = float(row.get("team_air_yards_per_game", 0) or 0) * games
        team_tgts = float(row.get("team_targets_per_game", 0) or 0) * games
        air_yards_share = safe_pct(air_yards, team_air_yds) if team_air_yds > 0 else None
        target_share = safe_pct(attempts, team_tgts) if team_tgts > 0 else None

        # WOPR stays on 0-~1.5 index scale (not percent)
        if target_share is not None and air_yards_share is not None:
            wopr = round(1.5 * (target_share / 100) + 0.7 * (air_yards_share / 100), 3)
        else:
            wopr = None

        # Approximate snap share from target share (percent scale)
        snap_share = min(99.9, round((target_share or 0) * 2.8, 1))
        # snapTrend: percent-point change from the target-implied snap baseline
        snap_trend_val = round((snap_share - 70) * 0.3, 2)

        # Red zone utilization (2025 only, percent scale)
        if rz_col and total_rz > 0 and "receiver_player_name" in rz_pbp.columns:
            rz_targets = int((rz_pbp["receiver_player_name"] == name).sum())
            red_zone_util = safe_pct(rz_targets, total_rz)
        else:
            red_zone_util = None

        expected_tds = round(attempts * 0.05, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        actual_fp = yards_total * 0.1 + touchdowns * 6
        fpoe = round(actual_fp * 0.18, 2)

        # Trends from 2025 data only
        trends = compute_weekly_trends(pbp_2025, name)
        usage_vol = trends["usageVolatility"] or 0.0
        role_stab = trends["roleStability"] or 5.0

        # Context labels use career data
        career_arc = classify_career_arc(pos, age, career_epa_per_play, snap_share, career_games, games)
        three_year = classify_three_year_context(pos, career_epa_per_play, snap_share, career_games, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        all_players.append({
            "player": full_name,
            "team": team,
            "pos": pos,
            "age": age,
            "games": games,
            "snapShare": snap_share,
            "targetShare": round(target_share, 1) if target_share is not None else None,
            "airYardsShare": round(air_yards_share, 1) if air_yards_share is not None else None,
            "wopr": wopr,
            "redZoneUtil": round(red_zone_util, 1) if red_zone_util is not None else None,
            "goalLineCarries": 0,
            "snapTrend": snap_trend_val,
            "routeParticipation": route_participation.get(full_name, route_participation.get(name)),
            "threeYearContext": three_year,
            "careerArc": career_arc,
            "epaPlay": epa_per_play,
            "successRate": success_rate,
            "tdOverExpected": td_over_expected,
            "fpoe": fpoe,
            "explosivePlayRate": explosive_rate,
            "breakawayRunRate": 0.0,
            "deepTargetRate": round(deep_target_rate, 1),
            "sustainability": sustainability,
            "rollingSnapTrend": trends["rollingSnapTrend"],
            "rollingTargetTrend": trends["rollingTargetTrend"],
            "usageVolatility": trends["usageVolatility"],
            "roleStability": trends["roleStability"],
            "routeGrowth": trends["routeGrowth"],
        })

    # -----------------------------------------------------------------------
    # Pure rushers (RB / FB)
    # -----------------------------------------------------------------------
    for _, row in rushing.iterrows():
        name = row["player_name"]
        if pd.isna(name):
            continue

        full_name = resolve_full_name(
            name, player_lookup, abbreviated_lookup,
            team_disambig=team_disambig, pbp_team=str(row.get("team", "")),
            pos_hint="RB",
        )
        if any(p["player"] == full_name and p.get("pos") in ("RB", "FB") for p in all_players):
            continue

        sleeper_info = player_lookup.get(full_name, {})
        # QBs who also rush are processed in the QB loop with their full
        # passing stats — if we let them through here their entire stat card
        # would be built from rushing plays only (e.g. 22 carries instead of
        # 212 pass attempts), producing wildly wrong EPA and success rate.
        if sleeper_info.get("position") == "QB":
            continue

        attempts = int(row.get("attempts", 0))
        if attempts < MIN_PLAYS:
            continue

        team = str(row.get("team", sleeper_info.get("team", "UNK")))
        pos = sleeper_info.get("position", "RB")
        age = sleeper_info.get("age")
        games = int(row.get("games", 1))

        ctx = career_stats.get(name, {"career_games": games, "career_epa_per_play": 0.0})
        career_games = ctx["career_games"]
        career_epa_per_play = ctx["career_epa_per_play"]

        total_epa = float(row.get("total_epa", 0))
        successful = float(row.get("successful_plays", 0))
        explosive = int(row.get("explosive_plays", 0))
        breakaway = int(row.get("breakaway_runs", 0))
        gl_carries = int(row.get("goal_line_carries", 0))
        touchdowns = int(row.get("touchdowns", 0))
        yards_total = float(row.get("yards_gained_total", 0))

        epa_per_play = round(total_epa / attempts, 3) if attempts else 0.0
        success_rate = safe_pct(successful, attempts)
        explosive_rate = safe_pct(explosive, attempts)
        breakaway_rate = safe_pct(breakaway, attempts)

        # Snap share from 2025 team rush attempts (percent scale)
        team_rushes = float(team_rush_attempts.get(team, attempts))
        snap_share = min(99.9, round(safe_pct(attempts, team_rushes) * 1.5, 1))
        snap_trend_val = round((snap_share - 50) * 0.2, 2)

        expected_tds = round(attempts * 0.04, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        actual_fp = yards_total * 0.1 + touchdowns * 6
        fpoe = round(actual_fp * 0.18, 2)

        trends = compute_weekly_trends(pbp_2025, name)
        usage_vol = trends["usageVolatility"] or 0.0
        role_stab = trends["roleStability"] or 5.0

        career_arc = classify_career_arc(pos, age, career_epa_per_play, snap_share, career_games, games)
        three_year = classify_three_year_context(pos, career_epa_per_play, snap_share, career_games, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        all_players.append({
            "player": full_name,
            "team": team,
            "pos": pos,
            "age": age,
            "games": games,
            "snapShare": snap_share,
            # Receiver opportunity metrics are null for pure rushers
            "targetShare": None,
            "airYardsShare": None,
            "wopr": None,
            "redZoneUtil": None,
            "goalLineCarries": gl_carries,
            "snapTrend": snap_trend_val,
            "routeParticipation": None,
            "threeYearContext": three_year,
            "careerArc": career_arc,
            "epaPlay": epa_per_play,
            "successRate": success_rate,
            "tdOverExpected": td_over_expected,
            "fpoe": fpoe,
            "explosivePlayRate": explosive_rate,
            "breakawayRunRate": round(breakaway_rate, 1),
            "deepTargetRate": None,
            "sustainability": sustainability,
            "rollingSnapTrend": trends["rollingSnapTrend"],
            "rollingTargetTrend": None,   # not applicable for pure rushers
            "usageVolatility": trends["usageVolatility"],
            "roleStability": trends["roleStability"],
            "routeGrowth": None,          # not applicable for pure rushers
        })

    # -----------------------------------------------------------------------
    # Quarterbacks
    # -----------------------------------------------------------------------
    for _, row in passing.iterrows():
        name = row["player_name"]
        if pd.isna(name):
            continue

        full_name = resolve_full_name(
            name, player_lookup, abbreviated_lookup,
            team_disambig=team_disambig, pbp_team=str(row.get("team", "")),
            pos_hint="QB",
        )
        if any(p["player"] == full_name and p.get("pos") == "QB" for p in all_players):
            continue

        attempts = int(row.get("attempts", 0))
        if attempts < MIN_PLAYS:
            continue

        sleeper_info = player_lookup.get(full_name, {})
        sleeper_pos = sleeper_info.get("position", "")
        if sleeper_pos and sleeper_pos in ("WR", "TE", "RB", "FB"):
            continue
        team = str(row.get("team", sleeper_info.get("team", "UNK")))
        pos = "QB"
        age = sleeper_info.get("age")
        games = int(row.get("games", 1))

        ctx = career_stats.get(name, {"career_games": games, "career_epa_per_play": 0.0})
        career_games = ctx["career_games"]
        career_epa_per_play = ctx["career_epa_per_play"]

        total_epa = float(row.get("total_epa", 0))
        successful = float(row.get("successful_plays", 0))
        explosive = int(row.get("explosive_plays", 0))
        deep_tgts = int(row.get("deep_targets", 0))
        touchdowns = int(row.get("touchdowns", 0))
        rz_att_count = int(row.get("rz_attempts", 0))
        yards_total  = int(row.get("yards_total", 0))
        air_yards_total = float(row.get("air_yards_total", 0))

        # QB rushing stats (matched by the same abbreviated PBP name)
        _rd      = qb_rush_agg.get(name, {})
        rush_att = int(_rd.get("rush_att", 0))
        rush_yds = int(_rd.get("rush_yds", 0))
        rush_td  = int(_rd.get("rush_td",  0))

        epa_per_play     = round(total_epa / attempts, 3) if attempts else 0.0
        success_rate     = safe_pct(successful, attempts)
        explosive_rate   = safe_pct(explosive, attempts)
        deep_target_rate = safe_pct(deep_tgts, attempts)
        ypa              = round(yards_total / attempts, 2) if attempts else 0.0
        air_yards_per_att = round(air_yards_total / attempts, 2) if attempts else 0.0

        expected_tds     = round(attempts * 0.055, 1)
        td_over_expected = round(touchdowns - expected_tds, 2)

        # Real QB snap share: this QB's dropbacks / team's total dropbacks × 100.
        qb_dbs   = qb_dropbacks_map.get(name, attempts)
        team_dbs = int(team_dropbacks.get(team, max(1, qb_dbs)))
        snap_share = round(min(99.9, (qb_dbs / team_dbs) * 100), 1)
        snap_trend_val = 0.0

        # Derived per-game / alias fields
        dropbacks_per_game = round(qb_dbs / max(1, games), 1)
        rush_att_per_game  = round(rush_att / max(1, games), 2)
        deep_attempt_rate  = round(deep_target_rate, 1)   # same metric, preferred key

        # play_action_rate and scramble_rate — None until pull script is re-run
        # with the updated PBP_COLUMNS that include play_action and qb_scramble.
        play_action_rate: float | None = None
        scramble_rate: float | None    = None
        designed_rush_rate: float | None = None
        if qb_play_action_map:
            pa_count = int(qb_play_action_map.get(name, 0))
            play_action_rate = round(safe_pct(pa_count, qb_dbs), 1)
        if qb_scramble_map:
            sc_count = int(qb_scramble_map.get(name, 0))
            scramble_rate    = round(safe_pct(sc_count, qb_dbs), 1)
            # designed runs = total QB rushes minus scrambles
            designed_rush_count = max(0, rush_att - sc_count)
            designed_rush_rate  = round(safe_pct(designed_rush_count, qb_dbs), 1)

        # Experience tier — prefer nflverse years_of_experience (true career
        # length); fall back to estimating from our 3-season PBP window.
        years_exp = exp_lookup.get(full_name) or years_exp_from_games(career_games)
        experience_tier = classify_experience_tier(years_exp)

        trends = compute_weekly_trends(pbp_2025, name)
        usage_vol = trends["usageVolatility"] or 0.0
        role_stab = trends["roleStability"] or 9.0

        career_arc    = classify_career_arc(pos, age, career_epa_per_play, snap_share, career_games, games)
        three_year    = classify_three_year_context(pos, career_epa_per_play, snap_share, career_games, games)
        sustainability = classify_sustainability(success_rate, usage_vol, role_stab, games)

        all_players.append({
            "player": full_name,
            "team": team,
            "pos": pos,
            "age": age,
            "games": games,
            "snapShare": snap_share,
            # Receiver opportunity metrics are not applicable for QBs
            "targetShare": None,
            "airYardsShare": None,
            "wopr": None,
            "redZoneUtil": None,
            "goalLineCarries": 0,
            "snapTrend": snap_trend_val,
            "routeParticipation": None,
            "threeYearContext": three_year,
            "careerArc": career_arc,
            "epaPlay": epa_per_play,
            "successRate": success_rate,
            "tdOverExpected": td_over_expected,
            "fpoe": round(epa_per_play * attempts * 0.3, 2),
            "explosivePlayRate": explosive_rate,
            "breakawayRunRate": None,        # not applicable for QBs
            "deepTargetRate": round(deep_target_rate, 1),
            "sustainability": sustainability,
            "rollingSnapTrend": trends["rollingSnapTrend"],
            "rollingTargetTrend": None,      # not applicable for QBs
            "usageVolatility": trends["usageVolatility"],
            "roleStability": trends["roleStability"],
            "routeGrowth": None,             # not applicable for QBs
            # QB-specific advanced fields
            "rzAtt": rz_att_count,
            "rushAtt": rush_att,
            "rushYds": rush_yds,
            "rushTd": rush_td,
            "dropbacksPerGame": dropbacks_per_game,
            "rushAttPerGame": rush_att_per_game,
            "deepAttemptRate": deep_attempt_rate,
            "ypa": ypa,
            "airYardsPerAtt": air_yards_per_att,
            "careerGames": career_games,
            "experienceTier": experience_tier,
            # Scheme/role rates — None until pull_nflverse_data re-runs with new PBP_COLUMNS
            "playActionRate": play_action_rate,
            "scrambleRate": scramble_rate,
            "designedRushRate": designed_rush_rate,
            # Composite scores — filled in second pass after all QBs are collected
            "relativeEfficiency": None,
            "efficiencyScore": None,
            "opportunityScore": None,
            "usageScore": None,
            "playerScore": None,
        })

    # ─── Post-loop: QB composite scores ───────────────────────────────────────
    # Composite scores need the full QB distribution for z-scores, so they are
    # computed in a second pass after all QB records are collected.
    _qbs = [p for p in all_players if p.get("pos") == "QB"]

    if len(_qbs) >= 3:

        def _extract(field: str) -> list[float]:
            return [float(p.get(field) or 0) for p in _qbs]

        def _z_scores(vals: list[float]) -> list[float]:
            arr = np.array(vals, dtype=float)
            std = float(arr.std())
            if std < 1e-9:
                return [0.0] * len(vals)
            return list((arr - arr.mean()) / std)

        def _clip_scale(z: float, clip: float = 2.5) -> float:
            """Clamp z to [-clip, clip], then map linearly to [0, 100]."""
            return round(max(0.0, min(100.0, (max(-clip, min(clip, z)) + clip) / (2 * clip) * 100)), 1)

        # Season-average EPA/play used as baseline for relative_efficiency.
        season_avg_epa = round(float(np.mean(_extract("epaPlay"))), 3)

        # efficiency_score inputs (approved weights 0.35/0.25/0.15/0.10/0.10/0.05)
        z_epa  = _z_scores(_extract("epaPlay"))
        z_sr   = _z_scores(_extract("successRate"))
        z_ypa  = _z_scores(_extract("ypa"))
        z_air  = _z_scores(_extract("airYardsPerAtt"))
        z_exp  = _z_scores(_extract("explosivePlayRate"))
        z_tdo  = _z_scores(_extract("tdOverExpected"))

        # opportunity_score inputs (approved weights 0.40/0.30/0.20/0.10)
        z_snap = _z_scores(_extract("snapShare"))
        z_dbpg = _z_scores(_extract("dropbacksPerGame"))
        z_rz   = _z_scores(_extract("rzAtt"))
        z_rapg = _z_scores(_extract("rushAttPerGame"))

        # usage_score inputs (approved weights 0.45/0.30/0.20/0.05; volatility negated)
        z_stab  = _z_scores(_extract("roleStability"))
        z_vol   = _z_scores(_extract("usageVolatility"))
        z_trend = _z_scores(_extract("rollingSnapTrend"))
        z_dar   = _z_scores(_extract("deepAttemptRate"))

        for qi, qb in enumerate(_qbs):
            rel_eff = round(float(qb.get("epaPlay") or 0) - season_avg_epa, 3)

            eff_raw = (
                0.35 * z_epa[qi]  +
                0.25 * z_sr[qi]   +
                0.15 * z_ypa[qi]  +
                0.10 * z_air[qi]  +
                0.10 * z_exp[qi]  +
                0.05 * z_tdo[qi]
            )
            opp_raw = (
                0.40 * z_snap[qi] +
                0.30 * z_dbpg[qi] +
                0.20 * z_rz[qi]   +
                0.10 * z_rapg[qi]
            )
            usg_raw = (
                0.45 *  z_stab[qi]  +
                0.30 * (-z_vol[qi]) +   # lower volatility = better
                0.20 *  z_trend[qi] +
                0.05 *  z_dar[qi]
            )

            eff_score = _clip_scale(eff_raw)
            opp_score = _clip_scale(opp_raw)
            usg_score = _clip_scale(usg_raw)
            p_score   = round(0.45 * eff_score + 0.35 * opp_score + 0.20 * usg_score, 1)

            qb["relativeEfficiency"] = rel_eff
            qb["efficiencyScore"]    = eff_score
            qb["opportunityScore"]   = opp_score
            qb["usageScore"]         = usg_score
            qb["playerScore"]        = p_score
            qb["seasonAvgEpa"]       = season_avg_epa   # baseline for tooltip

    print(f"Total players processed: {len(all_players)}")
    return all_players


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_metrics(players: list[dict]) -> None:
    """Save computed metrics to JSON."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(players, f, indent=2)
    print(f"Saved metrics for {len(players)} players to {OUTPUT_PATH}")


def main():
    pbp_all, pbp_2025, stats, sleeper_players, nfl_players = load_data()
    players = compute_metrics(pbp_all, pbp_2025, stats, sleeper_players, nfl_players)
    save_metrics(players)
    print("Metric computation complete.")


if __name__ == "__main__":
    main()
