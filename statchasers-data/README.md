# StatChasers Data Pipeline

A Python data pipeline that pulls NFL data from nflverse and exports clean
per-position **Advanced Stats** JSON files for the
[StatChasers](https://statchasers.com) dashboard.

The pipeline produces one Advanced Stats dataset per position (QB, RB, WR, TE),
each covering the **2023, 2024 and 2025** regular seasons plus a combined
all-seasons file.

---

## What This Does

1. **Pulls player identity data** from the Sleeper API (names, teams, positions) for `playerId` resolution.
2. **Pulls data** from nflverse via `nflreadpy` — play-by-play, season-total player stats, PFR advanced stats, NextGen Stats, snap counts, participation, and the player registry — stored as Parquet under `data/raw/`.
3. **Builds the Advanced Stats datasets** — one self-contained builder per position, joined on the nflverse `gsis` id and resolved to a Sleeper `playerId`.

---

## Repository Structure

```
statchasers-data/
│
├── scripts/
│   ├── pull_sleeper_players.py       # Fetch player metadata from the Sleeper API
│   ├── pull_nflverse_data.py         # Download nflverse data → data/raw/*.parquet
│   ├── schema_config.py              # PlayerIdResolver (name → Sleeper playerId)
│   ├── build_qb_advanced_stats.py    # QB Advanced Stats
│   ├── build_rb_advanced_stats.py    # RB Advanced Stats
│   ├── build_wr_advanced_stats.py    # WR Advanced Stats
│   └── build_te_advanced_stats.py    # TE Advanced Stats
│
├── data/
│   ├── raw/                          # Raw source parquet (not committed except sleeper_players.json)
│   └── processed/                    # (unused intermediate dir)
│
├── output/
│   ├── qb_advanced_stats_{2023,2024,2025,all}.json   (+ qb_advanced_stats.json alias → 2025)
│   ├── rb_advanced_stats_{2023,2024,2025,all}.json   (+ rb_advanced_stats.json alias → 2025)
│   ├── wr_advanced_stats_{2023,2024,2025,all}.json   (+ wr_advanced_stats.json alias → 2025)
│   └── te_advanced_stats_{2023,2024,2025,all}.json   (+ te_advanced_stats.json alias → 2025)
│
├── requirements.txt
├── README.md
└── .github/workflows/update-data.yml # Automated weekly update workflow
```

Each output file uses the envelope:

```json
{
  "updated_at": "...",
  "season": 2025,
  "week": 18,
  "table": "wr_advanced_stats",
  "columns": [ { "key": "...", "label": "...", "type": "...", "defaultVisible": true }, ... ],
  "rows":    [ { "playerId": "6794", "playerName": "Justin Jefferson", ... }, ... ]
}
```

Every row leads with the shared identity fields: `playerId`, `playerName`,
`position`, `team`, `age`, `season`, `games`. Counting totals come from official
season `player_stats`; `snapPct` is the true season snap share (mean per-game
`offense_pct`); `routes` is participation-based (skill-position appearances on
pass plays); `playerId` is resolved against the Sleeper registry.

---

## Data Sources

### Sleeper API
- **Endpoint**: `https://api.sleeper.app/v1/players/nfl` (public, no key)
- Provides player names / teams / positions for `playerId` resolution.

### nflverse / nflfastR (via `nflreadpy`)
Pulled by `pull_nflverse_data.py` into `data/raw/`:
- **Play-by-play** (`load_pbp`) — EPA, success, air yards, distance buckets, red-zone / end-zone, longest plays.
- **Season-total player stats** (`load_player_stats(summary_level="reg")`) — counting stats, receiving (incl. `target_share`, `air_yards_share`, `wopr`), fantasy points, fumbles, `passing_cpoe`, sack data.
- **PFR advanced stats** (`load_pfr_advstats`, weekly + seasonal) — pass: pocket time, pressure %, blitz/hurry/knockdown, scrambles; rush: YBC/YAC, broken tackles; rec: broken tackles, drops, INTs when targeted.
- **NextGen Stats passing** (`load_nextgen_stats`, season row) — average time to throw.
- **Snap counts** (`load_snap_counts`) — true snap share.
- **Participation** (`load_participation`) — routes.
- **Player registry** (`load_players`) — `birth_date` (age) and `pfr_id`↔`gsis` mapping.

---

## The Advanced Stats datasets

### QB Advanced Stats
`build_qb_advanced_stats.py` → `output/qb_advanced_stats_{2023,2024,2025,all}.json`
(+ `qb_advanced_stats.json` alias for 2025) — a volume / pressure leaderboard
covering every QB with ≥1 regular-season pass attempt. Columns include passing &
rushing counting stats, distance buckets, deep-attempt %, fantasy points,
efficiency metrics (`epaPerPlay`, `successRate`, `cpoe`), and PFR/NextGen
pressure metrics (pocket time, time to throw, blitz/hurry/knockdown, pressure %).
`epaPerPlay` and `successRate` are computed over **all** of the QB's plays — every
play where he is the passer or rusher (pass attempts, sacks, scrambles, designed
runs and kneels), each counted once — which matches the standard public
"QB EPA per play" denominator. `cpoe` is the season completion-percentage-over-expected
from `passing_cpoe`.

### RB Advanced Stats
`build_rb_advanced_stats.py` → `output/rb_advanced_stats_{2023,2024,2025,all}.json`
(+ alias) — a volume / efficiency / elusiveness leaderboard covering every RB
with ≥1 regular-season touch. Columns include rushing & receiving counting stats,
usage (`snapPct`, `routes`, `redZoneOpportunities`, `redZoneTargets`,
`goalLineCarries`, `endZoneTargets`, `targetSharePct`), efficiency (`epaPerPlay`,
`explosiveRunPct`, `breakawayRunPct`, `yardsPerRouteRun`), distance buckets, and
PFR contact/elusiveness metrics (`yardsBeforeContactPerAttempt`,
`yardsAfterContactPerAttempt`, `brokenTackles`, `rushAttemptsPerBrokenTackle`,
`tackleEludedRate`). Notes:

- `explosiveRunPct` (rush ≥10 yds), `breakawayRunPct` (rush ≥15 yds), broken
  tackles and YBC/YAC per attempt match the established PFR-based definitions.
- `epaPerPlay` is EPA over all plays the RB was involved in (carries + targets).
- `longestRushTouchdown` is the yardage of the player's longest rushing TD.
- PFR provides no tackles-for-loss field for rushers, so `tacklesForLoss` /
  `tacklesForLossYards` are derived from negative-yardage rushes and carry the
  same values as `rushAttForNegativeYards`.

### WR Advanced Stats
`build_wr_advanced_stats.py` → `output/wr_advanced_stats_{2023,2024,2025,all}.json`
(+ alias) — a target-volume / efficiency / separation leaderboard covering every
WR with ≥1 regular-season target. Columns include target & route usage
(`snapPct`, `routes`, `targetsPerRouteRun`, `targetSharePct`, `airYardsSharePct`,
`wopr`, `airYardsPerTarget`), receiving production, efficiency (`epaPerPlay`,
`successRate`, `yardsPerRouteRun`, `yardsBefore/AfterCatchPerReception`),
reception buckets, and PFR contact metrics (`brokenTackles`, `drops`, `dropPct`,
`interceptionsWhenTargeted`). Notes:

- `epaPerPlay` is EPA per target; `successRate` is the % of targets with positive EPA.
- `routes` is participation-based; `routePct` = routes / team pass plays; per-route/per-rec rates use official receiving yards.
- `catchableTargets` = receptions + drops; `totalYards` = receiving + rushing yards; `fumbles` = receiving + rushing + sack fumbles; `airYardsPerReception` = air yards / receptions.
- `contestedCatchRate` and `yardsAfterContactPerReception` have **no source** in nflverse / PFR (charting / receiver-contact metrics) and are emitted as `null`.

### TE Advanced Stats
`build_te_advanced_stats.py` → `output/te_advanced_stats_{2023,2024,2025,all}.json`
(+ alias) — uses the **same column set and definitions as the WR dataset**,
filtered to TEs with ≥1 regular-season target.

---

## Running Locally

```bash
pip install -r requirements.txt

# Data ingest
python scripts/pull_sleeper_players.py
python scripts/pull_nflverse_data.py

# Build the four Advanced Stats datasets
python scripts/build_qb_advanced_stats.py
python scripts/build_rb_advanced_stats.py
python scripts/build_wr_advanced_stats.py
python scripts/build_te_advanced_stats.py
```

Each builder is self-contained — it reads the parquet files in `data/raw/` and
writes its five JSON files to `output/`.

---

## GitHub Actions Automation

`.github/workflows/update-data.yml`:

- **Schedule**: every **Tuesday at 6 AM UTC** (plus manual `workflow_dispatch`).
- **Steps**: install deps → pull Sleeper + nflverse data → run the four Advanced Stats builders → commit the updated `output/*_advanced_stats*.json` files back to the repo.
- No secrets required — all data sources are public.

---

## Frontend Integration

Fetch the files directly from GitHub raw, e.g.:

```
https://raw.githubusercontent.com/statchasersff-bit/statchasers-data/master/statchasers-data/output/wr_advanced_stats_2025.json
https://raw.githubusercontent.com/statchasersff-bit/statchasers-data/master/statchasers-data/output/qb_advanced_stats_all.json
```

The frontend renders each table from the `columns` array (`key` / `label` /
`type` / `defaultVisible`) rather than hardcoding fields.

---

## Notes

- Output covers **QB, RB, WR, TE** for the **2023–2025** regular seasons (plus a combined `all` file per position).
- Parquet files in `data/raw/` are not committed (regenerated on each run); `data/raw/sleeper_players.json` **is** committed so player lookups work without re-running the Sleeper pull.
