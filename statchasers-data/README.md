# StatChasers Data Pipeline

A Python data pipeline that automatically pulls NFL player data, computes advanced analytics metrics, and exports clean JSON files for the [StatChasers](https://statchasers.com) Performance Analytics dashboard.

---

## What This Does

1. **Pulls player identity data** from the Sleeper API (names, teams, positions, ages)
2. **Pulls play-by-play data** from nflverse via `nflreadpy` (EPA, success rate, targets, snaps, etc.)
3. **Computes advanced metrics** per player (WOPR, air yards share, EPA/play, explosive rate, trend lines, etc.)
4. **Exports clean JSON files** optimized for fast web fetches — no raw play-by-play, no unnecessary fields

---

## Repository Structure

```
statchasers-data/
│
├── scripts/
│   ├── pull_sleeper_players.py      # Step 1: Fetch player metadata from Sleeper API
│   ├── pull_nflverse_data.py        # Step 2: Download play-by-play from nflverse
│   ├── compute_player_metrics.py    # Step 3: Compute all advanced metrics
│   └── build_performance_analytics.py  # Step 4: Assemble and export final JSON
│
├── data/
│   ├── raw/                         # Raw source data (not committed except Sleeper JSON)
│   └── processed/                   # Intermediate computed metrics
│
├── output/
│   ├── performance_analytics_latest.json  # Always the most current data
│   └── performance_analytics_2025.json    # Season-specific snapshot
│
├── requirements.txt
├── README.md
└── .github/workflows/update-data.yml     # Automated weekly update workflow
```

---

## Data Sources

### Sleeper API
- **Endpoint**: `https://api.sleeper.app/v1/players/nfl`
- **No API key required** — public endpoint
- Provides: player names, teams, positions, birth dates
- Used to compute: player age, identity matching

### nflverse / nflfastR (via `nflreadpy`)
- Play-by-play data for 2024 and 2025 seasons
- Provides: EPA, success rate, yards, air yards, targets, rush attempts, touchdowns, snap data
- Automatically downloaded via `nflreadpy.load_pbp()` and `nflreadpy.load_player_stats()`

---

## Metrics Computed

Each player object includes:

| Category | Metrics |
|---|---|
| **Identity** | player, team, pos, age, games |
| **Usage** | snapShare, targetShare, airYardsShare, wopr, redZoneUtil, goalLineCarries |
| **Efficiency** | epaPlay, successRate, tdOverExpected, fpoe, explosivePlayRate, breakawayRunRate, deepTargetRate |
| **Trends** | snapTrend, rollingSnapTrend, rollingTargetTrend, usageVolatility, roleStability, routeGrowth |
| **Context** | threeYearContext, careerArc, sustainability |

---

## Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (in order)

```bash
python scripts/pull_sleeper_players.py
python scripts/pull_nflverse_data.py
python scripts/compute_player_metrics.py
python scripts/build_performance_analytics.py
```

Each script can also be run individually for debugging.

### 3. Output files

After running, the output directory will contain:

- `output/performance_analytics_latest.json` — current season data
- `output/performance_analytics_2025.json` — 2025 season snapshot

---

## GitHub Actions Automation

The workflow at `.github/workflows/update-data.yml`:

- **Schedule**: Runs every **Tuesday at 6 AM UTC** (covers Monday Night Football data)
- **Manual trigger**: Can be run on-demand via the GitHub Actions UI
- **Steps**: Installs dependencies → runs all 4 scripts in order → commits updated JSON back to the repo

### Setting up the workflow

1. Push this repository to GitHub
2. No secrets required — all data sources are public
3. The workflow will auto-commit updated JSON files with timestamped commit messages

---

## Frontend Integration

The StatChasers frontend fetches data via:

```
https://raw.githubusercontent.com/<your-org>/statchasers-data/main/output/performance_analytics_latest.json
```

### Output format

```json
{
  "meta": {
    "generated_at": "2025-10-14T06:12:33+00:00",
    "season": 2025,
    "player_count": 312,
    "source": "StatChasers Data Pipeline"
  },
  "players": [
    {
      "player": "Ja'Marr Chase",
      "team": "CIN",
      "pos": "WR",
      "age": 25,
      "games": 17,
      "snapShare": 89.4,
      "targetShare": 29.1,
      "airYardsShare": 41.3,
      "wopr": 0.78,
      "redZoneUtil": 31.2,
      "goalLineCarries": 0,
      "snapTrend": "+7.4%",
      "threeYearContext": "Elite WR1 trajectory",
      "careerArc": "Ascending",
      "epaPlay": 0.18,
      "successRate": 57.2,
      "tdOverExpected": 3.1,
      "fpoe": 18.4,
      "explosivePlayRate": 14.7,
      "breakawayRunRate": 0.0,
      "deepTargetRate": 22.1,
      "sustainability": "Sustainable",
      "rollingSnapTrend": "+9.1%",
      "rollingTargetTrend": "+5.4%",
      "usageVolatility": 4.2,
      "roleStability": 8.9,
      "routeGrowth": "+6.8%"
    }
  ]
}
```

---

## Notes

- The pipeline only outputs **QB, WR, RB, TE** positions
- Players with fewer than 2 games or 20 plays are excluded from output
- Parquet files in `data/raw/` are not committed to git (large binary files) — they are regenerated on each pipeline run
- `data/raw/sleeper_players.json` **is** committed so the player lookup is available without re-running the Sleeper pull
