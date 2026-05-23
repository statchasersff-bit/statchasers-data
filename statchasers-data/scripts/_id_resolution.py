"""
_id_resolution.py
─────────────────
Shared helper for dropping name-collision plays during PBP aggregation.

Problem
-------
Build scripts resolve PBP abbreviated names (e.g. "T.Hill") + posteam to a
canonical full name and then group plays by that full name.  When two real
players share the same abbreviated name on different teams (e.g.
Tyreek Hill on MIA vs Taysom Hill on NO), the resolver can fail and merge
both players' plays under one full name — inflating games / yards / TDs
for the player who "won" the resolution.

Fix
---
After name resolution, drop any play whose PBP `<id_col>` doesn't match the
canonical gsis_id (from `nflverse_players.parquet` keyed by `display_name`).
Plays whose resolved name isn't present in nflverse are kept untouched so
rookies or rarely-tracked players don't disappear.

Usage
-----
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _id_resolution import build_canonical_id_lookup, filter_canonical_id

    lookup = build_canonical_id_lookup(nfl_df, position="WR")
    plays  = filter_canonical_id(plays, "_fn", "receiver_player_id", lookup)
"""
from __future__ import annotations

import pandas as pd


def build_canonical_id_lookup(
    nfl_df: pd.DataFrame | None,
    position: str | None = None,
) -> dict[str, str]:
    """
    Build {display_name → gsis_id} from nflverse_players.

    When a `display_name` is shared by multiple gsis_ids (e.g. David Johnson
    appears as both an RB and a TE), `position` is used as a tiebreaker so
    the lookup returns the canonical id for the position-specific script.
    """
    if (
        nfl_df is None
        or "gsis_id"      not in nfl_df.columns
        or "display_name" not in nfl_df.columns
    ):
        return {}
    df = nfl_df.dropna(subset=["gsis_id", "display_name"]).copy()
    if position is not None and "position" in df.columns:
        df["__pos_pref"] = (df["position"] != position).astype(int)
        df = df.sort_values("__pos_pref", kind="mergesort")
    return (
        df.drop_duplicates("display_name", keep="first")
          .set_index("display_name")["gsis_id"]
          .to_dict()
    )


def filter_canonical_id(
    plays: pd.DataFrame,
    name_col: str,
    id_col: str,
    lookup: dict[str, str],
) -> pd.DataFrame:
    """
    Keep only plays whose `id_col` matches the canonical id for `name_col`.

    Plays whose resolved name isn't in `lookup` pass through unchanged so
    new/unknown players aren't silently dropped.
    """
    if not lookup or id_col not in plays.columns or name_col not in plays.columns:
        return plays
    expected = plays[name_col].map(lookup)
    keep = expected.isna() | (plays[id_col].astype(str) == expected.astype(str))
    return plays[keep].copy()
