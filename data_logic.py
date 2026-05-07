import warnings
warnings.filterwarnings("ignore")

import time
import random
import sys
import json
import os
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import soccerdata as sd


CUSTOM_LEAGUES = {
    "NED-Eredivisie": {
        "FBref": "Eredivisie",
        "season_start": "Aug",
        "season_end": "May"
    },
    "POR-Primeira Liga": {
        "FBref": "Primeira Liga",
        "season_start": "Aug",
        "season_end": "May"
    },
    "BEL-First Division A": {
        "FBref": "Belgian Pro League",
        "season_start": "Aug",
        "season_end": "May"
    },
    "TUR-Super Lig": {
        "FBref": "Süper Lig",
        "season_start": "Aug",
        "season_end": "May"
    },
    "MEX-Liga MX": {
        "FBref": "Liga MX",
        "season_start": "Jan",
        "season_end": "Dec"
    },
    "USA-MLS": {
        "FBref": "Major League Soccer",
        "season_start": "Feb",
        "season_end": "Nov"
    },
    "SAU-Pro League": {
        "FBref": "Saudi Pro League",
        "season_start": "Aug",
        "season_end": "May"
    },
    "INT-Champions League": {
        "FBref": "UEFA Champions League",
        "season_start": "Aug",
        "season_end": "May"
    },
}

_sd_dir = Path(os.environ.get("SOCCERDATA_DIR", Path.home() / "soccerdata"))
_config_dir = _sd_dir / "config"
_config_dir.mkdir(parents=True, exist_ok=True)
_league_dict_path = _config_dir / "league_dict.json"

_existing = {}
if _league_dict_path.exists():
    try:
        _existing = json.loads(_league_dict_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pass

_existing.update(CUSTOM_LEAGUES)
_league_dict_path.write_text(json.dumps(_existing, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Custom league_dict.json written to {_league_dict_path}")

import soccerdata._config as _sd_config
importlib.reload(_sd_config)
import soccerdata._common as _sd_common
importlib.reload(_sd_common)
import soccerdata.fbref as _sd_fbref
importlib.reload(_sd_fbref)
importlib.reload(sd)
print(f"soccerdata reloaded, available leagues: {len(sd.FBref.available_leagues())}")

SEASONS = ["2324", "2425", "2526"]
SINGLE_YEAR_SEASONS = ["2024", "2025", "2026"]

LEAGUE_SELECTOR = None

LEAGUES = {
    "ENG-Premier League":   {"tier": 1, "difficulty": 5.0, "country": "England",     "short": "Premier League",   "single_year": False},
    "ESP-La Liga":          {"tier": 1, "difficulty": 4.8, "country": "Spain",        "short": "La Liga",          "single_year": False},
    "GER-Bundesliga":       {"tier": 1, "difficulty": 4.7, "country": "Germany",      "short": "Bundesliga",       "single_year": False},
    "ITA-Serie A":          {"tier": 1, "difficulty": 4.6, "country": "Italy",        "short": "Serie A",          "single_year": False},
    "FRA-Ligue 1":          {"tier": 1, "difficulty": 4.3, "country": "France",       "short": "Ligue 1",          "single_year": False},
    "INT-Champions League": {"tier": 1, "difficulty": 5.5, "country": "Europe",       "short": "UCL",              "single_year": False},
    "NED-Eredivisie":       {"tier": 2, "difficulty": 3.5, "country": "Netherlands",  "short": "Eredivisie",       "single_year": False},
    "POR-Primeira Liga":    {"tier": 2, "difficulty": 3.4, "country": "Portugal",     "short": "Primeira Liga",    "single_year": False},
    "BEL-First Division A": {"tier": 2, "difficulty": 3.2, "country": "Belgium",      "short": "First Division A", "single_year": False},
    "TUR-Super Lig":        {"tier": 2, "difficulty": 3.1, "country": "Turkey",       "short": "Super Lig",        "single_year": False},
    "MEX-Liga MX":          {"tier": 2, "difficulty": 2.9, "country": "Mexico",       "short": "Liga MX",          "single_year": True},
    "USA-MLS":              {"tier": 2, "difficulty": 2.8, "country": "USA",          "short": "MLS",              "single_year": True},
    "SAU-Pro League":       {"tier": 2, "difficulty": 2.7, "country": "Saudi Arabia", "short": "Saudi Pro League", "single_year": False},
}

if LEAGUE_SELECTOR is not None:
    matched = {k: v for k, v in LEAGUES.items()
               if v["short"].lower() == LEAGUE_SELECTOR.strip().lower()}
    if not matched:
        valid = [v["short"] for v in LEAGUES.values()]
        print(f"✗ Unknown LEAGUE_SELECTOR '{LEAGUE_SELECTOR}'. Valid options: {valid}")
        sys.exit(1)
    ACTIVE_LEAGUES = matched
    print(f" Single-league mode: {LEAGUE_SELECTOR}")
else:
    ACTIVE_LEAGUES = LEAGUES
    print(f" All-leagues mode: {len(ACTIVE_LEAGUES)} leagues")

FBREF_STAT_TYPES = ["standard", "shooting", "keeper", "playing_time", "misc"]

if LEAGUE_SELECTOR is not None:
    _safe_name = LEAGUE_SELECTOR.strip().lower().replace(" ", "_")
    PARTIAL_FILE = f"player_stats_{_safe_name}.partial.csv"
else:
    PARTIAL_FILE = "player_stats_all_leagues.partial.csv"

DELAY_MIN                = 90
DELAY_MAX                = 150
COOLDOWN_BETWEEN_SEASONS = 180
COOLDOWN_BETWEEN_LEAGUES = 300
MAX_RETRIES              = 3
RETRY_BACKOFF            = 120


def random_delay(min_s=DELAY_MIN, max_s=DELAY_MAX, reason="Next request"):
    t = int(random.uniform(min_s, max_s))
    print(f"    ⏳ {reason} — waiting {t}s...", flush=True)
    for remaining in range(t, 0, -15):
        if remaining < t:
            print(f"       ...{remaining}s remaining", flush=True)
        time.sleep(min(15, remaining))


def fetch_fbref_stat(league, season, stat_type):
    for attempt in range(1, MAX_RETRIES + 1):
        fbref = None
        try:
            fbref = sd.FBref(leagues=league, seasons=season, no_store=False)
            df = fbref.read_player_season_stats(stat_type=stat_type)
            return df
        except ValueError as e:
            print(f"\n Skipping {stat_type} ({league} {season}) — parse error: {e}")
            return None
        except Exception as e:
            wait = RETRY_BACKOFF * attempt
            print(f"\n Attempt {attempt}/{MAX_RETRIES} failed for "
                  f"{stat_type} ({league} {season}): {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                random_delay(wait, wait + 30, reason=f"Retry {attempt+1} cooldown")
        finally:
            try:
                if fbref is not None and hasattr(fbref, "_driver") and fbref._driver:
                    fbref._driver.quit()
            except Exception:
                pass
    print(f"    ✗ Gave up on {stat_type} for {league} {season}.")
    return None


def standardise_columns(df):
    if df.index.name and str(df.index.name).lower().strip() in ("player", "name"):
        df = df.reset_index()
    elif isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(c).lower().strip() for c in col
                     if str(c).strip() not in ("", "nan")]
            new_cols.append("_".join(parts) if parts else "unnamed")
        df.columns = new_cols
    else:
        df.columns = [str(c).lower().strip() for c in df.columns]

    if "player" not in df.columns:
        for candidate in ("player", "name", "player_player", "unnamed:_0"):
            if candidate in df.columns:
                df.rename(columns={candidate: "player"}, inplace=True)
                break

    return df


def get_player_key(df):
    cols = df.columns.tolist()
    name_col = next((c for c in cols if c == "player"), None) or \
               next((c for c in cols if "player" in c and "squad" not in c
                     and "nation" not in c), None) or \
               next((c for c in cols if c == "name"), None)
    team_col = next((c for c in cols if c in ("squad", "team")), None) or \
               next((c for c in cols if "squad" in c or "team" in c), None)
    season_col = next((c for c in cols if "season" in c), None)

    keys = []
    for col in [name_col, team_col, season_col]:
        if col:
            keys.append(df[col].astype(str).str.lower().str.strip())
    if keys:
        return keys[0].str.cat(keys[1:], sep="||")
    return pd.Series(range(len(df)), index=df.index).astype(str)


all_frames = []

for league_idx, (league, meta) in enumerate(ACTIVE_LEAGUES.items()):
    print(f"\n{'='*60}")
    print(f"League {league_idx+1}/{len(ACTIVE_LEAGUES)}: {league}")
    print(f"{'='*60}")

    if league_idx > 0:
        random_delay(COOLDOWN_BETWEEN_LEAGUES, COOLDOWN_BETWEEN_LEAGUES + 60,
                     reason=f"League cooldown before {league}")

    league_seasons = SINGLE_YEAR_SEASONS if meta.get("single_year") else SEASONS

    for season_idx, season in enumerate(league_seasons):
        print(f"\n  Season: {season}")
        season_frames = {}

        if season_idx > 0:
            random_delay(COOLDOWN_BETWEEN_SEASONS, COOLDOWN_BETWEEN_SEASONS + 60,
                         reason=f"Season cooldown before {season}")

        for stat_type in FBREF_STAT_TYPES:
            print(f"Fetching: {stat_type}...", end=" ", flush=True)
            df = fetch_fbref_stat(league, season, stat_type)

            if df is not None:
                df = standardise_columns(df)
                if "player" not in df.columns:
                    print(f"\n WARNING: no 'player' column in {stat_type}. "
                          f"Columns: {df.columns.tolist()[:10]}")
                df["_key"] = get_player_key(df)
                season_frames[stat_type] = df
                print("✓")
            else:
                print("✗ (skipped)")

            if stat_type != FBREF_STAT_TYPES[-1]:
                next_type = FBREF_STAT_TYPES[FBREF_STAT_TYPES.index(stat_type) + 1]
                random_delay(reason=f"Before fetching '{next_type}'")

        if "standard" not in season_frames:
            print(f"  ✗ No standard stats for {league} {season}, skipping.")
            continue

        base = season_frames["standard"].copy()

        for stat_type, df in season_frames.items():
            if stat_type == "standard":
                continue
            all_cols = [c for c in df.columns if c != "_key"]
            if not all_cols:
                continue
            sub = df[["_key"] + all_cols].copy()
            sub.rename(columns={c: f"{c}__{stat_type}" for c in all_cols}, inplace=True)
            base = base.merge(sub, on="_key", how="left",
                              suffixes=("", f"_dup_{stat_type}"))

        base["league"]                 = league
        base["country"]                = meta["country"]
        base["league_tier"]            = meta["tier"]
        base["club_league_difficulty"] = meta["difficulty"]
        base["season"]                 = season

        all_frames.append(base)
        print(f"  ✓ Merged {len(base)} player-rows for {league} {season}")

        if all_frames:
            partial = pd.concat(all_frames, ignore_index=True, sort=False)
            partial.to_csv(PARTIAL_FILE, index=False)
            print(f"Saved to {PARTIAL_FILE} ({len(partial):,} rows so far)")

print(f"\n Concluded. Final file: {PARTIAL_FILE}")