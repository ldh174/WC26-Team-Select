import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "."
OUT_CSV  = "master.csv"
MIN_90S  = 1.0

REDUNDANT_PREFIXES = [
    "league__", "season__", "team__", "player__",
    "nation__", "pos__",    "age__",  "born__",
]
REDUNDANT_EXACT = [
    "90s__shooting", "standard_gls__shooting", "standard_pk__shooting",
    "standard_pkatt__shooting", "playing time_mp__keeper",
    "playing time_starts__keeper", "playing time_min__keeper",
    "playing time_90s__keeper", "playing time_mp__playing_time",
    "playing time_90s__playing_time", "performance_crdy__misc",
    "performance_crdr__misc", "90s__misc", "starts_starts__playing_time",
    "playing time_min__playing_time",
]
EMPTY_COLS = ["pk_won", "pk_conceded"]
SEASON_MAP = {"2024": "2324", "2025": "2425", "2026": "2526"}

RENAME = {
    "_key":                             "player_key",
    "playing time_mp":                  "mp",
    "playing time_starts":              "starts",
    "playing time_min":                 "min",
    "playing time_90s":                 "90s",
    "performance_gls":                  "gls",
    "performance_ast":                  "ast",
    "performance_g+a":                  "g_a",
    "performance_g-pk":                 "gls_npk",
    "performance_pk":                   "pk",
    "performance_pkatt":                "pkatt",
    "performance_crdy":                 "yellow",
    "performance_crdr":                 "red",
    "per 90 minutes_gls":               "gls_p90",
    "per 90 minutes_ast":               "ast_p90",
    "per 90 minutes_g+a":               "g_a_p90",
    "per 90 minutes_g-pk":              "gls_npk_p90",
    "per 90 minutes_g+a-pk":            "g_a_npk_p90",
    "standard_sh__shooting":            "sh",
    "standard_sot__shooting":           "sot",
    "standard_sot%__shooting":          "sot_pct",
    "standard_sh/90__shooting":         "sh_p90",
    "standard_sot/90__shooting":        "sot_p90",
    "standard_g/sh__shooting":          "g_per_sh",
    "standard_g/sot__shooting":         "g_per_sot",
    "performance_ga__keeper":           "gk_ga",
    "performance_ga90__keeper":         "gk_ga90",
    "performance_sota__keeper":         "gk_sota",
    "performance_saves__keeper":        "gk_saves",
    "performance_save%__keeper":        "gk_save_pct",
    "performance_w__keeper":            "gk_w",
    "performance_d__keeper":            "gk_d",
    "performance_l__keeper":            "gk_l",
    "performance_cs__keeper":           "gk_cs",
    "performance_cs%__keeper":          "gk_cs_pct",
    "penalty kicks_pkatt__keeper":      "gk_pk_faced",
    "penalty kicks_pka__keeper":        "gk_pk_allowed",
    "penalty kicks_pksv__keeper":       "gk_pk_saved",
    "penalty kicks_pkm__keeper":        "gk_pk_missed",
    "penalty kicks_save%__keeper":      "gk_pk_save_pct",
    "playing time_mn/mp__playing_time": "mn_per_mp",
    "playing time_min%__playing_time":  "min_pct",
    "starts_mn/start__playing_time":    "mn_per_start",
    "starts_compl__playing_time":       "starts_completed",
    "subs_subs__playing_time":          "subs",
    "subs_mn/sub__playing_time":        "mn_per_sub",
    "subs_unsub__playing_time":         "unsub",
    "team success_ppm__playing_time":   "ppm",
    "team success_ong__playing_time":   "ong",
    "team success_onga__playing_time":  "onga",
    "team success_+/-__playing_time":   "plus_minus",
    "team success_+/-90__playing_time": "plus_minus_p90",
    "team success_on-off__playing_time":"on_off",
    "performance_2crdy__misc":          "second_yellow",
    "performance_fls__misc":            "fouls",
    "performance_fld__misc":            "fouled",
    "performance_off__misc":            "offsides",
    "performance_crs__misc":            "crosses",
    "performance_int__misc":            "interceptions",
    "performance_tklw__misc":           "tkl_won",
    "performance_pkwon__misc":          "pk_won_misc",
    "performance_pkcon__misc":          "pk_conceded_misc",
    "performance_og__misc":             "own_goals",
}

NUMERIC_COLS = [
    "age", "born", "mp", "starts", "min", "90s",
    "gls", "ast", "g_a", "gls_npk", "pk", "pkatt", "yellow", "red",
    "gls_p90", "ast_p90", "g_a_p90", "gls_npk_p90", "g_a_npk_p90",
    "sh", "sot", "sot_pct", "sh_p90", "sot_p90", "g_per_sh", "g_per_sot",
    "gk_ga", "gk_ga90", "gk_sota", "gk_saves", "gk_save_pct",
    "gk_w", "gk_d", "gk_l", "gk_cs", "gk_cs_pct",
    "gk_pk_faced", "gk_pk_allowed", "gk_pk_saved", "gk_pk_missed", "gk_pk_save_pct",
    "mn_per_mp", "min_pct", "mn_per_start", "starts_completed",
    "subs", "mn_per_sub", "unsub", "ppm", "ong", "onga",
    "plus_minus", "plus_minus_p90", "on_off",
    "second_yellow", "fouls", "fouled", "offsides", "crosses",
    "interceptions", "tkl_won", "pk_won_misc", "pk_conceded_misc", "own_goals",
    "league_tier", "club_league_difficulty",
]


def load_all_csvs():
    SKIP = {
        "master.csv", "features.csv",
        "player_national_performances.csv", "player_profiles.csv",
    }
    paths = [p for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
             if os.path.basename(p) not in SKIP]
    if not paths:
        raise FileNotFoundError(f"No CSV files found in '{DATA_DIR}'")
    frames = []
    for path in sorted(paths):
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)
        print(f"  {os.path.basename(path):30s}  {df.shape[0]:>5} rows")
    raw = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {raw.shape[0]} rows x {raw.shape[1]} cols")
    return raw


def drop_redundant_cols(df):
    drops = list(set(
        [c for c in df.columns if any(c.startswith(p) for p in REDUNDANT_PREFIXES)] +
        [c for c in REDUNDANT_EXACT if c in df.columns] +
        [c for c in EMPTY_COLS if c in df.columns]
    ))
    return df.drop(columns=drops)


def dedup_keys(df):
    def priority(league):
        if pd.isna(league):            return 3
        if league.startswith("INT-"):  return 2
        if league.startswith("USA-"):  return 1
        return 0
    df["_p"] = df["league"].apply(priority)
    df = df.sort_values("_p").drop_duplicates(subset=["player_key"], keep="first")
    return df.drop(columns=["_p"])


def main():
    print("WC26 | Data Cleaning")
    print("  Loading CSVs...", flush=True)
    df = load_all_csvs()

    drops = list(set(
        [c for c in df.columns if any(c.startswith(p) for p in REDUNDANT_PREFIXES)] +
        [c for c in REDUNDANT_EXACT if c in df.columns] +
        [c for c in EMPTY_COLS if c in df.columns]
    ))
    df = df.drop(columns=drops)

    rename_map = {k: v for k, v in RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["player", "team", "league", "nation", "country", "pos"]:
        if col in df.columns:
            df[col] = df[col].str.strip()
    df["nation"]  = df["nation"].str.upper()
    df["season"]  = df["season"].astype(str).str.strip().replace(SEASON_MAP)

    before = len(df)
    df = df[df["90s"] >= MIN_90S].copy()
    print(f"  Filtered to {len(df)} rows (removed {before - len(df)} with < {MIN_90S} 90s)")

    before = len(df)
    df = dedup_keys(df)
    print(f"  Deduplicated to {len(df)} rows (removed {before - len(df)} UCL/MLS overlaps)")

    filled = df["age"].isnull().sum()
    df["age"] = df["age"].fillna(2026 - df["born"])
    print(f"  Filled {filled} age nulls from born year")

    df = df.drop(columns=["pk_won_misc", "pk_conceded_misc"], errors="ignore")

    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved {OUT_CSV}  ({len(df)} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    main()