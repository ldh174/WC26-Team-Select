import pandas as pd
import numpy as np

IN_CSV  = "master.csv"
OUT_CSV = "features.csv"

SEASON_WEIGHTS = {"2526": 0.50, "2425": 0.30, "2324": 0.20}
MIN_90S_FW = MIN_90S_MF = MIN_90S_DF = MIN_90S_GK = 5.0
MIN_SOTA_GK = 20
SEASONS_MULTIPLIER = {1: 0.80, 2: 0.92, 3: 1.00}

FWD_WEIGHTS = {
    "gls_adj": 0.30, "ast_adj": 0.15, "shot_quality": 0.15,
    "involvement": 0.15, "sh_adj": 0.10, "on_off": 0.10, "discipline": 0.05,
}
MID_WEIGHTS = {
    "g_a_adj": 0.35, "involvement": 0.20, "interceptions": 0.10,
    "tkl_won": 0.10, "on_off": 0.20, "discipline": 0.05,
}
DEF_WEIGHTS = {
    "tkl_won": 0.25, "interceptions": 0.25, "plus_minus_p90": 0.20,
    "fouls": 0.10, "on_off": 0.15, "discipline": 0.05,
}
GK_WEIGHTS = {
    "gk_save_pct": 0.30, "gk_ga90": 0.25, "gk_cs_pct": 0.20,
    "gk_pk_save_pct": 0.10, "on_off": 0.15,
}

# Players FBref misclassifies
OVERRIDES = {
    # england
    "nathaniel clyne": "DF", "marcus rashford": "FW",
    "oliver scarles":  "DF", "ryan sessegnon":  "DF",
    # germany
    "kai havertz": "FW",
    # argentina
    "marcelo herrera": "DF", "nahuel molina": "DF",
    # brazil
    "claudio falcão": "DF", "yan couto": "DF",
    "vitor costa": "DF", "vanderson": "DF",
    "caio henrique oliveira silva": "DF",
    # usa
    "sean zawadzki": "DF", "timothy weah": "FW",
    # belgium
    "louis patris": "DF",
    # croatia
    "josip juranović": "DF",
    # colombia
    "daniel muñoz": "DF",
    # senegal
    "ismaila sarr": "FW", "el hadji malick diouf": "DF",
    # morocco
    "brahim díaz": "FW",
    # portugal
    "bernardo silva": "FW", "nuno mendes": "DF", "flávio nazinho": "DF",
    "raphaël guerreiro": "DF", "diogo dalot": "DF",
    # france
    "désiré doué": "FW", "rayan cherki": "MF",
    # spain
    "lamine yamal": "FW", "nico williams": "FW", 
    "marcos llorente": "DF", "sergio gómez": "DF",
    "pedri": "MF", "gavi": "MF",
    # usa
    "joe scally": "DF",
    # netherlands
    "denzel dumfries": "DF",
    # croatia
    "borna sosa": "DF", "duje ćaleta-car": "DF",
    # misc
    "luca langoni": "DF",
}


def add_eligibility_flags(df):
    df["primary_pos"] = df["pos"].str.split(",").str[0].str.strip()
    for name, pos in OVERRIDES.items():
        mask = df["player"].str.strip().str.lower() == name
        if mask.any():
            df.loc[mask, "primary_pos"] = pos
    df["is_gk"] = df["primary_pos"] == "GK"
    df["is_df"] = df["primary_pos"] == "DF"
    df["is_mf"] = df["primary_pos"] == "MF"
    df["is_fw"] = df["primary_pos"] == "FW"
    df.loc[df["is_fw"] & (df["90s"] < MIN_90S_FW), "is_fw"] = False
    df.loc[df["is_mf"] & (df["90s"] < MIN_90S_MF), "is_mf"] = False
    df.loc[df["is_df"] & (df["90s"] < MIN_90S_DF), "is_df"] = False
    df.loc[df["is_gk"] & (df["90s"] < MIN_90S_GK), "is_gk"] = False
    df["gk_save_pct_adj"] = df["gk_save_pct"]
    low_sota = df["is_gk"] & (df["gk_sota"].fillna(0) < MIN_SOTA_GK)
    df.loc[low_sota, "gk_save_pct_adj"] = np.nan
    return df


def add_league_adjusted(df):
    diff     = df["club_league_difficulty"].fillna(1.0)
    safe_90s = df["90s"].replace(0, np.nan)
    df["gls_adj"] = df["gls_p90"]     * diff
    df["ast_adj"] = df["ast_p90"]     * diff
    df["g_a_adj"] = df["g_a_p90"]     * diff
    df["sh_adj"]  = df["sh_p90"]      * diff
    df["sot_adj"] = df["sot_p90"]     * diff
    df["npg_adj"] = df["gls_npk_p90"] * diff
    df["tkl_won_adj"]       = (df["tkl_won"].fillna(0)       / safe_90s) * diff
    df["interceptions_adj"] = (df["interceptions"].fillna(0) / safe_90s) * diff
    df["crosses_adj"]       = (df["crosses"].fillna(0)       / safe_90s) * diff
    df["fouls_adj"]         = (df["fouls"].fillna(0)         / safe_90s) * diff
    return df


def add_derived_ratios(df):
    df["involvement"]  = ((df["gls"] + df["ast"]) / df["ong"].replace(0, np.nan)).fillna(0).clip(0, 1)
    df["shot_quality"] = df["g_per_sot"].fillna(0)
    df["discipline"]   = (
        df["yellow"].fillna(0) * 1 +
        df["second_yellow"].fillna(0) * 2 +
        df["red"].fillna(0) * 3
    ) / df["90s"].replace(0, np.nan)
    df["discipline"] = df["discipline"].fillna(0)
    return df


def add_position_composites(df):
    def safe(col):
        return df[col].fillna(0) if col in df.columns else pd.Series(0, index=df.index)

    df["fwd_raw"] = (
        safe("gls_adj")      * FWD_WEIGHTS["gls_adj"] +
        safe("ast_adj")      * FWD_WEIGHTS["ast_adj"] +
        safe("shot_quality") * FWD_WEIGHTS["shot_quality"] +
        safe("involvement")  * FWD_WEIGHTS["involvement"] +
        safe("sh_adj")       * FWD_WEIGHTS["sh_adj"] +
        safe("on_off")       * FWD_WEIGHTS["on_off"] -
        safe("discipline")   * FWD_WEIGHTS["discipline"]
    )
    df["mid_raw"] = (
        safe("g_a_adj")           * MID_WEIGHTS["g_a_adj"] +
        safe("involvement")       * MID_WEIGHTS["involvement"] +
        safe("interceptions_adj") * MID_WEIGHTS["interceptions"] +
        safe("tkl_won_adj")       * MID_WEIGHTS["tkl_won"] +
        safe("on_off")            * MID_WEIGHTS["on_off"] -
        safe("discipline")        * MID_WEIGHTS["discipline"]
    )
    df["def_raw"] = (
        safe("tkl_won_adj")       * DEF_WEIGHTS["tkl_won"] +
        safe("interceptions_adj") * DEF_WEIGHTS["interceptions"] +
        safe("plus_minus_p90")    * DEF_WEIGHTS["plus_minus_p90"] +
        safe("on_off")            * DEF_WEIGHTS["on_off"] -
        safe("fouls_adj")         * DEF_WEIGHTS["fouls"] -
        safe("discipline")        * DEF_WEIGHTS["discipline"]
    )
    max_ga90 = df.loc[df["is_gk"], "gk_ga90"].max() if df["is_gk"].any() else 3.0
    df["gk_ga90_inv"] = (max_ga90 - df["gk_ga90"].fillna(max_ga90)).clip(lower=0)
    df["gk_raw"] = (
        safe("gk_save_pct_adj") * GK_WEIGHTS["gk_save_pct"] +
        safe("gk_ga90_inv")     * GK_WEIGHTS["gk_ga90"] +
        safe("gk_cs_pct")       * GK_WEIGHTS["gk_cs_pct"] +
        safe("gk_pk_save_pct")  * GK_WEIGHTS["gk_pk_save_pct"] +
        safe("on_off")          * GK_WEIGHTS["on_off"]
    )
    df.loc[df["primary_pos"] != "FW", "fwd_raw"] = np.nan
    df.loc[df["primary_pos"] != "MF", "mid_raw"] = np.nan
    df.loc[df["primary_pos"] != "DF", "def_raw"] = np.nan
    df.loc[df["primary_pos"] != "GK", "gk_raw"]  = np.nan
    return df


def minmax_normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series.fillna(0) * 0
    return (series - mn) / (mx - mn) * 100


def add_normalized_scores(df):
    for raw_col, score_col, flag in [
        ("fwd_raw", "fwd_score", "is_fw"),
        ("mid_raw", "mid_score", "is_mf"),
        ("def_raw", "def_score", "is_df"),
        ("gk_raw",  "gk_score",  "is_gk"),
    ]:
        df[score_col] = np.nan
        mask = df[flag] & df[raw_col].notna()
        df.loc[mask, score_col] = minmax_normalize(df.loc[mask, raw_col])
    return df


def add_form_trend(df):
    df = df.sort_values(["player", "team", "season"])
    for col, new_col in [
        ("gls_p90", "trend_gls_p90"), ("ast_p90", "trend_ast_p90"),
        ("g_a_p90", "trend_g_a_p90"), ("gls_adj", "trend_gls_adj"),
        ("ast_adj", "trend_ast_adj"),
    ]:
        if col in df.columns:
            df[new_col] = df.groupby(["player", "team"])[col].diff()
    return df


def add_temporal_weighted_score(df):
    df["_pt"] = df["player"] + "||" + df["team"]
    df["_ps"] = np.nan
    df.loc[df["primary_pos"] == "FW", "_ps"] = df.loc[df["primary_pos"] == "FW", "fwd_score"]
    df.loc[df["primary_pos"] == "MF", "_ps"] = df.loc[df["primary_pos"] == "MF", "mid_score"]
    df.loc[df["primary_pos"] == "DF", "_ps"] = df.loc[df["primary_pos"] == "DF", "def_score"]
    df.loc[df["primary_pos"] == "GK", "_ps"] = df.loc[df["primary_pos"] == "GK", "gk_score"]

    weighted_scores, seasons_played = {}, {}
    for key, grp in df.groupby("_pt"):
        grp = grp[grp["_ps"].notna()]
        if grp.empty:
            continue
        total_w, total_s, n = 0.0, 0.0, 0
        for _, row in grp.iterrows():
            w = SEASON_WEIGHTS.get(str(row["season"]), 0.0)
            if w > 0:
                total_s += row["_ps"] * w
                total_w += w
                n += 1
        if total_w == 0:
            continue
        weighted_scores[key] = round((total_s / total_w) * SEASONS_MULTIPLIER.get(min(n, 3), 1.0), 2)
        seasons_played[key]  = float(n)

    df["weighted_score"]  = df["_pt"].map(weighted_scores)
    df["seasons_in_data"] = df["_pt"].map(seasons_played)
    df = df.drop(columns=["_pt", "_ps"])
    return df


def add_form_trend_bonus(df):
    def bonus(val):
        return float(np.clip(val * 6, -5, 5)) if not pd.isna(val) else 0.0

    df["_tb"] = 0.0
    df.loc[df["primary_pos"] == "FW", "_tb"] = df.loc[df["primary_pos"] == "FW", "trend_gls_adj"].apply(bonus)
    df.loc[df["primary_pos"] == "MF", "_tb"] = df.loc[df["primary_pos"] == "MF", "trend_g_a_p90"].apply(bonus)
    df.loc[df["primary_pos"] == "DF", "_tb"] = df.loc[df["primary_pos"] == "DF", "trend_gls_adj"].apply(lambda x: bonus(x) * 0.5)
    df["form_trend_bonus"] = df["_tb"].round(2)
    df["weighted_score"]   = (df["weighted_score"] + df["form_trend_bonus"]).clip(0, 100).round(2)
    return df.drop(columns=["_tb"])


def add_caps_multiplier(df):
    if "nt_caps" not in df.columns:
        return df

    def caps_mult(caps):
        if pd.isna(caps):
            return 1.0
        caps = max(0, caps)
        if caps == 0:
            return 0.92
        return min(1.12, 0.92 + (caps / 100) * 0.20)

    df["caps_multiplier"] = df["nt_caps"].apply(caps_mult).round(3)
    df["weighted_score"]  = (df["weighted_score"] * df["caps_multiplier"]).clip(0, 100).round(2)
    return df


def main():
    print("WC26 | Feature Engineering")
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  Loaded {IN_CSV}  ({len(df)} rows x {len(df.columns)} cols)")
    print("  Processing...", flush=True)
    df = add_eligibility_flags(df)
    df = add_league_adjusted(df)
    df = add_derived_ratios(df)
    df = add_position_composites(df)
    df = add_normalized_scores(df)
    df = add_form_trend(df)
    df = add_temporal_weighted_score(df)
    df = add_form_trend_bonus(df)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved {OUT_CSV}  ({len(df)} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    main()