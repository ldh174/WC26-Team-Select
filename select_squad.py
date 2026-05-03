import argparse
import os
import pandas as pd
import numpy as np

IN_CSV  = "features.csv"
OUT_DIR = "squads"

SQUAD_SLOTS = {"GK": 3, "DF": 8, "MF": 8, "FW": 7}
SCORE_COL   = {"GK": "gk_score", "DF": "def_score", "MF": "mid_score", "FW": "fwd_score"}

NATION_MAP = {
    "ENG": "England",       "GER": "Germany",       "FRA": "France",
    "ESP": "Spain",         "ITA": "Italy",         "POR": "Portugal",
    "NED": "Netherlands",   "BEL": "Belgium",       "SUI": "Switzerland",
    "AUT": "Austria",       "SWE": "Sweden",        "NOR": "Norway",
    "DEN": "Denmark",       "POL": "Poland",        "CZE": "Czech Republic",
    "CRO": "Croatia",       "SRB": "Serbia",        "SVN": "Slovenia",
    "SVK": "Slovakia",      "HUN": "Hungary",       "ROU": "Romania",
    "UKR": "Ukraine",       "RUS": "Russia",        "TUR": "Turkey",
    "GRE": "Greece",        "SCO": "Scotland",      "WAL": "Wales",
    "NIR": "Northern Ireland", "IRL": "Ireland",    "ISL": "Iceland",
    "FIN": "Finland",       "ALB": "Albania",       "MNE": "Montenegro",
    "BIH": "Bosnia and Herzegovina", "MKD": "North Macedonia",
    "GEO": "Georgia",       "KVX": "Kosovo",        "BUL": "Bulgaria",
    "ISR": "Israel",        "MDA": "Moldova",       "KAZ": "Kazakhstan",
    "BRA": "Brazil",        "ARG": "Argentina",     "COL": "Colombia",
    "URU": "Uruguay",       "CHI": "Chile",         "ECU": "Ecuador",
    "PER": "Peru",          "PAR": "Paraguay",      "MEX": "Mexico",
    "USA": "United States", "CAN": "Canada",        "CRC": "Costa Rica",
    "SEN": "Senegal",       "MAR": "Morocco",       "EGY": "Egypt",
    "NGA": "Nigeria",       "GHA": "Ghana",         "CMR": "Cameroon",
    "CIV": "Ivory Coast",   "MLI": "Mali",          "TUN": "Tunisia",
    "ALG": "Algeria",       "RSA": "South Africa",  "GAB": "Gabon",
    "CPV": "Cape Verde",    "GAM": "Gambia",        "GMB": "Gambia",
    "JPN": "Japan",         "KOR": "South Korea",   "IRN": "Iran",
    "SAU": "Saudi Arabia",  "KSA": "Saudi Arabia",  "AUS": "Australia",
}

COUNTRY_TO_CODES = {}
for code, name in NATION_MAP.items():
    COUNTRY_TO_CODES.setdefault(name.lower(), []).append(code)

# Runtime position corrections (FBref label != national team role)
POSITION_OVERRIDES = {
    "nathaniel clyne": "DF", "marcus rashford": "FW",
    "oliver scarles":  "DF", "ryan sessegnon":  "DF",
    "kai havertz":     "FW", "joshua kimmich":  "DF",
    "marcelo herrera": "DF", "sean zawadzki":   "DF",
    "claudio falcão":  "DF", "yan couto":       "DF",
    "vitor costa":     "DF",
}

# Score multipliers for players whose stats are suppressed by injury/move
REPUTATION_BOOSTS = {
    "marc-andré ter stegen": 1.25,  # injury-reduced seasons
    "florian wirtz":         1.25,  # Liverpool move hurt 2526 score
    "kai havertz":           1.20,  # 3.5 90s in 2526
    "phil foden":            1.15,  # injury-reduced at Man City
    "declan rice":           1.15,  # rotation reduced 90s
    "luka modrić":           1.20,  # limited 90s in Serie A data
}


def load_features():
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  Loaded {IN_CSV}  ({len(df)} rows x {len(df.columns)} cols)")
    return df


def apply_overrides(df):
    n_pos, n_boost = 0, 0
    for name, pos in POSITION_OVERRIDES.items():
        mask = df["player"].str.lower() == name
        if mask.any():
            df.loc[mask, "primary_pos"] = pos
            n_pos += mask.sum()
    for name, mult in REPUTATION_BOOSTS.items():
        mask = df["player"].str.lower() == name
        if mask.any():
            df.loc[mask, "weighted_score"] = (df.loc[mask, "weighted_score"] * mult).round(2)
            n_boost += mask.sum()
    print(f"  {len(POSITION_OVERRIDES)} position overrides ({n_pos} rows)  |  "
          f"{len(REPUTATION_BOOSTS)} reputation boosts ({n_boost} rows)")
    return df


def get_candidate_pool(df, country, season=None):
    codes = COUNTRY_TO_CODES.get(country.lower(), [])
    if not codes:
        raise ValueError(f"Unknown country: {country}")

    pool = df[df["nation"].isin(codes)].copy()
    if season:
        pool = pool[pool["season"] == str(season)]

    pool = pool.sort_values("weighted_score", ascending=False, na_position="last")
    pool = pool.drop_duplicates(subset=["player"], keep="first")
    print(f"  {country}: {len(pool)} eligible players")
    return pool


def select_position_group(pool, position, n_slots, score_col):
    group = pool[(pool["primary_pos"] == position) & pool[score_col].notna()].copy()
    if position == "FW":
        before = len(group)
        group  = group[(group["gls_adj"].fillna(0) > 0) | (group["ast_adj"].fillna(0) > 0)]
        if len(group) < before:
            print(f"    Removed {before - len(group)} zero-contribution FW candidates")
    group    = group.sort_values("weighted_score", ascending=False)
    selected = group.head(n_slots).copy()
    selected["pick"] = range(1, len(selected) + 1)
    selected["position_group"] = position
    return selected, group


def save_squad(squad_parts, country):
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for position, (selected, full_pool) in squad_parts.items():
        sel = selected.copy()
        sel["role"] = sel["pick"].apply(lambda p: "starter" if p == 1 else "backup" if p == 2 else "sub")
        selected_names = set(sel["player"].str.lower())
        alternates = full_pool[~full_pool["player"].str.lower().isin(selected_names)].head(3).copy()
        if len(alternates):
            alternates["pick"] = range(len(sel) + 1, len(sel) + len(alternates) + 1)
            alternates["position_group"] = position
            alternates["role"] = "alternate"
        frames.append(sel)
        if len(alternates):
            frames.append(alternates)
    squad_df = pd.concat(frames, ignore_index=True)
    fpath    = os.path.join(OUT_DIR, f"{country.lower().replace(' ', '_')}_squad.csv")
    squad_df.to_csv(fpath, index=False)
    print(f"  Saved: {fpath}")
    return squad_df


def run_selection(country, season=None, top_n=5):
    print(f"\nWC26 | {country}")
    df = load_features()
    df = apply_overrides(df)
    pool = get_candidate_pool(df, country, season)
    squad_parts = {}
    for position, n_slots in SQUAD_SLOTS.items():
        selected, full_pool = select_position_group(pool, position, n_slots, SCORE_COL[position])
        squad_parts[position] = (selected, full_pool)
        print(f"  {position}: {len(selected)}/{n_slots} filled")

    print()
    for position, (selected, full_pool) in squad_parts.items():
        print(f"  {position}:")
        for _, row in selected.iterrows():
            role  = "S" if row["pick"] == 1 else "B" if row["pick"] == 2 else " "
            score = f"{row['weighted_score']:.1f}" if pd.notna(row.get("weighted_score")) else "N/A"
            print(f"    {role} {row['pick']:>2}. {row['player']:<25} {row.get('team',''):<22} {score}")

        if top_n:
            print(f"\n    --- full {position} pool (top {top_n}) ---")
            for _, row in full_pool.head(top_n).iterrows():
                score = f"{row['weighted_score']:.1f}" if pd.notna(row.get("weighted_score")) else "N/A"
                print(f"         {row['player']:<25} {row.get('team',''):<22} {score}")
        print()

    save_squad(squad_parts, country)


def run_all():
    TOP_15 = [
        "Argentina", "France",        "England",       "Spain",    "Brazil",
        "Portugal",  "Belgium",       "Netherlands",   "Germany",  "Colombia",
        "Morocco",   "United States", "Japan",         "Croatia",  "Senegal",
    ]

    print("WC26 | All Nations")
    df = load_features()
    df = apply_overrides(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    results, failed = {}, []

    for country in TOP_15:
        print(f"\n--- {country.upper()} ---")
        try:
            pool = get_candidate_pool(df, country)
            squad_parts = {}
            for position, n_slots in SQUAD_SLOTS.items():
                selected, full_pool = select_position_group(pool, position, n_slots, SCORE_COL[position])
                squad_parts[position] = (selected, full_pool)

            for position, (selected, _) in squad_parts.items():
                print(f"  {position}:")
                for _, row in selected.iterrows():
                    role  = "S" if row["pick"] == 1 else "B" if row["pick"] == 2 else " "
                    score = f"{row['weighted_score']:.1f}" if pd.notna(row.get("weighted_score")) else "N/A"
                    print(f"    {role} {row['pick']:>2}. {row['player']:<25} {row.get('team',''):<22} {score}")

            squad_df = save_squad(squad_parts, country)
            results[country] = squad_df

        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(country)

    print(f"\nDONE — {len(results)}/{len(TOP_15)} nations saved to {OUT_DIR}/")
    if failed:
        print(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="WC26 squad selector")
    parser.add_argument("--country", "-c", type=str, default=None)
    parser.add_argument("--all",     "-a", action="store_true")
    parser.add_argument("--season",  "-s", type=str, default=None)
    parser.add_argument("--top",     "-t", type=int, default=5)
    args = parser.parse_args()

    if args.all:
        run_all()
    elif args.country:
        run_selection(args.country, args.season, args.top)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()