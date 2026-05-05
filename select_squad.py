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

EXCLUSIONS = {
    # retired
    "manuel neuer", "thomas müller", "james milner", "antoine griezmann",
    "ángel di maría", "daley blind", "luuk de jong", "simon mignolet",
    "milan badelj", "iago aspas", "hugo lloris", "karim benzema",
    "dejan lovren", "david ospina",
    # rip
    "diogo jota",
    # not NT regulars
    "noah atubolu", "robert-jan vanwesemael",
    "maximiliano moralez", "mauro icardi",
    "jean butez",
    "jizz hornkamp",
    "takahiro akimoto",
    "m'baye niang",
    "gonçalo paciência",
    "michy batshuayi",
    "marcus coco", "djibril sidibé", "soungoutou magassa", "lilian raolisoa",
    "lewis cook", "james garner",
    "diego rico", "aitor ruibal", "césar azpilicueta",
    "pedro bicalho", "junior messias",
    "joão mendes", "zé carlos", "manuel manu",
    "bamba dieng", "emre can", "thilo kehrer",
    "milan iloski", "jonathan dean", "sebastian berhalter",
    "maya yoshida",
    # injured / unavailable
    "hugo ekitike", "juan foyth", "takumi minamino", "rodrygo",
    "serge gnabry", "xavi simons", "éder militão", "rodri",
    "iglesias", "kike salas", "yarek gasiorowski",
    "kyle smith", "cristian roldan", "max arfsten",
    "lukas kübler", "dominik kohr", "matthias ginter",
    "marc-andré ter stegen",
}

POSITION_OVERRIDES = {
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
    # spain
    "marcos llorente": "DF", "sergio gómez": "DF",
    # usa
    "joe scally": "DF",  
    # croatia
    "borna sosa": "DF", "duje ćaleta-car": "DF",
    # senegal
    "ismaila sarr": "FW", "el hadji malick diouf": "DF",
    # morocco
    "brahim díaz": "FW",
    # portugal 
    "bernardo silva": "FW", "nuno mendes": "DF",
    "flávio nazinho": "DF", "raphaël guerreiro": "DF", "diogo dalot": "DF",
    # france/spain
    "désiré doué": "FW", "lamine yamal": "FW", "nico williams": "FW",
    # misc fbref inconsistencies
    "luca langoni": "DF",
}

REPUTATION_BOOSTS = {
    # argentina 
    "rodrigo de paul":      1.15,
    "nahuel molina":        1.15,
    "lisandro martínez":    1.10,  
    "julián álvarez":       1.15,  
    "alexis mac allister":  1.05,  
    "enzo fernández":       1.05,  
    # france 
    "ibrahima konaté":      1.05,  
    "william saliba":       1.06,  
    "rayan cherki":         1.20,
    "michael olise":        1.15,
    "désiré doué":          1.15,
    "eduardo camavinga":    1.15,
    # england
    "reece james":          1.25,
    "john stones":          1.20,
    "jude bellingham":      1.10,
    "anthony gordon":       1.30,
    "phil foden":           1.15,
    "declan rice":          1.15,
    # spain 
    "lamine yamal":         1.20,
    "nico williams":        1.15,
    "pedri":                1.20,
    "gavi":                 1.25,
    "fermín lópez":         1.15,
    "pau cubarsí":          1.20,
    "aymeric laporte":      1.20,
    "dani olmo":            1.20,  
    "mikel oyarzabal":      1.25,  
    # brazil 
    "alisson":              1.05,  
    "ederson":              1.15,
    "marquinhos":           1.20, 
    "caio henrique oliveira silva": 1.10,  
    "gabriel magalhães":    1.15,
    "lucas paquetá":        1.15,
    "bruno guimarães":      1.05, 
    "vinicius júnior":      1.10,  
    # portugal 
    "vitinha":              1.80,  
    "joão neves":           1.60,  
    "bernardo silva":       1.20,
    "rafael leão":          1.05,
    "gonçalo inácio":       1.10,
    "nuno mendes":          1.15,
    # belgium
    "thibaut courtois":     1.12,  
    "arthur theate":        1.08,
    "amadou onana":         1.25, 
    "romelu lukaku":        1.10,
    "kevin de bruyne":      1.15,
    # netherlands
    "bart verbruggen":      1.20,
    "ryan gravenberch":     1.14,
    "tijjani reijnders":    1.12,
    "denzel dumfries":      1.20,
    "frenkie de jong":      1.15,
    "micky van de ven":     1.15,
    "memphis":              1.05,
    # germany 
    "marc-andré ter stegen": 1.05,  
    "florian wirtz":        1.25,
    "oliver baumann":       1.20,  
    "jonathan tah":         1.20,
    "david raum":           1.15,
    "nico schlotterbeck":   1.15,
    "kai havertz":          1.20,
    "joshua kimmich":       1.16,
    "jamal musiala":        1.05,
    "leroy sané":           1.16,
    # colombia 
    "james rodríguez":      1.30,  
    "johan mojica":         1.12,
    "jhon arias":           1.05,
    # morocco
    "brahim díaz":          2.00,
    "bilal el khannouss":   1.30,
    "nayef aguerd":         1.20,
    "issa diop":            1.05,
    "neil el aynaoui":      1.15,
    "abde ezzalzouli":      1.05,
    "ayoub el kaabi":       1.08,
    # usa
    "matt turner":          1.16,
    "tyler adams":          1.10,
    "yunus musah":          1.25,  
    "weston mckennie":      1.15,
    "timothy weah":         1.15,
    "folarin balogun":      1.05,
    "christian pulisic":    1.20,  
    # japan
    "zion suzuki":          1.30,
    "koki machida":         1.06,
    "hiroki ito":           1.15,
    "shogo taniguchi":      1.12,
    "tsuyoshi watanabe":    1.10,
    "wataru endo":          1.20,
    "kaoru mitoma":         1.15,  
    "takefusa kubo":        1.12,
    "endrick":              1.40,  
    "loïs openda":          1.15,  
    "takehiro tomiyasu":    1.12,  
    # croatia
    "luka modrić":          1.20,
    "dominik livaković":    1.30,
    "josip juranović":      1.20,
    "duje ćaleta-car":      1.12,
    "borna sosa":           1.20,
    "mario pašalić":        1.05,
    "nikola vlašić":        1.05,
    # senegal 
    "ismaila sarr":         1.50,  
    "sadio mané":           1.15,
    "ismail jakobs":        1.10,
    "pape matar sarr":      1.05,
    "lamine camara":        1.06,
    "nicolas jackson":      1.06,
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
            df.loc[mask, "weighted_score"] = (df.loc[mask, "weighted_score"] * mult).clip(0, 100).round(2)
            n_boost += mask.sum()
    print(f"  {len(POSITION_OVERRIDES)} position overrides ({n_pos} rows)  |  "
          f"{len(REPUTATION_BOOSTS)} reputation boosts ({n_boost} rows)")
    return df


def get_candidate_pool(df, country, season=None):
    codes = COUNTRY_TO_CODES.get(country.lower(), [])
    if not codes:
        raise ValueError(f"Unknown country: {country}")
    pool = df[df["nation"].isin(codes)].copy()
    pool = pool[~pool["player"].str.lower().isin(EXCLUSIONS)]
    if season:
        pool = pool[pool["season"] == str(season)]
    pool = pool.sort_values(["weighted_score", "season"], ascending=[False, False], na_position="last")
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

    if results:
        combined = pd.concat(results.values(), ignore_index=True)
        combined_path = os.path.join(OUT_DIR, "squads_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Combined: {combined_path}  ({len(combined)} rows)")


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