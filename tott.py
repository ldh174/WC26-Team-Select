import pandas as pd
import argparse
import os

SQUAD_CSV = "squads/squads_combined.csv"
OUT_DIR   = "squads"
FORMATION = {"GK": 1, "DF": 4, "MF": 3, "FW": 3}


def load(u23=False):
    df = pd.read_csv(SQUAD_CSV, low_memory=False)
    df = df.sort_values(["weighted_score", "season"], ascending=[False, False], na_position="last")
    df = df.drop_duplicates(subset=["player"], keep="first")
    if u23:
        df = df[df["age"] <= 23]
    return df


def build_tott(df, label):
    print(f"\nWC26 | {label}")
    print("-" * 55)
    rows = []
    for pos, n in FORMATION.items():
        pool  = df[df["position_group"] == pos].copy()
        picks = pool.head(n).copy()
        picks["tott_type"]     = label
        picks["tott_position"] = pos
        rows.append(picks)
        for _, row in picks.iterrows():
            score = f"{row['weighted_score']:.1f}" if pd.notna(row["weighted_score"]) else "N/A"
            age   = f"age {int(row['age'])}" if pd.notna(row["age"]) else ""
            print(f"  {pos}  {row['player']:<25} {row['team']:<22} {score}  {age}")
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="WC26 team of the tournament")
    parser.add_argument("--u23",  action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    frames = []

    if args.both:
        frames.append(build_tott(load(u23=False), "Team of the Tournament"))
        frames.append(build_tott(load(u23=True),  "U23 Team of the Tournament"))
    elif args.u23:
        frames.append(build_tott(load(u23=True),  "U23 Team of the Tournament"))
    else:
        frames.append(build_tott(load(u23=False), "Team of the Tournament"))

    if frames:
        out = pd.concat(frames, ignore_index=True)
        path = os.path.join(OUT_DIR, "tott.csv")
        out.to_csv(path, index=False)
        print(f"\n  Saved: {path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()