import re
import unicodedata
import pandas as pd

NT_CSV       = "player_national_performances.csv"
PROFILES_CSV = "player_profiles.csv"
FEATURES_CSV = "features.csv"


def normalize(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip().replace("-", " ")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    print("WC26 | Merging NT caps")

    nt       = pd.read_csv(NT_CSV, low_memory=False)
    profiles = pd.read_csv(PROFILES_CSV, low_memory=False)
    features = pd.read_csv(FEATURES_CSV, low_memory=False)

    caps = (
        nt.groupby("player_id")["matches"]
        .sum()
        .reset_index()
        .rename(columns={"matches": "nt_caps"})
    )
    caps = caps.merge(
        profiles[["player_id", "player_slug"]].drop_duplicates("player_id"),
        on="player_id", how="left"
    )
    caps["name_key"] = caps["player_slug"].apply(normalize)
    caps = caps[caps["name_key"] != ""].drop_duplicates("name_key")[["name_key", "nt_caps"]]

    features["name_key"] = features["player"].apply(normalize)
    features = features.merge(caps, on="name_key", how="left")
    features = features.drop(columns=["name_key"])

    matched = features["nt_caps"].notna().sum()
    total   = len(features)
    print(f"  Matched {matched}/{total} rows ({matched/total*100:.1f}%)")

    # Apply caps multiplier to weighted_score
    # 0 caps = *0.92, 50 caps = *1.05, 100+ caps = *1.12
    def caps_mult(caps):
        if pd.isna(caps):
            return 1.0
        return min(1.12, 0.92 + (max(0, caps) / 100) * 0.20)

    features["caps_multiplier"] = features["nt_caps"].apply(caps_mult).round(3)
    features["weighted_score"]  = (
        features["weighted_score"] * features["caps_multiplier"]
    ).clip(0, 100).round(2)

    top = (
        features[features["nt_caps"].notna()]
        .drop_duplicates("player")
        .nlargest(8, "nt_caps")[["player", "team", "nt_caps", "caps_multiplier", "weighted_score"]]
    )
    print("  Top caps in dataset:")
    print(top.to_string(index=False))

    features.to_csv(FEATURES_CSV, index=False)
    print(f"  Saved {FEATURES_CSV} with nt_caps column")


if __name__ == "__main__":
    main()