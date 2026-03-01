"""
FakeRadar — Real-Time Account Analyzer
=======================================
Usage:
  python predict.py --account_id ACC00123
  python predict.py --csv my_accounts.csv

Outputs:
  - Bot Probability (0–1)
  - Authenticity Score (0–100)
  - Behavioral anomaly flags
  - Recommended intervention
"""

import pickle, json, sys, argparse
import numpy as np
import pandas as pd
import os

MODEL_PATH  = "../models/bot_detector.pkl"
MULTI_PATH  = "../models/campaign_classifier.pkl"
ISO_PATH    = "../models/isolation_forest.pkl"
DATA_PATH   = "../data/cresci2017_reproduced.csv"

FEATURE_COLS = [
    "followers_count", "friends_count", "statuses_count", "listed_count",
    "favourites_count", "account_age_days", "tweets_per_day",
    "follower_friend_ratio", "timing_regularity", "burst_score",
    "url_ratio", "retweet_ratio", "hashtag_density", "mention_ratio",
    "linguistic_consistency", "engagement_rate", "network_clustering",
    "profile_completeness", "name_entropy", "coordinated_score",
    "has_profile_image", "has_geo_enabled", "has_url_in_profile"
]


def load_models():
    with open(MODEL_PATH, "rb") as f:  binary_model = pickle.load(f)
    with open(MULTI_PATH,  "rb") as f: multi_model  = pickle.load(f)
    with open(ISO_PATH,    "rb") as f: iso_data     = pickle.load(f)
    return binary_model, multi_model, iso_data


def compute_authenticity(bot_prob, row, anomaly_score):
    a_norm = min(max((anomaly_score - 0.37) / (0.59 - 0.37), 0), 1)
    raw = (
        (1 - bot_prob) * 50 +
        (1 - a_norm) * 20 +
        row.get("profile_completeness", 0.5) * 10 +
        min(row.get("engagement_rate", 0) / 10, 1) * 10 -
        row.get("timing_regularity", 0) * 15 -
        row.get("burst_score", 0) * 10 -
        row.get("coordinated_score", 0) * 15
    )
    return round(float(np.clip(raw, 0, 100)), 2)


def get_flags(row):
    flags = []
    if row.get("timing_regularity", 0) > 0.7:
        flags.append(("⚠️  Robotic posting regularity", "HIGH"))
    if row.get("burst_score", 0) > 0.6:
        flags.append(("⚠️  Abnormal activity burst pattern", "HIGH"))
    if row.get("url_ratio", 0) > 0.7:
        flags.append(("🔗  Excessive URL promotion", "HIGH"))
    if row.get("linguistic_consistency", 0) > 0.7:
        flags.append(("📝  Low content diversity (repetitive messages)", "HIGH"))
    if row.get("coordinated_score", 0) > 0.6:
        flags.append(("🤝  Coordinated network behavior", "HIGH"))
    if row.get("follower_friend_ratio", 1) < 0.1 and row.get("friends_count", 0) > 100:
        flags.append(("👥  Follow-heavy with few followers", "MED"))
    if row.get("profile_completeness", 1) < 0.2:
        flags.append(("🧩  Incomplete profile", "MED"))
    if row.get("name_entropy", 3) > 4.2:
        flags.append(("🔤  High username randomness", "MED"))
    if row.get("engagement_rate", 1) < 0.01:
        flags.append(("📉  Near-zero engagement rate", "MED"))
    if not flags:
        flags.append(("✅  No anomalous behaviors detected", "NONE"))
    return flags


def get_intervention(bot_prob):
    if bot_prob > 0.8:
        return "🚨 SUSPEND — Investigate IP clusters, report to platform Trust & Safety"
    elif bot_prob > 0.5:
        return "⚠️  FLAG — Manual review, reduce algorithmic amplification, require 2FA"
    else:
        return "✅  GENUINE — Continue passive monitoring, periodic re-evaluation"


def analyze(account_data: dict, models):
    binary_model, multi_model, iso_data = models
    X = pd.DataFrame([account_data])[FEATURE_COLS].fillna(0)

    bot_prob  = float(binary_model.predict_proba(X)[0, 1])
    campaign  = multi_model.predict(X)[0]
    X_scaled  = iso_data["scaler"].transform(X)
    anomaly   = float(-iso_data["model"].score_samples(X_scaled)[0])
    auth      = compute_authenticity(bot_prob, account_data, anomaly)
    flags     = get_flags(account_data)
    action    = get_intervention(bot_prob)

    verdict   = "BOT" if bot_prob > 0.5 else "GENUINE"

    return {
        "account_id":        account_data.get("account_id", "unknown"),
        "verdict":           verdict,
        "bot_probability":   round(bot_prob, 4),
        "authenticity_score": auth,
        "anomaly_score":     round(anomaly, 4),
        "predicted_campaign": campaign,
        "behavioral_flags":  [{"flag": f, "severity": s} for f, s in flags],
        "recommended_action": action
    }


def pretty_print(result):
    print("\n" + "─"*55)
    print(f"  FakeRadar Analysis — Account: {result['account_id']}")
    print("─"*55)
    verdict_color = "\033[91m" if result["verdict"] == "BOT" else "\033[92m"
    reset = "\033[0m"
    print(f"  Verdict:           {verdict_color}{result['verdict']}{reset}")
    print(f"  Bot Probability:   {result['bot_probability']:.1%}")
    print(f"  Authenticity Score:{result['authenticity_score']}/100")
    print(f"  Anomaly Score:     {result['anomaly_score']:.4f}")
    print(f"  Predicted Campaign:{result['predicted_campaign']}")
    print("\n  Behavioral Flags:")
    for item in result["behavioral_flags"]:
        sev_map = {"HIGH": "\033[91m", "MED": "\033[93m", "NONE": "\033[92m"}
        col = sev_map.get(item["severity"], "")
        print(f"    {col}{item['flag']}{reset}")
    print(f"\n  Recommended Action:\n    {result['recommended_action']}")
    print("─"*55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FakeRadar — Bot Detection CLI")
    parser.add_argument("--account_id", help="Analyze specific account from dataset")
    parser.add_argument("--csv",        help="Analyze all accounts in a CSV file")
    parser.add_argument("--json",       action="store_true", help="Output as JSON")
    args = parser.parse_args()

    models = load_models()
    df = pd.read_csv(DATA_PATH)

    if args.account_id:
        row = df[df["account_id"] == args.account_id]
        if row.empty:
            print(f"Account {args.account_id} not found. Using random sample.")
            row = df.sample(1)
        account = row.iloc[0].to_dict()
        result  = analyze(account, models)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            pretty_print(result)

    elif args.csv:
        ext_df  = pd.read_csv(args.csv)
        results = []
        for _, row in ext_df.iterrows():
            r = analyze(row.to_dict(), models)
            results.append(r)
            pretty_print(r)
        out = args.csv.replace(".csv", "_analyzed.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {out}")

    else:
        # Demo: analyze 5 random accounts
        print("\n📊 Demo — Analyzing 5 Random Accounts\n")
        sample = df.sample(5, random_state=99)
        for _, row in sample.iterrows():
            result = analyze(row.to_dict(), models)
            pretty_print(result)


if __name__ == "__main__":
    main()
