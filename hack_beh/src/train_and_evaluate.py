"""
Fake Engagement Detection - Full ML Pipeline
=============================================
Models:
  1. Random Forest (primary classifier - interpretable)
  2. Gradient Boosting (best performance)
  3. Isolation Forest (unsupervised anomaly detection - novel angle)
  4. Logistic Regression (baseline)

Outputs:
  - Bot probability score (0-1)
  - Authenticity score (0-100)
  - Bot type classification (genuine/spambot1/spambot2/spambot3)
  - Behavioral anomaly explanation via feature importance
  - All evaluation plots
"""

import warnings
warnings.filterwarnings("ignore")
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    f1_score, accuracy_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.pipeline import Pipeline

DATA_PATH   = "../data/cresci2017_reproduced.csv"
MODEL_DIR   = "../models"
OUTPUT_DIR  = "../outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "followers_count", "friends_count", "statuses_count", "listed_count",
    "favourites_count", "account_age_days", "tweets_per_day",
    "follower_friend_ratio", "timing_regularity", "burst_score",
    "url_ratio", "retweet_ratio", "hashtag_density", "mention_ratio",
    "linguistic_consistency", "engagement_rate", "network_clustering",
    "profile_completeness", "name_entropy", "coordinated_score",
    "has_profile_image", "has_geo_enabled", "has_url_in_profile"
]

PALETTE = {
    "genuine":   "#2DD4BF",
    "spambot_1": "#F87171",
    "spambot_2": "#FB923C",
    "spambot_3": "#A78BFA",
    "bot":       "#EF4444",
}

# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} accounts | {df['is_bot'].sum():,} bots | {(~df['is_bot'].astype(bool)).sum():,} genuine")
    X = df[FEATURE_COLS].fillna(0)
    y_binary = df["is_bot"]
    y_multi  = df["account_type"]
    return df, X, y_binary, y_multi


# ─────────────────────────────────────────────────────────────────────────────
def train_binary_classifier(X, y):
    """Primary Bot vs Genuine classifier."""
    print("\n" + "="*60)
    print("  BINARY CLASSIFICATION: Bot vs Genuine")
    print("="*60)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.85, random_state=42
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0))
        ]),
    }

    results = {}
    best_auc, best_name, best_model = 0, None, None

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        auc  = roc_auc_score(y_te, y_prob)
        f1   = f1_score(y_te, y_pred, average="weighted")
        acc  = accuracy_score(y_te, y_pred)
        results[name] = {"auc": auc, "f1": f1, "acc": acc,
                         "y_prob": y_prob, "y_pred": y_pred}
        print(f"\n{name}:")
        print(f"  AUC-ROC : {auc:.4f}")
        print(f"  F1      : {f1:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    print(f"\n✅ Best Model: {best_name} (AUC={best_auc:.4f})")
    y_prob_best = results[best_name]["y_prob"]
    y_pred_best = results[best_name]["y_pred"]
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred_best, target_names=["Genuine", "Bot"]))

    # Cross-validation
    cv = cross_val_score(best_model, X, y, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1)
    print(f"5-Fold CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    # Save model
    with open(f"{MODEL_DIR}/bot_detector.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model, best_name, X_tr, X_te, y_tr, y_te, results


# ─────────────────────────────────────────────────────────────────────────────
def train_multiclass(X, y_multi):
    """Bot type classifier (genuine / spambot1 / spambot2 / spambot3)."""
    print("\n" + "="*60)
    print("  MULTICLASS: Bot Campaign Type")
    print("="*60)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_multi, test_size=0.2,
                                               random_state=42, stratify=y_multi)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(classification_report(y_te, y_pred))
    with open(f"{MODEL_DIR}/campaign_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    return model, X_te, y_te, y_pred


# ─────────────────────────────────────────────────────────────────────────────
def train_anomaly_detector(X):
    """Unsupervised Isolation Forest — detects novel bot patterns."""
    print("\n" + "="*60)
    print("  UNSUPERVISED ANOMALY DETECTION: Isolation Forest")
    print("="*60)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso = IsolationForest(n_estimators=200, contamination=0.40,
                          random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    anomaly_scores = -iso.score_samples(X_scaled)  # higher = more anomalous
    print(f"Anomaly score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
    with open(f"{MODEL_DIR}/isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": iso, "scaler": scaler}, f)
    return iso, scaler, anomaly_scores


# ─────────────────────────────────────────────────────────────────────────────
def compute_authenticity_score(df, bot_prob, anomaly_scores):
    """
    Composite Authenticity Score (0=bot, 100=genuine).
    Combines ML probability + anomaly signal + behavioral heuristics.
    """
    # Normalize anomaly to 0-1
    a_min, a_max = anomaly_scores.min(), anomaly_scores.max()
    anomaly_norm = (anomaly_scores - a_min) / (a_max - a_min + 1e-9)

    # Behavioral heuristics
    timing_penalty   = df["timing_regularity"].values * 15
    burst_penalty    = df["burst_score"].values * 10
    coord_penalty    = df["coordinated_score"].values * 15
    profile_bonus    = df["profile_completeness"].values * 10
    engagement_bonus = np.clip(df["engagement_rate"].values / 10, 0, 1) * 10

    raw = (
        (1 - bot_prob) * 50 +           # ML signal (50 pts)
        (1 - anomaly_norm) * 20 +        # Anomaly signal (20 pts)
        profile_bonus +                   # Profile completeness (10 pts)
        engagement_bonus -                # Engagement quality (10 pts)
        timing_penalty -                  # Timing regularity penalty
        burst_penalty -                   # Burst pattern penalty
        coord_penalty                     # Coordination penalty
    )
    return np.clip(raw, 0, 100).round(2)


# ─────────────────────────────────────────────────────────────────────────────
def generate_intervention(row):
    """Rule-based behavioral anomaly explanation."""
    flags = []
    if row["timing_regularity"] > 0.7:
        flags.append("⚠️ Robotic posting regularity detected")
    if row["burst_score"] > 0.6:
        flags.append("⚠️ Abnormal activity burst pattern")
    if row["url_ratio"] > 0.7:
        flags.append("🔗 Excessive URL promotion (spam signal)")
    if row["linguistic_consistency"] > 0.7:
        flags.append("📝 Low content diversity (repetitive messages)")
    if row["coordinated_score"] > 0.6:
        flags.append("🤝 Coordinated network behavior detected")
    if row["follower_friend_ratio"] < 0.1 and row["friends_count"] > 100:
        flags.append("👥 Follow-heavy, few followers (bot pattern)")
    if row["profile_completeness"] < 0.2:
        flags.append("🧩 Incomplete profile (missing image/bio/location)")
    if row["name_entropy"] > 4.2:
        flags.append("🔤 High username randomness (generated name)")
    if row["engagement_rate"] < 0.01:
        flags.append("📉 Near-zero engagement rate")
    if not flags:
        flags.append("✅ No anomalous behaviors detected")
    return flags


# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, model_name, ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    rf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-15:]
    colors  = ["#EF4444" if importances[i] > np.percentile(importances, 75) else "#94A3B8"
               for i in indices]
    ax.barh([FEATURE_COLS[i] for i in indices], importances[indices], color=colors)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"Top 15 Behavioral Triggers\n({model_name})", fontsize=13, fontweight="bold")
    ax.axvline(np.median(importances), color="#64748B", linestyle="--", linewidth=1, alpha=0.6)
    if show:
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_roc_curves(X_te, y_te, results, ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#EF4444", "#3B82F6", "#10B981"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Bot Detection", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if show:
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_confusion_matrix(y_te, y_pred, ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn_r",
                xticklabels=["Genuine", "Bot"],
                yticklabels=["Genuine", "Bot"],
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    if show:
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_bot_probability_distribution(df, bot_prob, ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    genuine_prob = bot_prob[df["is_bot"] == 0]
    bot_prob_bot = bot_prob[df["is_bot"] == 1]
    ax.hist(genuine_prob, bins=40, alpha=0.7, color=PALETTE["genuine"],
            label=f"Genuine (n={len(genuine_prob):,})", density=True)
    ax.hist(bot_prob_bot, bins=40, alpha=0.7, color=PALETTE["bot"],
            label=f"Bots (n={len(bot_prob_bot):,})", density=True)
    ax.axvline(0.5, color="#1E293B", linestyle="--", lw=2, label="Decision threshold")
    ax.set_xlabel("Bot Probability Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Bot Probability Distribution by Account Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    if show:
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/probability_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_authenticity_by_type(df, auth_scores, ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df.copy()
    df_plot["authenticity_score"] = auth_scores
    order = ["genuine", "spambot_3", "spambot_2", "spambot_1"]
    colors = [PALETTE.get(t, "#94A3B8") for t in order]
    data_by_type = [df_plot[df_plot["account_type"] == t]["authenticity_score"].values
                    for t in order]
    bp = ax.boxplot(data_by_type, patch_artist=True, notch=True,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(["Genuine", "Spambot 3\n(Sophisticated)", "Spambot 2\n(Advanced)", "Spambot 1\n(Simple)"], fontsize=9)
    ax.set_ylabel("Authenticity Score (0-100)", fontsize=11)
    ax.set_title("Authenticity Score by Account Type", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if show:
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/authenticity_by_type.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_behavioral_radar(df, ax=None):
    """Radar chart comparing behavioral fingerprints."""
    show = ax is None
    categories = ["timing_regularity", "burst_score", "url_ratio",
                  "linguistic_consistency", "coordinated_score",
                  "retweet_ratio", "hashtag_density"]
    labels = ["Timing\nRegularity", "Burst\nScore", "URL\nRatio",
              "Linguistic\nConsistency", "Coordinated\nScore",
              "Retweet\nRatio", "Hashtag\nDensity"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    type_colors = {"genuine": PALETTE["genuine"], "spambot_1": PALETTE["spambot_1"],
                   "spambot_2": PALETTE["spambot_2"], "spambot_3": PALETTE["spambot_3"]}
    type_labels = {"genuine": "Genuine", "spambot_1": "Spambot 1",
                   "spambot_2": "Spambot 2", "spambot_3": "Spambot 3"}

    for atype, color in type_colors.items():
        vals = df[df["account_type"] == atype][categories].mean().tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color,
                label=type_labels[atype])
        ax.fill(angles, vals, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Behavioral Fingerprint Radar\nby Account Type", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    if show:
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/behavioral_radar.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_anomaly_scatter(df, anomaly_scores, bot_prob, ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    colors = [PALETTE.get(t, "#94A3B8") for t in df["account_type"]]
    sc = ax.scatter(anomaly_scores, bot_prob, c=colors, alpha=0.35, s=12)
    ax.axhline(0.5, color="#1E293B", linestyle="--", lw=1.5, alpha=0.7, label="Bot threshold")
    ax.axvline(np.percentile(anomaly_scores, 70), color="#DC2626", linestyle="--",
               lw=1.5, alpha=0.7, label="Anomaly threshold")
    ax.set_xlabel("Isolation Forest Anomaly Score", fontsize=11)
    ax.set_ylabel("ML Bot Probability", fontsize=11)
    ax.set_title("ML Probability vs Anomaly Score\n(Color by Account Type)", fontsize=13, fontweight="bold")
    patches = [mpatches.Patch(color=PALETTE[t], label=t.replace("_", " ").title())
               for t in ["genuine", "spambot_1", "spambot_2", "spambot_3"]]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(alpha=0.2)
    if show:
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/anomaly_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
def build_master_dashboard(df, model, model_name, X_te, y_te, results, anomaly_scores, bot_prob_full, auth_scores):
    """Single comprehensive dashboard figure."""
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0F172A")
    gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.38)

    title_kw   = {"color": "white", "fontsize": 12, "fontweight": "bold"}
    label_kw   = {"color": "#CBD5E1", "fontsize": 9}
    tick_color = "#94A3B8"

    def style_ax(ax):
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors=tick_color, labelsize=8)
        ax.xaxis.label.set_color(tick_color)
        ax.yaxis.label.set_color(tick_color)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(alpha=0.15, color="#475569")
        return ax

    # ── Row 0: Header ──
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor("#0F172A")
    ax_header.axis("off")
    ax_header.text(0.5, 0.7, "🔍  Fake Engagement Detection System",
                   transform=ax_header.transAxes, ha="center", va="center",
                   fontsize=22, fontweight="bold", color="white")
    ax_header.text(0.5, 0.1,
                   f"Dataset: Cresci-2017 (5,827 accounts)  │  Best Model: {model_name}  │  AUC: {max(r['auc'] for r in results.values()):.4f}  │  Behavioral Features: {len(FEATURE_COLS)}",
                   transform=ax_header.transAxes, ha="center", va="center",
                   fontsize=11, color="#94A3B8")

    # ── Row 1: 4 metric panels ──
    stats = [
        ("Total Accounts", f"{len(df):,}", "#2DD4BF"),
        ("Bot Accounts",   f"{df['is_bot'].sum():,}", "#EF4444"),
        ("Best AUC-ROC",   f"{max(r['auc'] for r in results.values()):.4f}", "#A78BFA"),
        ("Avg Auth Score\n(Genuine)", f"{auth_scores[df['is_bot']==0].mean():.1f}/100", "#34D399"),
    ]
    for i, (title, value, color) in enumerate(stats):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor(color + "22")
        ax.axis("off")
        for spine in ax.spines.values(): spine.set_edgecolor(color)
        ax.text(0.5, 0.65, value, transform=ax.transAxes, ha="center",
                fontsize=22, fontweight="bold", color=color)
        ax.text(0.5, 0.25, title, transform=ax.transAxes, ha="center",
                fontsize=10, color="#CBD5E1")

    # ── Row 2, Col 0-1: ROC Curves ──
    ax_roc = style_ax(fig.add_subplot(gs[2, :2]))
    colors_roc = ["#EF4444", "#3B82F6", "#10B981"]
    for (name, res), color in zip(results.items(), colors_roc):
        fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC={res['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], "--", color="#475569", lw=1)
    ax_roc.set_title("ROC Curves — Bot Detection", **title_kw)
    ax_roc.set_xlabel("False Positive Rate", **label_kw)
    ax_roc.set_ylabel("True Positive Rate", **label_kw)
    ax_roc.legend(fontsize=8, facecolor="#1E293B", edgecolor="#475569", labelcolor="white")

    # ── Row 2, Col 2-3: Feature Importance ──
    ax_fi = style_ax(fig.add_subplot(gs[2, 2:]))
    rf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-12:]
    feat_colors = ["#EF4444" if importances[i] > np.percentile(importances, 75) else "#64748B"
                   for i in top_idx]
    ax_fi.barh([FEATURE_COLS[i] for i in top_idx], importances[top_idx],
               color=feat_colors)
    ax_fi.set_title("Top Behavioral Triggers", **title_kw)
    ax_fi.set_xlabel("Feature Importance", **label_kw)

    fig.suptitle("", y=0)
    plt.savefig(f"{OUTPUT_DIR}/master_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor="#0F172A")
    plt.close()
    print(f"\n✅ Master dashboard saved → {OUTPUT_DIR}/master_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
def build_individual_plots(df, model, model_name, X_te, y_te, y_pred, results, anomaly_scores, bot_prob_full, auth_scores):
    print("\nGenerating individual plots...")

    plot_roc_curves(X_te, y_te, results)
    print("  ✓ ROC curves")

    best_result = max(results.items(), key=lambda x: x[1]["auc"])
    plot_confusion_matrix(y_te, best_result[1]["y_pred"])
    print("  ✓ Confusion matrix")

    plot_feature_importance(model, model_name)
    print("  ✓ Feature importance")

    plot_bot_probability_distribution(df, bot_prob_full)
    print("  ✓ Probability distribution")

    plot_authenticity_by_type(df, auth_scores)
    print("  ✓ Authenticity by type")

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    plot_behavioral_radar(df, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/behavioral_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Behavioral radar")

    plot_anomaly_scatter(df, anomaly_scores, bot_prob_full)
    print("  ✓ Anomaly scatter")


# ─────────────────────────────────────────────────────────────────────────────
def save_results_json(df, bot_prob_full, auth_scores, anomaly_scores, best_model_name, best_auc):
    """Save full predictions with explanations."""
    df_out = df.copy()
    df_out["bot_probability"]    = bot_prob_full.round(4)
    df_out["authenticity_score"] = auth_scores
    df_out["anomaly_score"]      = anomaly_scores.round(4)
    df_out["predicted_bot"]      = (bot_prob_full > 0.5).astype(int)

    df_out.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)

    # Sample explanations for 5 accounts
    sample = df_out.sample(5, random_state=42)
    explanations = []
    for _, row in sample.iterrows():
        flags = generate_intervention(row)
        explanations.append({
            "account_id":         row["account_id"],
            "account_type":       row["account_type"],
            "bot_probability":    round(row["bot_probability"], 3),
            "authenticity_score": round(row["authenticity_score"], 1),
            "behavioral_flags":   flags,
            "recommendation":     (
                "🚨 Suspend / investigate account immediately"
                if row["bot_probability"] > 0.8 else
                "⚠️ Flag for manual review"
                if row["bot_probability"] > 0.5 else
                "✅ Account appears genuine"
            )
        })

    summary = {
        "model":       best_model_name,
        "auc_roc":     round(best_auc, 4),
        "total_accounts": len(df),
        "flagged_bots":   int((bot_prob_full > 0.5).sum()),
        "avg_auth_genuine": round(float(auth_scores[df["is_bot"]==0].mean()), 2),
        "avg_auth_bot":     round(float(auth_scores[df["is_bot"]==1].mean()), 2),
        "sample_explanations": explanations
    }

    with open(f"{OUTPUT_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved → {OUTPUT_DIR}/predictions.csv")
    print(f"✅ Summary saved → {OUTPUT_DIR}/results_summary.json")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "🔍 " * 20)
    print("   FAKE ENGAGEMENT DETECTION SYSTEM — CRESCI-2017")
    print("🔍 " * 20)

    df, X, y_binary, y_multi = load_data()

    # 1. Binary bot detection
    best_model, best_name, X_tr, X_te, y_tr, y_te, results = train_binary_classifier(X, y_binary)

    # 2. Campaign type multiclass
    multi_model, Xm_te, ym_te, ym_pred = train_multiclass(X, y_multi)

    # 3. Isolation Forest anomaly detection
    iso_model, scaler, anomaly_scores = train_anomaly_detector(X)

    # 4. Full dataset bot probabilities
    bot_prob_full = best_model.predict_proba(X)[:, 1]

    # 5. Authenticity score
    auth_scores = compute_authenticity_score(df, bot_prob_full, anomaly_scores)

    print(f"\n📊 Authenticity Score Summary:")
    for atype in ["genuine", "spambot_1", "spambot_2", "spambot_3"]:
        mask = df["account_type"] == atype
        print(f"  {atype:12s}: {auth_scores[mask].mean():.1f} ± {auth_scores[mask].std():.1f}")

    # 6. Plots
    best_result = max(results.items(), key=lambda x: x[1]["auc"])
    build_individual_plots(
        df, best_model, best_name, X_te, y_te,
        best_result[1]["y_pred"], results, anomaly_scores, bot_prob_full, auth_scores
    )

    # 7. Master dashboard
    build_master_dashboard(
        df, best_model, best_name, X_te, y_te,
        results, anomaly_scores, bot_prob_full, auth_scores
    )

    # 8. Save results
    summary = save_results_json(df, bot_prob_full, auth_scores, anomaly_scores,
                                best_name, best_result[1]["auc"])

    print("\n" + "="*60)
    print("✅  PIPELINE COMPLETE")
    print(f"   Model:       {summary['model']}")
    print(f"   AUC-ROC:     {summary['auc_roc']}")
    print(f"   Flagged Bots:{summary['flagged_bots']:,} / {summary['total_accounts']:,}")
    print(f"   Auth (genuine): {summary['avg_auth_genuine']}/100")
    print(f"   Auth (bot):     {summary['avg_auth_bot']}/100")
    print("="*60)


if __name__ == "__main__":
    main()
