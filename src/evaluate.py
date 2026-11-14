"""Evaluate detectors on TEST rows with persistence alerts and detection latency.

Alert rule: at hour t an alert fires if all of the last k consecutive hours
(including t) were flagged, for k in {1, 2, 3}. Detection latency is measured
from each attack window's first_modified_time to the first alert at or after
that time and at or before end_time.

Run from the project root:
    python -m src.evaluate
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.features import SPLIT_TIME
from src.models import binary_metrics, pred_columns

PRED_PATH = Path("data/processed/predictions.parquet")
ATTACKS_PATH = Path("data/processed/attacks.parquet")
REPORTS = Path("reports")
K_VALUES = (1, 2, 3)

SCORE_COLUMNS = {
    "z_score": ("z_score", True),          # (column, use abs)
    "isolation_forest": ("if_score", False),
    "random_forest": ("rf_proba", False),
}


def persistence_alert(flags, groups, k):
    """Alert at t if all of the last k hours (including t) are flagged, per meter."""
    flags = flags.astype("int8")
    if k == 1:
        return flags
    rolled = flags.groupby(groups, observed=True).transform(
        lambda s: s.rolling(k, min_periods=k).sum()
    )
    return (rolled == k).astype("int8")


def row_metrics_table(test, variants):
    """Build precision/recall/F1/FAR for every model/threshold/k variant."""
    rows = []
    for label, flag_col in variants.items():
        model, mode = label.split("/")
        for k in K_VALUES:
            alert = persistence_alert(test[flag_col], test["LCLid"], k)
            m = binary_metrics(test["label"], alert)
            rows.append({
                "model": model, "threshold": mode, "k": k,
                "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "false_alarm_rate": m["false_alarm_rate"],
                "false_alerts_per_meter_day": _false_alerts_per_meter_day(
                    test, alert),
            })
    return pd.DataFrame(rows)


def _false_alerts_per_meter_day(test, alert):
    """Mean false-alert hours per meter per calendar day on the test period."""
    false = (alert.to_numpy() == 1) & (test["label"].to_numpy() == 0)
    n_meters = test["LCLid"].nunique()
    n_days = test["timestamp"].dt.normalize().nunique()
    return false.sum() / (n_meters * n_days)


def latency_table(test, attacks, variants):
    """Detection latency from first_modified_time for every variant and k.

    Returns (summary, by_type) DataFrames. Latency is in hours; undetected
    windows contribute to the undetected count only.
    """
    test_attacks = attacks[attacks["start_time"] >= SPLIT_TIME].copy()
    summaries, by_types = [], []
    for label, flag_col in variants.items():
        model, mode = label.split("/")
        for k in K_VALUES:
            alert = persistence_alert(test[flag_col], test["LCLid"], k)
            alert_idx = test.loc[alert.to_numpy() == 1, ["LCLid", "timestamp"]]
            latencies = []
            for _, atk in test_attacks.iterrows():
                hits = alert_idx[
                    (alert_idx["LCLid"] == atk["LCLid"])
                    & (alert_idx["timestamp"] >= atk["first_modified_time"])
                    & (alert_idx["timestamp"] <= atk["end_time"])
                ]
                if hits.empty:
                    latencies.append({
                        "attack_id": atk["attack_id"],
                        "attack_type": atk["attack_type"],
                        "latency_h": np.nan,
                        "detected": 0,
                    })
                else:
                    first = hits["timestamp"].min()
                    latencies.append({
                        "attack_id": atk["attack_id"],
                        "attack_type": atk["attack_type"],
                        "latency_h": (first - atk["first_modified_time"])
                        / pd.Timedelta(hours=1),
                        "detected": 1,
                    })
            lat = pd.DataFrame(latencies)
            detected = lat[lat["detected"] == 1]["latency_h"]
            summaries.append({
                "model": model, "threshold": mode, "k": k,
                "n_windows": len(lat),
                "n_detected": int(lat["detected"].sum()),
                "n_undetected": int((lat["detected"] == 0).sum()),
                "detection_rate": lat["detected"].mean(),
                "latency_mean_h": detected.mean() if len(detected) else np.nan,
                "latency_median_h": detected.median() if len(detected) else np.nan,
            })
            for atype, grp in lat.groupby("attack_type", observed=True):
                det = grp[grp["detected"] == 1]["latency_h"]
                by_types.append({
                    "model": model, "threshold": mode, "k": k,
                    "attack_type": atype,
                    "n_windows": len(grp),
                    "n_detected": int(grp["detected"].sum()),
                    "n_undetected": int((grp["detected"] == 0).sum()),
                    "detection_rate": grp["detected"].mean(),
                    "latency_mean_h": det.mean() if len(det) else np.nan,
                    "latency_median_h": det.median() if len(det) else np.nan,
                })
    return pd.DataFrame(summaries), pd.DataFrame(by_types)


def recall_by_type(test, variants, k=1):
    """Share of attacked TEST rows flagged (after persistence) by attack_type."""
    attacked = test[test["label"] == 1]
    rows = []
    for label, flag_col in variants.items():
        model, mode = label.split("/")
        alert = persistence_alert(test[flag_col], test["LCLid"], k)
        a = alert.loc[attacked.index]
        for atype, idx in attacked.groupby("attack_type", observed=True).groups.items():
            rows.append({
                "model": model, "threshold": mode, "k": k,
                "attack_type": atype,
                "recall": a.loc[idx].mean(),
            })
    return pd.DataFrame(rows)


def plot_confusion(test, variants, out_dir):
    """Confusion matrices for RF and IF at far2, k=1."""
    targets = {
        "isolation_forest/far2": "if_pred_far2",
        "random_forest/far2": "rf_pred_far2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (label, col) in zip(axes, targets.items()):
        alert = persistence_alert(test[col], test["LCLid"], 1)
        cm = confusion_matrix(test["label"], alert, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{label}, k=1")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices_far2_k1.png", dpi=150)
    plt.close(fig)


def plot_roc_pr(test, out_dir):
    """ROC and precision-recall curves from raw scores on TEST rows."""
    y = test["label"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (col, use_abs) in SCORE_COLUMNS.items():
        score = test[col].abs().to_numpy() if use_abs else test[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        prec, rec, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        axes[1].plot(rec, prec, label=f"{name} (AP={auc(rec, prec):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC (TEST)")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (TEST)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_recall_bars(recall_df, out_dir, k=1):
    """Grouped bar chart of row-level recall by attack_type (k fixed)."""
    sub = recall_df[recall_df["k"] == k].copy()
    sub["variant"] = sub["model"] + "/" + sub["threshold"]
    pivot = sub.pivot(index="attack_type", columns="variant", values="recall")
    order = ["scaling", "zeroing", "spike", "ramp_down"]
    pivot = pivot.reindex([a for a in order if a in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(11, 4.5), width=0.8)
    ax.set(ylabel="Recall", xlabel="Attack type",
           title=f"TEST recall by attack type (k={k})")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "recall_by_attack_type.png", dpi=150)
    plt.close()


def plot_latency(by_type, out_dir, threshold="far2", k=1):
    """Mean latency (hours) by attack_type for one threshold mode and k."""
    sub = by_type[(by_type["threshold"] == threshold) & (by_type["k"] == k)].copy()
    pivot = sub.pivot(index="attack_type", columns="model", values="latency_mean_h")
    order = ["scaling", "zeroing", "spike", "ramp_down"]
    pivot = pivot.reindex([a for a in order if a in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(9, 4.5))
    ax.set(ylabel="Mean latency (hours)", xlabel="Attack type",
           title=f"Detection latency by attack type ({threshold}, k={k})")
    ax.legend(fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_by_attack_type.png", dpi=150)
    plt.close()


def evaluate():
    """Run the full TEST evaluation, save tables/plots and return the tables."""
    preds = pd.read_parquet(PRED_PATH)
    attacks = pd.read_parquet(ATTACKS_PATH)
    test = preds[preds["split"] == "test"].sort_values(["LCLid", "timestamp"]).reset_index(drop=True)
    variants = pred_columns()

    metrics = row_metrics_table(test, variants)
    latency, latency_by_type = latency_table(test, attacks, variants)
    recall = recall_by_type(test, variants, k=1)

    REPORTS.mkdir(exist_ok=True)
    plot_confusion(test, variants, REPORTS)
    plot_roc_pr(test, REPORTS)
    plot_recall_bars(recall, REPORTS, k=1)
    plot_latency(latency_by_type, REPORTS, threshold="far2", k=1)

    metrics_out = metrics.copy()
    metrics_out["section"] = "row_metrics"
    lat_out = latency.copy()
    lat_out["section"] = "latency"
    lat_type_out = latency_by_type.copy()
    lat_type_out["section"] = "latency_by_type"
    recall_out = recall.copy()
    recall_out["section"] = "recall_by_type"
    all_tables = pd.concat([metrics_out, lat_out, lat_type_out, recall_out],
                           ignore_index=True, sort=False)
    all_tables.to_csv(REPORTS / "metrics.csv", index=False)
    return metrics, latency, latency_by_type, recall


def main():
    """Evaluate on TEST, write reports/ and print the key tables."""
    metrics, latency, latency_by_type, recall = evaluate()

    print(f"TEST evaluation ({PRED_PATH})")
    print("\nRow-level metrics (with persistence k):")
    print(metrics.round(4).to_string(index=False))

    print("\nFalse alerts per meter per day:")
    print(metrics.pivot_table(index=["model", "threshold"], columns="k",
                              values="false_alerts_per_meter_day")
          .round(3).to_string())

    print("\nDetection latency (from first_modified_time), overall:")
    cols = ["model", "threshold", "k", "n_windows", "n_detected", "n_undetected",
            "detection_rate", "latency_mean_h", "latency_median_h"]
    print(latency[cols].round(3).to_string(index=False))

    print("\nDetection latency by attack_type (far2, k=1):")
    sub = latency_by_type[(latency_by_type["threshold"] == "far2")
                          & (latency_by_type["k"] == 1)]
    print(sub.round(3).to_string(index=False))

    print("\nRecall by attack_type (k=1):")
    print(recall.pivot_table(index=["model", "threshold"], columns="attack_type",
                             values="recall").round(3).to_string())

    print(f"\nWrote {REPORTS / 'metrics.csv'} and plots in {REPORTS}/")


if __name__ == "__main__":
    main()
