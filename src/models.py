"""Train the three detectors, pick thresholds on validation, and score all rows.

Time-based scheme: the train split is cut by time into "fit" (first 80%) and
"validation" (last 20%). Models are fitted on fit rows, thresholds are chosen on
validation rows, and test rows are only scored with the frozen models.

Detectors:
    z-score          |baseline_deviation / per-meter normal std| > threshold
    Isolation Forest -score_samples >= threshold, fitted on normal fit rows
    Random Forest    predict_proba >= threshold, fitted on all fit rows

Each detector gets two thresholds, both chosen on validation:
    f1   - maximises F1 (column <prefix>_pred)
    far2 - gives a 2% false-alarm rate on normal rows (column <prefix>_pred_far2)

Run from the project root:
    python -m src.models
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.features import FEATURE_COLUMNS

IN_PATH = Path("data/processed/features.parquet")
OUT_PATH = Path("data/processed/predictions.parquet")
MODELS_DIR = Path("models")
FIT_FRAC = 0.8
TARGET_FAR = 0.02
Z_GRID = np.round(np.arange(0.5, 10.0001, 0.05), 2)
EPS = 1e-6
SEED = 42

# name -> (column prefix, strict ">" comparison instead of ">=")
DETECTORS = {
    "z_score": ("z", True),
    "isolation_forest": ("if", False),
    "random_forest": ("rf", False),
}
THRESHOLD_MODES = {"f1": "_pred", "far2": "_pred_far2"}

OUTPUT_COLUMNS = ["LCLid", "timestamp", "split", "label", "attack_type", "attack_id",
                  "z_score", "z_pred", "z_pred_far2",
                  "if_score", "if_pred", "if_pred_far2",
                  "rf_proba", "rf_pred", "rf_pred_far2"]


def fit_val_masks(df, fit_frac=FIT_FRAC):
    """Split train rows by time into fit (first fit_frac) and validation masks."""
    train = df["split"] == "train"
    times = np.sort(df.loc[train, "timestamp"].unique())
    val_start = pd.Timestamp(times[int(len(times) * fit_frac)])
    fit = (train & (df["timestamp"] < val_start)).to_numpy()
    val = (train & (df["timestamp"] >= val_start)).to_numpy()
    return fit, val, val_start


def binary_metrics(y_true, y_pred):
    """Return precision, recall, F1 and false-alarm rate FP / (FP + TN)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm_rate": far}


def best_f1_on_grid(y_true, score, grid):
    """Return the grid threshold t maximising F1 of (score > t)."""
    f1s = [binary_metrics(y_true, (score > t).astype(int))["f1"] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def best_f1_threshold(y_true, score):
    """Return the score threshold t maximising F1 of (score >= t) over all cut points."""
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, EPS)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def far_threshold(normal_score, far=TARGET_FAR):
    """Return the threshold flagging a fraction `far` of normal rows."""
    return float(np.quantile(normal_score, 1 - far))


def flag(score, threshold, strict):
    """Apply a threshold: score > t if strict, else score >= t."""
    return ((score > threshold) if strict else (score >= threshold)).astype("int8")


def fit_zscore(df, fit):
    """Per-meter std of baseline_deviation on normal fit rows."""
    normal = df[fit & (df["label"] == 0).to_numpy()]
    std = normal.groupby("LCLid", observed=True)["baseline_deviation"].std()
    std.index = std.index.astype(str)
    return std


def zscore(df, meter_std):
    """Signed z-score of baseline_deviation using the fitted per-meter std."""
    std = df["LCLid"].astype(str).map(meter_std).to_numpy(dtype=float)
    return df["baseline_deviation"].to_numpy() / np.maximum(std, EPS)


def train_and_predict(df):
    """Fit all detectors, choose both thresholds on validation and score every row.

    Returns (predictions, isolation_forest, random_forest, thresholds).
    """
    fit, val, val_start = fit_val_masks(df)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label"].to_numpy()
    y_val = y[val]
    normal_val = val & (y == 0)

    meter_std = fit_zscore(df, fit)
    z = zscore(df, meter_std)

    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    iso.fit(X[fit & (y == 0)])
    if_score = -iso.score_samples(X)

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
    rf.fit(X[fit], y[fit])
    rf_proba = rf.predict_proba(X)[:, 1]

    decision = {"z_score": np.abs(z), "isolation_forest": if_score, "random_forest": rf_proba}
    preds = df.copy()
    preds["z_score"] = z
    preds["if_score"] = if_score
    preds["rf_proba"] = rf_proba

    thresholds = {"val_start": str(val_start), "feature_columns": FEATURE_COLUMNS}
    for name, (prefix, strict) in DETECTORS.items():
        score = decision[name]
        chosen = {
            "f1": (best_f1_on_grid(y_val, score[val], Z_GRID) if name == "z_score"
                   else best_f1_threshold(y_val, score[val])),
            "far2": far_threshold(score[normal_val]),
        }
        entry = {"rule": ("abs(z_score) > threshold" if strict
                          else f"{prefix}_{'proba' if name == 'random_forest' else 'score'}"
                               " >= threshold")}
        for mode, suffix in THRESHOLD_MODES.items():
            preds[prefix + suffix] = flag(score, chosen[mode], strict)
            entry[f"threshold_{mode}"] = chosen[mode]
            entry[f"validation_{mode}"] = binary_metrics(y_val, preds[prefix + suffix].to_numpy()[val])
        thresholds[name] = entry
    thresholds["z_score"]["meter_std"] = meter_std.round(6).to_dict()

    return preds[OUTPUT_COLUMNS], iso, rf, thresholds


def pred_columns():
    """Map a readable label like 'random_forest/f1' to its prediction column."""
    return {f"{name}/{mode}": prefix + suffix
            for name, (prefix, _) in DETECTORS.items()
            for mode, suffix in THRESHOLD_MODES.items()}


def test_report(preds):
    """Return (metrics, recall_by_type, event_detection) tables for TEST rows."""
    test = preds[preds["split"] == "test"]
    cols = pred_columns()
    metrics = pd.DataFrame({label: binary_metrics(test["label"], test[col])
                            for label, col in cols.items()}).T

    attacked = test[test["label"] == 1]
    recall = attacked.groupby("attack_type", observed=True)[list(cols.values())].mean()
    recall.columns = list(cols)

    events = attacked.groupby("attack_id")
    detected = events[list(cols.values())].max()
    detected.columns = list(cols)
    types = events["attack_type"].first()
    event_rate = detected.groupby(types, observed=True).mean()
    event_rate.loc["all"] = detected.mean()
    event_rate.insert(0, "n_windows", types.value_counts().reindex(event_rate.index)
                      .fillna(len(types)).astype(int))
    return metrics, recall, event_rate


def main():
    """Train detectors, save models, thresholds and predictions, print a test summary."""
    df = pd.read_parquet(IN_PATH)
    preds, iso, rf, thresholds = train_and_predict(df)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    preds.to_parquet(OUT_PATH, index=False)

    print(f"Validation starts at {thresholds['val_start']}")
    print("Thresholds chosen on validation:")
    thr = pd.DataFrame({name: {
        "rule": thresholds[name]["rule"],
        "max_F1": thresholds[name]["threshold_f1"],
        "FAR_2%": thresholds[name]["threshold_far2"],
        "val_F1@maxF1": thresholds[name]["validation_f1"]["f1"],
        "val_F1@FAR2%": thresholds[name]["validation_far2"]["f1"],
    } for name in DETECTORS}).T
    print(thr.to_string(float_format=lambda v: f"{v:.4f}"))

    metrics, recall, event_rate = test_report(preds)
    n_test = (preds["split"] == "test").sum()
    print(f"\nTEST rows only ({n_test:,} rows, "
          f"{preds.loc[preds['split'] == 'test', 'label'].mean():.2%} attacked):")
    print(metrics.round(4).to_string())
    print("\nTEST recall per attack_type (share of attacked rows flagged):")
    print(recall.T.round(3).to_string())
    print("\nTEST event-level detection rate (window detected if any row flagged):")
    print(event_rate.T.round(3).to_string())


if __name__ == "__main__":
    main()
