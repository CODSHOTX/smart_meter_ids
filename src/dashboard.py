"""Helpers for the Streamlit dashboard: persistence alerts and event grouping.

Suspected attack types are derived only from observable readings and baselines,
never from the ground-truth attack_type label.
"""

import numpy as np
import pandas as pd

MODEL_OPTIONS = {
    "Random Forest": {
        "key": "random_forest",
        "score": "rf_proba",
        "pred": {"f1": "rf_pred", "far2": "rf_pred_far2"},
        "score_label": "P(attack)",
    },
    "Isolation Forest": {
        "key": "isolation_forest",
        "score": "if_score",
        "pred": {"f1": "if_pred", "far2": "if_pred_far2"},
        "score_label": "anomaly score",
    },
    "Z-score": {
        "key": "z_score",
        "score": "z_score",
        "pred": {"f1": "z_pred", "far2": "z_pred_far2"},
        "score_label": "|z|",
        "abs_score": True,
    },
}

ZERO_KWH = 0.02
SPIKE_RATIO = 3.0
THEFT_RATIO = 0.7


def persistence_alert(flags, groups, k):
    """Alert at t if all of the last k hours (including t) are flagged, per meter."""
    flags = flags.astype("int8")
    if k == 1:
        return flags
    rolled = flags.groupby(groups, observed=True).transform(
        lambda s: s.rolling(k, min_periods=k).sum()
    )
    return (rolled == k).astype("int8")


def decision_score(df, model_cfg):
    """Return the model's decision score (abs for z-score)."""
    s = df[model_cfg["score"]].to_numpy(dtype=float)
    return np.abs(s) if model_cfg.get("abs_score") else s


def suspect_type(energy, baseline, peak_score):
    """Heuristic label from readings vs baseline. Never uses ground-truth labels."""
    energy = np.asarray(energy, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if len(energy) == 0:
        return "unknown"
    near_zero = (energy < ZERO_KWH).mean()
    ratio = energy / (baseline + 0.01)
    if near_zero >= 0.5:
        return "possible meter outage/zeroing"
    if len(energy) <= 3 and (ratio >= SPIKE_RATIO).any():
        return "possible spike / data injection"
    if (ratio.max() >= SPIKE_RATIO) and (ratio.mean() > 1.2):
        return "possible spike / data injection"
    if (ratio.mean() <= THEFT_RATIO) or ((ratio < THEFT_RATIO).mean() >= 0.5):
        return "possible energy theft (reduced readings)"
    if peak_score >= 0.8:
        return "possible spike / data injection"
    return "possible energy theft (reduced readings)"


def group_alert_events(df, alert, score, baseline_col="baseline_median"):
    """Collapse consecutive alert hours per meter into event rows.

    Expects df sorted by LCLid, timestamp with columns LCLid, timestamp,
    energy_kwh and baseline_col. alert and score are aligned Series/arrays.
    """
    out = df[["LCLid", "timestamp", "energy_kwh", baseline_col]].copy()
    out["alert"] = np.asarray(alert, dtype="int8")
    out["score"] = np.asarray(score, dtype=float)
    out = out[out["alert"] == 1]
    if out.empty:
        return pd.DataFrame(columns=[
            "LCLid", "start", "end", "duration_h", "peak_score", "suspected_type",
        ])

    events = []
    for meter, grp in out.groupby("LCLid", observed=True):
        grp = grp.sort_values("timestamp")
        gap = grp["timestamp"].diff().dt.total_seconds().fillna(0) > 3600
        grp = grp.assign(event_id=gap.cumsum())
        for _, ev in grp.groupby("event_id"):
            energy = ev["energy_kwh"].to_numpy()
            baseline = ev[baseline_col].to_numpy()
            peak = float(ev["score"].max())
            events.append({
                "LCLid": meter,
                "start": ev["timestamp"].iloc[0],
                "end": ev["timestamp"].iloc[-1],
                "duration_h": int(len(ev)),
                "peak_score": peak,
                "suspected_type": suspect_type(energy, baseline, peak),
            })
    return pd.DataFrame(events).sort_values(["start", "LCLid"]).reset_index(drop=True)
