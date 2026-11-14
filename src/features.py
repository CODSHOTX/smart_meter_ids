"""Build per-meter detection features from attacked hourly data.

All rolling features look strictly backwards: the window for hour t covers
hours t-N .. t-1, so the current reading is compared with its own past.

Two baselines per (day_of_week, hour):
    fixed    - median energy_kwh per meter fitted on rows before split_time only
    adaptive - median of the same weekday-hour in each of the previous 4 weeks

Neither baseline uses labels. Features are left unscaled.

Run from the project root:
    python -m src.features
"""

from pathlib import Path

import pandas as pd

IN_PATH = Path("data/processed/attacked.parquet")
OUT_PATH = Path("data/processed/features.parquet")
SPLIT_TIME = pd.Timestamp("2013-09-13 12:00:00")
EPS = 0.01  # kWh; keeps ratios bounded when the reference consumption is ~0
LOW_KWH = 0.02
ADAPTIVE_WEEKS = 4
WEEK = 168

ID_COLUMNS = ["LCLid", "timestamp", "split", "label", "attack_type", "attack_id"]
FEATURE_COLUMNS = [
    "energy_kwh",
    "hour",
    "day_of_week",
    "is_weekend",
    "rolling24_mean",
    "rolling24_std",
    "rolling168_mean",
    "rate_of_change",
    "ratio_to_rolling24",
    "baseline_median",
    "baseline_deviation",
    "baseline_ratio",
    "baseline_ratio_mean3",
    "baseline_ratio_mean6",
    "baseline_ratio_mean12",
    "baseline_ratio_mean24",
    "low_count6",
    "low_count12",
    "rolling24_min",
    "rolling24_max",
    "rolling24_std_rel",
    "adaptive_baseline",
    "adaptive_deviation",
    "adaptive_ratio",
]


def _past_rolling(values, groups, window, stat):
    """Rolling stat over hours t-window .. t-1 per meter; NaN until the window is full."""
    prev = values.groupby(groups, observed=True).shift(1)
    return prev.groupby(groups, observed=True).transform(
        lambda s: getattr(s.rolling(window, min_periods=window), stat)()
    )


def fit_baseline(df, split_time):
    """Median energy_kwh per (LCLid, day_of_week, hour) from rows before split_time."""
    train = df[df["timestamp"] < split_time]
    return (train.groupby(["LCLid", "day_of_week", "hour"], observed=True)["energy_kwh"]
            .median()
            .rename("baseline_median"))


def adaptive_baseline(df, weeks=ADAPTIVE_WEEKS):
    """Median energy_kwh at the same weekday-hour in each of the previous `weeks` weeks.

    NaN until all `weeks` past values exist.
    """
    g = df.groupby("LCLid", observed=True)["energy_kwh"]
    lags = pd.concat([g.shift(WEEK * k) for k in range(1, weeks + 1)], axis=1)
    return lags.median(axis=1, skipna=False)


def build_features(df, split_time=SPLIT_TIME):
    """Return a feature table with ID_COLUMNS + FEATURE_COLUMNS.

    Expects one row per meter per hour with no gaps. Each meter's first
    ADAPTIVE_WEEKS weeks are dropped as warm-up, since the adaptive baseline
    (the longest look-back) is incomplete there.
    """
    split_time = pd.Timestamp(split_time)
    out = df.sort_values(["LCLid", "timestamp"]).reset_index(drop=True)
    steps = out.groupby("LCLid", observed=True)["timestamp"].diff().dropna()
    if (steps != pd.Timedelta(hours=1)).any():
        raise ValueError("each meter must have contiguous hourly timestamps")

    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.day_of_week
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")

    groups = out["LCLid"]
    energy = out["energy_kwh"]
    prev = energy.groupby(groups, observed=True).shift(1)
    out["rolling24_mean"] = _past_rolling(energy, groups, 24, "mean")
    out["rolling24_std"] = _past_rolling(energy, groups, 24, "std")
    out["rolling168_mean"] = _past_rolling(energy, groups, WEEK, "mean")
    out["rate_of_change"] = energy - prev
    out["ratio_to_rolling24"] = energy / (out["rolling24_mean"] + EPS)

    out = out.join(fit_baseline(out, split_time), on=["LCLid", "day_of_week", "hour"])
    out["baseline_deviation"] = energy - out["baseline_median"]
    out["baseline_ratio"] = energy / (out["baseline_median"] + EPS)

    for window in (3, 6, 12, 24):
        out[f"baseline_ratio_mean{window}"] = _past_rolling(
            out["baseline_ratio"], groups, window, "mean")
    is_low = (energy < LOW_KWH).astype(float)
    out["low_count6"] = _past_rolling(is_low, groups, 6, "sum")
    out["low_count12"] = _past_rolling(is_low, groups, 12, "sum")
    out["rolling24_min"] = _past_rolling(energy, groups, 24, "min")
    out["rolling24_max"] = _past_rolling(energy, groups, 24, "max")
    out["rolling24_std_rel"] = out["rolling24_std"] / (out["baseline_median"] + EPS)

    out["adaptive_baseline"] = adaptive_baseline(out)
    out["adaptive_deviation"] = energy - out["adaptive_baseline"]
    out["adaptive_ratio"] = energy / (out["adaptive_baseline"] + EPS)

    out["split"] = (out["timestamp"] >= split_time).map({False: "train", True: "test"})
    out = out[out["adaptive_baseline"].notna()].reset_index(drop=True)
    return out[ID_COLUMNS + FEATURE_COLUMNS]


def main():
    """Build features from attacked.parquet, save them and print a summary."""
    df = pd.read_parquet(IN_PATH)
    feats = build_features(df, SPLIT_TIME)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT_PATH, index=False)

    print(f"Shape: {feats.shape}")
    print(f"Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print("\nRows per split:")
    print(feats["split"].value_counts().to_string())
    print("\nNaNs per feature column:")
    print(feats[FEATURE_COLUMNS].isna().sum().to_string())


if __name__ == "__main__":
    main()
