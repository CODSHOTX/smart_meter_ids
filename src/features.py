"""Build per-meter detection features from attacked hourly data.

All rolling features look strictly backwards: the window for hour t covers
hours t-N .. t-1, so the current reading is compared with its own past.
The per-meter (day_of_week, hour) baseline is fitted on rows before
split_time only and never uses labels. Features are left unscaled.

Run from the project root:
    python -m src.features
"""

from pathlib import Path

import pandas as pd

IN_PATH = Path("data/processed/attacked.parquet")
OUT_PATH = Path("data/processed/features.parquet")
SPLIT_TIME = pd.Timestamp("2013-09-13 12:00:00")
EPS = 0.01  # kWh; keeps ratios bounded when the reference consumption is ~0

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
]


def _past_rolling(prev, groups, window, stat):
    """Rolling stat over the previous `window` hours per meter; NaN until full."""
    return prev.groupby(groups, observed=True).transform(
        lambda s: getattr(s.rolling(window, min_periods=window), stat)()
    )


def fit_baseline(df, split_time):
    """Median energy_kwh per (LCLid, day_of_week, hour) from rows before split_time."""
    train = df[df["timestamp"] < split_time]
    return (train.groupby(["LCLid", "day_of_week", "hour"], observed=True)["energy_kwh"]
            .median()
            .rename("baseline_median"))


def build_features(df, split_time=SPLIT_TIME):
    """Return a feature table with ID_COLUMNS + FEATURE_COLUMNS.

    Expects one row per meter per hour with no gaps. Rows in the first 168
    hours of each meter (incomplete weekly window) are dropped.
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
    prev = out.groupby(groups, observed=True)["energy_kwh"].shift(1)
    out["rolling24_mean"] = _past_rolling(prev, groups, 24, "mean")
    out["rolling24_std"] = _past_rolling(prev, groups, 24, "std")
    out["rolling168_mean"] = _past_rolling(prev, groups, 168, "mean")
    out["rate_of_change"] = out["energy_kwh"] - prev
    out["ratio_to_rolling24"] = out["energy_kwh"] / (out["rolling24_mean"] + EPS)

    out = out.join(fit_baseline(out, split_time), on=["LCLid", "day_of_week", "hour"])
    out["baseline_deviation"] = out["energy_kwh"] - out["baseline_median"]
    out["baseline_ratio"] = out["energy_kwh"] / (out["baseline_median"] + EPS)

    out["split"] = (out["timestamp"] >= split_time).map({False: "train", True: "test"})
    out = out[out["rolling168_mean"].notna()].reset_index(drop=True)
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
