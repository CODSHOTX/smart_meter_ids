"""Load London Smart Meters half-hourly blocks and build a clean hourly subset.

Pipeline: read blocks -> clean values/timestamps -> restrict to a common date
range -> pick the meters with the fewest missing readings -> resample to hourly
sums -> fill short gaps -> save as long-format parquet.

Run from the project root:
    python -m src.load_data
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/halfhourly_dataset/halfhourly_dataset")
OUT_PATH = Path("data/processed/hourly.parquet")
BLOCKS = [f"block_{i}.csv" for i in range(4)]

START = pd.Timestamp("2013-01-01")
END = pd.Timestamp("2014-01-01")  # exclusive
N_METERS = 100
MAX_GAP_HOURS = 3


def read_blocks(raw_dir=RAW_DIR, blocks=BLOCKS):
    """Read half-hourly CSV blocks into one DataFrame.

    Returns columns LCLid (category), tstp (datetime64), energy_kwh (float),
    with 'Null' readings converted to NaN and duplicate rows dropped.
    """
    frames = []
    for name in blocks:
        df = pd.read_csv(raw_dir / name, dtype=str)
        df.columns = ["LCLid", "tstp", "energy_kwh"]
        energy = df["energy_kwh"].str.strip().replace("Null", np.nan)
        df["energy_kwh"] = pd.to_numeric(energy, errors="raise")
        df["tstp"] = pd.to_datetime(df["tstp"], format="ISO8601")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["LCLid", "tstp"])
    df["LCLid"] = df["LCLid"].astype("category")
    return df


def select_meters(df, start=START, end=END, n=N_METERS):
    """Return the n meter IDs with the fewest missing half-hours in [start, end).

    A half-hour counts as missing if the row is absent or its value is NaN.
    Ties are broken by meter ID so the selection is deterministic.
    """
    expected = len(pd.date_range(start, end, freq="30min", inclusive="left"))
    window = df[(df["tstp"] >= start) & (df["tstp"] < end)]
    valid = window.groupby("LCLid", observed=True)["energy_kwh"].count()
    missing = (expected - valid).rename("missing").reset_index()
    missing = missing.sort_values(["missing", "LCLid"]).head(n)
    return missing["LCLid"].astype(str).tolist(), missing


def fill_short_gaps(wide, max_gap=MAX_GAP_HOURS):
    """Linearly interpolate NaN runs of at most max_gap steps, per column.

    Longer gaps and gaps at the series edges are left as NaN.
    """
    filled = wide.interpolate(method="linear", limit_area="inside")
    is_nan = wide.isna()
    run_len = is_nan.apply(lambda s: s.groupby((~s).cumsum()).transform("sum"))
    return filled.mask(is_nan & (run_len > max_gap))


def to_hourly(df, meters, start=START, end=END, max_gap=MAX_GAP_HOURS):
    """Resample selected meters to hourly kWh sums and fill short gaps.

    An hour is NaN unless both of its half-hours are present, so missing data
    is never mistaken for low consumption. Returns long format with columns
    LCLid, timestamp, energy_kwh.
    """
    sub = df[df["LCLid"].isin(meters) & (df["tstp"] >= start) & (df["tstp"] < end)]
    wide = sub.pivot(index="tstp", columns="LCLid", values="energy_kwh")
    wide.columns = wide.columns.astype(str)
    grid = pd.date_range(start, end, freq="30min", inclusive="left")
    wide = wide.reindex(index=grid, columns=meters)
    hourly = wide.resample("h").sum(min_count=2)
    hourly = fill_short_gaps(hourly, max_gap)
    hourly.index.name = "timestamp"
    long = hourly.reset_index().melt(
        id_vars="timestamp", var_name="LCLid", value_name="energy_kwh"
    )
    long["LCLid"] = long["LCLid"].astype("category")
    return long[["LCLid", "timestamp", "energy_kwh"]]


def build_hourly(out_path=OUT_PATH):
    """Run the full pipeline, save the parquet file and return the DataFrame."""
    raw = read_blocks()
    meters, missing = select_meters(raw)
    print(f"Raw rows: {len(raw):,}, meters: {raw['LCLid'].nunique()}, "
          f"Null readings: {raw['energy_kwh'].isna().sum():,}")
    print(f"Missing half-hours among selected meters: "
          f"min={missing['missing'].min()}, max={missing['missing'].max()}")
    hourly = to_hourly(raw, meters)
    print(f"Hourly NaN after filling short gaps: {hourly['energy_kwh'].isna().sum()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out_path, index=False)
    return hourly


if __name__ == "__main__":
    df = build_hourly()
    print(df.head())
    print(df.shape)
