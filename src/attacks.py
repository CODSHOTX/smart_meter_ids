"""Inject synthetic consumption attacks into hourly smart-meter data.

Attack types:
    scaling   - readings multiplied by one random factor in [0.3, 0.6] (theft)
    zeroing   - readings set to 0
    spike     - a random subset of readings multiplied by 3-6x each
    ramp_down - factor falling gradually from 1.0 to 0.3 across the window

Windows never cross the train/test boundary (first TRAIN_FRAC of the timeline)
and windows on the same meter never overlap. Only readings that were actually
tampered with get label 1, so for spike windows the unspiked hours stay normal.

Run from the project root:
    python -m src.attacks
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/hourly.parquet")
OUT_PATH = Path("data/processed/attacked.parquet")
ATTACKS_PATH = Path("data/processed/attacks.parquet")

ATTACK_TYPES = ["scaling", "zeroing", "spike", "ramp_down"]
TRAIN_FRAC = 0.7
METER_FRAC = 0.3
MIN_HOURS, MAX_HOURS = 6, 72
WINDOWS_PER_METER = (25, 45)
MIN_GAP_HOURS = 24
SPIKE_PROB = 0.3


def split_time(df, train_frac=TRAIN_FRAC):
    """Return the first timestamp of the test period (train is timestamp < split)."""
    grid = np.sort(df["timestamp"].unique())
    return pd.Timestamp(grid[int(len(grid) * train_frac)])


def _place_windows(rng, seg_start, seg_len, n):
    """Return n (start, length) windows inside [seg_start, seg_start + seg_len).

    Windows are in time order and separated by at least MIN_GAP_HOURS, so they
    can neither overlap nor touch.
    """
    lengths = rng.integers(MIN_HOURS, MAX_HOURS, size=n, endpoint=True)
    slack = seg_len - lengths.sum() - MIN_GAP_HOURS * (n - 1)
    if slack < 0:
        raise ValueError(f"{n} windows do not fit in a {seg_len}-hour segment")
    offsets = np.sort(rng.integers(0, slack, size=n, endpoint=True))
    preceding = np.concatenate([[0], np.cumsum(lengths[:-1] + MIN_GAP_HOURS)])
    starts = seg_start + offsets + preceding
    return list(zip(starts.tolist(), lengths.tolist()))


def _tamper(rng, values, attack_type):
    """Apply one attack to a window of readings.

    Returns the new values and a boolean mask of the readings that were tampered.
    """
    n = len(values)
    hit = np.ones(n, dtype=bool)
    if attack_type == "scaling":
        new = values * rng.uniform(0.3, 0.6)
    elif attack_type == "zeroing":
        new = np.zeros(n)
    elif attack_type == "spike":
        hit = rng.random(n) < SPIKE_PROB
        if not hit.any():
            hit[rng.integers(n)] = True
        new = values.copy()
        new[hit] *= rng.uniform(3.0, 6.0, size=hit.sum())
    elif attack_type == "ramp_down":
        new = values * np.linspace(1.0, 0.3, n + 1)[1:]
    else:
        raise ValueError(f"unknown attack type: {attack_type}")
    return new, hit


def inject_attacks(df, seed=42):
    """Inject attack windows into a long-format hourly DataFrame.

    Expects columns LCLid, timestamp, energy_kwh with one row per meter per
    hour on a common time grid. Returns (attacked, attacks):

    attacked: input rows sorted by LCLid and timestamp, with energy_kwh holding
        the attacked values and extra columns energy_kwh_original, label (0/1),
        attack_type ('none' if not attacked) and attack_id (-1 if not attacked).
    attacks: one row per window with attack_id, LCLid, attack_type, start_time
        and end_time, where start/end are the first and last tampered hours
        (inclusive).
    """
    rng = np.random.default_rng(seed)
    out = df.sort_values(["LCLid", "timestamp"]).reset_index(drop=True)
    grid = np.sort(out["timestamp"].unique())
    n_hours = len(grid)
    split_idx = int(n_hours * TRAIN_FRAC)

    rows_by_meter = out.groupby("LCLid", observed=True).indices
    if any(len(rows) != n_hours for rows in rows_by_meter.values()):
        raise ValueError("every meter needs exactly one row per hour on a common grid")

    energy = out["energy_kwh"].to_numpy(dtype=float, copy=True)
    label = np.zeros(len(out), dtype="int8")
    attack_type = np.full(len(out), "none", dtype=object)
    attack_id = np.full(len(out), -1, dtype="int32")
    records = []

    meters = sorted(rows_by_meter)
    n_attacked = round(len(meters) * METER_FRAC)
    for meter in sorted(rng.choice(meters, size=n_attacked, replace=False)):
        rows = rows_by_meter[meter]
        n = int(rng.integers(*WINDOWS_PER_METER, endpoint=True))
        n_train = round(n * TRAIN_FRAC)
        windows = (_place_windows(rng, 0, split_idx, n_train)
                   + _place_windows(rng, split_idx, n_hours - split_idx, n - n_train))
        for start, length in windows:
            kind = str(rng.choice(ATTACK_TYPES))
            idx = rows[start:start + length]
            energy[idx], hit = _tamper(rng, energy[idx], kind)
            tampered = idx[hit]
            aid = len(records)
            label[tampered] = 1
            attack_type[tampered] = kind
            attack_id[tampered] = aid
            hours = np.flatnonzero(hit) + start
            records.append({
                "attack_id": aid,
                "LCLid": meter,
                "attack_type": kind,
                "start_time": pd.Timestamp(grid[hours[0]]),
                "end_time": pd.Timestamp(grid[hours[-1]]),
            })

    out["energy_kwh_original"] = out["energy_kwh"]
    out["energy_kwh"] = energy
    out["label"] = label
    out["attack_type"] = pd.Categorical(attack_type, categories=["none"] + ATTACK_TYPES)
    out["attack_id"] = attack_id
    attacks = pd.DataFrame(records)
    attacks["attack_type"] = pd.Categorical(attacks["attack_type"], categories=ATTACK_TYPES)
    return out, attacks


def main():
    """Load hourly data, inject attacks, save both tables and print a summary."""
    df = pd.read_parquet(IN_PATH)
    attacked, attacks = inject_attacks(df, seed=42)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    attacked.to_parquet(OUT_PATH, index=False)
    attacks.to_parquet(ATTACKS_PATH, index=False)

    split = split_time(df)
    in_test = attacks["start_time"] >= split
    print(f"Attacked meters: {attacks['LCLid'].nunique()} of {df['LCLid'].nunique()}")
    print(f"Train/test split at {split}: {(~in_test).sum()} train windows, "
          f"{in_test.sum()} test windows")
    print("\nAttack windows per type:")
    print(attacks["attack_type"].value_counts().sort_index().to_string())
    print(f"\nAttacked rows: {attacked['label'].sum():,} of {len(attacked):,} "
          f"({attacked['label'].mean():.2%})")
    print("\nFirst 5 attacks:")
    print(attacks.head().to_string(index=False))


if __name__ == "__main__":
    main()
