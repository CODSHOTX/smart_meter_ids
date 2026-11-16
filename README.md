# Cyberattack Detection System for Smart Electricity Metering Networks

An anomaly-based prototype that flags abnormal smart-meter consumption behaviour (energy theft, meter outage or zeroing, data spikes and gradual tampering) and displays suspicious events in an interactive dashboard.

**Author:** [Your name] | **Supervisor:** [Supervisor name] | **Institution:** [Institution] | **Year:** 2026

---

## 1. Overview

Smart metering improves energy management but introduces cyber risks such as meter tampering and unauthorised data manipulation. This project develops and evaluates a detection system that learns what normal consumption looks like and flags readings that deviate from it.

**Aim:** To develop and evaluate an anomaly-based cyberattack detection system for smart electricity metering networks.

**Objectives and where they are met**

| # | Objective | Where |
|---|-----------|-------|
| 1 | Develop an anomaly detection model for abnormal smart-meter consumption patterns | `src/features.py`, `src/models.py` |
| 2 | Implement the model using a public smart-meter dataset with a simulated attack environment | `src/load_data.py`, `src/attacks.py` |
| 3 | Evaluate with accuracy, precision, recall, F1-score, false-alarm rate and detection latency | `src/evaluate.py`, `reports/` |

**Deliverable:** a prototype that flags abnormal smart-meter behaviour and displays suspicious events (`app.py`, Streamlit dashboard).

**Scope:** consumption-based attacks only. Network-traffic attacks are out of scope (see Limitations).

---

## 2. Method in brief

1. **Data.** The London Smart Meters dataset (half-hourly readings), resampled to hourly. 100 meters with complete data over all of 2013 (8,760 hours each) were selected from the first three block files.
2. **Attack simulation.** Real readings are treated as normal, and four attack types are injected as labelled windows of 6 to 72 hours:
   - **Scaling:** readings multiplied by 0.3 to 0.6 (energy theft)
   - **Zeroing:** readings set to 0 (meter outage or bypass)
   - **Spike:** isolated readings multiplied by 3 to 6 (data injection)
   - **Ramp-down:** readings reduced gradually from 1.0 to about 0.3 of normal (gradual tampering)

   About 3 to 6% of meter-hours are attacked. Attacked meters in the training period and the test period are **disjoint**, so the models cannot learn "this meter is attack-prone".
3. **Time-based split.** Training uses data before `2013-09-13 12:00`, and testing uses data after it. No random splitting. Inside the training period, the last 20% (by time) is a validation set used only for threshold selection.
4. **Features (past data only).** Calendar features, rolling statistics, rate of change, ratios to the meter's own baseline (a fixed training-period baseline and an adaptive 4-week baseline), and persistence features that describe how the last several hours behaved. Rolling windows never include the current or future hours.
5. **Leakage control.** A test replaces all post-split data with random numbers and rebuilds the features. Training features must stay identical. This test passes.
6. **Models.**
   - **Z-score baseline:** deviation from baseline divided by the meter's own normal variability.
   - **Isolation Forest (unsupervised):** 200 trees, trained on normal rows only.
   - **Random Forest (supervised):** 200 trees, depth 12, `class_weight='balanced'`.
7. **Thresholds** are chosen on the validation set only, in two ways: maximum F1 (`f1`) and a 2% false-alarm budget (`far2`).
8. **Persistence rule (k).** An alert is raised only if the reading was flagged for `k` consecutive hours (k = 1, 2 or 3), which trades detection speed for fewer false alarms.

---

## 3. Results (test period)

Models are compared at the same false-alarm budget (`far2` threshold). The test period contains 333 attack windows and about 4% attacked rows.

### Overall (far2, k = 1)

| Model | Precision | Recall | Attack windows detected | Median latency |
|-------|-----------|--------|-------------------------|----------------|
| Z-score | 0.048 | 0.064 | 39% | 3 h |
| Isolation Forest | 0.221 | 0.274 | 45% | 9 h |
| **Random Forest** | **0.549** | **0.599** | **90%** | 3 h |

### Recommended operating point: Random Forest, far2, k = 2

| Metric | Value |
|--------|-------|
| Precision | 0.648 |
| Recall (rows) | 0.523 |
| F1-score | 0.579 |
| False-alarm rate | 1.2% |
| False-flagged hours per meter per day | 0.28 |
| Attack windows detected | 80% |
| Median detection latency | 5 h |

### Random Forest by attack type (far2, k = 1)

| Attack type | Row recall | Windows detected | Median latency |
|-------------|-----------|------------------|----------------|
| Zeroing | 0.997 | 100% | 0 h |
| Spike | 0.560 | 89% | 4 h |
| Scaling | 0.458 | 87% | 6 h |
| Ramp-down | 0.271 | 81% | 19 h |

The recommended operating point is k=2, and its per-type breakdown shows a side effect that the overall 80% hides. Spikes are isolated hours, so requiring two consecutive flagged hours hurts them badly:

RF, far2	k=1	k=2	k=3
Spike windows detected	89%	56%	25%
Zeroing	100%	100%	100%
Scaling	87%	82%	77%
Ramp-down	81%	76%	71%

### Key findings

- Random Forest clearly outperforms the unsupervised models at equal false-alarm rates, and its validation-chosen threshold transferred to unseen months and unseen meters.
- Isolation Forest detects gross anomalies (zeroing and spikes) but almost never scaling or ramp-down. Subtle theft that resembles a quieter household needs labelled data or richer context.
- Ramp-down is detected late rather than missed, because it starts close to normal behaviour.
- Raising `k` cuts false alarms sharply but lowers detection and increases latency.

Full tables are in `reports/metrics.csv`, and figures (confusion matrices, ROC and precision-recall curves, recall and latency by attack type) are in `reports/`. Exact values may differ slightly if the pipeline is rerun with different seeds.

---

## 4. Project structure

```
smart_meter_ids/
├── data/
│   ├── halfhourly_dataset/      # raw London Smart Meters files (not included)
│   ├── informations_households.csv
│   └── processed/               # generated: hourly, attacked, features, predictions
├── src/
│   ├── load_data.py             # load, clean, resample to hourly
│   ├── attacks.py               # synthetic attack injection and labels
│   ├── features.py              # leakage-safe feature engineering
│   ├── models.py                # z-score, Isolation Forest, Random Forest
│   └── evaluate.py              # metrics, persistence rule, latency, plots
├── models/                      # saved models and chosen thresholds (generated)
├── reports/                     # metrics.csv and figures (generated)
├── notebooks/                   # exploration
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 5. Setup

**Requirements:** Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows (PowerShell: .\venv\Scripts\Activate.ps1)
source venv/bin/activate         # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

**Get the data.** Download the *Smart meters in London* dataset from Kaggle. Place the `halfhourly_dataset` folder and `informations_households.csv` inside `data/`. The raw data is not included in this repository.

---

## 6. Running the pipeline

Run the steps in order from the project root, with the virtual environment active:

```bash
python src/load_data.py     # -> data/processed/hourly.parquet
python src/attacks.py       # -> attacked.parquet, attacks.parquet
python src/features.py      # -> features.parquet (runs the leakage test)
python src/models.py        # -> predictions.parquet, models/
python src/evaluate.py      # -> reports/metrics.csv and figures
```

Then launch the dashboard:

```bash
streamlit run app.py
```

All random processes use fixed seeds (`42`) for reproducibility.

---

## 7. Dashboard

Full step-by-step usage guide: **[docs/dashboard_manual.md](docs/dashboard_manual.md)**.

The Streamlit dashboard reads the generated files and provides:

- **Controls:** model (Random Forest, Isolation Forest, Z-score), threshold variant (`f1` or `far2`), persistence `k`, meter, and date range. The default is Random Forest, far2, k = 2.
- **Summary cards:** precision, recall, F1, false alerts per meter per day, windows detected, and median latency.
- **Consumption chart:** a meter's readings with alert hours marked in red, with an optional ground-truth overlay (original readings and true attack windows).
- **Suspicious events table:** consecutive alert hours grouped into events with start, end, duration, and peak score. The suspected attack type is a heuristic derived only from observable features, never from the ground-truth label.
- **Fleet view:** all meters ranked by number of alert events.
- **Replay mode:** a play button and time slider that step through the test period to simulate real-time monitoring.

---

## 8. Limitations

- Attacks are **synthetic** and come from the same generator used for training, so performance on real attacks may differ. Validation on a labelled real-attack dataset (for example the SGCC electricity theft dataset) is future work.
- Only **consumption-based** attacks are covered. Network-traffic attacks (DoS, replay, command injection) are not addressed.
- One year of data from a single UK dataset, so results may not transfer to other regions or load patterns.
- **Seasonal drift:** the fixed baseline is built from January to mid-September, so the test period (autumn and winter) produces more false alarms for the unsupervised models.
- Attacks are denser in the training period than in the test period, which is a deliberate design choice to keep the overall attacked share at 3 to 6%.
- The reported false-alert rate counts flagged **hours** per meter per day, not distinct alert episodes.
- Per-attack-type results rest on 75 to 96 test windows each, so percentages carry roughly 8 to 10 points of uncertainty.

## 9. Future work

- Adaptive or periodically retrained baselines to handle seasonal drift.
- Network-level features and datasets (for example ToN-IoT or CICIDS) for communication attacks.
- Evaluation on real labelled theft data.
- A hybrid model that feeds Isolation Forest scores into the Random Forest.

---

## 10. Tools and acknowledgements

Python, pandas, NumPy, scikit-learn, Plotly, Matplotlib, Streamlit, PyArrow. Data: *Smart meters in London* (UK Power Networks via the London Datastore, distributed on Kaggle). Parts of the code were developed with AI assistance (Cursor and Claude) and reviewed and tested by the author.