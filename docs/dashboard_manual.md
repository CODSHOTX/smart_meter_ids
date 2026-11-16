# Dashboard user manual

Smart Meter Attack Detection — Streamlit app (`app.py`)

---

## 1. What the dashboard is

A **read-only** interface over results the pipeline already produced. It does **not** retrain models or inject attacks. You choose a detector and filters; it shows alerts, charts, and precomputed metrics for the **test period** (from **2013-09-13 12:00** onward).

---

## 2. Starting the app

From the project root, with the virtual environment active and the pipeline already run:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

On macOS / Linux:

```bash
source .venv/bin/activate   # or: source venv/bin/activate
streamlit run app.py
```

Open the URL Streamlit prints (usually **http://localhost:8501**).

**Needed files** (created by the pipeline):

- `data/processed/predictions.parquet`
- `data/processed/attacked.parquet`
- `data/processed/features.parquet`
- `data/processed/attacks.parquet`
- `reports/metrics.csv`
- `models/thresholds.json`

If any are missing, run the pipeline first (see the project README, §6).

---

## 3. Layout at a glance

| Area | Purpose |
|------|---------|
| **Sidebar** | Model, threshold, persistence *k*, meter, date range, ground truth |
| **Summary cards** | Test-set KPIs for the current model / threshold / *k* |
| **Meter chart** | One meter’s consumption + red alert points |
| **Suspicious events** | Alert runs grouped into events + heuristic type |
| **Fleet view** | Meters ranked by how many alert events they have |
| **Replay mode** | Step through time as if monitoring live |

Defaults: **Random Forest**, **far2**, **k = 2**.

---

## 4. Sidebar controls

### Model

- **Random Forest** — supervised; usually strongest (recommended)
- **Isolation Forest** — unsupervised; good on zeroing/spikes, weak on gradual theft
- **Z-score** — simple baseline vs fixed weekday–hour pattern

### Threshold variant

- **far2** — tuned for ~2% false-alarm rate on validation (default; better for demos)
- **f1** — maximises F1 on validation (often more sensitive, more noise)

The sidebar caption shows the decision rule and numeric threshold from `models/thresholds.json`.

### Persistence *k* (1, 2, or 3)

An **alert** fires at hour *t* only if the model flagged **all of the last *k* hours** (including *t*).

- *k* = 1 → every flagged hour is an alert
- *k* = 2 or 3 → fewer false alarms, slightly later detection

Recommended demo setting: **k = 2**.

### Meter

Which household (`LCLid`) to show in the main chart and (by default) in the events table.

### Date range

Limits charts, events, fleet ranking, and replay to that window inside the test period. Summary cards still reflect the **full** test evaluation in `metrics.csv`.

### Show ground truth

When checked:

- dashed line = true energy (`energy_kwh_original`)
- shaded bands = real injected attack windows

Use this for demos and evaluation. Leave it **off** to mimic an operator who does not know the labels.

---

## 5. Summary cards

| Card | Meaning |
|------|---------|
| **Precision** | Of alerted hours, fraction that were truly attacked |
| **Recall** | Of attacked hours, fraction that were alerted |
| **F1** | Balance of precision and recall |
| **False alerts / meter / day** | Average false-alert hours per meter per day |
| **Windows detected** | Attack windows with ≥1 alert / total test windows |
| **Median latency (h)** | Median hours from first changed reading to first alert (detected windows only) |

These come from `reports/metrics.csv` and do **not** change when you narrow the date range.

---

## 6. Meter chart

- **Solid blue line** — reported energy (what the detector sees)
- **Red dots** — hours that raised an alert under the current model / threshold / *k*
- Hover for time and kWh

**Tip:** Pick a busy meter from Fleet view, then narrow the date range around its alerts so the chart is readable.

---

## 7. Suspicious events

Consecutive alert hours on one meter are merged into one **event**:

| Column | Meaning |
|--------|---------|
| `LCLid` | Meter ID |
| `start` / `end` | First and last alert hour |
| `duration_h` | Length in hours |
| `peak_score` | Highest model score in the event |
| `suspected_type` | Heuristic guess (see below) |

By default the table is for the **selected meter**. Tick **Show events for all meters** for the whole fleet in the date range.

**Suspected types** (from readings + baseline only — **not** from labels):

| Heuristic label | Rough rule |
|-----------------|------------|
| possible meter outage/zeroing | Many hours near 0 kWh |
| possible spike / data injection | Short burst or large jump vs baseline |
| possible energy theft (reduced readings) | Sustained drop vs baseline |

Treat these as **hints**. They can be wrong.

---

## 8. Fleet view

All meters in the date range, sorted by:

1. number of alert events
2. total alert hours

Also shows last alert time and the most common suspected type. Use this to decide which meter to open next.

---

## 9. Replay mode

Simulates real-time monitoring over the **selected date range**:

1. Drag **Replay cursor**, or click **Play ▶** (advances **6 hours**), or **Reset**
2. Left chart — selected meter, only data **up to** the cursor
3. **Live alert feed** — meters with new alerts in that 6-hour step
4. Caption — how many alert events have accumulated so far

Ground truth still follows the sidebar checkbox.

---

## 10. Suggested walkthrough (demo)

1. Leave **Random Forest · far2 · k = 2**.
2. Turn **Show ground truth** on.
3. In **Fleet view**, open a meter with many events.
4. Select that meter; shrink **Date range** around the alerts.
5. Check that red dots land inside shaded attack windows.
6. Read **Suspicious events** and compare the heuristic to the shaded labels.
7. Turn ground truth **off**, then use **Replay** to step through the same period.

**Compare models:** same meter and dates, switch Model / Threshold / *k* and watch how red dots and fleet rankings change.

---

## 11. Common issues

| Problem | What to try |
|---------|-------------|
| App won’t start | Activate venv; run from project root; `pip install -r requirements.txt` |
| Empty charts / warnings | Run full pipeline so processed files and `reports/metrics.csv` exist |
| No alerts on a meter | Many meters are never attacked in the test period; use Fleet view |
| Too many red dots | Raise *k* (e.g. 3) or use **far2** |
| Too few alerts | Lower *k* to 1 or try **f1** |
| Page slow after filter change | Normal on first load; data is cached after that |

---

## 12. What the dashboard does *not* do

- Retrain or retune thresholds
- Change attacks or features
- Show the training period (TEST only)
- Guarantee that “suspected type” matches the true attack
