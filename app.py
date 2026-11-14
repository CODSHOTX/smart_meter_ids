"""Streamlit dashboard for the smart-meter attack detection prototype.

Run from the project root:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import (
    MODEL_OPTIONS,
    decision_score,
    group_alert_events,
    persistence_alert,
)
from src.features import SPLIT_TIME

PRED_PATH = Path("data/processed/predictions.parquet")
ATTACKED_PATH = Path("data/processed/attacked.parquet")
ATTACKS_PATH = Path("data/processed/attacks.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
METRICS_PATH = Path("reports/metrics.csv")
THRESHOLDS_PATH = Path("models/thresholds.json")

REPLAY_STEP_H = 6


@st.cache_data(show_spinner="Loading data…")
def load_data():
    """Load processed tables once and join the columns the dashboard needs."""
    preds = pd.read_parquet(PRED_PATH)
    attacked = pd.read_parquet(
        ATTACKED_PATH, columns=["LCLid", "timestamp", "energy_kwh", "energy_kwh_original"]
    )
    features = pd.read_parquet(
        FEATURES_PATH, columns=["LCLid", "timestamp", "energy_kwh", "baseline_median"]
    )
    attacks = pd.read_parquet(ATTACKS_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    thresholds = json.loads(THRESHOLDS_PATH.read_text())

    df = preds.merge(
        attacked[["LCLid", "timestamp", "energy_kwh", "energy_kwh_original"]],
        on=["LCLid", "timestamp"],
        how="left",
    )
    df = df.merge(
        features[["LCLid", "timestamp", "baseline_median"]],
        on=["LCLid", "timestamp"],
        how="left",
    )
    df = df.sort_values(["LCLid", "timestamp"]).reset_index(drop=True)
    return df, attacks, metrics, thresholds


def sidebar_controls(df):
    """Render sidebar widgets and return the current selection."""
    st.sidebar.header("Controls")
    model_name = st.sidebar.selectbox(
        "Model", list(MODEL_OPTIONS), index=list(MODEL_OPTIONS).index("Random Forest")
    )
    threshold = st.sidebar.radio(
        "Threshold variant",
        ["far2", "f1"],
        index=0,
        format_func=lambda x: {
            "far2": "far2 — ~2% false-alarm rate on validation",
            "f1": "f1 — maximises F1 on validation",
        }[x],
    )
    k = st.sidebar.select_slider("Persistence k", options=[1, 2, 3], value=2)

    test = df[df["split"] == "test"]
    meters = sorted(test["LCLid"].astype(str).unique())
    meter = st.sidebar.selectbox("Meter", meters, index=0)

    t_min = test["timestamp"].min().to_pydatetime()
    t_max = test["timestamp"].max().to_pydatetime()
    date_range = st.sidebar.slider(
        "Date range",
        min_value=t_min,
        max_value=t_max,
        value=(t_min, t_max),
        format="YYYY-MM-DD",
    )
    show_truth = st.sidebar.checkbox("Show ground truth", value=False)
    return model_name, threshold, k, meter, date_range, show_truth


def summary_cards(metrics, model_key, threshold, k):
    """Top-row KPI cards from reports/metrics.csv."""
    row = metrics[
        (metrics["section"] == "row_metrics")
        & (metrics["model"] == model_key)
        & (metrics["threshold"] == threshold)
        & (metrics["k"] == k)
    ]
    lat = metrics[
        (metrics["section"] == "latency")
        & (metrics["model"] == model_key)
        & (metrics["threshold"] == threshold)
        & (metrics["k"] == k)
    ]
    if row.empty or lat.empty:
        st.warning("No metrics found for this selection in reports/metrics.csv.")
        return

    r, l = row.iloc[0], lat.iloc[0]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Precision", f"{r['precision']:.3f}")
    c2.metric("Recall", f"{r['recall']:.3f}")
    c3.metric("F1", f"{r['f1']:.3f}")
    c4.metric("False alerts / meter / day", f"{r['false_alerts_per_meter_day']:.2f}")
    c5.metric("Windows detected", f"{int(l['n_detected'])} / {int(l['n_windows'])}")
    c6.metric("Median latency (h)", f"{l['latency_median_h']:.1f}")


def meter_chart(meter_df, alert, attacks, meter, show_truth):
    """Plotly line chart of consumption with alert markers and optional truth."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=meter_df["timestamp"], y=meter_df["energy_kwh"],
        mode="lines", name="Reported energy",
        line=dict(color="#1f4e79", width=1.5),
    ))
    flagged = meter_df[alert.to_numpy() == 1]
    if not flagged.empty:
        fig.add_trace(go.Scatter(
            x=flagged["timestamp"], y=flagged["energy_kwh"],
            mode="markers", name="Alert",
            marker=dict(color="#c0392b", size=7, symbol="circle"),
        ))
    if show_truth:
        fig.add_trace(go.Scatter(
            x=meter_df["timestamp"], y=meter_df["energy_kwh_original"],
            mode="lines", name="True energy",
            line=dict(color="#7f8c8d", width=1.2, dash="dash"),
        ))
        windows = attacks[
            (attacks["LCLid"].astype(str) == meter)
            & (attacks["end_time"] >= meter_df["timestamp"].min())
            & (attacks["start_time"] <= meter_df["timestamp"].max())
        ]
        for i, (_, w) in enumerate(windows.iterrows()):
            fig.add_vrect(
                x0=w["start_time"], x1=w["end_time"],
                fillcolor="rgba(192, 57, 43, 0.12)", line_width=0,
                annotation_text=str(w["attack_type"]) if i < 8 else "",
                annotation_position="top left",
                annotation_font_size=10,
            )
    fig.update_layout(
        height=380, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title=None, yaxis_title="kWh / hour",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch')


def fleet_table(events):
    """Meters ranked by number of alert events in the selected range."""
    if events.empty:
        st.info("No alert events in this range.")
        return
    fleet = (events.groupby("LCLid", observed=True)
             .agg(n_events=("start", "size"),
                  total_alert_hours=("duration_h", "sum"),
                  last_alert=("end", "max"),
                  top_suspect=("suspected_type",
                               lambda s: s.value_counts().index[0]))
             .sort_values(["n_events", "total_alert_hours"], ascending=False)
             .reset_index())
    st.dataframe(fleet, width='stretch', hide_index=True)


def replay_panel(view, alert, score, baseline_col, attacks, show_truth, meter):
    """Time-slider replay that reveals data and new alerts up to the cursor."""
    st.subheader("Replay mode")
    times = pd.date_range(view["timestamp"].min(), view["timestamp"].max(),
                          freq=f"{REPLAY_STEP_H}h")
    if "replay_idx" not in st.session_state:
        st.session_state.replay_idx = 0

    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("Play ▶", width='stretch'):
            st.session_state.replay_idx = min(
                st.session_state.replay_idx + 1, len(times) - 1
            )
    with c2:
        if st.button("Reset", width='stretch'):
            st.session_state.replay_idx = 0

    idx = st.slider(
        "Replay cursor",
        min_value=0,
        max_value=max(len(times) - 1, 0),
        value=st.session_state.replay_idx,
        format="%d",
    )
    st.session_state.replay_idx = idx
    cursor = times[idx]
    st.caption(f"Showing data up to **{cursor}**  ·  step = {REPLAY_STEP_H} h")

    visible = view[view["timestamp"] <= cursor].copy()
    vis_alert = alert.loc[visible.index]
    meter_vis = visible[visible["LCLid"].astype(str) == meter]
    meter_alert = vis_alert.loc[meter_vis.index]

    left, right = st.columns([2, 1])
    with left:
        meter_chart(meter_vis, meter_alert, attacks, meter, show_truth)
    with right:
        st.markdown("**Live alert feed**")
        step_start = cursor - pd.Timedelta(hours=REPLAY_STEP_H)
        new = visible[(vis_alert.to_numpy() == 1)
                      & (visible["timestamp"] > step_start)
                      & (visible["timestamp"] <= cursor)]
        if new.empty:
            st.write("No new alerts in this step.")
        else:
            tmp = new.assign(score=score.loc[new.index].to_numpy())
            feed = (tmp.groupby("LCLid", observed=True)
                    .agg(hours=("timestamp", "count"),
                         latest=("timestamp", "max"),
                         peak_score=("score", "max"))
                    .reset_index()
                    .sort_values("latest", ascending=False))
            feed["peak_score"] = feed["peak_score"].round(3)
            st.dataframe(feed, width='stretch', hide_index=True)

        events = group_alert_events(visible, vis_alert, score.loc[visible.index],
                                    baseline_col)
        st.caption(f"{len(events)} alert events so far across the fleet.")


def main():
    st.set_page_config(page_title="Smart Meter Attack Detection", layout="wide")
    st.title("Smart Meter Attack Detection")
    st.caption(
        f"TEST period from {SPLIT_TIME}. Metrics are precomputed on the full test set; "
        "charts and tables below follow the sidebar filters."
    )

    df, attacks, metrics, thresholds = load_data()
    model_name, threshold, k, meter, date_range, show_truth = sidebar_controls(df)
    cfg = MODEL_OPTIONS[model_name]
    model_key = cfg["key"]
    pred_col = cfg["pred"][threshold]
    thr_key = f"threshold_{threshold}"
    thr_val = thresholds[model_key][thr_key]
    st.sidebar.caption(
        f"Decision rule: `{thresholds[model_key]['rule']}`  \n"
        f"Chosen {threshold} threshold = **{thr_val:.4f}**"
    )

    summary_cards(metrics, model_key, threshold, k)

    # Scope: TEST rows in the selected date range
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    view = df[
        (df["split"] == "test")
        & (df["timestamp"] >= start)
        & (df["timestamp"] <= end)
    ].copy()
    if view.empty:
        st.warning("No rows in the selected range.")
        return

    flags = view[pred_col]
    alert = persistence_alert(flags, view["LCLid"], k)
    score = pd.Series(decision_score(view, cfg), index=view.index)
    events = group_alert_events(view, alert, score)

    # --- Meter detail ---
    st.subheader(f"Meter {meter}")
    meter_df = view[view["LCLid"].astype(str) == meter]
    meter_alert = alert.loc[meter_df.index]
    meter_chart(meter_df, meter_alert, attacks, meter, show_truth)

    # --- Suspicious events ---
    st.subheader("Suspicious events")
    st.info(
        "Suspected attack type is a **heuristic** from the reported readings and the "
        "fixed weekday-hour baseline only. It does **not** use ground-truth labels "
        "(`attack_type` / `label`) and can be wrong."
    )
    meter_events = events[events["LCLid"].astype(str) == meter]
    show = events if st.checkbox("Show events for all meters", value=False) else meter_events
    if show.empty:
        st.write("No alert events for this selection.")
    else:
        display = show.copy()
        display["peak_score"] = display["peak_score"].round(3)
        st.dataframe(display, width='stretch', hide_index=True)

    # --- Fleet view ---
    st.subheader("Fleet view")
    st.caption("Meters ranked by number of alert events in the selected range.")
    fleet_table(events)

    # --- Replay ---
    replay_panel(view, alert, score, "baseline_median", attacks, show_truth, meter)


if __name__ == "__main__":
    main()
