from __future__ import annotations

from datetime import date
from io import BytesIO
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Microcredit Forecast & Staff Dashboard", page_icon="📊", layout="wide")

REQUIRED_COLUMNS = ["Loan Amount", "Disburse Date", "Loan Product", "Branch", "RO"]
OPTIONAL_NUMERIC = ["Collected", "Collected Capital", "Collected Profit", "Outstanding Balance", "Default Amount", "Past Due Count"]

@st.cache_data(show_spinner=False)
def load_report(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(file_bytes))
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    # Keep only the columns needed for aggregated reporting. Customer names, phone numbers,
    # customer IDs and loan reference numbers are intentionally excluded from the app dataset.
    keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_NUMERIC if c in raw.columns]
    df = raw[keep].copy()
    df["Disburse Date"] = pd.to_datetime(df["Disburse Date"], errors="coerce")
    for col in ["Loan Amount"] + [c for c in OPTIONAL_NUMERIC if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Loan Product", "Branch", "RO"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df = df.dropna(subset=["Disburse Date", "Loan Amount"])
    df = df[df["Loan Amount"] >= 0]
    return df



@st.cache_data(show_spinner=False)
def load_audit_log(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load only audit fields needed for aggregated monitoring."""
    if file_name.lower().endswith(".csv"):
        raw = pd.read_csv(BytesIO(file_bytes))
    else:
        raw = pd.read_excel(BytesIO(file_bytes))

    required = ["Who", "Entity Type", "Created At"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError("Missing required audit column(s): " + ", ".join(missing))

    # Do not carry free-text descriptions, emails, IP addresses, user agents,
    # or personal names into the dashboard's working table.
    keep = ["Who", "Entity Type", "Created At"]
    if "Where" in raw.columns:
        keep.append("Where")
    audit = raw[keep].copy()

    audit["Created At"] = pd.to_datetime(audit["Created At"], errors="coerce")
    audit = audit.dropna(subset=["Created At"])
    for col in ["Who", "Entity Type"] + (["Where"] if "Where" in audit.columns else []):
        audit[col] = (
            audit[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        )
    return audit


def render_audit_dashboard(audit_df: pd.DataFrame) -> None:
    """Render aggregate audit monitoring without exposing raw transaction details."""
    st.subheader("Audit Trail & Activity Monitoring")
    st.caption(
        "Aggregated activity only. Raw event descriptions, personal names, email addresses, "
        "IP addresses and browser identifiers are not displayed."
    )

    if audit_df.empty:
        st.warning("No valid audit events were found after parsing event timestamps.")
        return

    min_date = audit_df["Created At"].min().date()
    max_date = audit_df["Created At"].max().date()
    date_range = st.date_input(
        "Audit event date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="audit_date_range",
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_audit = audit_df[
            audit_df["Created At"].dt.date.between(start_date, end_date)
        ].copy()
    else:
        filtered_audit = audit_df.iloc[0:0].copy()

    entity_values = sorted(filtered_audit["Entity Type"].unique().tolist())
    selected_entities = st.multiselect(
        "Activity category",
        entity_values,
        default=entity_values,
        key="audit_entity_types",
    )
    filtered_audit = filtered_audit[
        filtered_audit["Entity Type"].isin(selected_entities)
    ].copy()

    filtered_audit["_Actor class"] = np.where(
        filtered_audit["Who"].str.strip().str.lower().eq("system"),
        "Explicit system",
        "Other / unidentified actor",
    )
    actor_classes = ["Explicit system", "Other / unidentified actor"]
    available_actor_classes = [
        label for label in actor_classes
        if label in filtered_audit["_Actor class"].unique()
    ]
    selected_actor_classes = st.multiselect(
        "Actor class",
        available_actor_classes,
        default=available_actor_classes,
        key="audit_actor_classes",
    )
    filtered_audit = filtered_audit[
        filtered_audit["_Actor class"].isin(selected_actor_classes)
    ].copy()

    if filtered_audit.empty:
        st.info("No audit events match the selected filters.")
        return

    system_mask = filtered_audit["_Actor class"].eq("Explicit system")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Audit events", f"{len(filtered_audit):,}")
    k2.metric("Activity categories", f"{filtered_audit['Entity Type'].nunique():,}")
    k3.metric("Distinct actors", f"{filtered_audit['Who'].nunique():,}")
    k4.metric("Explicit system events", f"{int(system_mask.sum()):,}")

    left, right = st.columns(2)
    with left:
        category_counts = (
            filtered_audit["Entity Type"].value_counts()
            .rename_axis("Activity category")
            .reset_index(name="Events")
        )
        fig_category = px.bar(
            category_counts,
            x="Activity category",
            y="Events",
            title="Events by activity category",
        )
        fig_category.update_layout(height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig_category, use_container_width=True)

    with right:
        daily = (
            filtered_audit.set_index("Created At")
            .resample("D")
            .size()
            .rename("Events")
            .reset_index()
        )
        fig_daily = px.line(
            daily,
            x="Created At",
            y="Events",
            title="Daily audit event volume",
            markers=True,
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)

    actor_breakdown = pd.DataFrame(
        {
            "Actor class": ["Explicit system", "Other / unidentified actor"],
            "Events": [int(system_mask.sum()), int((~system_mask).sum())],
        }
    )
    st.plotly_chart(
        px.bar(actor_breakdown, x="Actor class", y="Events", title="System vs other events"),
        use_container_width=True,
    )

    st.subheader("Recent activity summary")
    # No names, emails, IP addresses, user-agent strings or free-text event descriptions.
    safe = filtered_audit[["Created At", "Entity Type"]].copy()
    safe["Actor class"] = filtered_audit["_Actor class"]
    if "Where" in filtered_audit.columns:
        safe["Application area"] = filtered_audit["Where"].astype(str)
    safe = safe.sort_values("Created At", ascending=False).head(20)
    st.dataframe(safe, use_container_width=True, hide_index=True)
    st.caption(
        "The 'system' label is based only on the literal Who value. Other actors are not "
        "necessarily employees, and this view does not determine whether an event was authorised."
    )


def fmt_naira(value) -> str:
    if pd.isna(value):
        return "—"
    return "₦" + f"{value:,.0f}"


def make_series(df: pd.DataFrame, frequency: str, metric: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    tmp = df.set_index("Disburse Date").sort_index()
    if metric == "Disbursement amount":
        series = tmp["Loan Amount"].resample(frequency).sum()
    else:
        series = tmp["Loan Amount"].resample(frequency).count()
    if len(series):
        # Fill periods with no disbursements as zero so the timeline remains continuous.
        series = series.fillna(0).astype(float)
    return series


def forecast_series(series: pd.Series, horizon: int):
    """Compare naive and moving-average baselines with ETS where the data supports it."""
    series = series.astype(float).dropna()
    if len(series) < 6:
        return None, None, "At least 6 time periods are needed for a basic forecast. Try weekly frequency for more observations."

    test_n = max(2, min(6, int(round(len(series) * 0.2))))
    if len(series) - test_n < 3:
        test_n = max(1, len(series) - 3)
    train, test = series.iloc[:-test_n], series.iloc[-test_n:]

    candidates = {}
    # Last-observation baseline
    candidates["Naive baseline"] = np.repeat(train.iloc[-1], len(test))
    # Trailing 3-period mean baseline
    candidates["3-period moving average"] = np.repeat(train.tail(min(3, len(train))).mean(), len(test))

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(train, trend="add", damped_trend=True, initialization_method="estimated").fit(optimized=True)
            candidates["Exponential smoothing"] = np.asarray(model.forecast(len(test)))
    except Exception:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(train, initialization_method="estimated").fit(optimized=True)
                candidates["Exponential smoothing"] = np.asarray(model.forecast(len(test)))
        except Exception:
            pass

    actual = test.to_numpy(dtype=float)
    evaluation = []
    for name, preds in candidates.items():
        preds = np.asarray(preds, dtype=float)
        mae = float(np.mean(np.abs(actual - preds)))
        rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))
        denom = np.sum(np.abs(actual))
        wape = float(np.sum(np.abs(actual - preds)) / denom * 100) if denom else np.nan
        evaluation.append({"Model": name, "MAE": mae, "RMSE": rmse, "WAPE (%)": wape})
    scores = pd.DataFrame(evaluation).sort_values("MAE")
    best_name = scores.iloc[0]["Model"]

    # Fit the selected model to all available observations for future estimates.
    if best_name == "Naive baseline":
        future_values = np.repeat(series.iloc[-1], horizon).astype(float)
    elif best_name == "3-period moving average":
        future_values = np.repeat(series.tail(min(3, len(series))).mean(), horizon).astype(float)
    else:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_model = ExponentialSmoothing(series, trend="add", damped_trend=True, initialization_method="estimated").fit(optimized=True)
                future_values = np.asarray(full_model.forecast(horizon), dtype=float)
        except Exception:
            future_values = np.repeat(series.iloc[-1], horizon).astype(float)
            best_name = "Naive baseline (fallback)"

    future_values = np.maximum(future_values, 0)
    if isinstance(series.index, pd.DatetimeIndex):
        freq = pd.infer_freq(series.index)
        if freq is None:
            # Use the known index spacing from the selected resampling frequency.
            freq = "D"
        future_index = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
    else:
        future_index = pd.RangeIndex(start=len(series), stop=len(series) + horizon)
    forecast = pd.Series(future_values, index=future_index, name="Forecast")
    return forecast, scores, best_name

st.title("📊 Microcredit Forecasting & Staff Performance")
st.caption("Historical loan disbursements, staff portfolio indicators, and short-term time-series forecasts.")

with st.sidebar:
    st.header("Data sources")
    uploaded = st.file_uploader("Upload your bank's General Report", type=["xlsx"], key="loan_report_upload")
    activity_file = st.file_uploader(
        "Upload CBA Audit Trail",
        type=["xlsx", "csv"],
        key="audit_trail_upload",
    )
    st.caption("Loan reporting excludes customer identifiers from its working table.")
    st.caption("Audit monitoring uses aggregated fields and does not display raw event descriptions, names, emails, IP addresses or browser identifiers.")

audit_df = None
audit_load_error = None
if activity_file is not None:
    try:
        audit_df = load_audit_log(activity_file.getvalue(), activity_file.name)
    except Exception as exc:
        audit_load_error = str(exc)


if uploaded is None:
    st.info("Upload **General_Report** to use forecasting and portfolio analysis. You can still use the audit section independently.")
    if audit_load_error:
        st.error(f"Could not read the audit log: {audit_load_error}")
    elif audit_df is not None:
        render_audit_dashboard(audit_df)
    else:
        st.markdown("Upload the General Report or the CBA Audit Trail in the sidebar to begin.")
    st.stop()

try:
    df = load_report(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read this report: {exc}")
    st.stop()

if df.empty:
    st.error("No valid records remain after checking disbursement dates and loan amounts.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    min_date = df["Disburse Date"].min().date()
    max_date = df["Disburse Date"].max().date()
    date_range = st.date_input("Disbursement date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    branches = sorted(df["Branch"].unique().tolist())
    selected_branches = st.multiselect("Branch", branches, default=branches)
    products = sorted(df["Loan Product"].unique().tolist())
    selected_products = st.multiselect("Loan product", products, default=products)
    officers = sorted(df["RO"].unique().tolist())
    selected_officers = st.multiselect("Relationship Officer (RO)", officers, default=officers)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered = df[
    df["Disburse Date"].dt.date.between(start_date, end_date)
    & df["Branch"].isin(selected_branches)
    & df["Loan Product"].isin(selected_products)
    & df["RO"].isin(selected_officers)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

# KPI row
loan_count = len(filtered)
disbursed_total = filtered["Loan Amount"].sum()
collected_total = filtered["Collected"].sum() if "Collected" in filtered else np.nan
outstanding_total = filtered["Outstanding Balance"].sum() if "Outstanding Balance" in filtered else np.nan
default_total = filtered["Default Amount"].sum() if "Default Amount" in filtered else np.nan
past_due_loans = int((filtered["Past Due Count"].fillna(0) > 0).sum()) if "Past Due Count" in filtered else None

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Loans disbursed", f"{loan_count:,}")
k2.metric("Total disbursed", fmt_naira(disbursed_total))
k3.metric("Collected*", fmt_naira(collected_total))
k4.metric("Outstanding balance*", fmt_naira(outstanding_total))
k5.metric("Loans with past-due count*", f"{past_due_loans:,}" if past_due_loans is not None else "—")
st.caption("*Collection, outstanding balance and past-due figures are portfolio snapshot values from the uploaded report, not dated collection transactions. Do not interpret them as collections made during the selected date range.")

st.divider()
tab_forecast, tab_staff, tab_portfolio, tab_audit, tab_data = st.tabs(
    ["Forecasting", "Staff performance", "Portfolio monitoring", "Audit trail", "Data checks"]
)

with tab_forecast:
    st.subheader("Time-series forecast")
    st.write("Forecasts are based on the **Disburse Date** and **Loan Amount** in this report. They estimate future disbursement volume, not future repayments or employee visits.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.selectbox("Forecast metric", ["Disbursement amount", "Number of loans"])
    with c2:
        period_label = st.selectbox("Time interval", ["Weekly", "Monthly", "Daily"], index=0)
    with c3:
        horizon = st.slider("Periods to forecast", min_value=2, max_value=12, value=4)
    frequency = {"Weekly": "W-SUN", "Monthly": "MS", "Daily": "D"}[period_label]
    series = make_series(filtered, frequency, metric)
    if period_label == "Monthly" and len(series) < 12:
        st.warning("Monthly history is limited. Treat this as an exploratory forecast; weekly data may provide more observations but can be noisier.")
    if len(series) < 6:
        st.warning(f"Only {len(series)} {period_label.lower()} periods are available after filtering. Broaden the filters or choose a finer interval to obtain at least 6 periods.")
    else:
        forecast, scores, best_model = forecast_series(series, horizon)
        if forecast is not None:
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name="Historical actual"))
            chart.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Forecast", line=dict(dash="dash")))
            chart.update_layout(height=430, xaxis_title="Period", yaxis_title="Naira" if metric == "Disbursement amount" else "Number of loans", legend_title="Series", margin=dict(l=10, r=10, t=30, b=10))
            if metric == "Disbursement amount":
                chart.update_yaxes(tickprefix="₦", tickformat=",.0f")
            st.plotly_chart(chart, use_container_width=True)
            best = scores.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected model", best_model)
            m2.metric("Backtest MAE", f"{best['MAE']:,.2f}")
            m3.metric("Backtest RMSE", f"{best['RMSE']:,.2f}")
            st.markdown("**Model comparison (chronological holdout)**")
            st.dataframe(scores.round(2), use_container_width=True, hide_index=True)
            st.caption("MAE and RMSE are calculated on the last observed periods held out from training. Lower is better. With limited history, these scores can vary substantially; compare against the baseline before using forecasts for operational decisions.")
            out = forecast.rename_axis("Forecast period").reset_index()
            out.columns = ["Forecast period", "Forecast value"]
            if metric == "Disbursement amount":
                out["Forecast value"] = out["Forecast value"].round(0).map(fmt_naira)
            else:
                out["Forecast value"] = out["Forecast value"].round(0).astype(int)
            st.markdown("**Forecast values**")
            st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            st.warning("Unable to create a forecast. Try a wider date range or fewer filters.")

with tab_staff:
    st.subheader("Relationship Officer (RO) performance")
    st.caption("This report supports loan portfolio output analysis by RO. It does not contain daily visit logs, applications received, or follow-up activity records.")
    staff = filtered.groupby("RO", dropna=False).agg(
        Loans=("Loan Amount", "size"),
        Amount_Disbursed=("Loan Amount", "sum"),
        Average_Loan_Size=("Loan Amount", "mean"),
    ).reset_index().sort_values("Amount_Disbursed", ascending=False)
    if "Collected" in filtered.columns:
        coll = filtered.groupby("RO")["Collected"].sum().rename("Portfolio_Collected_Snapshot")
        staff = staff.merge(coll, on="RO", how="left")
    if "Outstanding Balance" in filtered.columns:
        outst = filtered.groupby("RO")["Outstanding Balance"].sum().rename("Outstanding_Snapshot")
        staff = staff.merge(outst, on="RO", how="left")
    left, right = st.columns(2)
    with left:
        top = staff.head(15).sort_values("Amount_Disbursed", ascending=True)
        fig = px.bar(top, x="Amount_Disbursed", y="RO", orientation="h", title="Top 15 ROs by disbursement amount", labels={"Amount_Disbursed": "Amount disbursed (₦)", "RO": "Relationship Officer"})
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig2 = px.bar(staff.head(15).sort_values("Loans", ascending=True), x="Loans", y="RO", orientation="h", title="Top 15 ROs by number of loans", labels={"Loans": "Number of loans", "RO": "Relationship Officer"})
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, use_container_width=True)
    display_staff = staff.copy()
    display_staff["Amount_Disbursed"] = display_staff["Amount_Disbursed"].map(fmt_naira)
    display_staff["Average_Loan_Size"] = display_staff["Average_Loan_Size"].map(fmt_naira)
    for col in ["Portfolio_Collected_Snapshot", "Outstanding_Snapshot"]:
        if col in display_staff.columns:
            display_staff[col] = display_staff[col].map(fmt_naira)
    st.dataframe(display_staff, use_container_width=True, hide_index=True)
    csv = staff.to_csv(index=False).encode("utf-8")
    st.download_button("Download RO summary (CSV)", data=csv, file_name="ro_performance_summary.csv", mime="text/csv")

with tab_portfolio:
    st.subheader("Portfolio monitoring (snapshot indicators)")
    p1, p2 = st.columns(2)
    with p1:
        prod = filtered.groupby("Loan Product").agg(Loans=("Loan Amount", "size"), Amount_Disbursed=("Loan Amount", "sum")).reset_index().sort_values("Amount_Disbursed", ascending=False)
        fig = px.bar(prod, x="Loan Product", y="Amount_Disbursed", title="Disbursement amount by loan product", labels={"Amount_Disbursed": "Amount disbursed (₦)"})
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        if "Default Amount" in filtered.columns:
            arrears = filtered.groupby("Loan Product")["Default Amount"].sum().reset_index().sort_values("Default Amount", ascending=False)
            fig = px.bar(arrears, x="Loan Product", y="Default Amount", title="Default amount by loan product (snapshot)", labels={"Default Amount": "Default amount (₦)"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("The uploaded report does not contain a Default Amount column.")
    if "Past Due Count" in filtered.columns:
        overdue = filtered.copy()
        overdue["Past Due Status"] = np.where(overdue["Past Due Count"].fillna(0) > 0, "Past due count > 0", "No past due count")
        counts = overdue["Past Due Status"].value_counts().rename_axis("Status").reset_index(name="Loan count")
        fig = px.pie(counts, names="Status", values="Loan count", title="Loans by past-due-count indicator")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Default Amount and Past Due Count are interpreted as values in the supplied report. Confirm the bank's definitions before using them as official delinquency KPIs.")

with tab_audit:
    if audit_load_error:
        st.error(f"Could not read the audit log: {audit_load_error}")
        st.info("Check the audit file format and confirm it contains Who, Entity Type and Created At columns.")
    elif audit_df is not None:
        render_audit_dashboard(audit_df)
    else:
        st.info("Upload the CBA Audit Trail in the sidebar to view audit activity.")
        st.caption("The audit trail can be loaded independently of the General Report.")

with tab_data:
    st.subheader("Data checks and coverage")
    info = pd.DataFrame({
        "Check": ["Rows used", "Date range", "Branches", "Relationship Officers", "Loan products", "Missing disbursement dates removed", "Missing loan amounts removed"],
        "Value": [f"{len(filtered):,}", f"{filtered['Disburse Date'].min():%Y-%m-%d} to {filtered['Disburse Date'].max():%Y-%m-%d}", str(filtered["Branch"].nunique()), str(filtered["RO"].nunique()), str(filtered["Loan Product"].nunique()), str(int(df["Disburse Date"].isna().sum())), str(int(df["Loan Amount"].isna().sum()))]
    })
    st.dataframe(info, use_container_width=True, hide_index=True)
    st.markdown("**Fields not used in the app's analysis table**")
    st.write("Customer names, loan reference numbers, customer IDs, mobile phone numbers and alternative phone numbers are excluded from the working table.")
    st.markdown("**For true daily activity tracking, add an activity log**")
    st.write("Recommended fields: Activity Date, Staff/RO ID, Branch, Activity Type (visit, follow-up, application, collection), Activity Count, Target, Outcome. Keep customer-level details out of this log unless strictly required and approved.")
