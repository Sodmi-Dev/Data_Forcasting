# Copilot instructions

## Commands

- Install the pinned dependencies: `python -m pip install -r requirements.txt`
- Start the dashboard: `python -m streamlit run app.py`
- Check Python syntax: `python -m py_compile app.py`

There is no configured build, test suite, or linter in this repository, so there is no single-test command.

## Application architecture

- `app.py` contains both the data-processing helpers and the Streamlit UI; the app is organized as a single top-to-bottom Streamlit script, not as separate routes or service modules.
- `load_report` reads uploaded Excel bytes, validates the required report columns, converts dates and numeric fields, normalizes category values, and drops unusable rows. It deliberately keeps only required reporting fields and recognized optional numeric indicators; keep customer names, contact details, IDs, and loan references out of the working DataFrame.
- The UI stops early when there is no upload, the report is invalid or empty, or the chosen filters return no records. When adding dashboard features, preserve these guards and apply the shared date, branch, product, and RO filters consistently.
- The dashboard then derives KPIs and renders four tabs: time-series forecasts, RO aggregates, portfolio snapshot indicators, and data coverage checks. RO and portfolio views are report-based aggregates, not daily staff activity or dated repayment histories.
- Forecasting resamples the filtered disbursement data, compares naive and moving-average baselines with exponential smoothing when it can be fitted, and selects a model using a chronological holdout. Keep this evaluation separate from the final fit used for future estimates; forecasts concern disbursements, not repayments.

## Data and UI conventions

- Report column names are exact, case-sensitive source headers. `REQUIRED_COLUMNS` define the minimum accepted report; fields in `OPTIONAL_NUMERIC` are included only when present, and UI metrics/charts must remain conditional on those fields.
- Keep Excel parsing and normalization in `load_report`, time-series construction in `make_series`, and forecast selection/evaluation in `forecast_series`; the Streamlit page consumes those helpers.
- The upload is cached by its bytes with `st.cache_data`. Avoid persisting uploaded report data or exposing customer-level fields in charts, exports, or logs.
- Keep report-snapshot fields such as collected, outstanding, default, and past-due values described as snapshots. They should not be presented as transactions or time-window activity.
- CSV export currently contains grouped RO summary values, not the underlying loan rows.
