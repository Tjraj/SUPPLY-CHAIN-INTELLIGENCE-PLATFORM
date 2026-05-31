# modules/forecast.py
# PURPOSE: Forecast future demand using 3 methods built from scratch
# Methods: Moving Average → Exponential Smoothing → Prophet (benchmark)
# Interviewers WILL ask you to explain the math behind each one

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── STEP 1: PREPARE TIME SERIES DATA ──────────────────────────────────────────
def prepare_time_series(df, freq="M"):
    """
    Aggregate orders into a time series at monthly frequency.

    Input : cleaned delivered orders DataFrame
    Output: DataFrame with columns [ds, y] — Prophet-compatible format
            ds = date, y = order count
    """

    # Count orders per month
    ts = (
        df.groupby("order_date")
        .agg(
            order_count   = ("order_id",          "count"),
            total_revenue = ("total_order_value",  "sum"),
            avg_price     = ("price",              "mean")
        )
        .reset_index()
    )

    # Convert order_date to datetime
    ts["order_date"] = pd.to_datetime(ts["order_date"])

    # Resample to monthly frequency
    ts = (
        ts.set_index("order_date")
        .resample("ME")
        .agg(
            order_count   = ("order_count",   "sum"),
            total_revenue = ("total_revenue", "sum"),
            avg_price     = ("avg_price",     "mean")
        )
        .reset_index()
    )

    # Rename to Prophet format
    ts.rename(columns={"order_date": "ds", "order_count": "y"}, inplace=True)

    # Drop incomplete months with very low counts
    ts = ts[ts["y"] > 10].copy()

    print(f"  Time series prepared: {len(ts)} monthly data points")
    print(f"  Date range: {ts['ds'].min().strftime('%Y-%m')} to {ts['ds'].max().strftime('%Y-%m')}")
    print(f"  Avg monthly orders: {ts['y'].mean():.0f}")

    return ts


# ── STEP 2: MOVING AVERAGE FORECAST ───────────────────────────────────────────
def moving_average_forecast(ts, window=3, periods=6):
    """
    Simple Moving Average (SMA) Forecast — built from scratch using NumPy.

    MATH EXPLANATION (for interviews):
    SMA_t = (y_t + y_(t-1) + y_(t-2) + ... + y_(t-window+1)) / window

    We take the last 'window' months, average them,
    and use that as the forecast for the next month.
    Then slide the window forward and repeat.

    window  = how many past months to average
    periods = how many future months to forecast
    """

    values    = ts["y"].values.copy()
    last_date = ts["ds"].iloc[-1]

    forecasts = []
    extended  = list(values)

    for i in range(periods):
        window_values = extended[-window:]
        prediction    = np.mean(window_values)
        extended.append(prediction)

        next_date = last_date + pd.DateOffset(months=i+1)
        forecasts.append({
            "ds"     : next_date,
            "yhat"   : round(prediction, 0),
            "method" : "Moving Average",
            "window" : window
        })

    forecast_df = pd.DataFrame(forecasts)

    print(f"\n  Moving Average (window={window}) — {periods}-month forecast:")
    for _, row in forecast_df.iterrows():
        print(f"    {row['ds'].strftime('%Y-%m')}: {row['yhat']:,.0f} orders")

    return forecast_df


# ── STEP 3: EXPONENTIAL SMOOTHING FORECAST ────────────────────────────────────
def exponential_smoothing_forecast(ts, alpha=0.3, periods=6):
    """
    Simple Exponential Smoothing (SES) — built from scratch using NumPy.

    MATH EXPLANATION (for interviews):
    S_t = alpha * y_t + (1 - alpha) * S_(t-1)

    where:
        S_t   = smoothed value at time t (our forecast)
        y_t   = actual value at time t
        alpha = smoothing factor (0 to 1)

    High alpha (0.8) = weights recent data heavily = reacts fast to changes
    Low alpha  (0.2) = weights all history equally = more stable/slow

    We use alpha=0.3 as a balanced default.
    Forecast for all future periods = last smoothed value (flat forecast)
    """

    values    = ts["y"].values
    last_date = ts["ds"].iloc[-1]

    # Initialize: first smoothed value = first actual value
    smoothed = [values[0]]

    # Apply exponential smoothing formula across all historical data
    for t in range(1, len(values)):
        s_t = alpha * values[t] + (1 - alpha) * smoothed[-1]
        smoothed.append(s_t)

    # Forecast for all future periods = last smoothed value
    last_smoothed = smoothed[-1]

    forecasts = []
    for i in range(periods):
        next_date = last_date + pd.DateOffset(months=i+1)
        forecasts.append({
            "ds"     : next_date,
            "yhat"   : round(last_smoothed, 0),
            "method" : "Exponential Smoothing",
            "alpha"  : alpha
        })

    forecast_df = pd.DataFrame(forecasts)

    print(f"\n  Exponential Smoothing (alpha={alpha}) — {periods}-month forecast:")
    for _, row in forecast_df.iterrows():
        print(f"    {row['ds'].strftime('%Y-%m')}: {row['yhat']:,.0f} orders")

    return forecast_df


# ── STEP 4: PROPHET FORECAST (FIXED) ──────────────────────────────────────────
def prophet_forecast(ts, periods=6):
    """
    Facebook Prophet forecast — used as benchmark comparison.

    FIX APPLIED:
    - Changed seasonality_mode from 'multiplicative' to 'additive'
    - Added changepoint_prior_scale=0.05 to reduce overfitting
    - These two changes fix the negative forecast problem on small datasets

    WHY THIS MATTERS (interview answer):
    Multiplicative mode assumes seasonal swings SCALE with the trend.
    With only 21 data points that's unreliable — it overfit badly.
    Additive mode assumes seasonal swings stay CONSTANT — safer with
    limited data. changepoint_prior_scale controls how flexible the
    trend line is — lower = less flexible = less overfitting.
    """

    try:
        from prophet import Prophet

        prophet_df = ts[["ds", "y"]].copy()

        model = Prophet(
            yearly_seasonality      = True,
            weekly_seasonality      = False,   # monthly data — no weekly pattern
            daily_seasonality       = False,
            seasonality_mode        = "additive",  # ← FIXED: was multiplicative
            changepoint_prior_scale = 0.05,        # ← ADDED: prevents overfitting
            interval_width          = 0.80
        )
        model.fit(prophet_df)

        future   = model.make_future_dataframe(periods=periods, freq="ME")
        forecast = model.predict(future)

        future_forecast = forecast.tail(periods)[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()
        future_forecast["method"] = "Prophet"
        future_forecast["yhat"]   = future_forecast["yhat"].clip(lower=0).round(0)

        print(f"\n  Prophet (additive, fixed) — {periods}-month forecast:")
        for _, row in future_forecast.iterrows():
            print(f"    {row['ds'].strftime('%Y-%m')}: {row['yhat']:,.0f} orders "
                  f"(range: {max(0,row['yhat_lower']):,.0f} – {row['yhat_upper']:,.0f})")

        return future_forecast

    except ImportError:
        print("\n  Prophet not installed — skipping")
        return pd.DataFrame()

    except Exception as e:
        print(f"\n  Prophet forecast failed: {e}")
        return pd.DataFrame()


# ── STEP 5: COMPARE ALL THREE METHODS ─────────────────────────────────────────
def compare_forecasts(ma_forecast, es_forecast, prophet_forecast_df):
    """
    Side-by-side comparison of all three forecasting methods.
    This is what you show on the dashboard — and in interviews.
    """

    print("\n── Forecast Comparison (all methods) ───────────────")
    print(f"{'Month':<12} {'Moving Avg':>12} {'Exp Smooth':>12} {'Prophet':>12}")
    print("─" * 52)

    months = ma_forecast["ds"].tolist()

    for i, month in enumerate(months):
        ma_val = ma_forecast.iloc[i]["yhat"]
        es_val = es_forecast.iloc[i]["yhat"]

        if not prophet_forecast_df.empty and i < len(prophet_forecast_df):
            p_val = prophet_forecast_df.iloc[i]["yhat"]
            p_str = f"{p_val:>12,.0f}"
        else:
            p_str = f"{'N/A':>12}"

        print(f"{month.strftime('%Y-%m'):<12} {ma_val:>12,.0f} {es_val:>12,.0f} {p_str}")

    print("─" * 52)


# ── STEP 6: CATEGORY LEVEL FORECAST ───────────────────────────────────────────
def forecast_by_category(df, top_n=5, periods=6):
    """
    Run Moving Average forecast for top N product categories.
    This shows SKU-level forecasting — rare in fresher portfolios.
    """

    print(f"\n── Category-Level Forecasts (Top {top_n}) ──────────────")

    top_cats = df["category"].value_counts().head(top_n).index.tolist()
    category_forecasts = {}

    for cat in top_cats:
        cat_df = df[df["category"] == cat].copy()

        cat_ts = (
            cat_df.groupby("order_date")["order_id"]
            .count()
            .reset_index()
        )
        cat_ts["order_date"] = pd.to_datetime(cat_ts["order_date"])
        cat_ts = (
            cat_ts.set_index("order_date")
            .resample("ME")["order_id"]
            .sum()
            .reset_index()
        )
        cat_ts.rename(columns={"order_date": "ds", "order_id": "y"}, inplace=True)
        cat_ts = cat_ts[cat_ts["y"] > 0]

        if len(cat_ts) < 4:
            continue

        cat_forecast          = moving_average_forecast(cat_ts, window=3, periods=periods)
        cat_forecast["category"] = cat
        category_forecasts[cat]  = cat_forecast

        avg_forecast = cat_forecast["yhat"].mean()
        print(f"  {cat:<30} → avg {periods}mo forecast: {avg_forecast:,.0f} orders/month")

    return category_forecasts


# ── MAIN RUNNER ────────────────────────────────────────────────────────────────
def run_forecasting(df_delivered):
    """
    Full forecasting pipeline.
    Input : delivered orders DataFrame from clean.py
    Output: dict with all forecast results
    """

    print("\n" + "="*55)
    print("DEMAND FORECASTING MODULE")
    print("="*55)

    ts = prepare_time_series(df_delivered)

    print("\nRunning forecasting methods...")
    ma_fc      = moving_average_forecast(ts, window=3, periods=6)
    es_fc      = exponential_smoothing_forecast(ts, alpha=0.3, periods=6)
    prophet_fc = prophet_forecast(ts, periods=6)

    compare_forecasts(ma_fc, es_fc, prophet_fc)

    cat_forecasts = forecast_by_category(df_delivered, top_n=5, periods=6)

    results = {
        "time_series"        : ts,
        "moving_average"     : ma_fc,
        "exp_smoothing"      : es_fc,
        "prophet"            : prophet_fc,
        "category_forecasts" : cat_forecasts
    }

    combined = pd.concat([ma_fc, es_fc], ignore_index=True)
    if not prophet_fc.empty:
        combined = pd.concat(
            [combined, prophet_fc[["ds", "yhat", "method"]]],
            ignore_index=True
        )

    combined.to_csv("data/processed/forecasts.csv", index=False)
    print("\nSaved: data/processed/forecasts.csv")

    return results


# ── TEST ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading cleaned data...")
    df_delivered = pd.read_csv(
        "data/processed/orders_delivered.csv",
        parse_dates=["order_purchase_timestamp"]
    )
    print(f"Loaded: {df_delivered.shape[0]:,} rows")

    results = run_forecasting(df_delivered)
    print("\nForecasting complete ✅")