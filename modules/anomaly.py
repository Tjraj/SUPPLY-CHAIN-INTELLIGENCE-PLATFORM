# modules/anomaly.py
# PURPOSE: Detect anomalies in supply chain data
# Method : Custom IsolationForest wrapper that flags BUSINESS-MEANINGFUL anomalies
# Interviewers will ask: "What does your anomaly detection actually catch?"

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")


# ── STEP 1: ORDER LEVEL ANOMALY DETECTION ─────────────────────────────────────
def detect_order_anomalies(df):
    """
    Detect anomalies at the individual ORDER level.

    Features used:
    - price              : unusually high or low order price
    - freight_value      : unusually high shipping cost
    - delivery_days      : unusually long or short delivery
    - payment_value      : unusually high payment amount
    - payment_installments: unusually high number of installments

    IsolationForest MATH (for interviews):
    - Builds random decision trees that try to ISOLATE each data point
    - Anomalies are isolated faster (fewer splits needed)
    - contamination=0.05 means we expect ~5% of data to be anomalous
    - Returns -1 for anomaly, 1 for normal
    """

    print("\n── Order-Level Anomaly Detection ───────────────────")

    # Select features for anomaly detection
    features = [
        "price",
        "freight_value",
        "delivery_days",
        "payment_value",
        "payment_installments"
    ]

    # Only use rows where all features are available
    df_model = df[features].copy()
    df_model  = df_model.fillna(df_model.median())

    # Train IsolationForest
    iso_forest = IsolationForest(
        n_estimators  = 100,        # number of trees
        contamination = 0.05,       # expect 5% anomalies
        random_state  = 42,         # reproducible results
        n_jobs        = -1          # use all CPU cores
    )

    # Fit and predict — returns 1 (normal) or -1 (anomaly)
    predictions    = iso_forest.fit_predict(df_model)
    anomaly_scores = iso_forest.score_samples(df_model)

    # Add results back to dataframe
    df = df.copy()
    df["is_anomaly"]    = (predictions == -1).astype(int)
    df["anomaly_score"] = anomaly_scores  # more negative = more anomalous

    total    = len(df)
    flagged  = df["is_anomaly"].sum()
    pct      = flagged / total * 100

    print(f"  Total orders analyzed : {total:,}")
    print(f"  Anomalies flagged     : {flagged:,} ({pct:.1f}%)")

    return df


# ── STEP 2: CLASSIFY ANOMALY TYPES ────────────────────────────────────────────
def classify_anomaly_types(df):
    """
    IsolationForest tells us WHAT is anomalous but not WHY.
    This function adds business context by classifying each anomaly type.

    Types we detect:
    1. Price spike       — order price >> category average
    2. Freight spike     — shipping cost >> normal range
    3. Delivery delay    — delivered way after estimate
    4. Fast delivery     — delivered suspiciously fast (data error?)
    5. High payment      — payment value >> order price (overpayment?)
    6. Stockout signal   — sudden drop in category orders
    """

    print("\n── Classifying Anomaly Types ────────────────────────")

    anomalies = df[df["is_anomaly"] == 1].copy()

    # ── Type 1: Price Spike
    # Price is more than 3 standard deviations above the mean
    price_mean = df["price"].mean()
    price_std  = df["price"].std()
    anomalies["is_price_spike"] = (
        anomalies["price"] > price_mean + 3 * price_std
    ).astype(int)

    # ── Type 2: Freight Spike
    freight_mean = df["freight_value"].mean()
    freight_std  = df["freight_value"].std()
    anomalies["is_freight_spike"] = (
        anomalies["freight_value"] > freight_mean + 3 * freight_std
    ).astype(int)

    # ── Type 3: Delivery Delay
    # Delivered more than 14 days after estimated date
    anomalies["is_severe_delay"] = (
        anomalies["delay_days"] > 14
    ).astype(int)

    # ── Type 4: Suspiciously Fast Delivery
    # Delivered in less than 1 day — likely a data error
    anomalies["is_fast_delivery"] = (
        anomalies["delivery_days"] < 1
    ).astype(int)

    # ── Type 5: Payment Mismatch
    # Payment value is more than 2x the order price
    anomalies["is_payment_mismatch"] = (
        anomalies["payment_value"] > anomalies["price"] * 2
    ).astype(int)

    # ── Summary of anomaly types
    print(f"  Price spikes         : {anomalies['is_price_spike'].sum():,}")
    print(f"  Freight spikes       : {anomalies['is_freight_spike'].sum():,}")
    print(f"  Severe delays (14d+) : {anomalies['is_severe_delay'].sum():,}")
    print(f"  Fast delivery (<1d)  : {anomalies['is_fast_delivery'].sum():,}")
    print(f"  Payment mismatches   : {anomalies['is_payment_mismatch'].sum():,}")

    return anomalies


# ── STEP 3: DETECT MONTHLY ORDER SPIKES ───────────────────────────────────────
def detect_monthly_spikes(df):
    """
    Detect months where order volume was unusually high or low.
    Uses rolling mean + standard deviation threshold.

    MATH (for interviews):
    Upper threshold = rolling_mean + (2 * rolling_std)
    Lower threshold = rolling_mean - (2 * rolling_std)

    Any month outside this band is flagged as a spike or drop.
    This is called a "control chart" in supply chain management.
    """

    print("\n── Monthly Volume Spike Detection ───────────────────")

    # Aggregate to monthly order counts
    monthly = (
        df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))
        .agg(
            order_count   = ("order_id",          "count"),
            avg_price     = ("price",              "mean"),
            total_revenue = ("total_order_value",  "sum")
        )
        .reset_index()
    )
    monthly["order_purchase_timestamp"] = monthly[
        "order_purchase_timestamp"
    ].dt.to_timestamp()
    monthly.rename(columns={"order_purchase_timestamp": "month"}, inplace=True)

    # Rolling statistics (3-month window)
    monthly["rolling_mean"] = (
        monthly["order_count"].rolling(window=3, min_periods=1).mean()
    )
    monthly["rolling_std"]  = (
        monthly["order_count"].rolling(window=3, min_periods=1).std().fillna(0)
    )

    # Flag spikes and drops
    monthly["upper_threshold"] = monthly["rolling_mean"] + 2 * monthly["rolling_std"]
    monthly["lower_threshold"] = monthly["rolling_mean"] - 2 * monthly["rolling_std"]

    monthly["is_spike"] = (
        monthly["order_count"] > monthly["upper_threshold"]
    ).astype(int)
    monthly["is_drop"]  = (
        monthly["order_count"] < monthly["lower_threshold"]
    ).astype(int)

    spikes = monthly[monthly["is_spike"] == 1]
    drops  = monthly[monthly["is_drop"]  == 1]

    print(f"  Months analyzed      : {len(monthly)}")
    print(f"  Volume spikes found  : {len(spikes)}")
    print(f"  Volume drops found   : {len(drops)}")

    if len(spikes) > 0:
        print(f"\n  Spike months:")
        for _, row in spikes.iterrows():
            print(f"    {row['month'].strftime('%Y-%m')}: "
                  f"{row['order_count']:,} orders "
                  f"(threshold: {row['upper_threshold']:,.0f})")

    if len(drops) > 0:
        print(f"\n  Drop months:")
        for _, row in drops.iterrows():
            print(f"    {row['month'].strftime('%Y-%m')}: "
                  f"{row['order_count']:,} orders "
                  f"(threshold: {row['lower_threshold']:,.0f})")

    return monthly


# ── STEP 4: DETECT SUPPLIER ANOMALIES ─────────────────────────────────────────
def detect_supplier_anomalies(df):
    """
    Flag suppliers whose performance metrics are anomalous.
    Metrics: avg delivery days, late rate, avg review score
    """

    print("\n── Supplier Anomaly Detection ───────────────────────")

    # Aggregate per seller
    supplier_stats = (
        df.groupby("seller_id")
        .agg(
            total_orders    = ("order_id",       "count"),
            avg_delivery    = ("delivery_days",  "mean"),
            late_rate       = ("is_late",        "mean"),
            avg_review      = ("review_score",   "mean"),
            avg_price       = ("price",          "mean")
        )
        .reset_index()
    )

    # Only analyze suppliers with 10+ orders (enough data)
    supplier_stats = supplier_stats[
        supplier_stats["total_orders"] >= 10
    ].copy()

    # Flag bad suppliers using simple thresholds
    # Late rate > 20% is a red flag
    supplier_stats["high_late_rate"] = (
        supplier_stats["late_rate"] > 0.20
    ).astype(int)

    # Avg review score < 3.0 is a red flag
    supplier_stats["low_review_score"] = (
        supplier_stats["avg_review"] < 3.0
    ).astype(int)

    # Avg delivery > 20 days is a red flag
    supplier_stats["slow_delivery"] = (
        supplier_stats["avg_delivery"] > 20
    ).astype(int)

    # Flag suppliers with ANY red flag
    supplier_stats["is_flagged"] = (
        (supplier_stats["high_late_rate"]   == 1) |
        (supplier_stats["low_review_score"] == 1) |
        (supplier_stats["slow_delivery"]    == 1)
    ).astype(int)

    flagged = supplier_stats[supplier_stats["is_flagged"] == 1]

    print(f"  Suppliers analyzed   : {len(supplier_stats):,}")
    print(f"  Flagged suppliers    : {len(flagged):,}")
    print(f"  High late rate (20%+): {supplier_stats['high_late_rate'].sum():,}")
    print(f"  Low review (<3.0)    : {supplier_stats['low_review_score'].sum():,}")
    print(f"  Slow delivery (20d+) : {supplier_stats['slow_delivery'].sum():,}")

    return supplier_stats


# ── MAIN RUNNER ────────────────────────────────────────────────────────────────
def run_anomaly_detection(df):
    """
    Full anomaly detection pipeline.
    Input : cleaned delivered orders DataFrame
    Output: dict with all anomaly results
    """

    print("\n" + "="*55)
    print("ANOMALY DETECTION MODULE")
    print("="*55)

    # Run all detection methods
    df_with_anomalies = detect_order_anomalies(df)
    anomalies_df      = classify_anomaly_types(df_with_anomalies)
    monthly_df        = detect_monthly_spikes(df_with_anomalies)
    supplier_df       = detect_supplier_anomalies(df_with_anomalies)

    # Save results
    anomalies_df.to_csv("data/processed/anomalies.csv",        index=False)
    monthly_df.to_csv("data/processed/monthly_spikes.csv",     index=False)
    supplier_df.to_csv("data/processed/supplier_anomalies.csv", index=False)

    print("\n── Files Saved ──────────────────────────────────────")
    print("  data/processed/anomalies.csv")
    print("  data/processed/monthly_spikes.csv")
    print("  data/processed/supplier_anomalies.csv")

    results = {
        "orders_with_anomalies" : df_with_anomalies,
        "anomalies"             : anomalies_df,
        "monthly_spikes"        : monthly_df,
        "supplier_anomalies"    : supplier_df
    }

    print("\nAnomaly detection complete ✅")
    return results


# ── TEST ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading cleaned data...")
    df = pd.read_csv(
        "data/processed/orders_delivered.csv",
        parse_dates=["order_purchase_timestamp"]
    )
    print(f"Loaded: {df.shape[0]:,} rows")

    results = run_anomaly_detection(df)

    # Quick summary
    print("\n── Final Anomaly Summary ────────────────────────────")
    anomalies = results["anomalies"]
    print(f"  Total anomalous orders : {len(anomalies):,}")
    print(f"  Top anomaly states     :")
    if "customer_state" in anomalies.columns:
        top_states = anomalies["customer_state"].value_counts().head(3)
        for state, count in top_states.items():
            print(f"    {state}: {count:,} anomalies")