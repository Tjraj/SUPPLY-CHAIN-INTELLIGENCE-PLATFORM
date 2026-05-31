# modules/clean.py
# PURPOSE: Clean the master table and engineer new features

import pandas as pd
import numpy as np

# ── STEP 1: PARSE ALL DATE COLUMNS ────────────────────────────────────────────
def parse_dates(df):
    """
    Convert all timestamp columns from strings to proper datetime objects.
    This is required before we can calculate delivery times, delays etc.
    """
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date"
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            # errors="coerce" turns unparseable dates into NaT instead of crashing

    print(f"  Dates parsed: {len(date_columns)} columns converted")
    return df


# ── STEP 2: HANDLE MISSING VALUES ─────────────────────────────────────────────
def handle_missing_values(df):
    """
    Deal with NaN values intelligently based on column type.
    We never drop rows — we fill or flag missing data.
    """

    before = df.isnull().sum().sum()

    # Numeric columns — fill with median (robust to outliers)
    numeric_cols = [
        "price", "freight_value", "payment_value",
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
        "review_score", "payment_installments"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Category column — fill unknown categories with "uncategorized"
    if "category" in df.columns:
        df["category"] = df["category"].fillna("uncategorized")

    if "product_category_name" in df.columns:
        df["product_category_name"] = df["product_category_name"].fillna("uncategorized")

    # City and state — fill with "unknown"
    for col in ["customer_city", "seller_city", "customer_state", "seller_state"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Payment type — fill with "unknown"
    if "payment_type" in df.columns:
        df["payment_type"] = df["payment_type"].fillna("unknown")

    after = df.isnull().sum().sum()
    print(f"  Missing values: {before:,} → {after:,} remaining")
    return df


# ── STEP 3: FEATURE ENGINEERING ───────────────────────────────────────────────
def engineer_features(df):
    """
    Create new columns that are more useful for analysis than raw data.
    These are the columns recruiters will ask you about in interviews.
    """

    # ── 3A: Delivery time in days (actual)
    # How long did it actually take from purchase to delivery?
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    # Negative or zero delivery days are data errors — set to NaN
    df.loc[df["delivery_days"] <= 0, "delivery_days"] = np.nan

    # ── 3B: Was the order delivered late?
    # Compare actual delivery vs estimated delivery date
    df["is_late"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)
    # 1 = late, 0 = on time

    # ── 3C: Days early or late (positive = late, negative = early)
    df["delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    # ── 3D: Total order value (price + freight)
    df["total_order_value"] = df["price"] + df["freight_value"]

    # ── 3E: Order month and year (for time series forecasting later)
    df["order_year"]  = df["order_purchase_timestamp"].dt.year
    df["order_month"] = df["order_purchase_timestamp"].dt.month
    df["order_date"]  = df["order_purchase_timestamp"].dt.date

    # ── 3F: Order day of week (0=Monday, 6=Sunday)
    df["order_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

    # ── 3G: Is weekend order?
    df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)

    # ── 3H: Approval time in hours
    df["approval_hours"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600
    df.loc[df["approval_hours"] < 0, "approval_hours"] = np.nan

    print(f"  New features created: delivery_days, is_late, delay_days,")
    print(f"    total_order_value, order_year, order_month, order_date,")
    print(f"    order_dayofweek, is_weekend, approval_hours")
    return df


# ── STEP 4: REMOVE OUTLIERS ────────────────────────────────────────────────────
def remove_outliers(df):
    """
    Remove extreme outliers that would distort forecasting.
    We use IQR method — standard statistical approach.
    We only cap values, never delete rows.
    """

    def cap_outliers(series, lower_pct=0.01, upper_pct=0.99):
        """Cap values at 1st and 99th percentile"""
        lower = series.quantile(lower_pct)
        upper = series.quantile(upper_pct)
        return series.clip(lower, upper)

    # Cap price outliers
    if "price" in df.columns:
        before_max = df["price"].max()
        df["price"] = cap_outliers(df["price"])
        after_max  = df["price"].max()
        print(f"  Price capped: max {before_max:,.2f} → {after_max:,.2f}")

    # Cap freight outliers
    if "freight_value" in df.columns:
        df["freight_value"] = cap_outliers(df["freight_value"])

    # Cap delivery days outliers
    if "delivery_days" in df.columns:
        df["delivery_days"] = cap_outliers(
            df["delivery_days"].dropna()
        ).reindex(df.index)

    print(f"  Outliers capped at 1st/99th percentile")
    return df


# ── STEP 5: STANDARDIZE TEXT COLUMNS ──────────────────────────────────────────
def standardize_text(df):
    """
    Clean up text columns — consistent casing, strip whitespace.
    Inconsistent city names ('Sao Paulo' vs 'sao paulo') break groupby.
    """

    text_cols = [
        "customer_city", "seller_city",
        "customer_state", "seller_state",
        "category", "payment_type", "order_status"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()           # remove leading/trailing spaces
                .str.lower()           # convert to lowercase
                .str.replace("_", " ") # replace underscores with spaces
            )

    print(f"  Text standardized: lowercase + stripped whitespace")
    return df


# ── STEP 6: FILTER VALID ORDERS ONLY ──────────────────────────────────────────
def filter_valid_orders(df):
    """
    Keep only orders that are useful for analysis.
    Remove cancelled/unavailable orders for forecasting
    but keep them flagged for anomaly detection.
    """

    before = len(df)

    # Flag cancelled orders instead of dropping
    df["is_cancelled"] = (df["order_status"] == "canceled").astype(int)

    # For forecasting, we only want delivered orders
    # We'll create a separate clean subset
    df_delivered = df[df["order_status"] == "delivered"].copy()

    after = len(df_delivered)
    print(f"  Orders filtered: {before:,} total → {after:,} delivered")
    print(f"  Cancelled orders flagged: {df['is_cancelled'].sum():,}")

    return df, df_delivered  # return both full and delivered-only


# ── MAIN CLEANING PIPELINE ─────────────────────────────────────────────────────
def run_cleaning(master_df):
    """
    Full cleaning pipeline — runs all steps in order.
    Input:  raw master DataFrame from ingest.py
    Output: cleaned full DataFrame + delivered-only DataFrame
    """

    print("\nRunning data cleaning pipeline...")
    print("─" * 50)

    df = master_df.copy()

    df = parse_dates(df)
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = remove_outliers(df)
    df = standardize_text(df)
    df, df_delivered = filter_valid_orders(df)

    print("─" * 50)
    print(f"Cleaning complete!")
    print(f"  Full dataset    : {df.shape[0]:,} rows | {df.shape[1]} columns")
    print(f"  Delivered only  : {df_delivered.shape[0]:,} rows | {df_delivered.shape[1]} columns")

    # Save both versions
    df.to_csv("data/processed/master_orders_clean.csv", index=False)
    df_delivered.to_csv("data/processed/orders_delivered.csv", index=False)
    print(f"\n  Saved: data/processed/master_orders_clean.csv")
    print(f"  Saved: data/processed/orders_delivered.csv")

    return df, df_delivered


# ── TEST: Run directly ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load the master table we built in ingest.py
    print("Loading master table...")
    master = pd.read_csv("data/processed/master_orders.csv")
    print(f"Loaded: {master.shape[0]:,} rows")

    # Run cleaning
    df_clean, df_delivered = run_cleaning(master)

    # Quick summary stats
    print("\n── Quick Stats ──────────────────────────────────")
    print(f"  Avg delivery days  : {df_delivered['delivery_days'].mean():.1f}")
    print(f"  Late delivery rate : {df_delivered['is_late'].mean()*100:.1f}%")
    print(f"  Avg order value    : R$ {df_delivered['total_order_value'].mean():.2f}")
    print(f"  Avg review score   : {df_delivered['review_score'].mean():.2f} / 5.0")
    print(f"  Top 3 categories   :")
    top3 = df_delivered["category"].value_counts().head(3)
    for cat, count in top3.items():
        print(f"    {cat}: {count:,} orders")