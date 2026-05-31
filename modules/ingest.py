# modules/ingest.py
# PURPOSE: Load all 9 Olist CSVs and merge them into one master orders table

import pandas as pd
import os

# ── STEP 1: DEFINE FILE PATHS ──────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw/"

# Each CSV file mapped to a short name we'll use in code
FILES = {
    "orders"       : "olist_orders_dataset.csv",
    "order_items"  : "olist_order_items_dataset.csv",
    "customers"    : "olist_customers_dataset.csv",
    "products"     : "olist_products_dataset.csv",
    "sellers"      : "olist_sellers_dataset.csv",
    "payments"     : "olist_order_payments_dataset.csv",
    "reviews"      : "olist_order_reviews_dataset.csv",
    "translation"  : "product_category_name_translation.csv",
}

# ── STEP 2: LOAD ALL CSVs INTO A DICTIONARY ────────────────────────────────────
def load_raw_data():
    """
    Loads all CSV files from data/raw/ into a dictionary of DataFrames.
    Returns: dict like {"orders": df, "customers": df, ...}
    """
    dataframes = {}

    for name, filename in FILES.items():
        filepath = os.path.join(RAW_DATA_PATH, filename)

        # Check file exists before loading
        if not os.path.exists(filepath):
            print(f"WARNING: {filename} not found — skipping")
            continue

        df = pd.read_csv(filepath)
        dataframes[name] = df
        print(f"Loaded {name:15s} → {df.shape[0]:>7,} rows  |  {df.shape[1]} columns")

    print(f"\nTotal files loaded: {len(dataframes)}")
    return dataframes


# ── STEP 3: MERGE ALL INTO ONE MASTER TABLE ────────────────────────────────────
def build_master_table(dataframes):
    """
    Merges all DataFrames into one master orders table.
    The 'orders' table is the backbone — everything joins to it.
    Returns: single merged DataFrame
    """

    print("\nBuilding master table...")

    # Start with the orders backbone
    master = dataframes["orders"].copy()
    print(f"  Base (orders)          → {master.shape[0]:>7,} rows")

    # ── JOIN 1: Add order items (what was ordered + price + freight)
    # One order can have multiple items, so this expands the rows
    master = master.merge(
        dataframes["order_items"],
        on="order_id",
        how="left"
    )
    print(f"  After + order_items    → {master.shape[0]:>7,} rows")

    # ── JOIN 2: Add customer info (city, state)
    master = master.merge(
        dataframes["customers"],
        on="customer_id",
        how="left"
    )
    print(f"  After + customers      → {master.shape[0]:>7,} rows")

    # ── JOIN 3: Add product details (category, weight, dimensions)
    master = master.merge(
        dataframes["products"],
        on="product_id",
        how="left"
    )
    print(f"  After + products       → {master.shape[0]:>7,} rows")

    # ── JOIN 4: Add seller info (seller city, state)
    master = master.merge(
        dataframes["sellers"],
        on="seller_id",
        how="left"
    )
    print(f"  After + sellers        → {master.shape[0]:>7,} rows")

    # ── JOIN 5: Add payment info (payment type, value)
    # Aggregate payments first — one order can have multiple payment methods
    payments_agg = dataframes["payments"].groupby("order_id").agg(
        payment_type    = ("payment_type",  "first"),
        payment_value   = ("payment_value", "sum"),
        payment_installments = ("payment_installments", "max")
    ).reset_index()

    master = master.merge(
        payments_agg,
        on="order_id",
        how="left"
    )
    print(f"  After + payments       → {master.shape[0]:>7,} rows")

    # ── JOIN 6: Add review scores
    # Keep only the most recent review per order
    reviews_clean = dataframes["reviews"].sort_values(
        "review_answer_timestamp", ascending=False
    ).drop_duplicates(subset="order_id")

    master = master.merge(
        reviews_clean[["order_id", "review_score"]],
        on="order_id",
        how="left"
    )
    print(f"  After + reviews        → {master.shape[0]:>7,} rows")

    # ── JOIN 7: Translate product categories to English
    master = master.merge(
        dataframes["translation"],
        on="product_category_name",
        how="left"
    )
    print(f"  After + translation    → {master.shape[0]:>7,} rows")

    # Rename translated column to something cleaner
    master.rename(
        columns={"product_category_name_english": "category"},
        inplace=True
    )

    print(f"\nMaster table complete → {master.shape[0]:,} rows | {master.shape[1]} columns")
    return master


# ── STEP 4: SAVE THE MASTER TABLE ─────────────────────────────────────────────
def save_master_table(master):
    """
    Saves the merged master table to data/processed/
    """
    output_path = "data/processed/master_orders.csv"
    master.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


# ── STEP 5: MAIN RUNNER ────────────────────────────────────────────────────────
def run_ingestion():
    """
    Full pipeline: load → merge → save
    Call this function from app.py or test directly
    """
    dataframes  = load_raw_data()
    master      = build_master_table(dataframes)
    save_master_table(master)
    return master


# ── TEST: Run this file directly to verify ────────────────────────────────────
if __name__ == "__main__":
    master = run_ingestion()

    print("\n── Column List ──────────────────────────────")
    for col in master.columns:
        print(f"  {col}")

    print("\n── Sample Row ───────────────────────────────")
    print(master.head(1).T)  # .T flips it so it's easier to read