# modules/sql_analytics.py
# PURPOSE: Run KPI queries using SQLite (local) and DuckDB (analytical)
# Why two databases?
#   SQLite  = standard relational queries — familiar to all interviewers
#   DuckDB  = columnar analytics engine — faster for aggregations on large data
# Having BOTH on your resume is a strong signal
print("Script started...")
import pandas as pd
import numpy as np
import sqlite3
import duckdb
import os
import warnings
warnings.filterwarnings("ignore")


# ── STEP 1: SETUP SQLITE DATABASE ─────────────────────────────────────────────
def setup_sqlite(df):
    """
    Load the master orders table into a local SQLite database.
    SQLite stores the DB as a single file — easy to version control.
    """

    print("\n── Setting Up SQLite Database ───────────────────────")

    db_path = "data/processed/supply_chain.db"

    # Connect (creates file if it doesn't exist)
    conn = sqlite3.connect(db_path)

    # Write dataframe to SQLite table
    df.to_sql(
        name      = "orders",
        con       = conn,
        if_exists = "replace",   # overwrite if table exists
        index     = False
    )

    # Verify
    cursor = conn.execute("SELECT COUNT(*) FROM orders")
    row_count = cursor.fetchone()[0]
    print(f"  SQLite DB created    : {db_path}")
    print(f"  Rows loaded          : {row_count:,}")

    return conn


# ── STEP 2: CORE KPI QUERIES — SQLITE ─────────────────────────────────────────
def run_kpi_queries(conn):
    """
    Core supply chain KPIs — these are the numbers on Page 2 of the dashboard.
    Every supply chain analyst is expected to know these metrics.
    """

    print("\n── Core KPI Queries (SQLite) ────────────────────────")

    kpis = {}

    # ── KPI 1: Total Orders
    result = pd.read_sql("""
        SELECT COUNT(DISTINCT order_id) AS total_orders
        FROM orders
    """, conn)
    kpis["total_orders"] = int(result["total_orders"].iloc[0])
    print(f"  Total Orders         : {kpis['total_orders']:,}")

    # ── KPI 2: Total Revenue
    result = pd.read_sql("""
        SELECT ROUND(SUM(total_order_value), 2) AS total_revenue
        FROM orders
        WHERE total_order_value > 0
    """, conn)
    kpis["total_revenue"] = float(result["total_revenue"].iloc[0])
    print(f"  Total Revenue        : R$ {kpis['total_revenue']:,.2f}")

    # ── KPI 3: Average Order Value
    result = pd.read_sql("""
        SELECT ROUND(AVG(total_order_value), 2) AS avg_order_value
        FROM orders
        WHERE total_order_value > 0
    """, conn)
    kpis["avg_order_value"] = float(result["avg_order_value"].iloc[0])
    print(f"  Avg Order Value      : R$ {kpis['avg_order_value']:,.2f}")

    # ── KPI 4: On-Time Delivery Rate
    result = pd.read_sql("""
        SELECT
            ROUND(
                100.0 * SUM(CASE WHEN is_late = 0 THEN 1 ELSE 0 END)
                / COUNT(*), 2
            ) AS on_time_rate
        FROM orders
        WHERE delivery_days IS NOT NULL
    """, conn)
    kpis["on_time_delivery_rate"] = float(result["on_time_rate"].iloc[0])
    print(f"  On-Time Delivery %   : {kpis['on_time_delivery_rate']}%")

    # ── KPI 5: Average Delivery Days
    result = pd.read_sql("""
        SELECT ROUND(AVG(delivery_days), 1) AS avg_delivery_days
        FROM orders
        WHERE delivery_days > 0
    """, conn)
    kpis["avg_delivery_days"] = float(result["avg_delivery_days"].iloc[0])
    print(f"  Avg Delivery Days    : {kpis['avg_delivery_days']} days")

    # ── KPI 6: Average Review Score
    result = pd.read_sql("""
        SELECT ROUND(AVG(review_score), 2) AS avg_review
        FROM orders
        WHERE review_score IS NOT NULL
    """, conn)
    kpis["avg_review_score"] = float(result["avg_review"].iloc[0])
    print(f"  Avg Review Score     : {kpis['avg_review_score']} / 5.0")

    # ── KPI 7: Total Unique Customers
    result = pd.read_sql("""
        SELECT COUNT(DISTINCT customer_unique_id) AS unique_customers
        FROM orders
    """, conn)
    kpis["unique_customers"] = int(result["unique_customers"].iloc[0])
    print(f"  Unique Customers     : {kpis['unique_customers']:,}")

    # ── KPI 8: Total Unique Sellers
    result = pd.read_sql("""
        SELECT COUNT(DISTINCT seller_id) AS unique_sellers
        FROM orders
        WHERE seller_id IS NOT NULL
    """, conn)
    kpis["unique_sellers"] = int(result["unique_sellers"].iloc[0])
    print(f"  Unique Sellers       : {kpis['unique_sellers']:,}")

    return kpis


# ── STEP 3: REVENUE ANALYSIS QUERIES ──────────────────────────────────────────
def run_revenue_analysis(conn):
    """
    Revenue breakdown queries — by category, state, and payment type.
    """

    print("\n── Revenue Analysis (SQLite) ────────────────────────")

    # Revenue by category
    revenue_by_category = pd.read_sql("""
        SELECT
            category,
            COUNT(*)                              AS order_count,
            ROUND(SUM(total_order_value), 2)      AS total_revenue,
            ROUND(AVG(total_order_value), 2)      AS avg_order_value,
            ROUND(AVG(review_score), 2)           AS avg_review
        FROM orders
        WHERE category != 'uncategorized'
        GROUP BY category
        ORDER BY total_revenue DESC
        LIMIT 10
    """, conn)
    print(f"  Revenue by category  : {len(revenue_by_category)} categories")

    # Revenue by state
    revenue_by_state = pd.read_sql("""
        SELECT
            customer_state                        AS state,
            COUNT(*)                              AS order_count,
            ROUND(SUM(total_order_value), 2)      AS total_revenue,
            ROUND(AVG(delivery_days), 1)          AS avg_delivery_days
        FROM orders
        WHERE customer_state != 'unknown'
        GROUP BY customer_state
        ORDER BY total_revenue DESC
        LIMIT 10
    """, conn)
    print(f"  Revenue by state     : {len(revenue_by_state)} states")

    # Revenue by payment type
    revenue_by_payment = pd.read_sql("""
        SELECT
            payment_type,
            COUNT(*)                              AS order_count,
            ROUND(SUM(payment_value), 2)          AS total_revenue,
            ROUND(AVG(payment_installments), 1)   AS avg_installments
        FROM orders
        WHERE payment_type != 'unknown'
        GROUP BY payment_type
        ORDER BY total_revenue DESC
    """, conn)
    print(f"  Revenue by payment   : {len(revenue_by_payment)} payment types")

    return {
        "by_category" : revenue_by_category,
        "by_state"    : revenue_by_state,
        "by_payment"  : revenue_by_payment
    }


# ── STEP 4: SUPPLIER PERFORMANCE QUERIES ──────────────────────────────────────
def run_supplier_analysis(conn):
    """
    Supplier performance dashboard queries.
    These map directly to what supply chain teams monitor daily.
    """

    print("\n── Supplier Performance (SQLite) ────────────────────")

    supplier_performance = pd.read_sql("""
        SELECT
            seller_id,
            seller_city,
            seller_state,
            COUNT(*)                                        AS total_orders,
            ROUND(AVG(delivery_days), 1)                   AS avg_delivery_days,
            ROUND(100.0 * SUM(is_late) / COUNT(*), 1)      AS late_rate_pct,
            ROUND(AVG(review_score), 2)                    AS avg_review_score,
            ROUND(SUM(total_order_value), 2)               AS total_revenue,
            ROUND(AVG(price), 2)                           AS avg_price
        FROM orders
        WHERE seller_id IS NOT NULL
        GROUP BY seller_id, seller_city, seller_state
        HAVING total_orders >= 10
        ORDER BY total_orders DESC
        LIMIT 50
    """, conn)

    print(f"  Suppliers analyzed   : {len(supplier_performance):,}")
    print(f"  Avg late rate        : {supplier_performance['late_rate_pct'].mean():.1f}%")
    print(f"  Avg delivery days    : {supplier_performance['avg_delivery_days'].mean():.1f}")

    return supplier_performance


# ── STEP 5: MONTHLY TRENDS — DUCKDB ───────────────────────────────────────────
def run_duckdb_analytics(df):
    """
    DuckDB is faster than SQLite for analytical queries on large DataFrames.
    It runs directly on the DataFrame — no file needed.

    Use DuckDB when:
    - Running window functions (LAG, LEAD, RANK)
    - Aggregating millions of rows
    - Running complex multi-join analytics

    INTERVIEW POINT: DuckDB is increasingly used in modern data stacks
    (alongside dbt, Polars, Arrow). Mentioning it signals you follow
    the data engineering ecosystem.
    """

    print("\n── DuckDB Analytics ─────────────────────────────────")

    # Register DataFrame as a DuckDB table
    con = duckdb.connect()
    con.register("orders", df)

    # Monthly revenue trend with MoM growth
    monthly_trend = con.execute("""
        SELECT
            order_year,
            order_month,
            COUNT(*)                                    AS order_count,
            ROUND(SUM(total_order_value), 2)            AS monthly_revenue,
            ROUND(AVG(delivery_days), 1)                AS avg_delivery_days,
            ROUND(AVG(review_score), 2)                 AS avg_review,

            -- Month over Month growth using LAG window function
            ROUND(
                100.0 * (COUNT(*) - LAG(COUNT(*)) OVER (
                    ORDER BY order_year, order_month
                )) / NULLIF(LAG(COUNT(*)) OVER (
                    ORDER BY order_year, order_month
                ), 0),
            1) AS mom_growth_pct

        FROM orders
        WHERE order_year IS NOT NULL
          AND order_month IS NOT NULL
        GROUP BY order_year, order_month
        ORDER BY order_year, order_month
    """).df()

    print(f"  Monthly trend rows   : {len(monthly_trend)}")

    # Category ranking with RANK()
    category_ranking = con.execute("""
        SELECT
            category,
            COUNT(*)                            AS order_count,
            ROUND(SUM(total_order_value), 2)    AS revenue,
            ROUND(AVG(review_score), 2)         AS avg_review,
            RANK() OVER (
                ORDER BY SUM(total_order_value) DESC
            )                                   AS revenue_rank
        FROM orders
        WHERE category != 'uncategorized'
        GROUP BY category
        ORDER BY revenue_rank
        LIMIT 15
    """).df()

    print(f"  Category rankings    : {len(category_ranking)} categories ranked")

    # State-level delivery performance
    state_delivery = con.execute("""
        SELECT
            customer_state                              AS state,
            COUNT(*)                                    AS order_count,
            ROUND(AVG(delivery_days), 1)                AS avg_delivery_days,
            ROUND(100.0 * SUM(is_late) / COUNT(*), 1)  AS late_rate_pct,
            ROUND(AVG(review_score), 2)                 AS avg_review
        FROM orders
        WHERE customer_state != 'unknown'
          AND delivery_days  > 0
        GROUP BY customer_state
        HAVING COUNT(*) >= 100
        ORDER BY avg_delivery_days DESC
    """).df()

    print(f"  States analyzed      : {len(state_delivery)}")

    con.close()

    return {
        "monthly_trend"    : monthly_trend,
        "category_ranking" : category_ranking,
        "state_delivery"   : state_delivery
    }


# ── MAIN RUNNER ────────────────────────────────────────────────────────────────
def run_sql_analytics(df):
    """
    Full SQL analytics pipeline.
    Input : cleaned delivered orders DataFrame
    Output: dict with all analytics results
    """

    print("\n" + "="*55)
    print("SQL ANALYTICS MODULE")
    print("="*55)

    # SQLite analytics
    conn             = setup_sqlite(df)
    kpis             = run_kpi_queries(conn)
    revenue_analysis = run_revenue_analysis(conn)
    supplier_perf    = run_supplier_analysis(conn)
    conn.close()

    # DuckDB analytics
    duckdb_results   = run_duckdb_analytics(df)

    # Save key outputs
    revenue_analysis["by_category"].to_csv(
        "data/processed/revenue_by_category.csv", index=False
    )
    revenue_analysis["by_state"].to_csv(
        "data/processed/revenue_by_state.csv", index=False
    )
    supplier_perf.to_csv(
        "data/processed/supplier_performance.csv", index=False
    )
    duckdb_results["monthly_trend"].to_csv(
        "data/processed/monthly_trend.csv", index=False
    )
    duckdb_results["category_ranking"].to_csv(
        "data/processed/category_ranking.csv", index=False
    )

    print("\n── Files Saved ──────────────────────────────────────")
    print("  data/processed/revenue_by_category.csv")
    print("  data/processed/revenue_by_state.csv")
    print("  data/processed/supplier_performance.csv")
    print("  data/processed/monthly_trend.csv")
    print("  data/processed/category_ranking.csv")

    results = {
        "kpis"             : kpis,
        "revenue_analysis" : revenue_analysis,
        "supplier_perf"    : supplier_perf,
        "duckdb"           : duckdb_results
    }

    print("\nSQL analytics complete ✅")
    return results


# ── TEST ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading cleaned data...")
    df = pd.read_csv(
        "data/processed/orders_delivered.csv",
        parse_dates=["order_purchase_timestamp"]
    )
    print(f"Loaded: {df.shape[0]:,} rows")

    results = run_sql_analytics(df)

    # Print top 5 categories by revenue
    print("\n── Top 5 Categories by Revenue ─────────────────────")
    top5 = results["revenue_analysis"]["by_category"].head(5)
    for _, row in top5.iterrows():
        print(f"  {row['category']:<30} R$ {row['total_revenue']:>12,.2f}")

    # Print top 3 states
    print("\n── Top 3 States by Revenue ──────────────────────────")
    top3 = results["revenue_analysis"]["by_state"].head(3)
    for _, row in top3.iterrows():
        print(f"  {row['state']:<10} R$ {row['total_revenue']:>12,.2f}")