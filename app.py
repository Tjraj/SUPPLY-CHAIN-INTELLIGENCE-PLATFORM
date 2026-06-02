# app.py
# Supply Chain Intelligence Platform
# Dark UI — matches AI Churn Predictor & AI Data Cleaner aesthetic
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Supply Chain Intelligence Platform",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# DARK THEME CSS — matches Churn Predictor / Data Cleaner style exactly
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0e1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: #0e1117 !important; border-bottom: 1px solid #1e2535; }
.main .block-container { padding: 2rem 2.5rem 2rem 2.5rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1e2535 !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px !important;
    border-radius: 6px !important;
    transition: background 0.2s !important;
    font-size: 0.88rem !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1e2535 !important; color: #e2e8f0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; font-size: 0.82rem !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Metric cards (st.metric) ── */
[data-testid="stMetric"] {
    background: #1a1f2e !important;
    border: 1px solid #1e2535 !important;
    border-radius: 10px !important;
    padding: 1.2rem 1.4rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #8b92a5 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] > div { font-size: 0.78rem !important; font-weight: 500 !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #1a1f2e !important;
    border: 1px solid #1e2535 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #94a3b8 !important; font-size: 0.88rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e2535 !important;
    border: 1px solid #2d3748 !important;
    color: #94a3b8 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #2d3748 !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: #1e2535 !important;
    border: 1px solid #3b82f6 !important;
    color: #3b82f6 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #3b82f6 !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
}

/* ── Selectbox / Slider / File uploader ── */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background: #1a1f2e !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-baseweb="select"] * { color: #e2e8f0 !important; background: #1a1f2e !important; }
[data-baseweb="popover"] { background: #1a1f2e !important; border: 1px solid #2d3748 !important; }
[data-baseweb="option"]:hover { background: #2d3748 !important; }

[data-testid="stSlider"] > div > div > div { background: #3b82f6 !important; }

[data-testid="stFileUploader"] {
    background: #1a1f2e !important;
    border: 1px dashed #2d3748 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #3b82f6 !important; }

/* ── DataFrames / Tables ── */
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    border: 1px solid #1e2535 !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] > div {
    border-radius: 8px !important;
}

/* ── Alerts ── */
[data-testid="stSuccess"] { background: #0d2b1e !important; border: 1px solid #16a34a !important; border-radius: 8px !important; color: #4ade80 !important; }
[data-testid="stError"]   { background: #2b0d0d !important; border: 1px solid #dc2626 !important; border-radius: 8px !important; color: #f87171 !important; }
[data-testid="stWarning"] { background: #2b2000 !important; border: 1px solid #d97706 !important; border-radius: 8px !important; color: #fbbf24 !important; }
[data-testid="stInfo"]    { background: #0d1f2b !important; border: 1px solid #0ea5e9 !important; border-radius: 8px !important; color: #38bdf8 !important; }

/* ── Code blocks ── */
[data-testid="stCode"] { background: #161b27 !important; border: 1px solid #1e2535 !important; border-radius: 8px !important; }
code { color: #7dd3fc !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Plotly charts background ── */
.js-plotly-plot, .plotly { background: transparent !important; }
[data-testid="stPlotlyChart"] { background: #1a1f2e !important; border-radius: 10px !important; border: 1px solid #1e2535 !important; padding: 4px !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: #3b82f6 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom-color: #3b82f6 !important;
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

/* ── Select slider ── */
[data-testid="stSlider"] { color: #e2e8f0 !important; }

/* hide default streamlit branding ── */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
[data-testid="stHeader"] { height: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS — section headers, stat cards, dividers
# ══════════════════════════════════════════════════════════════════════════════

def section_header(num: str, title: str):
    """Numbered section header like '01  UPLOAD ──────'"""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1.8rem 0 1rem 0;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                   font-weight:600;color:#3b82f6;letter-spacing:0.05em;">{num}</span>
      <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.18em;
                   color:#94a3b8;text-transform:uppercase;">{title}</span>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,#1e2535,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)


def page_title(icon: str, title: str, subtitle: str = ""):
    """Large page title with optional subtitle"""
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
      <div style="display:inline-block;background:linear-gradient(135deg,#1e293b,#0f172a);
                  border:1px solid #1e2535;border-radius:12px;
                  padding:1.4rem 2rem;width:100%;box-sizing:border-box;">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:0.3rem;">
          <span style="font-size:1.8rem;">{icon}</span>
          <span style="font-size:1.6rem;font-weight:700;color:#f1f5f9;
                       font-family:'Inter',sans-serif;letter-spacing:-0.02em;">{title}</span>
        </div>
        {"" if not subtitle else f'<p style="margin:0;color:#64748b;font-size:0.88rem;padding-left:3.2rem;">{subtitle}</p>'}
      </div>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label: str, value: str, color: str = "#f1f5f9", border_color: str = "#3b82f6"):
    """Single stat card with colored top border and value"""
    return f"""
    <div style="background:#1a1f2e;border:1px solid #1e2535;border-radius:10px;
                padding:1.2rem 1.4rem;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{border_color};"></div>
      <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;
                  text-transform:uppercase;color:#8b92a5;margin-bottom:0.5rem;">{label}</div>
      <div style="font-size:1.9rem;font-weight:700;color:{color};
                  font-family:'JetBrains Mono',monospace;line-height:1.1;">{value}</div>
    </div>
    """


def render_stat_row(metrics: list):
    """
    metrics = list of (label, value, color, border_color)
    color/border_color optional — defaults to white/blue
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        label = m[0]; value = m[1]
        color  = m[2] if len(m) > 2 else "#f1f5f9"
        border = m[3] if len(m) > 3 else "#3b82f6"
        with col:
            st.markdown(stat_card(label, value, color, border), unsafe_allow_html=True)


def dark_plotly(fig, height=420):
    """Apply dark theme to any plotly figure."""
    fig.update_layout(
        paper_bgcolor="#1a1f2e",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#94a3b8", family="Inter"),
        height=height,
        margin=dict(l=16, r=16, t=40, b=16),
        legend=dict(
            bgcolor="#111827",
            bordercolor="#2d3748",
            borderwidth=1,
            font=dict(color="#94a3b8", size=11),
        ),
        xaxis=dict(
            gridcolor="#1e2535",
            linecolor="#2d3748",
            tickfont=dict(color="#64748b", size=10),
            title_font=dict(color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="#1e2535",
            linecolor="#2d3748",
            tickfont=dict(color="#64748b", size=10),
            title_font=dict(color="#64748b"),
        ),
    )
    fig.update_traces(marker_line_width=0)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN MAPPING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS = {
    "order_id"                  : "Order ID (unique identifier per order)",
    "order_purchase_timestamp"  : "Order Date / Purchase Timestamp",
    "price"                     : "Item Price / Unit Price",
    "freight_value"             : "Shipping / Freight Cost",
    "order_delivered_date"      : "Actual Delivery Date",
    "order_estimated_date"      : "Estimated Delivery Date",
    "customer_state"            : "Customer Region / State / City",
    "seller_id"                 : "Seller / Supplier ID",
    "seller_city"               : "Seller City (optional)",
    "category"                  : "Product Category",
    "payment_type"              : "Payment Method",
    "payment_value"             : "Total Payment Amount",
    "review_score"              : "Customer Rating / Review Score",
}

OPTIONAL_FIELDS = ["seller_city", "review_score", "payment_type", "payment_value"]


def apply_column_mapping(df_raw, mapping):
    df = pd.DataFrame()
    for internal_col, user_col in mapping.items():
        if user_col and user_col != "-- Not Available --" and user_col in df_raw.columns:
            df[internal_col] = df_raw[user_col].values
        elif internal_col not in OPTIONAL_FIELDS:
            df[internal_col] = None

    for date_col in ["order_purchase_timestamp", "order_delivered_date", "order_estimated_date"]:
        if date_col in df.columns and df[date_col] is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if "order_delivered_date" in df.columns and "order_purchase_timestamp" in df.columns:
        df["delivery_days"] = (df["order_delivered_date"] - df["order_purchase_timestamp"]).dt.days
        df.loc[df["delivery_days"] <= 0, "delivery_days"] = np.nan

    if "order_delivered_date" in df.columns and "order_estimated_date" in df.columns:
        df["is_late"]    = (df["order_delivered_date"] > df["order_estimated_date"]).astype(float)
        df["delay_days"] = (df["order_delivered_date"] - df["order_estimated_date"]).dt.days

    for col in ["price", "freight_value", "payment_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    if "price" in df.columns and "freight_value" in df.columns:
        df["total_order_value"] = df["price"] + df["freight_value"]
    elif "price" in df.columns:
        df["total_order_value"] = df["price"]
    elif "payment_value" in df.columns:
        df["total_order_value"] = df["payment_value"]

    for col in ["category", "customer_state", "payment_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()

    if "review_score" in df.columns:
        df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")

    if "seller_id" not in df.columns or df["seller_id"].isna().all():
        df["seller_id"] = "unknown_seller"

    if "customer_unique_id" not in df.columns:
        df["customer_unique_id"] = df["order_id"] if "order_id" in df.columns else range(len(df))

    if "order_purchase_timestamp" in df.columns:
        df["order_year"]  = df["order_purchase_timestamp"].dt.year
        df["order_month"] = df["order_purchase_timestamp"].dt.month
        df["order_date"]  = df["order_purchase_timestamp"].dt.date

    return df


def show_column_mapper(df_raw):
    section_header("03", "Map Your Columns")
    st.markdown("""
    <p style="color:#64748b;font-size:0.88rem;margin-bottom:1rem;">
    Map your CSV columns to the platform's standard fields.
    All 6 pages will then work fully with your data.
    </p>
    """, unsafe_allow_html=True)

    csv_cols = ["-- Not Available --"] + list(df_raw.columns)

    def best_guess(internal_name, csv_columns):
        keywords = {
            "order_id"                 : ["order_id","orderid","order","id","transaction"],
            "order_purchase_timestamp" : ["date","purchase","timestamp","order_date","created"],
            "price"                    : ["price","unit_price","item_price","amount","cost"],
            "freight_value"            : ["freight","shipping","delivery_cost","ship"],
            "order_delivered_date"     : ["delivered","delivery_date","actual_delivery","received"],
            "order_estimated_date"     : ["estimated","expected","planned","eta"],
            "customer_state"           : ["state","region","city","location","customer_state","area"],
            "seller_id"                : ["seller","supplier","vendor","seller_id","supplier_id"],
            "seller_city"              : ["seller_city","supplier_city","origin"],
            "category"                 : ["category","product_category","type","class","segment"],
            "payment_type"             : ["payment","payment_type","method","pay_method"],
            "payment_value"            : ["payment_value","total","amount_paid","revenue"],
            "review_score"             : ["review","rating","score","stars","feedback"],
        }
        hints = keywords.get(internal_name, [])
        for col in csv_columns:
            col_lower = col.lower().replace(" ", "_")
            for hint in hints:
                if hint in col_lower or col_lower in hint:
                    return col
        return "-- Not Available --"

    col1, col2 = st.columns(2)
    fields = list(REQUIRED_FIELDS.items())
    half   = len(fields) // 2

    for i, (internal_col, description) in enumerate(fields):
        guess  = best_guess(internal_col, df_raw.columns.tolist())
        is_opt = internal_col in OPTIONAL_FIELDS
        label  = f"{'⚪ ' if is_opt else '🔴 '}{description}"
        target = col1 if i < half else col2
        with target:
            st.selectbox(label, options=csv_cols,
                         index=csv_cols.index(guess) if guess in csv_cols else 0,
                         key=f"map_{internal_col}")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    mapping = {
        ic: (st.session_state.get(f"map_{ic}")
             if st.session_state.get(f"map_{ic}") != "-- Not Available --" else None)
        for ic in REQUIRED_FIELDS
    }
    missing_required = [
        REQUIRED_FIELDS[f] for f in REQUIRED_FIELDS
        if f not in OPTIONAL_FIELDS and not mapping.get(f)
    ]

    if missing_required:
        st.warning(f"⚠️ Please map required fields: {', '.join(missing_required[:3])}")

    c1, c2 = st.columns(2)
    with c1:
        apply_clicked = st.button("✅ Apply Mapping & Load Dashboard",
                                  type="primary", use_container_width=True,
                                  disabled=bool(missing_required), key="apply_mapping_btn")
    with c2:
        reset_clicked = st.button("🔄 Reset & Use Olist Dataset",
                                  use_container_width=True, key="reset_mapping_btn")

    if reset_clicked:
        for k in ["uploaded_df","mapping","mapped_df","pending_mapping","dataset_name"]:
            st.session_state.pop(k, None)
        st.rerun()

    if apply_clicked and not missing_required:
        final_mapping = {
            ic: (st.session_state.get(f"map_{ic}")
                 if st.session_state.get(f"map_{ic}") != "-- Not Available --" else None)
            for ic in REQUIRED_FIELDS
        }
        with st.spinner("Processing your data..."):
            try:
                mapped = apply_column_mapping(df_raw, final_mapping)
                st.session_state["mapped_df"] = mapped
                st.session_state["mapping"]   = final_mapping
                st.success(f"✅ Mapped — {len(mapped):,} rows ready.")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Mapping error: {e}")


# ── DATA LOADERS ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_olist():
    """Load and cache Olist dataset — runs once, cached for all users."""
    try:
        return pd.read_csv(
            "data/processed/orders_delivered.csv",
            parse_dates=["order_purchase_timestamp"]
        )
    except FileNotFoundError:
        return None

def get_active_df():
    # Return custom uploaded dataset if present
    if "mapped_df" in st.session_state and st.session_state["mapped_df"] is not None:
        return st.session_state["mapped_df"]
    # Load Olist with caching — fast for all pages, all users
    return _load_olist()


def load_analytics():
    files = {
        "monthly_trend"   : "data/processed/monthly_trend.csv",
        "revenue_category": "data/processed/revenue_by_category.csv",
        "revenue_state"   : "data/processed/revenue_by_state.csv",
        "supplier_perf"   : "data/processed/supplier_performance.csv",
        "anomalies"       : "data/processed/anomalies.csv",
        "monthly_spikes"  : "data/processed/monthly_spikes.csv",
        "forecasts"       : "data/processed/forecasts.csv",
    }
    data = {}
    for key, path in files.items():
        try:    data[key] = pd.read_csv(path)
        except: data[key] = pd.DataFrame()
    return data


def is_olist(df):
    return "mapped_df" not in st.session_state or st.session_state["mapped_df"] is None


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # App branding
        st.markdown("""
        <div style="padding:1.2rem 0.5rem 0.5rem 0.5rem;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="font-size:1.4rem;">🚚</span>
            <span style="font-size:1.05rem;font-weight:700;color:#f1f5f9;
                         font-family:'Inter',sans-serif;letter-spacing:-0.01em;">
              Supply Chain
            </span>
          </div>
          <div style="padding-left:2.4rem;">
            <span style="font-size:0.65rem;font-weight:700;letter-spacing:0.18em;
                         color:#3b82f6;text-transform:uppercase;">
              Intelligence Platform
            </span>
          </div>
        </div>
        <hr style="border:none;border-top:1px solid #1e2535;margin:0.8rem 0;">
        """, unsafe_allow_html=True)

        page = st.radio("Navigate", options=[
            "📁  Setup & Data Upload",
            "📊  Executive KPI Dashboard",
            "📈  Demand Forecasting",
            "🚨  Anomaly Detection",
            "🏭  Supplier Performance",
            "🤖  AI Report & Download"
        ], label_visibility="collapsed")

        st.markdown("<hr style='border:none;border-top:1px solid #1e2535;margin:0.8rem 0;'>",
                    unsafe_allow_html=True)

        if "mapped_df" in st.session_state and st.session_state["mapped_df"] is not None:
            name = st.session_state.get("dataset_name", "Custom Dataset")
            rows = len(st.session_state["mapped_df"])
            st.markdown(f"""
            <div style="background:#0d2b1e;border:1px solid #16a34a;border-radius:8px;
                        padding:0.8rem 1rem;margin-bottom:0.8rem;">
              <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                          text-transform:uppercase;color:#4ade80;margin-bottom:3px;">
                Active Dataset
              </div>
              <div style="font-size:0.82rem;color:#f1f5f9;font-weight:600;
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                📂 {name}
              </div>
              <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">
                {rows:,} rows loaded
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Switch Dataset", use_container_width=True, key="sw_btn"):
                for k in ["mapped_df","uploaded_df","mapping","dataset_name"]:
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            st.markdown("""
            <div style="background:#0d1f2b;border:1px solid #0ea5e9;border-radius:8px;
                        padding:0.8rem 1rem;margin-bottom:0.8rem;">
              <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                          text-transform:uppercase;color:#38bdf8;margin-bottom:3px;">
                Active Dataset
              </div>
              <div style="font-size:0.82rem;color:#f1f5f9;font-weight:600;">
                📦 Olist Dataset
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border:none;border-top:1px solid #1e2535;margin:0.8rem 0;'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.65rem;color:#334155;text-align:center;
                    letter-spacing:0.08em;padding-bottom:0.5rem;">
          XGBoost · Prophet · IsoForest · OpenRouter
        </div>
        """, unsafe_allow_html=True)

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SETUP & DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def page_setup():
    page_title("📁", "Setup & Data Upload",
               "Use the built-in Olist dataset or upload any supply chain CSV")

    section_header("01", "Choose Data Source")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid #1e2535;border-radius:10px;
                    padding:1.4rem;margin-bottom:1rem;">
          <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.14em;
                      text-transform:uppercase;color:#3b82f6;margin-bottom:0.6rem;">
            Built-in Dataset
          </div>
          <div style="font-size:0.88rem;color:#94a3b8;line-height:1.6;">
            📦 110,197 delivered orders<br>
            🗓️ 2016 – 2018 · 45 features<br>
            ⚡ Pre-computed analytics (fast)
          </div>
        </div>
        """, unsafe_allow_html=True)
        df_olist = None
        try:
            df_olist = pd.read_csv("data/processed/orders_delivered.csv")
        except FileNotFoundError:
            pass
        if df_olist is not None:
            if st.button("▶  Load Olist Dataset", type="primary", use_container_width=True):
                for k in ["mapped_df","uploaded_df","mapping","dataset_name"]:
                    st.session_state.pop(k, None)
                st.success("✅ Olist dataset active")
                st.rerun()
        else:
            st.error("❌ Run: python modules/clean.py")

    with c2:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid #1e2535;border-radius:10px;
                    padding:1.4rem;margin-bottom:1rem;">
          <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.14em;
                      text-transform:uppercase;color:#06b6d4;margin-bottom:0.6rem;">
            Upload Your CSV
          </div>
          <div style="font-size:0.88rem;color:#94a3b8;">
            Upload any supply chain CSV and map your columns
            to activate all 6 dashboard pages with your data.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1a1500;border:1px solid #854d0e;border-radius:8px;
                    padding:0.7rem 1rem;margin-bottom:0.8rem;">
          <span style="font-size:0.78rem;color:#fbbf24;">
            ⚠️ <b>Session Notice:</b> Uploaded data is stored in browser session memory.
            It will be lost if you refresh the page or reconnect.
            Re-upload your CSV if that happens.
          </span>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"],
                                    label_visibility="collapsed")
        if uploaded:
            try:
                import io
                bytes_data = uploaded.read()
                # Try UTF-8 first, fallback to latin-1 to handle encoding issues
                try:
                    df_raw = pd.read_csv(io.BytesIO(bytes_data), encoding="utf-8")
                except UnicodeDecodeError:
                    df_raw = pd.read_csv(io.BytesIO(bytes_data), encoding="latin-1")
                if st.session_state.get("dataset_name") != uploaded.name:
                    st.session_state["uploaded_df"]  = df_raw
                    st.session_state["dataset_name"] = uploaded.name
                    st.session_state.pop("mapped_df", None)
                    st.session_state.pop("mapping", None)
                st.success(f"✅ **{uploaded.name}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
                if "mapped_df" not in st.session_state:
                    st.info("👇 Map your columns below")
            except Exception as e:
                st.error(f"❌ {e}")

    if "uploaded_df" in st.session_state and (
        "mapped_df" not in st.session_state or st.session_state["mapped_df"] is None
    ):
        show_column_mapper(st.session_state["uploaded_df"])

    df = get_active_df()
    if df is not None:
        section_header("02", "Dataset Preview")
        active_label = (
            st.session_state.get("dataset_name", "Custom")
            if "mapped_df" in st.session_state and st.session_state["mapped_df"] is not None
            else "Olist"
        )
        missing_pct = f"{df.isnull().sum().sum() / (df.shape[0]*df.shape[1])*100:.1f}%"
        render_stat_row([
            ("Rows",        f"{len(df):,}",       "#f1f5f9", "#3b82f6"),
            ("Columns",     f"{df.shape[1]}",      "#f1f5f9", "#06b6d4"),
            ("Missing",     missing_pct,           "#fbbf24", "#d97706"),
            ("Memory (MB)", f"{df.memory_usage(deep=True).sum()/1024**2:.1f}", "#f1f5f9", "#8b5cf6"),
        ])
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        with st.expander(f"▶  Preview — {active_label} (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)
        with st.expander("▶  Column Details"):
            info = pd.DataFrame({
                "Column"  : df.columns,
                "Type"    : df.dtypes.values,
                "Non-Null": df.count().values,
                "Nulls"   : df.isnull().sum().values,
            })
            st.dataframe(info, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — KPI DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_kpi_dashboard():
    page_title("📊", "Executive KPI Dashboard",
               "Real-time supply chain performance at a glance")

    df = get_active_df()
    if df is None:
        st.error("❌ No data loaded. Go to Setup page first.")
        return

    def sm(col):
        if col not in df.columns: return None
        v = df[col].mean()
        return v if pd.notna(v) else None
    def ss(col):
        if col not in df.columns: return None
        v = df[col].sum()
        return v if pd.notna(v) else None
    def sn(col):
        return df[col].nunique() if col in df.columns else None

    section_header("01", "Core KPIs")

    total_revenue = ss("total_order_value")
    avg_val       = sm("total_order_value")
    late_mean     = sm("is_late")
    on_time       = (1 - late_mean) * 100 if late_mean is not None else None

    render_stat_row([
        ("Total Orders",     f"{sn('order_id'):,}" if sn('order_id') else "N/A",
         "#f1f5f9", "#3b82f6"),
        ("Total Revenue",    f"R${total_revenue/1e6:.2f}M" if total_revenue else "N/A",
         "#4ade80", "#16a34a"),
        ("Avg Order Value",  f"R${avg_val:.2f}" if avg_val else "N/A",
         "#f1f5f9", "#06b6d4"),
        ("On-Time Delivery", f"{on_time:.1f}%" if on_time else "N/A",
         "#4ade80" if on_time and on_time >= 80 else "#f87171", "#16a34a"),
    ])
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    avg_del   = sm("delivery_days")
    avg_rev   = sm("review_score")
    n_buyers  = sn("customer_unique_id")
    n_sellers = sn("seller_id")

    render_stat_row([
        ("Avg Delivery Days",  f"{avg_del:.1f}d" if avg_del else "N/A",
         "#fbbf24", "#d97706"),
        ("Avg Review Score",   f"{avg_rev:.2f} / 5.0" if avg_rev else "N/A",
         "#c084fc", "#8b5cf6"),
        ("Unique Customers",   f"{n_buyers:,}" if n_buyers else "N/A",
         "#f1f5f9", "#3b82f6"),
        ("Active Sellers",     f"{n_sellers:,}" if n_sellers else "N/A",
         "#f1f5f9", "#06b6d4"),
    ])

    if "order_purchase_timestamp" in df.columns and "total_order_value" in df.columns:
        section_header("02", "Monthly Revenue & Orders")
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        monthly = (
            df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))
            .agg(order_count=("order_id","count"), monthly_revenue=("total_order_value","sum"))
            .reset_index()
        )
        monthly["order_purchase_timestamp"] = monthly["order_purchase_timestamp"].dt.to_timestamp()
        monthly["month_label"] = monthly["order_purchase_timestamp"].dt.strftime("%Y-%m")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Monthly Revenue (R$)", "Order Volume"),
                            vertical_spacing=0.10)
        fig.add_trace(go.Bar(x=monthly["month_label"], y=monthly["monthly_revenue"],
                             name="Revenue", marker_color="#3b82f6",
                             marker_opacity=0.85), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly["month_label"], y=monthly["order_count"],
                                 name="Orders", mode="lines+markers",
                                 line=dict(color="#06b6d4", width=2),
                                 marker=dict(size=4)), row=2, col=1)
        dark_plotly(fig, height=460)
        fig.update_layout(
            annotations=[
                dict(text="Monthly Revenue (R$)", x=0.5, y=1.02, xref="paper",
                     yref="paper", showarrow=False, font=dict(color="#64748b", size=11)),
                dict(text="Order Volume", x=0.5, y=0.44, xref="paper",
                     yref="paper", showarrow=False, font=dict(color="#64748b", size=11)),
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

    section_header("03", "Category & Region Breakdown")
    a, b = st.columns(2)

    if "category" in df.columns and "total_order_value" in df.columns:
        with a:
            cat_rev = (df.groupby("category")["total_order_value"]
                       .sum().reset_index()
                       .sort_values("total_order_value", ascending=True)
                       .tail(10))
            fig = px.bar(cat_rev, x="total_order_value", y="category",
                         orientation="h", title="Top 10 Categories by Revenue",
                         color="total_order_value",
                         color_continuous_scale=["#1e3a5f","#3b82f6","#06b6d4"])
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                               title_font=dict(color="#94a3b8", size=12))
            dark_plotly(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

    if "customer_state" in df.columns and "total_order_value" in df.columns:
        with b:
            state_rev = (df.groupby("customer_state")["total_order_value"]
                         .sum().reset_index()
                         .sort_values("total_order_value", ascending=False)
                         .head(10))
            fig = px.pie(state_rev, values="total_order_value",
                         names="customer_state", title="Revenue by Region",
                         hole=0.5,
                         color_discrete_sequence=["#3b82f6","#06b6d4","#8b5cf6",
                                                  "#f59e0b","#10b981","#f43f5e",
                                                  "#a78bfa","#34d399","#fbbf24","#fb7185"])
            fig.update_traces(textfont_color="white", textfont_size=10)
            fig.update_layout(title_font=dict(color="#94a3b8", size=12))
            dark_plotly(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

    section_header("04", "Payment & Delivery")
    a, b = st.columns(2)

    if "payment_type" in df.columns:
        with a:
            pay = df["payment_type"].value_counts().reset_index()
            pay.columns = ["payment_type","count"]
            fig = px.pie(pay, values="count", names="payment_type",
                         title="Payment Methods", hole=0.45,
                         color_discrete_sequence=["#3b82f6","#06b6d4","#8b5cf6","#f59e0b","#10b981"])
            fig.update_traces(textfont_color="white", textfont_size=10)
            fig.update_layout(title_font=dict(color="#94a3b8", size=12))
            dark_plotly(fig, 360)
            st.plotly_chart(fig, use_container_width=True)

    if "delivery_days" in df.columns:
        with b:
            d_vals = df["delivery_days"].dropna()
            fig = px.histogram(d_vals, nbins=40, title="Delivery Days Distribution",
                               color_discrete_sequence=["#3b82f6"])
            fig.update_traces(marker_line_width=0, opacity=0.8)
            fig.add_vline(x=float(d_vals.mean()), line_dash="dash",
                          line_color="#f59e0b", line_width=2,
                          annotation_text=f"Avg {d_vals.mean():.1f}d",
                          annotation_font_color="#f59e0b")
            fig.update_layout(title_font=dict(color="#94a3b8", size=12))
            dark_plotly(fig, 360)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
def page_forecasting():
    page_title("📈", "Demand Forecasting",
               "Moving Average · Exponential Smoothing · Prophet")

    df = get_active_df()
    an = load_analytics()

    if df is None:
        st.error("❌ No data loaded."); return
    if "order_purchase_timestamp" not in df.columns:
        st.error("❌ No date column found."); return

    section_header("01", "Forecast Settings")
    a, b = st.columns(2)
    with a:
        periods = st.slider("Forecast Horizon (months)", 1, 12, 6)
    with b:
        method = st.selectbox("Highlight Method",
                              ["Moving Average","Exponential Smoothing","Prophet"])

    section_header("02", "Historical + Forecast")

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    hist = (df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["order_id"]
            .count().reset_index())
    hist.columns = ["month","order_count"]
    hist["month"] = hist["month"].dt.to_timestamp()

    # ── Minimum data guard ────────────────────────────────────────────────
    n_months = len(hist)
    if n_months < 3:
        st.markdown(f"""
        <div style="background:#2b1a00;border:1px solid #d97706;border-radius:10px;
                    padding:1.2rem 1.6rem;margin:1rem 0;">
          <div style="font-size:0.8rem;font-weight:700;color:#fbbf24;
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">
            Insufficient Data for Forecasting
          </div>
          <div style="font-size:0.88rem;color:#94a3b8;line-height:1.6;">
            Your dataset contains only <b style="color:#f59e0b">{n_months} month(s)</b>
            of order data. Forecasting requires a minimum of
            <b style="color:#f59e0b">3 months</b> of historical data to produce
            reliable predictions.<br><br>
            <b>What you can do:</b><br>
            - Upload a dataset with more historical order dates<br>
            - Ensure the Order Date column is correctly mapped<br>
            - Check that date parsing succeeded (Setup page > Column Details)
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if n_months < 6:
        st.markdown(f"""
        <div style="background:#1a1f2e;border:1px solid #d97706;border-radius:8px;
                    padding:0.8rem 1.2rem;margin-bottom:1rem;">
          <span style="font-size:0.8rem;color:#fbbf24;">
            ⚠️ Only {n_months} months of data available.
            Forecast accuracy improves with more history (12+ months recommended).
          </span>
        </div>
        """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["order_count"],
                             name="Historical", mode="lines+markers",
                             line=dict(color="#3b82f6", width=2),
                             marker=dict(size=4, color="#3b82f6"),
                             fill="tozeroy",
                             fillcolor="rgba(59,130,246,0.08)"))

    if is_olist(df) and not an["forecasts"].empty:
        fc = an["forecasts"].copy()
        fc["ds"] = pd.to_datetime(fc["ds"])
        colors = {"Moving Average":"#f59e0b",
                  "Exponential Smoothing":"#10b981",
                  "Prophet":"#f43f5e"}
        for mname, color in colors.items():
            mdf = fc[fc["method"] == mname].head(periods)
            if len(mdf):
                fig.add_trace(go.Scatter(
                    x=mdf["ds"], y=mdf["yhat"], name=mname,
                    mode="lines+markers",
                    line=dict(color=color, width=3 if mname==method else 1.5, dash="dot"),
                    marker=dict(size=5)))
    else:
        values   = hist["order_count"].values
        extended = list(values)
        fdates, fvals = [], []
        last = hist["month"].iloc[-1]
        for i in range(periods):
            p = np.mean(extended[-3:])
            extended.append(p)
            fdates.append(last + pd.DateOffset(months=i+1))
            fvals.append(round(p, 0))
        fig.add_trace(go.Scatter(x=fdates, y=fvals, name="MA Forecast",
                                 mode="lines+markers",
                                 line=dict(color="#f59e0b", width=2, dash="dot"),
                                 marker=dict(size=5)))
        alpha    = 0.3
        smoothed = [values[0]]
        for t in range(1, len(values)):
            smoothed.append(alpha * values[t] + (1-alpha) * smoothed[-1])
        es_val   = smoothed[-1]
        es_dates = [last + pd.DateOffset(months=i+1) for i in range(periods)]
        fig.add_trace(go.Scatter(x=es_dates, y=[round(es_val,0)]*periods,
                                 name="ES Forecast",
                                 mode="lines+markers",
                                 line=dict(color="#10b981", width=2, dash="dot"),
                                 marker=dict(size=5)))

    last_date = hist["month"].max()
    max_y     = int(hist["order_count"].max() * 1.15)
    fig.add_vrect(x0=last_date, x1=last_date + pd.DateOffset(months=periods),
                  fillcolor="rgba(59,130,246,0.04)", line_width=0,
                  annotation_text="Forecast Zone", annotation_position="top left",
                  annotation_font_color="#3b82f6", annotation_font_size=10)
    fig.add_vline(x=last_date, line_dash="dash", line_color="#64748b",
                  line_width=1)
    fig.update_layout(xaxis_title="Month", yaxis_title="Orders",
                      hovermode="x unified")
    dark_plotly(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    if "category" in df.columns:
        section_header("03", "Category-Level Forecast")
        top_cats = df["category"].value_counts().head(8).index.tolist()
        cat      = st.selectbox("Select Category", top_cats)
        cat_df   = df[df["category"] == cat].copy()
        cm = (cat_df.groupby(cat_df["order_purchase_timestamp"].dt.to_period("M"))
              ["order_id"].count().reset_index())
        cm.columns = ["month","order_count"]
        cm["month"] = cm["month"].dt.to_timestamp()

        ext = list(cm["order_count"].values)
        fdates2, fvals2 = [], []
        last2 = cm["month"].iloc[-1]
        for i in range(periods):
            p = np.mean(ext[-3:])
            ext.append(p)
            fdates2.append(last2 + pd.DateOffset(months=i+1))
            fvals2.append(round(p, 0))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cm["month"], y=cm["order_count"],
                                  name="Historical", mode="lines+markers",
                                  line=dict(color="#3b82f6", width=2),
                                  fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
        fig2.add_trace(go.Scatter(x=fdates2, y=fvals2, name="MA Forecast",
                                  mode="lines+markers",
                                  line=dict(color="#f59e0b", width=2, dash="dot")))
        fig2.update_layout(title=f"Forecast — {cat.title()}",
                           title_font=dict(color="#94a3b8", size=13))
        dark_plotly(fig2, 380)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def page_anomalies():
    page_title("🚨", "Anomaly Detection",
               "Isolation Forest · Statistical Control Limits")

    df = get_active_df()
    an = load_analytics()

    if df is None:
        st.error("❌ No data loaded."); return

    if is_olist(df) and not an["anomalies"].empty:
        anom = an["anomalies"]

        section_header("01", "Anomaly Summary")
        render_stat_row([
            ("Total Orders",      f"{len(df):,}",       "#f1f5f9", "#3b82f6"),
            ("Anomalies Flagged", f"{len(anom):,}",     "#f87171", "#dc2626"),
            ("Price Spikes",
             f"{int(anom['is_price_spike'].sum()) if 'is_price_spike' in anom.columns else 0:,}",
             "#fbbf24", "#d97706"),
            ("Severe Delays",
             f"{int(anom['is_severe_delay'].sum()) if 'is_severe_delay' in anom.columns else 0:,}",
             "#f87171", "#dc2626"),
        ])

        section_header("02", "Price vs Delivery Anomaly Map")
        plot = anom[["price","delivery_days","category",
                     "customer_state","anomaly_score"]].dropna().head(2000)
        fig = px.scatter(plot, x="delivery_days", y="price",
                         color="anomaly_score",
                         color_continuous_scale=["#dc2626","#f59e0b","#16a34a"],
                         hover_data=["category","customer_state"],
                         opacity=0.7)
        fig.update_traces(marker=dict(size=5))
        dark_plotly(fig, 480)
        st.plotly_chart(fig, use_container_width=True)

        section_header("03", "Monthly Volume Control Chart")
        if not an["monthly_spikes"].empty:
            sp = an["monthly_spikes"].copy()
            sp["month"] = pd.to_datetime(sp["month"])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=sp["month"], y=sp["upper_threshold"],
                                      name="Upper +2σ", mode="lines",
                                      line=dict(color="#dc2626", dash="dash", width=1),
                                      fill=None))
            fig2.add_trace(go.Scatter(x=sp["month"], y=sp["lower_threshold"],
                                      name="Lower -2σ", mode="lines",
                                      line=dict(color="#d97706", dash="dash", width=1),
                                      fill="tonexty",
                                      fillcolor="rgba(217,119,6,0.06)"))
            fig2.add_trace(go.Scatter(x=sp["month"], y=sp["order_count"],
                                      name="Order Volume", mode="lines+markers",
                                      line=dict(color="#3b82f6", width=2),
                                      marker=dict(size=5)))
            dark_plotly(fig2, 380)
            st.plotly_chart(fig2, use_container_width=True)

        section_header("04", "Top Anomalous Orders")
        cols = [c for c in ["order_id","price","freight_value","delivery_days",
                             "customer_state","category","anomaly_score"]
                if c in anom.columns]
        top_anom = anom[cols].sort_values("anomaly_score").head(20).reset_index(drop=True)
        st.dataframe(top_anom, use_container_width=True)

    else:
        
        num_cols     = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in ["price","freight_value","delivery_days",
                                    "total_order_value","payment_value"] if c in num_cols]
        if len(feature_cols) < 2:
            feature_cols = num_cols[:5]
        if not feature_cols:
            st.warning("⚠️ No numeric columns found."); return

        # ── Progress bar for large datasets ──────────────────────────────
        n_rows = len(df)
        bar_placeholder = st.empty()
        status_placeholder = st.empty()

        with bar_placeholder:
            progress = st.progress(0, text="Preparing data for anomaly detection...")

        status_placeholder.markdown(
            f'''<p style="color:#64748b;font-size:0.82rem;">
            Analysing <b style="color:#3b82f6">{n_rows:,} rows</b> across
            <b style="color:#3b82f6">{len(feature_cols)} features</b> —
            this may take a few seconds for large datasets.</p>''',
            unsafe_allow_html=True
        )

        progress.progress(15, text="Filling missing values...")
        df_model = df[feature_cols].fillna(df[feature_cols].median())

        progress.progress(35, text="Initialising Isolation Forest model...")
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=100, contamination=0.05,
                              random_state=42, n_jobs=-1)

        progress.progress(55, text=f"Training model on {n_rows:,} rows...")
        preds  = iso.fit_predict(df_model)

        progress.progress(80, text="Scoring anomalies...")
        scores = iso.score_samples(df_model)

        progress.progress(95, text="Building results...")
        df_work = df.copy()
        df_work["is_anomaly"]    = (preds == -1).astype(int)
        df_work["anomaly_score"] = scores
        anom_live = df_work[df_work["is_anomaly"] == 1]

        progress.progress(100, text="Done!")
        import time; time.sleep(0.4)
        bar_placeholder.empty()
        status_placeholder.empty()

        section_header("01", "Live Detection Results")
        render_stat_row([
            ("Total Records",     f"{len(df):,}",           "#f1f5f9", "#3b82f6"),
            ("Anomalies Flagged", f"{len(anom_live):,}",    "#f87171", "#dc2626"),
            ("Anomaly Rate",      "5.0%",                   "#fbbf24", "#d97706"),
            ("Features Used",     f"{len(feature_cols)}",   "#f1f5f9", "#8b5cf6"),
        ])

        if len(feature_cols) >= 2:
            section_header("02", f"{feature_cols[0]} vs {feature_cols[1]}")
            plot = df_work[[feature_cols[0], feature_cols[1], "anomaly_score"]].dropna().head(2000)
            fig = px.scatter(plot, x=feature_cols[1], y=feature_cols[0],
                             color="anomaly_score",
                             color_continuous_scale=["#dc2626","#f59e0b","#16a34a"],
                             opacity=0.7)
            fig.update_traces(marker=dict(size=5))
            dark_plotly(fig, 460)
            st.plotly_chart(fig, use_container_width=True)

        section_header("03", "Top Anomalous Records")
        display = [c for c in feature_cols + ["anomaly_score"] if c in df_work.columns]
        top_anom = df_work[display].sort_values("anomaly_score").head(20)
        if top_anom.empty:
            st.info("ℹ️ No anomalous records to display.")
        else:
            st.dataframe(top_anom.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SUPPLIER PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_supplier():
    page_title("🏭", "Supplier Performance",
               "Revenue · On-Time Rate · Review Scores · Risk Map")

    df = get_active_df()
    if df is None:
        st.error("❌ No data loaded."); return

    # ── safe helpers to avoid NaN display ────────────────────────────────────
    def _fmt_mean(col, fmt, suffix=""):
        if col not in df.columns: return "N/A"
        v = df[col].mean()
        return f"{v:{fmt}}{suffix}" if pd.notna(v) else "N/A"

    def _fmt_nunique(col):
        return f"{df[col].nunique():,}" if col in df.columns else "N/A"

    section_header("01", "Fleet KPIs")
    render_stat_row([
        ("Total Sellers",    _fmt_nunique("seller_id"),          "#f1f5f9", "#3b82f6"),
        ("Avg Late Rate",    _fmt_mean("is_late",  ".1%"),       "#f87171", "#dc2626"),
        ("Avg Delivery Days",_fmt_mean("delivery_days",".1f","d"),"#fbbf24","#d97706"),
        ("Avg Review Score", _fmt_mean("review_score",".2f"),    "#c084fc", "#8b5cf6"),
    ])

    if "seller_id" in df.columns:
        # Build agg dict using named aggregation to avoid rename bugs
        agg_dict = {"order_id": ("order_id", "count")}
        if "total_order_value" in df.columns: agg_dict["total_revenue"]      = ("total_order_value", "sum")
        if "delivery_days"     in df.columns: agg_dict["avg_delivery_days"]  = ("delivery_days",     "mean")
        if "is_late"           in df.columns: agg_dict["late_rate_pct"]      = ("is_late",           "mean")
        if "review_score"      in df.columns: agg_dict["avg_review_score"]   = ("review_score",      "mean")
        if "price"             in df.columns: agg_dict["avg_price"]          = ("price",             "mean")

        sup = df.groupby("seller_id").agg(**agg_dict).reset_index()
        sup = sup.rename(columns={"order_id": "total_orders"})

        if "late_rate_pct" in sup.columns:
            sup["late_rate_pct"] = sup["late_rate_pct"] * 100

        # Safe min_orders: always show data — cap at 5 max
        min_orders = min(5, max(1, sup["total_orders"].quantile(0.05)))
        sup_filtered = sup[sup["total_orders"] >= min_orders]
        # Fallback: if filter removes everything, show all
        if sup_filtered.empty:
            sup_filtered = sup.copy()

        section_header("02", "Supplier Tables")
        a, b = st.columns(2)
        if "total_revenue" in sup_filtered.columns:
            with a:
                st.markdown('<p style="color:#3b82f6;font-size:0.8rem;font-weight:600;'
                            'letter-spacing:0.08em;text-transform:uppercase;">🏆 Top 10 by Revenue</p>',
                            unsafe_allow_html=True)
                top = sup_filtered.nlargest(min(10, len(sup_filtered)), "total_revenue").copy()
                top["seller_id"] = top["seller_id"].astype(str).str[:8] + "..."
                st.dataframe(top.reset_index(drop=True), use_container_width=True)
        else:
            with a:
                st.info("ℹ️ No revenue data available.")

        if "avg_review_score" in sup_filtered.columns:
            with b:
                st.markdown('<p style="color:#f87171;font-size:0.8rem;font-weight:600;'
                            'letter-spacing:0.08em;text-transform:uppercase;">⚠️ Bottom 10 by Review</p>',
                            unsafe_allow_html=True)
                bot = sup_filtered.nsmallest(min(10, len(sup_filtered)), "avg_review_score").copy()
                bot["seller_id"] = bot["seller_id"].astype(str).str[:8] + "..."
                st.dataframe(bot.reset_index(drop=True), use_container_width=True)
        else:
            with b:
                st.info("ℹ️ No review score data available.")

        has_late   = "late_rate_pct"    in sup_filtered.columns and sup_filtered["late_rate_pct"].notna().any()
        has_review = "avg_review_score" in sup_filtered.columns and sup_filtered["avg_review_score"].notna().any()

        if has_late and has_review:
            section_header("03", "Supplier Risk Map")
            sup_plot = sup_filtered.dropna(subset=["late_rate_pct","avg_review_score"]).copy()
            if not sup_plot.empty:
                sz  = "total_orders"      if "total_orders"      in sup_plot.columns else None
                col = "avg_delivery_days" if "avg_delivery_days" in sup_plot.columns and sup_plot["avg_delivery_days"].notna().any() else None
                if sz:
                    sup_plot[sz] = sup_plot[sz].fillna(1)
                fig = px.scatter(sup_plot, x="late_rate_pct", y="avg_review_score",
                                 size=sz, color=col,
                                 color_continuous_scale=["#16a34a","#d97706","#dc2626"],
                                 hover_data=["seller_id","total_orders"] if "total_orders" in sup_plot.columns else None,
                                 opacity=0.75)
                fig.add_hline(y=3.0, line_dash="dash", line_color="#f43f5e", line_width=1.5,
                              annotation_text="Min Review 3.0", annotation_font_color="#f43f5e")
                fig.add_vline(x=20, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                              annotation_text="Max Late 20%", annotation_font_color="#f59e0b")
                fig.update_layout(xaxis_title="Late Rate (%)", yaxis_title="Avg Review Score")
                dark_plotly(fig, 500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Not enough data to render Risk Map.")

    if "customer_state" in df.columns and "delivery_days" in df.columns:
        valid_del = df["delivery_days"].dropna()
        if len(valid_del) > 0:
            section_header("04", "Delivery Days by Region")
            state = (df.groupby("customer_state")
                     .agg(avg_delivery=("delivery_days","mean"),
                          order_count=("order_id","count"))
                     .reset_index()
                     .dropna(subset=["avg_delivery"])
                     .sort_values("avg_delivery", ascending=False)
                     .head(15))
            if not state.empty:
                overall_avg = float(valid_del.mean())
                fig2 = px.bar(state, x="customer_state", y="avg_delivery",
                              color="avg_delivery",
                              color_continuous_scale=["#16a34a","#d97706","#dc2626"])
                fig2.add_hline(y=overall_avg,
                               line_dash="dash", line_color="#3b82f6", line_width=1.5,
                               annotation_text=f"Avg {overall_avg:.1f}d",
                               annotation_font_color="#3b82f6")
                fig2.update_layout(coloraxis_showscale=False,
                                   xaxis_title="Region", yaxis_title="Avg Delivery Days")
                dark_plotly(fig2, 400)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("ℹ️ No valid delivery day data by region.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — AI REPORT & DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def page_ai_report():
    page_title("🤖", "AI Report & Download",
               "OpenRouter AI · Executive Briefs · PDF Export")

    df = get_active_df()
    if df is None:
        st.error("❌ No data loaded."); return

    lines = ["Supply Chain KPI Summary:"]
    if "order_id"          in df.columns: lines.append(f"- Total Orders: {df['order_id'].nunique():,}")
    if "total_order_value" in df.columns: lines.append(f"- Total Revenue: {df['total_order_value'].sum():,.2f}")
    if "total_order_value" in df.columns: lines.append(f"- Avg Order Value: {df['total_order_value'].mean():.2f}")
    if "is_late"           in df.columns: lines.append(f"- On-Time Rate: {(1-df['is_late'].mean())*100:.1f}%")
    if "delivery_days"     in df.columns: lines.append(f"- Avg Delivery Days: {df['delivery_days'].mean():.1f}")
    if "review_score"      in df.columns: lines.append(f"- Avg Review Score: {df['review_score'].mean():.2f}/5.0")
    if "seller_id"         in df.columns: lines.append(f"- Active Sellers: {df['seller_id'].nunique():,}")
    if "category"          in df.columns: lines.append(f"- Top Category: {df['category'].value_counts().index[0]}")
    if "customer_state"    in df.columns: lines.append(f"- Top Region: {df['customer_state'].value_counts().index[0]}")
    kpis = "\n".join(lines)

    section_header("01", "KPI Summary")
    st.code(kpis, language=None)

    section_header("02", "Generate AI Report")
    col1, col2 = st.columns(2)
    with col1:
        report_type = st.selectbox("Report Type", [
            "Executive Summary","Risk Analysis",
            "Supplier Performance Review",
            "Demand Forecast Summary",
            "Full Supply Chain Report",
        ])
    with col2:
        model_choice = st.selectbox("AI Model (OpenRouter)", [
            "openrouter/owl-alpha",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "openai/gpt-oss-120b:free",
            "poolside/laguna-m.1:free",
            "z-ai/glm-4.5-air:free",
        ])

    tone = st.select_slider("Report Tone",
                            options=["Concise","Balanced","Detailed"], value="Balanced")
    tone_instructions = {
        "Concise" : "Be very concise. Use bullet points. Maximum 300 words.",
        "Balanced": "Be thorough but focused. Use headers and bullets. 500–700 words.",
        "Detailed": "Write a comprehensive detailed report with headers and sub-sections. 800–1200 words.",
    }

    if st.button("🚀  Generate AI Report", type="primary", use_container_width=True):
        with st.spinner("Calling OpenRouter AI..."):
            try:
                from dotenv import load_dotenv
                import requests
                load_dotenv()
                try:
                    api_key = st.secrets["OPENROUTER_API_KEY"]
                except Exception:
                    api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    st.error("❌ OPENROUTER_API_KEY not found in .env or Streamlit secrets")
                    st.code("OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxx", language="bash")
                    return
                prompt = f"""You are a senior supply chain analyst writing for C-suite executives.
Write a professional {report_type} based on the following KPI data.
{tone_instructions[tone]}

{kpis}

Structure:
1. Executive Summary
2. Key Findings
3. Risk Areas & Concerns
4. Actionable Recommendations
5. Conclusion
"""
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type" : "application/json",
                    "HTTP-Referer" : "https://supply-chain-platform.streamlit.app",
                    "X-Title"      : "Supply Chain Intelligence Platform",
                }
                payload = {
                    "model"     : model_choice,
                    "messages"  : [{"role":"user","content":prompt}],
                    "max_tokens": 1500,
                }
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                     headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if "choices" in data and data["choices"]:
                    st.session_state["ai_report"]      = data["choices"][0]["message"]["content"]
                    st.session_state["ai_report_type"] = report_type
                    st.session_state.pop("pdf_bytes", None)
                    st.success("✅ Report generated!")
                elif "error" in data:
                    st.error(f"❌ {data['error'].get('message', data['error'])}")
                    st.info("Try a different model or wait 30s if rate limited.")
                else:
                    st.error("❌ Unexpected response"); st.json(data)
            except Exception as e:
                st.error(f"❌ {type(e).__name__}: {e}")

    if "ai_report" in st.session_state:
        section_header("03", "Generated Report")
        rtype = report_type
        st.markdown(f"""
        <div style="background:#1a1f2e;border:1px solid #1e2535;border-radius:10px;
                    padding:1.6rem 2rem;line-height:1.7;color:#cbd5e1;font-size:0.9rem;">
        {st.session_state["ai_report"].replace(chr(10),"<br>")}
        </div>
        """, unsafe_allow_html=True)

        section_header("04", "Export")
        report_md = f"# Supply Chain {rtype}\n\n{kpis}\n\n---\n\n{st.session_state['ai_report']}"
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button("📄  .txt", data=st.session_state["ai_report"],
                               file_name="supply_chain_report.txt", mime="text/plain",
                               use_container_width=True)
        with col2:
            st.download_button("📝  .md", data=report_md,
                               file_name="supply_chain_report.md", mime="text/markdown",
                               use_container_width=True)
        with col3:
            if st.button("📑  Generate PDF", type="primary", use_container_width=True):
                with st.spinner("Building PDF..."):
                    try:
                        from modules.pdf_report import generate_pdf_report
                        dataset_name = st.session_state.get("dataset_name","Supply Chain Dataset")
                        pdf_bytes = generate_pdf_report(
                            df=df, report_type=rtype,
                            ai_text=st.session_state["ai_report"],
                            dataset_name=dataset_name)
                        st.session_state["pdf_bytes"] = pdf_bytes
                        st.success("✅ PDF ready!")
                    except Exception as e:
                        st.error(f"❌ PDF error: {e}")
        with col4:
            if st.button("🔄  Regenerate", use_container_width=True):
                for k in ["ai_report","pdf_bytes"]: st.session_state.pop(k, None)
                st.rerun()
        if "pdf_bytes" in st.session_state:
            st.download_button("⬇️  Download PDF Report",
                               data=st.session_state["pdf_bytes"],
                               file_name="supply_chain_report.pdf",
                               mime="application/pdf",
                               use_container_width=True, type="primary")


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # Initialize all session state keys on first load
    # This prevents "SessionInfo not initialized" errors on Streamlit Cloud
    defaults = {
        "mapped_df"      : None,
        "uploaded_df"    : None,
        "mapping"        : None,
        "dataset_name"   : None,
        "ai_report"      : None,
        "ai_report_type" : None,
        "pdf_bytes"      : None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    page = render_sidebar()
    if   page == "📁  Setup & Data Upload":     page_setup()
    elif page == "📊  Executive KPI Dashboard": page_kpi_dashboard()
    elif page == "📈  Demand Forecasting":      page_forecasting()
    elif page == "🚨  Anomaly Detection":       page_anomalies()
    elif page == "🏭  Supplier Performance":    page_supplier()
    elif page == "🤖  AI Report & Download":    page_ai_report()

if __name__ == "__main__":
    main()