import streamlit as st

st.title("Test — Supply Chain Platform")
st.write("If you see this, basic Streamlit works.")

st.write("Testing imports one by one...")

try:
    import pandas as pd
    st.success("✅ pandas OK")
except Exception as e:
    st.error(f"❌ pandas: {e}")

try:
    import numpy as np
    st.success("✅ numpy OK")
except Exception as e:
    st.error(f"❌ numpy: {e}")

try:
    import plotly.express as px
    st.success("✅ plotly OK")
except Exception as e:
    st.error(f"❌ plotly: {e}")

try:
    from sklearn.ensemble import IsolationForest
    st.success("✅ sklearn OK")
except Exception as e:
    st.error(f"❌ sklearn: {e}")

try:
    from fpdf import FPDF
    st.success("✅ fpdf2 OK")
except Exception as e:
    st.error(f"❌ fpdf2: {e}")

try:
    from dotenv import load_dotenv
    st.success("✅ python-dotenv OK")
except Exception as e:
    st.error(f"❌ python-dotenv: {e}")

try:
    import requests
    st.success("✅ requests OK")
except Exception as e:
    st.error(f"❌ requests: {e}")

try:
    import duckdb
    st.success("✅ duckdb OK")
except Exception as e:
    st.error(f"❌ duckdb: {e}")

try:
    from prophet import Prophet
    st.success("✅ prophet OK")
except Exception as e:
    st.error(f"❌ prophet: {e}")

try:
    from modules.pdf_report import generate_pdf_report
    st.success("✅ pdf_report module OK")
except Exception as e:
    st.error(f"❌ pdf_report module: {e}")

st.write("All import tests done.")