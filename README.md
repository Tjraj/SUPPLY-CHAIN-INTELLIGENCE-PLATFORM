# Supply Chain Intelligence Platform


[Live App](https://supply-chain-intelligence-platform-vxtmzjfkdahzqpdqufmzph.streamlit.app)

# Business Problem
Supply chain teams need instant visibility into demand trends, supplier anomalies, and delivery risks — without waiting for manual analyst reports. This platform automates all three with AI-generated insights.

# What It Does
- Upload any supply chain CSV and get instant analysis
- Demand Forecasting — Prophet time-series model predicts future demand with confidence intervals
- Anomaly Detection — IsolationForest flags unusual orders, deliveries, or supplier behaviour
- AI Insights — OpenRouter AI generates plain-English executive summaries of findings
- PDF Report Generation — Automated downloadable reports via FPDF2

# Tech Stack
| Tool | Purpose |

| Python | Core logic |
| Streamlit | Web application |
| Prophet | Time-series demand forecasting |
| IsolationForest | Anomaly detection |
| OpenRouter AI | Natural language insight generation |
| FPDF2 | Automated PDF reports |

# Key Features
- File upload support (CSV)
- Interactive forecast charts with confidence bands
- Anomaly scatter plots with flagged records highlighted
- One-click PDF report download
- AI-generated recommendations in plain English

# How to Run Locally
```bash
git clone https://github.com/Tjraj/SUPPLY-CHAIN-INTELLIGENCE-PLATFORM
cd SUPPLY-CHAIN-INTELLIGENCE-PLATFORM
pip install -r requirements.txt
streamlit run app.py
```


