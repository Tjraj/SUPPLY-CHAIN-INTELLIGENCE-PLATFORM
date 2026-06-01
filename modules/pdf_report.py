# modules/pdf_report.py
# Step 9 — PDF Report Generation
# Uses FPDF2. Call generate_pdf_report() from app.py page_ai_report().

from fpdf import FPDF
from datetime import datetime
import pandas as pd
import re


def _strip_markdown(text: str) -> str:
    text = re.sub(r"#{1,6}\s?", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",    r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text.strip()


def _safe(text) -> str:
    if text is None:
        return ""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


class SupplyChainPDF(FPDF):

    PURPLE = (124, 58, 237)
    DARK   = (30,  30,  40)
    GREY   = (100, 100, 110)
    LIGHT  = (240, 238, 252)
    WHITE  = (255, 255, 255)
    RED    = (220,  50,  50)
    GREEN  = ( 22, 163,  74)

    # Page width minus both margins = 210 - 20 - 20 = 170 mm usable
    USABLE_W = 170

    def __init__(self, title="Supply Chain Intelligence Report",
                 dataset_name="Olist Dataset"):
        super().__init__()
        self.report_title  = _safe(title)
        self.dataset_name  = _safe(dataset_name)
        self.generated_at  = datetime.now().strftime("%B %d, %Y  %H:%M")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*self.PURPLE)
        self.rect(0, 0, 210, 8, "F")
        self.set_y(11)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*self.GREY)
        self.cell(0, 5, _safe(self.report_title), align="L")
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.GREY)
        self.cell(0, 10,
            f"Supply Chain Intelligence Platform  |  Page {self.page_no()}  |  {self.generated_at}",
            align="C")

    def cover_page(self, report_type: str):
        self.add_page()
        self.set_fill_color(*self.PURPLE)
        self.rect(0, 0, 210, 72, "F")
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(*self.WHITE)
        self.set_y(16)
        self.cell(0, 20, "Supply Chain", align="C")
        self.ln(14)
        self.set_font("Helvetica", "", 16)
        self.cell(0, 10, "Intelligence Platform", align="C")
        self.set_y(82)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*self.DARK)
        self.cell(0, 14, _safe(report_type), align="C")
        self.ln(12)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*self.GREY)
        self.cell(0, 8, f"Dataset: {self.dataset_name}", align="C")
        self.ln(8)
        self.cell(0, 8, f"Generated: {self.generated_at}", align="C")
        self.ln(20)
        self.set_draw_color(*self.PURPLE)
        self.set_line_width(0.8)
        self.line(30, self.get_y(), 180, self.get_y())

    def section_heading(self, text: str):
        self.ln(5)
        self.set_fill_color(*self.LIGHT)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.PURPLE)
        self.cell(self.USABLE_W, 9, _safe(text), ln=True, fill=True)
        self.ln(2)

    def sub_heading(self, text: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.DARK)
        self.cell(self.USABLE_W, 7, _safe(text), ln=True)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.DARK)
        clean = _safe(_strip_markdown(text))
        self.multi_cell(self.USABLE_W, 6, clean)
        self.ln(1)

    def bullet_list(self, items: list):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.DARK)
        for item in items:
            clean = _safe(_strip_markdown(str(item)))
            # indent + bullet
            self.set_x(self.l_margin + 4)
            self.cell(5, 6, "-")
            self.multi_cell(self.USABLE_W - 9, 6, clean)
        self.ln(1)

    def kpi_row(self, metrics: list):
        """
        metrics = list of (label, value, delta) — max 4 per row.
        """
        n   = min(len(metrics), 4)
        gap = 3
        w   = (self.USABLE_W - gap * (n - 1)) / n   # guaranteed positive

        y_start = self.get_y()

        for i, (label, value, delta) in enumerate(metrics[:n]):
            x = self.l_margin + i * (w + gap)

            # card background
            self.set_fill_color(*self.LIGHT)
            self.set_draw_color(*self.PURPLE)
            self.set_line_width(0.3)
            self.rect(x, y_start, w, 22, "FD")

            # value
            self.set_xy(x + 1, y_start + 2)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*self.PURPLE)
            self.cell(w - 2, 7, _safe(str(value)), align="C")

            # label
            self.set_xy(x + 1, y_start + 10)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*self.GREY)
            self.cell(w - 2, 5, _safe(str(label)), align="C")

            # delta
            if delta:
                color = self.GREEN if str(delta).startswith("+") else self.RED
                self.set_xy(x + 1, y_start + 15)
                self.set_font("Helvetica", "B", 7)
                self.set_text_color(*color)
                self.cell(w - 2, 5, _safe(str(delta)), align="C")

        self.set_y(y_start + 26)

    def simple_table(self, headers: list, rows: list):
        n     = len(headers)
        col_w = self.USABLE_W / n          # evenly split usable width

        # header row
        self.set_fill_color(*self.PURPLE)
        self.set_text_color(*self.WHITE)
        self.set_font("Helvetica", "B", 8)
        for h in headers:
            self.cell(col_w, 7, _safe(str(h))[:20], border=1, fill=True, align="C")
        self.ln()

        # data rows
        self.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows[:20]):
            fill = i % 2 == 0
            self.set_fill_color(240, 240, 248) if fill else self.set_fill_color(*self.WHITE)
            self.set_text_color(*self.DARK)
            for cell in row:
                self.cell(col_w, 6, _safe(str(cell))[:22], border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)

    def ai_report_section(self, ai_text: str):
        clean = _safe(_strip_markdown(ai_text))
        parts = re.split(r"\n(?=\d+\.\s)", clean)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines   = part.split("\n", 1)
            heading = lines[0].strip()
            body    = lines[1].strip() if len(lines) > 1 else ""
            if heading:
                self.sub_heading(heading)
            if body:
                for para in re.split(r"\n[-*]\s", body):
                    para = para.strip()
                    if para:
                        self.body_text(para)
        self.ln(3)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    df: pd.DataFrame,
    report_type: str,
    ai_text: str,
    dataset_name: str = "Supply Chain Dataset",
) -> bytes:

    pdf = SupplyChainPDF(title=report_type, dataset_name=dataset_name)

    # ── Cover ─────────────────────────────────────────────────────────────────
    pdf.cover_page(report_type)

    # ── KPI page ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_heading("Key Performance Indicators")

    on_time_val = (
        f"{(1 - df['is_late'].mean())*100:.1f}%"
        if "is_late" in df.columns else "N/A"
    )
    rev = (f"${df['total_order_value'].sum()/1e6:.2f}M"
           if "total_order_value" in df.columns else "N/A")
    avg_ord = (f"${df['total_order_value'].mean():.2f}"
               if "total_order_value" in df.columns else "N/A")
    orders  = (f"{df['order_id'].nunique():,}"
               if "order_id" in df.columns else "N/A")

    pdf.kpi_row([
        ("Total Orders",     orders,     None),
        ("Total Revenue",    rev,         None),
        ("Avg Order Value",  avg_ord,     None),
        ("On-Time Delivery", on_time_val, None),
    ])

    del_days = (f"{df['delivery_days'].mean():.1f}d"
                if "delivery_days" in df.columns else "N/A")
    rev_score = (f"{df['review_score'].mean():.2f}/5"
                 if "review_score" in df.columns else "N/A")
    sellers  = (f"{df['seller_id'].nunique():,}"
                if "seller_id" in df.columns else "N/A")
    customers = (f"{df['customer_unique_id'].nunique():,}"
                 if "customer_unique_id" in df.columns else "N/A")

    pdf.kpi_row([
        ("Avg Delivery Days", del_days,  None),
        ("Avg Review Score",  rev_score, None),
        ("Active Sellers",    sellers,   None),
        ("Unique Customers",  customers, None),
    ])

    # top categories
    if "category" in df.columns and "total_order_value" in df.columns:
        pdf.section_heading("Top 10 Categories by Revenue")
        cat_df = (df.groupby("category")["total_order_value"]
                  .sum().reset_index()
                  .sort_values("total_order_value", ascending=False)
                  .head(10))
        cat_df["total_order_value"] = cat_df["total_order_value"].apply(lambda x: f"${x:,.0f}")
        pdf.simple_table(["Category", "Revenue"], cat_df.values.tolist())

    # top regions
    if "customer_state" in df.columns and "total_order_value" in df.columns:
        pdf.section_heading("Top 10 Regions by Revenue")
        reg_df = (df.groupby("customer_state")["total_order_value"]
                  .sum().reset_index()
                  .sort_values("total_order_value", ascending=False)
                  .head(10))
        reg_df["total_order_value"] = reg_df["total_order_value"].apply(lambda x: f"${x:,.0f}")
        pdf.simple_table(["Region", "Revenue"], reg_df.values.tolist())

    # ── Supplier page ─────────────────────────────────────────────────────────
    if "seller_id" in df.columns:
        pdf.add_page()
        pdf.section_heading("Supplier Performance Snapshot")

        agg = {"order_id": "count"}
        if "total_order_value" in df.columns: agg["total_order_value"] = "sum"
        if "is_late"           in df.columns: agg["is_late"]           = "mean"
        if "delivery_days"     in df.columns: agg["delivery_days"]     = "mean"

        sup = df.groupby("seller_id").agg(agg).reset_index()
        sup = sup[sup["order_id"] >= 5]

        if "total_order_value" in sup.columns:
            top10 = sup.nlargest(10, "total_order_value").copy()
            top10["seller_id"]         = top10["seller_id"].astype(str).str[:10] + "..."
            top10["total_order_value"] = top10["total_order_value"].apply(lambda x: f"${x:,.0f}")
            top10["order_id"]          = top10["order_id"].astype(int)

            cols   = ["seller_id", "order_id", "total_order_value"]
            labels = ["Seller ID", "Orders", "Revenue"]
            if "is_late" in top10.columns:
                top10["is_late"] = (top10["is_late"] * 100).apply(lambda x: f"{x:.1f}%")
                cols.append("is_late"); labels.append("Late %")
            if "delivery_days" in top10.columns:
                top10["delivery_days"] = top10["delivery_days"].apply(lambda x: f"{x:.1f}d")
                cols.append("delivery_days"); labels.append("Avg Days")

            pdf.sub_heading("Top 10 Suppliers by Revenue")
            pdf.simple_table(labels, top10[cols].values.tolist())

    # ── AI narrative page ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_heading(f"AI Generated Analysis")
    if ai_text:
        pdf.ai_report_section(ai_text)
    else:
        pdf.body_text("No AI narrative available. Generate a report on the AI Report page first.")

    # ── Disclaimer page ───────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_heading("Notes & Disclaimer")
    pdf.body_text(
        "This report was automatically generated by the Supply Chain Intelligence Platform. "
        "KPI figures are computed from the active dataset at time of export. "
        "AI narrative is generated via OpenRouter and may contain inaccuracies — "
        "verify critical figures against primary data sources before acting on recommendations. "
        "Forecasts use Moving Average and Exponential Smoothing models and should be treated "
        "as directional indicators only."
    )
    pdf.ln(4)
    pdf.sub_heading("Data Summary")
    pdf.bullet_list([
        f"Dataset: {dataset_name}",
        f"Total rows: {len(df):,}",
        f"Columns: {df.shape[1]}",
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    return bytes(pdf.output())