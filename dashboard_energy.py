import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import gspread
import json
from gspread_dataframe import get_as_dataframe
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Dashboard Giá Dầu & Xăng Dầu", layout="wide")
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<style>
.stDataFrame {font-size:15px !important;}
.stPlotlyChart {margin-bottom: 0.5rem !important;}
.block-container {padding: 0.7rem 0.2rem !important;}
@media (max-width: 800px) {
    .stDataFrame, th, td {font-size: 16px !important;}
}
</style>
""", unsafe_allow_html=True)

def get_gspread_client():
    import json
    from oauth2client.service_account import ServiceAccountCredentials
    import gspread

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client

@st.cache_data(ttl=300)
def load_sheet(sheet_name):
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID).worksheet(sheet_name)
    df = get_as_dataframe(sheet).dropna(how="all")
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    return df

SHEET_ID = st.secrets["SHEET_ID"]

# Định nghĩa tên các sheet
SHEET_OIL = "data"
SHEET_LOG = "log_run_history_2"
SHEET_ENRICH = "train_mop_enrich"
SHEET_ROLLING = "log_rolling_results"
SHEET_GCS = "data_giadinh"

# Sau đó mới load data
df_oil = load_sheet(SHEET_OIL)
df_mops = load_sheet(SHEET_LOG)
df_rolling = load_sheet(SHEET_ROLLING)
df_log = load_sheet(SHEET_LOG)
df_enrich = load_sheet(SHEET_ENRICH)
df_gcs = load_sheet(SHEET_GCS)

def vnd(val):
    try:
        v = float(str(val).replace(",", ".").replace(" ", ""))
        return f"{v:,.0f}".replace(",", ".")
    except:
        return str(val) if val is not None else "..."

def render_tab_oil(df, df_enrich):
    st.header("Giá dầu Brent & WTI")
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if not pd.api.types.is_datetime64_any_dtype(df['thoi_gian']):
        df['thoi_gian'] = pd.to_datetime(df['thoi_gian'], errors='coerce', dayfirst=True)
    df = df[df['thoi_gian'].notna()]
    df['brent'] = pd.to_numeric(df['brent'], errors='coerce')
    df['wti'] = pd.to_numeric(df['wti'], errors='coerce')

    all_dates = df['thoi_gian'].dt.date.unique()
    date_selected = st.sidebar.selectbox("Chọn ngày", all_dates, index=len(all_dates)-1)
    df_day = df[df['thoi_gian'].dt.date == date_selected]
    if df_day.empty:
        st.warning("Không có dữ liệu ngày này.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_day['thoi_gian'], y=df_day['brent'], name='Brent', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df_day['thoi_gian'], y=df_day['wti'], name='WTI', mode='lines+markers'))
    fig.update_layout(margin=dict(l=5, r=5, t=30, b=12), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_day.sort_values("thoi_gian", ascending=False).head(10), use_container_width=True)

def render_tab_mops(df_mops, df_enrich):
    st.header("Dự báo giá MOPS95")
    df = df_mops.copy()
    df["RunTime"] = pd.to_datetime(df["RunTime"], dayfirst=True, errors='coerce')
    df = df[df["RunTime"].notna()]
    df["date"] = df["RunTime"].dt.date
    all_dates = df["date"].unique()
    if len(all_dates) == 0:
        st.warning("Không có dữ liệu để hiển thị!")
        return
    date_selected = st.sidebar.selectbox("Chọn ngày dự báo", all_dates, index=len(all_dates)-1)
    df_ngay = df[df["date"] == date_selected]
    if df_ngay.empty:
        st.warning("Không có dự báo ngày này.")
        return
    runtime_options = df_ngay["RunTime"].dt.strftime("%H:%M:%S").tolist()
    runtime_map = dict(zip(runtime_options, df_ngay.index))
    runtime_selected = st.selectbox("Chọn thời điểm", runtime_options, index=len(runtime_options)-1)
    row = df_ngay.loc[runtime_map[runtime_selected]]

    df_enrich = df_enrich.copy()
    df_enrich["Date"] = pd.to_datetime(df_enrich["Date"], errors='coerce')
    yesterday = pd.to_datetime(date_selected) - pd.Timedelta(days=1)
    row_enrich = df_enrich[df_enrich["Date"].dt.date == yesterday]
    row_enrich = row_enrich.iloc[-1] if not row_enrich.empty else None

    def get_val(row, key):
        if row is None: return np.nan
        for c in row.index:
            if key.lower() in c.lower():
                try: return float(row[c])
                except: return row[c]
        return np.nan

    brent_today = get_val(row, "BrentClose")
    brent_yt = get_val(row_enrich, "BrentClose")
    wti_today = get_val(row, "WTIClose")
    wti_yt = get_val(row_enrich, "WTIClose")
    mops_yt = get_val(row_enrich, "MOP95")

    st.markdown(f"Brent: {vnd(brent_today)} | Hôm qua: {vnd(brent_yt)} | Δ: {vnd(brent_today-brent_yt) if pd.notna(brent_today) and pd.notna(brent_yt) else '...'}")
    st.markdown(f"WTI: {vnd(wti_today)} | Hôm qua: {vnd(wti_yt)} | Δ: {vnd(wti_today-wti_yt) if pd.notna(wti_today) and pd.notna(wti_yt) else '...'}")
    if pd.notna(mops_yt): st.markdown(f"MOPS95 hôm qua: {vnd(mops_yt)}")

    models = ["LinearRegression", "XGBoost", "RandomForest", "Polynomial"]
    st.markdown("### Dự báo các mô hình:")
    summary = []
    for m in models:
        pred = get_val(row, f"{m}_pred")
        mae = get_val(row, f"{m}_mae")
        delta = get_val(row, f"{m}_delta")
        if pd.notna(pred):
            summary.append({
                "Model": m, "Dự báo": vnd(pred), "MAE": vnd(mae), "Δ": vnd(delta)
            })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

def render_tab_mae(df_rolling):
    st.header("Sai số MAE Rolling")
    df = df_rolling.copy()
    df["RunTime"] = pd.to_datetime(df["RunTime"], dayfirst=True, errors='coerce')
    df = df[df["RunTime"].notna()]
    min_date = df["RunTime"].min().date()
    max_date = df["RunTime"].max().date()
    date_range = st.sidebar.date_input("Chọn khoảng ngày", [min_date, max_date], min_value=min_date, max_value=max_date, key="mae")
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 1: date_range = [date_range[0], date_range[0]]
    else: date_range = [date_range, date_range]
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df_view = df[(df["RunTime"] >= start_date) & (df["RunTime"] < end_date)]
    if df_view.empty:
        st.warning("Không có dữ liệu trong khoảng thời gian đã chọn.")
        return
    if "Model" in df_view.columns:
        models = df_view["Model"].dropna().unique().tolist()
        selected_models = st.multiselect("Chọn mô hình (MAE)", models, default=models, key="model_mae")
        if selected_models: df_view = df_view[df_view["Model"].isin(selected_models)]
        else:
            st.warning("Chưa chọn mô hình nào.")
            return
    df_view["RunTime_fmt"] = df_view["RunTime"].dt.strftime('%d/%m %H:%M')
    fig = px.bar(df_view, x="RunTime_fmt", y="MAE_Rolling", color="Model" if "Model" in df_view.columns else None, barmode="group")
    fig.update_layout(xaxis_title="Thời gian", yaxis_title="MAE (Rolling)", font=dict(size=13))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_view.sort_values("RunTime", ascending=False).head(10), use_container_width=True)

def render_tab_gcs_chuky(df):
    st.header("Dự báo GCS chu kỳ")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df[df["Date"].notna()]
    df = df.sort_values("Date").reset_index(drop=True)
    today = pd.Timestamp.now().date()
    df_today = df[df["Date"].dt.date == today]
    if not df_today.empty:
        row_today = df_today.iloc[-1]
    else:
        row_today = df.iloc[-1]
        today = row_today["Date"].date()
    st.markdown(f"**Ngày:** {row_today['Date'].strftime('%d/%m/%Y')}")
    st.markdown(f"Giá bán lẻ: {vnd(row_today.get('GIALE'))} ₫")
    st.markdown(f"GCS: {vnd(row_today.get('GCS'))} ₫")
    st.markdown(f"Chênh lệch: {vnd(row_today.get('Chênh lệch'))} ₫")
    st.dataframe(df.sort_values("Date", ascending=False).head(10), use_container_width=True)

def render_tab_14days(df):
    st.header("Giá cơ sở 14 ngày gần nhất")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date", ascending=False)
    show_cols = ["Date", "MOP95", "RATEVCB", "GIALE", "GCS_CHENHLECH", "PRICEBCT", "xangdau", "GCS_chenhlech", "Chenhlech", "GCS_heso"]
    df_view = df[show_cols].head(14).copy()
    df_view["Date"] = df_view["Date"].dt.strftime("%d/%m/%Y")
    for col in ["MOP95", "RATEVCB", "GIALE", "GCS_CHENHLECH", "PRICEBCT", "GCS_chenhlech", "Chenhlech", "GCS_heso", "xangdau"]:
        if col in df_view.columns: df_view[col] = df_view[col].apply(vnd)
    df_view = df_view.replace(["nan", "None", None, pd.NA, np.nan], "")
    st.dataframe(df_view, use_container_width=True, hide_index=True)

def render_tab_log(df_log, df_mops):
    st.header("Log các lần chạy & cảnh báo")
    st.dataframe(df_log.sort_values("RunTime", ascending=False).head(20))
    if "Delta" in df_mops.columns:
        st.write("Cảnh báo chênh lệch dự báo > 5 đồng:")
        df_alert = df_mops[df_mops["Delta"].abs() > 5]
        st.dataframe(df_alert.sort_values("RunTime", ascending=False).head(10))

tab_options = [
    "🛢️ Giá Dầu",
    "⛽ Dự Báo MOPS",
    "📉 MAE Rolling",
    "📝 Log/Báo Cáo",
    "📅 Dự báo GCS chu kỳ",
    "📋 Giá cơ sở 14 ngày"
]
tab_selected = st.sidebar.radio("Chuyên mục", tab_options, index=0)

if tab_selected == tab_options[0]:
    render_tab_oil(df_oil, df_enrich)
elif tab_selected == tab_options[1]:
    render_tab_mops(df_mops, df_enrich)
elif tab_selected == tab_options[2]:
    render_tab_mae(df_rolling)
elif tab_selected == tab_options[3]:
    render_tab_log(df_log, df_mops)
elif tab_selected == tab_options[4]:
    render_tab_gcs_chuky(df_gcs)
elif tab_selected == tab_options[5]:
    df_main = load_sheet("data_main")
    render_tab_14days(df_main)
