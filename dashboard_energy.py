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
    import numpy as np

    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    """, unsafe_allow_html=True)
    st.markdown("## MÔ HÌNH DỰ BÁO GIÁ PLATTS XĂNG 95:")

    df = df_mops.copy()
    df["RunTime"] = pd.to_datetime(df["RunTime"], dayfirst=True, errors='coerce')
    df = df[df["RunTime"].notna()]
    df = df.sort_values("RunTime", ascending=True)
    df["date"] = df["RunTime"].dt.date

    # Sidebar chọn ngày
    all_dates = df["date"].unique()
    if len(all_dates) == 0:
        st.warning("Không có dữ liệu để hiển thị!")
        return

    date_selected = st.sidebar.date_input(
        "Chọn ngày cần xem",
        value=all_dates[-1],
        min_value=min(all_dates),
        max_value=max(all_dates)
    )
    if isinstance(date_selected, list):
        date_selected = date_selected[0]
    date_selected = pd.to_datetime(date_selected).date()

    df_ngay = df[df["date"] == date_selected]
    if df_ngay.empty:
        st.warning("Không có dữ liệu dự báo cho ngày này!")
        return

    # Chọn thời điểm (RunTime)
    runtime_options = df_ngay["RunTime"].dt.strftime("%H:%M:%S").tolist()
    runtime_map = dict(zip(runtime_options, df_ngay.index))
    runtime_selected = st.selectbox("Chọn thời điểm (RunTime)", runtime_options, index=len(runtime_options)-1)
    row = df_ngay.loc[runtime_map[runtime_selected]]

    # Lấy hôm qua gần nhất có dữ liệu enrich
    df_enrich_ = df_enrich.copy()
    for col in df_enrich_.columns:
        if "date" in col.lower():
            df_enrich_["Date"] = pd.to_datetime(df_enrich_[col], dayfirst=True, errors='coerce')
    df_enrich_ = df_enrich_[df_enrich_["Date"].notna()].sort_values("Date")

    # Tìm row hôm qua gần nhất có dữ liệu
    row_enrich = df_enrich_[df_enrich_["Date"].dt.date < date_selected].iloc[-1] if not df_enrich_[df_enrich_["Date"].dt.date < date_selected].empty else None

    # === Hàm phụ ===
    def to_float(val):
        try:
            if pd.isna(val): return None
            if isinstance(val, (float, int)): return float(val)
            return float(str(val).replace(",", ".").replace(" ", ""))
        except: return None

    def format_vn(val):
        try:
            v = float(val)
            return f"{v:,.2f}".replace(",", ".")
        except: return str(val) if val is not None else "..."

    def sign_arrow(val):
        if val is None: return ""
        arrow = "🔼" if val > 0 else "🔽" if val < 0 else "⏺"
        color = "#198544" if val > 0 else "#d6223c" if val < 0 else "#787878"
        sign = "+" if val > 0 else ""
        return f"<span style='color:{color};font-weight:bold'>{arrow} {sign}{val:.2f}</span>"

    def safe_val(obj, key, fmt=".2f"):
        if obj is None: return "..."
        columns = [str(c).lower().replace(" ", "").replace("_", "") for c in obj.index]
        key_search = key.lower().replace(" ", "").replace("_", "")
        key_dict = dict(zip(columns, obj.index))
        real_key = key_dict.get(key_search)
        if real_key is not None and pd.notnull(obj[real_key]):
            val = obj[real_key]
            if isinstance(val, str):
                try: val = to_float(val)
                except: return str(val)
            if not isinstance(val, (int, float, np.number)):
                return str(val)
            return format_vn(val)
        return "..."

    def safe_time(obj, key):
        v = obj.get(key) if obj is not None else None
        if isinstance(v, pd.Timestamp) and pd.notnull(v):
            return v.strftime('%d/%m/%Y %H:%M:%S')
        return "N/A"

    run_time_str = safe_time(row, 'RunTime')
    brent_today = safe_val(row, "BrentClose")
    wti_today = safe_val(row, "WTIClose")
    brent_yt = safe_val(row_enrich, "BrentClose")
    wti_yt = safe_val(row_enrich, "WTIClose")
    mops_yt = safe_val(row_enrich, "MOP95")

    # Tính chênh lệch Brent, WTI hôm nay vs hôm qua
    try:
        brent_today_f = to_float(row["BrentClose"])
        brent_yt_f = to_float(row_enrich["BrentClose"]) if row_enrich is not None else None
        wti_today_f = to_float(row["WTIClose"])
        wti_yt_f = to_float(row_enrich["WTIClose"]) if row_enrich is not None else None
    except Exception:
        brent_today_f = brent_yt_f = wti_today_f = wti_yt_f = None

    delta_brent_val = brent_today_f - brent_yt_f if brent_today_f is not None and brent_yt_f is not None else None
    delta_wti_val = wti_today_f - wti_yt_f if wti_today_f is not None and wti_yt_f is not None else None

    # Giá MOPS thực tế hôm qua
    try:
        mops_yt_f = to_float(row_enrich["MOP95"]) if row_enrich is not None else None
    except Exception:
        mops_yt_f = None

    # Top 3 model sát nhất với Brent, tính khoảng dự báo
    models = ["LinearRegression", "XGBoost", "RandomForest", "Polynomial"]
    pred_range_min = pred_range_max = None
    delta_models = []

    if row_enrich is not None and pd.notnull(row_enrich["BrentClose"]) and pd.notnull(row["BrentClose"]):
        try:
            brent_today_f_ = to_float(row["BrentClose"])
            brent_yt_f_ = to_float(row_enrich["BrentClose"])
            delta_brent = brent_today_f_ - brent_yt_f_
        except Exception:
            delta_brent = None

        for m in models:
            pred = row.get(f"{m}_pred", None)
            mae = row.get(f"{m}_mae", None)
            delta = row.get(f"{m}_delta", None)
            if pred is None or mae is None or delta is None or delta_brent is None:
                continue
            try:
                pred = to_float(pred)
                mae = abs(to_float(mae))
                delta = to_float(delta)
            except Exception:
                continue
            pred_adj = pred + mae if delta < 0 else pred - mae
            dist = abs(abs(delta) - abs(delta_brent))
            delta_models.append((m, delta, mae, dist, pred, pred_adj))
        delta_models = sorted(delta_models, key=lambda x: x[3])  # Sát chênh lệch Brent nhất

        if len(delta_models) >= 3:
            preds_adj = [delta_models[i][5] for i in range(3)]
            pred_range_min = min(preds_adj)
            pred_range_max = max(preds_adj)
        elif len(delta_models) >= 2:
            preds_adj = [delta_models[i][5] for i in range(2)]
            pred_range_min = min(preds_adj)
            pred_range_max = max(preds_adj)

    # Chênh lệch khoảng dự báo và giá thực tế hôm qua
    delta_pred_min = delta_pred_max = None
    if mops_yt_f is not None and pred_range_min is not None and pred_range_max is not None:
        delta_pred_min = pred_range_min - mops_yt_f
        delta_pred_max = pred_range_max - mops_yt_f

    # UI block đẹp, màu nền, format số Việt Nam, tối ưu mobile
    st.markdown(
        f"""
        <div style='background: #EFF4FA; padding: 10px 12px; border-radius:12px; margin-bottom:10px;font-size:1.02em'>
            <b>⏰ Thời điểm dự báo:</b> <span style='color:#1558a7;font-size:1.1em'>{run_time_str}</span><br>
            <b>Brent hôm nay:</b> <span style='color:#347d39;font-size:1.17em'>{brent_today}</span> |
            <b>Hôm qua:</b> <span style='color:#7c7c7c'>{brent_yt}</span> {sign_arrow(delta_brent_val)}<br>
            <b>WTI hôm nay:</b> <span style='color:#a74814;font-size:1.17em'>{wti_today}</span> |
            <b>Hôm qua:</b> <span style='color:#7c7c7c'>{wti_yt}</span> {sign_arrow(delta_wti_val)}<br>
            <b>PLATTS XĂNG 95 hôm qua:</b> <span style='color:#336699'>{mops_yt}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # HIỂN THỊ KHOẢNG DỰ BÁO + CHÊNH LỆCH
    if pred_range_min is not None and pred_range_max is not None and mops_yt_f is not None:
        st.markdown(
            f"""<div style='background:#f7f0e3;padding:11px 12px;border-radius:12px;margin-bottom:10px;'>
            <b>🚦 Khoảng dự báo giá Platts xăng 95 (MOPS95):</b>
            <span style='color:#2A62B8;font-weight:bold;font-size:1.23rem'>
            {format_vn(pred_range_min)} – {format_vn(pred_range_max)}
            </span> đ/lít<br>
            <b>Chênh lệch với giá MOPS hôm qua:</b> 
            {sign_arrow(delta_pred_min)} đến {sign_arrow(delta_pred_max)}
            </div>""",
            unsafe_allow_html=True
        )
    elif pred_range_min is not None and pred_range_max is not None:
        st.markdown(
            f"""<div style='background:#f7f0e3;padding:11px 12px;border-radius:12px;margin-bottom:10px;'>
            <b>🚦 Khoảng dự báo giá Platts xăng 95 (MOPS95):</b>
            <span style='color:#2A62B8;font-weight:bold;font-size:1.23rem'>{format_vn(pred_range_min)} – {format_vn(pred_range_max)}</span> đ/lít
            </div>""",
            unsafe_allow_html=True
        )

    # Block 3 mô hình sát nhất với Brent
    if len(delta_models) >= 3:
        st.markdown("""
        <style>
            .model-brent-box {
                background: #f2f7ff;
                border-radius: 10px;
                padding: 10px 13px 8px 13px;
                margin-bottom: 0px;
            }
            .model-brent-title {
                color: #15477a;
                font-size: 1.09em;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .model-brent-row {
                font-size: 14px;
                margin-bottom: 2px;
            }
            .model-name {font-weight: 700; min-width: 82px; display: inline-block;}
            .model-label {min-width: 40px; display: inline-block; color: #666;}
            .model-value {font-weight: 600; color: #23356c;}
            .model-delta-up {color: #198544;}
            .model-delta-down {color: #d6223c;}
        </style>
        """, unsafe_allow_html=True)
        st.markdown(
            """<div class="model-brent-box">
                <div class="model-brent-title">📈 3 mô hình sát nhất với biến động Brent:</div>
            """, unsafe_allow_html=True)
        icons = {"LinearRegression": "🟦", "XGBoost": "🟧", "RandomForest": "🟩", "Polynomial": "🟪"}
        colors = {"RandomForest": "#2270d6", "Polynomial": "#6b34ad", "XGBoost": "#b86818", "LinearRegression": "#357824"}
        for m, delta, mae, dist, pred, pred_adj in delta_models[:3]:
            sign = "+" if delta > 0 else ""
            icon = icons.get(m, "")
            color = colors.get(m, "#444")
            arrow = "🔻 Giảm" if delta < 0 else "🔺 Tăng"
            delta_class = "model-delta-down" if delta < 0 else "model-delta-up"
            st.markdown(
                f"""
                <div class="model-brent-row">
                    <span class="model-name" style="color:{color}">{icon} {m}:</span>
                    <span class="model-label">Dự báo:</span>
                    <span class="model-value">{format_vn(pred)}</span> |
                    <span class="model-label">Δ:</span>
                    <span class="{delta_class}">{sign}{format_vn(delta)}</span> |
                    <span class="model-label">|Δ-Brent|=</span>
                    <span class="model-value">{format_vn(dist)}</span> |
                    <span class="model-label">MAE=</span>
                    <span class="model-value">{format_vn(mae)}</span>
                    <br>
                    <span class="model-label">→ Dự báo hiệu chỉnh:</span>
                    <span class="model-value" style="color:{color};font-size:1.07em;">{format_vn(pred_adj)}</span>
                    <span class="{delta_class}" style="font-size:0.98em;margin-left:5px">{arrow} {sign}{format_vn(delta)}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Bảng tóm tắt tất cả mô hình
    lines = ["┏━━━━━━━━━━━━━━━━━━━"]
    for m in models:
        pred = row.get(f"{m}_pred", None)
        mae = row.get(f"{m}_mae", None)
        delta = row.get(f"{m}_delta", None)
        if pred is None or mae is None or delta is None:
            continue
        try:
            pred = to_float(pred)
            mae = to_float(mae)
            delta = to_float(delta)
        except Exception:
            continue
        sign = "+" if delta > 0 else ""
        arrow = "Tăng" if delta > 0 else "Giảm"
        lines.append(f"┃ {m}: {format_vn(pred)} (MAE={format_vn(mae)}) ({arrow} {sign}{format_vn(delta)})")
    lines.append("┗━━━━━━━━━━━━━━━━━━━")
    st.code('\n'.join(lines), language="text")

    # (Tuỳ chọn) Hiện bảng chi tiết
    with st.expander("Bảng dữ liệu chi tiết ngày này"):
        st.dataframe(df_ngay, use_container_width=True)

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
