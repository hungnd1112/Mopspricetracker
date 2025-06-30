import streamlit as stMore
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

# ==== LOAD DATA ====
df_oil = load_sheet(SHEET_OIL)
df_mops = load_sheet(SHEET_LOG)
df_rolling = load_sheet(SHEET_ROLLING)
df_log = load_sheet(SHEET_LOG)      # hoặc đổi sang sheet log khác nếu muốn
df_enrich = load_sheet(SHEET_ENRICH)
df_gcs = load_sheet(SHEET_GCS)
df_main = load_sheet(SHEET_MAIN)
df_kq, df_thongke = load_kq_and_stat("kq_dubao")


# ==== SIDEBAR RADIO MENU ====
tab_options = [
    "🛢️ Giá Dầu",
    "⛽ Dự Báo MOPS",
    "📉 MAE Rolling",
    "📝 Log/Báo Cáo",
    "📅 Dự báo GCS chu kỳ",
    "📋 Giá cơ sở 14 ngày",
    "📋 Bảng tổng hợp"
]
tab_selected = st.sidebar.radio("Chọn chuyên mục", tab_options, index=0)

if tab_selected == tab_options[0]:
    render_tab_oil(df_oil, df_enrich)
elif tab_selected == tab_options[1]:
    render_tab_mops(df_mops, df_enrich, df_kq, df_thongke, df_log)
elif tab_selected == tab_options[2]:
    render_tab_mae(df_rolling)
elif tab_selected == tab_options[3]:
    render_tab_log_kq(df_kq)
elif tab_selected == tab_options[4]:
    df_gcs = load_sheet(SHEET_GCS)
    df_main = load_sheet(SHEET_MAIN)
    df_main["Date"] = pd.to_datetime(df_main["Date"], errors="coerce")
    render_tab_gcs_chuky(df_gcs, df_main)
elif tab_selected == tab_options[5]:
    render_tab_14days(df_main)
elif tab_selected == tab_options[6]:
    tab_bang_tong_hop(df_mops, df_enrich, df_gcs, df_brent_wti=None)

def to_float(val):
    if isinstance(val, str):
        val = val.replace(",", ".").replace(" ", "")
    try:
        return float(val)
    except Exception:
        return np.nan

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px

@st.cache_data
def plot_oil_price(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['thoi_gian'],
        y=df['brent'],
        mode='lines',
        name='Brent',
        line=dict(shape='spline', width=3, color='#2A62B8')
    ))
    fig.add_trace(go.Scatter(
        x=df['thoi_gian'],
        y=df['wti'],
        mode='lines',
        name='WTI',
        line=dict(shape='spline', width=3, color='#59A5F5')
    ))
    fig.update_layout(
        title="Giá dầu Brent & WTI (khung giờ Platts Singapore)",
        xaxis_title="Thời gian",
        yaxis_title="Giá dầu ($/thùng)",
        legend_title="Loại dầu",
        font=dict(size=14),
        height=430,
        margin=dict(l=18, r=12, t=44, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(180,180,180,0.15)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(180,180,180,0.18)')
    return fig

def render_tab_oil(df, df_enrich):
    import streamlit as st
    import pandas as pd
    import numpy as np

    st.header("🛢️ Biểu đồ giá dầu Brent & WTI")
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if not pd.api.types.is_datetime64_any_dtype(df['thoi_gian']):
        df['thoi_gian'] = pd.to_datetime(df['thoi_gian'], errors='coerce', dayfirst=True)
    df = df[df['thoi_gian'].notna()]
    df['brent'] = pd.to_numeric(df['brent'], errors='coerce')
    df['wti'] = pd.to_numeric(df['wti'], errors='coerce')

    # ========== LỌC NGÀY ==========
    today = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").date()
    min_date = df['thoi_gian'].min().date()
    max_date = df['thoi_gian'].max().date()
    if min_date <= today <= max_date:
        default_start = default_end = today
    else:
        default_start = default_end = max_date

    date_range = st.sidebar.date_input(
        "Chọn khoảng ngày",
        [default_start, default_end],
        min_value=min_date,
        max_value=max_date,
        key="oil"
    )
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 1:
            date_range = [date_range[0], date_range[0]]
    else:
        date_range = [date_range, date_range]

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df_view = df[(df['thoi_gian'] >= start_date) & (df['thoi_gian'] < end_date)]

    # ========== SO SÁNH ==========
    row_today_data = df_view.sort_values("thoi_gian", ascending=False).head(1)
    if not row_today_data.empty:
        brent_today = row_today_data["brent"].values[0]
        wti_today = row_today_data["wti"].values[0]
        date_today_compare = row_today_data["thoi_gian"].dt.date.values[0]
    else:
        brent_today = wti_today = date_today_compare = None

    df_enrich = df_enrich.copy()
    df_enrich["Date"] = pd.to_datetime(df_enrich["Date"], errors="coerce")
    yesterday = pd.to_datetime(date_today_compare) - pd.Timedelta(days=1) if date_today_compare is not None else today - pd.Timedelta(days=1)
    row_yt = df_enrich[df_enrich["Date"].dt.date == yesterday]
    if not row_yt.empty:
        brent_yt = row_yt["BrentClose"].values[-1]
        wti_yt = row_yt["WTIClose"].values[-1]
        date_yt_compare = yesterday
    else:
        df_yt = df_enrich[df_enrich["Date"].dt.date < date_today_compare] if date_today_compare is not None else df_enrich[df_enrich["Date"].dt.date < today]
        if not df_yt.empty:
            brent_yt = df_yt["BrentClose"].values[-1]
            wti_yt = df_yt["WTIClose"].values[-1]
            date_yt_compare = df_yt["Date"].dt.date.values[-1]
        else:
            brent_yt = wti_yt = date_yt_compare = None

    def format_delta(val):
        if val is None: return ""
        emoji = "▲" if val > 0 else "▼" if val < 0 else "■"
        color = "#23b74e" if val > 0 else "#d6223c" if val < 0 else "#555"
        sign = "+" if val > 0 else ""
        txt = "Tăng" if val > 0 else "Giảm" if val < 0 else "-"
        return f"<span style='color:{color};font-weight:600'>{emoji} {sign}{val:.2f} ({txt})</span>"

    def format_num(x):
        try:
            return f"{float(x):,.2f}".replace(",", ".")
        except: return str(x)

    # So sánh hôm nay/hôm qua
    if brent_today is not None and brent_yt is not None:
        d1 = pd.to_datetime(date_today_compare).strftime('%d/%m/%Y')
        d2 = pd.to_datetime(date_yt_compare).strftime('%d/%m/%Y')
        st.markdown(
            f"""<div style="background:#3057a4;padding:12px 18px 10px 18px;border-radius:8px;color:white;font-size:1.09em;font-weight:600;margin-bottom:16px">
            📊 <span style="font-size:1.07em">So sánh giá đóng cửa hôm nay và hôm qua</span><br>
            Ngày so sánh: <b>{d1}</b> & <b>{d2}</b>
            <br>
            <span style="margin-right:14px">Brent hôm nay: {format_num(brent_today)}</span> | 
            <span style="margin-right:12px">Hôm qua: {format_num(brent_yt)}</span>
            {format_delta(float(brent_today) - float(brent_yt))}
            <br>
            <span style="margin-right:14px">WTI hôm nay: {format_num(wti_today)}</span> | 
            <span style="margin-right:12px">Hôm qua: {format_num(wti_yt)}</span>
            {format_delta(float(wti_today) - float(wti_yt))}
            </div>""", unsafe_allow_html=True
        )

    # ========== BIỂU ĐỒ ==========
    st.markdown("##### Số dòng dữ liệu: " + str(len(df_view)))
    if not df_view.empty:
        fig = plot_oil_price(df_view)
        st.plotly_chart(fig, use_container_width=True)

    # ========== DỮ LIỆU MỚI NHẤT: HTML TABLE ==========

    st.markdown("""
    <style>
    .oil-table-custom {
        font-size: 15px !important;
        width: 100%;
        border-collapse: collapse;
        margin-top: 6px;
        margin-bottom: 14px;
        background: #fff;
        box-shadow: 0 2px 14px rgba(0,0,0,0.08);
        border-radius: 14px;
        overflow: hidden;
    }
    .oil-table-custom th, .oil-table-custom td {
        border: 1px solid #f0f1f6;
        padding: 7px 4px;
        text-align: center;
        vertical-align: middle;
    }
    .oil-table-custom th {
        background: #eaf3fa;
        font-weight: 600;
        color: #29518a;
    }
    .oil-table-custom tr:nth-child(even) td {
        background: #f7fbff;
    }
    .oil-table-custom tr:hover td {
        background: #eaf0fb;
    }
    @media (max-width: 700px) {
        .oil-table-custom th, .oil-table-custom td { padding: 5px 1px; font-size:0.98em;}
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("**Dữ liệu mới nhất (10 dòng gần nhất):**")

    df_last10 = df_view.sort_values("thoi_gian", ascending=False).head(10).copy()

    def vn_time(dt):
        try:
            return pd.to_datetime(dt).strftime('%d/%m/%Y %H:%M')
        except: return str(dt)

    # Định dạng số kiểu VN cho cột số
    for col in ['brent', 'wti']:
        if col in df_last10.columns:
            df_last10[col] = df_last10[col].apply(lambda x: format_num(x))
    if 'thoi_gian' in df_last10.columns:
        df_last10['thoi_gian'] = df_last10['thoi_gian'].apply(vn_time)

    # Tạo bảng HTML
    html = '<table class="oil-table-custom">'
    html += "<thead><tr>"
    for col in df_last10.columns:
        html += f"<th>{col.upper()}</th>"
    html += "</tr></thead><tbody>"

    for _, row in df_last10.iterrows():
        html += "<tr>"
        for col in df_last10.columns:
            html += f"<td>{row[col]}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

    if df_view.empty:
        st.warning("Không có dữ liệu trong khoảng thời gian đã chọn!")

# === Lưu ý ===
# - Bạn chỉ cần thay đoạn st.dataframe(...) ở cuối bằng đoạn HTML ở trên là xong.
# - Bảng sẽ co giãn đẹp, số, ngày luôn chuẩn Việt Nam.
# - Nếu cần bổ sung tô màu hoặc icon vào ô, báo mình nhé!

    # ======= DEBUG ngày hệ thống, ngày max =============
    # st.write(f"Dữ liệu lớn nhất: {max_date} | Ngày hệ thống: {today}")

import numpy as np

def to_float(val):
    if isinstance(val, str):
        val = val.replace(",", ".").replace(" ", "")
    try:
        return float(val)
    except Exception:
        return np.nan

def format_change(today, yesterday, unit=""):
    try:
        t, y = to_float(today), to_float(yesterday)
        if np.isnan(t) or np.isnan(y): return ""
        delta = t - y
        emoji = "🟢🔼" if delta > 0 else "🔴🔽"
        sign = "+" if delta > 0 else ""
        return f"({emoji} {sign}{delta:.2f}{unit} {'TĂNG' if delta > 0 else 'GIẢM'})"
    except: return ""

# ========= CACHE & FORMAT =========


def get_enrich_cols(df_enrich):
    cols = ["Date", "BrentClose", "WTIClose", "MOP95"]
    df = df_enrich[cols].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
    return df

def format_vn(x):
    if x is None or pd.isna(x): return ""
    try:
        return "{:,.2f}".format(float(x)).replace(",", "#").replace(".", ",").replace("#", ".")
    except:
        return str(x)

def sign_arrow(val):
    if val is None: return ""
    arrow = "🔼" if val > 0 else "🔽" if val < 0 else "⏺"
    color = "red" if val < 0 else "green" if val > 0 else "gray"
    return f"(<span style='color:{color};font-weight:bold'>{arrow} {'+' if val > 0 else ''}{val:.2f}</span>)"

def to_float(s):
    try:
        if pd.isna(s): return None
        if isinstance(s, (float, int)): return float(s)
        return float(str(s).replace(",", ".").replace(" ", ""))
    except: return None

# ========== HÀM CHÍNH Dự Báo MOPS ==========
def norm_text(s):
    import unicodedata, re
    s = str(s)
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode()
    s = re.sub(r'[^\w]', '', s.lower())  # Chỉ giữ chữ/số
    return s
    
def safe_str(val):
    if val is None:
        return ""
    try:
        s = str(val)
        return s.replace("\n", " ").replace("\r", " ")
    except:
        return ""


    import pandas as pd
    import streamlit as st
    df_kq = df_kq.copy()
    df_kq["Ngày"] = pd.to_datetime(df_kq["Ngày"], errors="coerce")
    mask = df_kq["Ngày"].dt.date == pd.to_datetime(date_selected).date()
    df_today = df_kq[mask]
    st.markdown(f"""<div style="background:#166ff4;padding:10px 18px 8px 18px;border-radius:8px;color:white;font-size:1.07em;font-weight:600;margin-bottom:16px">
    📌 <span style="font-size:1.08em">KẾT QUẢ DỰ BÁO CÁC MÔ HÌNH - NGÀY <b>{pd.to_datetime(date_selected).strftime('%d/%m/%Y')}</b></span>
    </div>""", unsafe_allow_html=True)
    if df_today.empty:
        st.warning("Không có dữ liệu dự báo cho ngày này.")
    else:
        st.dataframe(df_today, use_container_width=True)
  
        # Thống kê stick vàng (BestModel)
        if "BestModel" in df_kq.columns:
            best = df_kq.dropna(subset=["BestModel"])
            thong_ke_md = best["BestModel"].value_counts().to_dict()
            st.markdown("**THỐNG KÊ SỐ LẦN MÔ HÌNH ĐÚNG BESTMODEL (STICK VÀNG)**")
            st.table(pd.DataFrame(list(thong_ke_md.items()), columns=["Mô hình", "Số lần"]))

def find_col(df, keys):
    # keys là list các từ khóa ['xanh', 'stick xanh', ...]
    for col in df.columns:
        for k in keys:
            if k in col.lower().replace(" ", ""):
                return col
    return None

def render_block_stick_stat(df_thongke):
    import streamlit as st

    # Nhận diện cột mô hình (không lấy cột đầu)
    model_names = [c for c in df_thongke.columns if c.lower() not in ['mô hình','mo hinh','model','']]
    label_col = df_thongke.columns[0]  # cột label: Stick đỏ, Stick xanh

    # Lấy giá trị từng dòng
    stick_do = df_thongke.loc[df_thongke[label_col].str.lower().str.contains("đỏ"), model_names].values.flatten().tolist()
    stick_xanh = df_thongke.loc[df_thongke[label_col].str.lower().str.contains("xanh"), model_names].values.flatten().tolist()

    # --- Table STICK ĐỎ ---
    html_do = f"""
    <div style='margin-bottom:3px;'>
      <span style='font-size:1.09em; font-weight:600; color:#d6223c;'>
        <span style='font-size:1.16em;'>📌</span> STICK ĐỎ (Δ Dự báo sát nhất có 📌)
      </span>
    </div>
    <table style='width:100%;border-collapse:separate;border-spacing:0;background:#f1faf4;border-radius:14px 14px 0 0;'>
      <thead>
        <tr>
            {''.join([f"<th style='padding:9px 8px; background:#e6f6e8; color:#185040; font-weight:600; border-radius:10px 10px 0 0;text-align:center'>{m}</th>" for m in model_names])}
        </tr>
      </thead>
      <tbody>
        <tr>
            {''.join([f"<td style='text-align:center;font-weight:bold;color:#d6223c;background:#f6fcf6'>{int(float(v))}</td>" for v in stick_do])}
        </tr>
      </tbody>
    </table>
    """

    # --- Table STICK XANH ---
    html_xanh = f"""
    <div style='margin:13px 0 3px 0;'>
      <span style='font-size:1.09em; font-weight:600; color:#0d980a;'>
        <span style='font-size:1.13em;'>✅</span> STICK XANH (BestModel có ✅)
      </span>
    </div>
    <table style='width:100%;border-collapse:separate;border-spacing:0;background:#fefde7;border-radius:14px 14px 0 0;'>
      <thead>
        <tr>
            {''.join([f"<th style='padding:9px 8px; background:#fff6ce; color:#9d7d18; font-weight:600; border-radius:10px 10px 0 0;text-align:center'>{m}</th>" for m in model_names])}
        </tr>
      </thead>
      <tbody>
        <tr>
            {''.join([f"<td style='text-align:center;font-weight:bold;color:#f2a500;background:#fefde7'>{int(float(v))}</td>" for v in stick_xanh])}
        </tr>
      </tbody>
    </table>
    """

    st.markdown(html_do, unsafe_allow_html=True)
    st.markdown(html_xanh, unsafe_allow_html=True)

def render_block_kq_dubao(df_kq, date_selected):
    import streamlit as st
    import pandas as pd
    import numpy as np

    # === Tìm dòng header ===
    idx_header = df_kq[df_kq.iloc[:,0].astype(str).str.lower().str.contains("ngày")].index
    if len(idx_header) > 0:
        df_sub = df_kq.iloc[idx_header[0]+1:]
        df_sub.columns = df_kq.iloc[idx_header[0]]
        df_sub = df_sub.reset_index(drop=True)
    else:
        df_sub = df_kq.copy()

    # === Lọc ngày hoặc lấy max 10 dòng ===
    if "Ngày" in df_sub.columns:
        df_sub["Ngày"] = pd.to_datetime(df_sub["Ngày"], dayfirst=True, errors="coerce")
        filter_sub = df_sub[df_sub["Ngày"] == pd.to_datetime(date_selected)]
        df_sub = filter_sub if not filter_sub.empty else df_sub.tail(10)
    else:
        df_sub = df_sub.tail(10)

    # ==== Format số và màu ====
    def format_vn_number(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        try:
            val = float(val)
            s = f"{val:,.2f}".replace(",", ".")
            if val > 0:
                return f"<span style='color:#188544;font-weight:bold'>{s}</span>"
            elif val < 0:
                return f"<span style='color:#d6223c;font-weight:bold'>{s}</span>"
            else:
                return f"{s}"
        except:
            # Nếu là ngày hoặc text
            if str(val).lower() in ["nan", "nat", "none", ""]:
                return ""
            return str(val)

    # ==== Build bảng HTML ====
    headers = df_sub.columns.tolist()
    html = """
    <table class="tbl-block">
    <thead><tr>""" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr></thead><tbody>"

    for _, row in df_sub.iterrows():
        html += "<tr>"
        for h in headers:
            cell = row[h]
            # Nếu là số thì format, nếu không thì giữ nguyên
            try:
                cell_fmt = format_vn_number(cell)
            except:
                cell_fmt = str(cell) if cell is not None else ""
            html += f"<td>{cell_fmt}</td>"
        html += "</tr>"
    html += "</tbody></table>"

    st.markdown("<b>📝 Kết quả dự báo từng ngày:</b>", unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

def render_tab_mops(df_mops, df_enrich, df_kq, df_thongke, df_log):
    import numpy as np
    import streamlit as st
    import pandas as pd

    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
    .tbl-info-header, .tbl-block, .tbl-stick, .tbl-models {
        width: 100%; border-collapse: collapse; background: #f7fafd; margin-bottom: 16px; border-radius: 14px;
        font-size: 1.13em; overflow: hidden; box-shadow: 0 2px 7px rgba(30,60,150,.09);
    }
    .tbl-info-header td, .tbl-info-header th,
    .tbl-block td, .tbl-block th,
    .tbl-stick td, .tbl-stick th,
    .tbl-models td, .tbl-models th {
        border: none; padding: 9px 7px; font-size: 1.09em;
    }
    .tbl-info-header .label, .tbl-block .label, .tbl-stick .label {
        color: #224488; font-weight: 600; white-space:nowrap;
    }
    .tbl-info-header .val, .tbl-block .val, .tbl-stick .val, .tbl-models .val {
        font-weight: bold; color: #183b73;
    }
    .tbl-info-header .delta, .tbl-block .delta {
        font-weight: bold; font-size: 1.09em;
    }
    .tbl-stick th, .tbl-models th { background:#e5ecfa; color:#15477a; text-align:center;}
    .tbl-block th {background:#fae9e5; color:#b95216;}
    .tbl-models th {background:#e5ecfa; color:#1d477a;}
    .tbl-stick tr:last-child td, .tbl-models tr:last-child td { border-bottom:none; }
    .delta-green {color: #188544 !important; font-weight: bold;}
    .delta-red {color: #d6223c !important; font-weight: bold;}
    @media (max-width:600px){
        .tbl-info-header td, .tbl-info-header th,
        .tbl-block td, .tbl-block th,
        .tbl-stick td, .tbl-stick th,
        .tbl-models td, .tbl-models th {font-size:1em;padding:6px 3px;}
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## MÔ HÌNH DỰ BÁO GIÁ PLATTS XĂNG 95:")

    df = df_mops.copy()
    df["RunTime"] = pd.to_datetime(df["RunTime"], dayfirst=True, errors='coerce')
    df = df[df["RunTime"].notna()]
    df = df.sort_values("RunTime", ascending=True)
    df["date"] = df["RunTime"].dt.date

    # --- Sidebar: chọn ngày ---
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

    runtime_options = df_ngay["RunTime"].dt.strftime("%H:%M:%S").tolist()
    runtime_map = dict(zip(runtime_options, df_ngay.index))
    runtime_selected = st.selectbox("Chọn thời điểm (RunTime)", runtime_options, index=len(runtime_options)-1)
    row = df_ngay.loc[runtime_map[runtime_selected]]

    # ==== Các hàm phụ ====
    def to_float(val):
        try:
            if pd.isna(val): return None
            if isinstance(val, (float, int)): return float(val)
            return float(str(val).replace(",", ".").replace(" ", ""))
        except: return None

    def format_vn(val, dec=2, color_delta=False):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        try:
            val = float(val)
            s = f"{val:,.{dec}f}".replace(",", ".")
            if color_delta:
                if val < 0:
                    return f"<span class='delta-red'>{s}</span>"
                elif val > 0:
                    return f"<span class='delta-green'>{s}</span>"
                else:
                    return s
            return s
        except:
            return str(val)

    def sign_arrow(val):
        if val is None: return ""
        try: val = float(val)
        except: return ""
        arrow = "🔼" if val > 0 else "🔽" if val < 0 else "⏺"
        color = "#188544" if val > 0 else "#d6223c" if val < 0 else "#787878"
        sign = "+" if val > 0 else ""
        sval = f"{val:.2f}".replace("-", "")
        return f"<span style='color:{color};font-weight:bold'>{arrow} {sign}{sval}</span>"

    def safe_str(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return str(x)

    # Brent/WTI hôm qua từ train_mop_enrich
    def get_oil_enrich_yesterday(df_enrich, date_selected, col_gia, col_date='Date'):
        dfk = df_enrich.copy()
        dfk["Date"] = pd.to_datetime(dfk[col_date], dayfirst=True, errors='coerce').dt.date
        dfk = dfk.sort_values("Date")
        dates = dfk["Date"].dropna().unique()
        if len(dates) == 0:
            return None, None
        check = date_selected - pd.Timedelta(days=1)
        while (check.weekday() >= 5 or check not in dates) and check >= min(dates):
            check -= pd.Timedelta(days=1)
        if check in dates:
            row = dfk[dfk["Date"] == check].iloc[-1]
            return row.get(col_gia, None), check
        return None, None

    # Brent/WTI hôm nay từ log_run_history_2
    def get_oil_today_from_log(df_log, date_selected, col_gia, col_ngay='RunTime'):
        dfk = df_log.copy()
        dfk["RunTime_dt"] = pd.to_datetime(dfk[col_ngay], dayfirst=True, errors='coerce').dt.date
        row_today = dfk[dfk["RunTime_dt"] == date_selected]
        if not row_today.empty:
            return row_today.iloc[-1][col_gia]
        return None

    
    # MOPS hôm qua từ giá thật
    def get_mops_thuc_te_yesterday(df_kq, date_selected):
        if "Ngày" in df_kq.columns:
            df_kq = df_kq.copy()
            df_kq["Ngày_dt"] = pd.to_datetime(df_kq["Ngày"], dayfirst=True, errors='coerce').dt.date
        else:
            return None, None
        cond_gia_that = (df_kq["Nguồn giá MOP95"].astype(str).str.lower().str.contains("giá thật"))
        df_true = df_kq[cond_gia_that].sort_values("Ngày_dt")
        true_dates = df_true["Ngày_dt"].dropna().unique()
        if len(true_dates) == 0:
            return None, None
        check = date_selected - pd.Timedelta(days=1)
        while (check.weekday() >= 5 or check not in true_dates) and check >= min(true_dates):
            check -= pd.Timedelta(days=1)
        if check in true_dates:
            row = df_true[df_true["Ngày_dt"] == check].iloc[-1]
            return row["Giá MOP95 thật"], check
        return None, None

    # ==== Brent/WTI hôm qua (enrich) ====
    brent_yt, brent_yt_date = get_oil_enrich_yesterday(df_enrich, date_selected, col_gia="BrentClose")
    brent_yt_f = to_float(brent_yt)
    brent_yt_show = format_vn(brent_yt_f)

    wti_yt, wti_yt_date = get_oil_enrich_yesterday(df_enrich, date_selected, col_gia="WTIClose")
    wti_yt_f = to_float(wti_yt)
    wti_yt_show = format_vn(wti_yt_f)

    # ==== Brent/WTI hôm nay (log) ====
    brent_today_f = to_float(get_oil_today_from_log(df_log, date_selected, col_gia="BrentClose", col_ngay="RunTime"))
    wti_today_f = to_float(get_oil_today_from_log(df_log, date_selected, col_gia="WTIClose", col_ngay="RunTime"))
    brent_today_show = format_vn(brent_today_f)
    wti_today_show = format_vn(wti_today_f)

    delta_brent_val = brent_today_f - brent_yt_f if brent_today_f is not None and brent_yt_f is not None else None
    delta_wti_val = wti_today_f - wti_yt_f if wti_today_f is not None and wti_yt_f is not None else None

    # ==== MOPS hôm qua (giá thật) ====
    mops_yt, mops_yt_date = get_mops_thuc_te_yesterday(df_kq, date_selected)
    mops_yt_f = to_float(mops_yt)
    mops_yt_show = format_vn(mops_yt_f)

    run_time_str = row["RunTime"].strftime('%d/%m/%Y %H:%M:%S') if pd.notnull(row["RunTime"]) else "N/A"

    table_info = f"""
    <table class="tbl-info-header">
      <tr>
        <td class="label">⏰ Thời điểm dự báo:</td>
        <td class="val">{safe_str(run_time_str)}</td>
        <td></td>
      </tr>
      <tr>
        <td class="label">Brent hôm nay:</td>
        <td class="val" style="color:#347d39;">{safe_str(brent_today_show)}</td>
        <td class="label">Hôm qua: <span style="color:#7c7c7c;">{safe_str(brent_yt_show)}</span>
          <span style="color:#888;font-size:0.99em">{f"({brent_yt_date.strftime('%d/%m/%Y')})" if brent_yt_date else ""}</span>
          {sign_arrow(delta_brent_val)}
        </td>
      </tr>
      <tr>
        <td class="label">WTI hôm nay:</td>
        <td class="val" style="color:#a74814;">{safe_str(wti_today_show)}</td>
        <td class="label">Hôm qua: <span style="color:#7c7c7c;">{safe_str(wti_yt_show)}</span>
          <span style="color:#888;font-size:0.99em">{f"({wti_yt_date.strftime('%d/%m/%Y')})" if wti_yt_date else ""}</span>
          {sign_arrow(delta_wti_val)}
        </td>
      </tr>
      <tr>
        <td class="label">PLATTS XĂNG 95 hôm qua:</td>
        <td class="val" style="color:#336699;">{safe_str(mops_yt_show)}</td>
        <td style="color:#666;font-size:0.98em">
            {f"({mops_yt_date.strftime('%d/%m/%Y')})" if mops_yt_date else ""}
        </td>
      </tr>
    </table>
    """
    st.markdown(table_info, unsafe_allow_html=True)

    # ==== BLOCK 4: Khoảng dự báo và bảng mô hình ====
    models = ["LinearRegression", "XGBoost", "RandomForest", "Polynomial"]

    def get_delta_from_kq(df_kq, model, date_selected):
        if "Ngày" in df_kq.columns and "Model" in df_kq.columns:
            row = df_kq[(df_kq["Ngày"] == date_selected.strftime("%d/%m/%Y")) & (df_kq["Model"] == model)]
            if not row.empty:
                try:
                    return float(str(row["Δ Dự báo"].values[0]).replace(",", "."))
                except:
                    return None
        return None

    icons = {
        "LinearRegression": "🟦", "XGBoost": "🟧", "RandomForest": "🟩", "Polynomial": "🟪"
    }
    colors = {
        "RandomForest": "#2270d6", "Polynomial": "#6b34ad", "XGBoost": "#b86818", "LinearRegression": "#357824"
    }
    table_rows = ""
    pred_values = []
    for m in models:
        delta = get_delta_from_kq(df_kq, m, date_selected)
        if delta is None or mops_yt_f is None:
            continue
        pred = mops_yt_f + delta
        pred_values.append(pred)
        icon = icons.get(m, "")
        color = colors.get(m, "#444")
        pred_s = format_vn(pred)
        delta_s = format_vn(delta, color_delta=True)
        arrow = "🔺 Tăng" if delta > 0 else "🔻 Giảm" if delta < 0 else "⏺ Không đổi"
        table_rows += f"""
        <tr>
            <td style="text-align:center;font-size:1.15em">{icon}</td>
            <td style="font-weight:700;color:{color};">{safe_str(m)}</td>
            <td style="text-align:right;font-weight:bold">{pred_s}</td>
            <td style="text-align:right;">{delta_s} <span style='font-size:0.93em'>{arrow}</span></td>
        </tr>
        """

    pred_range_min = min(pred_values) if pred_values else None
    pred_range_max = max(pred_values) if pred_values else None
    delta_pred_min = pred_range_min - mops_yt_f if pred_range_min is not None and mops_yt_f is not None else None
    delta_pred_max = pred_range_max - mops_yt_f if pred_range_max is not None and mops_yt_f is not None else None

    if pred_range_min is not None and pred_range_max is not None and mops_yt_f is not None:
        st.markdown(
            f"""<table class="tbl-block">
            <thead><tr><th colspan="2">🚦 Khoảng dự báo giá Platts xăng 95</th></tr></thead>
            <tbody>
            <tr>
                <td><b>Khoảng dự báo:</b></td>
                <td><span style='color:#2A62B8;font-weight:bold;font-size:1.4rem'>
                {format_vn(pred_range_min)} – {format_vn(pred_range_max)}
                </span> đ/lít</td>
            </tr>
            <tr>
                <td><b>Chênh lệch với giá MOPS hôm qua:</b></td>
                <td>{format_vn(delta_pred_min, color_delta=True)} đến {format_vn(delta_pred_max, color_delta=True)}</td>
            </tr>
            </tbody></table>
            """,
            unsafe_allow_html=True
        )
    elif pred_range_min is not None and pred_range_max is not None:
        st.markdown(
            f"""<table class="tbl-block">
            <thead><tr><th colspan="2">🚦 Khoảng dự báo giá Platts xăng 95 (MOPS95)</th></tr></thead>
            <tbody>
            <tr>
                <td colspan="2"><span style='color:#2A62B8;font-weight:bold;font-size:1.4rem'>
                {format_vn(pred_range_min)} – {format_vn(pred_range_max)}
                </span> đ/lít</td>
            </tr>
            </tbody></table>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <table class="tbl-models">
          <thead>
            <tr>
              <th></th>
              <th>Mô hình</th>
              <th>Dự báo ($/thùng)</th>
              <th>Δ Dự báo</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
        """, unsafe_allow_html=True)
    
    # ==== BLOCK 2: Thống kê STICK ====
    render_block_stick_stat(df_thongke)

    # ==== BLOCK 3: Kết quả dự báo từng ngày ====
    render_block_kq_dubao(df_kq, date_selected)
    
    # ==== BLOCK 6: Bảng tổng hợp mô hình (giá đã hiệu chỉnh Delta) ====
    model_lines_html = ""
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
            if delta < 0:
                pred_adj = pred + abs(delta)
                delta_html = f"<span class='delta-red'>-{format_vn(abs(delta))}</span>"
                arrow = "🔻 Giảm"
            else:
                pred_adj = pred - abs(delta)
                delta_html = f"<span class='delta-green'>+{format_vn(abs(delta))}</span>"
                arrow = "🔺 Tăng"
        except Exception:
            continue
        model_lines_html += f"""
        <tr>
            <td style="font-weight:700">{safe_str(m)}</td>
            <td style="text-align:right;font-weight:bold">{format_vn(pred_adj)}</td>
            <td style="text-align:right">{delta_html}</td>
            <td style="text-align:right">{arrow}</td>
            <td style="text-align:right">{format_vn(mae)}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="tbl-models">
          <thead>
            <tr>
              <th>Mô hình</th>
              <th>Dự báo (hiệu chỉnh Delta)</th>
              <th>Delta</th>
              <th>Chiều</th>
              <th>MAE</th>
            </tr>
          </thead>
          <tbody>
            {model_lines_html}
          </tbody>
        </table>
        """, unsafe_allow_html=True)

    # --- (Tuỳ chọn) Hiện bảng chi tiết dataframe ngày này ---
    with st.expander("Bảng dữ liệu chi tiết ngày này"):
        df_show = df_ngay.replace([np.nan, None], "")
        st.dataframe(df_show, use_container_width=True)

def load_kq_and_stat(sheet_name="kq_dubao"):
    df_raw = load_sheet(sheet_name).reset_index(drop=True)
    # Tìm header của data chi tiết (có cột "Ngày")
    header_row_idx = None
    for i in range(len(df_raw)):
        vals = [str(x).strip() for x in df_raw.iloc[i].tolist()]
        if "Ngày" in vals or "ngày" in [x.lower() for x in vals] or "Date" in vals:
            header_row_idx = i
            break
    if header_row_idx is None:
        return pd.DataFrame(), pd.DataFrame()
    # Phần thống kê phía trên
    df_stat = df_raw.iloc[:header_row_idx].copy()
    # Auto tìm dòng có "Mô hình"/"Model"
    row_idx_model = None
    for idx in df_stat.index:
        val = str(df_stat.iloc[idx, 0]).lower()
        if val in ["mô hình", "model"]:
            row_idx_model = idx
            break
    if row_idx_model is not None:
        header_row = df_stat.iloc[row_idx_model].tolist()
        # Xác định số cột thực (bỏ NaN cuối)
        n_col_valid = len([x for x in header_row if pd.notna(x) and str(x).strip() != ""])
        df_thongke = df_stat.iloc[row_idx_model+1:row_idx_model+3, :n_col_valid].copy()
        df_thongke.columns = header_row[:n_col_valid]
        df_thongke = df_thongke.reset_index(drop=True)
    else:
        df_thongke = pd.DataFrame()
    # Data chi tiết
    df_kq = df_raw.iloc[header_row_idx:].copy()
    df_kq.columns = df_kq.iloc[0]
    df_kq = df_kq[1:].reset_index(drop=True)
    return df_kq, df_thongke


def find_date_column(cols):
    for name in cols:
        n = name.strip().lower().replace(' ', '')
        if n in ['ngay', 'date', 'ngày']:
            return name
    return None

######################################################## TAB MAE
import streamlit as st
import pandas as pd
import plotly.express as px

def render_tab_mae(df_rolling):
    import streamlit as st
    import plotly.express as px
    import pandas as pd
    import numpy as np

    st.header("📉 Sai số dự báo (MAE/Rolling)")

    # ==== Hàm HTML Table tiện ích ====
    def render_html_table(df, header_color="#eaf3fa", border="#bfe0ff", even_color="#f7fbff"):
        def format_vn_number(x, dec=2):
            try:
                if pd.isnull(x) or x == "":
                    return ""
                x = float(str(x).replace(",", "."))
                if dec == 0:
                    s = f"{int(round(x, 0)):,}".replace(",", ".")
                else:
                    s = f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")
                return s
            except Exception:
                return x

        df_view = df.copy()
        # Format số cho các cột MAE, số liệu
        for col in df_view.columns:
            if "mae" in col.lower() or "sai" in col.lower():
                df_view[col] = df_view[col].apply(lambda x: format_vn_number(x, 2))
        df_view = df_view.replace(["nan", "None", None, pd.NA, np.nan], "")

        st.markdown(f"""
        <style>
        .mae-table {{
            font-size: 15px !important;
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
            margin-bottom: 12px;
            background: #fff;
            box-shadow: 0 2px 14px rgba(0,0,0,0.08);
            border-radius: 14px;
            overflow: hidden;
        }}
        .mae-table th, .mae-table td {{
            border: 1px solid {border};
            padding: 8px 7px;
            text-align: center;
            vertical-align: middle;
        }}
        .mae-table th {{
            background: {header_color};
            font-weight: 600;
            color: #29518a;
        }}
        .mae-table tr:nth-child(even) td {{
            background: {even_color};
        }}
        .mae-table tr:hover td {{
            background: #eaf0fb;
        }}
        @media (max-width: 700px) {{
            .mae-table th, .mae-table td {{ padding: 5px 1px; font-size:0.97em;}}
        }}
        </style>
        """, unsafe_allow_html=True)

        # Tạo HTML
        html = '<div style="overflow-x:auto;"><table class="mae-table">'
        html += "<thead><tr>"
        for col in df_view.columns:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"
        for _, row in df_view.iterrows():
            html += "<tr>"
            for col in df_view.columns:
                html += f"<td>{row[col]}</td>"
            html += "</tr>"
        html += "</tbody></table></div>"
        st.markdown(html, unsafe_allow_html=True)

    # ==== Bắt đầu xử lý dữ liệu ====
    df_rolling = df_rolling.copy()
    for col in ["RunTime", "MAE_Rolling"]:
        if col not in df_rolling.columns:
            st.error(f"Không tìm thấy cột {col} trong dữ liệu!")
            return

    df_rolling["RunTime"] = pd.to_datetime(df_rolling["RunTime"], dayfirst=True, errors='coerce')
    df_rolling = df_rolling[df_rolling["RunTime"].notna()]

    if df_rolling.empty:
        st.warning("Không có dữ liệu MAE để hiển thị!")
        return

    min_date = df_rolling["RunTime"].min().date()
    max_date = df_rolling["RunTime"].max().date()

    date_range = st.sidebar.date_input(
        "Chọn khoảng ngày MAE",
        [min_date, max_date],
        min_value=min_date, max_value=max_date,
        key="mae"
    )
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 1:
            date_range = [date_range[0], date_range[0]]
    else:
        date_range = [date_range, date_range]

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df_view = df_rolling[(df_rolling["RunTime"] >= start_date) & (df_rolling["RunTime"] < end_date)]

    if "Model" in df_rolling.columns and not df_rolling["Model"].isnull().all():
        models = df_rolling["Model"].dropna().unique().tolist()
        selected_models = st.multiselect("Chọn mô hình (MAE)", models, default=models, key="model_mae")
        if selected_models:
            df_view = df_view[df_view["Model"].isin(selected_models)]
        else:
            st.warning("Chưa chọn mô hình nào.")
            return

    if df_view.empty:
        st.warning("Không có dữ liệu trong khoảng thời gian đã chọn.")
        return

    # Format RunTime đẹp hơn cho chart
    df_view = df_view.copy()
    df_view["RunTime_fmt"] = df_view["RunTime"].dt.strftime('%d/%m %H:%M')

    fig = px.bar(
        df_view, x="RunTime_fmt", y="MAE_Rolling", color="Model" if "Model" in df_view.columns else None, barmode="group",
        title="Rolling MAE theo từng mô hình"
    )
    fig.update_layout(xaxis_title="Thời gian", yaxis_title="MAE (Rolling)", font=dict(size=13))
    st.plotly_chart(fig, use_container_width=True)

     # ==== Hiển thị bảng đẹp ====
    st.markdown("**Dữ liệu MAE gần nhất:**")
    df_show = df_view.sort_values("RunTime", ascending=False).head(10).copy()
    if "RunTime_fmt" in df_show.columns:
        df_show = df_show.drop(columns=["RunTime_fmt"])
    render_html_table(df_show)


########################################################

########################################################Tab gia co so
import streamlit as st
import pandas as pd
import datetime
import pytz

def render_block(title, gia_le, gcs, chenhlech, chenhlech_msg, note=None, bg="#fff"):
    st.markdown(f"""
    <div style="
        background: {bg};
        border-radius: 18px;
        box-shadow: 0 2px 10px #0001;
        padding: 18px 18px 14px 18px;
        margin-bottom: 18px;
        border-left: 7px solid #1976d2;">
        <div style="font-size:21px; font-weight:700; color:#1956a6; margin-bottom:8px;">
            {title}
        </div>
        <div style="margin:2px 0 0 0;font-size:19px;">
            <b>• GIÁ BÁN LẺ:</b>
            <span style='color:#19832d;font-weight:700;font-size:22px'>{gia_le} ₫</span>
        </div>
        <div style="margin:2px 0 0 0;font-size:19px;">
            <b>• GCS:</b>
            <span style='color:#2270d6;font-weight:700;font-size:22px'>{gcs} ₫</span>
        </div>
        <div style="margin:2px 0 0 0;font-size:19px;">
            <b>• Chênh lệch:</b>
            <span style='color:#f2790c;font-weight:800;font-size:22px'>{chenhlech} ₫</span>
        </div>
        <div style="margin-top:6px;font-size:17px;">{chenhlech_msg}</div>
        {f'<div style="margin-top:8px;font-size:15px;color:#666;">{note}</div>' if note else ''}
    </div>
    """, unsafe_allow_html=True)

def format_vn_number(x, dec=0):
    try:
        if pd.isnull(x) or x == "" or str(x).lower() == "none":
            return ""
        x = float(str(x).replace(",", "."))
        if dec == 0:
            s = f"{int(round(x, 0)):,}".replace(",", ".")
        else:
            s = f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")
        return s
    except Exception:
        return ""

def chenh_lech_msg(delta):
    try:
        delta = float(delta)
        if delta > 0:
            return "<span style='color:#d6223c;font-weight:600'>📉 GIẢM GIÁ BÁN LẺ</span>"
        elif delta < 0:
            return "<span style='color:#1c8622;font-weight:600'>📈 TĂNG GIÁ BÁN LẺ</span>"
        else:
            return "<span style='color:#787878;font-weight:600'>- GIỮ NGUYÊN -</span>"
    except:
        return "..."

def render_tab_gcs_chuky(df_gcs, df_main=None):
    import datetime, pytz
    import pandas as pd
    import streamlit as st
    import numpy as np

    # ========== Tiện ích format số/ngày/chênh lệch ==========
    def format_vn_number(x, dec=0, color_delta=True):
        try:
            if pd.isnull(x) or x in ["", "nan", "NaN", "None", "NaT"]:
                return ""
            val = float(str(x).replace(",", "."))
            if dec == 0:
                s = f"{int(round(val, 0)):,}".replace(",", ".")
            else:
                s = f"{val:,.{dec}f}".replace(",", ".")
            if color_delta:
                if val > 0:
                    return f"<span style='color:#188544;font-weight:bold'>{s}</span>"
                elif val < 0:
                    return f"<span style='color:#d6223c;font-weight:bold'>{s}</span>"
                else:
                    return s
            else:
                return s
        except:
            if str(x).lower() in ["nan", "nat", "none", ""]:
                return ""
            return str(x)

    def chenh_lech_msg(delta):
        try:
            delta = float(str(delta).replace(",", "."))
            if delta > 0:
                return "<span style='color:#d6223c;font-weight:600'>📉 GIẢM GIÁ BÁN LẺ</span>"
            elif delta < 0:
                return "<span style='color:#1c8622;font-weight:600'>📈 TĂNG GIÁ BÁN LẺ</span>"
            else:
                return "<span style='color:#565'>—</span>"
        except:
            return "<span style='color:#565'>—</span>"

    # ========== Hàm render HTML Table ==========
    def render_html_table(df, header_color="#eaf3fa", border="#bfe0ff", even_color="#f7fbff"):
        df_view = df.copy()
        # Bôi màu số
        for col in df_view.columns:
            if any(key in col.upper() for key in [
                "GIALE", "GCS", "MOPSTB7", "MOP95", "CHÊNH", "PRICE", "RATEVCB", "USD", "Brent", "WTI", "XANG", "FIX"
            ]):
                try:
                    # Bôi màu delta (có thể phân biệt loại cột nếu cần)
                    df_view[col] = df_view[col].apply(lambda x: format_vn_number(x, 2, color_delta=True))
                except: pass
        df_view = df_view.replace(["nan", "None", None, pd.NA, "NaN", "NaT"], "")

        st.markdown(f"""
        <style>
        .custom-table {{
            font-size: 15px !important;
            width: 100%;
            border-collapse: collapse;
            margin-top: 6px;
            margin-bottom: 14px;
            background: #fff;
            box-shadow: 0 2px 14px rgba(0,0,0,0.08);
            border-radius: 14px;
            overflow: hidden;
        }}
        .custom-table th, .custom-table td {{
            border: 1px solid {border};
            padding: 8px 7px;
            text-align: center;
            vertical-align: middle;
        }}
        .custom-table th {{
            background: {header_color};
            font-weight: 600;
            color: #29518a;
        }}
        .custom-table tr:nth-child(even) td {{
            background: {even_color};
        }}
        .custom-table tr:hover td {{
            background: #eaf0fb;
        }}
        @media (max-width: 700px) {{
            .custom-table th, .custom-table td {{ padding: 5px 1px; font-size:0.97em;}}
        }}
        </style>
        """, unsafe_allow_html=True)

        html = '<div style="overflow-x:auto;"><table class="custom-table">'
        html += "<thead><tr>"
        for col in df_view.columns:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"
        for _, row in df_view.iterrows():
            html += "<tr>"
            for col in df_view.columns:
                html += f"<td>{row[col]}</td>"
            html += "</tr>"
        html += "</tbody></table></div>"
        st.markdown(html, unsafe_allow_html=True)

    # ========== CHUẨN HÓA DATA ==========
    df = df_gcs.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df[df["Date"].notna()]
    df = df.sort_values("Date").reset_index(drop=True)
    tz_vn = pytz.timezone("Asia/Ho_Chi_Minh")
    today_vn = datetime.datetime.now(tz_vn).date()

    # ========== XÁC ĐỊNH CHU KỲ ==========
    df['GIALE_shift'] = df['GIALE'].shift(1)
    change_idx = df['GIALE'].ne(df['GIALE_shift']).cumsum()

    df_today = df[df["Date"].dt.date == today_vn]
    if not df_today.empty:
        row_today = df_today.iloc[-1]
        today_idx = row_today.name
        today_cy = change_idx.iloc[today_idx]
    else:
        df_before = df[df["Date"].dt.date < today_vn]
        if df_before.empty:
            st.error("Không có dữ liệu cho hôm nay hoặc trước hôm nay.")
            return
        row_today = df_before.iloc[-1]
        today_idx = row_today.name
        today_cy = change_idx.iloc[today_idx]
        today_vn = row_today["Date"].date()
    today_str = row_today["Date"].strftime("%d/%m/%Y")

    cy_mask = (change_idx == today_cy)
    cy_df = df[cy_mask]
    row_cy_start = cy_df.iloc[0]
    row_cy_end = cy_df.iloc[-1]

    next_cy = today_cy + 1
    next_cy_mask = (change_idx == next_cy)
    next_cy_df = df[next_cy_mask]
    row_next = next_cy_df.iloc[0] if not next_cy_df.empty else None
    row_next_end = next_cy_df.iloc[-1] if not next_cy_df.empty else None

    # ========== GOM BLOCK DỰ BÁO THÀNH 1 BẢNG ==========
    blocks = []
    # Block hôm nay
    blocks.append({
        "Loại báo cáo": "📅 Hôm nay",
        "Chu kỳ / Ngày": row_today["Date"].strftime("%d/%m/%Y"),
        "Giá bán lẻ": format_vn_number(row_today.get('GIALE')),
        "Giá GCS": format_vn_number(row_today.get('GCS')),
        "Chênh lệch": format_vn_number(row_today.get('Chênh lệch'), color_delta=True),
        "Nhận định": chenh_lech_msg(row_today.get('Chênh lệch')),
    })
    # Block chu kỳ hiện tại
    cy_str = f"{row_cy_start['Date'].strftime('%d/%m/%Y')} → {row_cy_end['Date'].strftime('%d/%m/%Y')}"
    blocks.append({
        "Loại báo cáo": "📊 Chu kỳ hiện tại",
        "Chu kỳ / Ngày": cy_str,
        "Giá bán lẻ": format_vn_number(row_cy_end.get('GIALE')),
        "Giá GCS": format_vn_number(row_cy_end.get('GCS')),
        "Chênh lệch": format_vn_number(row_cy_end.get('Chênh lệch'), color_delta=True),
        "Nhận định": chenh_lech_msg(row_cy_end.get('Chênh lệch')),
    })
    # Block kỳ tới (nếu có)
    if row_next is not None and row_next_end is not None:
        next_cy_str = f"{row_next['Date'].strftime('%d/%m/%Y')} → {row_next_end['Date'].strftime('%d/%m/%Y')}"
        blocks.append({
            "Loại báo cáo": "🧮 Kỳ tới",
            "Chu kỳ / Ngày": next_cy_str,
            "Giá bán lẻ": format_vn_number(row_next.get('GIALE')),
            "Giá GCS": format_vn_number(row_next.get('GCS')),
            "Chênh lệch": format_vn_number(row_next.get('Chênh lệch'), color_delta=True),
            "Nhận định": chenh_lech_msg(row_next.get('Chênh lệch')),
        })

    # ========== RENDER BLOCK BẢNG DỰ BÁO ==========
    df_blocks = pd.DataFrame(blocks)
    st.markdown("""
    <style>
    .block-table {
        font-size: 1.07em; border-collapse: collapse; width: 100%;
        background: #fff; box-shadow: 0 2px 12px #9bc7e321; border-radius: 14px; overflow: hidden;
        margin-bottom: 18px;
    }
    .block-table th, .block-table td {
        border: 1px solid #d6e6f5; padding: 8px 10px; text-align: center; vertical-align: middle;
    }
    .block-table th {
        background: #eaf7ff; font-weight: 700; color: #0e7490;
    }
    .block-table tr:nth-child(even) td { background: #f7fbff; }
    .block-table td span { font-size: 1em; }
    </style>
    """, unsafe_allow_html=True)

    html = '<div style="overflow-x:auto;"><table class="block-table">'
    html += "<thead><tr>" + "".join([f"<th>{col}</th>" for col in df_blocks.columns]) + "</tr></thead><tbody>"
    for _, row in df_blocks.iterrows():
        html += "<tr>"
        for col in df_blocks.columns:
            html += f"<td>{row[col]}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

    # ==== Bảng dữ liệu chi tiết ====
    with st.expander("Bảng dữ liệu chi tiết"):
        show_df = df.drop(columns=["GIALE_shift"])
        show_df["Date"] = show_df["Date"].dt.strftime("%d/%m/%Y")
        render_html_table(show_df)

    # ==== Bảng lịch sử giá 14 ngày trước chu kỳ hiện tại ====
    if (df_main is not None) and (row_cy_start is not None):
        dfm = df_main.copy()
        dfm["Date"] = pd.to_datetime(dfm["Date"], errors="coerce")
        date_start = row_cy_start["Date"]
        df_hist = dfm[dfm["Date"] < date_start].sort_values("Date", ascending=False).head(14)
        if not df_hist.empty:
            st.markdown("#### 🕑 Lịch sử giá 14 ngày trước đầu kỳ hiện tại")
            df_hist["Date"] = df_hist["Date"].dt.strftime("%d/%m/%Y")
            df_hist.columns = [str(c).strip() for c in df_hist.columns]
            # Loại cột không cần
            cols_to_drop = [
                "BrentAvg", "BrentHigh", "BrentLow",
                "WTIAvg", "WTIHigh", "WTILow",
                "BrentClose", "WTIClose", "GCS_fix",
                "MOP95", "MOPSTB7", "USDTB7", "GCS_chuan", "RATEVCB"
            ]
            cols_to_drop_lower = [c.lower() for c in cols_to_drop]
            df_hist = df_hist[[c for c in df_hist.columns if c.lower() not in cols_to_drop_lower]]
            # Format số và màu
            for col in df_hist.columns:
                if col == "Date": continue
                df_hist[col] = df_hist[col].apply(lambda x: format_vn_number(x, 2, color_delta=True))
            render_html_table(df_hist)
        else:
            st.info("Không đủ dữ liệu để hiển thị lịch sử giá 14 ngày trước chu kỳ hiện tại.")
    else:
        st.info("Không có dữ liệu lịch sử giá (df_main).")


#################################################################
def render_tab_14days(df):
    import streamlit as st
    import pandas as pd
    import numpy as np

    # --- CSS custom cho bảng đẹp, rõ khác biệt, hỗ trợ mobile ---
    st.markdown("""
    <style>
    .table14-custom {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1.06em;
        width: 100%;
        border-collapse: collapse;
        margin-top: 14px;
        box-shadow: 0 4px 32px 0 rgba(0,0,0,0.10);
        border-radius: 16px;
        overflow: hidden;
        background: #fff;
    }
    .table14-custom th, .table14-custom td {
        border: 1px solid #f2f2f2;
        padding: 10px 5px;
        text-align: center;
        vertical-align: middle;
        min-width: 72px;
    }
    .table14-custom th {
        background: #e3f2fd;
        font-weight: bold;
        color: #1a345c;
        border-bottom: 2.5px solid #b5dafd;
        letter-spacing: 0.02em;
    }
    .table14-custom tr:nth-child(even) td {
        background: #f7fbff;
    }
    .table14-custom tr:hover td {
        background: #e8f4fd;
    }
    .table14-custom td.negative { color: #d6223c; font-weight: 700; background: #fbe9eb; }
    .table14-custom td.positive { color: #18823c; font-weight: 700; background: #eafaf0; }
    .table14-custom td.zero { color: #636363; background: #f9f9f9; }
    .table14-custom td { font-size: 1em; }
    @media (max-width: 700px) {
        .table14-custom th, .table14-custom td { padding: 6px 1px; font-size:0.97em;}
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("## 📋 Giá cơ sở 14 ngày gần nhất")

    # Chuẩn hóa số/ngày
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date", ascending=False)
    show_cols = [
        "Date", "MOP95", "RATEVCB", "GIALE", "GCS_CHENHLECH",
        "PRICEBCT", "xangdau", "GCS_chenhlech", "Chenhlech", "GCS_heso"
    ]
    df_view = df[show_cols].head(14).copy()
    df_view["Date"] = df_view["Date"].dt.strftime("%d/%m/%Y")

    def format_vn_number(x, dec=0):
        try:
            if pd.isnull(x) or x == "":
                return ""
            x = float(str(x).replace(",", "."))
            if dec == 0:
                s = f"{int(round(x, 0)):,}".replace(",", ".")
            else:
                s = f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")
            return s
        except Exception:
            return ""

    cols_int = ["MOP95", "RATEVCB", "GIALE", "GCS_CHENHLECH", "PRICEBCT",
                "GCS_chenhlech", "Chenhlech", "GCS_heso", "xangdau"]
    for col in cols_int:
        if col in df_view.columns:
            df_view[col] = df_view[col].apply(lambda x: format_vn_number(x, 0))
    df_view = df_view.replace(["nan", "None", None, pd.NA, np.nan], "")

    # === Tô màu theo giá trị (dương/âm/0) ===
    def highlight_class(val):
        try:
            val_ = float(str(val).replace(".", "").replace(",", "."))
            if val_ < 0: return "negative"
            if val_ > 0: return "positive"
            return "zero"
        except:
            return ""

    # === Tạo bảng HTML ===
    html = '<table class="table14-custom">'
    # Header
    html += "<thead><tr>"
    for col in df_view.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead>\n<tbody>"

    # Rows
    for i, row in df_view.iterrows():
        html += "<tr>"
        for col in df_view.columns:
            val = row[col]
            cell = str(val)
            cell_class = ""
            if col in ["Chenhlech", "GCS_chenhlech"]:
                cell_class = highlight_class(val)
            html += f'<td class="{cell_class}">{cell}</td>'
        html += "</tr>\n"
    html += "</tbody></table>"

    st.markdown(html, unsafe_allow_html=True)
    st.caption("🔴 Giá âm: màu đỏ nền hồng | 🟢 Giá dương: màu xanh nền nhạt | Ngày, số đã định dạng chuẩn VN")

################################################################################
def render_tab_log_kq(df_kq):
    import pandas as pd
    import streamlit as st
    import numpy as np

    # --- Format số Việt Nam có màu ---
    def format_vn_number(x, dec=2, color=True):
        try:
            if pd.isnull(x) or x in ["", "nan", "NaN", "None", "NaT"]:
                return ""
            val = float(str(x).replace(",", "."))
            s = f"{val:,.{dec}f}".replace(",", ".")
            if color:
                if val > 0:
                    return f"<span style='color:#188544;font-weight:bold'>{s}</span>"
                elif val < 0:
                    return f"<span style='color:#d6223c;font-weight:bold'>{s}</span>"
                else:
                    return s
            else:
                return s
        except:
            # Trả về gốc nếu không phải số
            if str(x).lower() in ["nan", "nat", "none", ""]:
                return ""
            return str(x)

    # --- Làm sạch ---
    df_data = df_kq.copy().replace(["nan", "NaN", "None", "NaT"], np.nan).dropna(axis=1, how="all")
    df_data = df_data.fillna("")

    # --- Lọc stick ---
    stick_xanh = df_data[df_data["Δ Dự báo sát nhất"].astype(str).str.contains("📌", na=False)].copy().tail(14)
    stick_vang = df_data[df_data["Sát nhất (BestModel)"].astype(str).str.contains("✅", na=False)].copy().tail(14)

    # --- Thống kê số lần ---
    def thong_ke_stick(df, title):
        models = ["LinearRegression", "Polynomial", "RandomForest", "XGBoost"]
        tk = df['Model'].value_counts().reindex(models, fill_value=0)
        return pd.DataFrame([tk.values], columns=models, index=[title])

    tk_xanh = thong_ke_stick(stick_xanh, "Số lần")
    tk_vang = thong_ke_stick(stick_vang, "Số lần")

    # --- Chuẩn hóa số và màu ---
    def format_df(df):
        new_df = df.copy()
        for c in new_df.columns:
            # Các cột số, delta, sai số, chênh lệch… đều sẽ bôi màu
            if any(k in c.lower() for k in ["giá", "delta", "sai số", "Δ", "chênh"]):
                new_df[c] = new_df[c].apply(lambda x: format_vn_number(x, 2, True))
            else:
                new_df[c] = new_df[c].replace(["nan", "NaN", "None", ""], "")
        return new_df.fillna("")

    stick_xanh = format_df(stick_xanh)
    stick_vang = format_df(stick_vang)

    def df_to_html(df, header_color="#e8faef"):
        if df is None or df.empty:
            return ""
        th_style = f'background:{header_color};font-weight:600;padding:7px 14px;text-align:center;'
        td_style = 'padding:7px 14px;text-align:center;'
        html = '<table style="border-collapse:collapse;width:100%;font-size:16px;border-radius:12px;overflow:hidden;">\n<tr>'
        for col in df.columns:
            html += f'<th style="{th_style}">{col}</th>'
        html += '</tr>\n'
        for _, row in df.iterrows():
            html += "<tr>"
            for val in row:
                html += f'<td style="{td_style}">{val}</td>'
            html += "</tr>\n"
        html += "</table>"
        return html

    # --- Render ---
    st.markdown("""
        <style>
        .kq-header {font-size:22px;font-weight:700;margin-bottom:2px;}
        .kq-badge {display:inline-block;padding:3px 11px;border-radius:8px;font-size:15px;font-weight:600;}
        .kq-xanh {background:#dff5e1;color:#1c8622;}
        .kq-vang {background:#fff7d1;color:#d97706;}
        .kq-table-block {margin-top:10px;margin-bottom:18px;}
        .kq-title-block {margin-top:18px;margin-bottom:2px;font-size:18px;font-weight:700;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="kq-header">📊 Thống kê kết quả dự báo</div>', unsafe_allow_html=True)
    st.markdown('<div class="kq-title-block" style="color:#e11d48;">📌 STICK ĐỎ (Δ Dự báo sát nhất có 📌)</div>', unsafe_allow_html=True)
    if not tk_xanh.empty:
        st.markdown(df_to_html(tk_xanh, header_color="#e8faef"), unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:gray;">Không có thống kê.</span>', unsafe_allow_html=True)

    st.markdown('<div class="kq-title-block" style="color:#ff9c00;">✅ STICK XANH (BestModel có ✅)</div>', unsafe_allow_html=True)
    if not tk_vang.empty:
        st.markdown(df_to_html(tk_vang, header_color="#fff7d1"), unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:gray;">Không có thống kê.</span>', unsafe_allow_html=True)

    st.markdown('<div class="kq-title-block" style="color:#e11d48;">📌 Các dòng có STICK ĐỎ (Δ Dự báo sát nhất có 📌) <b>(14 ngày gần nhất)</b></div>', unsafe_allow_html=True)
    if not stick_xanh.empty:
        st.markdown(df_to_html(stick_xanh, header_color="#f5fefd"), unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:gray;">Không có dòng nào.</span>', unsafe_allow_html=True)

    st.markdown('<div class="kq-title-block" style="color:#22c55e;">✅ Các dòng có STICK XANH (Sát nhất (BestModel) có ✅) <b>(14 ngày gần nhất)</b></div>', unsafe_allow_html=True)
    if not stick_vang.empty:
        st.markdown(df_to_html(stick_vang, header_color="#fff7e3"), unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:gray;">Không có dòng nào.</span>', unsafe_allow_html=True)

    st.caption("Thống kê toàn bộ dữ liệu, bảng chi tiết chỉ hiển thị 14 ngày gần nhất. Ký hiệu stick 📌, ✅ luôn được giữ lại.")

# --- Dùng: render_tab_log_kq(df_kq)

# --- Dùng: render_tab_log_kq(df_kq)

# --- Dùng: render_tab_log_kq(df_kq)


################################################################
import streamlit as st
import pandas as pd
import numpy as np

def tab_bang_tong_hop(df_mops, df_enrich, df_gcs, df_brent_wti=None):
    import pandas as pd
    import numpy as np
    import streamlit as st

    # CSS đẹp dùng chung
    st.markdown("""
    <style>
    .styled-table {
        width:100%;
        border-collapse:separate;
        border-spacing:0;
        font-size:1.10em;
        background: #fcfcfe;
        border-radius: 13px;
        overflow: hidden;
        box-shadow: 0 2px 12px 0 rgba(36,37,38,0.07);
        margin-bottom:18px;
    }
    .styled-table th {
        background: #ecf4fd;
        color: #225597;
        font-weight:700;
        padding:8px 7px;
        text-align:center;
        font-size:1.07em;
        border-bottom:1px solid #e2e5e7;
    }
    .styled-table td {
        padding:7px 7px;
        border-bottom:1px solid #eef2f6;
        text-align:right;
        font-size:1.09em;
    }
    .styled-table tr:last-child td { border-bottom:none;}
    .styled-table td.label {text-align:left;color:#264674;font-weight:600;background:#f8fbff;}
    .styled-table td.val {font-weight:700;}
    .styled-table td.pos {color:#1a8905;font-weight:700;}
    .styled-table td.neg {color:#d6223c;font-weight:700;}
    .styled-table td.neu {color:#444;font-weight:700;}
    .styled-table tr:hover td {background: #f6fafd;}
    @media (max-width:700px) {
      .styled-table {font-size:1em;}
      .styled-table th,.styled-table td{padding:5px 2px;}
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='font-size:2.1em'><span style='vertical-align:-0.13em;'>📋</span> BẢNG TỔNG HỢP DỮ LIỆU XĂNG DẦU</h1>", unsafe_allow_html=True)

    # ==== Chọn ngày ====
    df = df_mops.copy()
    df["RunTime"] = pd.to_datetime(df["RunTime"], dayfirst=True, errors='coerce')
    df["date"] = df["RunTime"].dt.date
    all_dates = df["date"].dropna().unique()
    date_selected = st.date_input(
        "Chọn ngày xem bảng tổng hợp",
        value=all_dates[-1],
        min_value=min(all_dates),
        max_value=max(all_dates)
    )
    if isinstance(date_selected, list):
        date_selected = date_selected[0]
    date_selected = pd.to_datetime(date_selected).date()

    # ==== Format số ====
    def format_number_no_sign(val):
        try:
            v = float(str(val).replace(",", "."))
            if np.isnan(v): return ""
            return f"{v:,.2f}".replace(",", ".")
        except:
            return ""

    def format_number(val, show_plus=True):
        try:
            v = float(str(val).replace(",", "."))
            if np.isnan(v): return ""
            color = "#1a8905" if v > 0 else "#d6223c" if v < 0 else "#444"
            sign = "+" if show_plus and v > 0 else ""
            return f"<span style='color:{color};font-weight:600'>{sign}{v:,.2f}".replace(",", ".") + "</span>"
        except:
            return ""

    def format_date(d):
        try:
            if pd.isnull(d): return ""
            return pd.to_datetime(d).strftime("%d/%m/%Y")
        except:
            return ""

    # ===== Giá MOPS95, Brent, WTI hôm nay/hôm qua =====
    df_enrich_ = df_enrich.copy()
    for col in df_enrich_.columns:
        if "date" in col.lower():
            df_enrich_["Date"] = pd.to_datetime(df_enrich_[col], dayfirst=True, errors='coerce')
    mops_today = df_enrich_[df_enrich_["Date"].dt.date == date_selected]
    mops_yest = df_enrich_[df_enrich_["Date"].dt.date == (date_selected - pd.Timedelta(days=1))]
    def to_float(x):
        try:
            v = float(str(x).replace(",", "."))
            return v if not np.isnan(v) else None
        except:
            return None
    mops95_today = to_float(mops_today["MOP95"].iloc[-1]) if not mops_today.empty else None
    mops95_yest = to_float(mops_yest["MOP95"].iloc[-1]) if not mops_yest.empty else None
    delta_mops = mops95_today - mops95_yest if (mops95_today is not None and mops95_yest is not None) else None

    # Brent/WTI phiên close 15h30
    def close_brent_wti(df, date):
        df_ = df[df['date'] == date]
        if df_.empty: return None, None
        return (to_float(df_["BrentClose"].iloc[-1]), to_float(df_["WTIClose"].iloc[-1]))
    if df_brent_wti is not None:
        brent_today, wti_today = close_brent_wti(df_brent_wti, date_selected)
        brent_yest, wti_yest = close_brent_wti(df_brent_wti, date_selected - pd.Timedelta(days=1))
    else:
        brent_today = to_float(mops_today["BrentClose"].iloc[-1]) if not mops_today.empty else None
        wti_today = to_float(mops_today["WTIClose"].iloc[-1]) if not mops_today.empty else None
        brent_yest = to_float(mops_yest["BrentClose"].iloc[-1]) if not mops_yest.empty else None
        wti_yest = to_float(mops_yest["WTIClose"].iloc[-1]) if not mops_yest.empty else None
    delta_brent = brent_today - brent_yest if (brent_today is not None and brent_yest is not None) else None
    delta_wti = wti_today - wti_yest if (wti_today is not None and wti_yest is not None) else None

    # ==== 1. BẢNG GIÁ HIỆN TẠI ====
    st.markdown(
        f"<h4 style='margin-bottom:7px'><span style='color:#1669be;font-size:1.15em'>1</span> Giá MOPS95 và dầu Brent/WTI phiên close 15h30 <span style='color:#1468be;'>({date_selected.strftime('%d/%m/%Y')})</span></h4>",
        unsafe_allow_html=True)
    table1_html = f"""
    <table class="styled-table">
        <tr>
            <th> </th>
            <th>Hôm qua</th>
            <th>Hôm nay</th>
            <th>Chênh lệch</th>
        </tr>
        <tr>
            <td class="label">MOPS95</td>
            <td class="val">{format_number_no_sign(mops95_yest)}</td>
            <td class="val">{format_number_no_sign(mops95_today)}</td>
            <td class="val">{format_number(delta_mops, True)}</td>
        </tr>
        <tr>
            <td class="label">Brent</td>
            <td class="val">{format_number_no_sign(brent_yest)}</td>
            <td class="val">{format_number_no_sign(brent_today)}</td>
            <td class="val">{format_number(delta_brent, True)}</td>
        </tr>
        <tr>
            <td class="label">WTI</td>
            <td class="val">{format_number_no_sign(wti_yest)}</td>
            <td class="val">{format_number_no_sign(wti_today)}</td>
            <td class="val">{format_number(delta_wti, True)}</td>
        </tr>
    </table>
    """
    st.markdown(table1_html, unsafe_allow_html=True)

    # ==== 2. KẾT QUẢ DỰ BÁO (so với giá MOPS thật) ====
    st.markdown(
        f"<h4 style='margin-top:18px;margin-bottom:7px'><span style='color:#1669be;font-size:1.15em'>2</span> Kết quả dự báo các mô hình <span style='color:#1468be;'>({date_selected.strftime('%d/%m/%Y')})</span></h4>",
        unsafe_allow_html=True)
    df_ngay = df[df["date"] == date_selected]
    if df_ngay.empty or mops95_today is None:
        st.info("Chưa có giá MOPS 95 chính thức ngày này hoặc không có dữ liệu dự báo.")
    else:
        row = df_ngay.iloc[-1]
        models = ["LinearRegression", "XGBoost", "RandomForest", "Polynomial"]
        model_html = """
        <table class="styled-table">
        <tr>
            <th>Mô hình</th>
            <th>Dự báo</th>
            <th>MOPS95 thực tế</th>
            <th>Chênh lệch</th>
            <th>MAE</th>
        </tr>
        """
        for m in models:
            pred = row.get(f"{m}_pred", None)
            mae = row.get(f"{m}_mae", None)
            if pred is None or mae is None: continue
            pred = to_float(pred)
            mae = to_float(mae)
            chenh = pred - mops95_today if (pred is not None and mops95_today is not None) else None
            chenh_cls = "pos" if (chenh is not None and chenh > 0) else "neg" if (chenh is not None and chenh < 0) else "neu"
            model_html += (
                f"<tr>"
                f"<td class='label' style='text-align:center'>{m}</td>"
                f"<td class='val'>{format_number_no_sign(pred)}</td>"
                f"<td class='val'>{format_number_no_sign(mops95_today)}</td>"
                f"<td class='{chenh_cls}'>{format_number(chenh, True)}</td>"
                f"<td class='val'>{format_number_no_sign(mae)}</td>"
                f"</tr>"
            )
        model_html += "</table>"
        st.markdown(model_html, unsafe_allow_html=True)

    # ==== 3. DỮ LIỆU GIÁ CƠ SỞ (GCS) XĂNG 95 - TOÀN BỘ BẢNG ====
    st.markdown(
        f"<h4 style='margin-top:18px;margin-bottom:7px'><span style='color:#1669be;font-size:1.15em'>3</span> Dữ liệu giá cơ sở (GCS) xăng 95 - Toàn bộ bảng</h4>",
        unsafe_allow_html=True)

    # Chuẩn hóa dataframe và lấy các cột mong muốn
    df_gcs_ = df_gcs.copy()
    def gen_nhandinh(chenh):
        try:
            v = float(str(chenh).replace(",", "."))
            if np.isnan(v): return ""
            if v > 0: return "📉 DỰ BÁO GIẢM GIÁ"
            if v < 0: return "📈 DỰ BÁO TĂNG GIÁ"
            return ""
        except: return ""
    df_gcs_["Date"] = pd.to_datetime(df_gcs_["Date"], errors="coerce")
    gcs_view = df_gcs_[["Date", "GCS", "GIALE", "Chênh lệch"]].copy()
    gcs_view = gcs_view[gcs_view["Date"].notna()]
    gcs_view = gcs_view.sort_values("Date", ascending=True).reset_index(drop=True)
    gcs_view["Nhận định"] = gcs_view["Chênh lệch"].map(gen_nhandinh)

    table_gcs_html = """
    <table class="styled-table">
        <tr>
            <th>Ngày</th>
            <th>GCS</th>
            <th>GIALE</th>
            <th>Chênh lệch</th>
            <th>Nhận định</th>
        </tr>
    """
    for _, row in gcs_view.iterrows():
        d = format_date(row["Date"])
        gcs = format_number_no_sign(row["GCS"])
        giale = format_number_no_sign(row["GIALE"])
        chenh = to_float(row["Chênh lệch"])
        chenh_cls = "pos" if (chenh is not None and chenh > 0) else "neg" if (chenh is not None and chenh < 0) else "neu"
        chenh_txt = format_number(row["Chênh lệch"], show_plus=True)
        nhan = row["Nhận định"]
        table_gcs_html += (
            f"<tr>"
            f"<td class='label' style='text-align:center'>{d}</td>"
            f"<td class='val'>{gcs}</td>"
            f"<td class='val'>{giale}</td>"
            f"<td class='{chenh_cls}'>{chenh_txt}</td>"
            f"<td class='val' style='text-align:center'>{nhan}</td>"
            f"</tr>"
        )
    table_gcs_html += "</table>"
    st.markdown(table_gcs_html, unsafe_allow_html=True)


# --- Ví dụ gọi thử ---
# tab_bang_tong_hop(df_mops, df_enrich, df_gcs)
