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
    st.header("🛢️ Biểu đồ giá dầu Brent & WTI")
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Đảm bảo 'thoi_gian' là datetime
    if not pd.api.types.is_datetime64_any_dtype(df['thoi_gian']):
        df['thoi_gian'] = pd.to_datetime(df['thoi_gian'], errors='coerce')
    df = df[df['thoi_gian'].notna()]
    df['brent'] = pd.to_numeric(df['brent'], errors='coerce')
    df['wti'] = pd.to_numeric(df['wti'], errors='coerce')

    # =================== LỌC NGÀY HÔM NAY HOẶC NGÀY MỚI NHẤT ===================
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

    # ================= SO SÁNH GIÁ ĐÓNG CỬA HÔM NAY VÀ HÔM QUA =================
    # Lấy giá close cuối cùng hôm nay từ data (trong khoảng ngày đang xem)
    row_today_data = df_view.sort_values("thoi_gian", ascending=False).head(1)
    if not row_today_data.empty:
        brent_today = row_today_data["brent"].values[0]
        wti_today = row_today_data["wti"].values[0]
        date_today_compare = row_today_data["thoi_gian"].dt.date.values[0]
    else:
        brent_today = wti_today = date_today_compare = None

    # Lấy giá đóng cửa hôm qua từ enrich (chỉ lấy dòng cuối cùng)
    df_enrich = df_enrich.copy()
    df_enrich["Date"] = pd.to_datetime(df_enrich["Date"], errors="coerce")
    # Lấy ngày hôm qua so với ngày hiện tại trong bảng df
    yesterday = pd.to_datetime(date_today_compare) - pd.Timedelta(days=1) if date_today_compare is not None else today - pd.Timedelta(days=1)
    row_yt = df_enrich[df_enrich["Date"].dt.date == yesterday]
    if not row_yt.empty:
        brent_yt = row_yt["BrentClose"].values[-1]
        wti_yt = row_yt["WTIClose"].values[-1]
        date_yt_compare = yesterday
    else:
        # Nếu không có hôm qua, lấy ngày gần nhất trước đó
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

    # Block so sánh - luôn hiện phía trên
    if brent_today is not None and brent_yt is not None:
        st.markdown(
            f"""<div style="background:#3057a4;padding:12px 18px 10px 18px;border-radius:8px;color:white;font-size:1.09em;font-weight:600;margin-bottom:16px">
            📊 <span style="font-size:1.07em">So sánh giá đóng cửa hôm nay và hôm qua</span><br>
            Ngày so sánh: <b>{date_today_compare.strftime('%d/%m/%Y')}</b> & <b>{pd.to_datetime(date_yt_compare).strftime('%d/%m/%Y')}</b>
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
    else:
        st.markdown(
            "<span style='color:#e87020;font-weight:600'>⚠️ Không đủ dữ liệu ngày hệ thống, hiển thị 2 ngày gần nhất có dữ liệu</span>",
            unsafe_allow_html=True
        )

    # ================= BIỂU ĐỒ =================
    st.markdown("##### Số dòng dữ liệu: " + str(len(df_view)))
    if not df_view.empty:
        fig = plot_oil_price(df_view)
        st.plotly_chart(fig, use_container_width=True)

    # ================= DỮ LIỆU MỚI NHẤT =================
    st.write("**Dữ liệu mới nhất:**")
    st.dataframe(df_view.sort_values("thoi_gian", ascending=False).head(10), use_container_width=True)
    if df_view.empty:
        st.warning("Không có dữ liệu trong khoảng thời gian đã chọn!")

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

# ========== HÀM CHÍNH ==========
def render_tab_mops(df_mops, df_enrich):
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
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

    # **Chọn dòng dự báo theo thời gian**
    runtime_options = df_ngay["RunTime"].dt.strftime("%H:%M:%S").tolist()
    runtime_map = dict(zip(runtime_options, df_ngay.index))
    runtime_selected = st.selectbox("Chọn thời điểm (RunTime)", runtime_options, index=len(runtime_options)-1)
    row = df_ngay.loc[runtime_map[runtime_selected]]

    # ==== LẤY GIÁ HÔM QUA TỪ ENRICH (tối ưu tốc độ/lọc) ====
    def get_enrich_cols(df_enrich):
        df_enrich_ = df_enrich.copy()
        for col in df_enrich_.columns:
            if "date" in col.lower():
                df_enrich_["Date"] = pd.to_datetime(df_enrich_[col], dayfirst=True, errors='coerce')
        return df_enrich_

    df_enrich_ = get_enrich_cols(df_enrich)
    yesterday = date_selected - pd.Timedelta(days=1)
    row_enrich = df_enrich_[df_enrich_["Date"].dt.date == yesterday]
    row_enrich = row_enrich.iloc[-1] if not row_enrich.empty else None

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

    # --- Tính chênh lệch Brent, WTI hôm nay vs hôm qua ---
    try:
        brent_today_f = to_float(row["BrentClose"])
        brent_yt_f = to_float(row_enrich["BrentClose"]) if row_enrich is not None else None
        wti_today_f = to_float(row["WTIClose"])
        wti_yt_f = to_float(row_enrich["WTIClose"]) if row_enrich is not None else None
    except Exception:
        brent_today_f = brent_yt_f = wti_today_f = wti_yt_f = None

    delta_brent_val = brent_today_f - brent_yt_f if brent_today_f is not None and brent_yt_f is not None else None
    delta_wti_val = wti_today_f - wti_yt_f if wti_today_f is not None and wti_yt_f is not None else None

    # --- Giá MOPS thực tế hôm qua ---
    try:
        mops_yt_f = to_float(row_enrich["MOP95"]) if row_enrich is not None else None
    except Exception:
        mops_yt_f = None

    # --- Top 3 model sát nhất với Brent, và tính khoảng dự báo ---
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

        # Ghi rõ chênh lệch Brent (nằm trong UI block đẹp)
        st.markdown(
            f"<div style='margin-bottom:4px;font-size:1.05em'><b>🔎 Chênh lệch Brent (Δ={delta_brent:.2f})</b></div>"
            if delta_brent is not None else "<b>🔎 Chênh lệch Brent: ...</b>", unsafe_allow_html=True
        )

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

    # --- Tính chênh lệch khoảng dự báo và giá thực tế hôm qua ---
    delta_pred_min = delta_pred_max = None
    if mops_yt_f is not None and pred_range_min is not None and pred_range_max is not None:
        delta_pred_min = pred_range_min - mops_yt_f
        delta_pred_max = pred_range_max - mops_yt_f

    # === UI block đẹp, màu nền, format số Việt Nam ===
    st.markdown(
        f"""
        <div style='background: #EFF4FA; padding: 12px 16px; border-radius:14px; margin-bottom:10px;'>
            <b>⏰ Thời điểm dự báo:</b> <span style='color:#1558a7;font-size:1.1em'>{run_time_str}</span><br>
            <b>Brent hôm nay:</b> <span style='color:#347d39;font-size:1.2em'>{brent_today}</span> |
            <b>Hôm qua:</b> <span style='color:#7c7c7c'>{brent_yt}</span> {sign_arrow(delta_brent_val)}<br>
            <b>WTI hôm nay:</b> <span style='color:#a74814;font-size:1.2em'>{wti_today}</span> |
            <b>Hôm qua:</b> <span style='color:#7c7c7c'>{wti_yt}</span> {sign_arrow(delta_wti_val)}<br>
            <b>PLATTS XĂNG 95 hôm qua:</b> <span style='color:#336699'>{mops_yt}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- HIỂN THỊ KHOẢNG DỰ BÁO + CHÊNH LỆCH ---
    if pred_range_min is not None and pred_range_max is not None and mops_yt_f is not None:
        st.markdown(
            f"""<div style='background:#f7f0e3;padding:14px 16px;border-radius:14px;margin-bottom:10px;'>
            <b>🚦 Khoảng dự báo giá Platts xăng 95 (MOPS95):</b>
            <span style='color:#2A62B8;font-weight:bold;font-size:1.6rem'>
            {format_vn(pred_range_min)} – {format_vn(pred_range_max)}
            </span> đ/lít<br>
            <b>Chênh lệch với giá MOPS hôm qua:</b> 
            {sign_arrow(delta_pred_min)} đến {sign_arrow(delta_pred_max)}
            </div>""",
            unsafe_allow_html=True
        )
    elif pred_range_min is not None and pred_range_max is not None:
        st.markdown(
            f"""<div style='background:#f7f0e3;padding:14px 16px;border-radius:14px;margin-bottom:10px;'>
            <b>🚦 Khoảng dự báo giá Platts xăng 95 (MOPS95):</b>
            <span style='color:#2A62B8;font-weight:bold;font-size:1.6rem'>{format_vn(pred_range_min)} – {format_vn(pred_range_max)}</span> đ/lít
            </div>""",
            unsafe_allow_html=True
        )

    # --- Block 3 mô hình sát nhất với Brent ---
    if len(delta_models) >= 3:
        st.markdown("""
        <style>
            .model-brent-box {
                background: #f2f7ff;
                border-radius: 10px;
                padding: 14px 18px 12px 18px;
                margin-bottom: 0px;
            }
            .model-brent-title {
                color: #15477a;
                font-size: 1.1em;
                font-weight: bold;
                margin-bottom: 7px;
            }
            .model-brent-row {
                font-size: 15px;
                margin-bottom: 3px;
            }
            .model-name {
                font-weight: 700;
                min-width: 100px;
                display: inline-block;
            }
            .model-label {
                min-width: 48px;
                display: inline-block;
                color: #666;
            }
            .model-value {
                font-weight: 600;
                color: #23356c;
            }
            .model-delta-up {color: #198544;}
            .model-delta-down {color: #d6223c;}
        </style>
        """, unsafe_allow_html=True)
        st.markdown(
            """<div class="model-brent-box">
                <div class="model-brent-title">📈 3 mô hình sát nhất với biến động Brent:</div>
            """, unsafe_allow_html=True)
        icons = {
            "LinearRegression": "🟦", "XGBoost": "🟧", "RandomForest": "🟩", "Polynomial": "🟪"
        }
        colors = {
            "RandomForest": "#2270d6", "Polynomial": "#6b34ad", "XGBoost": "#b86818", "LinearRegression": "#357824"
        }
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
                    <span class="model-label">sát ΔBrent |Δ|=</span>
                    <span class="model-value">{format_vn(dist)}</span> |
                    <span class="model-label">MAE=</span>
                    <span class="model-value">{format_vn(mae)}</span>
                    <br>
                    <span class="model-label">→ Dự báo hiệu chỉnh:</span>
                    <span class="model-value" style="color:{color};font-size:1.1em;">{format_vn(pred_adj)}</span>
                    <span class="{delta_class}" style="font-size:1em;margin-left:5px">{arrow} {sign}{format_vn(delta)}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Bảng tóm tắt tất cả mô hình ---
    st.markdown("""
    <pre style="background:#ecf2fa;padding:12px 14px 7px 14px;border-radius:11px;color:#243868;font-size:1.05em">
┏━━━━━━━━━━━━━━━━━━━
""", unsafe_allow_html=True)
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
        arrow = "🔺 Tăng" if delta > 0 else "🔻 Giảm"
        color = "#1c8622" if delta > 0 else "#d6223c"
        st.markdown(
            f"""┃ <b>{m}</b>: <b>{format_vn(pred)}</b> (MAE=<b>{format_vn(mae)}</b>) (<span style="color:{color}">{arrow} {sign}{format_vn(delta)}</span>)<br>""",
            unsafe_allow_html=True
        )
    st.markdown("""┗━━━━━━━━━━━━━━━━━━━
    </pre>
    """, unsafe_allow_html=True)

    # --- (Tuỳ chọn) Hiện bảng chi tiết ---
    with st.expander("Bảng dữ liệu chi tiết ngày này"):
        st.dataframe(df_ngay, use_container_width=True)



def render_tab_mae(df_rolling):
    st.header("📉 Sai số dự báo (MAE/Rolling)")
    df_rolling = df_rolling.copy()
    df_rolling["RunTime"] = pd.to_datetime(df_rolling["RunTime"], dayfirst=True, errors='coerce')
    df_rolling = df_rolling[df_rolling["RunTime"].notna()]
    min_date = df_rolling["RunTime"].min().date()
    max_date = df_rolling["RunTime"].max().date()
    date_range = st.sidebar.date_input("Chọn khoảng ngày MAE", [min_date, max_date], key="mae")
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df_view = df_rolling[(df_rolling["RunTime"] >= start_date) & (df_rolling["RunTime"] < end_date)]
    if "Model" in df_rolling.columns:
        models = df_rolling["Model"].unique().tolist()
        selected_models = st.multiselect("Chọn mô hình (MAE)", models, default=models, key="model_mae")
        df_view = df_view[df_view["Model"].isin(selected_models)]
    fig = px.bar(
        df_view, x="RunTime", y="MAE_Rolling", color="Model", barmode="group",
        title="Rolling MAE theo từng mô hình"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_view.sort_values("RunTime", ascending=False).head(10))

#Tab gia co so
import streamlit as st
import pandas as pd
import datetime
import pytz

def vnd(val):
    """Định dạng số thành tiền Việt."""
    try:
        v = float(str(val).replace(",", ".").replace(" ", ""))
        return f"{v:,.0f}".replace(",", ".")
    except:
        return str(val) if val is not None else "..."

def chenh_lech_msg(delta):
    """Đánh giá tăng/giảm giá bán lẻ."""
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

def render_tab_gcs_chuky(df_gcs):
    st.markdown("""
    <style>
        .block-gcs {
            background: #f8fafd;
            border-radius: 13px;
            padding: 14px 18px 12px 18px;
            margin-bottom: 18px;
            box-shadow: 0 1.5px 9px 0 rgba(33,70,120,0.07);
        }
        .block-gcs hr {
            margin: 8px 0 14px 0;
            border-top: 1.5px solid #e4e7ed;
        }
        .gcs-title {
            font-size: 19px;
            font-weight: 700;
            color: #2a4877;
        }
        .gcs-section {
            font-size: 16px;
            font-weight: 700;
            color: #314686;
        }
        .gcs-row {
            font-size: 15px;
            line-height: 2;
        }
        .gcs-label {
            min-width: 135px;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)

    import datetime, pytz

    def vnd(val):
        try:
            v = float(val)
            return f"{v:,.0f}".replace(",", ".")
        except: return str(val) if val is not None else "..."

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

    df = df_gcs.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df[df["Date"].notna()]
    df = df.sort_values("Date").reset_index(drop=True)
    tz_vn = pytz.timezone("Asia/Ho_Chi_Minh")
    today_vn = datetime.datetime.now(tz_vn).date()
    df_today = df[df["Date"].dt.date == today_vn]
    if not df_today.empty:
        row_today = df_today.iloc[-1]
    else:
        row_today = df.iloc[-1]
        today_vn = row_today["Date"].date()
    today_str = row_today["Date"].strftime("%d/%m/%Y")
    # Dò đầu kỳ
    df['GIALE_shift'] = df['GIALE'].shift(1)
    idx_starts = df.index[df['GIALE'] != df['GIALE_shift']].tolist()
    idx_starts.append(len(df))
    # Xác định chu kỳ hiện tại
    curr_start, curr_end = None, None
    for i in range(len(idx_starts)-1):
        start, end = idx_starts[i], idx_starts[i+1]
        if any(df.loc[start:end-1, "Date"].dt.date == today_vn):
            curr_start, curr_end = start, end
            break
    cy_df = df.iloc[curr_start:curr_end]
    row_cy_start = cy_df.iloc[0]
    row_cy_end = cy_df.iloc[-1]
    # Chu kỳ tới
    next_start = curr_end if curr_end < len(df) else None
    next_end = idx_starts[idx_starts.index(next_start)+1] if next_start is not None and idx_starts.index(next_start)+1 < len(idx_starts) else len(df)
    next_cy_df = df.iloc[next_start:next_end] if next_start is not None else None
    row_next = next_cy_df.iloc[0] if next_cy_df is not None and len(next_cy_df) else None

    # ==== 1. HÔM NAY ====
    st.markdown(
        f"""
        <div class="block-gcs">
            <span class="gcs-title">📅 DỰ BÁO GIÁ CƠ SỞ NGÀY <b>{today_str}</b></span>
            <hr />
            <div class="gcs-row">
                <span class="gcs-label"><b>• GIÁ BÁN LẺ:</b></span>
                <span style="color:#14781c;font-weight:600">{vnd(row_today.get('GIALE'))} ₫</span><br>
                <span class="gcs-label"><b>• GCS hiện tại:</b></span>
                <span style="color:#2270d6;font-weight:600">{vnd(row_today.get('GCS'))} ₫</span><br>
                <span class="gcs-label"><b>• Chênh lệch:</b></span>
                <span style="color:#d6223c;font-weight:600">{vnd(row_today.get('Chênh lệch'))} ₫</span><br>
                <span style="font-size:14px">{chenh_lech_msg(row_today.get('Chênh lệch'))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # ==== 2. CHU KỲ HIỆN TẠI ====
    cy_str = f"{row_cy_start['Date'].strftime('%d/%m/%Y')} → {row_cy_end['Date'].strftime('%d/%m/%Y')}"
    st.markdown(
        f"""
        <div class="block-gcs">
            <span class="gcs-section">📊 GCS CHU KỲ HIỆN TẠI ({cy_str})</span>
            <div class="gcs-row">
                <span class="gcs-label"><b>• GIÁ BÁN LẺ:</b></span>
                <span style="color:#14781c;font-weight:600">{vnd(row_cy_end.get('GIALE'))} ₫</span><br>
                <span class="gcs-label"><b>• GCS (cuối kỳ):</b></span>
                <span style="color:#2270d6;font-weight:600">{vnd(row_cy_end.get('GCS'))} ₫</span><br>
                <span class="gcs-label"><b>• Chênh lệch:</b></span>
                <span style="color:#d6223c;font-weight:600">{vnd(row_cy_end.get('Chênh lệch'))} ₫</span><br>
                <span style="font-size:14px">{chenh_lech_msg(row_cy_end.get('Chênh lệch'))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # ==== 3. KỲ TỚI (nếu có) ====
    if row_next is not None:
        next_cy_str = f"{row_next['Date'].strftime('%d/%m/%Y')} → {next_cy_df.iloc[-1]['Date'].strftime('%d/%m/%Y')}" if len(next_cy_df) else ""
        st.markdown(
            f"""
            <div class="block-gcs">
                <span class="gcs-section">🧮 DỰ BÁO GCS KỲ TỚI ({next_cy_str})</span>
                <div class="gcs-row">
                    <span class="gcs-label"><b>• GIÁ BÁN LẺ (dự tính):</b></span>
                    <span style="color:#14781c;font-weight:600">{vnd(row_next.get('GIALE'))} ₫</span><br>
                    <span class="gcs-label"><b>• GCS đầu kỳ (dự tính):</b></span>
                    <span style="color:#2270d6;font-weight:600">{vnd(row_next.get('GCS'))} ₫</span><br>
                    <span class="gcs-label"><b>• Chênh lệch:</b></span>
                    <span style="color:#d6223c;font-weight:600">{vnd(row_next.get('Chênh lệch'))} ₫</span><br>
                    <span style="font-size:14px">{chenh_lech_msg(row_next.get('Chênh lệch'))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==== Bảng dữ liệu chi tiết ====
    with st.expander("Bảng dữ liệu chi tiết"):
        st.dataframe(df.drop(columns=["GIALE_shift"]))

def render_tab_14days(df):
    st.markdown("""
    <style>
    .stDataFrame {font-size:15px !important;}
    .css-1d391kg, .css-1i8e3wl {font-size:15px !important;}
    th, td {text-align: center !important;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("## 📋 Giá cơ sở 14 ngày gần nhất")

    # Đảm bảo tên cột và loại dữ liệu
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date", ascending=False)
    show_cols = [
        "Date", "MOP95", "RATEVCB", "GIALE", "GCS_CHENHLECH",
        "PRICEBCT", "xangdau", "GCS_chenhlech", "Chenhlech", "GCS_heso"
    ]
    df_view = df[show_cols].head(14).copy()

    # Định dạng ngày/tháng
    df_view["Date"] = df_view["Date"].dt.strftime("%d/%m/%Y")

    # Chuẩn hóa số liệu VND
    def vnd(x):
        try:
            if pd.isnull(x): return ""
            x = float(x)
            if abs(x) >= 1_000:
                return f"{x:,.0f}".replace(",", ".")
            else:
                return f"{x:,.0f}"
        except: return ""

    for col in ["MOP95", "RATEVCB", "GIALE", "GCS_CHENHLECH", "PRICEBCT", "GCS_chenhlech", "Chenhlech", "GCS_heso", "xangdau"]:
        if col in df_view.columns:
            df_view[col] = df_view[col].apply(vnd)

    # Xử lý nan/None cho đẹp (chuyển thành "")
    df_view = df_view.replace(["nan", "None", None, pd.NA, np.nan], "")

    # Màu cho cột chênh lệch
    def highlight_chenh(val):
        try:
            val_ = float(str(val).replace(".", "").replace(",", "."))
            if val_ < 0:  return "color:#1c8622; font-weight:600"
            if val_ > 0:  return "color:#d6223c; font-weight:600"
            return "color:#636363;"
        except:
            return ""

    # Áp dụng style
    styled = df_view.style \
        .set_properties(**{"text-align": "center"}, subset=df_view.columns) \
        .applymap(highlight_chenh, subset=["Chenhlech", "GCS_chenhlech"]) \
        .format(na_rep="")

    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_tab_log(df_log, df_mops):
    st.header("📝 Báo cáo log các lần chạy, cảnh báo & raw data")
    st.dataframe(df_log.sort_values("RunTime", ascending=False).head(20))
    if "Delta" in df_mops.columns:
        st.write("**Cảnh báo chênh lệch dự báo > 5 đồng:**")
        df_alert = df_mops[df_mops["Delta"].abs() > 5]
        st.dataframe(df_alert.sort_values("RunTime", ascending=False).head(10))



# code tab chinh

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gspread
from gspread_dataframe import get_as_dataframe
from oauth2client.service_account import ServiceAccountCredentials

# CSS responsive cho mobile
st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
    .stDataFrame {font-size: 15px !important;}
    .stPlotlyChart {margin-bottom: 0.7rem !important;}
    .block-container {padding-top: 1rem !important;}
    @media (max-width: 800px) {
        .element-container {padding: 0.6rem 0.2rem;}
    }
    </style>
""", unsafe_allow_html=True)
SERVICE_ACCOUNT_JSON = st.secrets["GOOGLE_SERVICE_ACCOUNT"]

# Nếu là dạng string thì parse ra dict:
if isinstance(SERVICE_ACCOUNT_JSON, str):
    SERVICE_ACCOUNT_JSON = json.loads(SERVICE_ACCOUNT_JSON)

# Google API dùng dict luôn
from oauth2client.service_account import ServiceAccountCredentials

@st.cache_data(ttl=300)
def load_sheet(sheet_name):
    import gspread
    from gspread_dataframe import get_as_dataframe
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(SERVICE_ACCOUNT_JSON, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).worksheet(sheet_name)
    df = get_as_dataframe(sheet).dropna(how="all")
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    return df

# ==== GOOGLE SHEET CONFIG ====
SHEET_ID = "1t9CSWV_NUxG-9WOWIz6vRITcBwVnBgqwbAdITAKfmW4"
CREDENTIALS_FILE = "credentials.json"
SHEET_OIL = "data"
SHEET_LOG = "log_run_history_2"
SHEET_ENRICH = "train_mop_enrich"
SHEET_ROLLING = "log_rolling_results"
SHEET_GCS = "data_giadinh"

@st.cache_data(ttl=300)
# ==== LOAD DATA ====
df_oil = load_sheet(SHEET_OIL)
df_mops = load_sheet(SHEET_LOG)
df_rolling = load_sheet(SHEET_ROLLING)
df_log = load_sheet(SHEET_LOG)      # hoặc đổi sang sheet log khác nếu muốn
df_enrich = load_sheet(SHEET_ENRICH)
df_gcs = load_sheet(SHEET_GCS)

# ==== SIDEBAR RADIO MENU ====
tab_options = [
    "🛢️ Giá Dầu",
    "⛽ Dự Báo MOPS",
    "📉 MAE Rolling",
    "📝 Log/Báo Cáo",
    "📅 Dự báo GCS chu kỳ",
    "📋 Giá cơ sở 14 ngày"
]
tab_selected = st.sidebar.radio("Chọn chuyên mục", tab_options, index=0)

if tab_selected == tab_options[0]:
    render_tab_oil(df_oil,df_enrich)
elif tab_selected == tab_options[1]:
    render_tab_mops(df_mops, df_enrich)
elif tab_selected == tab_options[2]:
    render_tab_mae(df_rolling)
elif tab_selected == tab_options[3]:
    render_tab_log(df_log, df_mops)
elif tab_selected == tab_options[4]:
    df_gcs = load_sheet("data_giadinh")
    render_tab_gcs_chuky(df_gcs)
elif tab_selected == tab_options[5]:
    df_main = load_sheet("data_main")
    render_tab_14days(df_main)
