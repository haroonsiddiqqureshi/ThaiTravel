import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ตั้งค่าหน้า Dashboard
st.set_page_config(page_title="Thai Travel Dashboard", layout="wide", page_icon="🇹🇭")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_github_icon = """
<style>
/* ซ่อนปุ่ม Deploy และไอคอน GitHub มุมขวาบน */
.stAppDeployButton {display:none;}
[data-testid="stToolbar"] {visibility: hidden;}
</style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

st.title("🇹🇭 Thai Travel Statistics & Data Analysis")

# --- Helper Functions ---
thai_months_abbr = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
thai_months_full = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]

def format_thai_date(date_obj, full_month=False):
    if pd.isnull(date_obj): return ""
    month_idx = date_obj.month - 1
    year_thai = date_obj.year + 543
    month_name = thai_months_full[month_idx] if full_month else thai_months_abbr[month_idx]
    return f"{month_name} {year_thai}"

def format_number_with_unit(val):
    return f"{int(val):,} คน"

def clean_complex_string(val):
    if pd.isna(val): return 0
    val = str(val).replace(',', '').strip()
    if not val or val.lower() == 'nan': return 0
    
    if '-' in val:
        try:
            parts = val.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            pass
            
    if 'ชม.' in val or 'นาที' in val:
        h, m = 0, 0
        try:
            if 'ชม.' in val:
                parts = val.split('ชม.')
                h = float(parts[0])
                val = parts[1]
            if 'นาที' in val:
                m = float(val.replace('นาที', ''))
            return h * 60 + m
        except:
            pass
            
    try:
        return float(val)
    except:
        return 0

# --- 1. โหลดข้อมูลนักท่องเที่ยว (Time Series) ---
@st.cache_data 
def load_tourist_data():
    file_id = '1nm8yyywGVr7-q8BsdeitUGWZgOh9kBRt'
    csv_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    df = pd.read_csv(csv_url, header=[1])
    
    cols_to_drop = [col for col in df.columns if 'ม.ค.-ธ.ค.' in col or 'ม.ค.-ก.ค.' in col]
    df = df.drop(columns=cols_to_drop)
    
    regions_to_remove = ['ภาคกลางไม่รวมกรุงเทพฯ', 'ภาคตะวันออก', 'ภาคใต้', 'ภาคเหนือ', 'ภาคตะวันออกเฉียงเหนือ', 'ทั่วประเทศไทย']
    df = df[~df['Unnamed: 0'].isin(regions_to_remove)]
    
    df = df.rename(columns={'Unnamed: 0': 'Province'})
    df = df.dropna(subset=['Province'])
    df = df[df['Province'].astype(str).str.lower() != 'nan']
    df = df[~df['Province'].astype(str).str.contains('อ้างอิง', na=False)]

    data_cols = df.columns[1:] 
    new_date_cols = pd.date_range(start='2015-01-01', periods=len(data_cols), freq='MS')
    df.columns = ['Province'] + list(new_date_cols)

    for col in new_date_cols:
        df[col] = df[col].astype(str).str.replace(',', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# --- 2. โหลดข้อมูลปัจจัย + Target ---
@st.cache_data
def load_factor_and_target_data():
    file_id = '1fL7HTdKV7bZLGmayFw2JOO2tNZHKz3TE'
    csv_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    df = pd.read_csv(csv_url)
    
    if 'จังหวัด' in df.columns:
        df = df.rename(columns={'จังหวัด': 'Province'})
        
    df = df.dropna(subset=['Province'])
    df = df[df['Province'].astype(str).str.lower() != 'nan']
    df = df[~df['Province'].astype(str).str.contains('อ้างอิง', na=False)]

    column_mapping = {
        'สนามบิน\n (มี=1, ไม่มี=0)': 'สนามบิน',
        'รถไฟ\n (มี=1,ไม่มี=0)': 'รถไฟ',
        'ระยะห่างจากกทม.\n(กิโลเมตร) โดยประมาณ': 'ระยะห่างจากกรุงเทพ',
        'ระยะเวลาจากกทม. ไปยังจังหวัดต่างๆ\n (เดินทางโดยรถยนต์)': 'เดินทางโดยรถยนต์',
        'จำนวนการค้นหาบน\nFacebook ': 'จำนวนการค้นหาบน Facebook',
        'จำนวนการค้นหาบน\nTiktok': 'จำนวนการค้นหาบน Tiktok',
        'จำนวนการค้นหาบน\nInstagram ': 'จำนวนการค้นหาบน Instagram',
        'ช้อปปิ้ง': 'ช้อปปิ้ง',
        'กิจกรรมและความบันเทิง': 'กิจกรรมและความบันเทิง',
        'ประวัติศาสตร์ วัฒนธรรมและศาสนา': 'ประวัติศาสตร์ วัฒนธรรมและศาสนา',
        'พิพิธภัณฑ์': 'พิพิธภัณฑ์',
        'สวนสาธารณะ': 'สวนสาธารณะ',
        'วิถีชีวิตและชุมชน': 'วิถีชีวิตและชุมชน',
        'สวนสัตว์ สวนน้ำ และสวนสนุก': 'สวนสัตว์ สวนน้ำ และสวนสนุก',
        'ธรรมชาติ': 'ธรรมชาติ',
        'คาเฟ่': 'ความหนาแน่นของคาเฟ่',
        'ปัญหาด้านเศรษฐกิจและรายได้ประชากร': 'ปัญหาด้านเศรษฐกิจ',
        'ปัญหาโครงสร้างพื้นฐานและระบบคมนาคม': 'ปัญหาระบบการคมนาคม',
        'ปัญหาสิ่งแวดล้อมและมลพิษ': 'ปัญหามลพิษ',
        'ปัญหาภัยพิบัติและความเสี่ยงด้านสภาพภูมิอากาศ': 'ปัญหาความเสี่ยงด้านภัยพิบัติ',
        'ปัญหาการขยายตัวของเมืองและคุณภาพชีวิต': 'ปัญหาคุณภาพชีวิต',
        'ปัญหาโครงสร้างประชากรและการย้ายถิ่น': 'ปัญหาโครงสร้างประชากร',
        'จำนวนนักท่องเที่ยว': 'Total_Tourists'
    }

    new_df = pd.DataFrame()
    new_df['Province'] = df['Province']
    
    for csv_col, target_col in column_mapping.items():
        if csv_col in df.columns:
            new_df[target_col] = df[csv_col].apply(clean_complex_string)
        else:
            new_df[target_col] = 0
            
    new_df['ความสนใจต่อที่เที่ยว'] = (new_df['จำนวนการค้นหาบน Facebook'] + 
                                      new_df['จำนวนการค้นหาบน Tiktok'] + 
                                      new_df['จำนวนการค้นหาบน Instagram'])

    required_features = [
        'สนามบิน', 'รถไฟ', 'ระยะห่างจากกรุงเทพ', 'เดินทางโดยรถยนต์',
        'จำนวนการค้นหาบน Facebook', 'จำนวนการค้นหาบน Tiktok', 'จำนวนการค้นหาบน Instagram',
        'ช้อปปิ้ง', 'กิจกรรมและความบันเทิง', 'ประวัติศาสตร์ วัฒนธรรมและศาสนา', 'พิพิธภัณฑ์', 'สวนสาธารณะ', 'วิถีชีวิตและชุมชน', 'สวนสัตว์ สวนน้ำ และสวนสนุก', 'ธรรมชาติ',
        'ปัญหาด้านเศรษฐกิจ', 'ปัญหาระบบการคมนาคม', 'ปัญหามลพิษ', 'ปัญหาความเสี่ยงด้านภัยพิบัติ', 'ปัญหาคุณภาพชีวิต', 'ปัญหาโครงสร้างประชากร',
        'ความหนาแน่นของคาเฟ่',
        'ความสนใจต่อที่เที่ยว'
    ]
    
    final_cols = ['Province', 'Total_Tourists'] + [c for c in required_features if c in new_df.columns]
    df_final = new_df[final_cols]
    df_final = df_final.groupby('Province').last().reset_index()
    
    return df_final, required_features

# --- Main App Logic ---

try:
    with st.spinner('กำลังโหลดข้อมูลและเตรียมระบบ...'):
        df_tourist = load_tourist_data()
        df_model_data, feature_names_list = load_factor_and_target_data()
        
    province_list = sorted(df_tourist['Province'].unique())

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# สร้าง Tabs
tab1, tab2 = st.tabs(["📊 พยากรณ์รายจังหวัด (Forecast)", "🧠 วิเคราะห์ปัจจัย & SWOT"])

# ==============================================================================
# TAB 1: Forecast
# ==============================================================================
with tab1:
    # --- Control Panel ---
    with st.container(border=False):
        st.write("###### แผงควบคุม (Control Panel)")
        
        # Layout: แบ่งเป็น 4 ส่วนหลัก (Scope | Province | Analysis Mode | Time Resolution)
        col_scope, col_prov, col_mode, col_time = st.columns([1.5, 3, 1.5, 1.5])
        
        with col_scope:
            view_mode = st.radio(
                "📍 ขอบเขตข้อมูล",
                ["ภาพรวมทั้งประเทศ", "เจาะจงรายจังหวัด"],
                horizontal=False
            )

        with col_prov:
            # ถ้าเลือกภาพรวมทั้งประเทศ ให้ปิดการเลือกจังหวัด (Disable) เพื่อไม่ให้สับสน
            is_disabled = (view_mode == "ภาพรวมทั้งประเทศ")
            
            selected_province_val = st.selectbox(
                "🔎 เลือกจังหวัด",
                province_list,
                index=province_list.index('เชียงใหม่') if 'เชียงใหม่' in province_list else 0,
                disabled=is_disabled
            )
            
            selected_province = 'ทั่วประเทศไทย' if is_disabled else selected_province_val

        with col_mode:
            analysis_mode = st.radio(
                "📊 รูปแบบการวิเคราะห์",
                ["ข้อมูลจริง (Raw Data)", "พยากรณ์ (Forecast)"],
                horizontal=False,
                index=0
            )
            
        with col_time:
            time_resolution = st.radio(
                "📅 รูปแบบการแสดงข้อมูล",
                ["รายปี (Yearly)", "รายเดือน (Monthly)"],
                horizontal=False,
                index=0
            )

    # --- Logic Process ---
    if selected_province == 'ทั่วประเทศไทย' and 'ทั่วประเทศไทย' not in df_tourist['Province'].values:
        province_data = pd.DataFrame(df_tourist.drop(columns=['Province']).sum()).T
        province_data['Province'] = 'ทั่วประเทศไทย'
    else:
        province_data = df_tourist[df_tourist['Province'] == selected_province]
    
    # 1. จัดเตรียมข้อมูลพื้นฐานรายเดือน (Raw Monthly Data)
    raw_melted_df = province_data.melt(id_vars=['Province'], var_name='Date', value_name='Tourists')
    raw_melted_df['Date'] = pd.to_datetime(raw_melted_df['Date'])
    raw_melted_df = raw_melted_df.dropna(subset=['Tourists'])
    raw_melted_df = raw_melted_df.sort_values('Date')

    # เก็บค่าข้อมูลจริงเดือนล่าสุดไว้ใช้เชื่อมกราฟ
    last_date_raw = raw_melted_df['Date'].iloc[-1]

    # 2. แปลงข้อมูลตามความละเอียดที่เลือก (Yearly vs Monthly)
    if time_resolution == "รายปี (Yearly)":
        melted_df = raw_melted_df.copy()
        melted_df['Year'] = melted_df['Date'].dt.year
        # รวมจำนวนนักท่องเที่ยวตามปี
        melted_df = melted_df.groupby('Year', as_index=False)['Tourists'].sum()
        # จำลองวันที่เป็น 1 มกราคมของปีนั้นๆ เพื่อให้สร้างกราฟได้
        melted_df['Date'] = pd.to_datetime(melted_df['Year'].astype(str) + '-01-01')
        melted_df['ThaiDate'] = 'ปี ' + (melted_df['Year'] + 543).astype(str)
    else:
        melted_df = raw_melted_df.copy()
        melted_df['ThaiDate'] = melted_df['Date'].apply(lambda x: format_thai_date(x, full_month=True))

    st.divider()

    if not melted_df.empty:
        last_date = melted_df['Date'].iloc[-1]
        last_value = melted_df['Tourists'].iloc[-1]
        last_date_display = melted_df['ThaiDate'].iloc[-1]
        
        # Header Area
        col_head, col_met = st.columns([3, 1])
        with col_head:
            if analysis_mode == "ข้อมูลจริง (Raw Data)":
                st.subheader(f"สถิตินักท่องเที่ยวจริง: {selected_province}")
            else:
                st.subheader(f"พยากรณ์แนวโน้มล่วงหน้า: {selected_province}")
        
        with col_met:
            st.metric(label=f"ข้อมูลล่าสุด ({last_date_display})", value=f"{last_value:,.0f} คน")

        # -----------------------------------------------------
        # MODE 1: Raw Data (Default)
        # -----------------------------------------------------
        if analysis_mode == "ข้อมูลจริง (Raw Data)":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=melted_df['Date'], 
                y=melted_df['Tourists'], 
                mode='lines+markers', 
                name='ข้อมูลจริง', 
                line=dict(color='#1f77b4', width=3), 
                customdata=melted_df['ThaiDate'], 
                hovertemplate="<b>%{customdata}</b><br>จำนวน: %{y:,.0f} คน<extra></extra>"
            ))
            
            # การตั้งค่าแกน X ตามความละเอียดข้อมูล
            if time_resolution == "รายปี (Yearly)":
                tick_vals = melted_df['Date']
                tick_text = melted_df['ThaiDate']
            else:
                tick_vals = pd.date_range(start=melted_df['Date'].min(), end=melted_df['Date'].max(), freq='6MS')
                tick_text = [format_thai_date(d) for d in tick_vals]
                
            fig.update_layout(
                xaxis=dict(tickvals=tick_vals, ticktext=tick_text, title="ระยะเวลา (พ.ศ.)"), 
                yaxis=dict(title="จำนวนนักท่องเที่ยว (คน)"),
                height=500,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("#### ตารางข้อมูลสถิติย้อนหลัง")
            display_raw = melted_df[['ThaiDate', 'Tourists']].sort_values('ThaiDate', ascending=False).copy()
            display_raw['Tourists'] = display_raw['Tourists'].apply(format_number_with_unit)
            display_raw.columns = ['ช่วงเวลา', 'จำนวนนักท่องเที่ยว']
            st.dataframe(display_raw, use_container_width=True, hide_index=True)

        # -----------------------------------------------------
        # MODE 2: Forecast
        # -----------------------------------------------------
        else:
            with st.spinner("กำลังคำนวณการพยากรณ์..."):
                # ให้โมเดลเรียนรู้ด้วยข้อมูล "รายเดือน" (raw_melted_df) เสมอ เพื่อจับแพทเทิร์นฤดูกาลให้แม่นยำ
                prophet_df = raw_melted_df[['Date', 'Tourists']].rename(columns={'Date': 'ds', 'Tourists': 'y'})
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=12, freq='MS')
                forecast = m.predict(future)
                
                # นำผลลัพธ์พยากรณ์มา GroupBy ตามตัวเลือก
                if time_resolution == "รายปี (Yearly)":
                    forecast['Year'] = forecast['ds'].dt.year
                    forecast_plot = forecast.groupby('Year', as_index=False)[['yhat', 'yhat_lower', 'yhat_upper', 'trend']].sum()
                    forecast_plot['ds'] = pd.to_datetime(forecast_plot['Year'].astype(str) + '-01-01')
                    forecast_plot['ThaiDate'] = 'ปี ' + (forecast_plot['Year'] + 543).astype(str)
                    
                    history = forecast_plot[forecast_plot['ds'] <= last_date]
                    future_only = forecast_plot[forecast_plot['ds'] > last_date]
                else:
                    forecast['ThaiDate'] = forecast['ds'].apply(lambda x: format_thai_date(x, full_month=True))
                    forecast_plot = forecast
                    history = forecast_plot[forecast_plot['ds'] <= last_date_raw]
                    future_only = forecast_plot[forecast_plot['ds'] > last_date_raw]

                # สร้างกราฟเชื่อมต่อระหว่างข้อมูลจริงกับพยากรณ์
                if not future_only.empty and not history.empty:
                    future_plot_line = pd.concat([history.iloc[[-1]], future_only])
                else:
                    future_plot_line = future_only

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history['ds'], y=history['yhat'], mode='lines+markers', name='ข้อมูลจริง/เทรนด์', line=dict(color='#1f77b4', width=3), customdata=history['ThaiDate'], hovertemplate="<b>%{customdata}</b><br>จำนวน: %{y:,.0f} คน<extra></extra>"))
                fig.add_trace(go.Scatter(x=future_plot_line['ds'], y=future_plot_line['yhat'], mode='lines+markers', name='พยากรณ์อนาคต', line=dict(color='#FF4B4B', width=3, dash='dot'), customdata=future_plot_line['ThaiDate'], hovertemplate="<b>%{customdata}</b><br>พยากรณ์: %{y:,.0f} คน<extra></extra>"))
                fig.add_trace(go.Scatter(x=future_plot_line['ds'], y=future_plot_line['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=future_plot_line['ds'], y=future_plot_line['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 75, 75, 0.2)', showlegend=False, hoverinfo='skip'))
                
                # การตั้งค่าแกน X ของกราฟพยากรณ์
                if time_resolution == "รายปี (Yearly)":
                    tick_vals = forecast_plot['ds']
                    tick_text = forecast_plot['ThaiDate']
                else:
                    tick_vals = pd.date_range(start=forecast['ds'].min(), end=forecast['ds'].max(), freq='6MS')
                    tick_text = [format_thai_date(d) for d in tick_vals]
                    
                fig.update_layout(xaxis=dict(tickvals=tick_vals, ticktext=tick_text, title="ระยะเวลา (พ.ศ.)"), yaxis=dict(title="จำนวนนักท่องเที่ยว (คน)"), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)

                row2_col1, row2_col2 = st.columns([1, 2])
                with row2_col1:
                    st.write("#### ข้อมูลประวัติย้อนหลัง")
                    display_raw = melted_df[['ThaiDate', 'Tourists']].sort_values('ThaiDate', ascending=False).copy()
                    display_raw['Tourists'] = display_raw['Tourists'].apply(format_number_with_unit)
                    display_raw.columns = ['ช่วงเวลา', 'จำนวนนักท่องเที่ยว']
                    st.dataframe(display_raw, height=400, use_container_width=True, hide_index=True)

                with row2_col2:
                    if time_resolution == "รายเดือน (Monthly)":
                        st.write("#### องค์ประกอบของข้อมูล (Decomposition)")
                        fig_trend = px.line(forecast, x='ds', y='trend', custom_data=['ThaiDate'])
                        fig_trend.update_traces(line_color='#2ca02c')
                        fig_trend.update_layout(title="1. แนวโน้มระยะยาว (Trend)", height=200, xaxis_title=None)
                        st.plotly_chart(fig_trend, use_container_width=True)

                        fig_season = px.line(future_only, x='ds', y='yearly', markers=True, custom_data=['ThaiDate'])
                        fig_season.update_traces(line_color='#ff7f0e')
                        fig_season.update_xaxes(tickvals=future_only['ds'], ticktext=[thai_months_abbr[d.month-1] for d in future_only['ds']])
                        fig_season.update_layout(title="2. รูปแบบตามฤดูกาล (Seasonality)", height=200, xaxis_title=None)
                        st.plotly_chart(fig_season, use_container_width=True)
                    else:
                         st.info("การวิเคราะห์องค์ประกอบข้อมูล (Trend & Seasonality) จะแสดงผลเมื่อเลือกความละเอียดแบบ 'รายเดือน' เท่านั้น")

                st.markdown("---")
                if time_resolution == "รายปี (Yearly)":
                    st.subheader("ตารางพยากรณ์แนวโน้มรายปีล่วงหน้า")
                else:
                    st.subheader("ตารางพยากรณ์ 12 เดือนข้างหน้า")
                    
                display_forecast = future_only[['ThaiDate', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                    display_forecast[col] = display_forecast[col].apply(format_number_with_unit)
                display_forecast.columns = ['ช่วงเวลา', 'ค่าพยากรณ์', 'ขอบเขตล่าง', 'ขอบเขตบน']
                st.dataframe(display_forecast, use_container_width=True, hide_index=True)
    else:
        st.warning("ไม่มีข้อมูลเพียงพอสำหรับการแสดงผล")

# ==============================================================================
# TAB 2: AI Model & SWOT Analysis
# ==============================================================================
with tab2:
    st.header("วิเคราะห์ปัจจัยและ SWOT Analysis")

    with st.spinner("กำลังเรียนรู้ข้อมูล (Random Forest) ..."):
        train_data = df_model_data.copy()
        train_data = train_data[train_data['Province'] != 'กรุงเทพมหานคร']
        
        X = train_data[feature_names_list]
        y = train_data['Total_Tourists']
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        y_pred = rf_model.predict(X)
        accuracy_score = r2_score(y, y_pred)
        
        importances = rf_model.feature_importances_
        feature_imp_df = pd.DataFrame({'Feature': feature_names_list, 'Importance': importances})
        feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)

    # Global Insights
    st.subheader("ประสิทธิภาพโมเดลและความสำคัญของปัจจัย")
    
    col_acc, col_chart = st.columns([1, 3])
    with col_acc:
        st.info(f"**ความแม่นยำของโมเดล (R² Score):**")
        st.metric(label="R-Squared", value=f"{accuracy_score:.2%}")
        st.caption("ค่า R² ยิ่งเข้าใกล้ 100% แสดงว่าปัจจัยที่เลือกมา สามารถอธิบายจำนวนนักท่องเที่ยวได้ดีมาก")
    
    with col_chart:
        uncontrollable_vars = ['ระยะห่างจากกรุงเทพ', 'เดินทางโดยรถยนต์', 'รถไฟ', 'สนามบิน']
        result_vars = ['จำนวนการค้นหาบน Facebook', 'จำนวนการค้นหาบน Tiktok', 'จำนวนการค้นหาบน Instagram', 'ความสนใจต่อที่เที่ยว']
        actionable_vars = [f for f in feature_names_list if f not in uncontrollable_vars and f not in result_vars]
        
        top_features_plot = feature_imp_df.head(15).sort_values('Importance', ascending=True)
        fig_imp = px.bar(top_features_plot, x='Importance', y='Feature', orientation='h', 
                         text_auto='.3f', color='Importance', color_continuous_scale='Blues')
        fig_imp.update_layout(xaxis_title="ระดับความสำคัญ", yaxis_title="ปัจจัย", showlegend=False, height=400)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # SWOT Analysis
    sorted_provinces_df = train_data[['Province', 'Total_Tourists']].sort_values('Total_Tourists')
    sorted_provinces_list = sorted_provinces_df['Province'].tolist()
    
    st.subheader("SWOT Analysis: รายจังหวัด")
    
    col_sel_prov, col_dummy = st.columns([1, 2])
    with col_sel_prov:
        selected_swot_prov = st.selectbox("🔎 เลือกจังหวัด (เรียงจากนักท่องเที่ยวน้อย -> มาก)", sorted_provinces_list)

    if selected_swot_prov:
        prov_row = train_data[train_data['Province'] == selected_swot_prov].iloc[0]
        prov_val = prov_row['Total_Tourists']
        top_10_provinces = train_data.nlargest(10, 'Total_Tourists')
        
        st.caption(f"จำนวนนักท่องเที่ยว (รายปี): {prov_val:,.0f} คน")
        
        comparison_data = []
        for feature in feature_names_list:
            prov_v = prov_row[feature]
            bench_v = top_10_provinces[feature].mean()
            comparison_data.append({
                'Feature': feature,
                'Value': prov_v,
                'Benchmark (Top10)': bench_v,
                'Gap': prov_v - bench_v
            })
        comp_df = pd.DataFrame(comparison_data).set_index('Feature')

        # Action Plan
        st.subheader("กลยุทธ์เพื่อยกระดับการท่องเที่ยว")
        st.caption("แนะนำเฉพาะปัจจัยที่สามารถควบคุมและปรับปรุงได้ (Actionable Factors) เรียงตามความสำคัญจากมากไปน้อย")
        
        sorted_actionable_imp = feature_imp_df[feature_imp_df['Feature'].isin(actionable_vars)].sort_values('Importance', ascending=False)
        
        weaknesses_list = []
        opportunities_list = []
        strengths_list = []

        for idx, row_imp in enumerate(sorted_actionable_imp.itertuples()):
            feature = row_imp.Feature
            current = comp_df.loc[feature, 'Value']
            target = comp_df.loc[feature, 'Benchmark (Top10)']
            
            negative_keywords = ['ปัญหา', 'มลพิษ', 'ความเสี่ยง']
            is_negative_factor = any(kw in feature for kw in negative_keywords)
            
            item = {
                'feature': feature,
                'current': current,
                'target': target,
                'is_neg': is_negative_factor
            }
            
            if is_negative_factor:
                if current > target:
                    weaknesses_list.append(item)
                else:
                    strengths_list.append(item)
            else:
                if current < target:
                    opportunities_list.append(item)
                else:
                    strengths_list.append(item)

        active_sections = []
        
        if weaknesses_list:
            active_sections.append({
                "title": "🚨 จุดอ่อน / สิ่งที่ต้องเร่งแก้ไข (Weaknesses)",
                "items": weaknesses_list,
                "color_code": "red",
                "bg_color": "rgba(255, 0, 0, 0.05)",
                "status_text": "ต้องลด/แก้ไข"
            })
            
        if opportunities_list:
            active_sections.append({
                "title": "⚠️ โอกาสพัฒนา / สิ่งที่ต้องเพิ่ม (Opportunities)",
                "items": opportunities_list,
                "color_code": "orange",
                "bg_color": "rgba(255, 165, 0, 0.05)",
                "status_text": "ควรสร้าง/ส่งเสริมเพิ่ม"
            })
            
        if strengths_list:
            active_sections.append({
                "title": "✅ จุดแข็ง / ศักยภาพสูง (Strengths)",
                "items": strengths_list,
                "color_code": "green",
                "bg_color": "rgba(0, 128, 0, 0.05)",
                "status_text": "ดีเยี่ยม/ควบคุมได้ดี"
            })
            
        if active_sections:
            cols = st.columns(len(active_sections))
            
            for col, section in zip(cols, active_sections):
                with col:
                    if section['color_code'] == 'red':
                        st.error(section['title'])
                    elif section['color_code'] == 'orange':
                        st.warning(section['title'])
                    else:
                        st.success(section['title'])
                    
                    for item in section['items']:
                        status_display = "ควบคุมได้ดี" if (item['is_neg'] and section['color_code']=='green') else section['status_text']
                        if section['color_code'] == 'green' and not item['is_neg']:
                            status_display = "ศักยภาพสูง"

                        diff_text = ""
                        if section['color_code'] == 'orange':
                             diff_text = f"(ขาดอีก {abs(item['target'] - item['current']):.2f})"
                        elif section['color_code'] == 'red':
                             diff_text = f"(สูงกว่าเกณฑ์ {item['target']:.2f})"
                        else:
                             diff_text = f"(Top 10: {item['target']:.2f})"

                        st.markdown(f"""
                        <div style="background-color:{section['bg_color']}; padding:10px; border-radius:5px; margin-bottom:10px; border-left: 4px solid {section['color_code']};">
                            <b>{item['feature']}</b><br>
                            <small>ค่าปัจจุบัน: {item['current']:.2f} {diff_text}</small><br>
                            <span style="color:{section['color_code']}; font-weight:bold;">{status_display}</span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("ไม่พบข้อมูลวิเคราะห์สำหรับจังหวัดนี้")

        st.markdown("---")

        with st.expander("ดูข้อมูลเพิ่มเติม: Social Media & ปัจจัยพื้นฐาน (คลิกเพื่อขยาย)"):
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write("**📱 Social Media Stats**")
                st.dataframe(comp_df.loc[result_vars].style.format("{:,.2f}"), use_container_width=True)
            with col_info2:
                st.write("**🛣️ ปัจจัยพื้นฐาน (Uncontrollable)**")
                st.dataframe(comp_df.loc[uncontrollable_vars].style.format("{:,.2f}"), use_container_width=True)