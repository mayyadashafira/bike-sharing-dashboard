import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Bike Sharing Analysis Dashboard",
    page_icon="🚲",
    layout="wide"
)
st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background-color: #f3e5f5;
    }
    
    /* Judul dan Header */
    h1, h2, h3 {
        color: #4a148c;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#4a148c, #7b1fa2);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Metrics Box */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        border-left: 5px solid #7b1fa2;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        color: #4a148c;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #7b1fa2;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #4a148c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# DATA LOADING & WRANGLING
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("day.csv")
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/danwild/bike-sharing-prediction/master/Bike-Sharing-Dataset/day.csv"
        df = pd.read_csv(url)
    
    # Cleaning & Renaming sesuai analisis
    df.rename(columns={
        'yr': 'year',
        'mnth': 'month',
        'hum': 'humidity',
        'cnt': 'count'
    }, inplace=True)

    df['dteday'] = pd.to_datetime(df['dteday'])

    # Mapping Labels
    df['season_label'] = df['season'].map({
        1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'
    })

    df['weather_label'] = df['weathersit'].map({
        1: 'Clear / Partly Cloudy',
        2: 'Misty / Cloudy',
        3: 'Light Snow / Rain',
        4: 'Heavy Rain'
    })

    df['workingday_label'] = df['workingday'].map({
        0: 'Holiday / Weekend',
        1: 'Working Day'
    })
    
    return df

df = load_data()

# SIDEBAR FILTERS
st.sidebar.title("🚲 Filter Data")
st.sidebar.write("Sesuaikan tampilan data di sini.")

# Filter Tahun
year_list = [0, 1]
year_labels = {0: '2011', 1: '2012'}
selected_year_val = st.sidebar.multiselect("Pilih Tahun", year_list, format_func=lambda x: year_labels[x], default=year_list)

# Filter Musim
season_options = df['season_label'].unique()
selected_seasons = st.sidebar.multiselect("Pilih Musim", season_options, default=season_options)

# Filter Dataframe
if selected_year_val and selected_seasons:
    main_df = df[
        (df['year'].isin(selected_year_val)) & 
        (df['season_label'].isin(selected_seasons))
    ]
else:
    main_df = df.copy()
    
# MAIN PAGE LAYOUT
st.title("🚲 Bike Sharing Analysis Dashboard")
st.markdown("### 💜 Analisis Tren Penyewaan Sepeda & Prediksi")
st.write("Dashboard ini menyajikan hasil analisis mendalam mengenai faktor cuaca, musim, dan hari kerja terhadap penyewaan sepeda.")

# KEY METRICS
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Penyewaan", value=f"{main_df['count'].sum():,}")
with col2:
    st.metric("Rata-rata Harian", value=f"{int(main_df['count'].mean()):,}")
with col3:
    st.metric("Hari Teramai", value=f"{main_df['count'].max():,}")
with col4:
    st.metric("Hari Tersepi", value=f"{main_df['count'].min():,}")
st.markdown("---")

# TABS NAVIGATION
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🤖 Prediksi Random Forest", "📝 Kesimpulan & Insight"])

# EDA
with tab1:
    st.header("🔍 Eksplorasi Data Visual")
    
    # --- CUACA ---
    st.subheader("Pengaruh Cuaca terhadap Penyewaan")
    col_eda1, col_eda2 = st.columns([2, 1])
    
    with col_eda1:
        # Barplot Cuaca
        fig_weather, ax_weather = plt.subplots(figsize=(10, 5))
        order_weather = main_df.groupby('weather_label')['count'].mean().sort_values(ascending=False).index
        sns.barplot(x='weather_label', y='count', data=main_df, order=order_weather, palette="Purples_r", ax=ax_weather)
        ax_weather.set_ylabel("Rata-rata Penyewaan")
        ax_weather.set_xlabel("")
        ax_weather.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig_weather)
    with col_eda2:
        st.info("**Insight Cuaca:**")
        st.markdown("""
        - **Cerah (Clear):** Penyewaan tertinggi. Sumber pendapatan utama.
        - **Hujan/Salju:** Penurunan drastis (>50%). Cuaca buruk adalah risiko terbesar bisnis.
        - **Mendung:** Masih stabil, namun lebih rendah dari cuaca cerah.
        """)
    st.markdown("---")

    # --- HARI KERJA VS LIBUR ---
    st.subheader("Pola Hari Kerja vs Hari Libur")
    col_eda3, col_eda4 = st.columns([2, 1])
    
    with col_eda3:
        # Boxplot Workingday
        fig_work, ax_work = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='workingday_label', y='count', data=main_df, palette="Purples", ax=ax_work)
        ax_work.set_ylabel("Jumlah Penyewaan")
        ax_work.set_xlabel("")
        st.pyplot(fig_work)
        
    with col_eda4:
        st.info("**Insight Tipe Hari:**")
        st.markdown("""
        - **Working Day:** Rata-rata sedikit lebih tinggi. Didominasi komuter (pekerja/pelajar).
        - **Holiday:** Variasi data lebih besar (fluktuatif). Menandakan penggunaan rekreasi yang tidak menentu.
        - **Kesimpulan:** Fungsi ganda sepeda sebagai alat transportasi (utama) dan rekreasi (sekunder).
        """)

    st.markdown("---")

    # --- MUSIM & TREN ---
    st.subheader("Interaksi Musim & Tren Suhu")
    
    # Clustered Barplot Musim
    st.markdown("**Perbandingan Musim dan Tipe Hari (The Summer Flip)**")
    fig_season, ax_season = plt.subplots(figsize=(12, 6))
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    sns.barplot(x='season_label', y='count', hue='workingday_label', data=main_df, 
                order=season_order, palette="PuRd", ax=ax_season)
    ax_season.legend(title='Tipe Hari')
    st.pyplot(fig_season)
    st.caption("*Insight: Di Musim Panas (Summer), penyewaan Hari Libur justru mengalahkan Hari Kerja (Wisata).*")
    
    # Scatter Plot Suhu
    st.markdown("**Korelasi Suhu dengan Jumlah Penyewa**")
    fig_temp, ax_temp = plt.subplots(figsize=(12, 6))
    sns.regplot(x='temp', y='count', data=main_df, scatter_kws={'alpha':0.5, 'color':'#7b1fa2'}, line_kws={'color':'#4a148c'}, ax=ax_temp)
    ax_temp.set_xlabel("Suhu (Normalized)")
    ax_temp.set_ylabel("Jumlah Penyewaan")
    st.pyplot(fig_temp)
    st.caption("*Insight: Korelasi Positif Kuat. Semakin hangat, semakin banyak penyewa.*")

# MACHINE LEARNING
with tab2:
    st.header("🤖 Prediksi Jumlah Penyewaan (Random Forest)")
    st.write("Model ini menggunakan **Random Forest Regresi** untuk memprediksi permintaan berdasarkan input kondisi.")

    # --- MODEL TRAINING ---
    # Persiapan Data
    drop_cols = ['instant', 'dteday', 'casual', 'registered', 'count', 
                 'season_label', 'weather_label', 'workingday_label']
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['count']
    
    # Train Model
    @st.cache_resource
    def train_model(X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    model = train_model(X, y)
    
    # --- USER INPUT ---
    st.subheader("🎮 Coba Prediksi")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        in_season = st.selectbox("Musim", options=['Spring', 'Summer', 'Fall', 'Winter'])
        in_weather = st.selectbox("Cuaca", options=['Clear', 'Misty', 'Light Rain/Snow'])
        in_temp = st.slider("Suhu (0=Dingin, 1=Panas)", 0.0, 1.0, 0.5)

    with col_in2:
        in_year = st.selectbox("Tahun (0=2011, 1=2012)", options=[0, 1], index=1)
        in_month = st.slider("Bulan (1-12)", 1, 12, 8)
        in_humidity = st.slider("Kelembaban (0-1)", 0.0, 1.0, 0.6)

    with col_in3:
        in_is_working = st.radio("Apakah Hari Kerja?", ["Ya", "Tidak"])
        in_wind = st.slider("Kecepatan Angin (0-1)", 0.0, 1.0, 0.2)
    
    # Mapping Input ke Format Model
    season_map = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}
    weather_map = {'Clear': 1, 'Misty': 2, 'Light Rain/Snow': 3}
    
    val_season = season_map[in_season]
    val_weather = weather_map[in_weather]
    val_workingday = 1 if in_is_working == "Ya" else 0
    val_holiday = 0 if val_workingday == 1 else 1 # Asumsi sederhana
    val_weekday = 3 if val_workingday == 1 else 6 # Asumsi sederhana
    
    # Prediksi Button
    if st.button("🔮 Prediksi Sekarang"):
        # Urutan kolom harus sama persis dengan X saat training
        # X columns: ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'humidity', 'windspeed']
        # Kita pakai atemp = temp + sedikit noise/biasa
        input_data = [[val_season, in_year, in_month, val_holiday, val_weekday, val_workingday, val_weather, in_temp, in_temp, in_humidity, in_wind]]
        
        prediction = model.predict(input_data)[0]
        st.success(f"🚴 Estimasi Jumlah Penyewa: **{int(prediction):,}** sepeda")

    st.markdown("---")
    
    # --- FEATURE IMPORTANCE ---
    st.subheader("🧠 Feature Importance (Analisis Faktor)")
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette="Purples_r", ax=ax_imp)
    ax_imp.set_title("Faktor Paling Berpengaruh Terhadap Penyewaan")
    st.pyplot(fig_imp)
    st.markdown("""
    **Insight Model :**
    1. **Suhu (Temp/Atemp)** adalah faktor pertama.
    2. **Tahun (Year)** menunjukkan pertumbuhan bisnis jangka panjang.
    3. **Cuaca & Kelembaban** lebih penting daripada kecepatan angin.
    """)

# KESIMPULAN
with tab3:
    st.header("📝 Kesimpulan & Insight Strategis")
    
    st.markdown("""
    <div style="background-color: #ede7f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4a148c;">
        <h4> Suhu adalah Kunci Pendapatan</h4>
        <p>Faktor paling penentu laris atau tidaknya sepeda adalah suhu udara. Semakin hangat cuaca, penyewaan semakin meningkat. 
        Sebaliknya, hujan adalah musuh utama bisnis ini karena bisa membuat pendapatan anjlok lebih dari 50% dalam sehari.</p>
    </div>
    <br>
    <div style="background-color: #ede7f6; padding: 20px; border-radius: 10px; border-left: 5px solid #7b1fa2;">
        <h4> Sepeda untuk Bekerja (Komuter)</h4>
        <p>Secara umum, sepeda lebih banyak disewa pada Hari Kerja (Senin-Jumat) dibandingkan hari libur. 
        Ini membuktikan bahwa pelanggan utama adalah orang yang menggunakan sepeda sebagai alat transportasi rutin ke kantor atau sekolah.</p>
    </div>
    <br>
    <div style="background-color: #ede7f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ab47bc;">
        <h4> Keunikan Musim Panas (Wisata)</h4>
        <p>Ada pengecualian unik di Musim Panas: orang justru lebih banyak menyewa sepeda di Hari Libur. 
        Ini tandanya, saat musim panas, fungsi sepeda berubah dari "alat transportasi" menjadi "alat rekreasi/wisata".</p>
    </div>
    <br>
    <div style="background-color: #ede7f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ce93d8;">
        <h4> Pertumbuhan Bisnis Positif</h4>
        <p>Tren tahunan menunjukkan arah positif. Jumlah pelanggan di tahun 2012 jauh lebih banyak daripada tahun 2011, 
        menandakan bisnis ini semakin populer dan berkembang sehat.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("💜 **Bike Sharing Dashboard**")