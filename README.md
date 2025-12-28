# bike-sharing-dashboard
# 🚲 Proyek Analisis Data: Bike Sharing Dataset

Proyek ini bertujuan untuk menganalisis pola penyewaan sepeda pada sistem Bike Sharing. Analisis difokuskan pada pengaruh faktor lingkungan (cuaca, suhu, musim) dan faktor waktu (hari kerja vs hari libur) terhadap jumlah penyewaan sepeda. Proyek ini juga mencakup dashboard interaktif berbasis **Streamlit** untuk memvisualisasikan data dan melakukan prediksi sederhana menggunakan Machine Learning.

## 📂 Dataset
Dataset yang digunakan dalam proyek ini adalah **`day.csv`**.
Dataset ini berisi data harian penyewaan sepeda selama dua tahun (2011-2012) dengan fitur-fitur utama sebagai berikut:
- **dteday**: Tanggal pencatatan.
- **season**: Musim (1: Semi, 2: Panas, 3: Gugur, 4: Dingin).
- **mnth**: Bulan (1-12).
- **holiday**: Apakah hari libur nasional atau tidak.
- **weekday**: Hari dalam seminggu.
- **workingday**: Apakah hari kerja (bukan akhir pekan/libur).
- **weathersit**: Kondisi cuaca (1: Cerah, 2: Mendung, 3: Hujan Ringan, 4: Hujan Lebat).
- **temp**: Suhu yang dinormalisasi.
- **hum**: Kelembaban yang dinormalisasi.
- **windspeed**: Kecepatan angin yang dinormalisasi.
- **cnt**: Jumlah total penyewaan sepeda (Target).

## ⚙️ Cara Menjalankan Project

### 1. Pastikan telah menginstal **Python** (versi 3.9 atau lebih baru). Disarankan untuk menggunakan virtual environment.

### 2. Install semua library yang dibutuhkan dengan menjalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```

> Pastikan file `requirements.txt` berisi library berikut:
> * pandas
> * numpy
> * matplotlib
> * seaborn
> * streamlit
> * scikit-learn

### 3. Menjalankan Dashboard Streamlit

Untuk membuka dashboard interaktif, jalankan perintah berikut di terminal:
```bash
streamlit run app.py
```

Setelah dijalankan, dashboard akan otomatis terbuka di browser Anda (biasanya di alamat `http://localhost:8501`).

## 📊 Ringkasan Insight Bisnis
Berdasarkan analisis data yang telah dilakukan, berikut adalah temuan-temuan utama:

1. **Suhu adalah Faktor Utama:**
Terdapat korelasi positif yang kuat antara suhu dan jumlah penyewaan. Semakin hangat cuaca, semakin tinggi minat orang untuk menyewa sepeda. Cuaca buruk (hujan/salju) dapat menurunkan pendapatan secara drastis (>50%).
2. **Musim Gugur (Fall) adalah Periode Paling Baik:**
Musim gugur mencatatkan rata-rata penyewaan tertinggi dibandingkan musim lainnya yang paling optimal untuk strategi pemasaran dan ketersediaan stok maksimal.
3. **Pola Komuter vs Rekreasi:**
Pada umumnya, sepeda lebih banyak digunakan di **Hari Kerja** (Working Day), menandakan dominasi pengguna komuter (pekerja/pelajar). Namun, pada **Musim Panas (Summer)**, terjadi anomali di mana penyewaan di **Hari Libur** melonjak tinggi, menandakan pergeseran fungsi sepeda menjadi alat rekreasi/wisata.
4. **Pertumbuhan Bisnis Positif:**
Tren tahunan menunjukkan peningkatan yang sehat. Jumlah penyewaan pada tahun 2012 secara konsisten lebih tinggi dibandingkan tahun 2011, menunjukkan ekspansi pasar yang sukses.
