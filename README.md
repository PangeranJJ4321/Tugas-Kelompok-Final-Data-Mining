# 🎬 Analisis Prediktif Kesuksesan Finansial Film

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-green)](https://xgboost.readthedocs.io/en/stable/)
[![Imblearn](https://img.shields.io/badge/Imblearn-0.9%2B-purple)](https://imbalanced-learn.org/stable/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)](https://streamlit.io/)

Sebuah proyek analisis data dan Machine Learning komprehensif yang bertujuan untuk memprediksi kesuksesan finansial film. Proyek ini menggunakan model regresi untuk memprediksi pendapatan film dan model klasifikasi untuk mengkategorikan Return on Investment (ROI) ke dalam lima tingkat risiko yang berbeda. Dilengkapi dengan dashboard interaktif menggunakan Streamlit.

## 🌟 Fitur Utama

* **Prediksi Pendapatan Film:** Menggunakan model regresi (`RandomForestRegressor`, `XGBoostRegressor`) untuk memprediksi pendapatan bruto film dalam Dolar AS.
* **Klasifikasi Risiko ROI:** Mengkategorikan potensi keuntungan/kerugian film ke dalam 5 kelas risiko dinamis berbasis kuantil:
    * `Lowest Return`
    * `Low Return`
    * `Medium Return`
    * `Good Return`
    * `Highest Return`
* **Penanganan Imbalance Data Tingkat Lanjut:** Mengimplementasikan strategi canggih seperti `BorderlineSMOTE`, `SVMSMOTE`, `BalancedRandomForestClassifier`, dan *cost-sensitive learning* untuk meningkatkan performa klasifikasi pada kelas minoritas.
* **Rekayasa Fitur (Feature Engineering):** Penciptaan fitur-fitur baru yang fokus pada deteksi kasus ekstrem dan meningkatkan sinyal bagi model.
* **Hyperparameter Tuning:** Menggunakan `RandomizedSearchCV` dan `GridSearchCV` untuk mengoptimalkan kinerja model.
* **Dashboard Interaktif:** Aplikasi Streamlit untuk eksplorasi data, evaluasi model, dan prediksi film baru secara *real-time*.

## 🚀 Instalasi & Penggunaan

### Prasyarat

* Python 3.8+
* Git

### Langkah-langkah

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/NamaPenggunaAnda/nama-repo-anda.git](https://github.com/NamaPenggunaAnda/nama-repo-anda.git)
    cd nama-repo-anda
    ```

2.  **Buat Virtual Environment (Disarankan):**
    ```bash
    python -m venv venv
    # Di Windows
    .\venv\Scripts\activate
    # Di macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```
    (Jika `requirements.txt` belum ada, buat dengan perintah `pip freeze > requirements.txt` setelah menginstal semua library secara manual, atau daftar manual di bawah.)

    **Daftar Dependensi (`requirements.txt`):**
    ```
    pandas>=1.3.0
    numpy>=1.20.0
    scikit-learn>=1.0.0
    xgboost>=1.6.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
    imbalanced-learn>=0.9.0
    streamlit>=1.20.0
    joblib>=1.1.0
    ```

4.  **Latih Model:**
    Jalankan skrip pelatihan model. Ini akan melakukan pra-pemrosesan data, *feature engineering*, melatih model regresi dan klasifikasi (dengan *hyperparameter tuning* dan penanganan *imbalance data*), dan menyimpan model-model yang dilatih ke folder `models/`.
    ```bash
    python model.py
    ```
    *Catatan: Proses ini mungkin memakan waktu tergantung pada spesifikasi komputer Anda dan ukuran dataset.*

5.  **Jalankan Dashboard Streamlit:**
    Setelah model selesai dilatih dan disimpan, Anda dapat menjalankan aplikasi dashboard interaktif.
    ```bash
    streamlit run app_v2.py
    ```
    Aplikasi akan terbuka di browser web Anda (biasanya `http://localhost:8501`).

## ⚙️ Konfigurasi (Opsional)

* **Kurs Dolar ke Rupiah:** Anda dapat menyesuaikan kurs Dolar AS ke Rupiah Indonesia di sidebar dashboard Streamlit untuk mendapatkan konversi mata uang yang relevan.
* **Model untuk Prediksi:** Di halaman "Buat Prediksi" dashboard, Anda dapat memilih model regresi dan klasifikasi mana yang akan digunakan untuk menghasilkan prediksi.

