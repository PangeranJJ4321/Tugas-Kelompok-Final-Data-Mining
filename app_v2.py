import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from datetime import datetime
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Dashboard Prediksi Film 🎥 | Proyek Analitik Data",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling untuk Background Gradasi dan Beautification ---
st.markdown("""
<style>
    /* Background Gradasi Utama */
    .main > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f093fb 0%, #f5576c 100%);
        color: white; /* Ensure text visibility on gradient */
    }
    .sidebar .sidebar-content .stRadio > label {
        color: white; /* Ensure radio button labels are white */
    }
    .sidebar .sidebar-content .stNumberInput label {
        color: white; /* Ensure number input labels are white */
    }

    /* Custom CSS untuk Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        font-size: 6px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(31,38,135,0.37);
        color: white; /* Text color for metric cards */
    }
    .stMetric {
        font-size: 12px;
        color: white; /* Ensure metric values are white */
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7, #dda0dd);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 20px;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Subheader Styling */
    .sub-header {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 15px 0;
    }

    /* Info Box Styling */
    .info-box {
        background: linear-gradient(135deg, rgba(78,205,196,0.1) 0%, rgba(255,107,107,0.1) 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4ECDC4;
        backdrop-filter: blur(5px);
        color: white; /* Text color for info box */
    }

    /* Success Box Styling */
    .success-box {
        background: linear-gradient(135deg, rgba(150,206,180,0.2) 0%, rgba(69,183,209,0.2) 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #96CEB4;
        backdrop-filter: blur(5px);
        color: white; /* Text color for success box */
    }

    /* Warning Box Styling */
    .warning-box {
        background: linear-gradient(135deg, rgba(255,234,167,0.2) 0%, rgba(255,183,77,0.2) 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FFB74D;
        backdrop-filter: blur(5px);
        color: white; /* Text color for warning box */
    }

    /* Error Box Styling */
    .error-box {
        background: linear-gradient(135deg, rgba(255,107,107,0.2) 0%, rgba(238,82,83,0.2) 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF6B6B;
        backdrop-filter: blur(5px);
        color: white; /* Text color for error box */
    }

    /* DataFrame Styling */
    .stDataFrame { /* Target Streamlit's rendered dataframe directly */
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        overflow: hidden; /* Ensures border-radius is respected */
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    /* Selectbox Styling */
    .stSelectbox > label, .stMultiSelect > label {
        color: white; /* Label color for selectbox/multiselect */
    }
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        backdrop-filter: blur(10px);
        color: white; /* Selected item text color */
        border: 1px solid rgba(255,255,255,0.2);
    }
    .stSelectbox > div > div > div[data-baseweb="select"] > div {
        background-color: transparent !important; /* Remove default background */
    }
    .stSelectbox > div > div > div[data-baseweb="select"] > div > div {
        color: white !important; /* Ensure selected text is white */
    }

    /* Text Input Styling */
    .stTextInput > label, .stDateInput > label {
        color: white; /* Label color for text/date input */
    }
    .stTextInput > div > div > input, .stDateInput > div > div > input {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }

    /* Number Input Styling */
    .stNumberInput > label {
        color: white; /* Label color for number input */
    }
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }

    /* Slider Styling */
    .stSlider > label {
        color: white; /* Label color for slider */
    }
    .stSlider .st-fx { /* Track */
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .stSlider .st-fy { /* Fill */
        background: #4ECDC4;
    }
    .stSlider .st-fz { /* Thumb */
        background: #FF6B6B;
        border: 2px solid white;
    }
    .stSlider .st-c1 { /* Value text */
        color: white;
    }

    /* Sidebar Navigation Styling */
    .sidebar .sidebar-content .stRadio > div {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 10px;
        backdrop-filter: blur(10px);
    }
    .sidebar .sidebar-content .stRadio label {
        color: white !important; /* Ensure radio labels in sidebar are white */
    }
    .sidebar .sidebar-content .stRadio input:checked + div {
        background: #FF6B6B !important; /* Selected radio button background */
        color: white !important;
    }

    /* Plot Background */
    .plot-container {
        background: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31,38,135,0.37);
    }

    /* General text color in main content */
    .stMarkdown, .stText, .stCode {
        color: white;
    }

</style>
""", unsafe_allow_html=True)

# --- Variabel Global & Konfigurasi Awal ---
DEFAULT_DOLLAR_TO_RUPIAH_RATE = 15500
DATA_PATH = "data/raw/data_mentah.csv"
MODEL_REG_PATH = "models/regression/"
MODEL_CLS_PATH = "models/classification/"
SCALER_PATH = "models/regression/scaler.pkl"

# Definisikan semua genre yang mungkin untuk konsistensi
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
    'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
    'TV Movie', 'Thriller', 'War', 'Western'
]

# Definisikan semua kategori ROI yang mungkin untuk LabelEncoder
# Ini HARUS konsisten dengan definisi di model.py -> preprocess_data
ALL_ROI_CATEGORIES_ORDERED = [
    'Extreme Loss',
    'Significant Loss',
    'Marginal Profit',
    'Good Profit',
    'Blockbuster/High Profit'
]
label_encoder = LabelEncoder()
label_encoder.fit(ALL_ROI_CATEGORIES_ORDERED) # Fit label encoder dengan urutan yang pasti

# --- Fungsi Pembantu ---
@st.cache_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memproses data film mentah ke format yang dibutuhkan untuk pelatihan model.
    Menambahkan penanganan nilai NaN yang lebih robust.
    """
    processed_df = df.copy()

    if 'release_date' in processed_df.columns:
        processed_df['release_date'] = pd.to_datetime(processed_df['release_date'], errors='coerce')
        processed_df['release_year'] = processed_df['release_date'].dt.year.fillna(processed_df['release_date'].dt.year.mode()[0] if not processed_df['release_date'].dt.year.mode().empty else 2000).astype(int)
        processed_df['release_month'] = processed_df['release_date'].dt.month.fillna(processed_df['release_date'].dt.month.mode()[0] if not processed_df['release_date'].dt.month.mode().empty else 1).astype(int)
    else:
        st.markdown('<div class="warning-box">Kolom \'release_date\' tidak ditemukan. Menggunakan nilai default.</div>', unsafe_allow_html=True)
        processed_df['release_year'] = 2000
        processed_df['release_month'] = 1

    numerical_cols = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
    for col in numerical_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].replace(0, np.nan)
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        else:
            processed_df[col] = 0

    processed_df['budget_adjusted'] = processed_df['budget'].replace(0, processed_df['budget'].median() if processed_df['budget'].median() > 0 else 1)

    processed_df['ROI'] = ((processed_df['revenue'] - processed_df['budget_adjusted']) / processed_df['budget_adjusted']) * 100

    # Definisi 5 Kategori ROI yang baru
    conditions = [
        (processed_df['ROI'] < -75), # Extreme Loss
        (processed_df['ROI'] >= -75) & (processed_df['ROI'] < 0), # Significant Loss
        (processed_df['ROI'] >= 0) & (processed_df['ROI'] < 50), # Marginal Profit / Break-Even
        (processed_df['ROI'] >= 50) & (processed_df['ROI'] < 200), # Good Profit
        (processed_df['ROI'] >= 200) # Blockbuster / High Profit
    ]

    choices_map = [
        'Extreme Loss',
        'Significant Loss',
        'Marginal Profit',
        'Good Profit',
        'Blockbuster/High Profit'
    ]
    processed_df['ROI_category'] = np.select(conditions, choices_map, default='Unknown')

    # Handle if any 'Unknown' still exists (walaupun seharusnya tidak jika kondisi mencakup semua kemungkinan)
    if 'Unknown' in processed_df['ROI_category'].unique():
        processed_df.loc[processed_df['ROI_category'] == 'Unknown', 'ROI_category'] = 'Marginal Profit'


    processed_df['lang_en'] = (processed_df['original_language'] == 'en').astype(int) if 'original_language' in processed_df.columns else 0
    processed_df['lang_others'] = (processed_df['original_language'] != 'en').astype(int) if 'original_language' in processed_df.columns else 0

    if 'genres' in processed_df.columns:
        processed_df['genres'] = processed_df['genres'].fillna('')
        genre_df = processed_df['genres'].str.get_dummies(sep=', ')
        for genre in ALL_GENRES:
            genre_col_name = f'genre_{genre.lower().replace(" ", "_")}'
            genre_cols_matched = [col for col in genre_df.columns if col.lower() == genre.lower()]
            if genre_cols_matched:
                processed_df[genre_col_name] = genre_df[genre_cols_matched[0]]
            else:
                processed_df[genre_col_name] = 0
    else:
        for genre in ALL_GENRES:
            processed_df[f'genre_{genre.lower().replace(" ", "_")}'] = 0

    base_columns = [
        'release_year', 'release_month', 'budget', 'popularity', 'runtime',
        'vote_average', 'vote_count', 'lang_en', 'lang_others'
    ]
    genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in ALL_GENRES]
    target_columns = ['revenue', 'ROI', 'ROI_category']

    final_columns = base_columns + genre_columns + target_columns

    for col in final_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    for col in base_columns + genre_columns + ['revenue', 'ROI']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)

    return processed_df[final_columns]

@st.cache_data
def split_features_target(df: pd.DataFrame, _scaler: StandardScaler = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Memisahkan DataFrame menjadi fitur, target pendapatan, dan target kategori ROI.
    Menerapkan penskalaan fitur.
    """
    base_columns = [
        'release_year', 'release_month', 'budget', 'popularity', 'runtime',
        'vote_average', 'vote_count', 'lang_en', 'lang_others'
    ]
    genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in ALL_GENRES]
    feature_cols = base_columns + genre_columns

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].copy()
    y_regression = df['revenue']
    y_classification = df['ROI_category']

    numerical_features_to_scale = [
        'budget', 'popularity', 'runtime', 'vote_average', 'vote_count',
        'release_year', 'release_month'
    ]
    numerical_features_to_scale = [col for col in numerical_features_to_scale if col in X.columns]

    if numerical_features_to_scale:
        if _scaler is None:
            _scaler = StandardScaler()
            X[numerical_features_to_scale] = _scaler.fit_transform(X[numerical_features_to_scale])
        else:
            X[numerical_features_to_scale] = _scaler.transform(X[numerical_features_to_scale])
    else:
        _scaler = None

    return X, y_regression, y_classification, _scaler

# --- Muat Model (dengan caching dan penanganan error) ---
@st.cache_resource
def load_trained_models():
    """Memuat model yang sudah dilatih dari file."""
    models = {}

    # Muat scaler
    try:
        models['scaler'] = pickle.load(open(SCALER_PATH, 'rb'))
    except FileNotFoundError as e:
        st.markdown(f'<div class="error-box">🚨 Error: File Scaler tidak ditemukan di {SCALER_PATH}. Pastikan Anda sudah melatih model dan menyimpan scaler. Detail: {e}</div>', unsafe_allow_html=True)
        st.stop()
    except Exception as e:
        st.markdown(f'<div class="error-box">Terjadi kesalahan saat memuat Scaler: {e}</div>', unsafe_allow_html=True)
        st.stop()

    try:
        # Muat model regresi
        models['xgb_regressor'] = pickle.load(open(os.path.join(MODEL_REG_PATH, 'xgboost_regressor.pkl'), 'rb'))
        models['rf_regressor'] = pickle.load(open(os.path.join(MODEL_REG_PATH, 'random_forest_regressor.pkl'), 'rb'))

        # Muat model klasifikasi (dengan nama file yang disesuaikan setelah resampling)
        # Pastikan nama file ini sesuai dengan yang disimpan di model.py
        models['xgb_classifier'] = pickle.load(open(os.path.join(MODEL_CLS_PATH, 'xgboost_smotetomek_classifier.pkl'), 'rb'))
        models['rf_classifier'] = pickle.load(open(os.path.join(MODEL_CLS_PATH, 'random_forest_smotetomek_classifier.pkl'), 'rb'))

        # Muat feature importance jika ada
        models['xgb_reg_importance'] = pd.read_csv(os.path.join(MODEL_REG_PATH, 'xgboost_regressor_feature_importance.csv')) if os.path.exists(os.path.join(MODEL_REG_PATH, 'xgboost_regressor_feature_importance.csv')) else None
        models['rf_reg_importance'] = pd.read_csv(os.path.join(MODEL_REG_PATH, 'random_forest_regressor_feature_importance.csv')) if os.path.exists(os.path.join(MODEL_REG_PATH, 'random_forest_regressor_feature_importance.csv')) else None

        models['xgb_cls_importance'] = pd.read_csv(os.path.join(MODEL_CLS_PATH, 'xgboost_smotetomek_classifier_importance.csv')) if os.path.exists(os.path.join(MODEL_CLS_PATH, 'xgboost_smotetomek_classifier_importance.csv')) else None
        models['rf_cls_importance'] = pd.read_csv(os.path.join(MODEL_CLS_PATH, 'random_forest_smotetomek_classifier_importance.csv')) if os.path.exists(os.path.join(MODEL_CLS_PATH, 'random_forest_smotetomek_classifier_importance.csv')) else None

    except FileNotFoundError as e:
        st.markdown(f'<div class="error-box">🚨 Error: Salah satu file model atau feature importance tidak ditemukan. Pastikan Anda sudah melatih model dan menyimpannya di direktori `models/regression/` dan `models/classification/` dengan nama file yang benar (termasuk teknik resampling jika ada). Detail: {e}</div>', unsafe_allow_html=True)
        st.stop()
    except Exception as e:
        st.markdown(f'<div class="error-box">Terjadi kesalahan saat memuat model: {e}</div>', unsafe_allow_html=True)
        st.stop()
    return models

# Muat semua model saat aplikasi dimulai
models = load_trained_models()

# Dapatkan scaler dari model yang dimuat
global_scaler = models['scaler']

# --- Muat dan Proses Data untuk Evaluasi ---
@st.cache_data
def load_and_prepare_data_for_dashboard(file_path, _scaler_obj: StandardScaler):
    """
    Memuat dan mempersiapkan data utama untuk analisis dan evaluasi model.
    """
    if not os.path.exists(file_path):
        st.markdown(f'<div class="error-box">🚨 Error: File data mentah tidak ditemukan di {file_path}. Mohon pastikan file ada.</div>', unsafe_allow_html=True)
        st.stop()

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['budget', 'revenue'])
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

    if df.empty:
        st.markdown('<div class="error-box">Dataset kosong setelah pembersihan data. Tidak dapat melanjutkan.</div>', unsafe_allow_html=True)
        st.stop()

    processed_df = preprocess_data(df.copy())

    # Gunakan _scaler_obj yang dilewatkan saat memisahkan data
    X, y_regression, y_classification, _ = split_features_target(processed_df, _scaler=_scaler_obj)

    # Bagi data untuk evaluasi
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )

    # Untuk mendapatkan pembagian data klasifikasi asli (sebelum encoding/resampling)
    y_cls_train_original, y_cls_test_original = train_test_split(
        processed_df['ROI_category'], test_size=0.2, random_state=42, stratify=processed_df['ROI_category']
    )

    # Encode y_cls_test untuk evaluasi model klasifikasi
    y_cls_test_encoded = label_encoder.transform(y_cls_test_original)

    return processed_df, X_train, X_test, y_reg_train, y_reg_test, y_cls_train_original, y_cls_test_encoded

# Dapatkan data yang disiapkan
processed_df, X_train, X_test, y_reg_train, y_reg_test, y_cls_train_original, y_cls_test_encoded = load_and_prepare_data_for_dashboard(DATA_PATH, global_scaler)

# --- Header Utama Dashboard dengan Animasi ---
st.markdown('<h1 class="main-header">🎬 Analisis & Prediksi Proyek Film</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box"><p style="text-align: center; font-size: 1.1rem; margin: 0;"><em>Dashboard interaktif untuk memahami tren film, membandingkan performa model Machine Learning, dan memprediksi kesuksesan film baru.</em></p></div>', unsafe_allow_html=True)

# --- Sidebar Navigasi & Pengaturan Global ---
st.sidebar.markdown("# 🎯 Navigasi & Pengaturan")
page = st.sidebar.radio(
    "**Pilih Halaman**",
    ["🏠 Ikhtisar Data", "🔮 Buat Prediksi"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pengaturan Global")
st.session_state.dollar_to_rupiah_rate = st.sidebar.number_input(
    "💱 Kurs 1 USD ke IDR",
    min_value=10000,
    max_value=20000,
    value=DEFAULT_DOLLAR_TO_RUPIAH_RATE,
    step=100,
    help="Atur nilai tukar Dolar AS ke Rupiah Indonesia."
)

st.sidebar.markdown(f'<div class="success-box">💰 <strong>Kurs saat ini:</strong><br/>1 USD = Rp {st.session_state.dollar_to_rupiah_rate:,.0f}</div>', unsafe_allow_html=True)

# Tambahkan informasi tambahan di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Status Dashboard")
st.sidebar.markdown(f'<div class="info-box">✅ <strong>Dataset:</strong> {len(processed_df):,} film<br/>🤖 <strong>Models:</strong> Aktif<br/>🎯 <strong>Prediksi:</strong> Siap</div>', unsafe_allow_html=True)

# --- Halaman 1: Ikhtisar Data ---
if page == "🏠 Ikhtisar Data":
    st.markdown('<h2 class="sub-header">📊 Ikhtisar & Eksplorasi Data Film</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Bagian ini menyajikan gambaran umum tentang data film, termasuk distribusi kategori ROI dan pendapatan.</div>', unsafe_allow_html=True)

    # Container untuk dataset overview
    with st.container():
        st.markdown('<h3 class="sub-header">📋 Cuplikan Data yang Diproses</h3>', unsafe_allow_html=True)

        # Style dataframe dengan CSS custom
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.dataframe(
            processed_df.head(10),
            use_container_width=True,
            height=350
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="success-box">✨ Dataset berisi <strong>{len(processed_df):,}</strong> catatan film yang siap dianalisis dan dimodelkan.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Grid layout untuk visualisasi
    col_roi, col_revenue = st.columns([1, 1], gap="large")

    with col_roi:
        st.markdown('<h3 class="sub-header">🎯 Distribusi Kategori ROI</h3>', unsafe_allow_html=True)

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig_roi, ax_roi = plt.subplots(figsize=(8, 6))

        # Set style yang lebih cantik
        plt.style.use('seaborn-v0_8-darkgrid')
        # Sesuaikan warna jika diinginkan, sekarang ada 5 kategori
        # Contoh: Merah (rugi besar), Oranye (rugi signifikan), Biru muda (impas/profit tipis), Hijau (profit lumayan), Ungu (blockbuster)
        colors = ['#dc3545', '#fd7e14', '#17a2b8', '#28a745', '#6f42c1']

        sns.countplot(data=processed_df, x='ROI_category', ax=ax_roi,
                      palette=colors, order=ALL_ROI_CATEGORIES_ORDERED)

        ax_roi.set_title('Distribusi Kategori ROI Film', fontsize=16, fontweight='bold', pad=20, color='black')
        ax_roi.set_xlabel('Kategori ROI', fontsize=14, fontweight='bold', color='black')
        ax_roi.set_ylabel('Jumlah Film', fontsize=14, fontweight='bold', color='black')
        ax_roi.tick_params(axis='x', rotation=45, labelsize=12, colors='black')
        ax_roi.tick_params(axis='y', labelsize=12, colors='black')

        # Tambahkan grid dan styling
        ax_roi.grid(True, alpha=0.3)
        ax_roi.set_facecolor('#f8f9fa')

        # Tambahkan nilai di atas bar
        for i, p in enumerate(ax_roi.patches):
            ax_roi.annotate(f'{int(p.get_height())}',
                           (p.get_x() + p.get_width()/2., p.get_height()),
                           ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

        plt.tight_layout()
        st.pyplot(fig_roi)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <strong>📊 Kategori ROI menunjukkan potensi keuntungan film:</strong><br/>
        🔴 <strong>Extreme Loss</strong> (ROI < -75%) - Kerugian yang sangat besar<br/>
        🟠 <strong>Significant Loss</strong> (-75% &le; ROI < 0%) - Kerugian yang signifikan<br/>
        🔵 <strong>Marginal Profit</strong> (0% &le; ROI < 50%) - Impas atau keuntungan tipis<br/>
        🟢 <strong>Good Profit</strong> (50% &le; ROI < 200%) - Keuntungan lumayan<br/>
        🟣 <strong>Blockbuster/High Profit</strong> (ROI &ge; 200%) - Film yang sangat menguntungkan/blockbuster
        </div>
        """, unsafe_allow_html=True)

    with col_revenue:
        st.markdown('<h3 class="sub-header">💰 Distribusi Pendapatan Film</h3>', unsafe_allow_html=True)

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig_revenue, ax_revenue = plt.subplots(figsize=(8, 6))

        # Histogram dengan styling yang lebih menarik
        sns.histplot(data=processed_df, x='revenue', bins=50, kde=True, ax=ax_revenue,
                    color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=0.5)

        ax_revenue.set_title('Distribusi Pendapatan Film (USD)', fontsize=16, fontweight='bold', pad=20, color='black')
        ax_revenue.set_xlabel('Pendapatan (USD)', fontsize=14, fontweight='bold', color='black')
        ax_revenue.set_ylabel('Jumlah Film', fontsize=14, fontweight='bold', color='black')
        ax_revenue.ticklabel_format(style='plain', axis='x')
        ax_revenue.tick_params(labelsize=12, colors='black')

        # Styling tambahan
        ax_revenue.grid(True, alpha=0.3)
        ax_revenue.set_facecolor('#f8f9fa')

        plt.tight_layout()
        st.pyplot(fig_revenue)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <strong>📈 Insight:</strong> Visualisasi menunjukkan sebaran pendapatan film dalam dolar AS.
        Sebagian besar film cenderung memiliki pendapatan yang lebih rendah, dengan sedikit film
        berpendapatan sangat tinggi (distribusi ekor panjang).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Statistik ringkasan dengan styling yang lebih menarik
    st.markdown('<h3 class="sub-header">📊 Statistik Ringkasan Fitur Numerik</h3>', unsafe_allow_html=True)

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    stats_df = processed_df.describe().loc[['mean', 'std', 'min', 'max']].T
    st.dataframe(
        stats_df.style.format(precision=2),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Tabel di atas menyajikan statistik deskriptif untuk fitur-fitur numerik kunci dalam dataset yang telah diproses.
    Ini memberikan gambaran cepat tentang pusat, sebaran, dan rentang nilai data.
    </div>
    """, unsafe_allow_html=True)


# --- Halaman 2: Performa Model ---
# elif page == "📊 Performa Model":
#     st.markdown('<h2 class="sub-header">🚀 Perbandingan Performa Model Machine Learning</h2>', unsafe_allow_html=True)
#     st.markdown('<div class="info-box">Evaluasi performa model klasifikasi (untuk risiko ROI) dan model regresi (untuk pendapatan) menggunakan metrik standar.</div>', unsafe_allow_html=True)

#     st.markdown('<h3 class="sub-header">🎯 Evaluasi Model Klasifikasi (Prediksi Kategori ROI)</h3>', unsafe_allow_html=True)
#     col_cls_metrics, col_cls_plot = st.columns(2)

#     with col_cls_metrics:
#         st.markdown("<h4 style='color:white;'>Akurasi Model Klasifikasi</h4>", unsafe_allow_html=True)

#         rf_cls_accuracy = None
#         xgb_cls_accuracy = None

#         if models['rf_classifier'] is not None:
#             y_pred_rf_cls = models['rf_classifier'].predict(X_test)
#             rf_cls_accuracy = accuracy_score(y_cls_test_encoded, y_pred_rf_cls)
#             st.markdown(f'<div class="success-box"><strong>Random Forest Classifier Akurasi:</strong> `{rf_cls_accuracy:.4f}`</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="warning-box">Random Forest Classifier tidak dimuat.</div>', unsafe_allow_html=True)

#         if models['xgb_classifier'] is not None:
#             y_pred_xgb_cls = models['xgb_classifier'].predict(X_test)
#             xgb_cls_accuracy = accuracy_score(y_cls_test_encoded, y_pred_xgb_cls)
#             st.markdown(f'<div class="success-box"><strong>XGBoost Classifier Akurasi:</strong> `{xgb_cls_accuracy:.4f}`</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="warning-box">XGBoost Classifier tidak dimuat.</div>', unsafe_allow_html=True)

#         cls_models_names = []
#         cls_accuracy_scores = []
#         if rf_cls_accuracy is not None:
#             cls_models_names.append("Random Forest")
#             cls_accuracy_scores.append(rf_cls_accuracy)
#         if xgb_cls_accuracy is not None:
#             cls_models_names.append("XGBoost")
#             cls_accuracy_scores.append(xgb_cls_accuracy)

#         if cls_models_names:
#             st.markdown('<div class="plot-container">', unsafe_allow_html=True)
#             fig_cls_bar, ax_cls_bar = plt.subplots(figsize=(7, 4))
#             sns.barplot(x=cls_models_names, y=cls_accuracy_scores, palette='coolwarm', ax=ax_cls_bar)
#             ax_cls_bar.set_title('Perbandingan Akurasi Model Klasifikasi', fontsize=14, color='black')
#             ax_cls_bar.set_ylim(0, 1)
#             ax_cls_bar.set_ylabel('Akurasi', fontsize=12, color='black')
#             ax_cls_bar.tick_params(axis='x', colors='black')
#             ax_cls_bar.tick_params(axis='y', colors='black')
#             ax_cls_bar.set_facecolor('#f8f9fa') # Set plot background
#             plt.tight_layout()
#             st.pyplot(fig_cls_bar)
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown('<div class="info-box">Grafik ini menunjukkan akurasi model klasifikasi pada data uji. Akurasi adalah proporsi prediksi yang benar dari total prediksi.</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="info-box">Tidak ada model klasifikasi yang tersedia untuk perbandingan akurasi.</div>', unsafe_allow_html=True)

#     with col_cls_plot:
#         st.markdown("<h4 style='color:white;'>Laporan Klasifikasi & Confusion Matrix (XGBoost Classifier)</h4>", unsafe_allow_html=True)
#         if models['xgb_classifier'] is not None and 'y_pred_xgb_cls' in locals():
#             st.markdown('<div class="info-box">Laporan Klasifikasi:</div>', unsafe_allow_html=True)
#             y_pred_xgb_cls_decoded = label_encoder.inverse_transform(y_pred_xgb_cls)
#             y_cls_test_decoded = label_encoder.inverse_transform(y_cls_test_encoded)
#             st.code(classification_report(y_cls_test_decoded, y_pred_xgb_cls_decoded, target_names=ALL_ROI_CATEGORIES_ORDERED, zero_division=0))

#             st.markdown('<div class="plot-container">', unsafe_allow_html=True)
#             # Pastikan labels di confusion_matrix sesuai dengan ALL_ROI_CATEGORIES_ORDERED
#             cm = confusion_matrix(y_cls_test_decoded, y_pred_xgb_cls_decoded, labels=ALL_ROI_CATEGORIES_ORDERED)
#             fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
#                         xticklabels=ALL_ROI_CATEGORIES_ORDERED, yticklabels=ALL_ROI_CATEGORIES_ORDERED,
#                         linewidths=.5, linecolor='black')
#             ax_cm.set_title('Confusion Matrix - XGBoost Classifier', fontsize=14, color='black')
#             ax_cm.set_xlabel('Prediksi', fontsize=12, color='black')
#             ax_cm.set_ylabel('Aktual', fontsize=12, color='black')
#             ax_cm.tick_params(axis='x', colors='black')
#             ax_cm.tick_params(axis='y', colors='black')
#             plt.tight_layout()
#             st.pyplot(fig_cm)
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown('<div class="info-box">Laporan klasifikasi memberikan metrik presisi, recall, dan f1-score per kelas. Confusion matrix menunjukkan jumlah prediksi benar dan salah untuk setiap kategori risiko, dengan diagonal menunjukkan prediksi yang benar.</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="info-box">XGBoost Classifier tidak tersedia untuk menampilkan laporan dan confusion matrix, atau prediksinya belum dihitung.</div>', unsafe_allow_html=True)

#     st.markdown("---")

#     st.markdown('<h3 class="sub-header">💰 Evaluasi Model Regresi (Prediksi Pendapatan)</h3>', unsafe_allow_html=True)
#     col_reg_metrics, col_reg_plot = st.columns(2)

#     with col_reg_metrics:
#         st.markdown("<h4 style='color:white;'>Metrik Model Regresi (RMSE & R²)</h4>", unsafe_allow_html=True)
#         rf_reg_rmse, rf_reg_r2 = None, None
#         xgb_reg_rmse, xgb_reg_r2 = None, None

#         if models['rf_regressor'] is not None:
#             y_pred_rf_reg = models['rf_regressor'].predict(X_test)
#             rf_reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_rf_reg))
#             rf_reg_r2 = r2_score(y_reg_test, y_pred_rf_reg)
#             st.markdown(f'<div class="success-box"><strong>Random Forest Regressor RMSE:</strong> `{rf_reg_rmse:,.2f}`</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="success-box"><strong>Random Forest Regressor R²:</strong> `{rf_reg_r2:.4f}`</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="warning-box">Random Forest Regressor tidak dimuat.</div>', unsafe_allow_html=True)

#         if models['xgb_regressor'] is not None:
#             y_pred_xgb_reg = models['xgb_regressor'].predict(X_test)
#             xgb_reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_xgb_reg))
#             xgb_reg_r2 = r2_score(y_reg_test, y_pred_xgb_reg)
#             st.markdown(f'<div class="success-box"><strong>XGBoost Regressor RMSE:</strong> `{xgb_reg_rmse:,.2f}`</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="success-box"><strong>XGBoost Regressor R²:</strong> `{xgb_reg_r2:.4f}`</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="warning-box">XGBoost Regressor tidak dimuat.</div>', unsafe_allow_html=True)

#         reg_models_names = []
#         reg_rmse_scores = []
#         reg_r2_scores = []
#         if rf_reg_rmse is not None:
#             reg_models_names.append("Random Forest")
#             reg_rmse_scores.append(rf_reg_rmse)
#             reg_r2_scores.append(rf_reg_r2)
#         if xgb_reg_rmse is not None:
#             reg_models_names.append("XGBoost")
#             reg_rmse_scores.append(xgb_reg_rmse)
#             reg_r2_scores.append(xgb_reg_r2)

#         if reg_models_names:
#             st.markdown('<div class="plot-container">', unsafe_allow_html=True)
#             fig_rmse_bar, ax_rmse_bar = plt.subplots(figsize=(7, 4))
#             sns.barplot(x=reg_models_names, y=reg_rmse_scores, palette='plasma', ax=ax_rmse_bar)
#             ax_rmse_bar.set_title('Perbandingan RMSE Model Regresi', fontsize=14, color='black')
#             ax_rmse_bar.set_ylabel('RMSE (USD)', fontsize=12, color='black')
#             ax_rmse_bar.tick_params(axis='x', colors='black')
#             ax_rmse_bar.tick_params(axis='y', colors='black')
#             ax_rmse_bar.set_facecolor('#f8f9fa')
#             plt.tight_layout()
#             st.pyplot(fig_rmse_bar)
#             st.markdown('</div>', unsafe_allow_html=True)

#             st.markdown('<div class="plot-container">', unsafe_allow_html=True)
#             fig_r2_bar, ax_r2_bar = plt.subplots(figsize=(7, 4))
#             sns.barplot(x=reg_models_names, y=reg_r2_scores, palette='cividis', ax=ax_r2_bar)
#             ax_r2_bar.set_title('Perbandingan R² Model Regresi', fontsize=14, color='black')
#             ax_r2_bar.set_ylim(0, 1)
#             ax_r2_bar.set_ylabel('Skor R²', fontsize=12, color='black')
#             ax_r2_bar.tick_params(axis='x', colors='black')
#             ax_r2_bar.tick_params(axis='y', colors='black')
#             ax_r2_bar.set_facecolor('#f8f9fa')
#             plt.tight_layout()
#             st.pyplot(fig_r2_bar)
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown('<div class="info-box">RMSE (Root Mean Squared Error) mengukur rata-rata besarnya kesalahan prediksi dalam unit dolar AS (lebih rendah lebih baik). R-squared (R²) menunjukkan proporsi varians dalam variabel dependen yang dapat dijelaskan oleh model (lebih tinggi lebih baik, maksimal 1).</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="info-box">Tidak ada model regresi yang tersedia untuk perbandingan metrik.</div>', unsafe_allow_html=True)

#     with col_reg_plot:
#         st.markdown("<h4 style='color:white;'>Plot Aktual vs Prediksi (XGBoost Regressor)</h4>", unsafe_allow_html=True)
#         if models['xgb_regressor'] is not None and 'y_pred_xgb_reg' in locals():
#             st.markdown('<div class="plot-container">', unsafe_allow_html=True)
#             fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
#             ax_scatter.scatter(y_reg_test, y_pred_xgb_reg, alpha=0.6, color='darkblue')
#             ax_scatter.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
#             ax_scatter.set_xlabel('Pendapatan Aktual (USD)', fontsize=12, color='black')
#             ax_scatter.set_ylabel('Pendapatan Prediksi (USD)', fontsize=12, color='black')
#             ax_scatter.set_title('Pendapatan Aktual vs Prediksi (XGBoost Regressor)', fontsize=14, color='black')
#             ax_scatter.ticklabel_format(style='plain', axis='x')
#             ax_scatter.ticklabel_format(style='plain', axis='y')
#             ax_scatter.tick_params(axis='x', colors='black')
#             ax_scatter.tick_params(axis='y', colors='black')
#             ax_scatter.set_facecolor('#f8f9fa')
#             plt.tight_layout()
#             st.pyplot(fig_scatter)
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown('<div class="info-box">Plot ini memvisualisasikan seberapa dekat prediksi model dengan nilai pendapatan aktual. Titik-titik yang berada di dekat garis putus-putus merah menunjukkan prediksi yang akurat.</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="info-box">XGBoost Regressor tidak tersedia untuk menampilkan plot aktual vs prediksi, atau prediksinya belum dihitung.</div>', unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown('<h3 class="sub-header">🌟 Importansi Fitur (Top 10)</h3>', unsafe_allow_html=True)
#     st.markdown('<div class="info-box">Pilih model untuk melihat fitur-fitur yang paling berpengaruh dalam prediksinya. Fitur dengan nilai importansi yang lebih tinggi dianggap lebih penting.</div>', unsafe_allow_html=True)

#     col_imp_reg, col_imp_cls = st.columns(2)

#     with col_imp_reg:
#         available_reg_models = []
#         if models['xgb_regressor'] is not None:
#             available_reg_models.append("XGBoost Regressor")
#         if models['rf_regressor'] is not None:
#             available_reg_models.append("Random Forest Regressor")

#         if available_reg_models:
#             selected_reg_model_for_imp = st.selectbox(
#                 "Pilih Model Regresi:",
#                 available_reg_models,
#                 key="select_reg_imp"
#             )
#             importance_df_reg = None
#             if selected_reg_model_for_imp == "XGBoost Regressor" and models['xgb_reg_importance'] is not None:
#                 importance_df_reg = models['xgb_reg_importance']
#             elif selected_reg_model_for_imp == "Random Forest Regressor" and models['rf_reg_importance'] is not None:
#                 importance_df_reg = models['rf_reg_importance']

#             if importance_df_reg is not None and not importance_df_reg.empty:
#                 st.markdown('<div class="plot-container">', unsafe_allow_html=True)
#                 fig_reg_imp, ax_reg_imp = plt.subplots(figsize=(8, 6))
#                 sns.barplot(x='importance', y='feature', data=importance_df_reg.head(10), ax=ax_reg_imp, palette='magma')
#                 ax_reg_imp.set_title(f'Top 10 Importansi Fitur - {selected_reg_model_for_imp}', fontsize=14, color='black')
#                 ax_reg_imp.set_xlabel('Importansi', fontsize=12, color='black')
#                 ax_reg_imp.set_ylabel('Fitur', fontsize=12, color='black')
#                 ax_reg_imp.tick_params(axis='x', colors='black')
#                 ax_reg_imp.tick_params(axis='y', colors='black')
#                 ax_reg_imp.set_facecolor('#f8f9fa')
#                 plt.tight_layout()
#                 st.pyplot(fig_reg_imp)
#                 st.markdown('</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="warning-box">Data importansi fitur untuk {selected_reg_model_for_imp} tidak ditemukan atau kosong.</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="info-box">Tidak ada model regresi yang tersedia untuk melihat importansi fitur.</div>', unsafe_allow_html=True)

#     with col_imp_cls:
#         available_cls_models = []
#         if models['xgb_classifier'] is not None:
#             available_cls_models.append("XGBoost Classifier")
#         if models['rf_classifier'] is not None:
#             available_cls_models.append("Random Forest Classifier")

#         if available_cls_models:
#             selected_cls_model_for_imp = st.selectbox(
#                 "Pilih Model Klasifikasi:",
#                 available_cls_models,
#                 key="select_cls_imp"
#             )
#             importance_df_cls = None
#             if selected_cls_model_for_imp == "XGBoost Classifier" and models['xgb_cls_importance'] is not None:
#                 importance_df_cls = models['xgb_cls_importance']
#             elif selected_cls_model_for_imp == "Random Forest Classifier" and models['rf_cls_importance'] is not None:
#                 importance_df_cls = models['rf_cls_importance']

#             if importance_df_cls is not None and not importance_df_cls.empty:
#                 st.markdown('<div class="plot-container">', unsafe_allow_html=True)
#                 fig_cls_imp, ax_cls_imp = plt.subplots(figsize=(8, 6))
#                 sns.barplot(x='importance', y='feature', data=importance_df_cls.head(10), ax=ax_cls_imp, palette='cividis')
#                 ax_cls_imp.set_title(f'Top 10 Importansi Fitur - {selected_cls_model_for_imp}', fontsize=14, color='black')
#                 ax_cls_imp.set_xlabel('Importansi', fontsize=12, color='black')
#                 ax_cls_imp.set_ylabel('Fitur', fontsize=12, color='black')
#                 ax_cls_imp.tick_params(axis='x', colors='black')
#                 ax_cls_imp.tick_params(axis='y', colors='black')
#                 ax_cls_imp.set_facecolor('#f8f9fa')
#                 plt.tight_layout()
#                 st.pyplot(fig_cls_imp)
#                 st.markdown('</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="warning-box">Data importansi fitur untuk {selected_cls_model_for_imp} tidak ditemukan atau kosong.</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="info-box">Tidak ada model klasifikasi yang tersedia untuk melihat importansi fitur.</div>', unsafe_allow_html=True)


# --- Halaman 3: Buat Prediksi ---
elif page == "🔮 Buat Prediksi":
    st.markdown('<h2 class="sub-header">🔮 Prediksi Potensi Film Baru</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Masukkan detail film baru untuk memprediksi potensi pendapatan dan kategori risiko ROI-nya.</div>', unsafe_allow_html=True)

    st.markdown("<h3 style='color:white;'>📝 Masukkan Data Film</h3>", unsafe_allow_html=True)
    with st.form("movie_prediction_form"):
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            title = st.text_input("Judul Film", "The Next Blockbuster Movie")
            release_date = st.date_input("Tanggal Rilis", datetime.now().date())
            budget_usd = st.number_input("Anggaran (USD)", min_value=1000, value=50000000, step=1000000, format="%d")
            popularity = st.slider("Skor Popularitas", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
            runtime = st.number_input("Durasi (menit)", min_value=30, value=120, step=5, format="%d")

        with col_input2:
            vote_average = st.slider("Rata-rata Suara (1-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
            vote_count = st.number_input("Jumlah Suara", min_value=0, value=5000, step=100, format="%d")
            original_language = st.selectbox("Bahasa Asli", ["en", "id", "ja", "fr", "es", "ko", "zh", "de", "hi", "other"], index=0)
            selected_genres = st.multiselect("Pilih Genre", ALL_GENRES, default=["Action", "Science Fiction"])
            genres_str = ", ".join(selected_genres)

        st.markdown("---")
        st.markdown("<h3 style='color:white;'>⚙️ Pilih Model untuk Prediksi</h3>", unsafe_allow_html=True)
        col_model_choice_reg, col_model_choice_cls = st.columns(2)
        with col_model_choice_reg:
            available_reg_pred_models = []
            if models['xgb_regressor'] is not None:
                available_reg_pred_models.append("XGBoost Regressor")
            if models['rf_regressor'] is not None:
                available_reg_pred_models.append("Random Forest Regressor")

            if available_reg_pred_models:
                reg_model_choice = st.selectbox(
                    "Model Regresi (Pendapatan):",
                    available_reg_pred_models,
                    key="pred_reg_model"
                )
            else:
                reg_model_choice = None
                st.markdown('<div class="warning-box">Tidak ada model regresi yang tersedia untuk prediksi.</div>', unsafe_allow_html=True)

        with col_model_choice_cls:
            available_cls_pred_models = []
            if models['xgb_classifier'] is not None:
                available_cls_pred_models.append("XGBoost Classifier")
            if models['rf_classifier'] is not None:
                available_cls_pred_models.append("Random Forest Classifier")

            if available_cls_pred_models:
                cls_model_choice = st.selectbox(
                    "Model Klasifikasi (Risiko ROI):",
                    available_cls_pred_models,
                    key="pred_cls_model"
                )
            else:
                cls_model_choice = None
                st.markdown('<div class="warning-box">Tidak ada model klasifikasi yang tersedia untuk prediksi.</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Lakukan Prediksi 🚀")

    if submitted:
        if reg_model_choice is None or cls_model_choice is None:
            st.markdown('<div class="error-box">Tidak dapat melakukan prediksi. Pilih model regresi dan klasifikasi yang tersedia.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Menghasilkan prediksi..."):
                input_df_raw = pd.DataFrame([{
                    'title': title,
                    'release_date': release_date.strftime("%Y-%m-%d"),
                    'budget': budget_usd,
                    'popularity': popularity,
                    'runtime': runtime,
                    'vote_average': vote_average,
                    'vote_count': vote_count,
                    'original_language': original_language,
                    'genres': genres_str,
                    'revenue': 0, # Placeholder
                    'ROI': 0, # Placeholder
                    'ROI_category': 'Dummy' # Placeholder
                }])

                processed_input_df = preprocess_data(input_df_raw.copy())
                X_input, _, _, _ = split_features_target(processed_input_df.copy(), _scaler=global_scaler)

                expected_feature_cols = X_train.columns # Ambil dari X_train yang sudah di-cache
                for col in expected_feature_cols:
                    if col not in X_input.columns:
                        X_input[col] = 0
                X_input = X_input[expected_feature_cols] # Pastikan urutan dan kelengkapan kolom

                regression_model_to_use = models['xgb_regressor'] if reg_model_choice == "XGBoost Regressor" else models['rf_regressor']
                classification_model_to_use = models['xgb_classifier'] if cls_model_choice == "XGBoost Classifier" else models['rf_classifier']

                # Ensure models are loaded and not None before predicting
                if regression_model_to_use is None:
                    st.markdown('<div class="error-box">Model regresi tidak tersedia.</div>', unsafe_allow_html=True)
                    st.stop()
                if classification_model_to_use is None:
                    st.markdown('<div class="error-box">Model klasifikasi tidak tersedia.</div>', unsafe_allow_html=True)
                    st.stop()

                predicted_revenue_usd = regression_model_to_use.predict(X_input)[0]
                predicted_risk_encoded = classification_model_to_use.predict(X_input)[0]

                predicted_risk_category = label_encoder.inverse_transform([predicted_risk_encoded])[0]

                predicted_roi = ((predicted_revenue_usd - budget_usd) / budget_usd) * 100 if budget_usd > 0 else 0

                predicted_revenue_idr = predicted_revenue_usd * st.session_state.dollar_to_rupiah_rate
                budget_idr = budget_usd * st.session_state.dollar_to_rupiah_rate

                st.markdown("<h3 class='sub-header'>✨ Hasil Prediksi Anda:</h3>", unsafe_allow_html=True)
                st.markdown(f'<div class="success-box"><strong>Judul Film:</strong> <strong>{title}</strong></div>', unsafe_allow_html=True)

                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💲 Anggaran Awal", f"USD ${budget_usd:,.2f}")
                    st.metric("🇮🇩 Anggaran Awal", f"IDR Rp {budget_idr:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_pred2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💰 Pendapatan Diprediksi (USD)", f"USD ${predicted_revenue_usd:,.2f}")
                    st.metric("💵 Pendapatan Diprediksi (IDR)", f"IDR Rp {predicted_revenue_idr:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_pred3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("📈 ROI Diprediksi", f"{predicted_roi:.2f}%")
                    # Sesuaikan emoji berdasarkan kategori ROI yang baru
                    if predicted_risk_category == "Extreme Loss":
                        risk_emoji = "🔴"
                    elif predicted_risk_category == "Significant Loss":
                        risk_emoji = "🟠"
                    elif predicted_risk_category == "Marginal Profit":
                        risk_emoji = "🔵"
                    elif predicted_risk_category == "Good Profit":
                        risk_emoji = "🟢"
                    elif predicted_risk_category == "Blockbuster/High Profit":
                        risk_emoji = "🟣"
                    else:
                        risk_emoji = "⚪" # Default jika ada kategori tidak terduga
                    st.metric("⚠️ Tingkat Risiko Diprediksi", f"{risk_emoji} {predicted_risk_category}")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("<h3 class='sub-header'>📝 Detail Input Film yang Digunakan:</h3>", unsafe_allow_html=True)
                display_df = input_df_raw.drop(columns=['revenue', 'ROI', 'ROI_category'])

                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.dataframe(display_df.style.format({
                    'budget': "${:,.2f}",
                    'popularity': "{:.1f}",
                    'runtime': "{:d} menit",
                    'vote_average': "{:.1f}",
                    'vote_count': "{:,.0f}"
                }),
                use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="info-box">*) Prediksi ini dihasilkan menggunakan model <strong>{reg_model_choice}</strong> untuk pendapatan dan <strong>{cls_model_choice}</strong> untuk tingkat risiko ROI.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="info-box">Fitur riwayat prediksi tidak diaktifkan pada dashboard ini.</div>', unsafe_allow_html=True)