import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import warnings

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="HR Attrition (Model Based)", layout="wide")
warnings.filterwarnings("ignore")

# --- 2. FITUR TARGET (WAJIB SESUAI MODEL) ---
TARGET_FEATURES = [
    'Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement',
    'PerformanceRating', 'mean_work_time', 
    'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Human Resources', 'Department_Research & Development', 'Department_Sales',
    'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5',
    'EducationField_Human Resources', 'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
    'Gender_Female', 'Gender_Male',
    'JobLevel_1', 'JobLevel_2', 'JobLevel_3', 'JobLevel_4', 'JobLevel_5',
    'JobRole_Healthcare Representative', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'StockOptionLevel_0', 'StockOptionLevel_1', 'StockOptionLevel_2', 'StockOptionLevel_3',
    'EnvironmentSatisfaction_1', 'EnvironmentSatisfaction_2', 'EnvironmentSatisfaction_3', 'EnvironmentSatisfaction_4'
]

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'model_attrition.pkl')
    if os.path.exists(file_path):
        try:
            return joblib.load(file_path)
        except:
            return None
    return None

model = load_model()

# --- 4. DATA PROCESSING (PURE PREPROCESSING) ---
def read_csv_safe(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    return None

def process_data(df_gen, df_emp, df_mgr, df_in, df_out):
    # 1. Merge Data
    df = df_gen.merge(df_emp, on='EmployeeID', how='left').merge(df_mgr, on='EmployeeID', how='left')
    
    # 2. Hitung Jam Kerja (Mean Work Time)
    df_in.set_index(df_in.columns[0], inplace=True)
    df_out.set_index(df_out.columns[0], inplace=True)
    
    in_time_dt = df_in.apply(pd.to_datetime, errors='coerce')
    out_time_dt = df_out.apply(pd.to_datetime, errors='coerce')
    
    # Rata-rata jam kerja per hari
    mean_hours = (out_time_dt - in_time_dt).mean(axis=1).dt.total_seconds() / 3600
    df = df.merge(mean_hours.rename('mean_work_time'), left_on='EmployeeID', right_index=True, how='left')
    
    # 3. Cleaning Data
    drop_cols = ['Attrition', 'EmployeeCount', 'Over18', 'StandardHours']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # 4. Imputasi (Isi Data Kosong)
    # Penting agar model tidak error karena NaN
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    for c in num_cols: df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    for c in cat_cols:
        if len(df_clean[c].mode()) > 0: df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])
        else: df_clean[c] = df_clean[c].fillna('Unknown')
        
    return df, df_clean

# --- 5. UI UTAMA ---
st.title("📊 HR Attrition Dashboard (Model Only)")

if model is None:
    st.error("❌ Model 'model_attrition.pkl' tidak ditemukan di folder ini.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("1. Upload Data")
f_gen = st.sidebar.file_uploader("General Data", type='csv')
f_emp = st.sidebar.file_uploader("Employee Survey", type='csv')
f_mgr = st.sidebar.file_uploader("Manager Survey", type='csv')
f_in  = st.sidebar.file_uploader("In Time", type='csv')
f_out = st.sidebar.file_uploader("Out Time", type='csv')

st.sidebar.divider()
st.sidebar.header("2. Kalibrasi Model")
st.sidebar.markdown("Geser slider jika prediksi resign 0.")
# SLIDER: Solusi untuk masalah "Prediksi 0"
threshold = st.sidebar.slider(
    "Batas Sensitivitas (Threshold)", 
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Default ML adalah 0.5. Jika hasil 0, coba turunkan ke 0.3 atau 0.2."
)

# --- EXECUTION ---
if f_gen and f_emp and f_mgr and f_in and f_out:
    if st.button("🚀 PREDIKSI SEKARANG"):
        with st.spinner("Sedang menghitung prediksi model..."):
            try:
                # A. BACA FILE
                df_gen = read_csv_safe(f_gen)
                df_emp = read_csv_safe(f_emp)
                df_mgr = read_csv_safe(f_mgr)
                df_in = read_csv_safe(f_in)
                df_out = read_csv_safe(f_out)
                
                # B. PROSES DATA
                df_full, df_ready = process_data(df_gen, df_emp, df_mgr, df_in, df_out)
                
                # C. FORMAT KE BENTUK MODEL (59 Fitur)
                cols_to_str = ['Education', 'JobLevel', 'StockOptionLevel', 'EnvironmentSatisfaction']
                for col in df_ready.columns:
                    if df_ready[col].dtype == 'object' or col in cols_to_str:
                         df_ready[col] = df_ready[col].astype(str).str.replace('.0', '')
                
                # One-Hot & Reindex
                X_final = pd.get_dummies(df_ready).reindex(columns=TARGET_FEATURES, fill_value=0)
                
                # D. PREDIKSI (MENGGUNAKAN SLIDER)
                # Ambil probabilitas murni dari model (angka 0.0 - 1.0)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_final)[:, 1]
                else:
                    # Fallback jika model tidak punya proba (jarang terjadi di Random Forest)
                    st.warning("Model tidak mendukung probabilitas, menggunakan output kelas langsung.")
                    probs = model.predict(X_final).astype(float)
                
                # Tentukan Resign/Tidak berdasarkan Slider User
                preds = (probs >= threshold).astype(int)
                
                # E. HASIL OUTPUT
                output = df_full[['EmployeeID', 'Department', 'JobRole']].copy()
                output['Status'] = ['Resign 🔴' if x==1 else 'Stay 🟢' for x in preds]
                output['Probability'] = probs
                
                # Kategori Risiko (Murni dari Probabilitas Model)
                def get_risk_cat(p):
                    if p >= 0.6: return 'High Risk 🔴'
                    elif p >= 0.3: return 'Medium Risk 🟡'
                    else: return 'Low Risk 🟢'
                output['Risk Category'] = output['Probability'].apply(get_risk_cat)
                
                # --- F. TAMPILAN DASHBOARD ---
                st.success("Selesai! Hasil di bawah murni berasal dari Model.")
                
                # Hitung Statistik
                total_karyawan = len(output)
                total_resign = preds.sum()
                pct_attrition = (total_resign / total_karyawan) * 100
                
                # Tampilkan Kartu Metrik
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Karyawan", f"{total_karyawan:,}")
                col2.metric("Prediksi Resign", f"{total_resign:,}", help="Jumlah karyawan yang diprediksi keluar oleh model")
                col3.metric("Persentase Attrition", f"{pct_attrition:.1f}%", help="Persentase karyawan yang diprediksi resign")
                
                # Pesan Bantuan jika Hasil Masih 0
                if total_resign == 0:
                    st.info(f"💡 Prediksi masih 0? Coba geser **Batas Sensitivitas** di sidebar ke kiri (misal: 0.2 atau 0.3).")
                    # Tampilkan probabilitas tertinggi sebagai petunjuk
                    max_prob = probs.max()
                    st.write(f"ℹ️ *Info Debug: Probabilitas resign tertinggi yang ditemukan model adalah {max_prob:.1%}.*")
                
                # Tabel Detail
                st.subheader("📋 Detail Hasil Model")
                st.dataframe(output[['EmployeeID', 'Status', 'Risk Category', 'Probability', 'Department', 'JobRole']].sort_values('Probability', ascending=False))
                
                # Download
                st.download_button("Download CSV", output.to_csv(index=False).encode('utf-8'), "hasil_model_attrition.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Terjadi Error: {e}")
                import traceback
                st.text(traceback.format_exc())
else:
    st.info("Silakan upload semua file data di sidebar.")