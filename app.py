import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="HR Attrition Pro (AI Dashboard)", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings("ignore")

# Custom CSS for better aesthetics - Adaptive to theme
st.markdown("""
<style>
    .stMetric {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .priority-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
            return joblib.load(file_path), None
        except Exception as e:
            return None, f"Error setup environment: {str(e)}"
    return None, f"File {file_path} not found."

model, model_error = load_model()

# --- 4. DATA PROCESSING HELPERS ---
def read_csv_safe(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    return None

def preprocess_classic(df_gen, df_emp, df_mgr, df_in, df_out):
    # Merge Data
    df = df_gen.merge(df_emp, on='EmployeeID', how='left').merge(df_mgr, on='EmployeeID', how='left')
    
    # Calculate Mean Work Time
    df_in.set_index(df_in.columns[0], inplace=True)
    df_out.set_index(df_out.columns[0], inplace=True)
    in_time_dt = df_in.apply(pd.to_datetime, errors='coerce')
    out_time_dt = df_out.apply(pd.to_datetime, errors='coerce')
    mean_hours = (out_time_dt - in_time_dt).mean(axis=1).dt.total_seconds() / 3600
    df = df.merge(mean_hours.rename('mean_work_time'), left_on='EmployeeID', right_index=True, how='left')
    return df

def preprocess_simplified(df_single):
    # Mapping columns from single file to what the model expects
    # In hr_analytics_for_modeling.csv, 'AvgWorkingHours' is already calculated
    if 'AvgWorkingHours' in df_single.columns:
        df_single['mean_work_time'] = df_single['AvgWorkingHours']
    return df_single

def finalize_for_model(df_raw):
    # 1. Cleaning
    drop_cols = ['Attrition', 'EmployeeCount', 'Over18', 'StandardHours']
    df_clean = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns], errors='ignore')
    
    # 2. Imputation
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    for c in num_cols: df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    for c in cat_cols:
        if len(df_clean[c].mode()) > 0: df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])
        else: df_clean[c] = df_clean[c].fillna('Unknown')
    
    # 3. Format Strings for Categories
    cols_to_str = ['Education', 'JobLevel', 'StockOptionLevel', 'EnvironmentSatisfaction']
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' or col in cols_to_str:
             df_clean[col] = df_clean[col].astype(str).str.replace('.0', '')
    
    # 4. One-Hot & Reindex
    X_final = pd.get_dummies(df_clean).reindex(columns=TARGET_FEATURES, fill_value=0)
    return X_final

# --- 5. UI COMPONENTS ---
st.title("📊 Employee Attrition Analytics Dashboard")

if model is None:
    st.error(f"❌ {model_error}")
    st.stop()

# SIDEBAR
st.sidebar.header("Dashboard Settings")
mode = st.sidebar.radio("Upload Mode", ["Simplified (1 CSV)", "Classic (5 CSVs)"])

if mode == "Simplified (1 CSV)":
    st.sidebar.subheader("1. Upload Your Data")
    f_single = st.sidebar.file_uploader("Upload 'hr_analytics_for_modeling.csv'", type='csv')
else:
    st.sidebar.subheader("1. Upload Required Files")
    f_gen = st.sidebar.file_uploader("General Data", type='csv')
    f_emp = st.sidebar.file_uploader("Employee Survey", type='csv')
    f_mgr = st.sidebar.file_uploader("Manager Survey", type='csv')
    f_in  = st.sidebar.file_uploader("In Time Logs", type='csv')
    f_out = st.sidebar.file_uploader("Out Time Logs", type='csv')

st.sidebar.divider()
threshold = 0.32

# --- EXECUTION FLOW ---
data_ready = False
df_full = None

if mode == "Simplified (1 CSV)" and f_single:
    df_raw = read_csv_safe(f_single)
    df_full = preprocess_simplified(df_raw)
    data_ready = True
elif mode == "Classic (5 CSVs)" and all([f_gen, f_emp, f_mgr, f_in, f_out]):
    df_gen_raw = read_csv_safe(f_gen)
    df_emp_raw = read_csv_safe(f_emp)
    df_mgr_raw = read_csv_safe(f_mgr)
    df_in_raw = read_csv_safe(f_in)
    df_out_raw = read_csv_safe(f_out)
    df_full = preprocess_classic(df_gen_raw, df_emp_raw, df_mgr_raw, df_in_raw, df_out_raw)
    data_ready = True

if data_ready:
    if st.button("🚀 GO PREDICTTT!"):
        with st.spinner("Processing Model Predictions..."):
            try:
                # Prediction Logic
                X_final = finalize_for_model(df_full)
                probs = model.predict_proba(X_final)[:, 1]
                preds = (probs >= 0.32).astype(int)
                
                # Build Result DF
                results = df_full[['EmployeeID', 'Department', 'JobRole']].copy()
                results['Probability'] = probs
                results['Status'] = ['Resign 🔴' if x==1 else 'Stay 🟢' for x in preds]
                
                def get_risk_cat(p):
                    if p >= 0.6: return 'Critical High 🚨'
                    elif p >= 0.32: return 'Medium Risk 🟡'
                    else: return 'Low Risk 🟢'
                results['Risk Category'] = results['Probability'].apply(get_risk_cat)
                
                # --- VISUALIZATION SECTION ---
                st.success("Analysis Complete!")
                
                # 1. Metrics Header
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Workforce", f"{len(results):,}")
                m2.metric("Predicted Attrition", f"{preds.sum():,}", delta=f"{(preds.sum()/len(results)*100):.1f}% Rate")
                m3.metric("Critical High Risk", f"{len(results[results['Probability'] >= 0.6]):,}")
                
                st.divider()
                
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("Risk Distribution")
                    risk_counts = results['Risk Category'].value_counts()
                    fig_pie = px.pie(risk_counts, values=risk_counts.values, names=risk_counts.index, 
                                   color=risk_counts.index,
                                   color_discrete_map={'Critical High 🚨': '#ef5350', 'Medium Risk 🟡': '#ffca28', 'Low Risk 🟢': '#66bb6a'},
                                   hole=.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_right:
                    st.subheader("Attrition by Department")
                    dept_attr = results.groupby('Department')['Probability'].mean().sort_values(ascending=False).reset_index()
                    fig_bar = px.bar(dept_attr, x='Department', y='Probability', 
                                   title="Avg Risk per Dept", color='Probability',
                                   color_continuous_scale='Reds')
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.divider()
                
                # Feature Importance~
                st.subheader("Key Drivers of Attrition")
                if hasattr(model, 'steps'):
                    rf = model.steps[-1][1]
                    importances = pd.Series(rf.feature_importances_, index=TARGET_FEATURES).sort_values(ascending=False).head(8)
                    fig_imp = px.bar(importances, orientation='h', labels={'index': 'Feature', 'value': 'Importance Score'},
                                   title="Top 8 Global Factors", color=importances.values, color_continuous_scale='Viridis')
                    st.plotly_chart(fig_imp, use_container_width=True)

                # Priority Intervention List
                st.subheader("⚠️ Priority Intervention List")
                st.info("Employees listed here have over 30% probability of resignation. Immediate HR review recommended.")
                priority_list = results[results['Probability'] >= 0.3].sort_values('Probability', ascending=False)
                st.dataframe(priority_list[['EmployeeID', 'Department', 'JobRole', 'Probability', 'Risk Category']], use_container_width=True)
                
                # Full Data Table
                with st.expander("Show Full Employee List"):
                    st.dataframe(results.sort_values('Probability', ascending=False))
                
                # Downloada
                st.download_button("📥 Export Results to CSV", results.to_csv(index=False).encode('utf-8'), "hr_attrition_report.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                import traceback
                st.text(traceback.format_exc())
else:
    # Landing Page Info
    st.info("👋 Welcome! Please upload your data in the sidebar to begin.")
    st.markdown("""
    ### How it works:
    - **Simplified Mode**: Upload your model-ready single CSV file.
    - **Classic Mode**: Upload the raw HR datasets (General, Surveys, Attendance).
    - **AI Model**: We use a Random Forest algorithm (SMOTE-balanced) to predict attrition.
    """)