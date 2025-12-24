# ---------------------- Core Web Framework ----------------------
import streamlit as st

# ---------------------- Data Handling ----------------------
import pandas as pd
import numpy as np
import io

# ---------------------- Data Storage ----------------------
import sqlite3
from pathlib import Path

# ---------------------- Visualization ----------------------
import plotly.express as px

# ---------------------- Machine Learning ----------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re

# ---------------------- Reporting ----------------------
from reportlab.lib.pagesizes import letter

from io import BytesIO
from reportlab.lib import colors
import tempfile
import os

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    Image,
    TableStyle
)
from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# ---------------------- Email ----------------------
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.message import EmailMessage
import ssl
from reportlab.lib.utils import ImageReader

# ---------------------- Utilities ----------------------
from datetime import datetime
import warnings
import plotly.io as pio
pio.orca.config.executable = '/usr/bin/orca'
pio.orca.config.save()

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ---------------------- App Configuration ----------------------

# App metadata
APP_TITLE = "🩺 Global Health ETL + Analytics"
APP_LAYOUT = "wide"

# Temporary storage
CLEANED_CSV_NAME = "cleaned_global_health_data.csv"
SQLITE_DB_PATH = "cleaned_health_data.db"

# Email (loaded from Streamlit secrets)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = st.secrets.get("SENDER_EMAIL", "")
SENDER_PASSWORD = st.secrets.get("SENDER_PW", "")
DEFAULT_RECIPIENTS = (
    st.secrets.get("RECIPIENTS", "").split(",")
    if st.secrets.get("RECIPIENTS")
    else []
)

# Model configuration
model_configs= {
    "n_estimators": 300,
    "max_depth": 3,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1
}

# ======================================================================
# 1. ETL PIPELINE WITH ML PREDICTION CAPABILITIES
# ======================================================================

def clean_health_dataset(
    csv_path: str,
    output_path: str = 'cleaned_global_health_data.csv',
    sqlite_db_path: str = None,
    save_to_sqlite: bool = False
):
    """
    Specialized ETL pipeline for the Global Health Dataset.
    Handles issues specific to this dataset:
    • Special characters in country names (It@lĄ, T?u?r?k?e?y?, etc.)
    • Mixed numeric formatting (quotes, commas as decimals)
    • Disease name inconsistencies
    • Missing values in age groups and demographic columns
    • Inconsistent vaccine/treatment availability labels
    """
    print("=" * 80)
    print("GLOBAL HEALTH DATASET CLEANING PIPELINE")
    print("=" * 80)
    # ---------- 1. EXTRACTION ----------
    print("\n1. EXTRACTING DATA...")
    # Careful handling of mixed types and encoding fallback
    df = None
    encodings_to_try = ['utf-8', 'latin1', 'cp1252']
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                csv_path,
                encoding=enc,
                low_memory=False,
                na_values=['', 'NaN', 'NA', 'NULL', 'None', 'nan', 'N/A', 'n/a', '~none~', '?', '-'],
                keep_default_na=True
            )
            print(f"Successfully read with encoding: {enc}")
            break
        except UnicodeDecodeError:
            print(f"Failed to read with encoding '{enc}'. Trying next...")
            continue
    if df is None:
        raise ValueError("Could not read the CSV file with any common encoding (UTF-8, Latin-1, CP1252). "
                         "Try opening the file in a text editor to check its encoding, or specify a custom one.")
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows (original):")
    print(df.head())
    # ---------- 2. TRANSFORMATION ----------
    print("\n" + "=" * 80)
    print("2. TRANSFORMING DATA...")
    print("=" * 80)
    # 2a. Clean Country Names
    print("\n2a. Cleaning country names...")
    def clean_country_name(name):
        if pd.isna(name):
            return "Unknown"
        # Remove special characters but preserve spaces and actual letters
        # Keep common punctuation like hyphens and apostrophes
        name = str(name)
        # Specific known corrections
        corrections = {
            'It@lĄ': 'Italy',
            'T?u?r?k?e?y?': 'Turkey',
            'G%rmany': 'Germany',
            'Can@da': 'Canada',
            'Mex!co': 'Mexico',
            '?r?zil': 'Brazil',
            'Ind!a': 'India'  # Just in case
        }
        # Apply known corrections
        if name in corrections:
            return corrections[name]
        # Remove problematic characters but keep normal letters, spaces, hyphens
        cleaned = re.sub(r'[^a-zA-Z\s\-\.\']', '', name)
        cleaned = cleaned.strip()
        # Capitalize properly
        if cleaned:
            parts = cleaned.split()
            cleaned = ' '.join([p.capitalize() for p in parts])
        return cleaned if cleaned else "Unknown"
    df['Country'] = df['Country'].apply(clean_country_name)
    print(f"Unique countries after cleaning: {df['Country'].nunique()}")
    print(f"Sample countries: {df['Country'].unique()[:10]}")
    # 2b. Clean Disease Names
    print("\n2b. Cleaning disease names...")
    def clean_disease_name(name):
        if pd.isna(name):
            return "Unknown"
        name = str(name).strip()
        # Remove extra spaces and weird characters
        name = re.sub(r'[^\w\s\-\(\)\']', '', name)
        # Fix specific known issues
        name = name.replace('A!DS', 'AIDS')
        name = name.replace('Influen&za', 'Influenza')
        name = name.replace('Pol!o', 'Polio')
        # Remove extra whitespace
        name = ' '.join(name.split())
        # Capitalize first letter of each word (for proper nouns)
        # Preserve acronyms like COVID-19, HIV/AIDS
        if not any(x in name.upper() for x in ['COVID', 'HIV', 'AIDS', 'SARS', 'MERS']):
            name = name.title()
        return name
    df['Disease Name'] = df['Disease Name'].apply(clean_disease_name)
    print(f"Unique diseases: {df['Disease Name'].nunique()}")
    # 2c. Clean Year column
    print("\n2c. Cleaning year data...")
    # Convert Year to integer, handle float values like 2013.00
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Year'] = df['Year'].fillna(df['Year'].median())
    df['Year'] = df['Year'].astype(int)
    # Filter to reasonable years (1900-2100)
    df = df[(df['Year'] >= 1900) & (df['Year'] <= 2100)]
    # 2d. Clean numeric columns with special formatting
    print("\n2d. Cleaning numeric columns...")
    # List of columns that should be numeric
    numeric_columns = [
        'Country_pop', 'Incidence Rate mn (%)', 'Prevalence rate (%)',
        'Mortality Rate per 100 people (%)', 'Population affected',
        'Pop_affected(Male)', 'Pop_affected(Female)', 'Ages 0-18 (%)',
        'Ages 19-35 (%)', 'Ages 36-60 (%)', 'Ages 61+ (%)',
        'Pop_affected_U (%)', 'Pop_affected_R (%)', 'Healthcare Access (%)',
        'Doctors per 1000', 'Hospital Beds per 1000', 'Recovery Rate (%)',
        'DALYs', 'Improvement in 5 Years (%)', 'Average Annual Treatment Cost (USD)',
        'Composite Health Index (CHI)', 'Per Capita Income (USD)',
        'Education Index', 'Urbanization Rate (%)'
    ]
    def clean_numeric_value(value):
        if pd.isna(value):
            return np.nan
        value_str = str(value)
        # Remove quotes and commas used as decimal separators
        value_str = value_str.replace("'", "").replace(",", ".")
        # Remove any non-numeric characters except decimal point, minus sign
        value_str = re.sub(r'[^\d\.\-]', '', value_str)
        # Handle empty strings
        if value_str == '' or value_str == '.':
            return np.nan
        try:
            return float(value_str)
        except:
            return np.nan
    for col in numeric_columns:
        if col in df.columns:
            print(f" Cleaning {col}...")
            df[col] = df[col].apply(clean_numeric_value)
    # 2e. Clean categorical columns
    print("\n2e. Cleaning categorical columns...")
    # Clean Treatment Type
    if 'Treatment type' in df.columns:
        df['Treatment type'] = df['Treatment type'].fillna('Unknown')
        df['Treatment type'] = df['Treatment type'].str.capitalize()
    # Clean Vaccine/Treatment Availability
    if 'Availability of Vaccines/Treatment' in df.columns:
        availability_map = {
            'High': 'High',
            'High ': 'High',
            'high': 'High',
            'Medium': 'Medium',
            'medium': 'Medium',
            'Low': 'Low',
            'low': 'Low',
            'None': 'None',
            'none': 'None',
            '~none~': 'None',
            'NONE': 'None',
            'None ': 'None'
        }
        def clean_availability(val):
            if pd.isna(val):
                return 'Unknown'
            val_str = str(val).strip()
            return availability_map.get(val_str, 'Medium')  # Default to Medium if unknown
        df['Availability of Vaccines/Treatment'] = df['Availability of Vaccines/Treatment'].apply(clean_availability)
    # 2f. Handle missing values
    print("\n2f. Handling missing values...")
    # Fill missing population with country-year averages
    if 'Country_pop' in df.columns:
        country_year_avg = df.groupby(['Country', 'Year'])['Country_pop'].transform('median')
        df['Country_pop'] = df['Country_pop'].fillna(country_year_avg)
        df['Country_pop'] = df['Country_pop'].fillna(df['Country_pop'].median())
        df['Country_pop'] = df['Country_pop'].astype(int)
    # Fill missing demographic percentages (should sum to ~100% across age groups)
    age_cols = ['Ages 0-18 (%)', 'Ages 19-35 (%)', 'Ages 36-60 (%)', 'Ages 61+ (%)']
    for col in age_cols:
        if col in df.columns:
            df[col] = df[col].fillna(25)  # Distribute evenly if unknown
    # Fill missing rates with disease-country averages
    rate_cols = ['Incidence Rate mn (%)', 'Prevalence rate (%)', 'Mortality Rate per 100 people (%)']
    for col in rate_cols:
        if col in df.columns:
            disease_country_avg = df.groupby(['Disease Name', 'Country'])[col].transform('median')
            df[col] = df[col].fillna(disease_country_avg)
            df[col] = df[col].fillna(df[col].median())
    # 2g. Create derived columns
    print("\n2g. Creating derived columns...")
    # Population coverage percentage
    if all(col in df.columns for col in ['Population affected', 'Country_pop']):
        df['Population Coverage (%)'] = (df['Population affected'] / df['Country_pop'] * 100).round(2)
        # Cap at 100%
        df['Population Coverage (%)'] = df['Population Coverage (%)'].clip(upper=100)
    # Male/Female ratio
    if all(col in df.columns for col in ['Pop_affected(Male)', 'Pop_affected(Female)']):
        df['Gender Ratio (M:F)'] = (df['Pop_affected(Male)'] / df['Pop_affected(Female)']).round(2)
        df['Gender Ratio (M:F)'] = df['Gender Ratio (M:F)'].replace([np.inf, -np.inf], np.nan)
        df['Gender Ratio (M:F)'] = df['Gender Ratio (M:F)'].clip(lower=0.1, upper=10)
    # Urban/Rural ratio
    if all(col in df.columns for col in ['Pop_affected_U (%)', 'Pop_affected_R (%)']):
        df['Urban_Rural_Ratio'] = (df['Pop_affected_U (%)'] / df['Pop_affected_R (%)']).round(2)
        df['Urban_Rural_Ratio'] = df['Urban_Rural_Ratio'].replace([np.inf, -np.inf], np.nan)
        df['Urban_Rural_Ratio'] = df['Urban_Rural_Ratio'].clip(lower=0.1, upper=10)
    # Disease severity score (composite metric)
    if all(col in df.columns for col in ['Mortality Rate per 100 people (%)', 'DALYs']):
        df['Severity Score'] = (
            df['Mortality Rate per 100 people (%)'].fillna(0) * 0.7 +
            np.log1p(df['DALYs'].fillna(0)) * 0.3
        ).round(2)
    # 2h. Remove outliers from key metrics
    print("\n2h. Removing extreme outliers...")
    outlier_cols = [
        'Average Annual Treatment Cost (USD)',
        'Per Capita Income (USD)',
        'DALYs',
        'Country_pop'
    ]
    for col in outlier_cols:
        if col in df.columns:
            # Use IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            # Cap outliers instead of removing
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if n_outliers > 0:
                print(f" {col}: Capped {n_outliers} outliers")
    # 2i. Standardize column names
    print("\n2i. Standardizing column names...")
    column_rename = {
        'Country_pop': 'Country_Population',
        'Incidence Rate mn (%)': 'Incidence_Rate_per_million',
        'Prevalence rate (%)': 'Prevalence_Rate',
        'Mortality Rate per 100 people (%)': 'Mortality_Rate_per_100',
        'Population affected': 'Population_Affected',
        'Pop_affected(Male)': 'Affected_Male',
        'Pop_affected(Female)': 'Affected_Female',
        'Ages 0-18 (%)': 'Age_0_18_Pct',
        'Ages 19-35 (%)': 'Age_19_35_Pct',
        'Ages 36-60 (%)': 'Age_36_60_Pct',
        'Ages 61+ (%)': 'Age_61_Plus_Pct',
        'Pop_affected_U (%)': 'Urban_Population_Pct',
        'Pop_affected_R (%)': 'Rural_Population_Pct',
        'Healthcare Access (%)': 'Healthcare_Access_Pct',
        'Doctors per 1000': 'Doctors_per_1000',
        'Hospital Beds per 1000': 'Hospital_Beds_per_1000',
        'Treatment type': 'Treatment_Type',
        'Recovery Rate (%)': 'Recovery_Rate',
        'DALYs': 'DALYs',
        'Improvement in 5 Years (%)': 'Improvement_5_Years',
        'Average Annual Treatment Cost (USD)': 'Avg_Treatment_Cost_USD',
        'Availability of Vaccines/Treatment': 'Vaccine_Treatment_Availability',
        'Composite Health Index (CHI)': 'Health_Index',
        'Per Capita Income (USD)': 'Per_Capita_Income_USD',
        'Education Index': 'Education_Index',
        'Urbanization Rate (%)': 'Urbanization_Rate'
    }
    df = df.rename(columns={k: v for k, v in column_rename.items() if k in df.columns})
    

    # Final NaN sweep
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['Mortality_Rate_per_100', 'Prevalence_Rate', 'Incidence_Rate_per_million']:
            df[col] = df[col].fillna(0)
        else:
            median = df[col].median()
            df[col] = df[col].fillna(median if not pd.isna(median) else 0)

    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].fillna('Unknown')

    # Recalculate derived
    if all(c in df.columns for c in ['Population_Affected', 'Country_Population']):
        df['Population Coverage (%)'] = np.clip((df['Population_Affected'] / df['Country_Population'] * 100).round(2), 0, 100)

    if all(c in df.columns for c in ['Affected_Male', 'Affected_Female']):
        df['Gender Ratio (M:F)'] = np.clip((df['Affected_Male'] / df['Affected_Female'].clip(lower=0.1)).round(2), 0.1, 10)

    if all(c in df.columns for c in ['Pop_affected_U (%)', 'Pop_affected_R (%)']):  # Use original names if not renamed
        df['Urban_Rural_Ratio'] = np.clip((df['Pop_affected_U (%)'] / df['Pop_affected_R (%)'].clip(lower=0.1)).round(2), 0.1, 10)

    if all(c in df.columns for c in ['Mortality Rate per 100 people (%)', 'DALYs']):
        df['Severity Score'] = (df['Mortality Rate per 100 people (%)'].fillna(0) * 0.7 + np.log1p(df['DALYs'].fillna(0)) * 0.3).round(2)

    # 2j. Sort, dedupe, and reset index
    print("\n2j. Sorting, deduplicating, and resetting index...")
    # Drop exact duplicates before sorting
    initial_rows = len(df)
    df = df.drop_duplicates()  # Drops rows identical across ALL columns
    deduped_rows = len(df)
    if initial_rows > deduped_rows:
        print(f" ✓ Removed {initial_rows - deduped_rows:,} duplicate rows ({((initial_rows - deduped_rows) / initial_rows * 100):.1f}% reduction)")
    else:
        print(" ✓ No duplicates found")

    # Optional: Drop duplicates on key columns if you want uniqueness per Country-Year-Disease
    # df = df.drop_duplicates(subset=['Country', 'Year', 'Disease Name'])

    df = df.sort_values(['Country', 'Year', 'Disease Name'])
    df = df.reset_index(drop=True)
    df['Record_ID'] = df.index + 1
    # ---------- 3. LOAD ----------
    print("\n" + "=" * 80)
    print("3. LOADING CLEANED DATA...")
    print("=" * 80)
    # Save to CSV
    csv_output_path = Path(output_path)
    df.to_csv(csv_output_path, index=False, encoding='utf-8')
    print(f"\n✓ Cleaned data saved to CSV: {csv_output_path}")
 
# ---------------------------------------------ANALYTICS-----------------------------------------

# Top deadliest Disease
def top_deadliest_diseases(cleaned_df, top_n=10):
    df = (
        cleaned_df
        .groupby("Disease Name", as_index=False)
        .agg(Avg_Mortality_Rate=("Mortality_Rate_per_100", "mean"))
        .sort_values("Avg_Mortality_Rate", ascending=False)
        .head(top_n)
    )
    return df

# AVG Treatment per Disease
def avg_treatment_cost_by_disease(cleaned_df, top_n=10):
    df = (
        cleaned_df
        .groupby("Disease Name", as_index=False)
        .agg(Avg_Treatment_Cost=("Avg_Treatment_Cost_USD", "mean"))
        .sort_values("Avg_Treatment_Cost", ascending=False)
        .head(top_n)
    )
    return df

# Mortality trend over time
def mortality_trend_over_time(cleaned_df):
    """
    Analyze mortality rate trend over time.
    """
    df = (
        cleaned_df
        .groupby("Year", as_index=False)
        .agg(Avg_Mortality_Rate=("Mortality_Rate_per_100", "mean"))
        .sort_values("Year")
    )
    return df

# Health care access over Mortality rate
def healthcare_access_vs_mortality(cleaned_df):
    """
    Relationship between healthcare access and mortality rate.
    """
    df = cleaned_df[[
        "Healthcare_Access_Pct",
        "Mortality_Rate_per_100",
        "Disease Name"
    ]].dropna()

    return df

# Gender impact by disease
def prepare_gender_impact_df(df):
    gender_df = (
        df.groupby("Disease Name", as_index=False)
        .agg(
            Total_Male=("Affected_Male", "sum"),
            Total_Female=("Affected_Female", "sum")
        )
        .melt(
            id_vars="Disease Name",
            value_vars=["Total_Male", "Total_Female"],
            var_name="Gender",
            value_name="Affected Population"
        )
    )
    return gender_df


# Disease health care analysis
def disease_healthcare_burden_analysis(cleaned_df):
    """
    Compare disease burden with healthcare access and urbanization.
    """
    df = cleaned_df.groupby("Disease Name", as_index=False).agg(
        Avg_Affected_Population=("Population_Affected", "mean"),
        Avg_Healthcare_Access=("Healthcare_Access_Pct", "mean"),
        Avg_Urban_Pct=("Urban_Population_Pct", "mean"),
        Avg_Rural_Pct=("Rural_Population_Pct", "mean")
    )

    df = df.sort_values("Avg_Affected_Population", ascending=False).head(50)
    return df

def mortality_correlation_analysis(cleaned_df):
    corr_cols = [
        "Mortality_Rate_per_100",
        "Incidence_Rate_per_million",
        "Prevalence_Rate",
        "Population_Affected",
        "DALYs",
        "Severity Score",
        "Improvement_5_Years",
        "Healthcare_Access_Pct",
        "Doctors_per_1000",
        "Hospital_Beds_per_1000",
        "Population Coverage (%)",
        "Health_Index",
        "Per_Capita_Income_USD",
        "Education_Index",
        "Urban_Population_Pct",
        "Rural_Population_Pct",
        "Urbanization_Rate"
    ]

    corr_df = cleaned_df[[c for c in corr_cols if c in cleaned_df.columns]]

    corr_matrix = corr_df.corr()

    # Keep only correlations with mortality
    corr_matrix = corr_matrix.loc[["Mortality_Rate_per_100"]]

    return corr_matrix

# ----------------------------VISUALIZATION----------------------------------------------
def plot_top_deadliest_diseases(df):
    fig = px.bar(
        df,
        x="Disease Name",
        y="Avg_Mortality_Rate",
        title="Top Deadliest Diseases (Average Mortality Rate)",
        labels={"Avg_Mortality_Rate": "Avg Mortality Rate"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_avg_treatment_cost(df):
    fig = px.bar(
        df,
        x="Disease Name",
        y="Avg_Treatment_Cost",
        title="Average Annual Treatment Cost by Disease (USD)",
        labels={"Avg_Treatment_Cost": "Avg Treatment Cost (USD)"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_mortality_trend(df):
    fig = px.line(
        df,
        x="Year",
        y="Avg_Mortality_Rate",
        title="Global Mortality Rate Trend Over Time",
        markers=True,
        labels={"Avg_Mortality_Rate": "Average Mortality Rate"}
    )
    return fig

def plot_healthcare_vs_mortality(df):
    fig = px.scatter(
        df,
        x="Healthcare_Access_Pct",
        y="Mortality_Rate_per_100",
        color="Disease Name",
        title="Healthcare Access vs Mortality Rate",
        labels={
            "Healthcare_Access_Pct": "Healthcare Access (%)",
            "Mortality_Rate_per_100": "Mortality Rate"
        },
        opacity=0.7
    )
    return fig

def plot_gender_impact(df):
    fig = px.bar(
        df,
        x="Disease Name",
        y="Affected Population",
        color="Gender",
        title="Gender Impact by Disease",
        barmode="group"
    )
    return fig

def plot_disease_healthcare_bubble(df):
    fig = px.scatter(
        df,
        x="Avg_Healthcare_Access",
        y="Avg_Affected_Population",
        size="Avg_Urban_Pct",
        color="Disease Name",
        title="Healthcare Access vs Affected Population (Bubble Size = Urban %)",
        labels={
            "Avg_Healthcare_Access": "Avg Healthcare Access (%)",
            "Avg_Affected_Population": "Avg Affected Population"
        },
        hover_data=["Avg_Rural_Pct"]
    )
    return fig


def plot_mortality_correlation(corr_df):
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Key Factors Associated with Mortality Rate (Correlation Analysis)"
    )
    fig.update_xaxes(side="bottom")
    fig.update_traces(xgap=2, ygap=2)
    return fig

# -------------------------------------Mortality prediction model---------------------------------------
# ---------------------------------------------
# ML FEATURE PREPARATION
# ---------------------------------------------
def prepare_ml_dataset(cleaned_df):
    """
    Select and prepare features for mortality prediction.
    """
    feature_cols = [
        "Incidence_Rate_per_million",
        "Prevalence_Rate",
        "Population_Affected",
        "DALYs",
        "Severity Score",
        "Improvement_5_Years",
        "Healthcare_Access_Pct",
        "Doctors_per_1000",
        "Hospital_Beds_per_1000",
        "Population Coverage (%)",
        "Health_Index",
        "Per_Capita_Income_USD",
        "Education_Index",
        "Urban_Population_Pct",
        "Rural_Population_Pct",
        "Urbanization_Rate"
]


    # Keep only existing columns
    feature_cols = [c for c in feature_cols if c in cleaned_df.columns]

    X = cleaned_df[feature_cols]
    y = cleaned_df["Mortality_Rate_per_100"]

    return X, y, feature_cols


# ---------------------------------------------
# MODEL TRAINING
# ---------------------------------------------
def train_mortality_model(cleaned_df, model_config):
    """
    Train RandomForest model to predict mortality rate.
    """
    X, y, feature_cols = prepare_ml_dataset(cleaned_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(**model_config)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    metrics = {
        "R2_Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        #"RMSE": np.sqrt(mse),
        "Train_Samples": len(X_train),
        "Test_Samples": len(X_test)
    }

    return model, metrics, feature_cols
# ---------------------------------------------
# FEATURE IMPORTANCE (EXPLAINABILITY)
# ---------------------------------------------
def get_feature_importance(model, feature_cols):
    """
    Extract feature importance from trained RandomForest.
    """
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return importance_df


# ---------------------------------------------
# SINGLE PREDICTION HELPER (FOR STREAMLIT LATER)
# ---------------------------------------------
def predict_mortality(model, input_data, feature_cols, fallback_medians):
    """
    Predict mortality rate for a single input record.
    """
    input_df = pd.DataFrame([input_data])

    # Ensure all required features exist
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = fallback_medians.get(col, 0)

    # Keep correct column order
    input_df = input_df[feature_cols]

    prediction = model.predict(input_df)[0]

    return round(float(prediction), 4)

def plot_feature_importance(importance_df, top_n=15):
    """
    Plot top N feature importances using Plotly.
    """
    top_features = importance_df.head(top_n)

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Features Driving Mortality Predictions",
        labels={
            "Importance": "Relative Importance",
            "Feature": "Feature"
        }
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=500,
        margin=dict(l=120, r=40, t=60, b=40)
    )

    return fig

def format_feature_importance_table(importance_df):
    """
    Prepare feature importance table for reports.
    """
    df = importance_df.copy()
    df["Importance (%)"] = (df["Importance"] * 100).round(2)
    return df[["Feature", "Importance (%)"]]

# ------------------------------Streamlit UI phase---------------------------------------------
st.set_page_config(
    page_title="Global Health Analytics & Mortality Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Global Health Analytics & Mortality Prediction System")
st.markdown(
    "An end-to-end data application combining ETL, analytics, machine learning, and explainability."
)

#----------Side bar---------------------
# ================================
# SIDEBAR: DATA UPLOAD
# ================================
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Global Health Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    raw_df = None

    for enc in encodings:
        try:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
            st.sidebar.success(f"✅ Read CSV using {enc}")
            break
        except Exception:
            continue

    if raw_df is None or raw_df.empty:
        st.sidebar.error("❌ Failed to read CSV or file is empty.")
    else:
        with st.spinner("Running ETL pipeline..."):
            # Create a temporary file to pass to the function
            with io.BytesIO(uploaded_file.getvalue()) as temp_buffer:
                temp_path = "temp_uploaded_health_data.csv"
                with open(temp_path, "wb") as f:
                    f.write(temp_buffer.read())

            # Call your existing cleaning function
            cleaned_df = clean_health_dataset(
            csv_path=temp_path,
            output_path=CLEANED_CSV_NAME,
            sqlite_db_path=SQLITE_DB_PATH,
            save_to_sqlite=False  # or True if you want SQLite
            )

            # Load the cleaned CSV back into a DataFrame
            cleaned_df = pd.read_csv(CLEANED_CSV_NAME)

        # 🔐 VALIDATION (THIS IS CRITICAL)
        if cleaned_df is None or cleaned_df.empty or cleaned_df.shape[1] == 0:
            st.sidebar.error("❌ ETL returned empty dataset.")
        else:
            # ✅ STORE EXACTLY LIKE YOUR MENTOR
            st.session_state.cleaned_df = cleaned_df
            st.session_state.data_loaded = True

            st.sidebar.success("✅ Dataset cleaned & loaded successfully")

            # Preview
            st.subheader("Preview of Cleaned Dataset")
            st.dataframe(cleaned_df.head(), use_container_width=True)

            st.download_button(
                "Download Cleaned CSV",
                data=cleaned_df.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_global_health_data.csv",
                mime="text/csv"
            )
if not st.session_state.data_loaded:
    st.info("👈 Upload a dataset from the sidebar to begin.")
    st.stop()  # ⛔ STOPS EXECUTION SAFELY
cleaned_df = st.session_state.cleaned_df


#st.sidebar.header("🔎 Global Filters")
#st.stop()


st.sidebar.header("🔎 Global Filters")

if "Year" in cleaned_df.columns:
    selected_year = st.sidebar.selectbox(
        "Select Year",
        sorted(cleaned_df["Year"].dropna().unique())
    )
else:
    st.sidebar.warning("Year column not found")
    selected_year = None

if "Country" in cleaned_df.columns:
    selected_country = st.sidebar.selectbox(
        "Select Country",
        sorted(cleaned_df["Country"].dropna().unique())
    )
else:
    st.sidebar.warning("Country column not found")
    selected_country = None

filtered_df = cleaned_df.copy()

if selected_year is not None:
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]

if selected_country is not None:
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]
# --- MODEL SETUP ---
model, metrics, feature_cols = train_mortality_model(
    cleaned_df,
    model_configs
)

feature_medians = cleaned_df[feature_cols].median().to_dict()

# COMPUTE FEATURE IMPORTANCE ONCE (GLOBAL)
feature_importance_df = get_feature_importance(model, feature_cols)

# --- STREAMLIT UI ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analytics Dashboard",
    "📈 Advanced Insights",
    "🤖 ML Prediction",
    "🧠 Model Explainability"
])

with tab1:
    st.subheader("📊 Top-Level Health Insights")

    col1, col2 = st.columns(2)

    with col1:
        deadliest_df = top_deadliest_diseases(filtered_df)
        if not deadliest_df.empty:
            st.plotly_chart(
                plot_top_deadliest_diseases(deadliest_df),
                use_container_width=True
            )
        else:
            st.warning("No mortality data available.")

    with col2:
        treatment_df = avg_treatment_cost_by_disease(filtered_df)
        if not treatment_df.empty:
            st.plotly_chart(
                plot_avg_treatment_cost(treatment_df),
                use_container_width=True
            )
        else:
            st.warning("No treatment cost data available.")

#st.divider()

    # ---- ADDITIONAL VISUALS ----
    if selected_country is not None:
        country_trend_df = mortality_trend_over_time(
        cleaned_df[cleaned_df["Country"] == selected_country]
    )
    else:
        country_trend_df = mortality_trend_over_time(cleaned_df)

    st.plotly_chart(
        plot_mortality_trend(country_trend_df),
        use_container_width=True
    )


    healthcare_df = healthcare_access_vs_mortality(filtered_df)
    st.plotly_chart(
        plot_healthcare_vs_mortality(healthcare_df),
        use_container_width=True
    )

with tab2:
    st.subheader("Advanced Mortality Drivers & Population Insights")

    corr_df = mortality_correlation_analysis(cleaned_df)
    st.plotly_chart(
        plot_mortality_correlation(corr_df),
        use_container_width=True
    )

    healthcare_burden_df = disease_healthcare_burden_analysis(filtered_df)

    st.plotly_chart(
        plot_disease_healthcare_bubble(healthcare_burden_df),
        use_container_width=True
    )


    gender_df = prepare_gender_impact_df(cleaned_df)

    st.plotly_chart(
        plot_gender_impact(gender_df),
        use_container_width=True
)

with tab3:
    st.subheader("🤖 Mortality Rate Prediction")

    st.markdown("Adjust health and socioeconomic factors to predict mortality rate.")

    # --- MODEL PERFORMANCE ---
    st.markdown("### 📊 Model Performance (Test Set)")

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", f"{metrics['R2_Score']:.3f}")
    col2.metric("MAE", f"{metrics['MAE']:.4f}")
    col3.metric("MSE", f"{metrics['MSE']:.6f}")
    #col4.metric("RMSE", f"{metrics['RMSE']:.4f}")

    st.caption(
        "R² shows how well the model explains mortality variation. "
        "Lower MAE/MSE/RMSE indicate better prediction accuracy."
    )

    st.divider()

    # --- USER INPUT ---
    st.markdown("### 🧮 Input Features")

    input_data = {}
    for feature in feature_cols:
        input_data[feature] = st.number_input(
            feature,
            value=float(feature_medians.get(feature, 0))
        )

    if st.button("Predict Mortality Rate"):
        prediction = predict_mortality(
            model,
            input_data,
            feature_cols,
            feature_medians
        )

        st.success(
            f"**Predicted Mortality Rate:** {prediction} deaths per 100 people"
        )

with tab4:
    st.subheader("Model Explainability")

    importance_df = get_feature_importance(model, feature_cols)
    fig = plot_feature_importance(importance_df)

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        format_feature_importance_table(importance_df),
        use_container_width=True
    )
#-------------------------Generating PDF Report------------------------------------------
def save_plotly_fig(fig, filename):
    fig.write_image(img_buffer, format="png", engine="orca")
    return filename

def generate_pdf_report(
    cleaned_df,
    filtered_df,
    figures: dict,
    feature_importance_df,
    model_metrics: dict,
    selected_year,
    selected_country
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    elements = []
    # --------------------
    # Title
    # --------------------
    elements.append(Paragraph(
        "<b>Global Health Analytics Report</b>",
        styles["Title"]
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        f"<b>Country:</b> {selected_country} &nbsp;&nbsp; "
        f"<b>Year:</b> {selected_year}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    # --------------------
    # Dataset Summary
    # --------------------
    elements.append(Paragraph("<b>Dataset Overview</b>", styles["Heading2"]))
    elements.append(Paragraph(
        f"Total records analyzed: {len(filtered_df)}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    # --------------------
    # Charts
    # --------------------
    elements.append(Paragraph("<b>Key Analytics</b>", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    for title, fig in figures.items():
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format="png", scale=2)
        img_buffer.seek(0)

        elements.append(
            Image(
                img_buffer,      # ← Pass the BytesIO directly
                width=450,
                height=300
         )
        )
        elements.append(Spacer(1, 20))
    
    # --------------------
    # Feature Importance Table
    # --------------------
    elements.append(Paragraph(
        "<b>Model Feature Importance</b>",
        styles["Heading2"]
    ))

    table_data = [["Feature", "Importance(%)"]] + feature_importance_df.head(10).values.tolist()

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    # --------------------
    # Model Metrics
    # --------------------
    elements.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))

    for k, v in model_metrics.items():
        elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return buffer
# --- Build figures for PDF ---
deadliest_df = top_deadliest_diseases(filtered_df)
treatment_df = avg_treatment_cost_by_disease(filtered_df)
corr_df = mortality_correlation_analysis(cleaned_df)

fig_deadliest = plot_top_deadliest_diseases(deadliest_df)
fig_treatment = plot_avg_treatment_cost(treatment_df)
fig_corr = plot_mortality_correlation(corr_df)

if st.button("📄 Generate PDF Report"):
    pdf_buffer = generate_pdf_report(
        cleaned_df=cleaned_df,
        filtered_df=filtered_df,
        figures={
            "Top Deadliest Diseases": fig_deadliest,
            "Treatment Cost Analysis": fig_treatment,
            "Mortality Correlation Heatmap": fig_corr
        },
        feature_importance_df=feature_importance_df,
        model_metrics={
            "Model": "Random Forest",
            "R² Score": round(metrics["R2_Score"], 3),
            "MAE": round(metrics["MAE"], 4),
            "MSE": round(metrics["MSE"], 6)
        },
        selected_year=selected_year,
        selected_country=selected_country
    )


    st.download_button(
        label="⬇ Download PDF Report",
        data=pdf_buffer,
        file_name="Global_Health_Report.pdf",
        mime="application/pdf"
    )

# ----------------Automated Email Report-------------------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = st.secrets.get("SENDER_EMAIL", "")
SENDER_PASSWORD = st.secrets.get("SENDER_PW", "")

def send_pdf_report_email(
    pdf_buffer: BytesIO,
    recipient_email: str,
    selected_country,
    selected_year
):
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg["Subject"] = f"Global Health Report | {selected_country} - {selected_year}"

    msg.set_content(
        f"""
Hello,

Attached is your automated Global Health Analytics Report.

Filters applied:
• Country: {selected_country}
• Year: {selected_year}

This report includes:
• Mortality & treatment analytics
• Correlation insights
• Machine learning feature importance
• Model performance metrics

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Regards,
Global Health Analytics System
"""
    )

    msg.add_attachment(
        pdf_buffer.getvalue(),
        maintype="application",
        subtype="pdf",
        filename=f"Global_Health_Report_{selected_country}_{selected_year}.pdf"
    )

    context = ssl.create_default_context()

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)


st.sidebar.header("📧 Email Report")

email_recipient = st.sidebar.text_input(
    "Recipient Email",
    value=DEFAULT_RECIPIENTS[0] if DEFAULT_RECIPIENTS else ""
)

send_email_btn = st.sidebar.button("📤 Email PDF Report")


if send_email_btn:
    if not email_recipient:
        st.sidebar.error("Please enter a recipient email.")
    else:
        try:
            pdf_buffer = generate_pdf_report(
                cleaned_df=cleaned_df,
                filtered_df=filtered_df,
                figures={
                    "Top Deadliest Diseases": fig_deadliest,
                    "Treatment Cost Analysis": fig_treatment,
                    "Mortality Correlation Heatmap": fig_corr
                },
                feature_importance_df=feature_importance_df,
                model_metrics={
                    "Model": "Random Forest",
                    "R² Score": round(metrics["R2_Score"], 3),
                    "MAE": round(metrics["MAE"], 4),
                    "MSE": round(metrics["MSE"], 6)
                },
                selected_year=selected_year,
                selected_country=selected_country
            )

            send_pdf_report_email(
                pdf_buffer=pdf_buffer,
                recipient_email=email_recipient,
                selected_country=selected_country,
                selected_year=selected_year
            )

            st.sidebar.success("✅ Report emailed successfully!")

        except Exception as e:
            st.sidebar.error(f"❌ Failed to send email: {e}")


