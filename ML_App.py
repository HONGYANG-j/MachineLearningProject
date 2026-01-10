import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Child Health Vulnerability Analysis System",
    layout="wide"
)

# --------------------------------
# Data Loading & Cleaning
# --------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("mergednew.csv")

    df = df.dropna(subset=['rate'])

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    return df

df_clean = load_and_clean_data()

# --------------------------------
# Sidebar Navigation
# --------------------------------
st.sidebar.title("System Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["About this System", "Dataset Overview", "Train & Predict Model"]
)

# --------------------------------
# About
# --------------------------------
if menu == "About this System":
    st.header("Predicting Child Health Vulnerabilities in Malaysia")
    st.write("""
    This system applies a **tuned stacking regression model** to predict early childhood
    mortality rates using socio-economic and infrastructure indicators.
    """)

# --------------------------------
# Dataset Overview
# --------------------------------
elif menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(df_clean.head())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df_clean.select_dtypes(include=[np.number]).corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax
    )
    st.pyplot(fig)

# --------------------------------
# Model Training & Dashboard
# --------------------------------
elif menu == "Train & Predict Model":
    st.header("Tuned Stacking Regressor: Training & Prediction")

    if not XGB_AVAILABLE:
        st.error("XGBoost library not found.")
    elif df_clean is None:
        st.error("Dataset not loaded.")
    else:
        features = ['state', 'type', 'sex', 'piped_water', 'sanitation',
                    'electricity', 'income_mean', 'gini',
                    'poverty_absolute', 'cpi']
        target = 'rate'

        # -------------------------
        # MODEL TRAINING SECTION
        # -------------------------
        st.subheader("1️⃣ Model Training")

        if st.button("Train Tuned Stacking Regressor"):
            with st.spinner("Training ensemble model..."):
                X = df_clean[features].copy()
                y = df_clean[target]

                # Encode categorical features
                encoders = {}
                for col in ['state', 'type', 'sex']:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

                base_learners = [
                    ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
                    ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
                ]

                model = StackingRegressor(
                    estimators=base_learners,
                    final_estimator=LinearRegression()
                )

                model.fit(X_train, y_train)

                # Store everything in session
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.encoders = encoders
                st.session_state.features = features

                y_pred = model.predict(X_test)

                st.success("Model trained successfully!")

                c1, c2, c3 = st.columns(3)
                c1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
                c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                c3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")

        # -------------------------
        # PREDICTION SECTION
        # -------------------------
        if "model" in st.session_state:
            st.divider()
            st.subheader("2️⃣ Prediction Dashboard")

            input_data = []
            cols = st.columns(2)

            for i, f in enumerate(st.session_state.features):
                with cols[i % 2]:
                    if f in st.session_state.encoders:
                        val = st.selectbox(
                            f.replace('_', ' ').capitalize(),
                            st.session_state.encoders[f].classes_
                        )
                        input_data.append(
                            st.session_state.encoders[f].transform([val])[0]
                        )
                    else:
                        val = st.number_input(
                            f.replace('_', ' ').capitalize(),
                            value=float(df_clean[f].median())
                        )
                        input_data.append(val)

            if st.button("Generate Prediction"):
                X_final = st.session_state.scaler.transform([input_data])
                res = st.session_state.model.predict(X_final)[0]

                st.success(f"Predicted Mortality Rate: {res:.2f}")

                # Optional trend visualization (if year exists)
                if 'year' in df_clean.columns:
                    yearly_avg = df_clean.groupby('year')['rate'].mean().reset_index()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(yearly_avg['year'], yearly_avg['rate'], marker='o')
                    ax.axhline(res, color='red', linestyle='--')
                    ax.set_title("Trend vs Prediction")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Mortality Rate")
                    ax.grid(True)
                    st.pyplot(fig)
