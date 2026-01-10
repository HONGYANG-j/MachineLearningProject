import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# XGBoost Availability Check
# -----------------------------
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Child Health Vulnerability Analysis System",
    layout="wide"
)

# -----------------------------
# Data Loading & Cleaning
# -----------------------------
@st.cache_data
def load_and_clean_data():
    if not os.path.exists("mergednew.csv"):
        return None

    df = pd.read_csv("mergednew.csv")

    # Convert date → year
    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column (YYYY-MM-DD)")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Remove rows without target
    df = df.dropna(subset=["rate"])

    # Fill numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df

df_clean = load_and_clean_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("System Navigation")
menu = st.sidebar.radio(
    "Select Module",
    [
        "About this System",
        "Dataset Overview",
        "Model Training",
        "Prediction Dashboard"
    ]
)

# -----------------------------
# 1. About this System
# -----------------------------
if menu == "About this System":
    st.header("Predicting Child Health Vulnerabilities in Malaysia")

    st.write("""
    This system predicts early childhood mortality rates using a **Tuned Stacking
    Regressor** that integrates socio-economic, infrastructure, and **temporal
    (year-based)** indicators derived from real-world date records.
    """)

# -----------------------------
# 2. Dataset Overview
# -----------------------------
elif menu == "Dataset Overview":
    st.header("Dataset Overview")

    if df_clean is None:
        st.error("Dataset not found.")
        st.stop()

    st.dataframe(df_clean.head())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df_clean.select_dtypes(include=[np.number]).corr(),
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

# -----------------------------
# 3. Model Training
# -----------------------------
elif menu == "Model Training":
    st.header("Model Training – Tuned Stacking Regressor")

    if not XGB_AVAILABLE:
        st.error("XGBoost is required. Please install xgboost.")
        st.stop()

    if df_clean is None:
        st.error("Dataset not loaded.")
        st.stop()

    features = [
        "year",
        "state", "type", "sex",
        "piped_water", "sanitation",
        "electricity", "income_mean",
        "gini", "poverty_absolute", "cpi"
    ]
    target = "rate"

    if st.button("Train Tuned Stacking Regressor"):
        with st.spinner("Training ensemble model..."):
            X = df_clean[features].copy()
            y = df_clean[target]

            # Encode categorical features
            encoders = {}
            for col in ["state", "type", "sex"]:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            base_learners = [
                ("rf", RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    random_state=42
                )),
                ("xgb", XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=42
                ))
            ]

            model = StackingRegressor(
                estimators=base_learners,
                final_estimator=LinearRegression()
            )

            model.fit(X_train, y_train)

            # Save artifacts
            joblib.dump(model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(encoders, "encoders.pkl")
            joblib.dump(features, "feature_names.pkl")

            y_pred = model.predict(X_test)

            st.success("Model trained successfully!")

            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
            c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            c3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")

# -----------------------------
# 4. Prediction Dashboard
# -----------------------------
elif menu == "Prediction Dashboard":
    st.header("Prediction Dashboard")

    if not os.path.exists("model.pkl"):
        st.warning("Train the model first.")
        st.stop()

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    features = joblib.load("feature_names.pkl")

    input_data = []
    cols = st.columns(2)

    for i, f in enumerate(features):
        with cols[i % 2]:
            if f == "year":
                val = st.number_input(
                    "Year (Prediction Start)",
                    min_value=2023,
                    max_value=2050,
                    value=2023,
                    step=1
                )
                input_data.append(val)

            elif f in encoders:
                val = st.selectbox(
                    f.replace("_", " ").capitalize(),
                    encoders[f].classes_
                )
                input_data.append(encoders[f].transform([val])[0])

            else:
                val = st.number_input(
                    f.replace("_", " ").capitalize(),
                    value=float(df_clean[f].median())
                )
                input_data.append(val)

    if st.button("Generate Prediction"):
        X_input = scaler.transform([input_data])
        res = model.predict(X_input)[0]

        st.success(f"Predicted Child Mortality Rate: {res:.2f}")

        yearly_avg = df_clean.groupby("year")["rate"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(yearly_avg["year"], yearly_avg["rate"], marker="o", label="Historical Average")
        ax.axhline(res, color="red", linestyle="--", label="Predicted Rate")
        ax.set_title("Child Mortality Trend vs Prediction")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mortality Rate")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
