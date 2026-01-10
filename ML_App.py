import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =============================
# XGBoost Availability Check
# =============================
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Child Health Vulnerability Analysis System",
    layout="wide"
)

# =============================
# Data Loading & Cleaning
# =============================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("mergednew.csv")

    # Remove rows without target
    df = df.dropna(subset=['rate'])

    # Fill numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


df_clean = load_and_clean_data()

# =============================
# Sidebar Navigation
# =============================
st.sidebar.title("System Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["About this System", "Dataset Overview", "Train & Predict Model"]
)

# =============================
# About Page
# =============================
if menu == "About this System":
    st.header("Predicting Child Health Vulnerabilities in Malaysia")
    st.write("""
    This system applies a **tuned stacking regression model**
    to predict **early childhood mortality rates** using
    socio-economic, demographic, and infrastructure indicators.

    The model supports **future-year prediction starting from 2023**,
    enabling scenario-based planning and policy evaluation.
    """)

# =============================
# Dataset Overview
# =============================
elif menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(df_clean.head())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df_clean.select_dtypes(include=[np.number]).corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

# =============================
# Model Training & Prediction
# =============================
elif menu == "Train & Predict Model":
    st.header("Tuned Stacking Regressor: Training & Prediction")

    if not XGB_AVAILABLE:
        st.error("❌ XGBoost library not installed.")
        st.stop()

    # -------------------------
    # Feature & Target
    # -------------------------
    features = [
        'year',                    # ✅ year included
        'state', 'type', 'sex',
        'piped_water', 'sanitation',
        'electricity', 'income_mean',
        'gini', 'poverty_absolute',
        'cpi'
    ]
    target = 'rate'

    # -------------------------
    # Model Training
    # -------------------------
    st.subheader("1️⃣ Model Training")

    if st.button("Train Tuned Stacking Regressor"):
        with st.spinner("Training ensemble model..."):
            X = df_clean[features].copy()
            y = df_clean[target]

            # Encode categorical variables
            encoders = {}
            for col in ['state', 'type', 'sex']:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Base learners
            base_learners = [
                ('rf', RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    random_state=42
                )),
                ('xgb', XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=42
                ))
            ]

            # Stacking Regressor
            model = StackingRegressor(
                estimators=base_learners,
                final_estimator=LinearRegression()
            )

            model.fit(X_train, y_train)

            # Save to session
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.encoders = encoders
            st.session_state.features = features

            # Evaluation
            y_pred = model.predict(X_test)

            st.success("✅ Model trained successfully!")

            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
            c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            c3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")

    # -------------------------
    # Prediction Dashboard
    # -------------------------
    if "model" in st.session_state:
        st.divider()
        st.subheader("2️⃣ Prediction Dashboard")

        input_data = []
        cols = st.columns(2)

        for i, f in enumerate(st.session_state.features):
            with cols[i % 2]:

                # Year input (2023 onwards)
                if f == 'year':
                    val = st.number_input(
                        "Year",
                        min_value=2023,
                        max_value=2050,
                        value=2023,
                        step=1
                    )
                    input_data.append(val)

                # Categorical
                elif f in st.session_state.encoders:
                    val = st.selectbox(
                        f.replace('_', ' ').capitalize(),
                        st.session_state.encoders[f].classes_
                    )
                    input_data.append(
                        st.session_state.encoders[f].transform([val])[0]
                    )

                # Numerical
                else:
                    val = st.number_input(
                        f.replace('_', ' ').capitalize(),
                        value=float(df_clean[f].median())
                    )
                    input_data.append(val)

        if st.button("Generate Prediction"):
            X_input = pd.DataFrame(
                [input_data],
                columns=st.session_state.features
            )

            X_scaled_final = st.session_state.scaler.transform(X_input)
            prediction = st.session_state.model.predict(X_scaled_final)[0]

            st.success(f"📊 Predicted Mortality Rate: {prediction:.2f}")

            # -------------------------
            # Trend Visualization
            # -------------------------
            if 'year' in df_clean.columns:
                yearly_avg = (
                    df_clean
                    .groupby('year')['rate']
                    .mean()
                    .reset_index()
                )
