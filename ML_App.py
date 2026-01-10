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
    ["About this System", "Dataset Overview", "Model Training", "Prediction Dashboard"]
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
# Model Training (ONLY Stacking Regressor)
# --------------------------------
elif menu == "Model Training":
    st.header("Tuned Stacking Regressor Training")

    features = [
        'state', 'type', 'sex',
        'piped_water', 'sanitation', 'electricity',
        'income_mean', 'gini', 'poverty_absolute', 'cpi'
    ]
    target = 'rate'

    if st.button("Train Tuned Stacking Model"):
        X = df_clean[features].copy()
        y = df_clean[target]

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

        stacking_model = StackingRegressor(
            estimators=base_learners,
            final_estimator=LinearRegression()
        )

        stacking_model.fit(X_train, y_train)

        joblib.dump(stacking_model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(encoders, "encoders.pkl")
        joblib.dump(features, "feature_names.pkl")

        y_pred = stacking_model.predict(X_test)

        st.success("Tuned Stacking Regressor trained successfully")

        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")

# --------------------------------
# Prediction Dashboard
# --------------------------------
elif menu == "Prediction Dashboard":
    st.header("Prediction Dashboard")

    if not os.path.exists("model.pkl"):
        st.warning("Please train the model first.")
    else:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        features = joblib.load("feature_names.pkl")

        input_data = []
        cols = st.columns(2)

        for i, f in enumerate(features):
            with cols[i % 2]:
                if f in encoders:
                    val = st.selectbox(f, encoders[f].classes_)
                    input_data.append(encoders[f].transform([val])[0])
                else:
                    val = st.number_input(
                        f, value=float(df_clean[f].median())
                    )
                    input_data.append(val)

        if st.button("Predict"):
            X_final = scaler.transform([input_data])
            prediction = model.predict(X_final)[0]

            st.success(f"Predicted Mortality Rate: {prediction:.2f}")

            fig, ax = plt.subplots(figsize=(10, 3))
            sns.kdeplot(df_clean['rate'], fill=True, label="Historical")
            plt.axvline(prediction, color='red', linestyle='--', label="Prediction")
            plt.legend()
            st.pyplot(fig)
