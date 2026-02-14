import streamlit as st
import pandas as pd
import joblib

st.title("Credit Card Fraud Detection")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Model selection dropdown
model_option = st.selectbox(
    "Select a Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    # Drop Class column if present
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    if "Time" in data.columns:
        data = data.drop("Time", axis=1)

    # Scale data
    data_scaled = scaler.transform(data)

    # Load selected model
    if model_option == "Logistic Regression":
        model = joblib.load("model/logistic_model.pkl")
    elif model_option == "Decision Tree":
        model = joblib.load("model/decision_tree_model.pkl")
    elif model_option == "KNN":
        model = joblib.load("model/knn_model.pkl")
    elif model_option == "Naive Bayes":
        model = joblib.load("model/naive_bayes_model.pkl")
    elif model_option == "Random Forest":
        model = joblib.load("rmodel/andom_forest_model.pkl")
    else:
        model = joblib.load("model/xgboost_model.pkl")

    predictions = model.predict(data_scaled)

    st.write("Predictions:")
    data["Prediction"] = predictions
    st.write(data)

