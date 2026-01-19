# Modified app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import os
from PIL import Image

from models import Autoencoder, GCN
from predict import predict_autoencoder, predict_gnn
from utils import plot_confusion_matrix, plot_reconstruction_error_histogram

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🔍",
    layout="wide",
)

@st.cache_resource
def load_models():
    try:
        base_path = os.path.dirname(__file__)
        encoders = joblib.load(os.path.join(base_path, "encoders.pkl"))
        scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))

        input_dim = len(encoders) + 1
        autoencoder_model = Autoencoder(input_dim)
        autoencoder_model.load_state_dict(torch.load(os.path.join(base_path, "autoencoder_model.pth"), map_location=torch.device("cpu")))
        autoencoder_model.eval()

        gnn_model_path = os.path.join(base_path, "gnn_model.pth")
        gnn_available = os.path.exists(gnn_model_path)
        gnn_model = None
        if gnn_available:
            gnn_input_dim = len(encoders) + 3
            gnn_model = GCN(gnn_input_dim)
            gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=torch.device("cpu")))
            gnn_model.eval()

        threshold = 7.1

        return {
            "encoders": encoders,
            "scaler": scaler,
            "autoencoder_model": autoencoder_model,
            "gnn_model": gnn_model,
            "threshold": threshold,
            "gnn_available": gnn_available
        }

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def main():
    st.title("🔍 Credit Card Fraud Detection System")
    st.markdown("""
    This application uses machine learning models to detect potential credit card fraud:
    - **Autoencoder Model**: Detects anomalies in transaction patterns
    - **Graph Neural Network (GNN)**: Analyzes connections between transactions
    """)

    with st.spinner("Loading models..."):
        models_data = load_models()

    if not models_data:
        st.error("Failed to load models. Please check if model files exist.")
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Fraud Detection", "Model Insights", "About"])

    if page == "Fraud Detection":
        display_fraud_detection_page(models_data)
    elif page == "Model Insights":
        display_model_insights_page(models_data)
    else:
        display_about_page()

def display_fraud_detection_page(models_data):
    st.header("Transaction Analysis")
    st.markdown("Enter transaction details below to check for potential fraud.")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input("Amount (₹)", min_value=1.0, max_value=1000000.0, value=1000.0)
            merchant = st.selectbox("Merchant", list(models_data["encoders"]["Merchant"].classes_))
            cardholder_name = st.selectbox("Cardholder Name", list(models_data["encoders"]["Cardholder_Name"].classes_))

        with col2:
            transaction_type = st.selectbox("Transaction Type", list(models_data["encoders"]["Transaction_Type"].classes_))
            device_used = st.selectbox("Device Used", list(models_data["encoders"]["Device_Used"].classes_))
            location = st.selectbox("Location", list(models_data["encoders"]["Location"].classes_))

        run_analysis = st.form_submit_button("Analyze Transaction")

    if run_analysis:
        transaction = {
            "Amount": amount,
            "Merchant": merchant,
            "Cardholder_Name": cardholder_name,
            "Transaction_Type": transaction_type,
            "Device_Used": device_used,
            "Location": location
        }

        st.subheader("Analysis Results")
        progress_bar = st.progress(0)

        progress_bar.progress(25)
        autoencoder_result, error, top_features = predict_autoencoder(
            transaction,
            models_data["autoencoder_model"],
            models_data["encoders"],
            models_data["scaler"],
            models_data["threshold"]
        )

        progress_bar.progress(50)

        if models_data["gnn_available"] and models_data["gnn_model"]:
            gnn_result, fraud_neighbors = predict_gnn(
                transaction,
                models_data["gnn_model"],
                models_data["encoders"],
                models_data["scaler"]
            )
            progress_bar.progress(75)
        else:
            gnn_result = None
            fraud_neighbors = []

        progress_bar.progress(100)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Autoencoder Analysis")
            st.metric("Reconstruction Error", f"{error:.5f}")
            if autoencoder_result == "FRAUDULENT":
                st.error("⚠️ POTENTIALLY FRAUDULENT")
            else:
                st.success("✅ LEGITIMATE")

        with col2:
            st.subheader("Graph Neural Network Analysis")
            if gnn_result == "FRAUDULENT":
                st.error("⚠️ POTENTIALLY FRAUDULENT")
                if fraud_neighbors:
                    st.warning("Connected to previous frauds via:")
                    for attr in fraud_neighbors:
                        st.text(f"  - Shared {attr}")
                else:
                    st.info("No direct links to known frauds but pattern detected.")
            elif gnn_result == "LEGITIMATE":
                st.success("✅ LEGITIMATE")
            else:
                st.info("GNN model not available")

        st.subheader("Final Verdict")
        if autoencoder_result == "FRAUDULENT" or gnn_result == "FRAUDULENT":
            st.error("⚠️ This transaction is flagged as POTENTIALLY FRAUDULENT")
        else:
            st.success("✅ This transaction appears to be LEGITIMATE")

def display_model_insights_page(models_data):
    st.header("Model Insights")
    st.write("Explore how our fraud detection models work under the hood.")

    tab1, tab2 = st.tabs(["Autoencoder Model", "Graph Neural Network"])

    with tab1:
        st.subheader("Autoencoder Model")
        fig = plot_reconstruction_error_histogram(models_data["threshold"])
        if fig:
            st.pyplot(fig)
        fig = plot_confusion_matrix("autoencoder")
        if fig:
            st.pyplot(fig)

    with tab2:
        st.subheader("Graph Neural Network Model")
        fig = plot_confusion_matrix("gnn")
        if fig:
            st.pyplot(fig)

def display_about_page():
    st.header("About This Application")
    st.write("""
    This app uses Autoencoder and Graph Neural Networks for credit card fraud detection.
    - Trained on anonymized transaction data.
    - Detects anomalies and network-based fraud links.
    """)

if __name__ == "__main__":
    main()