
# Import required libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# 2. LOAD TRAINED MODEL
model = joblib.load("model.joblib")
feature_names = joblib.load("feature_names.joblib")
cm = joblib.load("confusion_matrix.joblib")

# 3. MAIN PROJECT HEADING
st.title("💳 Credit Card Fraud Detection Using Classification Techniques")

st.write(
    "This application detects fraudulent credit card transactions "
    "using Machine Learning Classification Techniques."
)

# 4. TRANSACTION INPUT SECTION
st.sidebar.header("Transaction Information")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV File",
    type=["csv"]
)

# 5. READ UPLOADED DATASET
if uploaded_file is not None:

    transaction_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Transaction Details")
    st.dataframe(transaction_data, width="stretch")

    # 6. PREDICT CREDIT CARD FRAUD
    missing_features = [
        col for col in feature_names
        if col not in transaction_data.columns
    ]

    if missing_features:
        st.error(f"Missing required features: {missing_features}")

    else:
        input_data = transaction_data[feature_names]

        if input_data.isnull().sum().sum() > 0:
            st.error("Please remove missing values from the dataset.")

        else:
            if st.button("Detect Fraud", type="primary"):

                predictions = model.predict(input_data)
                probabilities = model.predict_proba(input_data)[:, 1]

                results = transaction_data.copy()
                results["Predicted Class"] = predictions

                results["Prediction"] = [
                    "Fraudulent" if value == 1 else "Legitimate"
                    for value in predictions
                ]

                results["Fraud Probability"] = probabilities

                # DISPLAY PREDICTION RESULTS
                st.subheader("Fraud Detection Results")
                st.dataframe(results, width="stretch")

                # 7. DISPLAY FRAUD DETECTION SUMMARY
                legitimate_count = (predictions == 0).sum()
                fraud_count = (predictions == 1).sum()

                st.subheader("Transaction Summary")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Legitimate Transactions", legitimate_count)

                with col2:
                    st.metric("Fraudulent Transactions", fraud_count)

                # 8. SEPARATE TRANSACTION TABLES
                legitimate_transactions = results[
                    results["Prediction"] == "Legitimate"
                ]

                fraudulent_transactions = results[
                    results["Prediction"] == "Fraudulent"
                ]

                tab1, tab2 = st.tabs([
                    "Legitimate Transactions",
                    "Fraudulent Transactions"
                ])

                with tab1:
                    st.subheader("Legitimate Transactions")
                    st.write(f"Total: {len(legitimate_transactions)}")
                    st.dataframe(
                        legitimate_transactions.head(100),
                        width="stretch"
                    )

                with tab2:
                    st.subheader("Fraudulent Transactions")
                    st.write(f"Total: {len(fraudulent_transactions)}")
                    st.dataframe(
                        fraudulent_transactions.head(100),
                        width="stretch"
                    )

                # 9. DOWNLOAD PREDICTION RESULTS
                csv_data = results.to_csv(index=False).encode("utf-8")
                st.download_button(
    label="Download Prediction Results",
    data=csv_data,
    file_name="fraud_detection_results.csv",
    mime="text/csv",
    on_click="ignore"
)

                # 10. MODEL COMPARISON
                st.header("Classification Algorithm Comparison")

                comparison_df = pd.read_csv("model_comparison.csv")

                st.subheader("Model Performance Table")
                st.dataframe(comparison_df, width="stretch")

                st.subheader("Accuracy Comparison")
                st.bar_chart(
                    comparison_df.set_index("Algorithm")["Accuracy"]
                )

                st.subheader("F1 Score Comparison")
                st.bar_chart(
                    comparison_df.set_index("Algorithm")["F1 Score"]
                )

                # 11. CONFUSION MATRIX VISUALIZATION
                st.header("Confusion Matrix")

                fig, ax = plt.subplots()
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax
                )
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")
                st.pyplot(fig)
