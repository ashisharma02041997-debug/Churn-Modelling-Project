import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

# -----------------------------------------
# Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------------------
# Load Model and Preprocessing Files
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = load_model(
        BASE_DIR / "bank_churn_ann_model.keras",
        compile=False
    )

    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    encoder = joblib.load(BASE_DIR / "encoder.pkl")

    feature_columns = joblib.load(
        BASE_DIR / "feature_columns.pkl"
    )

    return model, scaler, encoder, feature_columns


try:
    model, scaler, encoder, feature_columns = load_artifacts()

except Exception as e:
    st.error(f"Error loading saved files: {e}")
    st.stop()


# -----------------------------------------
# Application Title
# -----------------------------------------

st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "This application predicts whether a bank customer "
    "is likely to churn using an Artificial Neural Network "
    "(ANN) optimized with KerasTuner."
)

st.divider()

# -----------------------------------------
# Customer Input Details
# -----------------------------------------

st.subheader("Enter Customer Details")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Account Balance",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

num_products = st.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_cr_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

is_active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

st.divider()

# -----------------------------------------
# Prediction
# -----------------------------------------

if st.button("Predict Customer Churn", type="primary"):

    # Step 1: Create DataFrame from user inputs

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })

    # Step 2: Encode categorical variables

    categorical_cols = ["Geography", "Gender"]

    encoded_data = encoder.transform(
        input_data[categorical_cols]
    )

    if hasattr(encoded_data, "toarray"):
        encoded_data = encoded_data.toarray()

    encoded_columns = encoder.get_feature_names_out(
        categorical_cols
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=input_data.index
    )

    # Step 3: Combine numerical and encoded features

    numerical_df = input_data.drop(
        columns=categorical_cols
    )

    final_input = pd.concat(
        [numerical_df, encoded_df],
        axis=1
    )

    # Step 4: Arrange features in training order

    final_input = final_input[feature_columns].copy()

    # Step 5: Apply the saved StandardScaler

    scale_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary"
    ]

    final_input[scale_cols] = scaler.transform(
        final_input[scale_cols]
    )

    # Step 6: Predict churn probability

    prediction_probability = float(
        model.predict(
            final_input.to_numpy(dtype=np.float32),
            verbose=0
        )[0][0]
    )

    # Step 7: Apply optimized threshold

    threshold = 0.4

    prediction = int(
        prediction_probability >= threshold
    )

    # -----------------------------------------
    # Display Prediction Results
    # -----------------------------------------

    st.subheader("Prediction Results")

    st.metric(
        "Churn Probability",
        f"{prediction_probability * 100:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn!"
        )

        st.write(
            "The customer has been identified as "
            "potentially at risk of leaving the bank."
        )

    else:

        st.success(
            "✅ Customer is unlikely to churn!"
        )

        st.write(
            "The customer is predicted to remain "
            "with the bank."
        )

    st.progress(
        min(max(prediction_probability, 0.0), 1.0)
    )

    st.caption(
        "Classification threshold: 0.40. "
        "Predictions are estimates and not guarantees."
    )

# -----------------------------------------
# Footer
# -----------------------------------------

st.divider()

st.caption(
    "Bank Customer Churn Prediction | "
    "Deep Learning ANN | KerasTuner"
)