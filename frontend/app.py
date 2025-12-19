import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Explainable Loan Approval", layout="centered")
st.title("🏦 Explainable AI – Loan Approval")

st.subheader("Applicant Details")

# -------------------------
# Inputs
# -------------------------
person_age = st.number_input("Age", 18, 100, 30)
person_gender = st.selectbox("Gender", ["Male", "Female"])
person_education = st.selectbox(
    "Education", ["High School", "Bachelor", "Master", "Doctorate"]
)
person_income = st.number_input("Annual Income", 0, step=1000)
person_emp_exp = st.number_input("Employment Experience (years)", 0, 50)
person_home_ownership = st.selectbox(
    "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_amnt = st.number_input("Loan Amount", 0)
loan_intent = st.selectbox(
    "Loan Intent",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)
loan_int_rate = st.number_input("Interest Rate (%)", 0.0)
loan_percent_income = st.number_input("Loan % of Income", 0.0)

credit_score = st.number_input("Credit Score", 300, 900)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", 0, 50)
previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Default", ["Yes", "No"]
)

# -------------------------
# Payload (must match backend)
# -------------------------
payload = {
    "person_age": person_age,
    "person_gender": 1 if person_gender == "Male" else 0,
    "person_education": ["High School", "Bachelor", "Master", "Doctorate"].index(person_education),
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"].index(person_home_ownership),
    "loan_amnt": loan_amnt,
    "loan_intent": ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"].index(loan_intent),
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": 1 if previous_loan_defaults_on_file == "Yes" else 0
}

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Loan Approval"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    if response.status_code != 200:
        st.error("❌ API Error")
        st.write(response.text)

    else:
        result = response.json()

        if "error" in result:
            st.error("❌ Backend Error")
            st.write(result["error"])

        else:
            decision = "✅ APPROVED" if result["prediction"] == 1 else "❌ REJECTED"
            st.subheader(decision)
            st.write("Confidence:", result["confidence"])

            # -------------------------
            # SHAP Explanation
            # -------------------------
            st.subheader("🔍 SHAP Explanation (Feature Impact)")

            shap_df = pd.DataFrame(
                list(result["shap_values"].items()),
                columns=["Feature", "SHAP Value"]
            ).sort_values(by="SHAP Value", key=abs, ascending=False)

            st.dataframe(shap_df)
            st.bar_chart(shap_df.set_index("Feature"))
