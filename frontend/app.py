import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="centered")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "credit_scoring_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return model, scaler, label_encoders, feature_columns

model, scaler, label_encoders, feature_columns = load_artifacts()

st.title("💳 Credit Scoring App")
st.write(
    "Cette application prédit le risque de défaut de paiement d'un client "
    "à partir d'un modèle Random Forest (Accuracy ~93%, ROC-AUC ~0.93)."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Âge", min_value=18, max_value=100, value=27)
    person_income = st.number_input("Revenu annuel ($)", min_value=0, value=45000, step=1000)
    person_home_ownership = st.selectbox("Statut logement", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    person_emp_length = st.number_input("Ancienneté emploi (années)", min_value=0.0, value=3.0, step=0.5)
    loan_intent = st.selectbox(
        "Motif du prêt",
        ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    )
    cb_person_cred_hist_length = st.number_input("Historique de crédit (années)", min_value=0, value=4)

with col2:
    loan_grade = st.selectbox("Grade du prêt", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Montant du prêt ($)", min_value=0, value=10000, step=500)
    loan_int_rate = st.number_input("Taux d'intérêt (%)", min_value=0.0, value=11.5, step=0.1)
    loan_percent_income = st.slider("% du revenu consacré au prêt", 0.0, 1.0, 0.22)
    cb_person_default_on_file = st.selectbox("Défaut de paiement antérieur ?", ["N", "Y"])

st.divider()

if st.button("🔍 Prédire", use_container_width=True):
    input_dict = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }

    encoded = dict(input_dict)
    for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        le = label_encoders[col]
        encoded[col] = int(le.transform([encoded[col]])[0])

    df_input = pd.DataFrame([encoded])[feature_columns]
    scaled_input = scaler.transform(df_input)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Résultat")
    if prediction == 1:
        st.error(f"⚠️ Risque de défaut élevé — probabilité: {probability:.1%}")
        st.write("**Recommandation : Refuser le prêt**")
    else:
        st.success(f"✅ Risque de défaut faible — probabilité: {probability:.1%}")
        st.write("**Recommandation : Accorder le prêt**")

    st.progress(float(probability))

st.divider()
st.caption("Modèle : Random Forest Classifier | Dataset : Credit Risk Dataset | Projet portfolio Data Science")
