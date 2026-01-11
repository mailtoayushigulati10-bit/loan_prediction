from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from time import time  # for cache-busting CSS

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load trained model and scaler
try:
    model = pickle.load(open("loan_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# Mapping for categorical features
gender_map = {"Female": 0, "Male": 1}
married_map = {"No": 0, "Yes": 1}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
education_map = {"Graduate": 0, "Not Graduate": 1}
self_employed_map = {"No": 0, "Yes": 1}
property_area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

@app.route("/")
def home():
    return render_template("index.html", time=int(time()))  # Pass time for CSS cache-busting

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template("index.html", prediction_text="Model not loaded!", time=int(time()))
    
    try:
        # Read form data
        gender = request.form.get("Gender")
        married = request.form.get("Married")
        dependents = request.form.get("Dependents")
        education = request.form.get("Education")
        self_employed = request.form.get("Self_Employed")
        applicant_income = int(request.form.get("ApplicantIncome"))
        coapplicant_income = int(request.form.get("CoapplicantIncome"))
        loan_amount = int(request.form.get("LoanAmount"))
        loan_amount_term = int(request.form.get("Loan_Amount_Term"))
        credit_history = int(request.form.get("Credit_History"))
        property_area = request.form.get("Property_Area")

        # Convert to model input
        features = [
            gender_map[gender],
            married_map[married],
            dependents_map[dependents],
            education_map[education],
            self_employed_map[self_employed],
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area_map[property_area]
        ]

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        result_text = "Approved" if prediction == 1 else "Not Approved"

        return render_template(
            "index.html",
            prediction_text=f"Loan Status: {result_text} (Probability: {probability:.2f})",
            time=int(time())
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", time=int(time()))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port, debug=True)
