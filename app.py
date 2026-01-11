from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from time import time  # For CSS cache-busting

# -------------------- FLASK APP --------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------- LOAD MODEL & SCALER --------------------
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    # Pass timestamp to avoid CSS caching
    return render_template("index.html", time=time())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- GET FORM DATA --------
        gender = request.form["Gender"]
        married = request.form["Married"]
        dependents = request.form["Dependents"]
        education = request.form["Education"]
        self_employed = request.form["Self_Employed"]
        applicant_income = int(request.form["ApplicantIncome"])
        coapplicant_income = int(request.form["CoapplicantIncome"])
        loan_amount = int(request.form["LoanAmount"])
        loan_amount_term = int(request.form["Loan_Amount_Term"])
        credit_history = int(request.form["Credit_History"])
        property_area = request.form["Property_Area"]

        # -------- LABEL ENCODER MAPPING --------
        gender_map = {"Female": 0, "Male": 1}
        married_map = {"No": 0, "Yes": 1}
        dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
        education_map = {"Graduate": 0, "Not Graduate": 1}
        self_employed_map = {"No": 0, "Yes": 1}
        property_area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

        # -------- FEATURE ORDER MUST MATCH TRAINING --------
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

        final_features = np.array([features])
        final_features_scaled = scaler.transform(final_features)

        # -------- PREDICTION --------
        prediction = model.predict(final_features_scaled)[0]
        probability = model.predict_proba(final_features_scaled)[0][1]

        result = "Approved" if prediction == 1 else "Not Approved"

        return render_template(
            "index.html",
            prediction_text=f"Loan Status: {result} (Approval Probability: {probability:.2f})",
            time=time()  # update CSS cache
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", time=time())

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    # Use PORT env variable provided by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
