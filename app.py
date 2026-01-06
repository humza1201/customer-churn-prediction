from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load("artifacts/model.pkl")
pipeline = joblib.load("artifacts/pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = {
            "gender": request.form["gender"],
            "SeniorCitizen": request.form["SeniorCitizen"],
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        df = pd.DataFrame([data])
        X = pipeline.transform(df)
        prob = model.predict_proba(X)[0][1]

        risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

        return render_template("index.html", probability=round(prob,2), risk=risk)

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
