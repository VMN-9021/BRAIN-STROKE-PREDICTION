from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained files
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
encoders = joblib.load("encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------
    # Get input from form
    # -----------------------------------------

    patient_data = pd.DataFrame([{
        "gender": request.form["gender"],
        "age": float(request.form["age"]),
        "hypertension": int(request.form["hypertension"]),
        "heart_disease": int(request.form["heart_disease"]),
        "ever_married": request.form["ever_married"],
        "work_type": request.form["work_type"],
        "Residence_type": request.form["Residence_type"],
        "avg_glucose_level": float(
            request.form["avg_glucose_level"]
        ),
        "bmi": float(request.form["bmi"]),
        "smoking_status": request.form["smoking_status"]
    }])


    # -----------------------------------------
    # Encode categorical columns
    # -----------------------------------------

    for column, encoder in encoders.items():

        value = patient_data[column].iloc[0]

        # Handle unknown category
        if value not in encoder.classes_:
            return render_template(
                "result.html",
                result="Invalid Input",
                probability=0
            )

        patient_data[column] = encoder.transform(
            patient_data[column].astype(str)
        )


    # -----------------------------------------
    # Handle missing values
    # -----------------------------------------

    patient_data = pd.DataFrame(
        imputer.transform(patient_data),
        columns=patient_data.columns
    )


    # -----------------------------------------
    # Scale data
    # -----------------------------------------

    patient_scaled = scaler.transform(
        patient_data
    )


    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    prediction = model.predict(
        patient_scaled
    )[0]


    # -----------------------------------------
    # Probability
    # -----------------------------------------

    probability = model.predict_proba(
        patient_scaled
    )[0][1] * 100


    # -----------------------------------------
    # Result
    # -----------------------------------------

    if prediction == 1:

        result = "Stroke Risk Detected"

    else:

        result = "No Stroke Risk Detected"


    return render_template(
        "result.html",
        result=result,
        probability=round(probability, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)