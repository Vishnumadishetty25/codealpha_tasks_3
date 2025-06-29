from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model components
model_bundle = joblib.load("disease_model_bundle.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
label_encoder = model_bundle["label_encoder"]
df_model = pd.read_json("patient_data.json")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        patient_id = int(request.form["patient_id"])
        patient = df_model[df_model["id"] == patient_id]

        if patient.empty:
            return render_template("index.html", prediction_text=f"No patient found with ID {patient_id}")

        input_features = patient.drop(columns=["id"])
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]
        disease = label_encoder.inverse_transform([prediction])[0]

        details = "<br>".join([f"<strong>{col}</strong>: {patient.iloc[0][col]}" for col in patient.columns])
        return render_template("index.html", prediction_text=f"🩺 Predicted Disease: <strong>{disease}</strong><br><br>{details}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)