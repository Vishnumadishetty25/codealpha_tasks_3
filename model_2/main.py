import tkinter as tk
from tkinter import messagebox
import json
import joblib
import pandas as pd

# === Load model and data ===
try:
    model_bundle = joblib.load("disease_model_bundle.pkl")
    with open("patient_data.json", "r") as f:
        patient_data = json.load(f)
except Exception as e:
    print("Error loading model or data:", e)
    exit()

# Unpack components
model = model_bundle["model"]
scaler = model_bundle["scaler"]
label_encoder = model_bundle["label_encoder"]
df_model = pd.DataFrame(patient_data)

# === GUI Function ===


def predict_disease():
    try:
        patient_id = int(patient_id_var.get())
        patient = df_model[df_model["id"] == patient_id]

        if patient.empty:
            messagebox.showerror(
                "Not Found", f"No patient with ID {patient_id}")
            return

        input_features = patient.drop(columns=["id"])
        input_scaled = scaler.transform(input_features)
        pred = model.predict(input_scaled)[0]
        disease_name = label_encoder.inverse_transform([pred])[0]

        details = "\n".join(
            [f"{col}: {patient.iloc[0][col]}" for col in patient.columns])
        messagebox.showinfo(
            "Prediction", f"🩺 Predicted Disease: {disease_name}\n\nPatient Info:\n{details}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# === GUI Layout ===
root = tk.Tk()
root.title("Disease Predictor by Patient ID")

tk.Label(root, text="Enter Patient ID:").grid(
    row=0, column=0, padx=10, pady=10)
patient_id_var = tk.StringVar()
tk.Entry(root, textvariable=patient_id_var).grid(
    row=0, column=1, padx=10, pady=10)

tk.Button(root, text="Predict Disease", command=predict_disease).grid(
    row=1, column=0, columnspan=2, pady=10)

root.mainloop()
