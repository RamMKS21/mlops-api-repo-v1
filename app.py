from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/health")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
