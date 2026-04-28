import os
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
from pymongo import MongoClient
from noshow_iq.model import predict, model_exists

app = Flask(__name__)
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
mongo_ok = False
predictions_col = None
training_runs_col = None
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db = client["noshow_iq"]
    predictions_col = db["predictions"]
    training_runs_col = db["training_runs"]
    mongo_ok = True
except Exception:
    mongo_ok = False

MODEL_FEATURES = ["gender","age","scholarship","hipertension","diabetes","alcoholism","handcap","sms_received","days_in_advance","appointment_hour"]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model_exists(), "mongo": mongo_ok})

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        col_map = {"Gender":"gender","Age":"age","Scholarship":"scholarship","Hipertension":"hipertension","Diabetes":"diabetes","Alcoholism":"alcoholism","Handcap":"handcap","SMS_received":"sms_received","ScheduledDay":"scheduled_day","AppointmentDay":"appointment_day","Neighbourhood":"neighbourhood","PatientId":"patient_id","AppointmentID":"appointment_id"}
        df = df.rename(columns=col_map)
        if "scheduled_day" in df.columns and "appointment_day" in df.columns:
            df["scheduled_day"] = pd.to_datetime(df["scheduled_day"])
            df["appointment_day"] = pd.to_datetime(df["appointment_day"])
            df["days_in_advance"] = (df["appointment_day"] - df["scheduled_day"]).dt.days
            df["appointment_hour"] = df["appointment_day"].dt.hour
        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"F": 0, "M": 1}).fillna(0)
        cleaned = df.to_dict(orient="records")[0]
        df = df[MODEL_FEATURES]
        risk, prob = predict(df)
        if prob >= 0.7:
            rec = "Send reminder SMS and call patient"
        elif prob >= 0.5:
            rec = "Send reminder SMS"
        else:
            rec = "No action needed"
        result = {"timestamp": datetime.utcnow().isoformat(), "raw_input": data, "cleaned_features": {k: str(v) for k,v in cleaned.items()}, "risk_level": risk, "probability": prob, "recommendation": rec}
        if mongo_ok and predictions_col is not None:
            predictions_col.insert_one(result.copy())
        result.pop("_id", None)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def history():
    if not mongo_ok or predictions_col is None:
        return jsonify([])
    docs = list(predictions_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
    return jsonify(docs)

@app.route("/stats", methods=["GET"])
def stats():
    if not mongo_ok or predictions_col is None:
        return jsonify({"message": "MongoDB not connected"})
    pipeline = [{"\$group": {"_id": None, "total_predictions": {"\$sum": 1}, "high_risk_count": {"\$sum": {"\$cond": [{"\$eq": ["\$risk_level", "high"]}, 1, 0]}}, "low_risk_count": {"\$sum": {"\$cond": [{"\$eq": ["\$risk_level", "low"]}, 1, 0]}}, "average_probability": {"\$avg": "\$probability"}}}]
    result = list(predictions_col.aggregate(pipeline))
    last_run = training_runs_col.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
    if result:
        r = result[0]
        r.pop("_id", None)
        r["last_trained"] = last_run["timestamp"] if last_run else None
        return jsonify(r)
    return jsonify({"message": "No predictions yet"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
