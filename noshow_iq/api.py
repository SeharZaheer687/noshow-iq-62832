from flask import Flask, request, jsonify
from datetime import datetime
from pymongo import MongoClient
import os
from noshow_iq.model import predict, model_exists

app = Flask(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["noshow_iq"]
predictions_col = db["predictions"]
training_runs_col = db["training_runs"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model_exists()})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()

    import pandas as pd
    df = pd.DataFrame([data])

    if "scheduledday" in df.columns:
        df["scheduledday"] = pd.to_datetime(df["scheduledday"])
    if "appointmentday" in df.columns:
        df["appointmentday"] = pd.to_datetime(df["appointmentday"])
        df["days_in_advance"] = (
            df["appointmentday"] - df["scheduledday"]
        ).dt.days.clip(lower=0)
        df["appointment_hour"] = df["scheduledday"].dt.hour
        df.drop(
            columns=["scheduledday", "appointmentday"],
            inplace=True
        )

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"F": 0, "M": 1})

    risk, prob = predict(df)

    if prob >= 0.7:
        rec = "Send reminder SMS and call patient"
    elif prob >= 0.5:
        rec = "Send reminder SMS"
    else:
        rec = "No action needed"

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "raw_input": data,
        "risk_level": risk,
        "probability": prob,
        "recommendation": rec
    }

    predictions_col.insert_one(result.copy())

    result.pop("_id", None)
    return jsonify(result)


@app.route("/history", methods=["GET"])
def history():
    docs = list(predictions_col.find(
        {}, {"_id": 0}
    ).sort("timestamp", -1).limit(20))
    return jsonify(docs)


@app.route("/stats", methods=["GET"])
def stats():
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$risk_level", "high"]}, 1, 0
                        ]
                    }
                },
                "low_risk_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$risk_level", "low"]}, 1, 0
                        ]
                    }
                },
                "average_probability": {"$avg": "$probability"}
            }
        }
    ]
    result = list(predictions_col.aggregate(pipeline))
    last_run = training_runs_col.find_one(
        {}, {"_id": 0}, sort=[("timestamp", -1)]
    )
    if result:
        r = result[0]
        r.pop("_id", None)
        r["last_trained"] = last_run["timestamp"] if last_run else None
        return jsonify(r)
    return jsonify({"message": "No predictions yet"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
