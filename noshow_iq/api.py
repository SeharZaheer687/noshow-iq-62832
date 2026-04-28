import os
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime, timezone
from pymongo import MongoClient

from noshow_iq.model import predict, model_exists

app = Flask(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

_mongo_client = None
_db = None


def get_db():
    global _mongo_client, _db
    if _db is None:
        _mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        _db = _mongo_client["noshow_iq"]
    return _db


def get_col(name):
    return get_db()[name]


@app.route("/health", methods=["GET"])
def health():
    mongo_status = False
    try:
        get_db().command("ping")
        mongo_status = True
    except Exception:
        mongo_status = False
    return jsonify({
        "status": "ok",
        "model_loaded": model_exists(),
        "mongo": mongo_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"F": 0, "M": 1})
            if df["gender"].isnull().any():
                df["gender"] = 0

        risk, prob = predict(df)
        risk_level = "High" if risk == "high" else "Low"

        if prob >= 0.7:
            rec = "Send reminder SMS and call patient"
        elif prob >= 0.5:
            rec = "Send reminder SMS"
        else:
            rec = "No action needed"

        cleaned_features = df.to_dict(orient="records")[0]

        doc = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_input": data,
            "cleaned_features": cleaned_features,
            "risk_level": risk_level,
            "probability": round(prob, 4),
            "recommendation": rec,
        }

        try:
            get_col("predictions").insert_one(doc.copy())
        except Exception:
            pass

        doc.pop("_id", None)
        return jsonify(doc)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    try:
        docs = list(
            get_col("predictions")
            .find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(20)
        )
        return jsonify(docs)
    except Exception:
        return jsonify([])


@app.route("/stats", methods=["GET"])
def stats():
    try:
        col = get_col("predictions")

        total = col.count_documents({})
        high = col.count_documents({"risk_level": "High"})
        low = col.count_documents({"risk_level": "Low"})

        probs = [
            d["probability"]
            for d in col.find({}, {"_id": 0, "probability": 1})
        ]
        avg_prob = round(sum(probs) / len(probs), 2) if probs else 0.0

        last_run = get_col("training_runs").find_one(
            {}, {"_id": 0, "timestamp": 1},
            sort=[("timestamp", -1)]
        )
        last_trained = last_run["timestamp"] if last_run else None

        return jsonify({
            "total_predictions": total,
            "high_risk_count": high,
            "low_risk_count": low,
            "average_probability": avg_prob,
            "last_trained": last_trained,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
