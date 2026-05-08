import time
from flask import Flask, jsonify, request
from database import get_collection
from rag import answer
import pandas as pd

app = Flask(__name__)


@app.route("/api/sensors", methods=["GET"])
def get_sensors():
    start = time.time()
    collection = get_collection()
    limit = int(request.args.get("limit", 100))
    location = request.args.get("location")
    query = {}
    if location:
        query["location"] = location
    records = list(collection.find(query, {"_id": 0, "embedding": 0}).limit(limit))
    elapsed = (time.time() - start) * 1000
    print(f"/api/sensors responded in {elapsed:.2f}ms")
    return jsonify(records)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    collection = get_collection()
    records = list(collection.find({}, {"_id": 0, "embedding": 0}))
    df = pd.DataFrame(records)
    stats = {
        "avg_temperature": round(df["temperature"].mean(), 2),
        "avg_salinity": round(df["salinity"].mean(), 2),
        "avg_dissolved_oxygen": round(df["dissolved_oxygen"].mean(), 2),
        "total_records": len(df)
    }
    return jsonify(stats)


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or not data.get("question"):
        return jsonify({"error": "No question provided"}), 400
    query = data["question"].strip()
    if not query:
        return jsonify({"error": "Question cannot be empty"}), 400
    result = answer(query)
    if not result.get("grounded"):
        result["warning"] = (
            "This answer may not be fully grounded in the sensor data. "
            "Please verify against the source records below."
        )
    return jsonify(result)


@app.route("/api/test-embedding", methods=["GET"])
def test_embedding():
    import requests as r
    import os
    HF_TOKEN = os.getenv('HF_API_TOKEN')
    response = r.post(
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": "test water temperature"}
    )
    return jsonify({
        "status": response.status_code,
        "response": response.json() if response.text else "empty"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)