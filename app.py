import time

from flask import Flask, jsonify, request
from database import get_collection
from rag import answer
import pandas as pd

# Create the Flask app — think of this as turning the server on
app = Flask(__name__)

# ---- ENDPOINT 1: Get all sensor data ----
# @app.route defines a URL path
# When someone visits /api/sensors, this function runs
@app.route("/api/sensors", methods=["GET"])
def get_sensors():
    """
        Returns sensor records with optional filtering.
        Excludes the embedding field — it's large and not useful to the frontend.
        """

    start = time.time()

    collection = get_collection()
    limit = int(request.args.get("limit", 100))
    location = request.args.get("location")

    # Build the query filter dynamically
    # If no location is provided, query is empty — returns all locations
    query = {}
    if location:
        query["location"] = location

    # embedding: 0 excludes the vector field from results
    # Sending 384 floats per record across the network is wasteful
    # when the frontend only needs the human-readable fields
    records = list(collection.find(query, {"_id": 0, "embedding": 0}).limit(limit))

    elapsed = (time.time() - start) * 1000
    print(f"/api/sensors responded in {elapsed:.2f}ms")

    return jsonify(records)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Returns aggregate summary statistics across all records.
    Uses pandas for computation rather than MongoDB aggregation
    to keep the code readable and easy to extend.
    """
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
    """
    Accepts a natural-language question and runs the full RAG pipeline.
    Returns the generated answer, source citations, and a grounding flag.

    Why POST and not GET:
    GET requests send data in the URL — visible in browser history and logs.
    POST sends data in the request body — more appropriate for user queries
    which may contain sensitive or personal information.
    """
    data = request.json

    # Validate that a question was actually provided
    if not data or not data.get("question"):
        # 400 Bad Request — standard HTTP code for malformed input
        return jsonify({"error": "No question provided"}), 400

    query = data["question"].strip()

    # Reject empty strings that passed the first check
    if not query:
        return jsonify({"error": "Question cannot be empty"}), 400

    # Run the full RAG pipeline
    result = answer(query)

    # If the answer validation check failed, add a warning
    # The answer is still returned — the frontend decides how to display it
    if not result.get("grounded"):
        result["warning"] = (
            "This answer may not be fully grounded in the sensor data. "
            "Please verify against the source records below."
        )

    return jsonify(result)

# This runs the server — debug=True means it auto-restarts when you save
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


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

