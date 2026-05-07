from sentence_transformers import SentenceTransformer
from database import get_collection
import requests as http_requests
import os

# Load embedding model once at startup
# This is small (90MB) and fine for Render's free tier
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Hugging Face Inference API — model runs on HF servers, not Render
# This means Render never loads torch or the full model into memory
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}


def generate_answer(prompt):
    response = http_requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    )
    result = response.json()
    if isinstance(result, list):
        return result[0].get("generated_text", "No answer generated.")
    return "No answer generated."


def retrieve(query, top_k=5):
    query_vector = embedder.encode(query).tolist()
    collection = get_collection()

    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": top_k
            }
        },
        {
            "$project": {
                "embedding": 0,
                "description": 0,
                "_id": 0,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])

    return list(results)


def build_context(records):
    context_lines = []
    for record in records:
        line = (
            f"Date: {record['date']} | "
            f"Location: {record['location']} | "
            f"Temperature: {record['temperature']}°C | "
            f"Salinity: {record['salinity']} ppt | "
            f"Dissolved Oxygen: {record['dissolved_oxygen']} mg/L | "
            f"Relevance Score: {record['score']:.3f}"
        )
        context_lines.append(line)
    return "\n".join(context_lines)


def validate_answer(answer_text, records):
    record_dates = [r["date"].split(" ")[0] for r in records]
    return any(date in answer_text for date in record_dates)


def answer(query):
    # Stage 1 — RETRIEVE
    relevant_records = retrieve(query, top_k=5)

    if not relevant_records:
        return {
            "answer": "No relevant sensor data was found for your question.",
            "sources": [],
            "grounded": False
        }

    # Stage 2 — AUGMENT
    context = build_context(relevant_records)

    prompt = f"""You are a water quality analyst.
Answer ONLY using the sensor data provided below.
Include the specific dates and locations from the data in your answer.
If the data does not contain enough information to answer, say so clearly.

Sensor Data:
{context}

Question: {query}

Answer:"""

    # Stage 3 — GENERATE
    result = generate_answer(prompt)

    # Stage 4 — VALIDATE
    is_grounded = validate_answer(result, relevant_records)

    # Stage 5 — RETURN
    return {
        "answer": result,
        "sources": relevant_records,
        "grounded": is_grounded
    }