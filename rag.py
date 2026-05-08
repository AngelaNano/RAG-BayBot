from database import get_collection
import requests as http_requests
import os
import numpy as np

HF_TOKEN = os.getenv('HF_API_TOKEN')
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
GENERATION_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"


def get_embedding(text):
    response = http_requests.post(
        EMBEDDING_API_URL,
        headers=HF_HEADERS,
        json={"inputs": text}
    )
    result = response.json()
    if isinstance(result, list):
        embedding = result[0]
        if isinstance(embedding[0], list):
            embedding = embedding[0]
        return embedding
    return None


def generate_answer(prompt):
    response = http_requests.post(
        GENERATION_API_URL,
        headers=HF_HEADERS,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    )
    result = response.json()
    if isinstance(result, list):
        return result[0].get("generated_text", "No answer generated.")
    return "No answer generated."


def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    Measures the angle between them — closer to 1.0 means more similar.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def retrieve(query, top_k=5):
    """
    Converts query to vector via HF API, then compares against
    stored embeddings in MongoDB using cosine similarity.
    """
    query_vector = get_embedding(query)

    if not query_vector:
        return []

    collection = get_collection()

    # Fetch all records with their stored embeddings
    records = list(collection.find(
        {},
        {"_id": 0, "date": 1, "temperature": 1, "salinity": 1,
         "dissolved_oxygen": 1, "location": 1, "embedding": 1}
    ))

    # Compute cosine similarity between query and every record
    for record in records:
        record["score"] = cosine_similarity(query_vector, record["embedding"])

    # Sort by score descending and return top_k
    records.sort(key=lambda x: x["score"], reverse=True)
    top_records = records[:top_k]

    # Remove embedding from returned records — not needed by frontend
    for record in top_records:
        record.pop("embedding", None)

    return top_records


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