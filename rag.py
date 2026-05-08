from database import get_collection
from sentence_transformers import SentenceTransformer
import requests as http_requests
import os

# Embedding model runs locally — only 90MB, fits in Render's free tier
# Only the generation model uses HF API to avoid loading torch locally
embedder = SentenceTransformer("all-MiniLM-L6-v2")

HF_TOKEN = os.getenv('HF_API_TOKEN')
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
GENERATION_API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"


def get_embedding(text):
    """
    Converts text to a 384-dimensional vector using the local
    sentence-transformers model. Runs on Render's server directly.
    Matches the vectors stored in MongoDB at seed time.
    """
    return embedder.encode(text).tolist()


def generate_answer(prompt):
    """
    Calls Hugging Face API to generate a natural language answer.
    Runs on HF servers — Render never loads torch or flan-t5 into memory.
    Retries up to 3 times in case the model is cold starting.
    """
    import time
    for attempt in range(3):
        response = http_requests.post(
            GENERATION_API_URL,
            headers=HF_HEADERS,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 200}}
        )
        if response.status_code == 503 or not response.text:
            time.sleep(20)
            continue
        try:
            result = response.json()
            if isinstance(result, list):
                return result[0].get("generated_text", "No answer generated.")
        except Exception:
            time.sleep(20)
            continue
    return "The AI model is still loading. Please try again in 30 seconds."


def retrieve(query, top_k=5):
    """
    Converts query to vector using local embedding model,
    then runs MongoDB Atlas Vector Search to find the most
    semantically relevant sensor records.
    """
    query_vector = get_embedding(query)
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
    """
    Formats retrieved records into a structured context string
    that gets injected into the AI prompt.
    """
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
    """
    Checks if the generated answer references actual dates
    from the retrieved records to verify grounding.
    """
    record_dates = [r["date"].split(" ")[0] for r in records]
    return any(date in answer_text for date in record_dates)


def answer(query):
    """
    Main RAG function — orchestrates the full pipeline:
    Retrieve → Augment → Generate → Validate → Return
    """
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