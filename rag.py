from database import get_collection
import requests as http_requests
import os
import time

HF_TOKEN = os.getenv('HF_API_TOKEN')
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

EMBEDDING_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
GENERATION_API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"


def get_embedding(text):
    for attempt in range(3):
        response = http_requests.post(
            EMBEDDING_API_URL,
            headers=HF_HEADERS,
            json={"inputs": text}
        )
        if response.status_code == 200 and response.text:
            result = response.json()
            if isinstance(result, list):
                if isinstance(result[0], float):
                    return result
                elif isinstance(result[0], list):
                    return result[0]
        time.sleep(5)
    return None


def generate_answer(prompt):
    for attempt in range(3):
        response = http_requests.post(
            GENERATION_API_URL,
            headers=HF_HEADERS,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 200}}
        )
        print(f"Attempt {attempt}: status={response.status_code}, text={response.text[:300]}")
        if response.status_code == 200 and response.text:
            try:
                result = response.json()
                print(f"Parsed result type: {type(result)}, value: {str(result)[:200]}")
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        return result[0].get("generated_text", "No answer generated.")
                    elif isinstance(result[0], str):
                        return result[0]
                elif isinstance(result, dict):
                    return result.get("generated_text", str(result))
            except Exception as e:
                print(f"Parse error: {e}")
        time.sleep(20)
    return "The AI model is still loading. Please try again in 30 seconds."


def retrieve(query, top_k=5):
    query_vector = get_embedding(query)
    if not query_vector:
        return []
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
    relevant_records = retrieve(query, top_k=5)
    if not relevant_records:
        return {
            "answer": "No relevant sensor data was found for your question.",
            "sources": [],
            "grounded": False
        }
    context = build_context(relevant_records)
    prompt = f"""You are a water quality analyst.
Answer ONLY using the sensor data provided below.
Include the specific dates and locations from the data in your answer.
If the data does not contain enough information to answer, say so clearly.

Sensor Data:
{context}

Question: {query}

Answer:"""
    result = generate_answer(prompt)
    is_grounded = validate_answer(result, relevant_records)
    return {
        "answer": result,
        "sources": relevant_records,
        "grounded": is_grounded
    }