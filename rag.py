from database import get_collection
import requests as http_requests
import os
import time

HF_TOKEN = os.getenv('HF_API_TOKEN')
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

EMBEDDING_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
GENERATION_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"


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
            json={"inputs": prompt}
        )
        print(f"Attempt {attempt}: status={response.status_code}")
        if response.status_code == 200 and response.text:
            try:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if "summary_text" in result[0]:
                        text = result[0]["summary_text"]
                        # Remove the prompt prefix if it appears in the response
                        if "Answer:" in text:
                            text = text.split("Answer:")[-1].strip()
                        elif "say so clearly." in text:
                            text = text.split("say so clearly.")[-1].strip()
                        return text
                    elif "generated_text" in result[0]:
                        return result[0]["generated_text"]
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
    prompt = f"""Use the sensor data provided below to answer this question. Include the specific dates and locations from the data in your answer. If the data does not contain enough information to answer, say so clearly.

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