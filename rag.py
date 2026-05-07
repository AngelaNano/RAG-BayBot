from sentence_transformers import SentenceTransformer
from transformers import pipeline
from database import get_collection

# ---- STEP 1: Load the embedding model ----
# This model converts text into a list of numbers (called a vector)
# "all-MiniLM-L6-v2" is a small, fast, accurate model from Hugging Face
# It loads once when the app starts — not on every question
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- STEP 2: Load the answer-generation model ----
# This is the AI that reads context and writes answers
# "google/flan-t5-base" is a good lightweight model for Q&A
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def retrieve(query, top_k=5):
    """
       Converts the user's question to a vector and uses MongoDB Atlas
       Vector Search to find the most semantically relevant sensor records.

       Why MongoDB Vector Search instead of sklearn:
       - The comparison happens inside the database before data is sent to Python
       - Only top_k records travel over the network instead of all 10,000
       - MongoDB's HNSW index makes this sub-millisecond even at millions of records
       - Matches what the resume and project documentation claim
       """
    # Convert the user's question into a vector
    # e.g. "average temperature March" becomes [0.23, -0.87, 0.45, ...]
    query_vector = embedder.encode(query).tolist()

    # Get all sensor records from MongoDB
    collection = get_collection()
    # $vectorSearch is MongoDB's native vector similarity operator
    # It runs entirely inside the database — Python receives only the results
    results = collection.aggregate([
        {
            "$vectorSearch": {
                # Must match the index name you created in Atlas exactly
                "index": "vector_index",

                # The field in each document that holds the stored vector
                "path": "embedding",

                # The query vector we just computed from the user's question
                "queryVector": query_vector,

                # How many candidates MongoDB considers internally
                # Higher = more accurate but slightly slower
                # Rule of thumb: numCandidates = 10 × limit
                "numCandidates": 50,

                # How many final results to return to Python
                "limit": top_k
            }
        },
        {
            # $project controls which fields are returned
            # 1 = include this field, 0 = exclude this field
            # We exclude "embedding" because we don't need the raw vector
            # in the response — it's large and not useful to display
            "$project": {
                "embedding": 0,  # exclude the vector (saves bandwidth)
                "description": 0,
                "_id": 0,
                # vectorSearchScore is MongoDB's similarity score for this result
                # 1.0 = perfect match, 0.0 = completely unrelated
                # We include it so we can show confidence in citations
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])

    # list() forces the aggregation cursor to execute and return all results
    # Without list(), results is a lazy cursor that hasn't run yet
    return list(results)


def build_context(records):
    """
    Formats the retrieved records into a structured context string
    that gets injected into the AI prompt.

    Why format this carefully:
    The language model reads this as plain text. The clearer and more
    structured the context is, the more accurately the model can
    answer based on it. Ambiguous formatting leads to ambiguous answers.
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

    # Join all lines with newlines so each record is on its own line
    return "\n".join(context_lines)


def validate_answer(answer, records):
    """
    Basic check to ensure the generated answer references actual data
    from the retrieved records rather than hallucinating values.

    Why this matters:
    Language models can sometimes ignore the provided context and generate
    plausible-sounding but incorrect answers from their training data.
    This check catches the most obvious cases of that.
    """
    # Extract all date strings from the retrieved records
    record_dates = [r["date"].split(" ")[0] for r in records]

    # Check if the answer mentions at least one date from the records
    # If it mentions no real dates it may be hallucinating
    mentions_real_data = any(date in answer for date in record_dates)

    return mentions_real_data


def answer(query):
    """
    Main RAG function — orchestrates the full pipeline:
    Retrieve → Augment → Generate → Validate → Return

    This is the only function called by app.py.
    All complexity is hidden inside retrieve(), build_context(),
    and validate_answer() so app.py stays clean.
    """

    # Stage 1 — RETRIEVE
    # MongoDB Vector Search returns the top 5 most relevant records
    relevant_records = retrieve(query, top_k=5)

    if not relevant_records:
        return {
            "answer": "No relevant sensor data was found for your question.",
            "sources": [],
            "grounded": False
        }

    # Stage 2 — AUGMENT
    # Format the records into a readable context block for the model
    context = build_context(relevant_records)

    # Build the full prompt
    # The instruction "Answer ONLY using the data below" is critical —
    # it tells the model to stay grounded in the retrieved records
    # and not draw on its own training knowledge
    prompt = f"""You are a water quality analyst. 
    Answer ONLY using the sensor data provided below.
    Include the specific dates and locations from the data in your answer.
    If the data does not contain enough information to answer, say so clearly.

    Sensor Data:
    {context}

    Question: {query}

    Answer:"""

    # Stage 3 — GENERATE
    # max_new_tokens limits response length — prevents runaway generation
    # do_sample=False means deterministic output — same question = same answer
    result = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )[0]["generated_text"]

    # Stage 4 — VALIDATE
    # Check whether the answer references real data from the records
    is_grounded = validate_answer(result, relevant_records)

    # Stage 5 — RETURN
    # Return the answer, the source records for citation display,
    # and the grounding flag so the frontend can warn if needed
    return {
        "answer": result,
        "sources": relevant_records,
        "grounded": is_grounded
    }