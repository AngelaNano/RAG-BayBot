from pymongo import MongoClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import random
from datetime import datetime, timedelta

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["baybot"]
collection = db["sensors"]

# Load the embedding model once before the loop
# Loading it inside the loop would reload it 10,000 times — extremely slow
# This model converts text into a 384-dimensional vector
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating sensor records...")
records = []
start_date = datetime(2024, 1, 1)

for i in range(10000):
    timestamp = start_date + timedelta(hours=i)
    month = timestamp.month
    base_temp = 20 + 8 * abs((month - 6) / 6 - 1)

    # Build a plain English description of each record
    # This is what gets converted to a vector and searched against
    # The richer this text is, the better the semantic search will be
    # We include all key fields so any question about them can match
    description = (
        f"On {timestamp.strftime('%Y-%m-%d')} at {timestamp.strftime('%H:%M')}, "
        f"sensors at {random.choice(['North Bay', 'South Bay', 'East Bay'])} recorded "
        f"a water temperature of {round(base_temp + random.uniform(-2, 2), 2)} degrees Celsius, "
        f"salinity of {round(random.uniform(28, 36), 2)} parts per thousand, "
        f"and dissolved oxygen of {round(random.uniform(6.5, 9.5), 2)} milligrams per liter."
    )

    # Convert the description to a vector right now at insert time
    # .tolist() converts numpy array to a plain Python list
    # MongoDB cannot store numpy arrays — only plain Python lists
    embedding = embedder.encode(description).tolist()

    record = {
        "date": timestamp.strftime("%Y-%m-%d %H:%M"),
        "temperature": round(base_temp + random.uniform(-2, 2), 2),
        "salinity": round(random.uniform(28, 36), 2),
        "dissolved_oxygen": round(random.uniform(6.5, 9.5), 2),
        "location": random.choice(["North Bay", "South Bay", "East Bay"]),
        "description": description,

        # This is the new field — the vector lives inside the document
        # MongoDB Vector Search will query this field directly
        "embedding": embedding
    }
    records.append(record)

    # Print progress every 1000 records so you know it's working
    if (i + 1) % 1000 == 0:
        print(f"  Generated {i + 1} records...")

# Insert all records at once — far faster than inserting one at a time
# insert_many() sends one network request instead of 10,000
print("Inserting into MongoDB...")
collection.insert_many(records)
print(f"✅ Inserted {len(records)} records with embeddings into MongoDB!")