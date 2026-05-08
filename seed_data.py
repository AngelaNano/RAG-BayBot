from pymongo import MongoClient
from dotenv import load_dotenv
from database import validate_record
import os
import random
import requests
import time
from datetime import datetime, timedelta

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["baybot"]
collection = db["sensors"]

HF_TOKEN = os.getenv('HF_API_TOKEN')
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
EMBEDDING_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


def get_embeddings_batch(texts):
    """Send multiple texts at once — much faster than one at a time"""
    for attempt in range(3):
        response = requests.post(
            EMBEDDING_URL,
            headers=HF_HEADERS,
            json={"inputs": texts}
        )
        if response.status_code == 200 and response.text:
            result = response.json()
            if isinstance(result, list):
                # Returns list of vectors, one per input
                embeddings = []
                for item in result:
                    if isinstance(item, list) and isinstance(item[0], float):
                        embeddings.append(item)
                    elif isinstance(item, list):
                        embeddings.append(item[0])
                return embeddings
        time.sleep(5)
    return None


print("Clearing existing records...")
collection.delete_many({})
print("Cleared!")

print("Generating sensor records...")
all_records = []
start_date = datetime(2024, 1, 1)
locations = ["North Bay", "South Bay", "East Bay"]

# Build all records first without embeddings
for i in range(10000):
    timestamp = start_date + timedelta(hours=i)
    month = timestamp.month
    base_temp = 20 + 8 * abs((month - 6) / 6 - 1)
    temperature = round(base_temp + random.uniform(-2, 2), 2)
    salinity = round(random.uniform(28, 36), 2)
    dissolved_oxygen = round(random.uniform(6.5, 9.5), 2)
    location = random.choice(locations)

    description = (
        f"On {timestamp.strftime('%Y-%m-%d')} at {timestamp.strftime('%H:%M')}, "
        f"sensors at {location} recorded "
        f"a water temperature of {temperature} degrees Celsius, "
        f"salinity of {salinity} parts per thousand, "
        f"and dissolved oxygen of {dissolved_oxygen} milligrams per liter."
    )

    all_records.append({
        "date": timestamp.strftime("%Y-%m-%d %H:%M"),
        "temperature": temperature,
        "salinity": salinity,
        "dissolved_oxygen": dissolved_oxygen,
        "location": location,
        "description": description,
    })

# Now get embeddings in batches of 32
print("Getting embeddings in batches...")
BATCH_SIZE = 32
inserted = 0

for i in range(0, len(all_records), BATCH_SIZE):
    batch_records = all_records[i:i + BATCH_SIZE]
    descriptions = [r["description"] for r in batch_records]

    embeddings = get_embeddings_batch(descriptions)

    if not embeddings or len(embeddings) != len(batch_records):
        print(f"  Batch {i} failed — skipping")
        continue

    # Add embeddings to records
    for record, embedding in zip(batch_records, embeddings):
        record["embedding"] = embedding

    collection.insert_many(batch_records)
    inserted += len(batch_records)

    if inserted % 500 == 0:
        print(f"  Inserted {inserted} records...")

print(f"✅ Done! Inserted {inserted} records with HF API embeddings.")
print("Creating indexes...")
collection.create_index([("date", 1)])
collection.create_index([("location", 1)])
print("✅ Indexes created!")