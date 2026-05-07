from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

try:
    client = MongoClient(os.getenv("MONGO_URI"))
    client.admin.command("ping")
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ Connection failed: {e}")