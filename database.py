from pymongo import MongoClient # import the MongoDB library
import os # lets us read environment variables
from dotenv import load_dotenv # reads our .env file

load_dotenv() # this line loads your .env file so os.getenv works below

def get_database():
    # os.getenv reads the MONGO_URI value from your .env file
    # This keeps your password out of your code
    client = MongoClient(os.getenv("MONGO_URI"))

    # "baybot" is the name of your database inside MongoDB
    return client["baybot"]

def get_collection():
    db = get_database()

    # "sensors" is the name of the table (called a collection in MongoDB)
    # This is where all 10,000 sensor records live
    return db["sensors"]

def validate_record(record):
    required_fields = ["date", "temperature", "salinity",
                       "dissolved_oxygen", "location"]

    for field in required_fields:
        if field not in record:
            raise ValueError(f"Missing required field: {field}")

    numeric_fields = ["temperature", "salinity", "dissolved_oxygen"]
    for field in numeric_fields:
        if not isinstance(record[field], (int, float)):
            raise ValueError(f"{field} must be a number")

    return True



