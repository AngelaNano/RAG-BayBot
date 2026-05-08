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

def get_global_stats():
    """
    Computes summary statistics across ALL records in MongoDB using
    the aggregation pipeline — without pulling any records into Python.
    """
    collection = get_collection()

    pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_temperature": {"$avg": "$temperature"},
                "avg_salinity": {"$avg": "$salinity"},
                "avg_dissolved_oxygen": {"$avg": "$dissolved_oxygen"},
                "total_records": {"$sum": 1},
                "max_temperature": {"$max": "$temperature"},
                "min_temperature": {"$min": "$temperature"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "avg_temperature": {"$round": ["$avg_temperature", 2]},
                "avg_salinity": {"$round": ["$avg_salinity", 2]},
                "avg_dissolved_oxygen": {"$round": ["$avg_dissolved_oxygen", 2]},
                "total_records": 1,
                "max_temperature": {"$round": ["$max_temperature", 2]},
                "min_temperature": {"$round": ["$min_temperature", 2]},
            }
        }
    ]

    result = list(collection.aggregate(pipeline))
    return result[0] if result else {}


