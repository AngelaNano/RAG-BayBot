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




