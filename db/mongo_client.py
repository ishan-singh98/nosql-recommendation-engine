import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise ValueError("MONGO_URI not found in .env")
        _client = MongoClient(uri)
    return _client

def get_db(db_name="ecommerce_reco"):
    return get_client()[db_name]