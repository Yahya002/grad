from pymongo import MongoClient
from app.config import MONGO_URI, DB_NAME

client = None
db = None

def init_db(app):
    global client, db
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

def get_db():
    return db