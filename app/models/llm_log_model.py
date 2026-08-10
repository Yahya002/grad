# app/models/llm_log_model.py

from app.database.mongo import get_db
from datetime import datetime

def save_llm_log(
    user_id,
    message,
    response,
    expected=None,
    source="chat"
):
    db = get_db()

    db.llm_logs.insert_one({
        "user_id": user_id,
        "message": message,
        "response": response,
        "expected": expected,
        "source": source,
        "validated": False,
        "corrected": None,
        "timestamp": datetime.utcnow()
    })