import json
from app.services.llm_service import call_llm

def analyze_message(message):
    raw = call_llm(message)

    try:
        data = json.loads(raw)
    except:
        return {
            "error": "invalid_json",
            "raw": raw
        }

    return data