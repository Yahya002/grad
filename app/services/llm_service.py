import requests
from app.config import OLLAMA_URL, MODEL_NAME

SYSTEM_PROMPT = """
أنت نظام تصنيف واستخراج بيانات.

مهمتك:
1. حدد intent فقط من:
- taxi
- delivery
- ask_directions
- ask_product
- greeting
- cancel
- unknown

2. استخرج البيانات الموجودة فعلاً في الرسالة فقط:
استخرج فقط المعلومات المذكورة حرفياً.

ممنوع التخمين.

إذا لم تجد معلومة:
ضع null.
Never infer missing values.

If the user did not explicitly mention a place, person, or phone number,
the value MUST be null.

Do not use values from previous examples.
Do not invent likely destinations.
"""

def call_llm(message):
    res = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": SYSTEM_PROMPT + f"\nMessage: {message}",
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string"
                },
                "from": {
                    "type": ["string", "null"]
                },
                "to": {
                    "type": ["string", "null"]
                },
                "name": {
                    "type": ["string", "null"]
                },
                "phone": {
                    "type": ["string", "null"]
                }
            },
            "required": [
                "intent",
                "from",
                "to",
                "name",
                "phone"
            ]
        },
        "options": {
            "temperature": 0
        }
    })

    return res.json()["response"]
