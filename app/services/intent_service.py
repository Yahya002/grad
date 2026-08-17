import json
import logging
from app.services.ml_service import call_llm

logger = logging.getLogger(__name__)

def analyze_message(message, session_status="idle"):
    """
    تحليل الرسالة واستخراج النوايا والكيانات.

    Args:
        message: User message text
        session_status: Current session status for location resolution

    Returns:
        dict:
        {
            "intents": [...],
            ...
        }
    """
    logger.info(f"🔍 بدء تحليل الرسالة: {message}")

    raw = call_llm(message, session_status)

    # call_llm may return either a dict or a JSON string.
    if isinstance(raw, dict):
        data = raw
        logger.info("✅ تم استلام نتيجة منظمة مباشرة من نموذج NLP")

    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
            logger.info("✅ تم تحليل JSON بنجاح")
        except json.JSONDecodeError:
            logger.error(f"❌ خطأ في تحليل JSON: {raw}")
            return {
                "error": "invalid_json",
                "raw": raw
            }

    else:
        logger.error(f"❌ نوع نتيجة غير متوقع: {type(raw)}")
        return {
            "error": "invalid_result_type",
            "raw": raw
        }

    # Handle old single-intent format.
    if "intent" in data:
        data["intents"] = [data["intent"]]
        del data["intent"]

        logger.info(
            f"🔄 تحويل intent واحد إلى intents array: {data['intents']}"
        )

    # Handle missing intents.
    elif "intents" not in data:
        data["intents"] = ["unknown"]
        logger.warning(
            "⚠️ لم يتم العثور على intents، استخدام unknown"
        )

    else:
        logger.info(
            f"✅ تم استلام مجموعة نوايا: "
            f"{data['intents']} ({len(data['intents'])} نوايا)"
        )

    return data