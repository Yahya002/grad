import json
import logging
from app.services.llm_service import call_llm

# إعداد logging
logger = logging.getLogger(__name__)

def analyze_message(message):
    """
    تحليل الرسالة واستخراج النوايا والكيانات
    
    المعاملات:
        message (str): الرسالة النصية من المستخدم
        
    المُرجع:
        dict: يحتوي على النوايا والكيانات المستخرجة
        - intents: مصفوفة النوايا فقط (بدون intent)
    """
    logger.info(f"🔍 بدء تحليل الرسالة: {message}")
    
    raw = call_llm(message)

    try:
        data = json.loads(raw)
        logger.info(f"✅ تم تحليل JSON بنجاح")
    except:
        logger.error(f"❌ خطأ في تحليل JSON: {raw}")
        return {
            "error": "invalid_json",
            "raw": raw
        }

    # Handle both single intent (old format) and multi-intent (new format)
    if "intent" in data:
        # Convert single intent to array for consistency
        data["intents"] = [data["intent"]]
        logger.info(f"🔄 تحويل intent واحد إلى intents array: {data['intent']}")
    elif "intents" not in data:
        data["intents"] = ["unknown"]
        logger.warning("⚠️ لم يتم العثور على intents، استخدام unknown")
    else:
        logger.info(f"✅ تم استلام مجموعة نوايا: {data['intents']} ({len(data['intents'])} نوايا)")
    
    # إزالة intent (النية الوحيدة) - نستخدم intents array فقط
    if "intent" in data:
        del data["intent"]
        logger.info(f"🗑️ إزالة intent للحفاظ على intents array فقط")
    
    return data