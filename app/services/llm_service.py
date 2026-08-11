import json
import requests
from app.config import MODEL_NAME, MODEL_TYPE, HF_API_KEY

# ==================== GLOBAL VARIABLES ====================
# النموذج والمُرمِّز المحلي (يُحمّل مرة واحدة فقط)
_model = None
_tokenizer = None

# ==================== AVAILABLE INTENTS ====================
# النوايا المتاحة في النظام
INTENTS = [
    "greeting",       # ترحيب
    "thank",          # شكر
    "taxi",           # طلب تكسي
    "delivery",       # طلب توصيل
    "ask_directions", # سؤال عن الاتجاهات
    "ask_product",    # سؤال عن منتج
    "cancel",         # إلغاء
    "unknown"         # غير معروف
]

# ==================== SYSTEM PROMPT ====================
# التعليمات التي تُرسل للنموذج لتحديد مهمته
SYSTEM_PROMPT = """
أنت نظام تصنيف واستخراج بيانات.

مهمتك:
1. حدد intents (يمكن أن يكون أكثر من نية) من:
- greeting (ترحيب)
- thank (شكر)
- taxi (طلب تكسي)
- delivery (طلب توصيل)
- ask_directions (سؤال عن الاتجاهات)
- ask_product (سؤال عن منتج)
- cancel (إلغاء)
- unknown (غير معروف)

2. استخرج البيانات الموجودة فعلاً في الرسالة فقط:
استخرج فقط المعلومات المذكورة حرفياً.

ممنوع التخمين.

إذا لم تجد معلومة:
ضع null.

If the user did not explicitly mention a place, person, or phone number,
the value MUST be null.

Do not use values from previous examples.
Do not invent likely destinations.

الرد يجب أن يكون JSON فقط بدون أي نص إضافي.
"""

# ==================== API CALLING FUNCTIONS ====================

def call_huggingface_router_api(message):
    """
    استدعاء Hugging Face Router API للنماذج الأونلاين
    API URL: https://router.huggingface.co/v1/chat/completions
    
    هذه الدالة تُستخدم للتجربة مع نماذج أونلاين قبل تدريب AraBERT
    
    المعاملات:
        message (str): الرسالة النصية من المستخدم
        
    المُرجع:
        str: JSON string يحتوي على النوايا والكيانات المستخرجة
    """
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Message: {message}"
            }
        ],
        "temperature": 0.1,  # قيمة منخفضة للحصول على نتائج أكثر دقة
        "max_tokens": 200,   # الحد الأقصى للرموز المُولّدة
        "response_format": {"type": "json_object"}  # فرض إرجاع JSON
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # استخراج المحتوى المُولّد
        content = result["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        return json.dumps({
            "error": "huggingface_router_api_error",
            "message": str(e)
        })

def load_model():
    """
    تحميل نموذج AraBERT المحلي والمُرمِّز
    هذه الدالة تُستخدم فقط عند استخدام MODEL_TYPE = "transformers"
    
    ⚠️ ملاحظة للتطوير المستقبلي ⚠️
    عند تدريب النموذج، يجب تعديل هذه الدالة لتحميل:
    - النموذج المُدرَّب مع classification head
    - النموذج المُدرَّب مع NER head
    
    المُرجع:
        tuple: (model, tokenizer) النموذج والمُرمِّز
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
    return _model, _tokenizer

# ==================== MAIN LLM CALL FUNCTION ====================

def call_llm(message):
    """
    الدالة الرئيسية لاستدعاء النموذج بناءً على MODEL_TYPE المُحدد في config.py
    
    الخيارات المتاحة:
    1. "huggingface_router" - استخدام Hugging Face Router API (أونلاين)
    2. "transformers" - استخدام نموذج AraBERT المحلي (بعد التدريب)
    
    المعاملات:
        message (str): الرسالة النصية من المستخدم
        
    المُرجع:
        str: JSON string بصيغة:
        {
            "intents": ["intent1", "intent2"],  // مصفوفة النوايا
            "intent": "intent1",                // النية الأساسية (للتوافق)
            "from": "location",                 // موقع الانطلاق
            "to": "destination",                // الواجهة
            "name": "user_name",               // اسم المستخدم
            "phone": "phone_number"             // رقم الهاتف
        }
    """
    if MODEL_TYPE == "huggingface_router":
        # ============================================
        # استخدام Hugging Face Router API
        # ============================================
        raw_output = call_huggingface_router_api(message)
        
        # محاولة تحليل JSON من المخرجات
        try:
            data = json.loads(raw_output)
            
            # التأكد من أن intents مصفوفة فقط
            if "intent" in data and "intents" not in data:
                data["intents"] = [data["intent"]]
            elif "intents" not in data:
                data["intents"] = ["unknown"]
            
            # إزالة intent (النية الوحيدة) - نستخدم intents array فقط
            if "intent" in data:
                del data["intent"]
            
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps({
                "error": "invalid_json",
                "raw": raw_output
            })
    
    elif MODEL_TYPE == "transformers":
        # ============================================
        # استخدام نموذج AraBERT المحلي
        # ============================================
        
        # ============================================
        # ⚠️ ملاحظة هامة للتطوير المستقبلي ⚠️
        # ============================================
        # هذا الكود حالياً يُرجع placeholder values
        # عند تدريب النموذج، يجب تعديل هذا الجزء لـ:
        # 
        # 1. تحميل النموذج المُدرَّب:
        #    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
        #    intent_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        #    ner_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        #
        # 2. تصنيف النوايا:
        #    intent_outputs = intent_model(**inputs)
        #    intent_predictions = torch.argmax(intent_outputs.logits, dim=-1)
        #    intent_label = INTENTS[intent_predictions[0].item()]
        #
        # 3. استخراج الكيانات (NER):
        #    ner_outputs = ner_model(**inputs)
        #    ner_predictions = torch.argmax(ner_outputs.logits, dim=-1)
        #    # معالجة predictions لاستخراج from, to, name, phone
        #
        # 4. معالجة مخرجات النموذج المُدرَّب
        # ============================================
        
        model, tokenizer = load_model()
        
        # ترميز النص المدخل
        inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
        
        # الحصول على مخرجات النموذج
        outputs = model(**inputs)
        
        # هيكل الاستجابة (سيتم استبداله بعد التدريب)
        response = {
            "intents": ["unknown"],  # سيتم التنبؤ به من classification head
            "from": None,            # سيتم استخراجه من NER head
            "to": None,              # سيتم استخراجه من NER head
            "name": None,            # سيتم استخراجه من NER head
            "phone": None,           # سيتم استخراجه من NER head
            "raw_embeddings": outputs.last_hidden_state.tolist()  # للتدريب والتصحيح
        }
        
        return json.dumps(response, ensure_ascii=False)
    
    else:
        return json.dumps({
            "error": "invalid_model_type",
            "message": f"Unknown MODEL_TYPE: {MODEL_TYPE}"
        })
