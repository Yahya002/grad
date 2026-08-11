MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "taxi_bot"

# ==================== MODEL CONFIGURATION ====================
# اختر واحد من الخيارات التالية:

# الخيار 1: استخدام Hugging Face Router API (للتجربة الأونلاين)
MODEL_TYPE = "huggingface_router"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"  # نموذج من Hugging Face
HF_API_KEY = "hf_api_key"  # احصل عليه من https://huggingface.co/settings/tokens

# الخيار 2: استخدام نموذج AraBERT المحلي (بعد التدريب)
# MODEL_TYPE = "transformers"
# MODEL_NAME = "aubmindlab/bert-base-arabertv02"  # أو مسار النموذج المُدرَّب الخاص بك

# الخيار 3: استخدام OpenRouter API
# MODEL_TYPE = "openrouter"
# MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"

# ==================== OLLAMA (الخيار القديم) ====================
# OLLAMA_URL = "http://192.168.100.87:11434/api/generate"
# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "mistral"