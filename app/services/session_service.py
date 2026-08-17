# =========================
# 🧠 In-memory sessions
# =========================

sessions = {}

def get_session(user_id):
    """الحصول على جلسة المستخدم"""
    session = sessions.get(user_id, {})
    if "status" not in session:
        session["status"] = "idle"
    return session

def update_session(user_id, new_data):
    """
    تحديث جلسة المستخدم بدمج البيانات الجديدة
    
    المعاملات:
        user_id (str): معرف المستخدم
        new_data (dict): البيانات الجديدة للدمج
        
    المُرجع:
        dict: الجلسة المحدثة
    """
    if user_id not in sessions:
        sessions[user_id] = {}

    # دمج البيانات (بس القيم الموجودة)
    for key, value in new_data.items():
        if value:
            # إذا كانت intents مصفوفة، ندمجها بدلاً من استبدالها
            if key == "intents" and isinstance(value, list):
                if key not in sessions[user_id]:
                    sessions[user_id][key] = []
                # إضافة النوايا الجديدة الفريدة فقط
                for intent in value:
                    if intent not in sessions[user_id][key]:
                        sessions[user_id][key].append(intent)
            else:
                sessions[user_id][key] = value

    return sessions[user_id]

def clear_session(user_id):
    """مسح جلسة المستخدم"""
    if user_id in sessions: 
        del sessions[user_id]

def set_session_status(user_id, status):
    """تعيين حالة الجلسة"""
    if user_id not in sessions:
        sessions[user_id] = {}
    sessions[user_id]["status"] = status
    return sessions[user_id]

def get_session_status(user_id):
    """الحصول على حالة الجلسة"""
    session = get_session(user_id)
    return session.get("status", "idle")

def has_pending_order(user_id):
    """
    التحقق مما إذا كان هناك طلب معلق للمعرف المستخدم
    
    المعاملات:
        user_id (str): معرف المستخدم
        
    المُرجع:
        bool: True إذا كان هناك طلب معلق
    """
    session = get_session(user_id)
    # نعتبر الطلب معلقاً إذا كانت هناك نوايا order-related
    order_intents = ["taxi", "delivery"]
    session_intents = session.get("intents", [])
    return any(intent in session_intents for intent in order_intents)