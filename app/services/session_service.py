# =========================
# 🧠 In-memory sessions
# =========================

sessions = {}

def get_session(user_id):
    return sessions.get(user_id, {})

def update_session(user_id, new_data):
    if user_id not in sessions:
        sessions[user_id] = {}

    # دمج البيانات (بس القيم الموجودة)
    for key, value in new_data.items():
        if value:
            sessions[user_id][key] = value

    return sessions[user_id]

def clear_session(user_id):
    if user_id in sessions: 
        del sessions[user_id]