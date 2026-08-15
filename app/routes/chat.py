from flask import Blueprint, request, jsonify, render_template
from app.database.mongo import get_db
from app.services.intent_service import analyze_message
from app.services.session_service import get_session, update_session, clear_session, has_pending_order
from app.utils.validator import validate
from app.models.order_model import save_order
from bson.objectid import ObjectId

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat-view")
def index():
    return render_template("chat.html")

@chat_bp.route("/orders", methods=["GET"])
def get_orders():
    db = get_db()
    orders = list(db.orders.find({}, {"_id": 0}))
    return jsonify(orders)

@chat_bp.route("/corrections")
def corrections():
    return render_template("corrections.html")

@chat_bp.route("/llm_logs", methods=["GET"])
def get_llm_logs():

    db = get_db()

    logs = list(db.llm_logs.find({}))

    for log in logs:
        log["_id"] = str(log["_id"])

    return jsonify(logs)

@chat_bp.route("/llm_logs_unvalidated", methods=["GET"])
def get_unvalidated_logs():

    db = get_db()

    logs = list(
        db.llm_logs.find({
            "validated": False
        })
    )

    for log in logs:
        log["_id"] = str(log["_id"])

    return jsonify(logs)

@chat_bp.route("/llm_logs/<id>", methods=["PATCH"])
def update_llm_log(id):

    data = request.json

    db = get_db()

    db.llm_logs.update_one(
        {
            "_id": ObjectId(id)
        },
        {
            "$set": {
                "corrected": data["corrected"],
                "validated": True
            }
        }
    )

    return jsonify({
        "success": True
    })

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    user_id = data.get("user_id", "default")

    ai_result = analyze_message(message)
    from app.models.llm_log_model import save_llm_log
    save_llm_log(user_id, message, ai_result)
    print("AI RESULT:", ai_result)
    if "error" in ai_result:
        return jsonify(ai_result)

    intents = ai_result.get("intents", [])
    
    # =========================
    # 🧠 التحقق من وجود طلب معلق
    # =========================
    has_pending = has_pending_order(user_id)
    print(f"🔍 طلب معلق للمستخدم {user_id}: {has_pending}")
    
    # =========================
    # 🎯 معالجة النوايا المتعددة بالترتيب
    # =========================
    replies = []
    session_updated = False
    order_saved = False
    
    # إذا كان هناك طلب معلق، نحدث الجلسة دائماً بغض النظر عن النوايا
    if has_pending:
        print("📝 تحديث الجلسة المعلقة")
        session = update_session(user_id, ai_result)
        session_updated = True
        
        # التحقق من اكتمال البيانات
        missing = validate(session)
        
        if not missing:
            # =========================
            # 💾 Save order
            # =========================
            clean_order = {
                "from": session.get("from"),
                "to": session.get("to"),
                "name": session.get("name"),
                "phone": session.get("phone"),
                "intents": session.get("intents", []),
                "status": "pending"
            }

            save_order(clean_order)
            clear_session(user_id)
            order_saved = True
            replies.append("تم تسجيل الطلب ✅")
        else:
            replies.append(f"محتاج: {', '.join(missing)}")
    
    # معالجة النوايا الحالية
    for intent in intents:
        print(f"🎯 معالجة النية: {intent}")
        
        # =========================
        # ❌ Cancel
        # =========================
        if intent == "cancel":
            clear_session(user_id)
            replies.append("تم إلغاء الطلب ❌")
            order_saved = False  # إلغاء حفظ الطلب
        
        # =========================
        # � Greeting
        # =========================
        elif intent == "greeting":
            if not has_pending:  # لا نرد بالترحيب إذا كان هناك طلب معلق
                replies.append("أهلاً وسهلاً �")
        
        # =========================
        # 🙏 Thank
        # =========================
        elif intent == "thank":
            if not has_pending:
                replies.append("عفواً! أنا هنا للمساعدة 😊")
        
        # =========================
        # ❓ Information requests (no session needed)
        # =========================
        elif intent == "ask_directions":
            if not has_pending:
                replies.append("أنا مساعد الطلبات، يمكنني مساعدتك في طلب تكسي أو توصيل. كيف يمكنني مساعدتك؟")
        
        elif intent == "ask_product":
            if not has_pending:
                replies.append("نحن نقدم خدمة التاكسي والتوصيل. يمكنك طلب تكسي أو توصيل طلباتك.")
        
        # =========================
        # 🧠 Session merge (for order-related intents)
        # =========================
        elif intent in ["taxi", "delivery"]:
            if not session_updated:  # لم يتم تحديث الجلسة بعد
                session = update_session(user_id, ai_result)
                session_updated = True
                
                # =========================
                # 🔍 Validation
                # =========================
                missing = validate(session)

                if missing:
                    replies.append(f"محتاج: {', '.join(missing)}")
                else:
                    # =========================
                    # 💾 Save order
                    # =========================
                    clean_order = {
                        "from": session.get("from"),
                        "to": session.get("to"),
                        "name": session.get("name"),
                        "phone": session.get("phone"),
                        "intents": session.get("intents", []),
                        "status": "pending"
                    }

                    save_order(clean_order)
                    clear_session(user_id)
                    order_saved = True
                    replies.append("تم تسجيل الطلب ✅")
        
        # =========================
        # ❓ Unknown intent
        # =========================
        elif intent == "unknown":
            if not has_pending:
                replies.append("لم أفهم جزء من رسالتك.")
    
    # إذا لم تكن هناك ردود، رد افتراضي
    if not replies:
        if has_pending:
            replies.append("أنا بانتظار باقي المعلومات المطلوبة.")
        else:
            replies.append("لم أفهم رسالتك. هل يمكنك التوضيح أكثر؟")
    
    # تجميع الردود في رسالة واحدة
    combined_reply = " | ".join(replies)
    
    return jsonify({
        "reply": combined_reply,
        "intents": intents,
        "replies": replies,  # ردود منفصلة لكل نية
        "has_pending_order": has_pending_order(user_id)  # حالة الطلب المعلق
    })