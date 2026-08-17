from flask import Blueprint, request, jsonify, render_template
from app.database.mongo import get_db
from app.services.intent_service import analyze_message
from app.services.session_service import get_session, update_session, clear_session, has_pending_order, set_session_status, get_session_status
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

    # Get session status before AI analysis for location resolution
    status = get_session_status(user_id)
    
    ai_result = analyze_message(message, session_status=status)
    from app.models.llm_log_model import save_llm_log
    save_llm_log(user_id, message, ai_result)
    print("AI RESULT:", ai_result)
    if "error" in ai_result:
        return jsonify(ai_result)

    intents = ai_result.get("intents", [])
    pickup = ai_result.get("from")
    dest = ai_result.get("to")
    
    # =========================
    # 🚨 Priority: Cancellation
    # =========================
    if "cancel" in intents:
        clear_session(user_id)
        return jsonify({
            "reply": "تم إلغاء الطلب ❌",
            "intents": intents,
            "replies": ["تم إلغاء الطلب ❌"],
            "has_pending_order": False
        })
    
    # =========================
    # 🎯 Get current session status
    # =========================
    print(f"🔍 Current status for {user_id}: {status}")
    
    reply = ""
    
    # =========================
    # 🔄 State Machine
    # =========================
    if status == "idle":
        # CASE "idle"
        if any(intent in intents for intent in ["taxi", "delivery", "request_taxi"]):
            # Check for greeting to prepend
            if "greeting" in intents:
                reply = "أهلا وسهلا. "
            
            # Evaluate entities
            if pickup and dest:
                # Both pickup & dest present: Confirm order
                session = update_session(user_id, ai_result)
                missing = validate(session)
                
                if not missing:
                    clean_order = {
                        "from": session.get("from"),
                        "to": session.get("to"),
                        "intents": session.get("intents", []),
                        "status": "pending"
                    }
                    save_order(clean_order)
                    clear_session(user_id)
                    set_session_status(user_id, "idle")
                    reply += "تم تأكيد طلبك ✅"
                else:
                    reply += f"محتاج: {', '.join(missing)}"
            elif pickup and not dest:
                # pickup ONLY: Save pickup, ask for destination
                session = update_session(user_id, ai_result)
                set_session_status(user_id, "waiting_for_dest")
                reply += "تم حفظ موقع الانطلاق. أين تريد الذهاب؟"
            elif dest and not pickup:
                # dest ONLY: Save dest, ask for pickup
                session = update_session(user_id, ai_result)
                set_session_status(user_id, "waiting_for_pickup")
                reply += "تم حفظ الوجهة. من أين تريد الانطلاق؟"
            else:
                # NO entities: Ask for pickup
                set_session_status(user_id, "waiting_for_pickup")
                reply += "من أين تريد الانطلاق؟"
        elif "greeting" in intents or "thank" in intents:
            reply = "أهلا وسهلا"
        else:
            reply = "لم أفهم رسالتك. هل يمكنك التوضيح أكثر؟"
    
    elif status == "waiting_for_pickup":
        # CASE "waiting_for_pickup"
        if pickup:
            # pickup present: Save pickup, ask for destination
            session = update_session(user_id, ai_result)
            set_session_status(user_id, "waiting_for_dest")
            reply = "تم حفظ موقع الانطلاق. أين تريد الذهاب؟"
        else:
            # Repeat pickup question
            reply = "من أين تريد الانطلاق؟"
    
    elif status == "waiting_for_dest":
        # CASE "waiting_for_dest"
        if dest:
            # dest present: Save dest, confirm order
            session = update_session(user_id, ai_result)
            missing = validate(session)
            
            if not missing:
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
                set_session_status(user_id, "idle")
                reply = "تم تأكيد طلبك ✅"
            else:
                reply = f"محتاج: {', '.join(missing)}"
        else:
            # Repeat destination question
            reply = "أين تريد الذهاب؟"
    
    elif status == "confirmed":
        # CASE "confirmed"
        if any(intent in intents for intent in ["taxi", "delivery", "request_taxi"]):
            # new request_taxi: Reset to idle, redirect to idle handling
            set_session_status(user_id, "idle")
            # Re-process as idle
            status = "idle"
            if "greeting" in intents:
                reply = "أهلا وسهلا. "
            
            if pickup and dest:
                session = update_session(user_id, ai_result)
                missing = validate(session)
                
                if not missing:
                    clean_order = {
                        "from": session.get("from"),
                        "to": session.get("to"),
                        "intents": session.get("intents", []),
                        "status": "pending"
                    }
                    save_order(clean_order)
                    clear_session(user_id)
                    set_session_status(user_id, "idle")
                    reply += "تم تأكيد طلبك ✅"
                else:
                    reply += f"محتاج: {', '.join(missing)}"
            elif pickup and not dest:
                session = update_session(user_id, ai_result)
                set_session_status(user_id, "waiting_for_dest")
                reply += "تم حفظ موقع الانطلاق. أين تريد الذهاب؟"
            elif dest and not pickup:
                session = update_session(user_id, ai_result)
                set_session_status(user_id, "waiting_for_pickup")
                reply += "تم حفظ الوجهة. من أين تريد الانطلاق؟"
            else:
                set_session_status(user_id, "waiting_for_pickup")
                reply += "من أين تريد الانطلاق؟"
        elif "thank" in intents:
            reply = "أهلا وسهلا"
        else:
            reply = "لم أفهم رسالتك. هل يمكنك التوضيح أكثر؟"
    
    return jsonify({
        "reply": reply,
        "intents": intents,
        "replies": [reply],
        "has_pending_order": has_pending_order(user_id)
    })