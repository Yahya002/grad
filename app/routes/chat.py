from flask import Blueprint, request, jsonify, render_template
from app.database.mongo import get_db
from app.services.intent_service import analyze_message
from app.services.session_service import get_session, update_session, clear_session
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

    intent = ai_result.get("intent")

    # =========================
    # 🟢 Greeting
    # =========================
    if intent == "greeting":
        return jsonify({"reply": "أهلاً وسهلاً 👋"})

    # =========================
    # 🧠 Session merge
    # =========================
    session = update_session(user_id, ai_result)

    # =========================
    # 🔍 Validation
    # =========================
    missing = validate(session)

    if missing:
        return jsonify({
            "reply": f"محتاج: {', '.join(missing)}"
        })

    # =========================
    # 💾 Save order
    # =========================
    clean_order = {
    "from": session.get("from"),
    "to": session.get("to"),
    "name": session.get("name"),
    "phone": session.get("phone"),
    "intent": session.get("intent", "unknown"),
    "status": "pending"
}

    save_order(clean_order)

    clear_session(user_id)

    return jsonify({
        "reply": "تم تسجيل الطلب ✅"
    })