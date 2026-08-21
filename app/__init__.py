from flask import Flask
from app.routes.chat import chat_bp
from app.database.mongo import init_db
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(
        app,
        origins="http://localhost:5173"
    )
    app.json.ensure_ascii = False
    init_db(app)

    app.register_blueprint(chat_bp)

    return app