from flask import Flask
from app.routes.chat import chat_bp
from app.database.mongo import init_db

def create_app():
    app = Flask(__name__)
    app.json.ensure_ascii = False
    init_db(app)

    app.register_blueprint(chat_bp)

    return app