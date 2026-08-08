from app import create_app
from flask import send_from_directory
import os
app = create_app()
@app.route("/")
def dashboard():
    return send_from_directory(os.getcwd(), "dashboard.html")
if __name__ == "__main__":
    app.run(debug=True, port=5000)