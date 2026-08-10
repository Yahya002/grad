from app.database.mongo import get_db

def save_order(order_data):
    db = get_db()
    return db.orders.insert_one(order_data)