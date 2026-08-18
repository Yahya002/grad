from app.database.mongo import get_db
from bson.objectid import ObjectId

def save_order(order_data):
    """
    حفظ طلب جديد في قاعدة البيانات
    
    المعاملات:
        order_data (dict): بيانات الطلب
        
    المُرجع:
        ObjectId: معرف الطلب المُحفوظ
    """
    db = get_db()
    return db.orders.insert_one(order_data)

def update_order_status(order_id, status):
    """
    تحديث حالة طلب موجود
    
    المعاملات:
        order_id (str): معرف الطلب
        status (str): الحالة الجديدة (pending, cancelled, completed, etc.)
        
    المُرجع:
        result: نتيجة عملية التحديث
    """
    db = get_db()
    return db.orders.update_one(
        {"_id": ObjectId(order_id)},
        {"$set": {"status": status}}
    )

def get_latest_order_by_user(user_id):
    """
    الحصول على آخر طلب للمستخدم
    
    المعاملات:
        user_id (str): معرف المستخدم
        
    المُرجع:
        dict: آخر طلب أو None
    """
    db = get_db()
    return db.orders.find_one(
        {"user_id": user_id},
        sort=[("created_at", -1)] if "created_at" in db.orders.find_one() else []
    )