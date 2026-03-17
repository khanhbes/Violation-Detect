import firebase_admin
from firebase_admin import credentials, firestore
import os, json
from datetime import datetime

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if type(obj).__name__ == 'DatetimeWithNanoseconds':
            return obj.isoformat()
        return super().default(obj)

current_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(current_dir, 'serviceAccountKey.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

print("--- USERS ---")
users = list(db.collection('users').stream())
for u in users:
    print(f"User ID: {u.id}, Email: {u.to_dict().get('email')}, Created: {u.to_dict().get('createdAt')}")

print("\n--- VIOLATIONS ---")
violations = list(db.collection('violations').stream())
for v in violations:
    data = v.to_dict()
    print(f"Violation ID: {v.id}, UserID: {data.get('userId')}, Created: {data.get('createdAt')}, Status: {data.get('status')}")

