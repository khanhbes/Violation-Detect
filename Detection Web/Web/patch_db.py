import firebase_admin
from firebase_admin import credentials, firestore
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(current_dir, 'serviceAccountKey.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Default to the most recently created user
TEST_USER_ID = "fxEbTaTqg2W1B05WDZ2NnKZPadb2"

violations = list(db.collection('violations').stream())
updated = 0
for v in violations:
    data = v.to_dict()
    if data.get('userId') in [None, "default_user", ""]:
        db.collection('violations').document(v.id).update({'userId': TEST_USER_ID})
        updated += 1

print(f"Updated {updated} violations to test user: {TEST_USER_ID}")
