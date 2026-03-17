import firebase_admin
from firebase_admin import credentials, firestore

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(current_dir, 'serviceAccountKey.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

VIOLATION_INFO = {
    'helmet':     {'fine': 10000},
    'no_helmet':  {'fine': 10000},
    'redlight':   {'fine': 30000},
    'sidewalk':   {'fine': 15000},
    'wrong_way':  {'fine': 30000},
    'wrong_lane': {'fine': 20000},
    'sign':       {'fine': 25000},
}

def update_all_fines():
    docs = db.collection('violations').stream()
    count = 0
    for doc in docs:
        data = doc.to_dict()
        v_type = data.get('type')
        fine = 20000 # default
        if v_type in VIOLATION_INFO:
            fine = VIOLATION_INFO[v_type]['fine']
        
        doc.reference.update({'fineAmount': fine})
        count += 1
    print(f"Updated {count} violations in Firestore.")

if __name__ == '__main__':
    update_all_fines()
