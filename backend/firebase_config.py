import firebase_admin
from firebase_admin import credentials

# Initialize Firebase Admin SDK with the service account key
cred = credentials.Certificate('config/firebase_service_account.json')  # Path to your Firebase service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://edora2-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase Realtime Database URL
})
