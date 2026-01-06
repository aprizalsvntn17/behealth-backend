import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-key-for-dev')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_models')
    DATASET_PATH = os.path.join(os.path.dirname(__file__), 'sethealth.json')
    
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.35))
    
    CORS_ORIGINS = [s.strip() for s in os.getenv('CORS_ORIGINS', '*').split(',') if s.strip()]