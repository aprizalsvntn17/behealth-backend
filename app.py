"""
Flask Application for Health Chatbot API
Main entry point for the backend server
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.chatbot import get_chatbot, reload_chatbot
from models.trainer import train_model
from utils.helpers import format_response, validate_chat_request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS for frontend
CORS(app, resources={
    r"/api/*": {
        "origins": Config.CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Global chatbot instance
chatbot = None


def initialize_chatbot():
    """Initialize or reload the chatbot."""
    global chatbot
    try:
        chatbot = get_chatbot(
            model_path=Config.MODEL_PATH,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )
        logger.info("Chatbot model loaded successfully")
        return True
    except FileNotFoundError as e:
        logger.warning(f"Model file not found: {e}")
        logger.warning("Please run 'python train.py' first to train the model.")
        return False


# ==========================================
# API ROUTES
# ==========================================

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        'name': 'Health Chatbot API',
        'version': '1.0.0',
        'description': 'API untuk Chatbot Layanan Kesehatan',
        'endpoints': {
            'POST /api/chat': 'Kirim pesan ke chatbot',
            'POST /api/train': 'Latih ulang model',
            'GET /api/health': 'Health check',
            'GET /api/intents': 'Daftar intent yang tersedia'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_loaded = chatbot is not None
    return jsonify(format_response(
        success=True,
        data={
            'status': 'healthy',
            'model_loaded': model_loaded,
            'num_intents': len(chatbot.classes) if model_loaded else 0
        }
    ))


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint - main API for chatbot interaction.
    """
    global chatbot
    
    # Check if model is loaded
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify(format_response(
                success=False,
                error="Model belum dilatih."
            )), 503
    
    # Get request data
    data = request.get_json()
    
    # Validate request
    is_valid, error_msg = validate_chat_request(data)
    if not is_valid:
        logger.warning(f"Invalid chat request: {error_msg}")
        return jsonify(format_response(
            success=False,
            error=error_msg
        )), 400
    
    # Get message
    message = data.get('message', '').strip()
    
    # Get chatbot response
    try:
        result = chatbot.get_response(message)
        
        return jsonify(format_response(
            success=True,
            data={
                'response': result['response'],
                'intent': result['intent'],
                'confidence': round(result['confidence'], 4)
            }
        ))
    
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        return jsonify(format_response(
            success=False,
            error="Terjadi kesalahan saat memproses pesan Anda."
        )), 500


@app.route('/api/train', methods=['POST'])
def train():
    """
    Training endpoint - retrain the model.
    """
    global chatbot
    
    classifier_type = request.args.get('classifier', 'logistic')
    
    try:
        logger.info(f"Starting model training with classifier: {classifier_type}")
        # Train model
        result = train_model(
            dataset_path=Config.DATASET_PATH,
            model_path=Config.MODEL_PATH,
            classifier_type=classifier_type
        )
        
        # Reload chatbot with new model
        chatbot = reload_chatbot(
            model_path=Config.MODEL_PATH,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )
        
        logger.info("Model training completed and reloaded")
        return jsonify(format_response(
            success=True,
            message="Model berhasil dilatih ulang!",
            data={
                'accuracy': round(result['accuracy'], 4),
                'cv_mean': round(result['cv_mean'], 4),
                'cv_std': round(result['cv_std'], 4),
                'num_classes': result['num_classes'],
                'num_samples': result['num_samples'],
                'classifier': classifier_type
            }
        ))
    
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        return jsonify(format_response(
            success=False,
            error=f"Gagal melatih model: {str(e)}"
        )), 500


@app.route('/api/intents', methods=['GET'])
def get_intents():
    """Get list of available intents."""
    global chatbot
    
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify(format_response(
                success=False,
                error="Model belum dimuat."
            )), 503
    
    return jsonify(format_response(
        success=True,
        data={
            'intents': chatbot.classes,
            'count': len(chatbot.classes)
        }
    ))


@app.route('/api/test', methods=['POST'])
def test_batch():
    """
    Test multiple messages at once.
    """
    global chatbot
    
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify(format_response(
                success=False,
                error="Model belum dimuat."
            )), 503
    
    data = request.get_json()
    messages = data.get('messages', [])
    
    if not messages or not isinstance(messages, list):
        return jsonify(format_response(
            success=False,
            error="Field 'messages' harus berupa array string"
        )), 400
    
    results = []
    for msg in messages[:20]:  # Limit to 20 messages
        try:
            result = chatbot.get_response(str(msg))
            results.append({
                'input': msg,
                'response': result['response'],
                'intent': result['intent'],
                'confidence': round(result['confidence'], 4)
            })
        except Exception as e:
            logger.error(f"Error in batch test for message '{msg}': {e}")
            results.append({
                'input': msg,
                'error': str(e)
            })
    
    return jsonify(format_response(
        success=True,
        data={
            'results': results,
            'count': len(results)
        }
    ))


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify(format_response(
        success=False,
        error="Endpoint tidak ditemukan"
    )), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify(format_response(
        success=False,
        error="Method tidak diizinkan"
    )), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify(format_response(
        success=False,
        error="Terjadi kesalahan internal server"
    )), 500


# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    logger.info("Starting Health Chatbot API Server...")
    
    # Try to initialize chatbot
    if initialize_chatbot():
        logger.info(f"Chatbot loaded with {len(chatbot.classes)} intents")
    else:
        logger.warning("Chatbot not loaded - run 'python train.py' first")
    
    logger.info(f"Server running on http://{Config.HOST}:{Config.PORT}")
    
    # Run server
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )