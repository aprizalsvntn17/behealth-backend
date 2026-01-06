import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.chatbot import get_chatbot, reload_chatbot
from models.trainer import train_model
from utils.helpers import format_response, validate_chat_request


app = Flask(__name__)
app.config.from_object(Config)

CORS(app, resources={
    r"/api/*": {
        "origins": Config.CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

chatbot = None


def initialize_chatbot():
    global chatbot
    try:
        chatbot = get_chatbot(
            model_path=Config.MODEL_PATH,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )
        return True
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Please run 'python train.py' first to train the model.")
        return False


@app.route('/', methods=['GET'])
def home():
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
    global chatbot
    
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify(format_response(
                success=False,
                error="Model belum dilatih. Jalankan 'python train.py' terlebih dahulu."
            )), 503
    
    data = request.get_json()
    
    is_valid, error_msg = validate_chat_request(data)
    if not is_valid:
        return jsonify(format_response(
            success=False,
            error=error_msg
        )), 400
    
    message = data.get('message', '').strip()
    
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
        return jsonify(format_response(
            success=False,
            error="Terjadi kesalahan saat memproses pesan Anda."
        )), 500


@app.route('/api/train', methods=['POST'])
def train():
    global chatbot
    
    classifier_type = request.args.get('classifier', 'logistic')
    
    try:
        result = train_model(
            dataset_path=Config.DATASET_PATH,
            model_path=Config.MODEL_PATH,
            classifier_type=classifier_type
        )
        
        chatbot = reload_chatbot(
            model_path=Config.MODEL_PATH,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )
        
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
        return jsonify(format_response(
            success=False,
            error=f"Gagal melatih model: {str(e)}"
        )), 500


@app.route('/api/intents', methods=['GET'])
def get_intents():
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
    for msg in messages[:20]:
        try:
            result = chatbot.get_response(str(msg))
            results.append({
                'input': msg,
                'response': result['response'],
                'intent': result['intent'],
                'confidence': round(result['confidence'], 4)
            })
        except Exception as e:
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


@app.errorhandler(404)
def not_found(error):
    return jsonify(format_response(
        success=False,
        error="Endpoint tidak ditemukan"
    )), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify(format_response(
        success=False,
        error="Method tidak diizinkan"
    )), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify(format_response(
        success=False,
        error="Terjadi kesalahan internal server"
    )), 500


if __name__ == '__main__':
    print("=" * 60)
    print("HEALTH CHATBOT API SERVER")
    print("=" * 60)
    
    if initialize_chatbot():
        print(f"Chatbot loaded with {len(chatbot.classes)} intents")
    else:
        print("Chatbot not loaded - run 'python train.py' first")
    
    print(f"\nStarting server on http://{Config.HOST}:{Config.PORT}")
    print("Press Ctrl+C to stop\n")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )