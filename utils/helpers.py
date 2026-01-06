import json
import os
from datetime import datetime


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath, indent=2):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_response(success, data=None, message=None, error=None):
    response = {
        'success': success,
        'timestamp': get_timestamp()
    }
    
    if data is not None:
        response['data'] = data
    
    if message:
        response['message'] = message
    
    if error:
        response['error'] = error
    
    return response


def validate_chat_request(data):
    if not data:
        return False, "Request body is required"
    
    if 'message' not in data:
        return False, "Field 'message' is required"
    
    message = data.get('message', '')
    
    if not isinstance(message, str):
        return False, "Field 'message' must be a string"
    
    if len(message) > 1000:
        return False, "Message too long (max 1000 characters)"
    
    return True, None