import os
import json
import random
import joblib
import numpy as np
from models.preprocessor import get_preprocessor


class HealthChatbot:
    
    def __init__(self, model_path, confidence_threshold=0.35):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.preprocessor = get_preprocessor()
        
        self.vectorizer = None
        self.classifier = None
        self.classes = None
        self.responses = None
        
        self._load_models()
    
    def _load_models(self):
        vectorizer_path = os.path.join(self.model_path, 'vectorizer.joblib')
        classifier_path = os.path.join(self.model_path, 'classifier.joblib')
        metadata_path = os.path.join(self.model_path, 'metadata.json')
        
        if not all(os.path.exists(p) for p in [vectorizer_path, classifier_path, metadata_path]):
            raise FileNotFoundError(
                f"Model files not found in {self.model_path}. "
                "Please train the model first using: python train.py"
            )
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.classes = metadata['classes']
        self.responses = metadata['responses']
        
        print(f"Chatbot loaded: {len(self.classes)} intents")
    
    def predict_intent(self, text):
        processed = self.preprocessor.preprocess(text)
        
        if not processed or len(processed.strip()) == 0:
            return 'fallback', 0.0
        
        X = self.vectorizer.transform([processed])
        intent = self.classifier.predict(X)[0]
        
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(X)[0]
            intent_idx = list(self.classifier.classes_).index(intent)
            confidence = probabilities[intent_idx]
        else:
            decision = self.classifier.decision_function(X)[0]
            if len(decision.shape) == 0 or decision.shape[0] == 1:
                confidence = float(abs(decision))
            else:
                intent_idx = list(self.classifier.classes_).index(intent)
                exp_scores = np.exp(decision - np.max(decision))
                probabilities = exp_scores / exp_scores.sum()
                confidence = probabilities[intent_idx]
        
        return intent, float(confidence)
    
    def get_response(self, text):
        if not text or not text.strip():
            return {
                'response': "Silakan masukkan pertanyaan Anda tentang kesehatan.",
                'intent': 'empty_input',
                'confidence': 0.0
            }
        
        intent, confidence = self.predict_intent(text)
        
        if confidence < self.confidence_threshold:
            fallback_responses = [
                "Maaf, saya kurang memahami pertanyaan Anda. Bisa dijelaskan lebih detail tentang keluhan atau gejala yang Anda rasakan?",
                "Mohon maaf, saya tidak mengerti. Silakan tanyakan tentang gejala kesehatan atau obat-obatan yang lebih spesifik.",
                "Saya belum bisa menjawab pertanyaan tersebut dengan baik. Coba tanyakan tentang keluhan kesehatan seperti: demam, batuk, sakit kepala, sakit perut, dll.",
                "Maaf, saya tidak yakin dengan pertanyaan Anda. Apakah Anda bisa menyebutkan gejala atau keluhan kesehatan yang lebih jelas?"
            ]
            
            return {
                'response': random.choice(fallback_responses),
                'intent': 'fallback',
                'confidence': confidence
            }
        
        responses_list = self.responses.get(intent, [])
        
        if responses_list:
            response = random.choice(responses_list)
        else:
            response = "Maaf, saya tidak memiliki informasi untuk pertanyaan tersebut."
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence
        }
    
    def chat(self, text):
        return self.get_response(text)


_chatbot = None

def get_chatbot(model_path, confidence_threshold=0.35):
    global _chatbot
    if _chatbot is None:
        _chatbot = HealthChatbot(model_path, confidence_threshold)
    return _chatbot

def reload_chatbot(model_path, confidence_threshold=0.35):
    global _chatbot
    _chatbot = HealthChatbot(model_path, confidence_threshold)
    return _chatbot