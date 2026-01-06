# ==========================================
# FILE: ml_model.py (ML Model - IMPROVED TYPE HINTS)
# ==========================================

import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessor import TextPreprocessor
import os
import logging

logger = logging.getLogger(__name__)

class MLModel:
    """Class untuk Machine Learning Model (Naive Bayes + TF-IDF)"""
    
    def __init__(self, dataset_path: str = 'datasets.json', model_dir: str = 'models'):
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        
        self.model: Optional[MultinomialNB] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.intents_data: Optional[Dict] = None
        self.labels: List[str] = []
        
        # Create models directory if not exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_dataset(self) -> None:
        """Load dataset dari JSON"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.intents_data = json.load(f)
            logger.info(f"✅ Dataset loaded: {len(self.intents_data['intents'])} intents")
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.dataset_path}")
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON in dataset file")
            raise
    
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """Prepare data untuk training"""
        X_raw: List[str] = []
        y: List[str] = []
        
        if not self.intents_data:
            raise ValueError("Dataset belum dimuat")
        
        for intent in self.intents_data['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                X_raw.append(pattern)
                y.append(tag)
        
        return X_raw, y
    
    def train_model(self) -> Tuple[float, str]:
        """Train Naive Bayes model dengan TF-IDF"""
        try:
            print("\n🔄 Starting training process...")
            logger.info("Training process started")
            
            # Load dataset
            self.load_dataset()
            
            # Prepare data
            X_raw, y = self.prepare_training_data()
            print(f"📊 Total samples: {len(X_raw)}")
            logger.info(f"Total training samples: {len(X_raw)}")
            
            # Preprocess
            print("🔄 Preprocessing texts...")
            X_processed = self.preprocessor.preprocess_batch(X_raw)
            
            # TF-IDF Vectorization
            print("🔄 Extracting TF-IDF features...")
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            X = self.vectorizer.fit_transform(X_processed)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            print("🔄 Training Naive Bayes model...")
            self.model = MultinomialNB(alpha=1.0)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            
            print(f"\n✅ Training completed!")
            print(f"📊 Accuracy: {accuracy * 100:.2f}%")
            logger.info(f"Training completed with accuracy: {accuracy * 100:.2f}%")
            
            # Save labels
            self.labels = list(set(y))
            
            # Save model
            self.save_model()
            
            return accuracy, report
        
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise
    
    def save_model(self) -> None:
        """Save model dan vectorizer"""
        try:
            model_path = os.path.join(self.model_dir, 'model.pkl')
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            print(f"💾 Model saved to {self.model_dir}/")
            logger.info(f"Model saved successfully to {self.model_dir}/")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise
    
    def load_model(self) -> bool:
        """Load trained model"""
        try:
            model_path = os.path.join(self.model_dir, 'model.pkl')
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                logger.warning("Saved model files not found")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.load_dataset()
            
            logger.info("Model loaded from saved files")
            print("✅ Model loaded from saved files")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
    
    def initialize(self) -> None:
        """Initialize model (load or train)"""
        if not self.load_model():
            logger.info("No saved model found, training new model...")
            self.train_model()
    
    def predict(self, text: str) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Predict intent dari input text"""
        try:
            # Preprocess
            processed = self.preprocessor.preprocess(text)
            
            # Transform to TF-IDF
            vector = self.vectorizer.transform([processed])
            
            # Predict probabilities
            probabilities = self.model.predict_proba(vector)[0]
            
            # Get prediction
            predicted_tag = self.model.predict(vector)[0]
            confidence = float(max(probabilities))
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3 = [
                {
                    'tag': self.model.classes_[i],
                    'confidence': float(probabilities[i])
                }
                for i in top_3_indices
            ]
            
            return predicted_tag, confidence, top_3
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}", exc_info=True)
            raise
    
    def get_response(self, tag: str) -> str:
        """Get random response berdasarkan tag"""
        try:
            for intent in self.intents_data['intents']:
                if intent['tag'] == tag:
                    return np.random.choice(intent['responses'])
            return "Maaf, saya tidak memahami pertanyaan Anda."
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}", exc_info=True)
            return "Terjadi kesalahan saat memproses respons."
    
    def predict_and_respond(self, user_input: str) -> Dict[str, Any]:
        """Predict dan return response"""
        predicted_tag, confidence, top_3 = self.predict(user_input)
        response = self.get_response(predicted_tag)
        
        return {
            'response': response,
            'intent': predicted_tag,
            'confidence': float(confidence),
            'top_predictions': top_3
        }
    
    def get_intents_list(self) -> List[Dict[str, Any]]:
        """Get list of all intents"""
        return [
            {
                'tag': intent['tag'],
                'patterns_count': len(intent['patterns']),
                'sample_pattern': intent['patterns'][0] if intent['patterns'] else ''
            }
            for intent in self.intents_data['intents']
        ]
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            X_raw, y = self.prepare_training_data()
            X_processed = self.preprocessor.preprocess_batch(X_raw)
            X = self.vectorizer.transform(X_processed)
            
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred, output_dict=True, zero_division=0)
            
            return {
                'accuracy': float(accuracy),
                'total_samples': len(X_raw),
                'classification_report': report
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'Multinomial Naive Bayes',
            'vectorizer': 'TF-IDF',
            'preprocessing': [
                'Lowercase',
                'Remove special characters',
                'Stopword removal (Bahasa Indonesia)',
                'Stemming (Sastrawi)'
            ],
            'total_intents': len(self.intents_data['intents']),
            'feature_size': len(self.vectorizer.get_feature_names_out()),
            'model_status': 'ready' if self.model else 'not_initialized'
        }
