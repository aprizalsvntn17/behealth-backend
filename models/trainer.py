import json
import os
import random
import joblib
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from models.preprocessor import get_preprocessor

logger = logging.getLogger(__name__)


class ChatbotTrainer:
    
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.preprocessor = get_preprocessor()
        
        os.makedirs(model_path, exist_ok=True)
        
        self.vectorizer = None
        self.classifier = None
        self.intents = None
        self.responses = None
        self.classes = None
        
    def load_dataset(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        patterns = []
        labels = []
        responses = {}
        
        for intent in data['intents']:
            tag = intent['tag']
            responses[tag] = intent['responses']
            
            for pattern in intent['patterns']:
                patterns.append(pattern)
                labels.append(tag)
        
        self.intents = data['intents']
        self.responses = responses
        self.classes = list(responses.keys())
        
        return patterns, labels, responses
    
    def augment_data(self, patterns, labels, augment_factor=1):
        augmented_patterns = list(patterns)
        augmented_labels = list(labels)
        
        for pattern, label in zip(patterns, labels):
            words = pattern.split()
            
            if len(words) <= 2:
                continue
            
            for _ in range(augment_factor):
                shuffled = words.copy()
                random.shuffle(shuffled)
                augmented_patterns.append(' '.join(shuffled))
                augmented_labels.append(label)
        
        return augmented_patterns, augmented_labels
    
    def train(self, classifier_type='logistic', augment=True, test_size=0.2):
        logger.info("Starting health chatbot model training")
        
        logger.info("Loading dataset...")
        patterns, labels, responses = self.load_dataset()
        logger.info(f"Loaded {len(patterns)} patterns across {len(self.classes)} classes")
        
        class_counts = Counter(labels)
        logger.info("Class distribution (top 5):")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"- {cls}: {count} samples")
        
        logger.info("Preprocessing patterns...")
        processed_patterns = self.preprocessor.preprocess_batch(patterns)
        
        if augment:
            logger.info("Augmenting data...")
            processed_patterns, labels = self.augment_data(processed_patterns, labels, augment_factor=1)
            logger.info(f"After augmentation: {len(processed_patterns)} samples")
        
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            processed_patterns, labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels
        )
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        logger.info("Initializing TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            min_df=1,
            max_df=0.9,
            sublinear_tf=True
        )
        
        classifiers = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic': LogisticRegression(
                max_iter=1000, 
                C=10, 
                class_weight='balanced',
                random_state=42
            ),
            'svm': LinearSVC(
                C=1.0, 
                max_iter=2000, 
                class_weight='balanced',
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced',
                random_state=42
            )
        }
        
        self.classifier = classifiers.get(classifier_type, classifiers['logistic'])
        logger.info(f"Using {classifier_type.upper()} classifier")
        
        logger.info("Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        logger.info(f"Feature dimensions: {X_train_tfidf.shape[1]}")
        
        logger.info("Training classifier...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        logger.info("Evaluating model...")
        y_pred = self.classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        logger.info("Running cross-validation (5-fold)...")
        X_all_tfidf = self.vectorizer.transform(processed_patterns)
        cv_scores = cross_val_score(self.classifier, X_all_tfidf, labels, cv=5)
        logger.info(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Log classification report
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))
        
        logger.info("Saving models...")
        self.save_models()
        logger.info(f"Models saved to: {self.model_path}")
        
        logger.info("Training completed successfully")
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'num_classes': len(self.classes),
            'num_samples': len(processed_patterns),
            'classifier_type': classifier_type
        }
    
    def save_models(self):
        vectorizer_path = os.path.join(self.model_path, 'vectorizer.joblib')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        classifier_path = os.path.join(self.model_path, 'classifier.joblib')
        joblib.dump(self.classifier, classifier_path)
        
        metadata = {
            'classes': self.classes,
            'responses': self.responses
        }
        metadata_path = os.path.join(self.model_path, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def train_model(dataset_path, model_path, classifier_type='logistic'):
    trainer = ChatbotTrainer(dataset_path, model_path)
    return trainer.train(classifier_type=classifier_type)