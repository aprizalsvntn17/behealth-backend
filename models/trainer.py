import json
import os
import random
import joblib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from models.preprocessor import get_preprocessor


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
        print("\n" + "="*60)
        print("TRAINING HEALTH CHATBOT MODEL")
        print("="*60)
        
        print("\n1. Loading dataset...")
        patterns, labels, responses = self.load_dataset()
        print(f"   Loaded {len(patterns)} patterns across {len(self.classes)} classes")
        
        class_counts = Counter(labels)
        print(f"\n   Class distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {cls}: {count} samples")
        
        print("\n2. Preprocessing patterns...")
        processed_patterns = self.preprocessor.preprocess_batch(patterns)
        print(f"   Preprocessing complete")
        
        if augment:
            print("\n3. Augmenting data...")
            processed_patterns, labels = self.augment_data(processed_patterns, labels, augment_factor=1)
            print(f"   After augmentation: {len(processed_patterns)} samples")
        
        print("\n4. Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            processed_patterns, labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels
        )
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
        print("\n5. Initializing TF-IDF vectorizer...")
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
        print(f"   Using {classifier_type.upper()} classifier")
        
        print("\n6. Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        print(f"   Feature dimensions: {X_train_tfidf.shape[1]}")
        
        print("\n7. Training classifier...")
        self.classifier.fit(X_train_tfidf, y_train)
        print("   Training complete")
        
        print("\n8. Evaluating model...")
        y_pred = self.classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n9. Cross-validation (5-fold)...")
        X_all_tfidf = self.vectorizer.transform(processed_patterns)
        cv_scores = cross_val_score(self.classifier, X_all_tfidf, labels, cv=5)
        print(f"   CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        print("\n10. Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        print("\n11. Saving models...")
        self.save_models()
        print(f"    Models saved to: {self.model_path}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        
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