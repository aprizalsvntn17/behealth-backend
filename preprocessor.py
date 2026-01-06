# ==========================================
# FILE: preprocessor.py (Text Preprocessing)
# ==========================================

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    """Class untuk preprocessing teks Bahasa Indonesia"""
    
    def __init__(self):
        # Inisialisasi Sastrawi
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
        stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = stopword_factory.create_stop_word_remover()
    
    def preprocess(self, text):
        """
        Preprocessing pipeline:
        1. Lowercase
        2. Remove special characters & numbers
        3. Remove extra spaces
        4. Remove stopwords
        5. Stemming
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        text = self.stopword_remover.remove(text)
        
        # Stemming
        text = self.stemmer.stem(text)
        
        return text
    
    def preprocess_batch(self, texts):
        """Preprocess multiple texts"""
        return [self.preprocess(text) for text in texts]


