import re
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class TextPreprocessor:
    
    def __init__(self):
        if NLTK_AVAILABLE:
            try:
                self.stopwords = set(stopwords.words('indonesian'))
            except:
                self.stopwords = set()
        else:
            self.stopwords = set()
        
        custom_stopwords = {
            'yang', 'dan', 'atau', 'untuk', 'pada', 'ke', 'dari', 'di',
            'sudah', 'akan', 'sangat', 'sekali', 'banget', 'dong', 'deh', 
            'sih', 'nih', 'yah', 'lalu', 'kemudian', 'adalah'
        }
        self.stopwords.update(custom_stopwords)
        
        self.health_keywords = {
            'sakit', 'pusing', 'demam', 'batuk', 'pilek', 'flu',
            'mual', 'muntah', 'diare', 'gatal', 'nyeri', 'pegal',
            'lemas', 'lelah', 'panas', 'dingin', 'bengkak', 'merah',
            'obat', 'vitamin', 'dokter', 'rumah', 'klinik', 'terima', 'kasih',
            'kepala', 'perut', 'dada', 'kaki', 'tangan', 'mata', 'hai', 'halo',
            'telinga', 'hidung', 'tenggorokan', 'gigi', 'kulit', 'salam',
            'jantung', 'paru', 'hati', 'ginjal', 'lambung', 'usus',
            'darah', 'tekanan', 'gula', 'kolesterol', 'asam', 'maag',
            'alergi', 'infeksi', 'virus', 'bakteri', 'radang', 'masuk', 'angin',
            'stroke', 'diabetes', 'hipertensi', 'asma', 'bye', 'sampai', 'jumpa',
            'migrain', 'vertigo', 'insomnia', 'stress', 'cemas', 'tidur',
            'anak', 'bayi', 'hamil', 'ibu', 'bapak', 'saya', 'aku', 'kamu',
            'help', 'bantuan', 'tolong', 'boleh', 'bisa', 'minta', 'tanya'
        }
        
        self.stopwords -= self.health_keywords
        
    def clean_text(self, text):
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
        else:
            tokens = text.split()
        return tokens
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens 
                if token not in self.stopwords and len(token) > 1]
    
    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts):
        return [self.preprocess(text) for text in texts]


_preprocessor = None

def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor()
    return _preprocessor