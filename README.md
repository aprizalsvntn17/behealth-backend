# Health Chatbot API

API Backend untuk Chatbot Layanan Kesehatan berbasis Machine Learning (NLP).

## Fitur

- Intent classification menggunakan TF-IDF + Logistic Regression
- Support untuk 40+ intent kesehatan
- REST API dengan Flask
- Cross-validation untuk akurasi model
- Production-ready dengan error handling

## Struktur Proyek

```
project/
├── models/
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── trainer.py
│   └── chatbot.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── trained_models/
├── datasets.json
├── config.py
├── app.py
├── train.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd health-chatbot
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Buat file `.env` atau edit yang sudah ada:

```env
SECRET_KEY=your-secret-key-here
DEBUG=False
HOST=0.0.0.0
PORT=5000
CONFIDENCE_THRESHOLD=0.35
CORS_ORIGINS=*
```

### 5. Train Model

```bash
python train.py
```

### 6. Run Server

```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

### 7. Run with Streamlit (UI Mode)

Untuk menjalankan antarmuka chat berbasis Streamlit:

```bash
streamlit run streamlit_app.py
```

Aplikasi akan berjalan di `http://localhost:8501`.

## API Endpoints

### 1. Health Check
```
GET /api/health
```

### 2. Chat
```
POST /api/chat
Content-Type: application/json

{
  "message": "saya sakit kepala"
}
```

Response:
```json
{
  "success": true,
  "timestamp": "2024-01-01 12:00:00",
  "data": {
    "response": "Untuk sakit kepala ringan...",
    "intent": "sakit_kepala_ringan",
    "confidence": 0.8542
  }
}
```

### 3. Get Intents
```
GET /api/intents
```

### 4. Train Model
```
POST /api/train?classifier=logistic
```

### 5. Test Batch
```
POST /api/test
Content-Type: application/json

{
  "messages": ["sakit kepala", "demam", "batuk"]
}
```

## Deployment

### Production Server (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t health-chatbot .
docker run -p 5000:5000 health-chatbot
```

## Model Performance

- Accuracy: >95%
- Cross-validation: ~95% (+/- 0.05)
- Intents: 40+
- Classifier: Logistic Regression with TF-IDF

## Tech Stack

- **Framework**: Flask 3.0
- **ML**: scikit-learn 1.3
- **NLP**: NLTK
- **Server**: Gunicorn (production)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| SECRET_KEY | Flask secret key | - |
| DEBUG | Debug mode | False |
| HOST | Server host | 0.0.0.0 |
| PORT | Server port | 5000 |
| CONFIDENCE_THRESHOLD | Min confidence for predictions | 0.35 |
| CORS_ORIGINS | Allowed CORS origins | * |

## License

MIT License

## Support

For issues and questions, please create an issue in the repository.