
import streamlit as st
import time
import os
import sys
import logging
from config import Config
from models.chatbot import get_chatbot
from models.trainer import train_model as run_training

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(
    page_title="Health Assistant Chatbot",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 12px;
    }
    .user-avatar {
        background-color: #007bff;
    }
    .bot-avatar {
        background-color: #28a745;
    }
    h1 {
        text-align: center;
        color: #1E88E5;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Halo! Saya adalah asisten kesehatan virtual Anda. Ada yang bisa saya bantu terkait keluhan kesehatan?"
    })

# Load Chatbot
@st.cache_resource
def load_bot():
    try:
        # Ensure model path exists
        if not os.path.exists(Config.MODEL_PATH):
            os.makedirs(Config.MODEL_PATH, exist_ok=True)
            return None
        
        # Check if model files exist
        required_files = ['classifier.joblib', 'vectorizer.joblib', 'metadata.json']
        if not all(os.path.exists(os.path.join(Config.MODEL_PATH, f)) for f in required_files):
            return None
            
        bot = get_chatbot(
            model_path=Config.MODEL_PATH,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )
        return bot
    except Exception as e:
        logger.error(f"Error loading chatbot: {e}")
        return None

chatbot = load_bot()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male--v1.png", width=64)
    st.title("Health Assistant")
    st.info("Chatbot ini dapat membantu mendiagnosa gejala awal dan memberikan saran kesehatan sederhana.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Reset Chat"):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Halo! Saya adalah asisten kesehatan virtual Anda. Ada yang bisa saya bantu terkait keluhan kesehatan?"
            })
            st.rerun()
            
    with col2:
        if st.button("🔄 Retrain"):
            with st.spinner("Melatih ulang model..."):
                try:
                    result = run_training(
                        dataset_path=Config.DATASET_PATH,
                        model_path=Config.MODEL_PATH
                    )
                    st.success(f"Training selesai! Akurasi: {result['accuracy']*100:.1f}%")
                    # Clear cache to reload model
                    load_bot.clear()
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal melatih: {str(e)}")

    st.markdown("---")
    st.markdown("### Tentang")
    st.caption("Versi 1.0.0")
    st.caption("Developed with ❤️")

# Main Chat Interface
st.title("🏥 Layanan Konsultasi Kesehatan")

if chatbot is None:
    st.warning("⚠️ Model belum tersedia. Silakan klik tombol **Retrain** di sidebar untuk melatih model pertama kali.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ketik keluhan Anda di sini..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Sedang menganalisis..."):
                # Simulate thinking time for better UX
                time.sleep(0.5)
                
                try:
                    response_data = chatbot.get_response(prompt)
                    response_text = response_data['response']
                    
                    # Optional: Show confidence in debug/expander
                    # with st.expander("Debug Info"):
                    #     st.json(response_data)
                        
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error("Maaf, terjadi kesalahan sistem.")
                    logger.error(f"Chat Error: {e}")
