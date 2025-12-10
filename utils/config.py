import os

from dotenv import load_dotenv

# Load environment variables from .env file into os.environ
load_dotenv()

# API keys for different LLM providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Dictionary of available model providers and their respective models
MODEL_OPTIONS = {
    "Gemini": {
        "playground": "https://ai.google.dev",
        "models": ["gemini-2.0-flash-exp", "gemini-1.5-flash"],
        "requires_api_key": True,
        "api_key_env": "GOOGLE_API_KEY"
    },
    "Groq": {
        "playground": "https://console.groq.com/",
        "models": ["llama-3.1-8b-instant", "llama3-70b-8192"],
        "requires_api_key": True,
        "api_key_env": "GROQ_API_KEY"
    },
    "Spanish LLM": {
        "playground": "local",
        "models": ["eva-mistral-7b-spanish"],
        "is_local": True,
        "requires_api_key": False
    }
}
# ==================== VECTOR STORE CONFIG ====================
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "platon_collection")

# ==================== EMBEDDING CONFIG ====================
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ==================== RAG PARAMETERS ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))