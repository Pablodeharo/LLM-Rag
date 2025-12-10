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
  "Groq": {
    "playground": "https://console.groq.com/",
    "models": ["llama-3.1-8b-instant", "llama3-70b-8192"]
  },
  "Gemini": {
    "playground": "https://ai.google.dev",
    "models": ["gemini-2.0-flash", "gemini-2.5-flash"]
  },
  "Spanish LLM": {
        "playground": "local",
        "models": ["eva-mistral-7b-spanish"],
        "is_local": True,
        "api_key_env": "HF_TOKEN"
    }
}
