"""
Vectorstore Handler Module
---------------------------------------------
All sources (Plato JSON + PDFs) available for all models
"""

import os
from typing import List, Optional
from utils.config import GOOGLE_API_KEY, MODEL_OPTIONS
from utils.pdf_handler import get_pdf_text, get_text_chunks
from utils.json_ingestor import load_platon_json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from datetime import datetime

# Persistence directories by provider
PERSIST_DIR = {
    "groq": "./data/groq_vector_store.chroma",
    "gemini": "./data/gemini_vector_store.chroma", 
    "spanish-llm": "./data/spanish_vector_store.chroma"
}

# Fixed path to Plato JSON
PLATON_JSON_PATH = "./data/platon_analisis_nlp.json"

def get_embeddings(model_provider: str): # Repasar Embeddings de los distintos modelos
    """Same as before - no changes"""
    if model_provider == "Groq":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif model_provider == "Gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
    elif model_provider == "Spanish llm":
        return HuggingFaceEmbeddings(
            model_name="hiiamsid/sentence_similarity_spanish_es",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        raise ValueError(f"Unsupported provider: {model_provider}")

# Variable de módulo
_PLATO_DOCS_CACHE: Optional[list[Document]] = None

def get_plato_documents() -> List[Document]:
    """
    Load enriched Plato documents from JSON.
    Executes ONCE and caches in memory.
    """
    global _PLATO_DOCS_CACHE

    if _PLATO_DOCS_CACHE is None:
        try:
            _PLATO_DOCS_CACHE = load_platon_json(PLATON_JSON_PATH)
            print(f"Plato loaded: {len(_PLATO_DOCS_CACHE)} documents")
        except Exception as e:
            print(f"Error loading plato: {e}")
            _PLATO_DOCS_CACHE = []
    return _PLATO_DOCS_CACHE

def get_or_create_vectorstore(uploaded_files: Optional[List], model_provider: str) -> Chroma:
    """Always includes Plato + optional PDFs"""
    
    embedding = get_embeddings(model_provider)
    persist_path = PERSIST_DIR.get(model_provider, f"./data/{model_provider}_store.chroma")
    
    # ================================
    # 1. Load or create vectorstore
    # ================================
    if os.path.exists(persist_path) and os.listdir(persist_path):
        print(f"🔄 Loading existing vectorstore from {persist_path}")
        vectorstore = Chroma(
            persist_directory=persist_path,
            embedding_function=embedding,
            collection_name=f"platonic_{model_provider}"
        )
        
        # Check if Plato is already loaded
        existing_sources = set()
        try:
            # Get metadata of existing documents
            results = vectorstore.get(limit=10000)  # Adjust limit if needed
            existing_sources = {
                doc.get("source", "") 
                for doc in results.get("metadatas", [])
            }
        except:
            pass
        
        plato_exists = "platon_analisis_nlp.json" in existing_sources
        
    else:
        print(f"🆕 Creating new vectorstore at {persist_path}")
        vectorstore = None
        plato_exists = False
    
    # ================================
    # 2. Prepare documents to add
    # ================================
    documents_to_add = []
    
    # Add Plato only if not present
    if not plato_exists:
        plato_docs = get_plato_documents()
        if plato_docs:
            documents_to_add.extend(plato_docs)
            print(f"🧠 Adding Platonic base: {len(plato_docs)} fragments")
    else:
        print(f"✓ Platonic base already loaded")
    
    # Add PDFs (always new)
    if uploaded_files:
        try:
            raw_text = get_pdf_text(uploaded_files)
            chunks = get_text_chunks(raw_text)
            
            pdf_documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": f"uploaded_pdf_{uploaded_files[0].name}",  # ← Nombre específico
                        "chunk_id": i,
                        "type": "user_pdf",
                        "provider": model_provider,
                        "timestamp": str(datetime.now())  # Para identificar uploads únicos
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
            documents_to_add.extend(pdf_documents)
            print(f"📄 Adding PDF chunks: {len(pdf_documents)}")
            
        except Exception as e:
            print(f"❌ Error processing PDFs: {e}")
    
    # ================================
    # 3. Create or update vectorstore
    # ================================
    if not documents_to_add:
        print("✓ No new documents to add")
        return vectorstore
    
    if vectorstore is None:
        # Create new
        vectorstore = Chroma.from_documents(
            documents=documents_to_add,
            embedding=embedding,
            persist_directory=persist_path,
            collection_name=f"platonic_{model_provider}"
        )
        print(f"✅ Created with {len(documents_to_add)} documents")
    else:
        # Add to existing
        vectorstore.add_documents(documents_to_add)
        print(f"✅ Added {len(documents_to_add)} new documents")
    
    return vectorstore