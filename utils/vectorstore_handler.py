"""
Vectorstore Handler Module
---------------------------------------------
Handles all sources (Plato JSON + PDFs) for local Spanish LLM
Groq / Gemini embeddings commented
"""

import os
from typing import List, Optional
from datetime import datetime
from utils.config import HF_TOKEN, GOOGLE_API_KEY
from utils.pdf_handler import get_pdf_text, get_text_chunks
from utils.json_ingestor import load_platon_json


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ================================
# Persistence directories by provider
# ================================
PERSIST_DIR = {
    # "Groq": "./data/groq_vector_store.chroma",
    # "Gemini": "./data/gemini_vector_store.chroma",
    "Spanish LLM": "./data/spanish_vector_store.chroma"
}

# ================================
# Fixed path to Plato JSON
# ================================
PLATON_JSON_PATH = "./data/platon_analisis_nlp.json"

# ================================
# Helper: Normalize provider name for ChromaDB
# ================================
def normalize_collection_name(provider: str) -> str:
    """
    Normalizes provider name to be ChromaDB-compatible.
    ChromaDB requires: 3-512 chars, [a-zA-Z0-9._-], starting/ending with alphanumeric.
    
    Args:
        provider: Model provider name (e.g., "Spanish LLM", "Groq")
    
    Returns:
        Normalized name (e.g., "platonic_spanish_llm", "platonic_groq")
    """
    # Convert to lowercase, replace spaces with underscores
    normalized = provider.lower().replace(" ", "_").replace("-", "_")
    # Remove any invalid characters
    normalized = "".join(c for c in normalized if c.isalnum() or c in "._-")
    # Ensure it starts and ends with alphanumeric
    normalized = normalized.strip("._-")
    # Add prefix
    return f"platonic_{normalized}"

# ================================
# Embeddings by provider
# ================================
def get_embeddings(model_provider: str):
    """
    #### Returns embedding function for a given model provider
    """
    if model_provider == "Spanish LLM":
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"}
        )
    
    elif model_provider == "Groq":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"}
        )
    
    elif model_provider == "Gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
    
    else:
        raise ValueError(f"Unsupported provider: {model_provider}")

# ================================
# Plato documents cache
# ================================
_PLATO_DOCS_CACHE: Optional[List[Document]] = None

def get_plato_documents() -> List[Document]:
    """
    #### Load enriched Plato documents from JSON
    Executes once and caches in memory
    """
    global _PLATO_DOCS_CACHE

    if _PLATO_DOCS_CACHE is None:
        try:
            _PLATO_DOCS_CACHE = load_platon_json(PLATON_JSON_PATH)
            print(f"🧠 Plato loaded: {len(_PLATO_DOCS_CACHE)} documents")
        except Exception as e:
            print(f"❌ Error loading Plato JSON: {e}")
            _PLATO_DOCS_CACHE = []

    return _PLATO_DOCS_CACHE

# ================================
# Vectorstore creation or loading
# ================================
def get_or_create_vectorstore(uploaded_files: Optional[List], model_provider: str) -> Chroma:
    """
    #### Loads or creates Chroma vectorstore
    Always includes Plato base + optional PDFs
    """

    embedding = get_embeddings(model_provider)
    persist_path = PERSIST_DIR.get(model_provider, f"./data/{model_provider.lower().replace(' ', '_')}_store.chroma")
    
    # Normalize collection name for ChromaDB
    collection_name = normalize_collection_name(model_provider)

    #### 1️⃣ Load existing vectorstore safely
    vectorstore = None
    plato_exists = False

    if os.path.exists(persist_path) and os.listdir(persist_path):
        try:
            print(f"🔄 Loading existing vectorstore from {persist_path}")
            vectorstore = Chroma(
                persist_directory=persist_path,
                embedding_function=embedding,
                collection_name=collection_name
            )

            #### Check if Plato JSON is already loaded
            existing_sources = set()
            try:
                results = vectorstore.get(limit=10000)
                existing_sources = {doc.get("source", "") for doc in results.get("metadatas", [])}
            except Exception:
                pass

            plato_exists = "platon_analisis_nlp.json" in existing_sources

        except Exception as e:
            print(f"⚠️ Error loading vectorstore: {e}. Creating new one.")
            vectorstore = None
            plato_exists = False
    else:
        print(f"🆕 Creating new vectorstore at {persist_path}")

    #### 2️⃣ Prepare documents to add
    documents_to_add = []

    #### Add Plato documents if not present
    if not plato_exists:
        plato_docs = get_plato_documents()
        if plato_docs:
            documents_to_add.extend(plato_docs)
            print(f"🧠 Adding Platonic base: {len(plato_docs)} fragments")
    else:
        print(f"✓ Platonic base already loaded")

    #### Add PDFs if provided
    if uploaded_files:
        try:
            for pdf in uploaded_files:
                raw_text = get_pdf_text([pdf])
                chunks = get_text_chunks(raw_text)

                pdf_documents = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": f"uploaded_pdf_{pdf.name}",
                            "chunk_id": i,
                            "type": "user_pdf",
                            "provider": model_provider,
                            "timestamp": str(datetime.now())
                        }
                    )
                    for i, chunk in enumerate(chunks)
                ]

                documents_to_add.extend(pdf_documents)
                print(f"📄 Adding PDF '{pdf.name}' chunks: {len(pdf_documents)}")

        except Exception as e:
            print(f"❌ Error processing PDFs: {e}")

    #### 3️⃣ Create or update vectorstore
    if not documents_to_add:
        if vectorstore:
            print("✓ No new documents to add")
            return vectorstore
        else:
            # Create empty vectorstore if none exists
            print("⚠️ Creating empty vectorstore structure")
            return Chroma(
                persist_directory=persist_path,
                embedding_function=embedding,
                collection_name=collection_name
            )

    if vectorstore is None:
        #### Create new vectorstore
        try:
            vectorstore = Chroma.from_documents(
                documents=documents_to_add,
                embedding=embedding,
                persist_directory=persist_path,
                collection_name=collection_name
            )
            print(f"✅ Created new vectorstore with {len(documents_to_add)} documents")
        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            # Fallback: in-memory vectorstore
            vectorstore = Chroma.from_documents(
                documents=documents_to_add,
                embedding=embedding,
                collection_name=f"{collection_name}_fallback"
            )
            print("⚠️ Using in-memory vectorstore (not persisted)")
    else:
        #### Add new documents to existing vectorstore
        try:
            vectorstore.add_documents(documents_to_add)
            print(f"✅ Added {len(documents_to_add)} new documents to existing vectorstore")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")

    return vectorstore