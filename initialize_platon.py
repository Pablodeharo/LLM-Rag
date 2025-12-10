#!/usr/bin/env python3
"""
Quick Initialization Script
Loads Plato JSON into all vectorstores
"""

import sys
import os
import shutil

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def initialize_all_stores():
    """Initializes vectorstores for all providers"""
    
    print("=" * 60)
    print("🧠 PLATONIC SYSTEM INITIALIZATION")
    print("=" * 60)
    
    # Import functions
    from utils.vectorstore_handler import (
        get_embeddings, 
        get_plato_documents,
        PERSIST_DIR
    )
    from langchain_community.vectorstores import Chroma
    
    # Load Plato documents (once)
    plato_docs = get_plato_documents()
    if not plato_docs:
        print("❌ Could not load Plato. Check the JSON.")
        return
    
    print(f"📚 Platonic corpus: {len(plato_docs)} fragments")
    
    # For each provider
    providers = ["groq", "gemini", "spanish-llm"]
    
    for provider in providers:
        print(f"\n🔧 Processing: {provider.upper()}")
        
        # 1. Remove old vectorstore (optional)
        persist_path = PERSIST_DIR.get(provider)
        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)
            print(f"   🗑️  Previous vectorstore removed")
        
        # 2. Get embeddings
        try:
            embedding = get_embeddings(provider)
        except Exception as e:
            print(f"   ❌ Embeddings error {provider}: {e}")
            continue
        
        # 3. Create new vectorstore
        try:
            vectorstore = Chroma.from_documents(
                documents=plato_docs,
                embedding=embedding,
                persist_directory=persist_path,
                collection_name=f"platonic_base_{provider}",
                persist=True
            )
            
            doc_count = vectorstore._collection.count()
            print(f"   ✅ Vectorstore created: {doc_count} documents")
            
        except Exception as e:
            print(f"   ❌ Error creating vectorstore: {e}")
    
    print("\n" + "=" * 60)
    print("✅ INITIALIZATION COMPLETED")
    print("=" * 60)
    print("\nYou can run the app:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    initialize_all_stores()