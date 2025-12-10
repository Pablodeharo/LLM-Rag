import streamlit as st
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chat_handler import (
    setup_session_state,
    render_chat_history,
    render_download_chat_history,
    handle_user_input,
    render_uploaded_files_expander
)
from utils.sidebar_handler import (
    render_model_selector,
    sidebar_file_upload,
    sidebar_provider_change_check,
    sidebar_utilities
)


def main():
    """
    Main entry point for the RAG PDFBot Streamlit app.
    Optimized to avoid pickling issues with LLMs and Vectorstores.
    """
    st.set_page_config(page_title="RAG PDFBot", layout="centered")
    st.title("👽 RAG PDFBot")
    st.caption("Chat with multiple PDFs :books:")

    # -------------------------------
    # 1️⃣ Initialize session state
    # -------------------------------
    setup_session_state()

    # -------------------------------
    # 2️⃣ Sidebar: model selection, PDF upload, utilities
    # -------------------------------
    with st.sidebar:
        with st.expander("⚙️ Configuration", expanded=True):
            model_provider, model = render_model_selector()
            sidebar_file_upload(model_provider)
            sidebar_provider_change_check(model_provider, model)
        sidebar_utilities()

    # -------------------------------
    # 3️⃣ Download chat history if exists
    # -------------------------------
    if st.session_state.chat_history:
        render_download_chat_history()

    # -------------------------------
    # 4️⃣ Show info if no PDFs uploaded
    # -------------------------------
    uploaded_files_key = f"uploaded_files_{st.session_state.uploader_key}"
    current_files = st.session_state.get(uploaded_files_key, [])

    # Improved message
    if not st.session_state.get("vector_store"):
        st.info("""
        🧠 **RAG PDFBot with Platonic Knowledge**
    
        **Included knowledge base:**
        • Semantic analysis of Plato's dialogues
        • Structured philosophical concepts
    
        **You can:**
        1. **Ask questions directly** about Platonic philosophy
        2. **Upload additional PDFs** to enrich context
        3. **Switch between models** (Groq, Gemini, Spanish-LLM)
    
        ⚠️ You don't need to upload PDFs to start.
        """)
    elif st.session_state.get("unsubmitted_files", False):
        st.warning("📄 New PDFs uploaded. Click 'Submit' to process them along with Plato.")

    # -------------------------------
    # 5️⃣ Show uploaded files summary
    # -------------------------------
    # FIXED: Removed vectorstore_path condition - use vector_store from session state
    if st.session_state.get("pdf_files", []):
        render_uploaded_files_expander()

    # -------------------------------
    # 6️⃣ Render chat history
    # -------------------------------
    if st.session_state.get("chat_history", []):
        render_chat_history()

    # -------------------------------
    # 7️⃣ Handle user input and LLM
    # -------------------------------
    vectorstore = st.session_state.get("vector_store")
    model_provider = st.session_state.get("model_provider", "").lower()
    model = st.session_state.get("model", "")

    # SIMPLIFIED CONDITION: vectorstore always exists (with Plato)
    if vectorstore and model and model_provider:
        try:
            from utils.llm_handler import get_llm_chain
            from utils.developer_mode import inspect_vectorstore
        
            llm_chain = get_llm_chain(model_provider, model, vectorstore)
            handle_user_input(model_provider, model, llm_chain)
        
            # Developer mode toggle button (moved to sidebar in next version)
            if st.session_state.get("show_developer_mode", False):
                inspect_vectorstore(vectorstore)
            
        except Exception as e:
            st.error(f"LLM chain error: {str(e)}")
            st.info("Check your API keys and connection.")

    # -------------------------------
    # 8️⃣ Developer mode toggle button in sidebar
    # -------------------------------
    with st.sidebar:
        if st.session_state.get("vector_store"):
            if st.button("🔧 Toggle Developer Mode"):
                st.session_state.show_developer_mode = not st.session_state.get("show_developer_mode", False)
                st.rerun()


if __name__ == "__main__":
    main()