"""
Sidebar Section Module
----------------------
Handles sidebar: model selection, PDF upload, reprocessing, and utilities.
"""

import streamlit as st
from utils.config import MODEL_OPTIONS
from utils.vectorstore_handler import get_or_create_vectorstore

def render_model_selector():
    """
    Render dropdowns for provider and model selection.
    """
    provider = st.selectbox(
        "🔌 Model Provider",
        options=list(MODEL_OPTIONS.keys()),
        key="model_provider"
    )

    models = MODEL_OPTIONS.get(provider, {}).get("models", [])
    model = st.selectbox(
        "🧠 Select Model",
        options=models,
        key="model",
        disabled=(not provider)
    )

    return provider, model

def render_upload_files_button():
    """
    Handles PDF file upload and submission.
    """
    uploaded_files = st.file_uploader(
        "📚 Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploaded_files_{st.session_state.uploader_key}"
    )

    if uploaded_files != st.session_state.get("pdf_files"):
        st.session_state.unsubmitted_files = True

    submitted = st.button(
        "➡️ Submit",
        disabled=(not st.session_state.get("model"))
    )

    return uploaded_files, submitted

def sidebar_file_upload(model_provider):
    """
    Load Plato + user PDFs into vectorstore.
    """
    uploaded_files, submitted = render_upload_files_button()
    if submitted:
        with st.spinner("🧠 Loading Platonic knowledge..."):
            try:
                vector_store = get_or_create_vectorstore(
                    uploaded_files if uploaded_files else [],
                    model_provider
                )
                st.session_state.update(
                    vector_store=vector_store,
                    pdf_files=uploaded_files if uploaded_files else [],
                    unsubmitted_files=False,
                    plato_loaded=True
                )
                msg = f"✅ Plato + {len(uploaded_files)} PDF(s) processed" if uploaded_files else "✅ Platonic base loaded"
                st.toast(msg, icon="🧠")
            except Exception as e:
                st.error(f"Error loading vectorstore: {str(e)}")

    return uploaded_files, submitted

def sidebar_provider_change_check(model_provider, model):
    """
    Reprocess PDFs if provider changes.
    """
    last = st.session_state.get("last_provider")
    if model_provider != last and model:
        st.session_state.last_provider = model_provider
        if st.session_state.get("pdf_files"):
            with st.spinner(f"Reprocessing PDFs with {model_provider}..."):
                try:
                    vector_store = get_or_create_vectorstore(
                        st.session_state.get("pdf_files"),
                        model_provider
                    )
                    st.session_state.vector_store = vector_store
                    st.toast("PDFs reprocessed successfully!", icon="🔁")
                except Exception as e:
                    st.error(f"Error reprocessing PDFs: {str(e)}")

def sidebar_utilities():
    """
    Utilities: reset, clear chat, undo last message.
    """
    with st.expander("🛠️ Utilities", expanded=False):
        col1, col2, col3 = st.columns(3)

        if col1.button("🔄 Reset"):
            st.session_state.clear()
            st.session_state.model_provider = None
            st.rerun()

        if col2.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.update(pdf_files=None, vector_store=None)
            st.session_state.uploader_key += 1
            st.toast("Chat and PDFs cleared.", icon="🧼")
            st.rerun()

        if col3.button("↩️ Undo") and st.session_state.get("chat_history"):
            st.session_state.chat_history.pop()
            st.toast("Last message removed.", icon="↩️")
            st.rerun()
