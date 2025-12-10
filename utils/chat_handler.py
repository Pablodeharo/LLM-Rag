"""
Chat Section Module
--------------------
Handles chat functionalities: session state, rendering, user queries,
uploaded PDFs, and chat download.
"""

import pandas as pd
import streamlit as st
from datetime import datetime

def setup_session_state():
    """
    Initialize necessary Streamlit session state variables.
    """
    defaults = {
        "chat_history": [],             # List of tuples: (question, answer, provider, model, pdfs, timestamp)
        "vector_store": None,           # Chroma vectorstore instance
        "pdf_files": [],                # Uploaded PDF files
        "last_provider": None,          # Last selected provider
        "unsubmitted_files": False,     # Tracks unsubmitted PDFs
        "uploader_key": 0,              # For resetting file_uploader
        "plato_loaded": False           # Flag to indicate Plato base loaded
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_chat_history():
    """
    Display chat history with user and AI messages.
    """
    for question, answer, *_ in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("ai"):
            st.markdown(answer)

def handle_user_input(model_provider, model, chain):
    """
    Handles user input and invokes the LLM chain to get responses.
    """
    disable_input = (
        st.session_state.get("unsubmitted_files") or
        (not st.session_state.get("pdf_files") and not st.session_state.get("plato_loaded")) or
        not chain
    )

    question = st.chat_input(
        "💬 Ask a Question",
        disabled=disable_input
    )

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            try:
                output = chain.invoke({"input": question})["answer"]
                st.markdown(output)
                pdf_names = [f.name for f in st.session_state.get("pdf_files", [])]
                st.session_state.chat_history.append(
                    (question, output, model_provider, model, pdf_names, datetime.now())
                )
            except Exception as e:
                st.error(f"Error invoking LLM chain: {str(e)}")

def render_uploaded_files_expander():
    """
    Displays the uploaded PDFs in an expander if files exist and are submitted.
    """
    uploaded_files = st.session_state.get("pdf_files", [])
    if uploaded_files and not st.session_state.get("unsubmitted_files"):
        with st.expander("📎 Uploaded Files"):
            for f in uploaded_files:
                st.markdown(f"- {f.name}")

def render_download_chat_history():
    """
    Provides a download button for chat history as CSV.
    """
    if not st.session_state.get("chat_history"):
        return

    df = pd.DataFrame(
        st.session_state.chat_history,
        columns=["Question", "Answer", "Provider", "Model", "PDF Files", "Timestamp"]
    )

    with st.expander("📥 Download Chat History"):
        st.sidebar.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="chat_history.csv",
            mime="text/csv"
        )
