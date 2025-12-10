"""
Platonic RAG Chatbot
--------------------
Socratic dialogue assistant powered by RAG (Retrieval-Augmented Generation)
Specialized in Plato's philosophy with optional PDF augmentation.

Author: [Tu Nombre]
Project: Portfolio LLM + RAG System
"""

import streamlit as st
from datetime import datetime

# Local modules
from utils.config import MODEL_OPTIONS
from utils.chat_handler import (
    setup_session_state,
    render_chat_history,
    handle_user_input,
    render_uploaded_files_expander,
    render_download_chat_history
)
from utils.sidebar_handler import (
    render_model_selector,
    sidebar_file_upload,
    sidebar_provider_change_check,
    sidebar_utilities
)
from utils.llm_handler import get_llm_chain
from utils.developer_mode import inspect_vectorstore


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Platonic RAG Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/rag-bot-chroma',
        'Report a bug': 'https://github.com/tu-usuario/rag-bot-chroma/issues',
        'About': """
        # Platonic RAG Assistant 🏛️
        
        A conversational AI assistant that applies the **Socratic Method** 
        to explore Plato's philosophy using Retrieval-Augmented Generation (RAG).
        
        **Tech Stack:**
        - LangChain for RAG orchestration
        - ChromaDB for vector storage
        - Local Spanish LLM (Mistral 7B)
        - Streamlit for UI
        
        **Features:**
        - Philosophical dialogue in Spanish
        - Pre-loaded Platonic corpus (Republic, Phaedo, etc.)
        - PDF upload for extended context
        - Conversational memory
        
        Version: 1.0.0
        """
    }
)


# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    /* ===== GLOBAL THEME ===== */
    :root {
        --primary-color: #3498DB;
        --secondary-color: #2ECC71;
        --accent-color: #E74C3C;
        --bg-dark: #1E1E1E;
        --bg-medium: #2D2D2D;
        --bg-light: #3A3A3A;
        --text-primary: #E0E0E0;
        --text-secondary: #B0B0B0;
        --border-color: #4A4A4A;
    }
    
    /* ===== MAIN CONTAINER ===== */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem 1rem;
    }
    
    /* ===== HEADER STYLING ===== */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5em;
        font-weight: 800;
        margin-bottom: 0.3em;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.3em;
        margin-bottom: 2em;
        font-style: italic;
        font-weight: 300;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #E2E8F0 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stButton label {
        color: #CBD5E1 !important;
        font-weight: 500;
    }
    
    /* ===== CHAT MESSAGES ===== */
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem !important;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%) !important;
        border-left: 3px solid #667eea;
    }
    
    .stChatMessage[data-testid="assistant"] {
        background: linear-gradient(135deg, #2ecc7115 0%, #3498db15 100%) !important;
        border-left: 3px solid #2ecc71;
    }
    
    /* ===== CHAT INPUT ===== */
    .stChatInput {
        border: 2px solid #475569 !important;
        border-radius: 12px !important;
        background-color: #1E293B !important;
        transition: all 0.3s ease;
    }
    
    .stChatInput:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stChatInput input {
        color: #E2E8F0 !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:disabled {
        background: #475569;
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {
        background-color: #1E293B !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background-color: #1E293B;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background-color: #1a2332;
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #233044 !important;
        border-color: #667eea !important;
    }
    
    .streamlit-expanderContent {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* ===== INFO/WARNING BOXES ===== */
    .info-box {
        padding: 1.5em;
        border-radius: 12px;
        background: linear-gradient(135deg, #3498db15 0%, #2ecc7115 100%);
        border-left: 4px solid #3498DB;
        margin: 1.5em 0;
        backdrop-filter: blur(10px);
        color: #E2E8F0;
    }
    
    .warning-box {
        padding: 1.5em;
        border-radius: 12px;
        background: linear-gradient(135deg, #f39c1215 0%, #e74c3c15 100%);
        border-left: 4px solid #F39C12;
        margin: 1.5em 0;
        backdrop-filter: blur(10px);
        color: #E2E8F0;
    }
    
    /* ===== SUCCESS/ERROR MESSAGES ===== */
    .stSuccess {
        background-color: #0f5132 !important;
        border-left: 4px solid #2ecc71 !important;
        color: #d1f2eb !important;
    }
    
    .stError {
        background-color: #58151c !important;
        border-left: 4px solid #e74c3c !important;
        color: #f8d7da !important;
    }
    
    .stWarning {
        background-color: #664d03 !important;
        border-left: 4px solid #f39c12 !important;
        color: #fff3cd !important;
    }
    
    .stInfo {
        background-color: #055160 !important;
        border-left: 4px solid #3498db !important;
        color: #cfe2ff !important;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #475569, transparent);
        margin: 2rem 0;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1E293B;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* ===== LINKS ===== */
    a {
        color: #667eea !important;
        text-decoration: none !important;
        transition: all 0.3s ease;
    }
    
    a:hover {
        color: #764ba2 !important;
        text-decoration: underline !important;
    }
    
    /* ===== FOOTER ===== */
    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 0.9em;
        padding: 2em 0;
        border-top: 1px solid #334155;
        margin-top: 3rem;
    }
    
    .footer-text a {
        color: #667eea !important;
        font-weight: 600;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5em;
        }
        
        .subtitle {
            font-size: 1.1em;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application logic
    """
    
    # Initialize session state
    setup_session_state()
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    
    st.markdown('<h1 class="main-title">🏛️ Platonic RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Diálogo socrático potenciado por IA y recuperación de conocimiento</p>', 
        unsafe_allow_html=True
    )
    
    # Project description
    with st.expander("ℹ️ Sobre este proyecto", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Propósito
            
            Este asistente aplica el **método socrático** para explorar la filosofía 
            de Platón mediante diálogos interactivos. Utiliza RAG (Retrieval-Augmented 
            Generation) para fundamentar sus respuestas en textos platónicos auténticos.
            
            ### 🧠 Características técnicas
            
            - **Base de conocimiento**: Corpus pre-procesado de diálogos platónicos 
              (República, Fedón, Banquete, etc.) con análisis NLP
            - **Vector Store**: ChromaDB con embeddings multilingües optimizados para español
            - **LLM Local**: Mistral 7B Spanish (eva-mistral) para inferencia local
            - **Memoria conversacional**: Mantiene contexto del diálogo para respuestas coherentes
            - **Extensible**: Permite cargar PDFs adicionales para ampliar el contexto
            
            ### 🎓 Metodología socrática
            
            El asistente **no da respuestas directas**, sino que:
            1. Examina las premisas de tus preguntas
            2. Genera aporía (perplejidad productiva)
            3. Guía hacia la anámnesis (recuerdo del conocimiento)
            4. Ofrece síntesis dialéctica citando fuentes platónicas
            """)
        
        with col2:
            st.info("""
            **🎓 Método Socrático**
            
            - ❓ No da respuestas directas
            - 🤔 Examina premisas
            - 💡 Genera perplejidad productiva
            - 📚 Cita fuentes platónicas
            """)
    
    # ========================================================================
    # SIDEBAR: MODEL SELECTION & FILE UPLOAD
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Model selection
        model_provider, model = render_model_selector()
        
        # Display model info
        if model_provider and model:
            playground = MODEL_OPTIONS[model_provider].get("playground", "")
            if playground and playground != "local":
                st.markdown(f"[🔗 Playground]({playground})")
            
            st.divider()
        
        # File upload section
        st.header("📚 Fuentes de conocimiento")
        
        # Info about Plato base
        if st.session_state.get("plato_loaded"):
            st.success("✅ Base platónica cargada")
        else:
            st.info("💡 La base platónica se cargará al enviar")
        
        # PDF upload
        uploaded_files, submitted = sidebar_file_upload(model_provider)
        
        # Handle provider changes
        if model_provider:
            sidebar_provider_change_check(model_provider, model)
        
        st.divider()
        
        # Utilities
        sidebar_utilities()
        
        st.divider()
        
        # Download chat history
        render_download_chat_history()
        
        # Developer mode (only if vectorstore exists)
        if st.session_state.get("vector_store"):
            inspect_vectorstore(st.session_state.vector_store)
    
    # ========================================================================
    # MAIN CHAT AREA
    # ========================================================================
    
    # Display uploaded files if any
    render_uploaded_files_expander()
    
    # Warning if files not submitted
    if st.session_state.get("unsubmitted_files"):
        st.warning("⚠️ Tienes archivos sin enviar. Haz clic en **Submit** en la barra lateral.")
    
    # Warning if no model selected
    if not model:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Selecciona un modelo</strong><br>
            Por favor, elige un proveedor y modelo en la barra lateral para comenzar el diálogo.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Warning if no vectorstore
    if not st.session_state.get("vector_store") and not st.session_state.get("plato_loaded"):
        st.markdown("""
        <div class="info-box">
            <strong>💡 ¿Cómo empezar?</strong><br>
            1. Selecciona un modelo en la barra lateral<br>
            2. (Opcional) Sube PDFs adicionales<br>
            3. Haz clic en <strong>Submit</strong> para cargar la base platónica<br>
            4. ¡Comienza tu diálogo socrático!
        </div>
        """, unsafe_allow_html=True)
    
    # Render chat history
    render_chat_history()
    
    # ========================================================================
    # LLM CHAIN SETUP & USER INPUT HANDLING
    # ========================================================================
    
    # Create LLM chain if vectorstore exists
    chain = None
    if st.session_state.get("vector_store"):
        try:
            chain = get_llm_chain(
                model_provider=model_provider,
                model=model,
                vectorstore=st.session_state.vector_store,
                use_smart_prompts=True
            )
        except Exception as e:
            st.error(f"❌ Error creando cadena LLM: {str(e)}")
    
    # Handle user input
    handle_user_input(model_provider, model, chain)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("""
    <div class="footer-text">
        🏛️ <strong>Platonic RAG Assistant</strong> | 
        <a href="https://github.com/tu-usuario/rag-bot-chroma" target="_blank">GitHub</a> | 
        <a href="https://tu-portfolio.com" target="_blank">Portfolio</a>
        <br><br>
        <span style="font-size: 0.85em; color: #64748B;">
            Construido con 💜 usando LangChain • ChromaDB • Streamlit • Mistral 7B
        </span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ### ❌ Error crítico
        
        Ha ocurrido un error inesperado:
        ```
        {str(e)}
        ```
        
        Por favor, intenta:
        1. Recargar la página
        2. Limpiar el chat con el botón 🧹 en la barra lateral
        3. Reiniciar la aplicación con el botón 🔄
        
        Si el problema persiste, revisa los logs del servidor.
        """)
        
        # Log error for debugging
        import traceback
        st.code(traceback.format_exc())