"""
LLM Handler Module
------------------

Handles LLM model initialization and RAG chain creation.

"""


from utils.config import GOOGLE_API_KEY, GROQ_API_KEY
from utils.prompts import get_socratic_prompt, select_prompt_by_question_type

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.llms import LlamaCpp

def get_llm_instance(model_provider: str, model: str):
    """
    Initializes and returns an LLM instance based on provider.
    
    Args:
        model_provider: Model provider ("Groq", "Gemini", "Spanish llm")
        model: Specific model name
        
    Returns:
        Configured LLM instance
        
    Raises:
        ValueError: If provider is not supported
    """
    if model_provider == "Groq":
        return ChatGroq(
            model=model, 
            api_key=GROQ_API_KEY,
            temperature=0.7,  # creativity
            max_tokens=500    # Concise responses
        ) # pop_k=
    
    elif model_provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model, 
            api_key=GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    elif model_provider == "Spanish llm":
        model_path = "./models/eva-mistral-7b-spanish.Q4_K_M.gguf"
        
        return LlamaCpp(
            model_path=model_path,
            n_ctx=2048,
            temperature=0.7,
            top_p=0.95,
            verbose=False,
            n_gpu_layers=33,
            max_tokens=500,
            stop=["Usuario:", "Human:"]
        )
    
    else:
        raise ValueError(f"Unsupported provider: {model_provider}")


def get_llm_chain(model_provider: str, model: str, vectorstore, use_smart_prompts: bool = True):
    """
    Builds a RAG (Retrieval-Augmented Generation) chain with Socratic method.
    
    Args:
        model_provider: LLM provider ("groq", "gemini", "spanish-llm")
        model: Specific model to use
        vectorstore: ChromaDB vectorstore for context retrieval
        use_smart_prompts: If True, select prompt based on question type
        
    Returns:
        LangChain chain ready for invoke() with {"input": "question"}
        
    Example:
        >>> chain = get_llm_chain("groq", "llama-3.1-8b-instant", vectorstore)
        >>> response = chain.invoke({"input": "¿Qué es la justicia?"})
        >>> print(response["answer"])
    """
    if not model:
        return None
    
    # Initialize LLM
    try:
        llm = get_llm_instance(model_provider, model)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None
    
    # Select Socratic prompt
    if use_smart_prompts:
        # This function will analyze the question and choose appropriate prompt
        prompt = get_socratic_prompt(use_history=False)
    else:
        # Use basic Socratic prompt
        prompt = get_socratic_prompt(use_history=False)
    
    # Configure retriever with optimized parameters
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,  # Top 3 most relevant chunks
            "fetch_k": 10  # Consider 10, return 3 (more precision)
        }
    )
    
    # Create modern RAG chain
    # Flow: question → retriever → context + question → LLM → response
    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )
    
    return chain


def get_conversational_chain(model_provider: str, model: str, vectorstore, memory):
    """
    Creates a conversational chain that maintains history (EXPERIMENTAL).
    
    Args:
        model_provider: LLM provider
        model: Specific model
        vectorstore: ChromaDB vectorstore
        memory: LangChain ConversationBufferMemory object
        
    Returns:
        Chain with conversational memory
        
    Note:
        To fully implement this you need:
        from langchain.memory import ConversationBufferMemory
        and adjust prompt to include MessagesPlaceholder
    """
    llm = get_llm_instance(model_provider, model)
    prompt = get_socratic_prompt(use_history=True)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # TODO: Implement with ConversationalRetrievalChain
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=retriever,
    #     memory=memory,
    #     combine_docs_chain_kwargs={"prompt": prompt}
    # )
    
    return create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )