"""
LLM Handler Module
---------------------------------------------
Pipeline-style RAG / Conversational chain
Spanish LLM local (Mistral/LLamaCpp) activo
Groq y Gemini comentados
"""

import os
import torch
from utils.prompts import get_socratic_prompt, select_prompt_by_question_type

#### LangChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

#### Providers
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_groq import ChatGroq
from langchain_community.llms import LlamaCpp
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ================================
# LLM initialization
# ================================
def get_llm_instance(model_provider: str, model: str):
    """
    #### Initialize LLM instance based on provider
    Only local Spanish LLM enabled
    """

    try:
        #### Spanish LLM local / Hugging Face
        if model_provider == "Spanish LLM":
            # Local GGUF Mistral / LlamaCpp
            model_path = "./models/eva-mistral-7b-spanish.Q4_K_M.gguf"
            return LlamaCpp(
                model_path=model_path,
                n_ctx=2048,
                temperature=0.7,
                top_p=0.95,
                verbose=False,
                n_gpu_layers=33,  #### Ajustar según GPU
                max_tokens=500,
                stop=["Usuario:", "Human:"]
            )

        #### Groq
        # elif model_provider == "Groq":
        #     return ChatGroq(model=model, api_key="YOUR_KEY", temperature=0.7, max_tokens=500)

        #### Google Gemini
        # elif model_provider == "Gemini":
        #     return ChatGoogleGenerativeAI(model=model, api_key="YOUR_KEY", temperature=0.7, convert_system_message_to_human=True)

        else:
            raise ValueError(f"Unsupported provider: {model_provider}")

    except Exception as e:
        print(f"❌ Error initializing LLM ({model_provider} - {model}): {e}")
        return None

# ================================
# Retrieval-Augmented Generation Chain
# ================================
def get_llm_chain(model_provider: str, model: str, vectorstore, user_question: str = None, use_smart_prompts: bool = True):
    """
    #### Builds a RAG chain (LLM + vectorstore)
    """

    if not model:
        return None

    #### Initialize LLM
    llm = get_llm_instance(model_provider, model)
    if llm is None:
        return None

    #### Select Socratic prompt
    if use_smart_prompts and user_question:
        try:
            prompt = select_prompt_by_question_type(user_question)
        except:
            prompt = get_socratic_prompt(use_history=False)
    else:
        prompt = get_socratic_prompt(use_history=False)

    #### Configure retriever
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
    except Exception as e:
        print(f"❌ Error configuring retriever: {e}")
        return None

    #### Create RAG chain
    try:
        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(llm, prompt)
        )
        return chain
    except Exception as e:
        print(f"❌ Error creating retrieval chain: {e}")
        return None

# ================================
# Conversational Chain with memory
# ================================
def get_conversational_chain(model_provider: str, model: str, vectorstore, memory: ConversationBufferMemory, user_question: str = None, use_smart_prompts: bool = True):
    """
    #### Creates a conversational RAG chain that keeps history
    """

    #### Initialize LLM
    llm = get_llm_instance(model_provider, model)
    if llm is None:
        return None

    #### Select Socratic prompt
    if use_smart_prompts and user_question:
        try:
            prompt = select_prompt_by_question_type(user_question)
        except:
            prompt = get_socratic_prompt(use_history=True)
    else:
        prompt = get_socratic_prompt(use_history=True)

    #### Configure retriever
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
    except Exception as e:
        print(f"❌ Error configuring retriever: {e}")
        return None

    #### Create conversational retrieval chain
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        return chain
    except Exception as e:
        print(f"❌ Error creating conversational chain: {e}")
        return None
