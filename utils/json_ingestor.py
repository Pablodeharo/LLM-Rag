"""
JSON Ingestor Module
--------------------
Converts Plato's semantic analysis JSON to LangChain Documents.
"""

import json
from typing import List
from langchain_core.documents import Document


def load_platon_json(filepath: str) -> List[Document]:
    """
    Loads Plato analysis JSON and converts to LangChain documents.
    
    Args:
        filepath: Path to JSON file with analysis
        
    Returns:
        List of LangChain Document objects
        
    Expected JSON structure:
    [
        {
            "titulo": "The Republic - Book VII",
            "tipo": "dialogue",
            "texto": "...",
            "conceptos_filosoficos": [...],
            "analisis_spacy": {...},
            "dialogo": "Republic",
            "libro": "Book VII"
        }
    ]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    for i, item in enumerate(data):
        # Extract metadata
        metadata = {
            'source': 'platon_analisis_nlp.json',
            'titulo': item.get('titulo', 'Unknown'),
            'tipo': item.get('tipo', 'fragment'),
            'dialogo': item.get('dialogo', ''),
            'libro': item.get('libro', ''),
            'chunk_id': i,
            
            # Philosophical concepts (comma-separated for filtering)
            'conceptos': ', '.join([
                c.get('concepto', '') 
                for c in item.get('conceptos_filosoficos', [])
            ]) if 'conceptos_filosoficos' in item else '',
            
            # Complexity metrics
            'complejidad_sintactica': item.get('analisis_spacy', {})
                                          .get('complejidad_sintactica', {})
                                          .get('avg_sentence_length', 0),
        }
        
        # Create document
        doc = Document(
            page_content=item.get('texto', ''),
            metadata=metadata
        )
        
        documents.append(doc)
    
    if not documents:
        raise ValueError(f"No documents loaded from {filepath}")
    
    print(f"📚 Loaded {len(documents)} Plato documents from JSON")
    return documents