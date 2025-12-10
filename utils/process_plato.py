#!/usr/bin/env python3
"""
process_plato.py
----------------
Semantic Analysis of Plato's texts for RAG/LLM pipelines.
Uses spaCy transformer model in Spanish with GPU support.
"""

import json
from pathlib import Path
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
from typing import List, Dict

# ------------------------------
# Config
# ------------------------------
INPUT_DIR = Path("./plato_texts")  # carpeta con textos de Plato en txt
OUTPUT_JSON = Path("./data/platon_analisis_nlp.json")
SPACY_MODEL = "es_dep_news_trf"

# Lista de conceptos filosóficos a extraer (ejemplo, puedes expandir)
PHILOSOPHICAL_CONCEPTS = [
    "alma", "cuerpo", "virtud", "bien", "justicia", "sabiduría",
    "conocimiento", "verdad", "belleza", "forma", "idea", "estado"
]

# ------------------------------
# Load spaCy model with GPU
# ------------------------------
print("🖥️ Loading spaCy model...")
import torch
print("GPU disponible:", torch.cuda.is_available())
print("GPU actual:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import spacy
spacy.require_gpu()  # usa GPU
nlp = spacy.load(SPACY_MODEL)

# ------------------------------
# Process each file
# ------------------------------
documents = []

for txt_file in tqdm(INPUT_DIR.glob("*.txt"), desc="Processing Plato texts"):
    text = txt_file.read_text(encoding="utf-8")
    
    doc: Doc = nlp(text)
    
    # Extract basic stats
    sentences = list(doc.sents)
    avg_sentence_len = sum(len(sent) for sent in sentences) / max(len(sentences), 1)
    
    # Named entities
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    
    # Concept frequencies (simple count)
    text_lower = text.lower()
    concepts_found = [c for c in PHILOSOPHICAL_CONCEPTS if c.lower() in text_lower]
    
    # Build document JSON
    documents.append({
        "titulo": txt_file.stem,
        "tipo": "dialogo" if "dialogue" in txt_file.stem.lower() else "escrito",
        "texto": text,
        "conceptos_filosoficos": [{"concepto": c} for c in concepts_found],
        "entidades": entities,
        "complejidad_sintactica": {
            "avg_sentence_length": avg_sentence_len,
            "n_sentences": len(sentences)
        },
        "dialogo": txt_file.stem,
        "libro": ""
    })

# ------------------------------
# Save JSON
# ------------------------------
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"✅ JSON generado con {len(documents)} documentos en {OUTPUT_JSON}")
