#!/usr/bin/env python3
"""
Setup SpaCy with GPU for Spanish Transformers
---------------------------------------------

This script:
1. Installs CuPy compatible with CUDA 12
2. Installs spaCy and required dependencies
3. Downloads the Spanish Transformer model
4. Checks GPU availability for spaCy
"""

import subprocess
import sys

def run(cmd):
    """Run a shell command."""
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        sys.exit(1)

# ------------------------------
# 1️⃣ Install packages
# ------------------------------
print("🚀 Installing CuPy for CUDA 12...")
run("pip install --upgrade pip")
run("pip install cupy-cuda12x")

print("🚀 Installing SpaCy and transformers...")
run("pip install spacy[transformers] torch torchvision torchaudio --upgrade")

# ------------------------------
# 2️⃣ Download Spanish transformer model
# ------------------------------
print("📥 Downloading Spanish Transformer model...")
run("python -m spacy download es_dep_news_trf")

# ------------------------------
# 3️⃣ Test GPU availability
# ------------------------------
print("🖥️ Testing GPU availability in SpaCy...")
import spacy
try:
    spacy.require_gpu()
    print("✅ SpaCy is using GPU!")
except Exception as e:
    print(f"⚠️ SpaCy GPU not available: {e}")

# ------------------------------
# 4️⃣ Load model to verify
# ------------------------------
try:
    nlp = spacy.load("es_dep_news_trf")
    print("✅ Model loaded successfully:", nlp.meta['name'])
except Exception as e:
    print(f"❌ Failed to load model: {e}")
