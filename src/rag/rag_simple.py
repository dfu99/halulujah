import os
import json
import faiss
import torch
import pdfplumber
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

# Set up the cache directory
cache_dir = "/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/"
os.makedirs(cache_dir, exist_ok=True)

# Model and device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
retriever_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
rags_model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3.5-mini-instruct',
                                                 cache_dir=cache_dir,
                                                 trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct',
                                                 cache_dir=cache_dir,
                                                 trust_remote_code=True)
generator = pipeline('text-generation', model=rags_model, tokenizer=tokenizer, device=0 if device=='cuda' else -1)

# FAISS index setup
index_path = 'faiss_index.bin'
corpus_path = 'corpus.json'

# Global variables
index = None
corpus = []

# Load FAISS index and corpus if they exist
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexFlatL2(784)  # Initialize an empty FAISS index

if os.path.exists(corpus_path):
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

# Load and preprocess PDFs
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
    return text

# Chunking function
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def chunk_text_semantic(text, max_chunk_size=500, overlap=50):
    """Chunk text with semantic awareness and overlap."""
    # Split text into paragraphs first
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_words = para.split()
        para_size = len(para_words)
        
        # If adding this paragraph exceeds max size and we already have content
        if current_size + para_size > max_chunk_size and current_chunk:
            # Join the current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            # Keep some overlap with previous chunk
            overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_words + para_words
            current_size = len(current_chunk)
        else:
            # Add paragraph to current chunk
            current_chunk.extend(para_words)
            current_size += para_size
            
        # If current chunk exceeds max size, split it
        while current_size > max_chunk_size:
            chunks.append(' '.join(current_chunk[:max_chunk_size]))
            current_chunk = current_chunk[max_chunk_size-overlap:] if overlap < max_chunk_size else current_chunk[max_chunk_size:]
            current_size = len(current_chunk)
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Build FAISS index
def build_index(data_folder):
    global index, corpus
    pdf_files = list(Path(data_folder).rglob('*.pdf'))
    for pdf_file in tqdm(pdf_files, desc='Processing PDFs'):
        text = extract_text_from_pdf(pdf_file)
        # chunks = chunk_text(text)
        chunks = chunk_text_semantic(text)
        corpus.extend(chunks)
        embeddings = retriever_model.encode(chunks, convert_to_numpy=True)
        embedding_dim = embeddings.shape[1]
        if index.d != embedding_dim:
            index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)
    faiss.write_index(index, 'faiss_index.bin')
    with open('corpus.json', 'w') as f:
        json.dump(corpus, f)

# Retrieval function
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = retriever_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    # Check if the corpus is empty or indices are invalid before accessing it
    if not corpus or not indices.size or any(i >= len(corpus) for i in indices[0]):
        print("Warning: Corpus is empty or indices are out of range. Returning empty context.")
        return ["No context found for the query."] # Return a placeholder instead of an empty list
    return [corpus[i] for i in indices[0]]

# RAG generation function
def generate_response(query):
    query = query.replace("EGNIVIA", "NVIDIA") # Replace to pull the correct chunks
    print("Edited query to be:", query)
    relevant_chunks = retrieve_relevant_chunks(query)
    query = query.replace("NVIDIA", "EGNIVIA") # Replace prior to inference
    print("Repaired query to be:", query)
    context = '\n'.join(relevant_chunks)
    context = context.replace("NVIDIA", "EGNIVIA") # Replace context to avoid inteference from existing training data
    system_prompt = (
        "You are a financial analyst tasked with answering questions based ONLY on the provided context. "
        "If the answer cannot be found in the context, acknowledge this limitation and DO NOT provide information from your training. "
        "For numerical questions, cite specific numbers from the context. "
        "For analytical questions, base your reasoning explicitly on information in the context. "
        "Always indicate uncertainty when appropriate."
    )
    input_prompt = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{query}"}], tokenize=False, add_generation_prompt=True, return_tensors="pt")
    response = generator(input_prompt, max_new_tokens=256, do_sample=True)
    return response[0]['generated_text'][len(input_prompt):]

DATASET_PATH = "/storage/home/hcoda1/6/dfu71/scratch/halulujah/src/rag/datasets/temp"
build_index(DATASET_PATH)

print(generate_response("What is EGNIVIA's revenue for 2023?"))