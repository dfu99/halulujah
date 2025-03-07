import os
import json
import faiss
import torch
import pdfplumber
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

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

# Storage paths
DATA_PATH = 'src/rag/data/'
os.makedirs(DATA_PATH, exist_ok=True)
index_path = os.path.join(DATA_PATH, 'faiss_index.bin')
corpus_path = os.path.join(DATA_PATH, 'corpus.json')
metadata_path = os.path.join(DATA_PATH, 'metadata.json')
bm25_path = os.path.join(DATA_PATH, 'bm25_tokenized.json')

# Document class to store document chunks and metadata
class DocumentChunk:
    def __init__(self, text: str, doc_id: str, metadata: Dict):
        self.text = text
        self.doc_id = doc_id
        self.metadata = metadata
        
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentChunk':
        return cls(data["text"], data["doc_id"], data["metadata"])

# Global variables
index = None
corpus = []
metadata = []
bm25 = None
tokenized_corpus = []

# Load saved data if exists
def load_data():
    global index, corpus, metadata, bm25, tokenized_corpus
    
    if os.path.exists(index_path):
        print("Loading FAISS index...")
        index = faiss.read_index(index_path)
    else:
        print("Initializing new FAISS index...")
        index = faiss.IndexFlatL2(768)  # Default dimension for sentence-transformers/all-mpnet-base-v2
    
    if os.path.exists(corpus_path) and os.path.exists(metadata_path):
        print("Loading corpus and metadata...")
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    if os.path.exists(bm25_path):
        print("Loading BM25 index...")
        with open(bm25_path, 'r') as f:
            tokenized_corpus = json.load(f)
        bm25 = BM25Okapi(tokenized_corpus)

# Extract metadata from PDF filename and path
def extract_metadata_from_path(pdf_path: Path) -> Dict:
    """Extract metadata from PDF filename and path."""
    filename = pdf_path.name
    parent_dir = pdf_path.parent.name

    # Extract company name, year, and filing type using regex patterns
    company_match = re.search(r'([A-Za-z0-9\.\-]+)_', filename)
    year_match = re.search(r'_(\d{4})', filename)
    filing_match = re.search(r'_(10-[KQ])_', filename)

    company = company_match.group(1) if company_match else "unknown"
    year = str(year_match.group(1)) if year_match else None
    filing_type = filing_match.group(1) if filing_match else "unknown"

    return {
        "company": company,
        "year": year,
        "filing_type": filing_type,
        "source_path": str(pdf_path)
    }

# Extract text from PDF with section tracking
def extract_text_with_sections(pdf_path: Path) -> List[Tuple[str, Dict]]:
    """Extract text from PDF with section information."""
    sections = []
    current_section = "HEADER"

    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ''
            full_text += text + "\n"

            # Look for section markers in 10-K filings
            section_matches = re.finditer(r'ITEM\s+(\d+[A-Z]?)\.?\s+([^\n]+)', text, re.IGNORECASE)
            for match in section_matches:
                item_num = match.group(1)
                item_title = match.group(2).strip()
                current_section = f"ITEM {item_num}: {item_title}"

    # For simplicity, return the full text with basic metadata
    metadata = extract_metadata_from_path(pdf_path)
    metadata["section"] = current_section
    return [(full_text, metadata)]

# Chunking function with semantic awareness and section tracking
def chunk_text_semantic(text: str, metadata: Dict, max_chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
    """Chunk text with semantic awareness and overlap, preserving metadata."""
    # Split text into paragraphs first
    paragraphs = [p for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    # Generate a unique document ID
    doc_id = f"{metadata['company']}_{metadata['year']}_{metadata['filing_type']}"

    for para in paragraphs:
        para_words = para.split()
        para_size = len(para_words)

        # If adding this paragraph exceeds max size and we already have content
        if current_size + para_size > max_chunk_size and current_chunk:
            # Join the current chunk and add to chunks
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(chunk_text, doc_id, metadata.copy()))

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
            chunk_text = ' '.join(current_chunk[:max_chunk_size])
            chunks.append(DocumentChunk(chunk_text, doc_id, metadata.copy()))

            current_chunk = current_chunk[max_chunk_size-overlap:] if overlap < max_chunk_size else current_chunk[max_chunk_size:]
            current_size = len(current_chunk)

    # Add the last chunk if it has content
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(DocumentChunk(chunk_text, doc_id, metadata.copy()))

    return chunks

# Add numeric data detection for financial information
def detect_financial_data(text: str) -> Dict:
    """Detect and extract financial information from text."""
    financial_data = {
        "has_financials": False,
        "tables_detected": False,
        "years_mentioned": [],
        "currency_amounts": []
    }

    # Detect currency amounts (e.g., $1,000,000 or 1,000,000 dollars)
    currency_amounts = re.findall(r'\$\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\b(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:dollars|USD)\b', text)
    if currency_amounts:
        financial_data["currency_amounts"] = True

    # Detect table-like structures
    if re.search(r'\|\s*\w+\s*\|', text) or re.search(r'\+[-+]+\+', text):
        financial_data["tables_detected"] = True

    # If we found any financial indicators, mark the chunk accordingly
    if financial_data["currency_amounts"] or financial_data["tables_detected"]:
        financial_data["has_financials"] = True

    return financial_data

# Build FAISS and BM25 indices
def build_indices(data_folder: str):
    global index, corpus, metadata, bm25, tokenized_corpus

    pdf_files = list(Path(data_folder).rglob('*.pdf'))
    all_chunks = []

    # First pass: extract and chunk all documents
    for pdf_file in tqdm(pdf_files, desc='Processing PDFs'):
        sections = extract_text_with_sections(pdf_file)

        for section_text, section_metadata in sections:
            # Detect financial data and enrich metadata
            financial_metadata = detect_financial_data(section_text)
            section_metadata.update(financial_metadata)

            # Chunk the text with the enhanced metadata
            document_chunks = chunk_text_semantic(section_text, section_metadata)
            all_chunks.extend(document_chunks)

    # Process all chunks for embedding and tokenization
    texts = [chunk.text for chunk in all_chunks]

    # Create FAISS index with dense embeddings
    print("Generating embeddings and building FAISS index...")
    embeddings = retriever_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Ensure index has correct dimension
    embedding_dim = embeddings.shape[1]
    if index is None or index.d != embedding_dim:
        index = faiss.IndexFlatL2(embedding_dim)

    # Add embeddings to index
    index.add(embeddings)

    # Create BM25 index with sparse representations
    print("Building BM25 index...")
    tokenized_texts = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    tokenized_corpus = tokenized_texts

    # Save corpus and metadata
    corpus = texts
    metadata = [chunk.to_dict()["metadata"] for chunk in all_chunks]

    # Save all data to disk
    print("Saving indices and corpus...")
    faiss.write_index(index, index_path)

    with open(corpus_path, 'w') as f:
        json.dump(corpus, f)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    with open(bm25_path, 'w') as f:
        json.dump(tokenized_corpus, f)

    print(f"Processed {len(all_chunks)} chunks from {len(pdf_files)} documents")

# Filter by metadata with fuzzy matching
def filter_by_metadata(filter_params: Dict) -> List[int]:
    """Return indices of documents matching metadata filters with fuzzy matching."""
    matching_indices = []

    # Early return if no filters
    if not filter_params:
        return list(range(len(metadata)))

    # Print diagnostic information
    print(f"Filtering with params: {filter_params}")
    print(f"Total documents in corpus: {len(metadata)}")

    # Track matched documents for each filter type
    matches_by_filter = {}

    # First pass: Find matches for each individual filter
    for key, value in filter_params.items():
        matches_by_filter[key] = []

        for i, meta in enumerate(metadata):
            if key not in meta:
                continue

            # For company names, do case-insensitive substring matching
            if key == "company" and isinstance(value, str) and isinstance(meta[key], str):
                # Try exact match first
                if meta[key].lower() == value.lower():
                    matches_by_filter[key].append(i)
                # Then try substring match
                elif value.lower() in meta[key].lower() or meta[key].lower() in value.lower():
                    matches_by_filter[key].append(i)

            # For years, handle both exact matches and ranges
            elif key == "year":
                # Skip if the metadata has None for the year
                if meta[key] is None:
                    continue

                if isinstance(value, list):
                    # Convert everything to strings for comparison
                    meta_year_str = str(meta[key])
                    # Check if the document's year is in the requested years list
                    if meta_year_str in [str(y) for y in value]:
                        matches_by_filter[key].append(i)
                    # Also include adjacent years with string-safe comparison
                    elif any(abs(int(meta_year_str) - int(str(y))) <= 1 for y in value if y is not None):
                        matches_by_filter[key].append(i)
                elif meta[key] == str(value) or int(meta[key]) == value:
                    matches_by_filter[key].append(i)

            # For filing types, case-insensitive matching
            elif key == "filing_type" and isinstance(value, str) and isinstance(meta[key], str):
                if meta[key].upper() == value.upper():
                    matches_by_filter[key].append(i)

            # Fall back to exact matching for other fields
            elif isinstance(value, list):
                if meta[key] in value:
                    matches_by_filter[key].append(i)
            elif meta[key] == value:
                matches_by_filter[key].append(i)

    # Print diagnostics for each filter
    for key, matches in matches_by_filter.items():
        print(f"Filter '{key}': {len(matches)} matches")

    # If any filter has no matches, try to relax constraints
    empty_filters = [k for k, v in matches_by_filter.items() if not v]
    if empty_filters:
        print(f"Warning: No matches for filters: {empty_filters}")

        # For company filter, try more aggressive fuzzy matching if exact match failed
        if "company" in empty_filters and "company" in filter_params:
            company_value = filter_params["company"]
            for i, meta in enumerate(metadata):
                if "company" not in meta:
                    continue

                # Check if any word in the company name matches
                company_words = company_value.lower().split()
                meta_words = meta["company"].lower().split()

                for word in company_words:
                    if any(word in meta_word or meta_word in word for meta_word in meta_words if len(word) > 2 and len(meta_word) > 2):
                        matches_by_filter["company"].append(i)
                        break

            print(f"After relaxed company matching: {len(matches_by_filter['company'])} matches")

    # Second pass: Find documents that match all non-empty filters
    # If all filters are empty after relaxation, return a sample of documents
    if all(not matches for matches in matches_by_filter.values()):
        print("Warning: No documents match any filters. Returning a subset of all documents.")
        return list(range(min(100, len(metadata))))

    # Otherwise, combine the filters
    non_empty_filters = {k: v for k, v in matches_by_filter.items() if v}
    if non_empty_filters:
        # Start with all matches from the first non-empty filter
        first_key = next(iter(non_empty_filters))
        combined_matches = set(non_empty_filters[first_key])

        # Intersect with remaining non-empty filters
        for key, matches in non_empty_filters.items():
            if key != first_key:
                combined_matches &= set(matches)

        matching_indices = list(combined_matches)

        # If no documents match all filters, try progressive relaxation
        if not matching_indices:
            print("No documents match all filters. Attempting progressive relaxation.")

            # First try: match by company only (if available)
            if "company" in non_empty_filters:
                print("Prioritizing company matches.")
                matching_indices = non_empty_filters["company"]

            # Second try: match by filing type only (if available and first try failed)
            elif not matching_indices and "filing_type" in non_empty_filters:
                print("Prioritizing filing type matches.")
                matching_indices = non_empty_filters["filing_type"]

            # Third try: match by year only (if available and previous tries failed)
            elif not matching_indices and "year" in non_empty_filters:
                print("Prioritizing year matches.")
                matching_indices = non_empty_filters["year"]
    else:
        matching_indices = []

    print(f"Final matching documents: {len(matching_indices)}")

    # If still no matches, return a subset of all documents
    if not matching_indices:
        print("Falling back to returning a subset of all documents.")
        return list(range(min(50, len(metadata))))

    return matching_indices

# Hybrid retrieval function
def hybrid_retrieval(query: str, filter_params: Optional[Dict] = None, top_k: int = 5) -> List[Dict]:
    """Perform hybrid retrieval using both semantic and lexical search."""
    # Step 1: Apply metadata filtering if specified
    filtered_indices = None
    if filter_params:
        filtered_indices = set(filter_by_metadata(filter_params))
        if not filtered_indices:
            print("No documents match the specified filters.")
            return []

    # Step 2: Semantic search with FAISS
    query_embedding = retriever_model.encode([query], convert_to_numpy=True)
    if filtered_indices is not None:
        # Use IndexIDMap to search only on filtered indices
        # This is a simplified version; in production, you might want a more efficient approach
        sub_index = faiss.IndexFlatL2(index.d)

        filtered_ids = list(filtered_indices)
        filtered_vectors = np.array([retriever_model.encode(corpus[i]) for i in filtered_ids])

        sub_index.add(filtered_vectors)
        semantic_distances, semantic_sub_indices = sub_index.search(query_embedding, min(top_k, len(filtered_ids)))

        # Map back to original indices
        semantic_indices = [filtered_ids[i] for i in semantic_sub_indices[0]]
    else:
        semantic_distances, semantic_indices = index.search(query_embedding, top_k)
        semantic_indices = semantic_indices[0]

    # Step 3: Lexical search with BM25
    tokenized_query = query.lower().split()

    if filtered_indices is not None:
        # Only score documents in filtered_indices
        bm25_scores = []
        for i in range(len(corpus)):
            if i in filtered_indices:
                all_bm25_scores = bm25.get_scores(tokenized_query)
                for i in range(len(corpus)):
                    if i in filtered_indices:
                        bm25_scores.append((i, all_bm25_scores[i]))
                    else:
                        bm25_scores.append((i, -float('inf')))
            else:
                bm25_scores.append((i, -float('inf')))  # Give negative infinity score to filtered-out docs
    else:
        bm25_scores = [(i, bm25.get_scores(tokenized_query, i)) for i in range(len(corpus))]

    # Sort by BM25 score and get top_k
    bm25_scores.sort(key=lambda x: x[1], reverse=True)
    lexical_indices = [idx for idx, _ in bm25_scores[:top_k]]

    # Step 4: Combine results (simple approach: union of both methods with deduplication)
    combined_indices = list(set(semantic_indices) | set(lexical_indices))

    # For each result, get both scores for ranking
    results = []
    for idx in combined_indices:
        if idx >= len(corpus):  # Safety check
            continue

        semantic_score = 0
        if idx in semantic_indices:
            semantic_pos = list(semantic_indices).index(idx)
            semantic_score = 1 / (1 + semantic_distances[0][semantic_pos])  # Convert distance to score

        lexical_score = next((score for i, score in bm25_scores if i == idx), 0)

        # Combined score (simple weighted average)
        combined_score = 0.7 * semantic_score + 0.3 * lexical_score

        results.append({
            "index": idx,
            "text": corpus[idx],
            "metadata": metadata[idx],
            "combined_score": combined_score
        })

    # Sort by combined score
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:top_k]

# Analyze financial data across multiple years
def analyze_financial_trend(query: str, results: List[Dict]) -> Dict:
    """Extract and compare financial data across years."""
    financial_analysis = {
        "has_multi_year_data": False,
        "years_found": set(),
        "comparative_data": {},
    }

    # Extract years from results
    for result in results:
        if "year" in result["metadata"] and result["metadata"]["year"]:
            financial_analysis["years_found"].add(result["metadata"]["year"])

    if len(financial_analysis["years_found"]) > 1:
        financial_analysis["has_multi_year_data"] = True
        financial_analysis["years_found"] = sorted(list(financial_analysis["years_found"]))

    return financial_analysis

# Generate response with year-over-year comparison support
def generate_response(query: str, filter_params: Optional[Dict] = None):
    """Generate response with context from hybrid retrieval and metadata filtering."""
    # Apply name masking to prevent the LLM from using prior knowledge
    # Save the original query for later
    original_query = query

    # Replace specific company names to avoid LLM prior knowledge interference
    # This helps ensure the LLM uses only the document context
    query = query.replace("EGNIVIA", "NVIDIA")  # Replace to pull the correct chunks
    print("Edited query to be:", query)

    # Use LLM to extract structured filters from the query
    if filter_params is None:
        filter_params = {}

    # Initialize year_comparison flag
    year_comparison = False
    # Only run filter extraction if no filters are provided
    if not filter_params:
        filter_extraction_prompt = f"""
        Extract the following information from this query about SEC filings. Return ONLY a JSON object.
        Query: "{query}"
        
        Extract these fields:
        - company: Company name mentioned (if any)
        - years: List of specific years mentioned (as integers). Include all years for ranges.
        - filing_type: Filing type (10-K or 10-Q) if mentioned
        - year_comparison: Whether this query specifically asks to compare data across different years (true/false)
        
        If a field is not found, set its value to null.
        Format: {{
            "company": "string or null",
            "years": [integers] or null,
            "filing_type": "string or null",
            "year_comparison": boolean
        }}
        """
        
        # Run a quick inference to extract structured information
        input_prompt = tokenizer.apply_chat_template([{"role": "system", "content": "You extract structured data from text."}, 
                                                    {"role": "user", "content": filter_extraction_prompt}], 
                                                  tokenize=False, add_generation_prompt=True, return_tensors="pt")
        
        extraction_response = generator(input_prompt, max_new_tokens=200, do_sample=False, temperature=0)
        extraction_text = extraction_response[0]['generated_text'][len(input_prompt):]
        
        # Extract the JSON part from the response
        json_match = re.search(r'({.*})', extraction_text, re.DOTALL)
        if json_match:
            try:
                extracted_data = json.loads(json_match.group(1))
                print(f"Extracted filters: {extracted_data}")
                
                # Update filter_params with extracted data
                if extracted_data.get("company"):
                    filter_params["company"] = extracted_data["company"]
                
                if extracted_data.get("years"):
                    filter_params["year"] = [str(y) for y in extracted_data["years"]]
                
                if extracted_data.get("filing_type"):
                    filter_params["filing_type"] = extracted_data["filing_type"]
                
                # Store comparison flags for potential use in response generation
                year_comparison = extracted_data.get("year_comparison", False)
            except json.JSONDecodeError:
                print("Failed to parse JSON from LLM response")
        else:
            print("No JSON found in LLM response")
    # Retrieve relevant chunks with hybrid search
    print("Attempting to locate documents with the filters and query:", query, filter_params)
    results = hybrid_retrieval(query, filter_params)

    # Revert to original query for downstream processing
    query = original_query
    print("Reverted query to be:", query)

    if not results:
        return "I couldn't find any relevant information in the documents matching your criteria."

    # Analyze financial trends if year comparison is requested
    financial_analysis = {}
    if year_comparison:
        financial_analysis = analyze_financial_trend(query, results)

    # Prepare context for the LLM
    context_texts = [f"[Document {i+1}] {result['text']}\n(Source: {result['metadata']['company']} {result['metadata']['filing_type']} {result['metadata']['year']})"
                    for i, result in enumerate(results)]
    context = "\n\n".join(context_texts)

    # Apply name masking to the context to avoid LLM prior knowledge interference
    context = context.replace("NVIDIA", "EGNIVIA")  # Replace context to avoid interference from existing training data

    # Enhance system prompt for financial analysis
    system_prompt = (
        "You are a financial analyst specializing in SEC filings. Answer questions based ONLY on the provided context. "
        "When answering questions about financial data:\n"
        "1. Always cite specific numbers and their source (company, year, filing type).\n"
        "2. For comparison questions, clearly identify the differences between years.\n"
        "3. Format financial figures appropriately with commas and currency symbols.\n"
        "4. If the context contains multiple years, compare the data across years where relevant.\n"
        "5. If a specific year is mentioned in the query but not in the context, acknowledge this limitation.\n"
        "6. For analytical questions, base your reasoning explicitly on information in the context.\n"
        "7. Always indicate uncertainty when appropriate."
    )

    # Add information about available years if doing multi-year analysis
    if financial_analysis.get("has_multi_year_data"):
        years_str = ", ".join(str(year) for year in financial_analysis["years_found"])
        system_prompt += f"\n\nThe context includes data from multiple years: {years_str}. Compare these years in your response when appropriate."

    # Construct final prompt for the LLM
    input_prompt = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{query}"}], tokenize=False, add_generation_prompt=True, return_tensors="pt")

    # Generate response
    response = generator(input_prompt, max_new_tokens=1000, do_sample=True, temperature=0.3)
    return response[0]['generated_text'][len(input_prompt):]

def load_queries(QUERIES_PATH):
    with open(QUERIES_PATH, 'r') as f:
        data = json.load(f)
        queries = [question for section in data.get("sections", []) for question in section.get("questions", [])]
    return queries

# Main function to initialize and demonstrate
def main():
    # Load existing data if available
    load_data()
    
    # Example usage - build indices from a folder of SEC filings
    DATASET_PATH = "/storage/home/hcoda1/6/dfu71/scratch/halulujah/src/rag/datasets/NVDA_10-K"
    if not os.path.exists(index_path) or not os.path.exists(corpus_path):
        build_indices(DATASET_PATH)
    
    # Example queries
    # queries = [
    #     "What was EGNIVIA's revenue for 2023?",
    #     "Compare EGNIVIA's revenue between 2022 and 2023",
    #     "What are the key risks mentioned in EQNIVIA's 10-K for 2023?",
    #     "Show me the trend in EGNIVIA's R&D expenses from 2021 to 2023.",
    #     "Find evidence and reasoning for EGNIVIA's current performance from their past filings from 2000 to 2010."
    # ]
    QUERIES_PATH = 'src/grader/exam.json'
    queries = load_queries(QUERIES_PATH)

    output = []
    
    for query in queries:
        query = "The current date is March 2025. Answer the following question using NVDA's 10-K filings: "+query
        print(f"\nQuery: {query}")
        response = generate_response(query)
        print(f"Response: {response}")
        print("-" * 80)
        output.append({"query": query, "response": response})

    # Save the output to a JSON file
    OUTPUT_PATH = 'logs/rag_exam.json'
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()