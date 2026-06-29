# Phase 5b: Embeddings, Vector Databases & RAG
**Months 12–13 | Difficulty: 7/10 | Label: 🔴 Must Learn**

> **Previous Phase**: [Phase 5a — LLMs & Prompt Engineering](./06_Phase5_Part1_LLMs_and_Prompt_Engineering.md)  
> **Next Phase**: [Phase 6 — AI Agents & MCP](./07_Phase6_AI_Agents_and_MCP.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Part A: Text Embeddings](#part-a-text-embeddings)
  - [What Are Embeddings?](#what-are-embeddings)
  - [Dense vs. Sparse Representations](#dense-vs-sparse-representations)
  - [Similarity Metrics](#similarity-metrics)
  - [Sentence Embeddings with Sentence-Transformers](#sentence-embeddings-with-sentence-transformers)
  - [The MTEB Leaderboard](#the-mteb-leaderboard)
- [Part B: Vector Databases](#part-b-vector-databases)
  - [Why Vector Databases Exist](#why-vector-databases-exist)
  - [Approximate Nearest Neighbor Search Algorithms](#approximate-nearest-neighbor-search-algorithms)
  - [Vector Database Comparison](#vector-database-comparison)
  - [ChromaDB: Local Development](#chromadb-local-development)
  - [Qdrant: Production-Ready](#qdrant-production-ready)
- [Part C: Retrieval-Augmented Generation (RAG)](#part-c-retrieval-augmented-generation-rag)
  - [The Core Problem RAG Solves](#the-core-problem-rag-solves)
  - [Basic RAG Pipeline](#basic-rag-pipeline)
  - [Chunking Strategies](#chunking-strategies)
  - [Hybrid Search](#hybrid-search)
  - [Re-Ranking](#re-ranking)
  - [Advanced RAG Patterns](#advanced-rag-patterns)
  - [RAG Evaluation with RAGAS](#rag-evaluation-with-ragas)
- [Resources](#resources)
- [Projects](#projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 12–13 (4 weeks) |
| **Daily Time** | 1–2 hours |
| **Difficulty** | 7/10 |
| **Label** | 🔴 Must Learn |
| **Prerequisites** | Phase 5a (LLMs, prompting), Phase 4a (word embeddings) |
| **Outcome** | Can build production-grade RAG systems; understands vector search; can evaluate retrieval quality |

---

## Part A: Text Embeddings

### What Are Embeddings?

An embedding is a dense, fixed-size vector representation that captures semantic meaning.

```
"I love pizza"  →  [0.12, -0.87, 0.43, ..., 0.21]  (768 floats)
"I enjoy pizza" →  [0.11, -0.89, 0.41, ..., 0.19]  (768 floats)
"Climate change"→  [-0.92, 0.23, -0.11, ..., 0.77] (768 floats)
```

Texts with similar meaning have vectors that are close together in the embedding space. This enables semantic search: instead of matching exact keywords, you find documents by meaning.

**Key properties**:
- Fixed-size regardless of input length
- Semantic similarity → geometric proximity
- Enables similarity computation via dot product
- Transferable: one embedding model works for many retrieval tasks

---

### Dense vs. Sparse Representations

| Property | Sparse (BM25/TF-IDF) | Dense (Neural) |
|----------|:--------------------:|:--------------:|
| Representation | One float per vocabulary word | 768 floats |
| Dimensionality | 50,000–100,000 | 256–4096 |
| Exact keyword match | Excellent | Poor |
| Semantic understanding | None | Excellent |
| Out-of-vocabulary words | Fails | Handles (subwords) |
| Multilingual | Requires separate models | Single multilingual model |
| Compute cost | Extremely fast | Requires GPU for large scale |
| Explainability | High (which words matched) | Low (dense vector) |

Modern production systems use **hybrid search**: combine sparse and dense retrieval.

---

### Similarity Metrics

```python
import numpy as np
import torch
import torch.nn.functional as F

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity: measures angle between vectors (range: -1 to 1)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product: magnitude AND direction (unbounded)."""
    return np.dot(a, b)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance: geometric distance (range: 0 to ∞, lower = more similar)."""
    return np.linalg.norm(a - b)

# When to use each metric:
metrics_guide = {
    "Cosine similarity": """
    - Most common for NLP embeddings
    - Use when vector magnitude is not meaningful
    - Invariant to vector scaling
    - Best for: semantic similarity, clustering
    """,
    "Dot product": """
    - Use when embeddings are normalized to unit sphere (then = cosine)
    - OpenAI text-embedding-ada-002 and most modern models are normalized
    - Fastest to compute
    - Best for: retrieval at scale (normalized embeddings)
    """,
    "Euclidean (L2)": """
    - Use when absolute magnitude matters
    - More sensitive to vector magnitude
    - Best for: image embeddings, molecular embeddings
    """,
}

# Demonstration: semantic similarity in embedding space
embeddings = {
    "I love Python programming": None,
    "I enjoy coding in Python":   None,
    "Python is my favorite language": None,
    "I hate programming":         None,
    "The weather is nice today":  None,
}

# After computing embeddings with a model:
# Similarity("I love Python", "I enjoy coding in Python") ≈ 0.92  (high)
# Similarity("I love Python", "The weather is nice today") ≈ 0.21  (low)
```

---

### Sentence Embeddings with Sentence-Transformers

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
# all-MiniLM-L6-v2: fast, small (80MB), good for local development
# BAAI/bge-large-en-v1.5: high quality, MTEB top performer
# text-embedding-ada-002: OpenAI's widely-used model (API only)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text
sentences = [
    "The transformer architecture changed AI.",
    "Attention mechanisms are the core of modern NLP.",
    "I like to eat pizza on Fridays.",
    "The weather is great today."
]

# Batch encoding (always use batch for efficiency)
embeddings = model.encode(
    sentences,
    batch_size=32,
    normalize_embeddings=True,   # Unit normalize (enables dot product = cosine)
    show_progress_bar=True
)

print(f"Embeddings shape: {embeddings.shape}")  # (4, 384)

# Compute similarity matrix
similarity_matrix = embeddings @ embeddings.T  # Dot product since normalized
print("\nSimilarity Matrix:")
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        if i < j:
            sim = similarity_matrix[i, j]
            print(f"  {sim:.3f}: '{s1[:30]}...' vs '{s2[:30]}...'")

# Using OpenAI embeddings (cloud, higher quality)
from openai import OpenAI
import os

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings from OpenAI API."""
    response = openai_client.embeddings.create(
        input=texts,
        model=model
    )
    return np.array([item.embedding for item in response.data])

# Embedding model selection guide
embedding_models = {
    "all-MiniLM-L6-v2":         {"dims": 384, "size": "80MB",  "speed": "fast",  "quality": "good",     "use_case": "local dev, prototyping"},
    "all-mpnet-base-v2":        {"dims": 768, "size": "420MB", "speed": "medium","quality": "better",   "use_case": "local, higher quality"},
    "BAAI/bge-large-en-v1.5":   {"dims": 1024,"size": "1.3GB", "speed": "slow",  "quality": "excellent","use_case": "production local"},
    "text-embedding-3-small":   {"dims": 1536,"size": "API",   "speed": "API",   "quality": "excellent","use_case": "prod API, cost-effective"},
    "text-embedding-3-large":   {"dims": 3072,"size": "API",   "speed": "API",   "quality": "best",     "use_case": "prod API, best quality"},
}
```

---

### The MTEB Leaderboard

**MTEB (Massive Text Embedding Benchmark)** is the standard benchmark for embedding models.

```
URL: https://huggingface.co/spaces/mteb/leaderboard

Key tasks evaluated:
- Retrieval (most important for RAG)
- Classification
- Clustering
- Semantic Textual Similarity (STS)
- Re-ranking

Model selection strategy:
1. Sort by "Retrieval" score (not overall average)
2. Filter by max_tokens >= your expected chunk size
3. Consider: size vs. quality tradeoff for your deployment
4. Always benchmark on YOUR data — MTEB is a proxy, not a guarantee

Top performers (as of late 2024):
- cohere-embed-v3 (API): best overall
- BAAI/bge-m3: best free/local, multilingual
- text-embedding-3-large: best OpenAI
- E5-mistral-7b: best parameter-efficient local
```

---

## Part B: Vector Databases

### Why Vector Databases Exist

A naive semantic search implementation:
```python
def naive_search(query_vec: np.ndarray, corpus_vecs: np.ndarray, top_k: int = 5):
    similarities = corpus_vecs @ query_vec  # Dot product (if normalized)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]
```

This works, but:
- **O(n) per query** — for 1M documents, you compute 1M dot products
- **All vectors in RAM** — 1M × 768 × 4 bytes ≈ 3GB just for vectors
- **No filtering** — can't combine with metadata filters efficiently

Vector databases solve these problems with:
- **ANN (Approximate Nearest Neighbor) indexing** — find approximate top-k in O(log n)
- **On-disk storage** with efficient memory management
- **Metadata filtering** alongside vector search
- **CRUD operations** — update/delete vectors
- **Horizontal scaling** — distribute across nodes

---

### Approximate Nearest Neighbor Search Algorithms

```python
# Understanding HNSW — the most widely used ANN algorithm
# HNSW = Hierarchical Navigable Small World graphs

# Conceptual explanation:
# Imagine a multilevel graph. At the top level: few nodes, long-range connections.
# At the bottom level: all nodes, short-range connections.
# Search: start at top, greedily navigate toward query, descend to lower levels.
# 
# Like navigating a city: highway → main road → street → address
# O(log n) approximate search, excellent recall-speed tradeoff

# IVF = Inverted File Index
# Cluster vectors into K centroids (like K-means)
# To search: find nearest centroid(s), then brute-force within cluster
# Much faster than exhaustive search; recall depends on n_probe parameter

# PQ = Product Quantization (compression)
# Split 768-dim vector into M sub-vectors of 768/M dims
# Quantize each sub-vector to a codebook of 256 entries
# Store 768-dim float32 (3072 bytes) as M bytes — 384x compression!
# Used for very large-scale deployment where RAM is the bottleneck

# Benchmarking ANN algorithms: ann-benchmarks.com
algorithms_comparison = {
    "HNSW": {
        "recall@10": "97-99%",
        "queries_per_second": "High",
        "memory": "High (all vectors in RAM)",
        "build_time": "Medium",
        "use_case": "General purpose, highest recall"
    },
    "IVF+PQ": {
        "recall@10": "85-95%",
        "queries_per_second": "Very high",
        "memory": "Low (compressed)",
        "build_time": "Long",
        "use_case": "Billion-scale, limited RAM"
    },
    "Flat (brute force)": {
        "recall@10": "100% (exact)",
        "queries_per_second": "Low",
        "memory": "High",
        "build_time": "None",
        "use_case": "< 100K vectors, ground truth for eval"
    },
}
```

---

### Vector Database Comparison

| Database | Hosting | ANN | Filtering | Scale | Best For |
|----------|---------|-----|-----------|-------|----------|
| **ChromaDB** | Local / Cloud | HNSW | Yes | Small-Medium | Prototyping, local dev |
| **Qdrant** | Local / Cloud | HNSW | Yes | Large | Production, self-hosted |
| **Weaviate** | Local / Cloud | HNSW | Yes | Large | GraphQL API, hybrid search |
| **Milvus** | Local / Cloud | HNSW, IVF | Yes | Billion-scale | Enterprise |
| **Pinecone** | Managed only | Proprietary | Yes | Large | Managed, no ops |
| **pgvector** | PostgreSQL ext | IVFFLAT, HNSW | Full SQL | Medium | Already using Postgres |
| **Databricks** | Managed | HNSW | Yes | Enterprise | Databricks ecosystem (your use case!) |

---

### ChromaDB: Local Development

```python
import chromadb
from chromadb.utils import embedding_functions
import uuid

# Initialize ChromaDB (persistent local storage)
client = chromadb.PersistentClient(path="./chroma_db")

# Use OpenAI embeddings automatically (or any other embedding function)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Create a collection (like a table in a traditional database)
collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}  # similarity metric
)

# Add documents
documents = [
    "Transformers use attention mechanisms to process sequences.",
    "RAG combines retrieval with language model generation.",
    "LLaMA is a family of open-source language models by Meta.",
    "Fine-tuning adapts pre-trained models to specific tasks.",
    "LoRA uses low-rank matrices to reduce fine-tuning parameters.",
]
metadatas = [
    {"topic": "architecture", "phase": 4},
    {"topic": "rag", "phase": 5},
    {"topic": "models", "phase": 5},
    {"topic": "fine-tuning", "phase": 7},
    {"topic": "fine-tuning", "phase": 7},
]

# ChromaDB automatically computes embeddings and stores them
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=[str(uuid.uuid4()) for _ in documents]
)

print(f"Collection count: {collection.count()}")

# Query (semantic search)
results = collection.query(
    query_texts=["How do transformers work?"],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
):
    print(f"  Distance: {dist:.4f} | Topic: {meta['topic']} | Doc: {doc}")

# Query with metadata filter
filtered_results = collection.query(
    query_texts=["model adaptation techniques"],
    n_results=3,
    where={"topic": "fine-tuning"},  # Only search fine-tuning documents
    include=["documents", "distances"]
)
```

---

### Qdrant: Production-Ready

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Qdrant (local Docker or cloud)
# docker run -p 6333:6333 qdrant/qdrant
client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "documents"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension

# Create collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

# Index documents
documents = [
    {"id": 1, "text": "RAG grounds LLMs in external knowledge.", "source": "textbook", "page": 15},
    {"id": 2, "text": "ChromaDB is great for local development.", "source": "blog", "page": 0},
    {"id": 3, "text": "Qdrant is optimized for production use.", "source": "docs", "page": 1},
]

vectors = embedding_model.encode([d["text"] for d in documents], normalize_embeddings=True)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=doc["id"],
            vector=vec.tolist(),
            payload={"text": doc["text"], "source": doc["source"], "page": doc["page"]}
        )
        for doc, vec in zip(documents, vectors)
    ]
)

# Semantic search
query = "How does retrieval augmented generation work?"
query_vector = embedding_model.encode([query], normalize_embeddings=True)[0]

search_results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector.tolist(),
    limit=3,
    with_payload=True
)

for result in search_results:
    print(f"Score: {result.score:.4f} | {result.payload['text']}")

# Filtered semantic search
filtered_results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector.tolist(),
    query_filter=Filter(
        must=[FieldCondition(
            key="source",
            match=MatchValue(value="docs")
        )]
    ),
    limit=3
)
```

---

## Part C: Retrieval-Augmented Generation (RAG)

### The Core Problem RAG Solves

```
Base LLM problems:
1. Knowledge cutoff: doesn't know about events after training
2. Private data: can't access your company's internal documents
3. Hallucination: invents facts when uncertain
4. Scale: can't memorize every document in a corpus

RAG solution:
1. Store your documents in a vector database
2. When user asks a question, retrieve the most relevant documents
3. Include retrieved documents in the LLM context as "reference material"
4. LLM answers the question using only the retrieved context

Result:
- Accurate answers grounded in real documents
- Citations possible (you know which documents were retrieved)
- No retraining needed when documents change
- Dramatically reduced hallucination
```

---

### Basic RAG Pipeline

```python
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import chromadb
import os

class BasicRAG:
    """
    Production-ready basic RAG implementation.
    
    Pipeline:
    1. Index phase: embed all documents, store in vector DB
    2. Query phase: embed query, retrieve top-k, generate answer
    """
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        collection_name: str = "rag_docs",
        top_k: int = 5
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model
        self.top_k = top_k
        
        # ChromaDB for local dev
        chroma_client = chromadb.PersistentClient(path="./rag_chroma")
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_documents(self, documents: list[dict]) -> None:
        """
        Index a list of documents.
        Each document: {"id": str, "text": str, "metadata": dict}
        """
        texts = [doc["text"] for doc in documents]
        
        # Compute embeddings
        embeddings = self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        ).tolist()
        
        self.collection.add(
            ids=[doc["id"] for doc in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[doc.get("metadata", {}) for doc in documents]
        )
        print(f"Indexed {len(documents)} documents. Total: {self.collection.count()}")
    
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Retrieve top-k most relevant documents for the query."""
        k = top_k or self.top_k
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text": doc,
                "metadata": meta,
                "similarity": 1 - dist  # Convert distance to similarity
            })
        
        return retrieved
    
    def generate(self, query: str, context_docs: list[dict]) -> str:
        """Generate answer using LLM with retrieved context."""
        
        # Format context
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc["metadata"].get("source", "unknown")
            context_parts.append(f"[Document {i} - Source: {source}]\n{doc['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context.

RULES:
1. Answer the question using ONLY the information in the provided context
2. If the context doesn't contain enough information to answer, say: "The provided context doesn't contain information to answer this question."
3. Cite which document(s) you used in your answer using [Document N] notation
4. Do not make up information not in the context"""
        
        user_message = f"""CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1  # Low temperature for factual answers
        )
        
        return response.choices[0].message.content
    
    def answer(self, query: str, verbose: bool = False) -> dict:
        """Full RAG pipeline: retrieve then generate."""
        retrieved = self.retrieve(query)
        
        if verbose:
            print(f"Query: {query}")
            print(f"Retrieved {len(retrieved)} documents:")
            for doc in retrieved:
                print(f"  [{doc['similarity']:.3f}] {doc['text'][:80]}...")
        
        answer = self.generate(query, retrieved)
        
        return {
            "query": query,
            "answer": answer,
            "sources": [doc["metadata"] for doc in retrieved],
            "n_retrieved": len(retrieved)
        }


# Usage
rag = BasicRAG()

# Index documents
docs = [
    {"id": "1", "text": "RAG (Retrieval-Augmented Generation) combines...", 
     "metadata": {"source": "chapter_5.pdf", "page": 1}},
    # ... more documents
]
rag.index_documents(docs)

# Query
result = rag.answer("What is RAG?", verbose=True)
print(result["answer"])
```

---

### Chunking Strategies

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)
import tiktoken

# ── Strategy 1: Fixed-Size with Overlap ──
# Simple, consistent, widely used
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # Target chunk size in characters
    chunk_overlap=200,         # Overlap between chunks (prevents cutting mid-sentence)
    separators=["\n\n", "\n", ". ", " ", ""]  # Split on these in order
)

# ── Strategy 2: Token-Aware Chunking ──
# Critical: LLM context limits are in TOKENS, not characters
# 1000 characters ≈ 200-250 tokens for English
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

token_splitter = TokenTextSplitter(
    chunk_size=256,           # tokens per chunk
    chunk_overlap=32          # overlap in tokens
)

# ── Strategy 3: Semantic Chunking ──
# Split at semantic boundaries rather than fixed sizes
# Better retrieval quality but slower to index

def semantic_chunk(text: str, embedding_model, similarity_threshold: float = 0.7) -> list[str]:
    """
    Split text at semantic boundary changes.
    If adjacent sentences are semantically different, split there.
    """
    sentences = text.split('. ')
    embeddings = embedding_model.encode(sentences, normalize_embeddings=True)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # If similarity drops significantly, start new chunk
        sim = float(np.dot(embeddings[i-1], embeddings[i]))
        if sim < similarity_threshold:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks

# ── Strategy 4: Hierarchical Chunking ──
# Store both large and small chunks; retrieve small, return large
# Also called "Parent-Child chunking"

# Index: small chunks (e.g., 256 tokens) for precise retrieval
# Context: large chunks (e.g., 1024 tokens) for comprehensive context

# This is the best strategy for production systems.
# Retrieval with small chunks = more precise
# Context with large chunks = more complete information

chunking_comparison = {
    "Fixed-size": {"quality": "OK", "speed": "Fast", "when": "Prototyping"},
    "Token-aware": {"quality": "Good", "speed": "Fast", "when": "Always use over char-based"},
    "Semantic": {"quality": "Better", "speed": "Slow (embedding per sentence)", "when": "When quality matters"},
    "Hierarchical": {"quality": "Best", "speed": "Medium", "when": "Production systems"},
}
```

---

### Hybrid Search

```python
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple

class HybridSearch:
    """
    Hybrid search: combine BM25 (sparse) with dense vector search.
    
    BM25 is excellent at:
    - Exact keyword matching
    - Technical terms, product codes, names
    
    Dense search is excellent at:
    - Semantic similarity
    - Paraphrase matching
    - Cross-language queries (with multilingual model)
    
    Hybrid: best of both worlds.
    """
    def __init__(self, documents: List[str], embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model
        
        # Build BM25 index
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Build dense embeddings
        self.dense_vecs = embedding_model.encode(documents, normalize_embeddings=True)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5  # Weight: 0 = BM25 only, 1 = dense only
    ) -> List[Tuple[int, float]]:
        """
        Hybrid search with RRF (Reciprocal Rank Fusion) normalization.
        
        RRF score = sum of 1/(k + rank) for each retrieval method
        This is more robust than score fusion (avoids scale mismatches).
        """
        # ── BM25 Search ──
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        
        # ── Dense Search ──
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        dense_scores = self.dense_vecs @ query_vec
        dense_ranking = np.argsort(dense_scores)[::-1]
        
        # ── RRF Fusion ──
        k = 60  # RRF constant (60 is standard)
        rrf_scores = {}
        
        for rank, idx in enumerate(bm25_ranking[:top_k * 3]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) / (k + rank + 1)
        
        for rank, idx in enumerate(dense_ranking[:top_k * 3]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (k + rank + 1)
        
        # Sort by combined RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
```

---

### Re-Ranking

```python
from sentence_transformers import CrossEncoder

class RAGWithReranking:
    """
    Two-stage retrieval with re-ranking.
    
    Stage 1 (bi-encoder): Fast, approximate retrieval
      - Embed query and documents separately
      - Cosine similarity (fast but less accurate)
      - Retrieve top-50 candidates
    
    Stage 2 (cross-encoder): Slow, precise scoring
      - Process query AND document together
      - Full attention over both → much more accurate
      - Re-rank top-50 to get top-5
    
    This is the standard pattern in production RAG systems.
    """
    def __init__(self, biencoder_name: str, crossencoder_name: str):
        # Stage 1: bi-encoder (fast retrieval)
        self.biencoder = SentenceTransformer(biencoder_name)
        
        # Stage 2: cross-encoder (accurate re-ranking)
        # ms-marco-MiniLM-L-6-v2 is the standard; trained on search relevance
        self.crossencoder = CrossEncoder(crossencoder_name)
    
    def retrieve_and_rerank(
        self,
        query: str,
        documents: list[str],
        initial_k: int = 50,
        final_k: int = 5
    ) -> list[tuple[str, float]]:
        
        # Stage 1: Bi-encoder fast retrieval
        query_vec = self.biencoder.encode([query], normalize_embeddings=True)[0]
        doc_vecs = self.biencoder.encode(documents, normalize_embeddings=True)
        
        scores = doc_vecs @ query_vec
        top_indices = np.argsort(scores)[-initial_k:][::-1]
        candidates = [(documents[i], scores[i]) for i in top_indices]
        
        # Stage 2: Cross-encoder re-ranking
        cross_inputs = [(query, doc) for doc, _ in candidates]
        cross_scores = self.crossencoder.predict(cross_inputs)
        
        # Re-rank by cross-encoder score
        reranked = sorted(
            [(candidates[i][0], float(cross_scores[i])) for i in range(len(candidates))],
            key=lambda x: x[1],
            reverse=True
        )
        
        return reranked[:final_k]

# Cohere Rerank API (cloud, highest quality)
import cohere

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def cohere_rerank(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    results = co.rerank(
        query=query,
        documents=documents,
        top_n=top_k,
        model="rerank-english-v3.0"
    )
    return [{"text": r.document["text"], "score": r.relevance_score} for r in results.results]
```

---

### Advanced RAG Patterns

```python
# ── Pattern 1: HyDE (Hypothetical Document Embeddings) ──
# Problem: query and document are in different "styles"
#   Query: "What is the capital of France?"
#   Document: "Paris is the capital city of France, situated..."
#   These look different even though they're semantically similar!
#
# HyDE solution:
#   1. Ask LLM to generate a hypothetical answer/document for the query
#   2. Embed the hypothetical document (not the query)
#   3. Use the hypothetical document embedding for retrieval
#   Result: better query-document alignment

def hyde_retrieve(query: str, llm_client, embedding_model, collection) -> list[dict]:
    # Step 1: Generate hypothetical document
    hyp_doc = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Write a factual paragraph that directly answers: {query}\nProvide only the paragraph, no introduction."
        }]
    ).choices[0].message.content
    
    # Step 2: Embed hypothetical document
    hyp_embedding = embedding_model.encode([hyp_doc], normalize_embeddings=True).tolist()
    
    # Step 3: Retrieve using hypothetical document embedding
    results = collection.query(
        query_embeddings=hyp_embedding,
        n_results=5
    )
    return results


# ── Pattern 2: Multi-Query Retrieval ──
# Problem: a single query may miss relevant docs phrased differently
# Solution: generate N variants of the query, retrieve for each, merge results

def multi_query_retrieve(query: str, llm_client, embedding_model, collection, n_variants: int = 4) -> list:
    # Generate query variants
    variants_response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Generate {n_variants} different ways to ask the following question.
Output only the questions, one per line, no numbering.

Question: {query}"""
        }]
    ).choices[0].message.content
    
    query_variants = [query] + variants_response.strip().split('\n')[:n_variants]
    
    # Retrieve for all variants
    all_docs = {}  # id → doc to deduplicate
    for variant in query_variants:
        vec = embedding_model.encode([variant], normalize_embeddings=True).tolist()
        results = collection.query(query_embeddings=vec, n_results=3, include=["documents", "ids"])
        for id, doc in zip(results["ids"][0], results["documents"][0]):
            all_docs[id] = doc
    
    return list(all_docs.values())


# ── Pattern 3: Contextual Compression ──
# Problem: retrieved chunks contain irrelevant content
# Solution: extract only the relevant portion of each chunk

def compress_chunk(query: str, chunk: str, llm_client) -> str:
    """Extract only the relevant portion of a chunk for the query."""
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Given this question: "{query}"
            
Extract from the following passage ONLY the sentences that are relevant to answering the question.
If nothing is relevant, output: "NOT RELEVANT"
Output only the extracted text, no explanation.

Passage:
{chunk}"""
        }]
    ).choices[0].message.content
    
    if "NOT RELEVANT" in response:
        return None
    return response


# ── Pattern 4: Corrective RAG (CRAG) ──
# Problem: retrieved documents may be irrelevant to the query
# Solution: evaluate each retrieved document; fall back to web search if all fail

def corrective_rag(query: str, rag_results: list[str], llm_client, web_search_fn) -> str:
    """Evaluate retrieved docs; fall back to web search if quality is poor."""
    
    def evaluate_relevance(doc: str) -> float:
        """Score how relevant a document is to the query (0-1)."""
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"""Rate how relevant this document is for answering: "{query}"
Score from 0.0 (irrelevant) to 1.0 (perfectly relevant).
Output ONLY a decimal number.

Document: {doc[:500]}"""}]
        ).choices[0].message.content
        try:
            return float(response.strip())
        except ValueError:
            return 0.0
    
    # Score all retrieved documents
    scored = [(evaluate_relevance(doc), doc) for doc in rag_results]
    
    # Partition into relevant and irrelevant
    relevant = [doc for score, doc in scored if score >= 0.7]
    ambiguous = [doc for score, doc in scored if 0.3 <= score < 0.7]
    
    # Decision logic:
    if not relevant and not ambiguous:
        # All documents are irrelevant → fall back to web search
        web_results = web_search_fn(query)
        context = "\n\n".join(web_results[:3])
        source_note = "[Web search fallback — knowledge base insufficient]"
    elif relevant:
        # Use only highly relevant documents
        context = "\n\n".join(relevant)
        source_note = f"[{len(relevant)} relevant documents found]"
    else:
        # Mixed — use ambiguous docs but note uncertainty
        context = "\n\n".join(ambiguous)
        source_note = "[Documents partially relevant — answer may be incomplete]"
    
    return context, source_note


# ── Pattern 5: Contextual Retrieval (Anthropic, 2024) ──
# Problem: chunks lose context when split from parent document
# "The total cost was $1.2 billion" — which deal? Which year?
# Solution: prepend a document-level context summary to each chunk BEFORE embedding

def add_contextual_context(
    document: str,
    chunks: list[str],
    llm_client
) -> list[str]:
    """Prepend document context to each chunk for better embedding quality."""
    
    # Generate document-level context once (not per chunk)
    doc_context = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"""Summarize this document in 2-3 sentences. Focus on: what it covers, the time period, and the main entities involved.

Document (first 2000 chars):
{document[:2000]}"""}]
    ).choices[0].message.content
    
    # Prepend context to each chunk before embedding
    contextual_chunks = []
    for chunk in chunks:
        contextual_chunk = f"[Document context: {doc_context}]\n\n{chunk}"
        contextual_chunks.append(contextual_chunk)
    
    return contextual_chunks
    # Anthropic reports: contextual retrieval reduces retrieval failures by 49%


# ── Pattern 6: Retrieval Quality Metrics ──
# RAGAS measures generation quality; these measure retrieval quality directly

def evaluate_retrieval(
    queries: list[str],
    retrieved_docs: list[list[str]],
    relevant_docs: list[list[str]]  # ground truth: which docs SHOULD be retrieved
) -> dict:
    """Compute standard retrieval metrics."""
    
    def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        return len(set(retrieved[:k]) & relevant) / k
    
    def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0
    
    def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    precisions, recalls, rrs = [], [], []
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        relevant_set = set(relevant)
        precisions.append(precision_at_k(retrieved, relevant_set, k=5))
        recalls.append(recall_at_k(retrieved, relevant_set, k=5))
        rrs.append(reciprocal_rank(retrieved, relevant_set))
    
    return {
        "precision@5": sum(precisions) / len(precisions),
        "recall@5": sum(recalls) / len(recalls),
        "MRR": sum(rrs) / len(rrs)       # Mean Reciprocal Rank
    }
    # Target: P@5 > 0.7, Recall@5 > 0.8, MRR > 0.85 for production RAG
```

---

### RAG Evaluation with RAGAS

```python
# RAGAS (RAG Assessment) - the standard framework for evaluating RAG pipelines
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # Are answers faithful to the retrieved context?
    answer_relevancy,       # Is the answer relevant to the question?
    context_precision,      # Are retrieved chunks relevant?
    context_recall,         # Were all relevant chunks retrieved?
)
from datasets import Dataset

# Prepare evaluation dataset
# You need: question, contexts (retrieved), answer (generated), ground_truth
eval_data = {
    "question": [
        "What is RAG?",
        "How does HNSW work?"
    ],
    "contexts": [
        ["RAG combines retrieval with LLM generation..."],
        ["HNSW is a graph-based ANN algorithm..."]
    ],
    "answer": [
        "RAG is a technique that retrieves relevant documents...",
        "HNSW uses a hierarchical graph structure..."
    ],
    "ground_truth": [
        "RAG (Retrieval-Augmented Generation) is a method...",
        "HNSW (Hierarchical Navigable Small World)..."
    ]
}

eval_dataset = Dataset.from_dict(eval_data)

# Run RAGAS evaluation
results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,       # Range: 0-1 (1 = no hallucination)
        answer_relevancy,   # Range: 0-1 (1 = perfectly relevant)
        context_precision,  # Range: 0-1 (1 = all retrieved docs relevant)
        context_recall,     # Range: 0-1 (1 = all needed docs retrieved)
    ]
)

print(results.to_pandas().to_string())

# Metric interpretation:
ragas_metrics = {
    "faithfulness": "Does the answer only use information from the context? (catches hallucination)",
    "answer_relevancy": "Does the answer address the question? (catches off-topic answers)",
    "context_precision": "Are the retrieved chunks relevant? (measures retrieval precision)",
    "context_recall": "Did we retrieve all the chunks needed? (measures retrieval recall)",
}

# What to optimize:
# Low faithfulness → LLM is hallucinating; add constraints to system prompt, check context quality
# Low answer relevancy → Prompt engineering issue; improve generation prompt
# Low context precision → Too many irrelevant chunks retrieved; tune retrieval threshold, use reranking
# Low context recall → Missing relevant documents; increase top_k, try multi-query, check chunking
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) | Docs | Free | Best hands-on RAG tutorial |
| 2 | [RAGAS GitHub](https://github.com/explodinggradients/ragas) | GitHub | Free | Essential: learn to evaluate your RAG |
| 3 | [Retrieval Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) | Paper | Free | Original RAG paper |
| 4 | [Advanced RAG Techniques (Pinecone)](https://www.pinecone.io/learn/advanced-rag/) | Blog | Free | Best blog series on advanced RAG patterns |
| 5 | [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | Tool | Free | Picking the right embedding model |
| 6 | [ANN-Benchmarks](https://ann-benchmarks.com/) | Tool | Free | Choosing the right vector index |

---

## Projects

### Project 14: Document QA System
**Difficulty**: 6/10 | **Time**: 1–2 weeks

Build a production-quality RAG system that:
1. Accepts PDF and text documents
2. Chunks with hierarchical strategy (small for retrieval, large for context)
3. Stores in ChromaDB (local) with metadata
4. Supports hybrid BM25 + dense search
5. Reranks with CrossEncoder
6. Returns answer with citations
7. Evaluated with RAGAS

**Extension for Databricks user**: Run the embedding computation on a Spark cluster for large document collections. Use Databricks Vector Search instead of ChromaDB.

---

### Project 15: RAG Evaluation Benchmark
**Difficulty**: 7/10 | **Time**: 1 week

Build a systematic evaluation framework:
1. Create 50 test questions from your document corpus
2. Generate answers with 3 different chunking strategies
3. Generate answers with and without reranking
4. Evaluate all variants with RAGAS
5. Build a comparison table showing which config wins on each metric
6. Document which configuration to use and why

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Not chunking at all (entire doc as context) | Exceeds context window; relevant content diluted | Always chunk; use hierarchical chunking |
| Chunks too small (< 100 tokens) | Not enough context for LLM | Use 256-512 token chunks minimum |
| Chunks too large (> 1000 tokens) | Imprecise retrieval; noisy context | Keep chunks semantically focused |
| Not overlapping chunks | Cuts mid-sentence; misses context | Always use 10-20% overlap |
| Using character count instead of tokens | Unpredictable chunk sizes | Always use token-based chunking |
| Skipping reranking | Lower quality answers | Always add a CrossEncoder reranker in production |
| Not evaluating with RAGAS | No objective quality measure | Build evaluation pipeline before production |
| Using same embedding model for all use cases | Suboptimal retrieval | Use MTEB to pick model for your specific task |

---

## Mastery Checklist

### Embeddings
- [ ] Understands why dense embeddings enable semantic search
- [ ] Can explain cosine similarity vs. dot product and when to use each
- [ ] Can use sentence-transformers to compute batch embeddings
- [ ] Knows how to navigate the MTEB leaderboard to pick an embedding model

### Vector Databases
- [ ] Understands HNSW algorithm at a conceptual level
- [ ] Can implement CRUD operations in ChromaDB
- [ ] Understands metadata filtering in vector databases
- [ ] Knows when to use Qdrant vs. ChromaDB vs. pgvector

### RAG
- [ ] Built a working RAG system from scratch
- [ ] Can explain the difference between basic and advanced RAG patterns
- [ ] Implemented at least one advanced pattern (HyDE, multi-query, or reranking)
- [ ] Evaluated RAG system with RAGAS and interpreted the metrics
- [ ] Understands the "lost in the middle" problem

---

## Moving to Phase 6

**Before proceeding to [Phase 6: AI Agents & MCP](./07_Phase6_AI_Agents_and_MCP.md), confirm:**

- [ ] Working RAG system with RAGAS evaluation scores
- [ ] Understand the difference between retrieval quality and generation quality
- [ ] Can debug a RAG system by identifying whether the problem is in retrieval or generation

**Why Phase 6 comes next**: RAG is a passive system — it retrieves and generates. Agents are active — they decide what to do, can call tools, and can interact with external systems. Phase 6 teaches you how to build systems where the LLM drives the workflow.

---

## Phase Completion & Readiness Assessment

> Complete this assessment before Phase 6. RAG is the most commonly deployed AI pattern in production — every AI engineer needs to build and evaluate it end to end.

---

### 1. Knowledge Checklist

**Embeddings**
- [ ] Dense vs. sparse embeddings — representations and use cases
- [ ] Cosine similarity, dot product, L2 distance — when each is correct
- [ ] Sentence transformer architecture: CLS pooling vs. mean pooling
- [ ] MTEB leaderboard: how to choose an embedding model for your use case
- [ ] Embedding model fine-tuning: contrastive loss, triplet loss, in-batch negatives
- [ ] Dimensionality vs. quality tradeoff in embedding models

**Vector Databases**
- [ ] ANN algorithms: HNSW, IVF, PQ — tradeoffs between recall, speed, memory
- [ ] Vector DB comparison: ChromaDB, Qdrant, Weaviate, Pinecone, pgvector, Databricks
- [ ] Metadata filtering and hybrid search (BM25 + dense)
- [ ] Index creation, collection/namespace management
- [ ] Upserts and deletes in a vector index
- [ ] Scalar quantisation and product quantisation for memory efficiency

**RAG Pipeline**
- [ ] Document ingestion: PDF/HTML/DOCX parsing
- [ ] Chunking strategies: fixed-size, token-aware, semantic, hierarchical
- [ ] Retrieval: dense retrieval, BM25, hybrid + RRF fusion
- [ ] Reranking: bi-encoder retrieve then cross-encoder rerank
- [ ] Context compression: remove irrelevant sentences before generation
- [ ] Advanced patterns: HyDE, multi-query, self-RAG

**Evaluation**
- [ ] RAGAS metrics: faithfulness, answer relevancy, context precision, context recall
- [ ] Why faithfulness is the most important metric
- [ ] How to build an evaluation dataset for your RAG system

---

### 2. Practical Skills Checklist

- [ ] Generate embeddings with sentence-transformers and compute similarity between texts
- [ ] Build a ChromaDB collection with metadata, insert documents, and query with filters
- [ ] Implement semantic chunking (not just fixed-size splitting)
- [ ] Build a full RAG pipeline from scratch (ingest → chunk → embed → store → retrieve → generate)
- [ ] Implement hybrid search (BM25 + dense + RRF fusion)
- [ ] Add CrossEncoder reranking to improve precision
- [ ] Evaluate a RAG system with RAGAS (all 4 metrics)
- [ ] Build HyDE (hypothetical document embedding) retrieval
- [ ] Implement multi-query retrieval

---

### 3. Coding Challenges

**Challenge A — RAG from Scratch**
```python
# Build a RAG pipeline with ZERO framework code (no LangChain):
# Ingest: load a PDF, extract text with PyMuPDF
# Chunk: 512 token chunks with 50 token overlap
# Embed: sentence-transformers (all-MiniLM-L6-v2)
# Store: ChromaDB with document metadata (page, source, chunk_id)
# Retrieve: top-5 by cosine similarity
# Generate: OpenAI with prompt template including retrieved context and citations
# Output: answer + list of (source, page, excerpt) citations
# Test: 10 questions over Databricks documentation
```

**Challenge B — Hybrid Search + Reranking**
```python
# Extend the RAG from Challenge A with:
# 1. BM25 sparse retrieval (using rank_bm25 library)
# 2. Dense retrieval (from Challenge A)
# 3. Reciprocal Rank Fusion to merge results
# 4. CrossEncoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
# 5. Compare top-5 results from: dense-only vs. hybrid vs. hybrid+rerank
# Show: 3 examples where hybrid+rerank returns better results than dense-only
```

**Challenge C — RAGAS Evaluation**
```python
# Create an evaluation dataset:
# - 30 questions with ground truth answers from the same document corpus
# - Run your RAG pipeline on all 30 questions
# - Evaluate using RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
# - Compare 3 configurations:
#     A: fixed-size chunks, no reranking
#     B: semantic chunks, no reranking
#     C: semantic chunks, CrossEncoder reranking
# - Produce: comparison table + bar chart for each metric
```

---

### 4. Mini Project

**Databricks Documentation Assistant** (Project 14 from the roadmap):
- Ingest: Databricks Python API docs and user guide (HTML scraping or PDFs)
- Pipeline: semantic chunking → BAAI/bge-large-en-v1.5 embeddings → ChromaDB
- Retrieval: hybrid BM25 + dense + CrossEncoder reranker
- UI: Gradio interface with: question input, answer display, sources section
- Evaluation: RAGAS on 30 manually created QA pairs
- Deploy: run locally with Docker Compose

---

### 5. Capstone Project

**RAG Evaluation Benchmark** (Project 15 from the roadmap):
- Corpus: 3 document sets (Databricks docs, Wikipedia subset, your own notes)
- Test configurations: 2×2×2 = 8 combinations:
  - Chunking: fixed vs. semantic
  - Retrieval: dense-only vs. hybrid
  - Reranking: none vs. CrossEncoder
- 30 test questions per corpus (90 total)
- Metrics: RAGAS faithfulness, relevancy, precision, recall + latency + cost
- Report: which configuration wins overall; which configuration to use when latency matters vs. when quality matters
- GitHub repo with reproducible evaluation script

---

### 6. Interview Questions

**Beginner**

1. **Q: What is a vector embedding and what is it used for?**
   A: A vector embedding is a dense numerical representation (e.g., 384 or 1536 floats) that encodes semantic meaning. Similar texts have similar embeddings. Used for: semantic search (find similar documents), clustering, RAG (retrieve relevant context), recommendation.

2. **Q: What is cosine similarity and why is it preferred over Euclidean distance for embeddings?**
   A: Cosine similarity = (A·B) / (|A||B|) — measures the angle between vectors, ignoring magnitude. Preferred because embedding magnitudes carry no semantic meaning (a longer embedding vector isn't "more similar" to anything). Cosine similarity focuses purely on direction.

3. **Q: What is RAG (Retrieval-Augmented Generation)?**
   A: RAG retrieves relevant documents from a knowledge base and includes them in the LLM's prompt. The LLM then generates an answer grounded in the retrieved content. Benefits: reduces hallucination, allows up-to-date information, enables citation.

4. **Q: What is chunking and why does chunk size matter?**
   A: Chunking splits documents into pieces for embedding and retrieval. Too small: individual chunks lack context, embeddings don't capture full meaning. Too large: irrelevant content dilutes the retrieved context, wastes context window tokens. Typical: 256–1024 tokens with 10–20% overlap.

5. **Q: What is BM25 and why use it alongside dense embeddings?**
   A: BM25 is a keyword-based ranking function (TF-IDF variant) that excels at finding exact keyword matches. Dense embeddings excel at semantic similarity. Hybrid search combines both: catches exact matches (product codes, names) that embeddings miss, plus semantic matches that BM25 misses.

6. **Q: What is reranking and why is it a two-stage process?**
   A: First stage: fast approximate retrieval with a bi-encoder (ANN search) — retrieve top 50. Second stage: accurate reranking with a cross-encoder — score all 50 pairs and return top 5. Cross-encoders are slow (can't use ANN) but more accurate than bi-encoders at relevance judgement.

7. **Q: What is HNSW and why is it used for vector search?**
   A: Hierarchical Navigable Small World graph. Builds a multi-layer graph where higher layers contain sparser, long-range connections and lower layers contain dense connections. Search starts at the top layer and descends. Achieves near-linear query time with high recall. Used by Qdrant, Weaviate, FAISS.

**Intermediate**

8. **Q: What is the difference between a bi-encoder and a cross-encoder for retrieval?**
   A: Bi-encoder: encodes query and document independently, then scores with dot product — O(1) per query (precomputed document embeddings). Cross-encoder: encodes query and document jointly (concatenated input) — O(N) per query. Cross-encoders are ~10× more accurate but can't be used for large-scale ANN retrieval.

9. **Q: Explain Reciprocal Rank Fusion (RRF) for hybrid search.**
   A: RRF combines rankings from multiple retrieval systems. Score(d) = Σ 1/(k + rank_i(d)) where k=60 is a constant and rank_i(d) is the rank of document d in system i. RRF is parameter-free and surprisingly robust — it doesn't require normalising scores between systems with different scales.

10. **Q: What is HyDE (Hypothetical Document Embedding)?**
    A: Instead of embedding the user's question (which looks different from answers), ask the LLM to generate a hypothetical answer, then embed that. The hypothetical answer looks more like real documents, improving retrieval accuracy. Tradeoff: one extra LLM call per query.

11. **Q: What is RAGAS faithfulness and how is it measured?**
    A: Faithfulness measures whether every claim in the answer can be traced back to the retrieved context. Computed by: (1) decompose the answer into atomic claims; (2) check if each claim is supported by the context; (3) faithfulness = supported_claims / total_claims. A faithful answer doesn't hallucinate — it only states what's in the context.

12. **Q: What is semantic chunking and why is it better than fixed-size chunking?**
    A: Semantic chunking splits on topic boundaries rather than arbitrary token counts. It embeds consecutive sentences, detects when the topic changes (cosine similarity drops significantly), and splits there. Result: each chunk is topically coherent. Chunks don't split in the middle of a concept.

13. **Q: What are the vectors in Databricks Vector Search and how does it integrate with Delta Lake?**
    A: Databricks Vector Search maintains a sync between a Delta table (source) and a vector index. When the Delta table updates, the index automatically re-embeds and syncs. This enables: build your pipeline in PySpark, store in Delta, and query semantically via the Vector Search API.

**Advanced**

14. **Q: How would you handle a RAG system where the context window fills up with irrelevant retrieved content?**
    A: (1) Increase reranking precision — retrieve 50, rerank, keep only top 3. (2) Implement context compression — LLM extracts only relevant sentences from each chunk before including in context. (3) Reduce chunk size. (4) Use multi-query retrieval with deduplication. (5) Implement long-context reranker (Cohere Rerank).

15. **Q: What is self-RAG and how does it differ from standard RAG?**
    A: Self-RAG (Asai et al., 2023): the model itself decides whether to retrieve, what to retrieve, and whether the retrieved content is relevant. It generates special tokens: [Retrieve], [Relevant], [Grounded], [Utility] — the model is trained to use these. More accurate than standard RAG but requires a specially fine-tuned model.

16. **Q: How does product quantisation reduce vector storage requirements?**
    A: PQ splits each d-dimensional vector into m subvectors of d/m dimensions each. Each subvector is quantised to one of k centroids (typically k=256). Storage: m bytes per vector instead of d×4 bytes. For d=1536, k=256, m=8: 8 bytes vs. 6144 bytes — 768× compression. Query time: approximate dot products via lookup tables.

17. **Q: What is multi-vector retrieval (ColBERT) and when is it better than single-vector?**
    A: ColBERT stores one embedding per token (not per document). At query time, each query token finds its best matching document token (MaxSim operation). This captures fine-grained token-level relevance. Better for: precise factual retrieval, code search. Worse for: storage (N × seq_len embeddings per document).

18. **Q: How would you design a production RAG system for a 100GB document corpus?**
    A: (1) Distributed ingestion: Spark UDF for parallel chunking and embedding. (2) Vector store: Qdrant or Databricks Vector Search (distributed). (3) Tiered retrieval: BM25 (inverted index, fast) → dense ANN → cross-encoder rerank. (4) Caching: Redis semantic cache for repeated queries. (5) Monitoring: RAGAS evaluation on 5% of queries, latency tracking.

19. **Q: What is the "embedding model alignment problem" in RAG?**
    A: The embedding model used to embed documents must be the same (or compatible) as the one used to embed queries. Mismatch causes poor retrieval. Also: if your domain is specialised (medical, legal, code), general-purpose embeddings may not capture domain-specific semantics well. Solution: domain-specific embedding models or fine-tuned embeddings.

20. **Q: How would you evaluate RAG answer quality when ground truth answers aren't available?**
    A: LLM-as-judge pipeline: (1) faithfulness: LLM checks if answer claims are supported by context; (2) answer completeness: LLM checks if answer addresses all parts of the question; (3) groundedness: LLM checks if answer could be derived solely from context. Generate synthetic ground truth using GPT-4 over your corpus, then use RAGAS.

---

### 7. Self-Assessment Quiz

- [ ] What is cosine similarity and write the formula.
- [ ] What is the difference between a bi-encoder and a cross-encoder?
- [ ] What are the 4 RAGAS metrics and what does each measure?
- [ ] What is HyDE and when is it better than standard query embedding?
- [ ] What is BM25 and when does it outperform dense retrieval?
- [ ] What is RRF and why is it parameter-free?
- [ ] What is HNSW and what does the hierarchy represent?
- [ ] Name 3 chunking strategies and a use case for each.
- [ ] What is context precision vs. context recall in RAGAS?
- [ ] What is the tradeoff between recall and latency in vector search (ef parameter in HNSW)?
- [ ] What is metadata filtering and why is it needed?
- [ ] What is semantic chunking?
- [ ] What is multi-query retrieval?
- [ ] How does a vector index handle deletes and updates?
- [ ] What is a namespace/collection in vector databases?
- [ ] What does mean pooling vs. CLS pooling produce in sentence transformers?
- [ ] What is in-batch negative sampling for embedding fine-tuning?
- [ ] What is context compression in RAG?
- [ ] What is the embedding dimension of all-MiniLM-L6-v2? Of text-embedding-3-small?
- [ ] What is the difference between approximate and exact nearest neighbour search?
- [ ] What is a hallucination in RAG specifically (different from bare LLM hallucination)?
- [ ] What is chunk overlap and why is it used?
- [ ] What is a parent-child chunking strategy?
- [ ] What is the MTEB benchmark?
- [ ] What is Reciprocal Rank Fusion's k parameter and what does changing it do?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 5b.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Using fixed-size chunks for all documents | Simple to implement | Profile your corpus; long documents with topic shifts need semantic chunking |
| Not separating embedding model from generation model | Assume both use the same | Always specify the embedding model explicitly; it's a separate choice from the LLM |
| Skipping reranking | Adds complexity | Always benchmark: bi-encoder-only vs. bi-encoder+cross-encoder; the improvement is often 10-20% |
| Not evaluating with RAGAS | Eyeballing outputs | RAGAS is automated; there's no excuse for not running it |
| Using a general embedding model for specialised domain | Convenience | Test domain-specific models (e.g., e5-mistral for code) against general models |
| Not filtering retrieved chunks by metadata | All docs treated equally | Date filters prevent stale information; source filters improve trustworthiness |
| Using the same index for different document types | Convenience | Separate indexes for separate content types improve precision |
| Not handling PDF extraction failures | Most PDFs extract fine | Some PDFs are scanned images; handle `None` text, fall back to OCR |

---

### 9. Readiness Criteria

You are ready for Phase 6 when **all** of the following are true:

- [ ] I built a RAG pipeline from scratch without any framework (Coding Challenge A)
- [ ] I implemented hybrid search + CrossEncoder reranking (Coding Challenge B)
- [ ] I ran RAGAS evaluation and compared 3 configurations (Coding Challenge C)
- [ ] I completed the Databricks Documentation Assistant (Mini Project)
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I can explain why faithfulness is RAGAS's most important metric

---

### 10. Revision Summary

```
EMBEDDINGS
─────────────────────────────────────────────────────
Cosine sim:    (A·B) / (|A||B|)  — direction only, ignore magnitude
Bi-encoder:    embed independently → fast ANN → approximate relevance
Cross-encoder: embed jointly → slow but accurate → used for reranking
MTEB:          leaderboard for embedding model selection

RAG PIPELINE
─────────────────────────────────────────────────────
Ingest    → chunk (semantic preferred) → embed → store in vector DB
Retrieve  → ANN search (bi-encoder) → BM25 → RRF fusion
Rerank    → CrossEncoder top-50 → top-5
Generate  → LLM with prompt: context + question → answer + citations

RAGAS METRICS
─────────────────────────────────────────────────────
Faithfulness:       are ALL claims in the answer in the context?
Answer Relevancy:   does the answer address the question?
Context Precision:  are retrieved chunks actually relevant?
Context Recall:     does retrieved context contain all needed information?

ADVANCED PATTERNS
─────────────────────────────────────────────────────
HyDE:         embed LLM-generated hypothetical answer instead of query
Multi-query:  generate 3 query variants → retrieve for all → deduplicate
Compression:  LLM condenses retrieved chunks to only relevant sentences
```

---

### 11. Next Phase Prerequisites

**What Phase 6 (Agents & MCP) requires from Phase 5b:**

| Phase 5b Concept | How Phase 6 Uses It |
|-----------------|----------------------|
| Vector databases | Agents use vector stores for long-term memory |
| RAG pipeline | Many agents use RAG to look up knowledge during task execution |
| Retrieval evaluation | Agents are evaluated end-to-end; retrieval quality affects agent quality |
| Embedding search | Agent memory retrieval uses similarity search |
| ChromaDB / Qdrant API | Agent memory modules are built on vector DBs |
| LLM generation with context | Agents pass retrieved context into generation just like RAG |

**The critical dependency**: Agents are RAG systems that also take actions. The memory component of an agent is a vector database. The knowledge lookup is a retrieval step. Every RAG concept carries directly into agent design.

---

*Phase 5b | Part of the [GenAI Engineer Roadmap](./00_README.md)*
