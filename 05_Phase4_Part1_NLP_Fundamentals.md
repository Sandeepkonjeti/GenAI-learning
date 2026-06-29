# Phase 4a: NLP Fundamentals
**Month 9–10 | Difficulty: 6/10 | Label: 🔴 Must Learn**

> **Previous Phase**: [Phase 3b — PyTorch](./04_Phase3_Part2_PyTorch.md)  
> **Next Phase**: [Phase 4b — Transformers](./05_Phase4_Part2_Transformers.md)

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Why NLP Fundamentals?](#why-nlp-fundamentals)
- [Topic 1: Text as Data](#topic-1-text-as-data)
  - [The Core NLP Problem](#the-core-nlp-problem)
  - [Tokenization](#tokenization)
  - [Text Preprocessing](#text-preprocessing)
- [Topic 2: Classical Representations](#topic-2-classical-representations)
  - [Bag of Words](#bag-of-words)
  - [TF-IDF](#tf-idf)
  - [Why Classical Representations Fall Short](#why-classical-representations-fall-short)
- [Topic 3: Word Embeddings](#topic-3-word-embeddings)
  - [The Distributional Hypothesis](#the-distributional-hypothesis)
  - [Word2Vec](#word2vec)
  - [GloVe and FastText](#glove-and-fasttext)
  - [Properties of Word Embeddings](#properties-of-word-embeddings)
- [Topic 4: Sequence Modeling](#topic-4-sequence-modeling)
  - [The Sequence Problem](#the-sequence-problem)
  - [RNNs for Text](#rnns-for-text)
  - [LSTMs and GRUs](#lstms-and-grus)
  - [Seq2Seq and the Birth of Attention](#seq2seq-and-the-birth-of-attention)
- [Topic 5: Modern Tokenization](#topic-5-modern-tokenization)
  - [Byte Pair Encoding (BPE)](#byte-pair-encoding-bpe)
  - [WordPiece and SentencePiece](#wordpiece-and-sentencepiece)
  - [Why Tokenization Matters for LLMs](#why-tokenization-matters-for-llms)
- [Resources](#resources)
- [Phase 4a Projects](#phase-4a-projects)
- [Common Mistakes](#common-mistakes)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 9–10 (3 weeks) |
| **Daily Time** | 1–2 hours |
| **Difficulty** | 6/10 |
| **Label** | 🔴 Must Learn |
| **Prerequisites** | Phase 3b (PyTorch fluent), Phase 1 (linear algebra, probability) |
| **Outcome** | Understands text representation from BoW to embeddings; can implement tokenization; understands sequence modeling limitations |

---

## Why NLP Fundamentals?

Before transformers replaced everything, NLP had decades of research. Understanding this history is not nostalgia — it is essential context:

1. **Word2Vec embeddings** gave the field its first insight into dense representations. Every embedding model today is a descendant of this idea.
2. **TF-IDF** is still used in hybrid search systems inside RAG pipelines you'll build in Phase 5.
3. **Seq2Seq with attention** was the direct precursor to transformers. Understanding it makes transformers intuitive, not magical.
4. **Tokenization** — BPE, WordPiece, SentencePiece — directly affects how LLMs reason. Understanding this is essential for prompt engineering and fine-tuning.

---

## Topic 1: Text as Data

### The Core NLP Problem

Neural networks work with numbers, not words. Every NLP system must solve the problem: **how do you represent language as numbers?**

The progression of solutions:
```
One-Hot Encoding → Bag of Words → TF-IDF → Word2Vec → Transformers
     (sparse)          (sparse)    (sparse)   (dense)    (contextual)
```

Each step solved a problem the previous approach couldn't handle.

---

### Tokenization

Before embedding text, you must split it into units (tokens). This seems simple but has profound implications for LLMs.

```python
# The progression of tokenization approaches

# ── Level 1: Character-level ──
text = "Hello, world!"
char_tokens = list(text)
print(char_tokens)  # ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']
# Problem: very long sequences; no word-level meaning

# ── Level 2: Word-level ──
import re
word_tokens = re.findall(r'\b\w+\b', text.lower())
print(word_tokens)  # ['hello', 'world']
# Problem: vocabulary size can be millions; OOV (out-of-vocabulary) words

# ── Level 3: Subword (modern approach) ──
# BPE, WordPiece — see Topic 5

# ── Practical tokenization with NLTK ──
import nltk
nltk.download('punkt', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize

text = "Dr. Smith went to Washington. He said, 'AI is transforming everything!'"
sentences = sent_tokenize(text)
for sent in sentences:
    tokens = word_tokenize(sent)
    print(f"  {tokens}")
```

---

### Text Preprocessing

```python
import re
import string
import unicodedata
from typing import List

class TextPreprocessor:
    """
    Text preprocessing pipeline — still useful for classical NLP
    and for cleaning data before feeding to LLMs.
    """
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,  # Keep for LLM inputs
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True
    ):
        self.lowercase = lowercase
        self.remove_punct = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_ws = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
    
    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode (é → e, etc.)
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
            text = text.encode('ascii', 'ignore').decode('utf-8')
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if self.remove_extra_ws:
            text = ' '.join(text.split())
        
        return text.strip()
    
    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

# ── Stemming vs. Lemmatization ──
# Only relevant for classical NLP; LLMs handle morphology automatically

from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet', quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ['running', 'ran', 'runs', 'easily', 'fairly', 'caring', 'cared']

print(f"{'Word':<12} {'Stem':<12} {'Lemma':<12}")
print("-" * 36)
for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)
    print(f"{word:<12} {stem:<12} {lemma:<12}")

# Stemming: fast, crude (running → run, but easily → easili)
# Lemmatization: slower, linguistically correct (running → run, easily → easily)
```

---

## Topic 2: Classical Representations

### Bag of Words

```python
import numpy as np
from collections import Counter
from typing import List, Dict

class BagOfWords:
    """
    Bag of Words: represent each document as a word count vector.
    
    COMPLETELY ignores word order and context.
    "The dog bit the man" == "The man bit the dog"
    
    This is why it works for some tasks (spam detection) but fails for others
    (sentiment analysis: "not good" ≠ "good").
    """
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
    
    def fit(self, documents: List[List[str]]):
        """Build vocabulary from training documents."""
        all_words = [word for doc in documents for word in doc]
        word_counts = Counter(all_words)
        # Assign index to each word
        self.vocabulary = {word: idx for idx, (word, _) in 
                          enumerate(word_counts.most_common())}
        return self
    
    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Convert documents to count vectors."""
        vectors = np.zeros((len(documents), len(self.vocabulary)))
        for i, doc in enumerate(documents):
            for word in doc:
                if word in self.vocabulary:
                    vectors[i, self.vocabulary[word]] += 1
        return vectors
    
    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        return self.fit(documents).transform(documents)

# Example
docs_raw = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "cats and dogs are pets"
]
docs_tokenized = [doc.split() for doc in docs_raw]

bow = BagOfWords()
X = bow.fit_transform(docs_tokenized)

print("Vocabulary (first 10):", list(bow.vocabulary.keys())[:10])
print("Vector shape:", X.shape)
print("Document 1 vector:")
for word, idx in sorted(bow.vocabulary.items(), key=lambda x: x[1])[:10]:
    if X[0, idx] > 0:
        print(f"  '{word}': {int(X[0, idx])}")
```

### TF-IDF

```python
import numpy as np
from typing import List
import math

class TFIDF:
    """
    TF-IDF: Term Frequency - Inverse Document Frequency
    
    TF(t, d) = count(t in d) / count(words in d)
    IDF(t) = log(N / count(documents containing t))
    TF-IDF(t, d) = TF(t, d) * IDF(t)
    
    Key insight: words that appear in MANY documents (the, and, is)
    have low IDF → low TF-IDF → effectively filtered out.
    Rare but document-specific words have high TF-IDF.
    
    Still used in production today for:
    - Hybrid search (keyword + semantic)
    - BM25 retrieval in RAG systems
    - Document clustering
    """
    def __init__(self, max_features: int = None, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary: dict = {}
        self.idf_: np.ndarray = None
    
    def fit(self, documents: List[List[str]]):
        N = len(documents)
        
        # Count document frequency for each term
        df = Counter()
        for doc in documents:
            df.update(set(doc))  # Count each term once per document
        
        # Filter by min_df and limit vocabulary
        vocab_terms = [(term, count) for term, count in df.items() 
                      if count >= self.min_df]
        vocab_terms.sort(key=lambda x: x[1], reverse=True)
        
        if self.max_features:
            vocab_terms = vocab_terms[:self.max_features]
        
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(vocab_terms)}
        
        # Compute IDF with smoothing (sklearn convention)
        n_vocab = len(self.vocabulary)
        self.idf_ = np.zeros(n_vocab)
        for term, idx in self.vocabulary.items():
            self.idf_[idx] = math.log((N + 1) / (df[term] + 1)) + 1
        
        return self
    
    def transform(self, documents: List[List[str]]) -> np.ndarray:
        vectors = np.zeros((len(documents), len(self.vocabulary)))
        
        for i, doc in enumerate(documents):
            doc_len = len(doc)
            if doc_len == 0:
                continue
            
            tf = Counter(doc)
            for term, count in tf.items():
                if term in self.vocabulary:
                    j = self.vocabulary[term]
                    tf_score = count / doc_len
                    vectors[i, j] = tf_score * self.idf_[j]
        
        # L2 normalization (standard)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)
        
        return vectors

# TF-IDF similarity search (still used in RAG hybrid search)
def tfidf_search(query: str, documents: List[str], top_k: int = 3) -> List[tuple]:
    """Search documents using TF-IDF similarity."""
    all_docs = [query] + documents
    tokenized = [doc.lower().split() for doc in all_docs]
    
    tfidf = TFIDF(max_features=10000)
    vectors = tfidf.fit_transform(tokenized)
    
    query_vec = vectors[0]
    doc_vecs = vectors[1:]
    
    # Cosine similarity (vectors are L2-normalized, so dot product = cosine sim)
    similarities = doc_vecs @ query_vec
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(documents[i], similarities[i]) for i in top_indices]
```

### Why Classical Representations Fall Short

```python
# The fundamental limitations of BoW and TF-IDF

examples = {
    "semantic gap": [
        "The car needs fuel.",
        "The automobile requires gasoline.",  # Same meaning, zero similarity in BoW!
    ],
    "word order matters": [
        "The dog bit the man.",
        "The man bit the dog.",  # Opposite meaning, identical BoW representation!
    ],
    "context ignored": [
        "I went to the bank to deposit money.",
        "I sat on the river bank to fish.",  # 'bank' has different meanings
    ]
}

print("Why BoW/TF-IDF fail:")
print()
for issue, pair in examples.items():
    print(f"Problem: {issue}")
    print(f"  Text 1: {pair[0]}")
    print(f"  Text 2: {pair[1]}")
    print(f"  BoW: These texts look {'identical' if issue == 'word order matters' else 'unrelated'}")
    print()

# Solution: dense word embeddings → then contextual embeddings (transformers)
```

---

## Topic 3: Word Embeddings

### The Distributional Hypothesis

> *"A word is characterized by the company it keeps."* — Firth (1957)

Words that appear in similar contexts have similar meanings. If "cat" and "dog" both appear near "pet", "fur", "food", and "veterinarian", they should have similar vector representations.

This single insight is the foundation of all word embeddings, sentence embeddings, and contextual representations in transformers.

---

### Word2Vec

Word2Vec (Mikolov et al., 2013) trains a neural network on a simple prediction task that forces it to learn meaningful word representations.

**Two architectures**:

| Model | Task | Intuition |
|-------|------|-----------|
| **CBOW** (Continuous Bag of Words) | Predict center word from context words | Given "The ___ sat on the mat", predict "cat" |
| **Skip-gram** | Predict context words from center word | Given "cat", predict "sat", "mat", "the" |

```python
import numpy as np
from typing import List, Tuple

class SimpleWord2Vec:
    """
    Simplified Word2Vec (Skip-gram) implementation.
    
    The key insight: the EMBEDDINGS are not the output of the network.
    They ARE the network itself — the weight matrix of the embedding layer.
    By training to predict context words, the weights self-organize 
    into meaningful vector spaces.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100, window: int = 2):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window = window
        
        # The embedding matrix IS the model
        # Each row = embedding vector for one word
        scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.W = np.random.randn(vocab_size, embedding_dim) * scale
        self.W_context = np.random.randn(embedding_dim, vocab_size) * scale
    
    def generate_training_pairs(
        self, token_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """Generate (center_word, context_word) pairs for skip-gram."""
        pairs = []
        n = len(token_ids)
        for i, center in enumerate(token_ids):
            start = max(0, i - self.window)
            end = min(n, i + self.window + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, token_ids[j]))
        return pairs
    
    def forward(self, center_idx: int) -> np.ndarray:
        """Forward pass: center word → predict context word logits."""
        # Look up center word embedding
        h = self.W[center_idx]              # (embedding_dim,)
        # Compute scores for all vocabulary words
        scores = h @ self.W_context         # (vocab_size,)
        # Softmax
        scores -= scores.max()
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum()
    
    def get_embedding(self, word_idx: int) -> np.ndarray:
        return self.W[word_idx].copy()
    
    def most_similar(self, word_idx: int, top_k: int = 5) -> List[int]:
        """Find most similar words by cosine similarity."""
        query = self.W[word_idx]
        # Vectorized cosine similarity
        norms = np.linalg.norm(self.W, axis=1)
        similarities = (self.W @ query) / (norms * np.linalg.norm(query) + 1e-10)
        # Exclude the query word itself
        similarities[word_idx] = -1
        return np.argsort(similarities)[::-1][:top_k].tolist()

# Using pre-trained Word2Vec with Gensim
# (The real implementation; don't train from scratch in practice)
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

# Load pre-trained GloVe embeddings
# glove_vectors = api.load('glove-wiki-gigaword-100')  # 100-dim GloVe

# Or train on your own corpus
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "rug"],
    ["cats", "and", "dogs", "are", "pets"],
]

model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # embedding dimension
    window=5,           # context window
    min_count=1,        # minimum word frequency
    sg=1,               # 1 = skip-gram, 0 = CBOW
    epochs=100
)

# Word embeddings are now available
cat_vector = model.wv['cat']          # shape: (100,)
dog_vector = model.wv['dog']
mat_vector = model.wv['mat']

# Cosine similarity between words
print(f"cat-dog similarity: {model.wv.similarity('cat', 'dog'):.4f}")
print(f"cat-mat similarity: {model.wv.similarity('cat', 'mat'):.4f}")
# cat and dog should be more similar than cat and mat
```

### GloVe and FastText

```python
# GloVe (Global Vectors) — key differences from Word2Vec
# Word2Vec: local context (window)
# GloVe: global co-occurrence statistics
# Both produce similar quality embeddings in practice

# FastText — adds subword information
# Represents words as bags of character n-grams
# "playing" = "pla" + "lay" + "ayi" + ... + "ing" + "playing"
# Advantage: can represent OOV (out-of-vocabulary) words by composing their n-grams!

from gensim.models import FastText

fasttext_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,     # minimum n-gram size
    max_n=6,     # maximum n-gram size
    epochs=100
)

# FastText can embed UNSEEN words
unknown_word_vector = fasttext_model.wv['unfamiliarword']  # Works!
# Word2Vec would fail with KeyError for unknown words

# This is why FastText is useful for:
# - Morphologically rich languages
# - Technical domains with many compound words
# - Misspellings and typos
```

### Properties of Word Embeddings

```python
import numpy as np

# The famous Word2Vec arithmetic:
# king - man + woman ≈ queen
# This works because the embedding space captures semantic relationships geometrically!

def analogy(model_wv, word_a, word_b, word_c, top_k=1):
    """
    Solve: word_a is to word_b as word_c is to ???
    vector(b) - vector(a) + vector(c)
    """
    # Get vectors
    vec_a = model_wv[word_a]
    vec_b = model_wv[word_b]
    vec_c = model_wv[word_c]
    
    # Compute target vector
    target = vec_b - vec_a + vec_c
    
    # Find nearest neighbors (excluding input words)
    result = model_wv.similar_by_vector(
        target, topn=top_k + 3  # Get extra, then filter
    )
    result = [(w, s) for w, s in result if w not in {word_a, word_b, word_c}]
    return result[:top_k]

# Using actual pre-trained GloVe (illustrative)
# print(analogy(glove_vectors, 'man', 'king', 'woman'))
# → [('queen', 0.77)]

# Understanding what the geometry means:
# The DIRECTION from 'man' to 'king' encodes "royalty"
# The DIRECTION from 'man' to 'woman' encodes "gender"
# These directions are approximately linear in the embedding space!

# Other linear relationships in word vectors:
# Paris - France + Germany ≈ Berlin   (capital cities)
# running - run + walk ≈ walking      (verb conjugation)
# good - better + bad ≈ worse         (comparative adjectives)

# Why this matters for LLMs:
# LLMs learn much richer, contextual representations
# But the fundamental idea — dense vectors capturing semantic relationships —
# is the same. The geometry of semantic space is a core concept.
```

---

## Topic 4: Sequence Modeling

### The Sequence Problem

Natural language is a **sequence** — word order matters fundamentally. Bag of Words ignores this. We need models that can:

1. Process words in order (left to right, or bidirectionally)
2. Maintain a memory of what was said earlier
3. Capture long-range dependencies ("the bank that the man who wore a hat went to was open")

### RNNs for Text

```python
import torch
import torch.nn as nn

class TextRNN(nn.Module):
    """
    Simple RNN for text classification.
    
    Key ideas:
    - Hidden state h_t acts as a "memory" 
    - Same weights W applied at each step (parameter sharing)
    - Final hidden state summarizes the entire sequence
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,    # (batch, seq, features)
            dropout=0.3
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids shape: (batch, seq_len)
        x = self.embedding(token_ids)   # (batch, seq_len, embed_dim)
        
        # RNN processes sequence; output = all hidden states
        # final_hidden = hidden state at last step
        output, final_hidden = self.rnn(x)
        
        # Use last step's hidden state for classification
        last_hidden = final_hidden[-1]   # (batch, hidden_dim)
        return self.classifier(last_hidden)

# Visualize the vanishing gradient problem
import numpy as np
import matplotlib.pyplot as plt

def simulate_gradient_flow(seq_len: int = 100, n_simulations: int = 100) -> np.ndarray:
    """
    Show how gradients shrink as they flow through RNN time steps.
    """
    gradient_norms = []
    
    for _ in range(n_simulations):
        # Each time step multiplies gradient by tanh'(z) ≈ random value ≤ 1
        grad = 1.0
        norms_per_step = [grad]
        
        for _ in range(seq_len):
            # tanh derivative: between 0 and 1
            tanh_grad = np.random.uniform(0.1, 0.99)
            # Weight matrix also multiplies gradient
            weight_factor = np.random.uniform(0.8, 1.2)  # near identity
            grad *= tanh_grad * weight_factor
            norms_per_step.append(abs(grad))
        
        gradient_norms.append(norms_per_step)
    
    return np.array(gradient_norms)

grads = simulate_gradient_flow(seq_len=100)
mean_grads = grads.mean(axis=0)

plt.figure(figsize=(10, 4))
plt.semilogy(mean_grads)
plt.xlabel('Time step (distance from loss)')
plt.ylabel('Gradient magnitude (log scale)')
plt.title('Vanishing Gradient in RNN\n(Gradient at early steps → 0)')
plt.axhline(0.01, color='r', linestyle='--', label='Near-zero threshold')
plt.legend(); plt.show()
```

### LSTMs and GRUs

```python
import torch
import torch.nn as nn

# LSTM (Long Short-Term Memory) adds gating mechanisms to control memory
# This largely solves the vanishing gradient problem for moderate sequence lengths

class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM for sentiment classification.
    
    Bidirectional: processes sequence left→right AND right→left
    Concatenates both directions for richer representation.
    This is the architecture behind early BERT-like models.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, 
                 n_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # PAD token doesn't contribute to gradients
        )
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,    # Process both directions
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        # *2 because bidirectional
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)
    
    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # Embed tokens
        x = self.dropout(self.embedding(token_ids))  # (B, T, embed_dim)
        
        # Pack padded sequences for efficient processing (optional but good practice)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Process through LSTM
        output, (hidden, cell) = self.lstm(x)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Concatenate last hidden states from both directions
        # hidden shape: (n_layers * 2, batch, hidden_dim)
        forward_hidden = hidden[-2]   # Last layer, forward direction
        backward_hidden = hidden[-1]  # Last layer, backward direction
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return self.classifier(self.dropout(combined))

# LSTM gates (conceptual) — what makes them work:
lstm_gates_explanation = """
LSTM has 4 gates that control information flow:

Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
              "What to FORGET from cell state"
              
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
              "What NEW information to let IN"
              
Cell update:  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
              "What new content to potentially add"
              
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
              "What to OUTPUT from cell state"

Cell state:   C_t = f_t * C_{t-1} + i_t * C̃_t
              (Forget old + add new)
              
Hidden state: h_t = o_t * tanh(C_t)

Key insight: the cell state C_t has a DIRECT path that gradients 
can flow through with just addition (no multiplication by weights).
This prevents vanishing gradients for moderate sequence lengths.
"""
print(lstm_gates_explanation)
```

### Seq2Seq and the Birth of Attention

```python
# The encoder-decoder architecture (Sutskever et al., 2014)
# This was used for machine translation BEFORE transformers

# Problem: encode "Je suis étudiant" (French) → decode "I am a student" (English)
# The encoder reads the source sentence into a fixed-size "context vector"
# The decoder generates the target sentence from this context vector

# LIMITATION: the entire source sentence must be compressed into ONE vector!
# For long sentences, information at the beginning gets lost.

# Solution: Bahdanau Attention (2015) — ALLOWED DECODER TO LOOK BACK AT ENCODER STATES
# This was the KEY insight that led to transformers.

# Attention (simplified):
# For each decoder step, compute attention scores over all encoder hidden states
# Use these scores to create a weighted sum of encoder states (context vector)
# Use this DYNAMIC context vector instead of a fixed one

class AttentionLayer(nn.Module):
    """
    Bahdanau-style attention.
    This is the direct precursor to transformer attention.
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int):
        super().__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attn_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    
    def forward(
        self, 
        decoder_hidden: torch.Tensor,    # (batch, decoder_dim) — current step
        encoder_outputs: torch.Tensor    # (batch, src_len, encoder_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Expand decoder hidden for broadcasting
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch, 1, decoder_dim)
        
        # Compute attention scores
        energy = torch.tanh(
            self.encoder_attn(encoder_outputs) +  # (batch, src_len, attn_dim)
            self.decoder_attn(decoder_hidden)      # (batch, 1, attn_dim) — broadcasts
        )
        attention_scores = self.v(energy).squeeze(-1)  # (batch, src_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch, src_len)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs                  # (batch, src_len, encoder_dim)
        ).squeeze(1)                         # (batch, encoder_dim)
        
        return context, attention_weights

# This attention mechanism was later generalized to the self-attention
# in transformers. Instead of attending from decoder to encoder,
# transformers have each position attend to ALL positions in the same sequence.
```

---

## Topic 5: Modern Tokenization

### Byte Pair Encoding (BPE)

BPE is the tokenization algorithm used by GPT-2, GPT-3, GPT-4, and most modern LLMs.

**Algorithm**:
1. Start with character-level vocabulary
2. Find the most frequent pair of tokens
3. Merge that pair into a new token
4. Repeat N times (N = vocabulary size target)

```python
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer from scratch.
    
    This is the algorithm used by GPT-2, GPT-3, GPT-4, LLaMA, etc.
    Understanding BPE is essential for understanding:
    - Why LLMs struggle with certain words (rare subwords)
    - How to count tokens for cost estimation
    - Why fine-tuning data format matters
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []  # Learned merge rules
        self.vocab: Dict[str, int] = {}
    
    def get_vocab(self, corpus: List[str]) -> Dict[str, int]:
        """Convert corpus to initial character-level vocabulary with word boundaries."""
        word_freqs = Counter()
        for text in corpus:
            words = text.lower().split()
            for word in words:
                # Add word boundary marker (</w> = end of word)
                word_with_end = ' '.join(list(word)) + ' </w>'
                word_freqs[word_with_end] += 1
        return dict(word_freqs)
    
    def get_pairs(self, vocab: Dict[str, int]) -> Dict[Tuple, int]:
        """Count all adjacent symbol pairs."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in the vocabulary."""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab:
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def fit(self, corpus: List[str]):
        """Learn BPE merges from corpus."""
        vocab = self.get_vocab(corpus)
        
        # Initial vocabulary: all characters
        all_chars = set()
        for word in vocab:
            all_chars.update(word.split())
        self.vocab = {char: idx for idx, char in enumerate(sorted(all_chars))}
        
        # Iteratively merge most frequent pairs
        n_merges = self.vocab_size - len(self.vocab)
        for i in range(n_merges):
            pairs = self.get_pairs(vocab)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # Merge the pair in vocabulary
            vocab = self.merge_vocab(best_pair, vocab)
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
        
        return self
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            word_tokens = list(word) + ['</w>']
            
            # Apply learned merges in order
            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        word_tokens[i] == pair[0] and 
                        word_tokens[i+1] == pair[1]):
                        new_tokens.append(''.join(pair))
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            token_ids.extend([self.vocab.get(t, 0) for t in word_tokens])
        
        return token_ids

# Demonstrate BPE
corpus = [
    "the cat sat on the mat",
    "the cat sat on the flat mat",
    "the cat sat",
    "cats are cute animals",
    "the dog sat on the rug",
    "dogs and cats are both pets",
] * 100  # Repeat to simulate larger corpus

tokenizer = BPETokenizer(vocab_size=200)
tokenizer.fit(corpus)

print("First 20 learned merges:")
for i, merge in enumerate(tokenizer.merges[:20]):
    print(f"  {i+1:2d}: {merge[0]} + {merge[1]} → {''.join(merge)}")

test_text = "the cat sat"
tokens = tokenizer.encode(test_text)
print(f"\nEncoding '{test_text}':")
print(f"  Token IDs: {tokens}")
```

### WordPiece and SentencePiece

```python
# WordPiece: used by BERT, BERT-based models
# Similar to BPE but uses a different merge criterion:
# maximize log-likelihood of training data

# Key difference from BPE:
# - BPE: merge most frequent pair
# - WordPiece: merge pair that maximizes language model likelihood
# - WordPiece uses ## prefix for continuation tokens: "playing" → ["play", "##ing"]

# SentencePiece: Google's tokenizer (used by T5, LLaMA, Gemma)
# Key difference: treats text as a stream of bytes (works for any language)
# No pre-tokenization step — handles spaces as regular characters
# Uses 'Ġ' or '▁' to indicate word starts

# Using Hugging Face tokenizers (production usage)
from transformers import AutoTokenizer

# GPT-2 / GPT-4 style (BPE)
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Hello, world! This is an example of GPT tokenization."

tokens = gpt2_tokenizer.encode(text)
decoded_tokens = [gpt2_tokenizer.decode([t]) for t in tokens]

print("GPT-2 (BPE) tokenization:")
for tok_id, tok_str in zip(tokens, decoded_tokens):
    print(f"  {tok_id:6d}: '{tok_str}'")

# BERT style (WordPiece)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = bert_tokenizer.tokenize(text)
print(f"\nBERT (WordPiece) tokenization:")
print(f"  {bert_tokens}")

# LLaMA style (SentencePiece)
# llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
```

### Why Tokenization Matters for LLMs

```python
# Critical tokenization effects you must understand

# 1. Token count affects cost and latency
text = "Summarize this document in 3 bullet points."
tokens = gpt2_tokenizer.encode(text)
print(f"Text: '{text}'")
print(f"Token count: {len(tokens)}")
print(f"Approx cost (GPT-4): ${len(tokens) * 0.00003:.5f}")

# 2. Numbers are tokenized differently
number_examples = ["1234567", "1,234,567", "1.234567"]
for num in number_examples:
    toks = gpt2_tokenizer.encode(num)
    print(f"'{num}' → {len(toks)} tokens: {[gpt2_tokenizer.decode([t]) for t in toks]}")
# LLMs struggle with arithmetic partly because numbers are split into odd subwords

# 3. Code tokenization
code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
code_tokens = gpt2_tokenizer.encode(code)
print(f"\nCode token count: {len(code_tokens)}")

# 4. Special tokens matter for fine-tuning
# Different models use different special tokens:
# GPT-2: <|endoftext|>
# LLaMA: <s>, </s>, [INST], [/INST]
# ChatML: <|im_start|>system\n...<|im_end|>
# Getting these wrong during fine-tuning causes catastrophic failures

# Example fine-tuning format (ChatML):
chatml_example = """<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
2+2 equals 4.
<|im_end|>"""
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [Stanford CS224N (NLP with Deep Learning)](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6zgCd4Xn6sGpOH) | YouTube | Free | Gold standard NLP course. Lectures 1–5 for this phase. |
| 2 | [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) | Course | Free | Practical, modern, uses Hugging Face tokenizers. Chapter 1–3 for this phase. |
| 3 | [Speech and Language Processing, 3rd ed. (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/) | Book | Free PDF | The NLP bible. Chapters 2–10 cover this phase. |
| 4 | [Jay Alammar's Blog](https://jalammar.github.io/) | Blog | Free | Best visual explanations of Word2Vec, transformers, LLMs. Read all NLP posts. |
| 5 | [The Illustrated Word2Vec (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/) | Blog | Free | Single best explanation of word embeddings. Required reading. |

**Best YouTube**: Stanford CS224N — especially lectures by Christopher Manning on word vectors.

---

## Phase 4a Projects

### Project 7: Sentiment Analysis Pipeline
**Difficulty**: 4/10 | **Time**: 1 week

Build an end-to-end sentiment classifier:
1. Data: IMDB movie reviews (50K reviews, binary positive/negative)
2. Implement TF-IDF from scratch → logistic regression baseline
3. Train BiLSTM with GloVe embeddings → compare
4. Evaluation: F1, confusion matrix, learning curves
5. Error analysis: what does the model get wrong?

```python
# Target comparison:
# Method               | F1 Score
# TF-IDF + LogReg      | ~0.85
# BiLSTM + GloVe       | ~0.90
# Fine-tuned BERT      | ~0.93 (you'll do this in Phase 4b)
```

### Project 8: Custom BPE Tokenizer
**Difficulty**: 4/10 | **Time**: 3 days

Build and train a BPE tokenizer on a text corpus:
1. Implement BPE from scratch (the code above)
2. Train on a domain-specific corpus (technical docs, code, etc.)
3. Compare token counts vs. GPT-2 tokenizer on the same text
4. Visualize most frequent merged tokens (word cloud)

---

## Common Mistakes

| Mistake | Why | Fix |
|---------|-----|-----|
| Skipping word embeddings | "Transformers are what matters" | Word2Vec intuition is the foundation of all embedding understanding |
| Not studying the attention precursor | Can't explain WHY transformers work | Study Bahdanau attention first |
| Ignoring tokenization details | Wrong fine-tuning data format | Understand BPE, special tokens, chat templates before Phase 7 |
| Treating RNNs as obsolete | They appear in audio models, time series | Understand them; don't go deep |
| Skipping the vanishing gradient problem | "Transformers don't have this" | Understanding the PROBLEM makes the SOLUTION (transformers) intuitive |

---

## Mastery Checklist

- [ ] Can explain TF-IDF and when it's still useful today (hybrid RAG search)
- [ ] Can explain the distributional hypothesis and why it enables word embeddings
- [ ] Understands skip-gram training objective and how it produces embeddings
- [ ] Can explain why `king - man + woman ≈ queen` works geometrically
- [ ] Understands BiLSTM architecture and why bidirectional helps
- [ ] Can explain the vanishing gradient problem quantitatively
- [ ] Understands why Seq2Seq attention was a breakthrough
- [ ] Can explain BPE algorithm step-by-step
- [ ] Understands how tokenization affects LLM reasoning and arithmetic
- [ ] Read "The Illustrated Word2Vec" by Jay Alammar
- [ ] Completed Projects 7 and 8

---

## Moving to Phase 4b

**Before proceeding to [Phase 4b: Transformers](./05_Phase4_Part2_Transformers.md), confirm:**

- [ ] Implemented BPE tokenizer from scratch
- [ ] Trained a BiLSTM sentiment classifier
- [ ] Can explain why RNNs have vanishing gradients
- [ ] Understands the Seq2Seq attention mechanism
- [ ] Read Jay Alammar's Word2Vec and Seq2Seq attention blog posts

**Why transformers come next**: You now understand the problem transformers solve (vanishing gradients, fixed context vectors) and the tools they were built from (attention). Phase 4b takes the Bahdanau attention mechanism you studied here and extends it into the full transformer architecture.

---

## Phase Completion & Readiness Assessment

> Complete this assessment **before** moving to Phase 4b (Transformers). NLP fundamentals explain *why* transformers were invented and what problem each component solves.

---

### 1. Knowledge Checklist

**Text Preprocessing**
- [ ] Tokenisation: word, character, subword (BPE, WordPiece, SentencePiece)
- [ ] Stop words, stemming, lemmatisation — when each is appropriate
- [ ] Vocabulary, OOV (out-of-vocabulary) tokens
- [ ] Text normalisation: lowercasing, punctuation, unicode handling

**Bag-of-Words & Classical NLP**
- [ ] Bag-of-Words representation: construction and limitations
- [ ] TF-IDF: TF formula, IDF formula, combined score, why IDF matters
- [ ] N-gram language models: bigram, trigram, Markov assumption
- [ ] Perplexity as a language model evaluation metric

**Word Embeddings**
- [ ] Word2Vec: CBOW vs. Skip-gram architectures
- [ ] Word2Vec training objective: maximise P(context|center word)
- [ ] Negative sampling: why it's used instead of full softmax
- [ ] Word arithmetic: king - man + woman = queen (why this works)
- [ ] GloVe: global co-occurrence vs. Word2Vec local windows
- [ ] FastText: subword information, why it handles OOV

**Sequence Models**
- [ ] RNN: hidden state, unrolling through time, weight sharing
- [ ] Vanishing gradient through sequences: why long-range dependencies fail
- [ ] LSTM: cell state, input gate, forget gate, output gate — each formula
- [ ] GRU: simplified LSTM, reset gate, update gate
- [ ] Bidirectional RNNs: why forward and backward context both matter
- [ ] Seq2Seq: encoder-decoder architecture
- [ ] Bahdanau attention: the original attention mechanism, alignment scores

---

### 2. Practical Skills Checklist

- [ ] Implement TF-IDF from scratch (without sklearn) and build a search engine
- [ ] Train Word2Vec with gensim and perform word arithmetic
- [ ] Implement a character-level or word-level n-gram language model from scratch
- [ ] Build an LSTM text classifier in PyTorch that uses pre-trained Word2Vec embeddings
- [ ] Implement Bahdanau attention from scratch in PyTorch
- [ ] Calculate perplexity for a language model on held-out data
- [ ] Compare TF-IDF vs. Word2Vec vs. sentence-transformers for text similarity

---

### 3. Coding Challenges

**Challenge A — TF-IDF Search Engine**
```python
# Implement from scratch (NumPy only, no sklearn):
# 1. TfidfVectorizer class with fit(corpus) and transform(texts) methods
# 2. Cosine similarity between a query and all documents
# 3. Return top-k most relevant documents with their scores
# Test: build a search engine over 1000 Wikipedia abstracts
# Validate: does it return sensible results for "machine learning" queries?
```

**Challenge B — LSTM Sentiment Classifier**
```python
# Build an LSTM classifier in PyTorch:
# - Embedding layer (initialised from pre-trained GloVe vectors)
# - 2-layer bidirectional LSTM
# - Attention pooling (not just last hidden state)
# - Linear classifier head
# Train on IMDb movie reviews (50k examples)
# Achieve > 87% test accuracy
# Compare: no pre-trained embeddings vs. GloVe init (show accuracy difference)
```

**Challenge C — Bahdanau Attention**
```python
# Implement the attention mechanism from scratch:
class BahdanauAttention(nn.Module):
    # score(q, k) = V * tanh(W1*q + W2*k)
    # weights = softmax(scores)
    # context = weights @ values
    pass

# Integrate into a seq2seq encoder-decoder for character-level translation
# Visualise the attention weights as a heatmap for 5 example pairs
```

---

### 4. Mini Project

**Comparison Study** (Project 8 from roadmap, fully completed):
- Dataset: IMDb sentiment (50k reviews, balanced)
- Implement and benchmark 4 approaches:
  1. TF-IDF + Logistic Regression
  2. Word2Vec embeddings + LSTM
  3. all-MiniLM-L6-v2 sentence transformer embeddings + LogReg
  4. Fine-tuned DistilBERT (just to see the gap)
- Report: accuracy, F1, latency per sample, memory footprint
- Error analysis: 20 examples that TF-IDF gets wrong but LSTM gets right

---

### 5. Capstone Project

**Text Similarity Engine**: Build a semantic search system without transformers:
- Corpus: 5000 news articles (BBC or Reuters)
- Embeddings: TF-IDF and average Word2Vec
- Implement: exact keyword search, TF-IDF semantic search, Word2Vec semantic search
- UI: Streamlit app where users enter a query and see ranked results
- Evaluation: manually annotate 50 test queries; precision@5 for each method
- Show: Word2Vec captures synonyms that TF-IDF completely misses

---

### 6. Interview Questions

**Beginner**

1. **Q: What is TF-IDF and why is IDF important?**
   A: TF (term frequency) measures how often a word appears in a document. IDF (inverse document frequency) = log(N / df) measures how rare the word is across the corpus. The product TF × IDF gives high scores to words that are frequent in the document but rare globally — capturing specificity rather than just frequency.

2. **Q: What is Word2Vec and how does it learn word representations?**
   A: Word2Vec trains a shallow neural network to predict context words (Skip-gram) or a center word from context (CBOW). Words appearing in similar contexts get similar embeddings. The trained embedding matrix captures semantic relationships as geometric vectors.

3. **Q: Why do LSTMs solve the vanishing gradient problem that RNNs have?**
   A: The LSTM cell state flows through time with only additive updates (controlled by gates). The gradient path through the cell state is largely unobstructed — the forget gate allows gradients to flow over hundreds of time steps. Compare: RNN multiplies Whh at every step; LSTM uses addition via cell state.

4. **Q: What is perplexity and what does it measure?**
   A: Perplexity = 2^(cross-entropy). It measures how "surprised" the model is by the test text — lower perplexity means better predictions. A perplexity of 100 means the model is as uncertain as choosing uniformly from 100 words at each step.

5. **Q: What is the Bahdanau attention mechanism?**
   A: It allows the decoder to look at all encoder hidden states (not just the last one) and compute a weighted sum based on relevance scores. The relevance is computed by a small neural network. This solves the bottleneck problem in seq2seq where all information had to pass through a single vector.

6. **Q: What is the difference between stemming and lemmatisation?**
   A: Stemming cuts words to their root using rules (e.g., "running" → "run", "better" → "bett"). Lemmatisation uses vocabulary and morphological analysis to return the correct base form (e.g., "better" → "good"). Lemmatisation is slower but more accurate.

7. **Q: What is a bigram language model?**
   A: A bigram model estimates P(word | previous word) = count(w_{i-1}, w_i) / count(w_{i-1}). The Markov assumption: next word depends only on the immediately previous word. Trigram: depends on previous two words. N-gram models fail at long-range dependencies.

**Intermediate**

8. **Q: Why do word embeddings capture analogical relationships like king - man + woman = queen?**
   A: Words in similar contexts get similar embeddings. "Man" and "king" appear in similar royal contexts; "woman" and "queen" similarly. The vector difference captures the "gender direction". Adding this direction to "king" gives a vector close to "queen" in the embedding space.

9. **Q: What is negative sampling in Word2Vec and why is it necessary?**
   A: The full softmax over the entire vocabulary (50k+ words) is computationally expensive. Negative sampling approximates it: instead of normalising over all words, sample k "negative" (non-context) words and train to distinguish them from the true context word. Reduces O(V) to O(k) per training step.

10. **Q: Explain the LSTM gates: what does each gate control?**
    A: Input gate i_t: controls how much new information to add to cell state. Forget gate f_t: controls how much of the old cell state to retain. Output gate o_t: controls what part of the cell state to expose as the hidden state. Cell state c_t: the "long-term memory" modified by i_t and f_t.

11. **Q: What is the difference between a language model and a text classifier?**
    A: A language model assigns probability P(word | context) to every possible next word — it models the distribution of text. A text classifier takes text as input and assigns a label (sentiment, topic). Language models can be used for generation; classifiers produce categorical outputs.

12. **Q: What is the bottleneck problem in seq2seq models?**
    A: The entire source sentence must be compressed into a single fixed-size context vector (the encoder's final hidden state). For long sentences, information is lost. Attention mechanisms solve this by allowing the decoder to directly access all encoder states.

13. **Q: What are subword tokenisation methods and why are they preferred?**
    A: BPE (Byte-Pair Encoding), WordPiece, and SentencePiece split text into subword units. Benefits: (1) handles OOV words by splitting into known subwords; (2) common words get single tokens (efficiency); (3) rare words split into meaningful parts (morphology). GPT-4 uses ~100k BPE tokens.

**Advanced**

14. **Q: Derive the TF-IDF score for a word w in document d.**
    A: TF(w,d) = count(w in d) / total_words(d) [or log(1 + count)]. IDF(w) = log(N / (1 + df(w))) where N = corpus size, df = documents containing w. TF-IDF(w,d) = TF(w,d) × IDF(w). The +1 in IDF prevents log(0) for words in all documents.

15. **Q: Why does the Transformer architecture make RNNs mostly obsolete for NLP?**
    A: RNNs process tokens sequentially — parallelisation is impossible during training. Transformers process all tokens in parallel using attention — training is much faster. Attention has O(1) path length between any two positions vs. O(n) for RNNs — capturing long-range dependencies is trivial. The only remaining advantage of RNNs is memory efficiency for very long sequences.

16. **Q: What is the difference between context-free and contextual word embeddings?**
    A: Word2Vec/GloVe: each word has one fixed embedding regardless of context. "Bank" has one vector. Contextual embeddings (BERT, ELMo): each word gets a different embedding depending on its context. "Bank" in "river bank" gets a different vector from "bank account". Contextual embeddings capture polysemy.

17. **Q: How does bidirectional LSTM improve over unidirectional LSTM?**
    A: Unidirectional LSTM only sees past context. Bidirectional: one LSTM reads forward, one reads backward; their hidden states are concatenated. Each token's representation incorporates both left and right context. Critical for tasks like NER, POS tagging where future context matters.

18. **Q: What is teacher forcing in seq2seq training and what is exposure bias?**
    A: Teacher forcing: feed the ground-truth previous token as input to the decoder (not the predicted token). Speeds up training. Exposure bias: at inference, the decoder sees its own (possibly wrong) predictions — a distribution mismatch from training. Causes error accumulation. Mitigated by scheduled sampling.

19. **Q: How does FastText handle out-of-vocabulary words?**
    A: FastText represents each word as a bag of character n-grams (e.g., "apple" → "<ap", "app", "ppl", "ple", "le>"). The word embedding is the average of its n-gram embeddings. For OOV words, it computes the average of the available n-gram embeddings. This also helps with morphologically related words.

20. **Q: What is the "distributional hypothesis" in NLP?**
    A: "Words that occur in the same contexts tend to have similar meanings" (Firth, 1957). This is the theoretical foundation for all word embedding methods. Word2Vec, GloVe, and BERT all learn from the statistical patterns of word co-occurrence, operationalising this hypothesis.

---

### 7. Self-Assessment Quiz

- [ ] What is the formula for TF-IDF? Write it out.
- [ ] What is the difference between Skip-gram and CBOW in Word2Vec?
- [ ] Name the three gates in an LSTM and what each controls.
- [ ] What is perplexity and is lower better or worse?
- [ ] Why can't bag-of-words models capture word order?
- [ ] What is cosine similarity and why is it preferred over Euclidean distance for embeddings?
- [ ] What is padding in sequence models and why is it needed?
- [ ] What is the difference between word-level and character-level language models?
- [ ] What does the forget gate in an LSTM learn to do?
- [ ] What is attention weight and what does a high weight mean?
- [ ] Why is GloVe different from Word2Vec in terms of training objective?
- [ ] What is the teacher forcing trick and what problem does it create?
- [ ] Name three advantages of subword tokenisation over word tokenisation.
- [ ] What is the Markov assumption in n-gram models?
- [ ] What is a context window in Word2Vec?
- [ ] Why are pre-trained embeddings better than random initialisation?
- [ ] What is the difference between a language model and a masked language model?
- [ ] What is beam search in text generation?
- [ ] What is the encoder-decoder architecture and what problem does it solve?
- [ ] What is attention in the context of seq2seq models?
- [ ] Why does adding a bidirectional LSTM improve NER tasks?
- [ ] What are the limitations of GloVe embeddings vs. BERT embeddings?
- [ ] How is cosine similarity computed between two word vectors?
- [ ] What is stop word removal and when should you NOT do it?
- [ ] What is a sentence embedding vs. a word embedding?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 4a.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Using just the last hidden state in seq2seq | Default implementation | Use attention pooling or concatenate forward + backward final states |
| Forgetting to freeze pre-trained embeddings initially | Not thinking about training dynamics | Freeze for first few epochs, then unfreeze with lower lr |
| Using TF-IDF for short texts | Good for documents, poor for short texts | For short texts (tweets, titles), sentence embeddings are better |
| Comparing word vectors with Euclidean distance | Seems natural | Embeddings have varying norms; always use cosine similarity |
| Not padding and masking variable-length sequences | Easiest implementation pads | Always create and use attention masks; padding tokens should not contribute to loss |
| Training Word2Vec on small corpora | Convenient | Word2Vec needs millions of sentences; use pre-trained GloVe or FastText for small datasets |
| Not understanding that Word2Vec is a byproduct | People think Word2Vec was designed to give embeddings | The embeddings are a side effect of training a word prediction task |

---

### 9. Readiness Criteria

You are ready for Phase 4b (Transformers) when **all** of the following are true:

- [ ] I implemented TF-IDF from scratch and used it for text search
- [ ] I built an LSTM text classifier with pre-trained embeddings and attention pooling
- [ ] I implemented Bahdanau attention from scratch in PyTorch
- [ ] I completed the comparison study mini project (all 4 approaches)
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can explain what problem attention was invented to solve
- [ ] I can answer at least 16/20 Interview Questions correctly

---

### 10. Revision Summary

```
TEXT REPRESENTATIONS
─────────────────────────────────────────────────────
BoW:        word counts, ignores order, high-dimensional sparse
TF-IDF:     TF × IDF — rewards specificity, punishes commonality
Word2Vec:   dense low-dim, semantic geometry, context-based
BERT/GPT:   contextual, one vector per token per context (polysemy-aware)

SEQUENCE MODELS
─────────────────────────────────────────────────────
RNN:   h_t = tanh(W_h * h_{t-1} + W_x * x_t)   vanishing gradient!
LSTM:  cell state + 3 gates = long-range memory
GRU:   2 gates (simpler LSTM, often comparable performance)
BiLSTM: forward + backward → each token sees full sentence context

ATTENTION (Bahdanau)
─────────────────────────────────────────────────────
score(q, k) = V * tanh(W1*q + W2*k)
weights      = softmax(scores)
context      = weights @ encoder_outputs   (attended summary)
→ Decoder gets to "look at" all encoder states, not just the last one
```

---

### 11. Next Phase Prerequisites

**What Phase 4b (Transformers) requires from Phase 4a:**

| Phase 4a Concept | How Phase 4b Uses It |
|-----------------|----------------------|
| Attention mechanism (Bahdanau) | Transformers extend this to self-attention |
| Softmax over attention scores | Identical in transformer attention |
| Sequence processing intuition | Why positional encoding is needed (no recurrence) |
| Tokenisation and vocabulary | BPE tokenisation used in GPT-2/LLaMA |
| Embedding layer | First layer of every transformer |
| Why RNNs fail on long sequences | Why transformers process all positions in parallel |
| Encoder-decoder architecture | BERT = encoder only; GPT = decoder only; T5 = encoder-decoder |

**The critical dependency**: The attention mechanism in transformers is Bahdanau attention generalised into self-attention (every token attends to every other token). If you understand Phase 4a attention, the transformer's attention formula Q @ K^T / sqrt(d_k) will click immediately rather than feel like a magic formula.

---

*Phase 4a | Part of the [GenAI Engineer Roadmap](./00_README.md)*
