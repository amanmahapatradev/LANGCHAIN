# Text Chunking in RAG: Fixed-Size vs. Recursive Chunking

---

## 1. What is Chunking & Why is it Critical?

In Retrieval-Augmented Generation (RAG) and LLM pipelines, **Chunking** is the process of breaking down large documents (PDFs, articles, books, web pages) into smaller, semantically coherent segments called **chunks**.

```
┌─────────────────┐       ┌──────────────┐       ┌────────────────────────┐
│  Raw Document   │ ────► │ Text Chunking│ ────► │ Chunks for Embeddings  │
│ (e.g., 50-page) │       │  (Splitter)  │       │ [Chunk 1] [Chunk 2]... │
└─────────────────┘       └──────────────┘       └────────────────────────┘
```

### Why Can't We Embed Entire Documents at Once?

1. **Embedding Model Input Limits:**
   Embedding models (such as OpenAI's `text-embedding-3-small` or Hugging Face's `all-MiniLM-L6-v2`) have maximum input token limits (e.g., 512, 2048, or 8192 tokens). Text exceeding this limit is silently truncated.
2. **Loss of Embedding Specificity (The Semantic Dilution Problem):**
   When an entire 50-page document is converted into a single vector, granular facts get averaged out. A vector representing "everything" matches "nothing" with high precision. Small, focused chunks yield distinct, highly relevant embeddings.
3. **LLM Context Window & Cost Optimization:**
   Injecting full documents into the prompt consumes massive token counts, increases latency, increases API costs, and risks the "lost-in-the-middle" phenomenon where LLMs overlook facts buried deep in context.

---

## 2. Core Concepts & Parameters

Every chunking strategy in LangChain revolves around three foundational parameters:

| Parameter | Definition | Typical Values | Impact |
| :--- | :--- | :--- | :--- |
| `chunk_size` | Maximum number of units (characters or tokens) in a single chunk. | 500 – 1500 chars (100 – 350 tokens) | Too small = fragmented context.<br>Too large = diluted embeddings. |
| `chunk_overlap` | Number of units shared between consecutive chunks. | 10% – 20% of `chunk_size` (e.g., 100–200) | Prevents breaking thoughts or sentences across chunk boundaries. |
| `length_function` | Method used to measure chunk length. | `len` (character count) or tiktoken / Hugging Face tokenizer | Character counts are fast; token counts match model limits. |

### Visualizing Chunk Overlap

```text
Raw Text: "LangChain is a framework for developing LLM applications with retrieval."

Chunk 1: [ LangChain is a framework for developing LLM ]
                                  └── Overlap ──┘
Chunk 2:                         [ for developing LLM applications with retrieval. ]
```

---

## 3. Fixed-Size Chunking (`CharacterTextSplitter`)

### 3.1 Definition & Core Concept
**Fixed-Size Chunking** is the simplest and most direct text-splitting technique. It divides a long document into fixed-length segments based on a strict count of characters or tokens, advancing through the document using a sliding window.

It operates without awareness of linguistic grammar, syntax trees, or semantic paragraphs.

---

### 3.2 The Sliding Window Mechanics & Mathematics

Fixed-size chunking moves through the document using a sliding window governed by two parameters:
* $S = \text{chunk\_size}$
* $O = \text{chunk\_overlap}$
* $\text{Stride (Step Size)} = S - O$

```
Character Index:
 0          20         40         60         80        100        120        140        160
 ├──────────┴──────────┴──────────┴──────────┴──────────┼──────────┴──────────┴──────────┤
 [────────────────── Chunk 1 (0 to 100) ────────────────]
                                            [────────────────── Chunk 2 (80 to 180) ───────────────]
                                            ▲──────────▲
                                         Overlap (80 to 100)
                                              (20 chars)
```

#### Mathematical Progression:
* **Chunk 1:** Indices $[0, S]$ $\rightarrow$ $[0, 100]$
* **Chunk 2:** Indices $[S - O, 2S - O]$ $\rightarrow$ $[80, 180]$
* **Chunk 3:** Indices $[2(S - O), 3S - 2O]$ $\rightarrow$ $[160, 260]$
* **Chunk $k$:** Starts at index $(k - 1) \times (S - O)$

---

### 3.3 Delimiter Handling & LangChain Behavior

LangChain implements fixed-size chunking via `CharacterTextSplitter`. It accepts a single `separator` (defaulting to `"\n\n"`):

1. **How `CharacterTextSplitter` Works:**
   - It first splits the raw text into pieces using the specified `separator` (e.g. `"\n"`).
   - It then iterates through these pieces and concatenates them until adding the next piece would exceed `chunk_size`.
   - When a chunk boundary is reached, it retains `chunk_overlap` characters from the end of the previous chunk and starts a new one.

2. **The LangChain Gotcha / Warning:**
   - If any individual piece between two separators is larger than `chunk_size`, `CharacterTextSplitter` **does not split that piece further**.
   - Instead, it emits a warning:
     ```text
     UserWarning: Created a chunk of size 145, which is larger than the specified 100
     ```
   - If you want strict character-by-character fixed slicing regardless of delimiters, you must set `separator=""`.

---

### 3.4 LangChain Code Implementation

```python
from langchain_text_splitters import CharacterTextSplitter

sample_text = """Retrieval-Augmented Generation (RAG) is an AI framework.
It fetches relevant facts from an external database before generating an answer.
This completely prevents hallucinations and provides accurate citations for users.
Embedding models convert these chunks into dense vector representations for similarity search."""

# 1. Fixed-size chunking with a delimiter
splitter = CharacterTextSplitter(
    separator="\n",        # Splits only on single newline characters
    chunk_size=120,        # Maximum chunk character target
    chunk_overlap=25,      # Shared overlap characters between chunks
    length_function=len,   # Measure by Python character length
    is_separator_regex=False
)

chunks = splitter.split_text(sample_text)

print(f"Total Chunks Created: {len(chunks)}\n" + "="*50)
for idx, chunk in enumerate(chunks):
    print(f"\n[Chunk {idx + 1}] ({len(chunk)} characters):")
    print(f'"{chunk}"')
```

---

### 3.5 Detailed Evaluation: Advantages vs. Disadvantages

#### Advantages:
* **Deterministic & Simple:** Extremely easy to understand, debug, and model mathematically.
* **Minimal Computational Overhead:** Pure string slicing $O(N)$ with zero parsing or recursive tree evaluations.
* **Uniform Memory / Token Allocation:** Chunks are roughly identical in size, making batch embedding cost and context window packing predictable.

#### Disadvantages:
* **Severe Semantic Fragmentation:**
  - Sentences can be cut in half mid-thought (e.g., `"The company's total revenue was $"` in Chunk 1 and `"5.4 billion for Q3."` in Chunk 2).
  - An embedding for `"5.4 billion for Q3"` without the subject has lost its semantic entity, making vector search match poorly.
* **Rigid Delimiter Dependency:** Relying on a single delimiter fails when text has mixed formatting (paragraphs, bullet points, code blocks).
* **Degraded RAG Retrieval Precision:** The vector database stores noisy, incomplete vectors, leading to lower cosine similarity scores when the user asks natural questions.

---

### 3.6 When Should You Use Fixed-Size Chunking?
* **Fixed-format logs:** Server logs, CSV-style text lines, or fixed-width database extracts where lines have uniform length.
* **Raw byte/character streams:** When parsing binary or raw serialized buffers without natural language grammar.
* **Quick prototyping & baselines:** Measuring retrieval speed or testing embedding pipelines without needing fine-tuned chunking.

---

## 4. Recursive Chunking (`RecursiveCharacterTextSplitter`)

### Definition & Mechanism
**Recursive Chunking** is the **gold standard and recommended default** for natural language documents.

Instead of relying on a single delimiter, it uses a **hierarchical list of separators** to split text progressively, preserving paragraphs, sentences, and words intact for as long as possible.

### Default Separator Hierarchy in LangChain:
```python
separators = ["\n\n", "\n", " ", ""]
```

### How the Recursive Algorithm Works Step-by-Step:

```text
Step 1: Try splitting by "\n\n" (Double Newline / Paragraphs)
  ├── Do resulting chunks fit within chunk_size?
  │     ├── YES ──► Keep them as individual chunks.
  │     └── NO  ──► Send oversized chunks to Step 2.
  ▼
Step 2: Try splitting the oversized chunk by "\n" (Single Newline / Lines)
  ├── Do resulting pieces fit within chunk_size?
  │     ├── YES ──► Keep them as chunks.
  │     └── NO  ──► Send oversized pieces to Step 3.
  ▼
Step 3: Try splitting by " " (Spaces / Words)
  ├── Do resulting pieces fit within chunk_size?
  │     ├── YES ──► Keep them as chunks.
  │     └── NO  ──► Send oversized words to Step 4.
  ▼
Step 4: Split by "" (Individual Characters)
  └── Ultimate fallback: split raw characters to enforce chunk_size.
  ▼
Final Step: Merge smaller adjacent pieces up to chunk_size without
            exceeding it, applying chunk_overlap at boundaries.
```

### LangChain Implementation

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

sample_document = """# Document Loaders in LangChain

In modern Retrieval-Augmented Generation (RAG) and downstream AI workflows, raw data resides across fragmented repositories—unstructured PDFs, customer service tickets, cloud databases, and intranet wikis.

Document Loaders form the foundational Extract phase of the LLM ETL pipeline. Their sole architectural mandate is to connect to external data sources, parse raw file bytes into normalized textual streams, and extract associated provenance metadata.

Every document loader in LangChain normalizes incoming content into the core langchain_core.documents.Document class. Downstream transformation modules receive a uniform data shape composed of page_content and metadata."""

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,             # Target chunk length
    chunk_overlap=40,           # Overlap length
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
    is_separator_regex=False
)

chunks = recursive_splitter.split_text(sample_document)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ({len(chunk)} chars) ---\n{chunk}\n")
```

### Trade-offs of Recursive Chunking

#### Advantages:
* **Maximum Semantic Preservation:** Keeps paragraphs and sentences intact wherever possible.
* **No Broken Words:** Words are never sliced mid-syllable because character splitting (`""`) is only the absolute last resort.
* **Higher Embedding Quality:** Coherent sentences translate to dense, accurate vector representations in the vector database.
* **Adaptable:** Works across diverse formats (Markdown, HTML, Code, Transcripts, PDFs).

#### Disadvantages:
* **Variable Chunk Sizes:** Chunks are not strictly identical in size (some may be 180 chars, others 195 chars) because it respects natural paragraph and sentence boundaries.
* **Slightly Higher Compute Overhead:** Requires recursion and boundary evaluations (negligible for most real-world workloads).

---

## 5. Side-by-Side Comparison: Fixed-Size vs. Recursive

| Feature | Fixed-Size Chunking (`CharacterTextSplitter`) | Recursive Chunking (`RecursiveCharacterTextSplitter`) |
| :--- | :--- | :--- |
| **Splitting Strategy** | Single delimiter or fixed length offset | Hierarchical fallback list (`["\n\n", "\n", " ", ""]`) |
| **Semantic Integrity** | Low (frequently breaks sentences/thoughts) | High (prioritizes paragraphs $\rightarrow$ sentences $\rightarrow$ words) |
| **Word Boundary Safety** | Risky (can chop words unless delimiter matches) | Guaranteed (only splits characters if a word > `chunk_size`) |
| **Chunk Size Uniformity**| Highly uniform | Slightly variable (respects natural semantic boundaries) |
| **LangChain Class** | `CharacterTextSplitter` | `RecursiveCharacterTextSplitter` |
| **Performance Overhead** | Minimal | Low (very fast in Python/C extensions) |
| **Best Used For** | Tabular logs, fixed-record data, simple quick tests | General prose, PDFs, articles, technical manuals, Markdown, Code |
| **Recommendation** | Specialized use-cases only | **Recommended default for all RAG pipelines** |

---

## 6. Language-Aware Recursive Chunking (Bonus Feature)

`RecursiveCharacterTextSplitter` supports specialized syntax-aware separators for programming and markup languages using `.from_language()`.

This ensures that functions, classes, and code blocks remain intact rather than getting severed mid-statement.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Python-aware recursive splitting
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=150,
    chunk_overlap=20
)

python_code = """
class CustomAPILoader(BaseLoader):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def lazy_load(self) -> Iterator[Document]:
        for item in fetch_remote_records(self.endpoint):
            yield Document(page_content=item["body"], metadata={"source": self.endpoint})
"""

code_chunks = python_splitter.split_text(python_code)
for idx, c in enumerate(code_chunks):
    print(f"--- Code Chunk {idx + 1} ---\n{c}\n")
```

### Supported Languages Include:
* `Language.PYTHON` (`def`, `class`, indents)
* `Language.MARKDOWN` (`#`, `##`, ````code blocks````)
* `Language.JS`, `Language.TS`, `Language.HTML`, `Language.CPP`, `Language.JAVA`

---

## 7. Practical Recommendations for RAG Pipelines

1. **Start with Recursive Chunking:**
   Always default to `RecursiveCharacterTextSplitter`. Switch to fixed-size chunking only when working with fixed-column or structured log files.
2. **Chunk Size Selection Guide:**
   - **300 – 600 characters (approx. 75 – 150 tokens):** Best for fine-grained, fact-lookup queries (e.g., FAQs, phone numbers, exact dates).
   - **800 – 1500 characters (approx. 200 – 375 tokens):** Best for general RAG, comprehension, synthesis, and summarization.
   - **> 2000 characters:** Often introduces noise into embeddings unless using long-context specialized embedding models.
3. **Always Include Overlap:**
   Use a `chunk_overlap` between **10% and 20%** of your `chunk_size`. This ensures that cross-sentence concepts are not isolated from each other during retrieval.
4. **Use Token Counting for Precise LLM Budgeting:**
   If you have tight token constraints, pass `length_function=tiktoken_len` or use `TokenTextSplitter` to avoid token conversion mismatches.

---

## 8. Traditional Chunking vs. Late Chunking

As RAG architectures evolved beyond basic text splitters, a major paradigm shift emerged regarding **when** chunking should occur relative to embedding generation:
* **Traditional Chunking:** *Split-First, Embed-Later*
* **Late Chunking:** *Embed-First, Split-Later* (Pioneered by Jina AI in 2024)

```text
========================================================================================
[TRADITIONAL CHUNKING: Split-First, Embed-Later]
Raw Document ──► Text Splitter ──► [Chunk 1] [Chunk 2] [Chunk 3]
                                         │         │         │
                                      Embed()   Embed()   Embed() (In Isolation)
                                         ▼         ▼         ▼
                                    [Vector 1] [Vector 2] [Vector 3]
========================================================================================
[LATE CHUNKING: Embed-First, Split-Later]
Raw Document (Full 8k context)
       │
       ▼
Full Document Transformer Layer (Bidirectional Attention across ALL tokens)
       │
       ▼ Token-level contextual representations H ∈ R^(N x d)
[Token 1] [Token 2] ... [Token 500] ... [Token 1000] ... [Token N]
 └─────────────┬─────────────┘ └─────────────┬─────────────┘
          Span [0:500]                  Span [500:1000]
          Mean-Pooling                   Mean-Pooling
               ▼                              ▼
          [Vector 1]                     [Vector 2]
    (Contains global context)      (Contains global context)
========================================================================================
```

---

### 8.1 Traditional Chunking (Split-Then-Embed)

#### How It Works:
1. The raw text document is sliced into discrete text segments (e.g. 500 characters each) using character, recursive, or sentence splitters.
2. Each text segment is sent to the embedding model independently.
3. The embedding model generates a single vector for each isolated chunk.

#### The Fundamental Flaws of Traditional Chunking:

1. **The Anaphora / Pronoun Ambiguity Problem:**
   Consider this document:
   * *Chunk 1:* `"Apple Inc. reported record-breaking revenue for the third quarter of 2024."`
   * *Chunk 2:* `"The company attributed this remarkable growth primarily to its expanding services division and strong iPhone sales."`
   
   When Chunk 2 is embedded in isolation:
   - The embedding model has **no idea** what `"The company"` or `"this remarkable growth"` refers to.
   - If a user asks *"Why did Apple grow in 2024?"*, a vector search on Chunk 2 will have poor similarity because the word `"Apple"` is completely missing from Chunk 2's isolated vector!

2. **The Context Cliff:**
   Even with a 10%–20% `chunk_overlap`, self-attention in the transformer is bounded strictly within the boundaries of the isolated chunk. Any global narrative, introductory definitions, or document metadata established earlier in the document are lost to later chunks.

---

### 8.2 Late Chunking (Embed-Then-Split)

#### Core Innovation:
Late Chunking leverages modern **long-context embedding models** (such as `jina-embeddings-v2` or `jina-embeddings-v3`, which support 8,192 tokens) to process the **entire document at once** through the transformer encoder before generating chunk vectors.

#### Step-by-Step Mechanics:

1. **Full-Sequence Self-Attention:**
   The entire document (up to 8,192 tokens) is fed into the transformer encoder. The multi-head bidirectional self-attention layers compute token interactions across the **entire text**:
   $$\mathbf{H} = \text{Transformer}(\text{Token}_1, \text{Token}_2, \dots, \text{Token}_N)$$
   Where $\mathbf{H} \in \mathbb{R}^{N \times d}$ represents the hidden representations for all $N$ tokens in dimension $d$.

2. **Boundary Span Identification:**
   Chunk boundaries (start and end token indices $[s_k, e_k]$) are defined on the original document using any chunking rule (e.g., recursive splitting, sentence boundaries, or paragraph marks).

3. **Span-Level Pooling:**
   Instead of pooling the entire document into a single vector, Late Chunking applies **mean pooling** strictly across the token embeddings corresponding to each chunk's span $[s_k, e_k]$:
   $$\vec{v}_k = \frac{1}{e_k - s_k} \sum_{i = s_k}^{e_k} \mathbf{H}_i$$

#### Why Late Chunking Solves the Context Problem:
Because token $\mathbf{H}_i$ inside Chunk 2 passed through self-attention alongside Chunk 1:
- The token `"The company"` in Chunk 2 **already attended to** `"Apple Inc."` in Chunk 1.
- The resulting vector $\vec{v}_2$ encodes the specific details of Chunk 2, but **retains the semantic identity** of Apple Inc. from the global document context!

---

### 8.3 Detailed Comparison: Traditional vs. Late Chunking

| Comparison Dimension | Traditional Chunking | Late Chunking |
| :--- | :--- | :--- |
| **Execution Sequence** | Split text $\rightarrow$ Embed each chunk | Embed full text $\rightarrow$ Split & Pool token spans |
| **Attention Scope** | Local (only tokens within the chunk attend to each other) | Global (tokens attend across the entire document) |
| **Pronoun / Entity Resolution** | Poor (pronouns like "he", "they", "it" lose their referents) | Excellent (pronouns retain attention links to earlier named entities) |
| **Model Requirement** | Any embedding model (even 512-token models) | Requires long-context models (e.g., $\ge 8192$ tokens like Jina, BGE-M3) |
| **Computational Efficiency** | Higher total overhead if batching many small chunks | Highly efficient (single forward pass through transformer for entire doc) |
| **Storage / Vector DB Impact** | Standard vectors ($d$-dimensional) | Identical standard vectors ($d$-dimensional, zero vector DB changes) |
| **Retrieval Accuracy (NDCG@10)** | Baseline | Up to **10% – 25% higher retrieval accuracy** on complex queries |

---

### 8.4 Code Example: Implementing Late Chunking

Late chunking is natively supported by Jina AI's embedding models and can be implemented with Python:

```python
from transformers import AutoModel
import torch

# 1. Load a long-context embedding model supporting late chunking
model_name = "jinaai/jina-embeddings-v2-base-en"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# 2. Sample document where context is established early
document = (
    "Albert Einstein was a German-born theoretical physicist. "
    "He developed the theory of relativity. "
    "His work is also known for its influence on the philosophy of science. "
    "In 1921, he received the Nobel Prize in Physics for his discovery of the law of the photoelectric effect."
)

# 3. Define chunk spans (using character or sentence boundaries)
chunks_text = [
    "Albert Einstein was a German-born theoretical physicist.",
    "He developed the theory of relativity.",
    "His work is also known for its influence on the philosophy of science.",
    "In 1921, he received the Nobel Prize in Physics for his discovery of the law of the photoelectric effect."
]

# 4. Late Chunking execution:
# The model computes token embeddings for the FULL document,
# then mean-pools representations over the boundary of each chunk.
chunk_embeddings = model.encode(
    [document], 
    return_chunk_embeddings=True, 
    chunk_annotations=[chunks_text]
)

print(f"Generated {len(chunk_embeddings[0])} contextualized chunk vectors!")
for i, emb in enumerate(chunk_embeddings[0]):
    print(f"Chunk {i + 1} Vector shape: {emb.shape}")
```

---

### 8.5 When to Choose Which Strategy

#### Choose Traditional Chunking when:
1. You are using proprietary API embedding models with strict token input limits or without token-level hidden state extraction (e.g., standard closed-source embedding endpoints).
2. Documents exceed the maximum context window of even long-context models ($> 32\text{k}$ tokens).
3. Fast, low-memory edge devices where loading large transformer attention matrices in RAM is impossible.

#### Choose Late Chunking when:
1. You are running open-source long-context embedding models (`jina-embeddings-v2/v3`, `BGE-M3`, `Nomic-Embed`).
2. Your documents are narrative, legal contracts, scientific papers, or multi-topic reports where pronouns and cross-paragraph references are frequent.
3. You want maximum retrieval precision without increasing downstream LLM prompt costs.

---

## 9. The Complete Chunking Framework: Architecture, Taxonomy & Decision Tree

Modern RAG architectures do not treat chunking as an ad-hoc string operation. Instead, chunking is formalized as an **extensible transformation framework** within the LLM data engineering stack.

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           THE CHUNKING FRAMEWORK                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   [Document Ingestion]                                                           │
│           │                                                                      │
│           ▼                                                                      │
│   [Chunking Strategy Selector]                                                   │
│   ├── Level 1: Fixed-Size (Character / Token)                                    │
│   ├── Level 2: Recursive Content-Aware (Paragraphs / Sentences / Words)          │
│   ├── Level 3: Document-Specific / Structural (Markdown / HTML / JSON / Code)    │
│   ├── Level 4: Semantic Embedding-Based (Distance / Similarity Thresholds)       │
│   └── Level 5: Late Chunking / Agentic (Transformer-level & LLM-driven)          │
│           │                                                                      │
│           ▼                                                                      │
│   [LangChain TextSplitter Abstraction]                                           │
│   (split_documents, metadata preservation, length validation)                    │
│           │                                                                      │
│           ▼                                                                      │
│   [Standardized Output: List[Document(page_content, metadata)]]                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### 9.1 The 5 Levels of the Chunking Taxonomy

The chunking framework categorizes all chunking methodologies across a 5-level maturity model based on **semantic awareness** and **computational complexity**:

```
High Semantic Awareness  ▲
                          │                                  [Level 5: Late / Agentic Chunking]
                          │                           [Level 4: Semantic Similarity Chunking]
                          │                    [Level 3: Document-Specific / Structural]
                          │             [Level 2: Recursive Content-Aware]
                          │      [Level 1: Fixed-Size]
                          └────────────────────────────────────────────────────────►
Low Compute                                                               High Compute
```

| Level | Strategy | Core Mechanism | Best Used For | LangChain Class / Tool |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Fixed-Size** | Slices by raw character or token count with sliding overlap. | Tabular rows, fixed logs, raw binary text. | `CharacterTextSplitter`, `TokenTextSplitter` |
| **2** | **Recursive** | Hierarchical fallback separators (`\n\n` $\rightarrow$ `\n` $\rightarrow$ space). | Articles, essays, documentation, books. | `RecursiveCharacterTextSplitter` |
| **3** | **Structural** | Splits based on document syntax (headers, tags, AST trees). | Markdown notes, HTML articles, Python/JS code. | `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter` |
| **4** | **Semantic** | Calculates cosine distance between sentence embeddings; splits on semantic shifts. | Heterogeneous text, interviews, speech transcripts. | `SemanticChunker` (`langchain_experimental`) |
| **5** | **Late / Agentic** | Computes full-context attention before span pooling, or uses LLM reasoning to segment. | Complex legal briefs, multi-hop RAG, scientific papers. | `Jina AI Late Chunking`, Agentic proposition splitters |

---

### 9.2 The LangChain `TextSplitter` Class Architecture

In LangChain, all chunking mechanisms inherit from a unified Object-Oriented Programming (OOP) hierarchy rooted in `BaseDocumentTransformer`.

```text
                BaseDocumentTransformer (Abstract Interface)
                           │
                           ▼
                      TextSplitter (Core Implementation Base)
             ┌─────────────┼───────────────────────────┐
             │                                         │
    CharacterTextSplitter                    RecursiveCharacterTextSplitter
             │                                         │
    TokenTextSplitter                ┌─────────────────┴─────────────────┐
                                     │                                   │
                         Language.PYTHON / Language.JS        MarkdownTextSplitter
```

#### Core Interface Contract:

All splitters in the framework implement three universal methods:

1. **`split_text(text: str) -> List[str]`**:
   Accepts a raw Python string and returns a list of raw string chunks.
2. **`create_documents(texts: List[str], metadatas: Optional[List[dict]]) -> List[Document]`**:
   Converts a list of raw strings and associated metadata dictionaries into standard `Document` objects.
3. **`split_documents(documents: List[Document]) -> List[Document]`**:
   Accepts a list of `Document` objects (e.g. from `PyPDFLoader`), applies chunking to each document's `page_content`, and **automatically clones and preserves the parent document's metadata** (e.g. `source`, `page`, timestamps) onto every resulting chunk!

#### Code Architecture Example:

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Incoming document with rich metadata from PyPDFLoader
source_doc = Document(
    page_content="LangChain provides modular abstractions. Chunking ensures vector stores retrieve precise context.",
    metadata={"source": "report.pdf", "page": 1, "author": "Research Team"}
)

# 2. Instantiate framework splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)

# 3. Transform documents (Metadata is preserved across all chunks!)
chunked_docs = splitter.split_documents([source_doc])

for idx, doc in enumerate(chunked_docs):
    print(f"Chunk {idx + 1}: '{doc.page_content}'")
    print(f"Preserved Metadata: {doc.metadata}\n")
```

---

### 9.3 The Chunking Decision Framework (How to Choose)

Use this architectural decision tree to determine the optimal chunking strategy for your specific RAG pipeline:

```text
                            START: What is your primary data format?
                                               │
             ┌─────────────────────────────────┼────────────────────────────────┐
             ▼                                 ▼                                ▼
     Structured Markup                  Natural Language                     Code / AST
     (Markdown, HTML)                    (PDFs, TXT, Docs)              (Python, JS, C++)
             │                                 │                                │
             ▼                                 ▼                                ▼
   Do you have explicit                Is semantic precision          Use Syntax-Aware Splitter:
   headers / section tags?             mission critical?              `RecursiveCharacterTextSplitter`
       ├── YES ──► Level 3:                   │                       `.from_language(Language.PYTHON)`
       │           `MarkdownHeaderTextSplitter`│
       └── NO  ──► Level 2:                   ├── HIGH ──► Do you use local long-context embeddings?
                   `RecursiveTextSplitter`    │             ├── YES ──► Level 5: `Late Chunking` (Jina/BGE)
                                              │             └── NO  ──► Level 4: `SemanticChunker`
                                              │
                                              └── STANDARD (Default) ──► Level 2: `RecursiveCharacterTextSplitter`
                                                                         (chunk_size: 800-1200, overlap: 10-15%)
```

---

### 9.4 The 4 Golden Rules of the Chunking Framework

1. **Default to Recursive Chunking:** When in doubt, `RecursiveCharacterTextSplitter` provides the best balance of semantic fidelity, performance, and reliability.
2. **Never Chunk Without Overlap:** Always maintain a **10%–20% overlap** to eliminate context cliffs across chunk boundaries.
3. **Preserve Document Metadata:** Always use `.split_documents()` rather than `.split_text()` so that filenames, page indices, and custom tags carry over to your vector store.
4. **Align Chunk Size with Embedding and Retrieval Task:**
   - Single-fact lookups (phone numbers, prices, dates) $\rightarrow$ Small chunks (200–500 chars).
   - Paragraph comprehension and reasoning $\rightarrow$ Medium chunks (800–1500 chars).
   - High-level summarization $\rightarrow$ Large chunks or Parent-Document / Hierarchical retrievers.

