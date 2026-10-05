# Retrieval-Augmented Generation (RAG): Complete Study Notes

---

## 1. What is RAG? (Definition & Core Concept)

**Retrieval-Augmented Generation (RAG)** is an AI framework that enhances the accuracy and reliability of Large Language Models (LLMs) by fetching relevant facts from an external, authoritative knowledge base before generating a response.

### Why is RAG Needed?
Standard LLMs suffer from three critical limitations:
1. **Knowledge Cutoff:** Models only know facts up to the date their training concluded.
2. **Hallucination:** When an LLM lacks information, it often invents plausible-sounding but factually incorrect statements.
3. **Lack of Private / Enterprise Data:** LLMs do not have access to proprietary databases, internal company documents, or private user files.

> **RAG Formula:**
> $$\text{Output} = \text{LLM}(\text{User Query} + \text{Retrieved External Context})$$

---

## 2. Comparison: Prompt Engineering vs. RAG vs. Fine-Tuning

| Metric | Prompt Engineering | RAG | Fine-Tuning |
| :--- | :--- | :--- | :--- |
| **Primary Purpose** | Guide reasoning & format | Supply up-to-date / private facts | Adapt style, tone, or syntax |
| **New Knowledge Access** | Minimal (fits in context window) | High (dynamic external databases) | Low (static; expensive to update) |
| **Setup Cost** | Very Low | Moderate | High (requires GPUs & datasets) |
| **Hallucination Risk** | Moderate to High | Low (grounded with source citations)| Moderate |
| **Maintenance** | Easy | Easy (update database files) | Hard (requires retraining) |

---

## 3. Core Components of a RAG System

A complete RAG architecture consists of six foundational components:

1. **Document Loaders:**
   - Ingest data from various file types: PDF, Markdown, Word, CSV, HTML, SQL, or web pages.
   - *LangChain modules:* `PyPDFLoader`, `DirectoryLoader`, `TextLoader`, `WebBaseLoader`.

2. **Text Splitters (Chunking):**
   - LLMs and embedding models have input token limits. Text splitters break long documents into manageable chunks while maintaining semantic coherence.
   - *Key parameters:*
     - `chunk_size`: Maximum tokens or characters per chunk (e.g., 500–1000).
     - `chunk_overlap`: Overlap between adjacent chunks (e.g., 100–200) to avoid losing context at chunk boundaries.
   - *LangChain module:* `RecursiveCharacterTextSplitter`.

3. **Embedding Models:**
   - Convert text chunks into high-dimensional vector arrays that represent mathematical semantics.
   - *Popular Models:* OpenAI `text-embedding-3-small`, HuggingFace `all-MiniLM-L6-v2`, BGE-Large.

4. **Vector Database (Vector Store):**
   - Specialized database designed to store high-dimensional vectors and perform fast nearest-neighbor search.
   - *Local/Embedded:* `ChromaDB`, `FAISS`.
   - *Cloud/Managed:* `Pinecone`, `Qdrant`, `Weaviate`, `Milvus`.

5. **Retriever:**
   - An interface that accepts a user query string, converts it to a vector, and queries the vector store for the top-$k$ most similar chunks.
   - *Search algorithms:* Cosine Similarity, Dot Product, Euclidean Distance, and MMR (Maximal Marginal Relevance to reduce redundancy).

6. **Generator (LLM):**
   - The foundation model (e.g., GPT-4o, Claude 3.5, Llama 3) that receives both the user prompt and the retrieved context chunks to generate a grounded, synthesized response.

---

## 4. Step-by-Step Working of RAG (The Two Pipelines)

The RAG workflow operates across two distinct phases:

```text
[PHASE 1: INGESTION PIPELINE (Offline / Pre-computation)]
Raw Documents (.pdf/.txt) 
      │
      ▼
Text Splitter (Chunking) 
      │
      ▼
Embedding Model (Vectorization)
      │
      ▼
Vector Database (Stored with Metadata & Index)

-----------------------------------------------------------------------

[PHASE 2: RETRIEVAL & GENERATION PIPELINE (Online / Real-time)]
User Query
      │
      ├───► Query Embedding ───► Vector DB Search (Top-k Chunks)
      │                                       │
      ▼                                       ▼
System Prompt Template  ◄─────────── Relevant Context Chunks
      │
      ▼
Augmented Prompt ("Answer using ONLY this context: ...")
      │
      ▼
LLM Generation ───► Final Response with Citations
```

### Detailed Execution Steps:
1. **Ingestion:**
   - Raw documents are collected and parsed.
   - Text is split into overlapping chunks.
   - Each chunk passes through the embedding model to produce an embedding vector.
   - Vectors and their original text metadata (source, page number) are stored in the vector database.
2. **Retrieval:**
   - When a user asks a question, the query is converted into an embedding using the exact same embedding model.
   - The vector store calculates mathematical similarity (e.g., Cosine Similarity) between the query vector and chunk vectors.
   - The top $k$ most relevant chunks are extracted.
3. **Augmentation:**
   - A prompt template is constructed combining:
     - System instructions (e.g., *"You are a helpful assistant. Use only the provided context."*)
     - The retrieved chunks
     - The user's original query
4. **Generation:**
   - The combined prompt is sent to the LLM.
   - The LLM synthesizes an accurate answer based solely on the injected context, minimizing hallucination.

---

## 5. Evolutionary Levels of RAG

1. **Naive RAG:**
   - Simple workflow: Split $\rightarrow$ Embed $\rightarrow$ Top-$k$ Search $\rightarrow$ Generate.
   - *Drawback:* Low precision, missing context, noise in retrieved chunks.

2. **Advanced RAG:**
   - **Pre-Retrieval:** Query expansion, query rewriting, HyDE (Hypothetical Document Embeddings).
   - **Post-Retrieval:** Reranking (using cross-encoders like Cohere Rerank or BGE Reranker) to score and filter chunks before sending to the LLM; Context Compression to discard irrelevant sentences.

3. **Modular / Agentic RAG:**
   - Employs autonomous AI agents with routing mechanisms.
   - Dynamically decides whether retrieval is required, searches multiple vector databases or web tools, and self-reflects (Corrective RAG / Self-RAG) to check if the answer is faithful to the retrieved documents.

---

## 6. Hands-On Development Setup (LangChain + ChromaDB + OpenAI)

### Step 1: Environment & Dependencies Installation
Ensure your virtual environment is active, then install the required packages:

```bash
pip install langchain langchain-openai langchain-community chromadb python-dotenv pypdf tiktoken
```

### Step 2: Environment Variables (`.env`)
Create a `.env` file in your project root:
```env
OPENAI_API_KEY="sk-proj-yourOpenAIKeyHere"
```

### Step 3: Complete End-to-End RAG Python Implementation
Here is a complete, modular, runnable RAG implementation using LangChain Expression Language (LCEL):

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

# 2. Ingestion: Prepare Sample Knowledge Base Document
sample_text = """
The LangChain Framework is an open-source development framework designed 
to simplify the creation of applications using large language models (LLMs).
It provides abstractions for chains, agents, memory, document loaders, 
vector stores, and retrieval-augmented generation (RAG).

Project Antigravity is an agentic coding workflow developed by the DeepMind team 
to assist developers in pair programming, planning, and tool execution.
"""

with open("knowledge_base.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)

# Load document
loader = TextLoader("knowledge_base.txt", encoding="utf-8")
docs = loader.load()

# 3. Chunking: Split documents into manageable segments
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# 4. Embeddings & Vector Store: Ingest chunks into ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_knowledge_base"
)

# 5. Retriever: Query the top 2 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 6. Prompt Template: Ground the LLM with retrieved context
prompt_template = """You are a helpful assistant for question-answering tasks.
Use ONLY the following pieces of retrieved context to answer the question.
If you do not know the answer based on the context, say that you do not know. 
Do not make up facts.

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 7. LLM Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. Construct RAG Chain using LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. Test Query
query = "What is Project Antigravity and who developed it?"
print(f"\nUser Question: {query}\n")

response = rag_chain.invoke(query)
print("RAG Response:\n", response)
```

---

## 7. RAG Evaluation: The RAG Triad

To measure whether your RAG pipeline is working accurately, evaluate three core dimensions (using frameworks such as **Ragas** or **TruLens**):

```text
       [User Query]
          /     \
         /       \
  Context         Answer
 Relevance       Relevance
       /           \
      /             \
[Retrieved Context] ── Faithfulness ──► [Generated Answer]
```

1. **Context Relevance:** Does the retrieved chunk actually contain the information necessary to answer the question?
2. **Faithfulness (Groundedness):** Is the generated answer 100% derived from the retrieved context, without introducing hallucinations?
3. **Answer Relevance:** Does the response directly and concisely resolve the user's inquiry?

---

## 8. Best Practices & Optimization Tips

- **Optimize Chunk Size:**
  - Small chunks ($100-250$ tokens) improve retrieval precision but lose macro context.
  - Large chunks ($800-1500$ tokens) preserve full context but introduce noise to the LLM.
- **Implement Re-ranking:** Add a cross-encoder model (e.g., Cohere Rerank) to reorder the top 20 retrieved candidates and pass only the top 3-5 to the LLM.
- **Always Include Metadata:** Attach source URLs, page numbers, authors, and timestamps to each chunk for citations and filtering.
- **Enforce Low Temperature:** Use `temperature=0.0` or `0.1` for the generator LLM to prevent creative drift and enforce factual adherence.
