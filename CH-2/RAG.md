RAG - REtrieval augmented Generation    
Introduction of RAG:
1. It is a technique that improves the quality of responses from Large Language Models (LLMs) by providing them with relevant information from an external knowledge base.
2. It is a way to reduce hallucinations in LLMs by providing them with relevant information from an external knowledge base.

The RAG Architechure ->

User Query -> Retriever -> Relevant Docs -> LLM with prompt -> Response

--------------------------------------------------------------------------------------------------------------------------------------
                                                        RAG ARCHITECHURE
--------------------------------------------------------------------------------------------------------------------------------------

                    User Question         ->         Retriever        ->     Relevant Docs          ->     LLM + Prompt     ->      Response
                      |                                   |                        |                          |                         |
                      V                                   V                        V                          V                         V
                  Text                                 Vector DB                Retreived Chunks          Context + Question        Augmented Response

--------------------------------------------------------------------------------------------------------------------------------------        

**The Complete RAG Pipeline**

basic RAG Chain ->
Chain Structure ------>

Context Query (Parallel inputes) | prompt (Template) | LLM (model) | Parser (Output)

**Parllel Input Processing**

Context <------- retriver ------> Query 
Question <------- RunnablePassthrough -------> Prompt 

**Prompt Template**

Answer based only on:

{context} 
Question: {question}

**Prompt patterns**:

1. Answer based only on:

{context} 
Question: {question}

2. **Instruction tuning prompts**:

Answer the following question based on the provided context. If the answer is not in the context, say "I don't know".

{context}

Question: {question}

RAG With Sources

**Retriver Output**
[Page change here (source: doc.pdf)
More content (source: guide.txt)
FAQ content (source: faq.md)] -------> format_docs_with_sources (Add source tags to each chunk) --------->

**Formatting Content**
[source: doc.pdf]
Page content......

[source: guide.txt]
More content......

[source: faq.md]
FAQ content......

1ST install the .venv and other models ----------> uv add langchain langchain-core langgraph langchain-openai langchain-anthropic python-dotenv langchain-google-genai

2ND install .env --------> touch .env (In Git-Bash) or New-Item .env (In PowerShell)

**Document Loder Overview**
