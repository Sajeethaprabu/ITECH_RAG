# Cryptography & Network Security RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer academic questions strictly from:

**William Stallings – Cryptography and Network Security: Principles and Practice (4th Edition)**

This system performs semantic retrieval over the textbook and generates structured, context-aware responses using a locally hosted Large Language Model (LLM). The entire pipeline is containerized using Docker.

---

# Project Overview

This project implements a complete Retrieval-Augmented Generation architecture using modern NLP tools.

The system:

- Extracts text from a PDF document  
- Splits text into semantic chunks  
- Generates vector embeddings  
- Stores embeddings in a vector database  
- Retrieves relevant context using semantic similarity  
- Generates structured academic responses grounded strictly in the source material  

The system runs locally using Docker and does not rely on external APIs.

---

# Architecture

The system follows a standard RAG pipeline:

User Question  
→ Streamlit Interface  
→ LangChain Retriever  
→ Qdrant Vector Database  
→ Relevant Context Chunks  
→ Ollama LLM  
→ Structured Academic Response  

---

# Technology Stack

### Frontend
- Streamlit

### LLM Serving
- Ollama

### Vector Database
- Qdrant

### Framework
- LangChain

### Embedding Model
- nomic-embed-text

### LLM Model
- qwen2.5:3b-instruct

### Containerization
- Docker  
- Docker Compose

---

# Project Structure


NLP-RAG/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── README.md
│
└── doc/
└── Place Stallings PDF here

Note: The textbook PDF is not included in this repository due to copyright restrictions.

---

---

# System Workflow

### Document Loading
The PDF is loaded using **PyPDFLoader**.

### Text Chunking
Text is split using **RecursiveCharacterTextSplitter**.

### Embedding Generation
Each chunk is converted into vector embeddings using Ollama.

### Vector Storage
Embeddings are stored in **Qdrant Vector Database** using cosine similarity.

### Semantic Retrieval
Top-K relevant chunks are retrieved based on query similarity.

### Response Generation
The LLM generates structured academic responses using only the retrieved context.

---

# Performance Considerations

The system is optimized for CPU-based environments:

- Uses **qwen2.5:3b-instruct** for balanced speed and quality  
- Context window limited to **4096 tokens**  
- Output token limits applied  
- Retrieval **Top-K tuned for performance**

---

