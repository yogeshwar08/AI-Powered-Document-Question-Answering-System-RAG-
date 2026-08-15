# 🚀 DocuMind — AI-Powered Document Question-Answering System

> A production-oriented Retrieval-Augmented Generation (RAG) application that enables users to ask questions across multiple PDF documents and receive context-aware, source-grounded answers using **LangChain, FAISS, Gemini Embeddings, and Google Gemini LLMs**.

---

## 📌 Project Overview

**DocuMind** is an end-to-end AI document intelligence system designed to retrieve relevant information from multiple PDF documents and generate context-aware answers using a Retrieval-Augmented Generation architecture.

Instead of sending a user's question directly to a Large Language Model (LLM), the system first performs semantic retrieval against a document knowledge base.

The retrieved document chunks are then provided to the LLM as contextual information, allowing the system to generate answers grounded in the available documents.

The system also maintains **document-level and page-level source attribution**, making responses more transparent and suitable for technical demonstrations and real-world document intelligence applications.

---

# 🎯 Project Objectives

The system is designed to:

- Process multiple PDF documents
- Extract text from PDF files
- Preprocess document content
- Split documents into meaningful chunks
- Generate semantic embeddings
- Store embeddings in a FAISS vector database
- Perform semantic similarity search
- Retrieve the most relevant document chunks
- Build contextual prompts dynamically
- Generate grounded responses using Google Gemini
- Provide document and page-level source attribution
- Expose the RAG pipeline through a Flask REST API
- Provide an interactive web interface
- Display executable Python examples in a compiler-style interface
- Provide response-time and retrieval metrics
- Support cloud deployment using Gunicorn

---

# 🧠 What is Retrieval-Augmented Generation?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with Large Language Models.

Instead of relying entirely on the knowledge stored inside an LLM, RAG retrieves relevant information from an external knowledge base and supplies that information to the LLM as context.

This helps the application produce responses that are more relevant to the provided documents.

---

## Traditional LLM Architecture

```text
User Question
      │
      ▼
   LLM
      │
      ▼
   Answer
####RAG Architecture
                    DOCUMENT INGESTION
                        │
                        ▼
                 Multiple PDFs
                        │
                        ▼
                PDF Text Extraction
                        │
                        ▼
                 Document Chunking
                        │
                        ▼
                Gemini Embeddings
                        │
                        ▼
                  FAISS Index
                        │
                        │
                        ▼
USER QUESTION ─────► Semantic Retrieval
                        │
                        ▼
                 Relevant Chunks
                        │
                        ▼
               Context Augmentation
                        │
                        ▼
                 Google Gemini
                        │
                        ▼
                Grounded Answer
                        │
                        ▼
             Sources + Page Numbers
####System Architecture 
┌─────────────────────────────────────────────────────────────┐
│                       USER INTERFACE                        │
│                  HTML / CSS / JavaScript                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      FLASK REST API                         │
│                                                             │
│  GET  /                                                     │
│  GET  /health                                               │
│  GET  /ready                                                │
│  GET  /api/stats                                            │
│  GET  /api/info                                             │
│  POST /api/ask                                              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     LANGCHAIN RAG LAYER                     │
│                                                             │
│  Query → Retriever → Context → Prompt → LLM                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FAISS VECTOR DATABASE                    │
│                                                             │
│              Semantic Similarity Retrieval                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 GOOGLE GEMINI EMBEDDINGS                    │
│                                                             │
│                 gemini-embedding-2                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     GOOGLE GEMINI LLM                       │
│                                                             │
│                    Answer Generation                        │
└─────────────────────────────────────────────────────────────┘