# 🚀 AI-Powered Document Question-Answering System

> A professional Retrieval-Augmented Generation (RAG) application that allows users to ask questions across multiple PDF documents and receive context-aware, grounded answers using LangChain, FAISS, Hugging Face embeddings, and Google Gemini.

---

## 📌 Project Overview

The **AI-Powered Document Question-Answering System** is an end-to-end RAG application designed to retrieve relevant information from a collection of PDF documents and generate accurate, context-aware responses.

Instead of sending the user's question directly to an LLM, the system first searches the document knowledge base for relevant information.

The retrieved information is then provided as context to the language model, allowing the model to generate an answer grounded in the uploaded documents.

The system supports **multiple PDF documents** and maintains document-level and page-level source information for retrieved content.

---

# 🎯 Project Objective

The primary objective of this project is to build a production-oriented document intelligence system that can:

- Process multiple PDF documents
- Extract and preprocess document content
- Split documents into meaningful chunks
- Convert chunks into semantic embeddings
- Store embeddings in a FAISS vector database
- Perform semantic similarity search
- Retrieve relevant document context
- Augment the user query with retrieved context
- Generate grounded answers using Google Gemini
- Display document and page-level sources
- Provide an interactive web interface
- Display Python examples in an interactive compiler-style interface

---

# 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with Large Language Models.

Instead of relying only on the knowledge stored inside an LLM, RAG retrieves relevant information from an external knowledge base and provides that information to the LLM as context.

### Traditional LLM

```text
User Question
      ↓
     LLM
      ↓
   Answer



### RAG Architecture

```text
User Question
      ↓
Query Embedding
      ↓
Vector Similarity Search
      ↓
FAISS Vector Database
      ↓
Relevant Document Chunks
      ↓
Context Augmentation
      ↓
Google Gemini
      ↓
Grounded Answer