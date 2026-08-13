# AI-Powered Document Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents and receive context-aware answers using semantic search, FAISS vector retrieval, and a locally hosted Llama 3.2 3B model through Ollama.

---

## Project Overview

This project implements an end-to-end RAG pipeline for document question answering.

Instead of sending the entire document directly to an LLM, the system:

1. Loads PDF documents.
2. Splits the document into smaller chunks.
3. Generates semantic embeddings using Sentence Transformers.
4. Stores the embeddings in a FAISS vector database.
5. Retrieves the most relevant document chunks for a user query.
6. Sends the retrieved context to a local Llama 3.2 3B model.
7. Generates an answer based on the retrieved information.
8. Displays the answer and source pages through a Streamlit interface.

The system runs locally and does not require a paid LLM API.

---

## Architecture

```text
                    PDF DOCUMENT
                         |
                         v
                  PDF Document Loader
                         |
                         v
                Recursive Text Splitter
                         |
                         v
              Sentence Transformer
                Embedding Model
                         |
                         v
                    FAISS Index
                         |
                  Vector Database
                         |
                         v
                  User Question
                         |
                         v
              Semantic Similarity Search
                         |
                         v
                 Relevant PDF Chunks
                         |
                         v
                  RAG Prompt Builder
                         |
                         v
                 Ollama Local Server
                         |
                         v
                  Llama 3.2 3B
                         |
                         v
                  Generated Answer
                         |
                         v
                  Streamlit Interface