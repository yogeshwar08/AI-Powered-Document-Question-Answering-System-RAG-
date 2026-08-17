🚀 DocuMind — AI-Powered Document Question-Answering System

A production-oriented Retrieval-Augmented Generation (RAG) application for asking questions across multiple PDF documents using LangChain, hybrid retrieval (FAISS + BM25), Gemini Embeddings, Google Gemini, and Flask.

📌 Project Overview

DocuMind is an end-to-end document intelligence application that retrieves relevant information from a multi-document PDF knowledge base and uses Google Gemini to generate clear, context-aware answers.

The system does not send every user question directly to the LLM. Instead, it first retrieves relevant document chunks using hybrid retrieval:

FAISS for semantic similarity search

BM25 for lexical / keyword search

Reciprocal Rank Fusion (RRF) to combine both rankings

The resulting context is passed through a LangChain LCEL prompt pipeline before answer generation. Document and page metadata are preserved to provide transparent source attribution.

🎯 Project Objectives

DocuMind is designed to:

Process multiple PDF documents

Extract text using LangChain PDF loaders

Split documents into meaningful chunks

Generate semantic embeddings using gemini-embedding-2

Store embeddings in a FAISS vector database

Perform semantic retrieval with FAISS

Perform keyword retrieval with BM25

Fuse semantic and lexical results using Reciprocal Rank Fusion (RRF)

Build contextual prompts dynamically

Generate grounded answers using Google Gemini

Provide document-level and page-level source attribution

Expose the RAG pipeline through a Flask REST API

Provide an interactive web interface

Open indexed PDFs directly from the knowledge-base sidebar

Display executable Python examples in a compiler-style interface

Display retrieval, generation, and total response-time metrics

Run with Gunicorn in a cloud deployment environment

🧠 What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) combines information retrieval with Large Language Models.

Instead of relying entirely on an LLM's internal knowledge, the application retrieves relevant information from an external knowledge base and supplies that information as context to the LLM.

This improves document relevance, source transparency, and control over the information used for answer generation.

🏗️ RAG Architecture

                         DOCUMENT INGESTION
                                  │
                                  ▼
                           Multiple PDFs
                                  │
                                  ▼
                         PDF Text Extraction
                                  │
                                  ▼
                         Recursive Chunking
                                  │
                                  ▼
                       Gemini Embedding 2
                                  │
                                  ▼
                            FAISS Index
                                  │
                                  │
                                  ▼
USER QUESTION ───────────────► HYBRID RETRIEVAL
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             FAISS Semantic                BM25 Keyword
                Search                         Search
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                       Reciprocal Rank Fusion
                                  │
                                  ▼
                         Top Relevant Chunks
                                  │
                                  ▼
                        Context Augmentation
                                  │
                                  ▼
                         LangChain LCEL
                                  │
                                  ▼
                         Google Gemini LLM
                                  │
                                  ▼
                    Grounded Answer + Sources

🔍 Why Hybrid Retrieval?

Semantic search and keyword search solve different retrieval problems.

Retrieval method

Strength

FAISS semantic search

Finds conceptually similar content and paraphrased questions

BM25 keyword search

Finds exact technical terms, names, acronyms, and phrases

RRF fusion

Combines both ranking signals into a unified result set

For a technical interview knowledge base, this is useful for queries containing exact terms such as FAISS, RAG, polymorphism, ACID, CNN, or overfitting, while still supporting semantically similar questions.

⚙️ Core Pipeline

1. PDF Collection
        ↓
2. PDF Parsing
        ↓
3. Recursive Text Chunking
        ↓
4. Gemini Semantic Embeddings
        ↓
5. FAISS Vector Database
        ↓
6. User Query
        ↓
7. FAISS Semantic Retrieval
        +
   BM25 Keyword Retrieval
        ↓
8. Reciprocal Rank Fusion
        ↓
9. Context Augmentation
        ↓
10. Grounded LangChain Prompt
        ↓
11. Google Gemini Generation
        ↓
12. Answer + Sources + Metrics

🧩 Technology Stack

Backend

Python 3.12

Flask

Gunicorn

AI / RAG

LangChain

LangChain LCEL

Google Gemini

Gemini Embedding 2

FAISS

BM25

Reciprocal Rank Fusion (RRF)

Document Processing

PyPDF / PyPDFLoader

RecursiveCharacterTextSplitter

Frontend

HTML5

CSS3

JavaScript

Development / Deployment

Git

GitHub

Gunicorn

RunXBuild / compatible Python hosting

📏 Chunking Configuration

The current document chunking configuration is:

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

The chunk size and overlap are configurable through environment variables:

CHUNK_SIZE=800
CHUNK_OVERLAP=150

This keeps chunks large enough to preserve context while maintaining useful retrieval granularity.

🗂️ Project Structure

AI-Powered-Document-Question-Answering-System-RAG/
│
├── app.py
├── main.py
├── query.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── documents/
│   ├── AI_Interview_Questions_Only.pdf
│   ├── Data_Science_Questions.pdf
│   ├── Machine_Learning_Notes.pdf
│   ├── Python_Interview_Questions.pdf
│   └── SQL_Interview_Questions.pdf
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
└── templates/
    └── index.html

Important deployment behavior

The FAISS vector database is built manually and then loaded by the production Flask application.

The production server should not rebuild the vector database during startup.

🔑 Environment Variables

Create a local .env file:

GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
LLM_MODEL=gemini-3.5-flash
EMBEDDING_MODEL=gemini-embedding-2
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=4
RRF_K=60

For deployment, configure these variables in the hosting platform's environment-variable settings.

⚠️ Never commit .env or API keys to GitHub.

📦 Installation

1. Clone the repository

git clone https://github.com/yogeshwar08/AI-Powered-Document-Question-Answering-System-RAG-.git
cd AI-Powered-Document-Question-Answering-System-RAG-

2. Create a virtual environment

Windows

python -m venv venv
venv\Scripts\activate

macOS / Linux

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

The hybrid retriever requires:

rank-bm25

📚 Build the Knowledge Base

Place the PDFs inside:

documents/

Then build the FAISS knowledge base locally:

python main.py

The process performs:

PDF discovery
   ↓
PDF parsing
   ↓
Chunking
   ↓
Gemini embeddings
   ↓
FAISS index creation
   ↓
vectorstore/index.faiss
vectorstore/index.pkl

After the vector database has been created, commit the vectorstore files so the deployed application can load them.

🧪 Run Locally

Query the RAG pipeline directly

python query.py

Start Flask

python app.py

Then open:

http://127.0.0.1:5000/

🌐 Flask API

Home

GET /

Returns the DocuMind web interface.

Health

GET /health

Lightweight health endpoint for deployment monitoring.

Knowledge-base statistics

GET /api/stats

Returns the indexed PDF count and document names.

Application information

GET /api/info

Returns application, model, framework, retrieval, and endpoint information.

Question answering

POST /api/ask

Example request:

{
  "question": "What is type casting in Python?",
  "top_k": 4
}

Example response structure:

{
  "success": true,
  "answer": "...",
  "sources": [
    {
      "document": "Python_Interview_Questions.pdf",
      "page": 2
    }
  ],
  "retrieval_time": 1.234,
  "generation_time": 8.421,
  "total_time": 9.655,
  "top_k": 4,
  "llm": "gemini-3.5-flash",
  "embedding_model": "gemini-embedding-2",
  "vector_database": "FAISS"
}

PDF viewer

GET /documents/<filename>

Example:

/documents/Python_Interview_Questions.pdf

This allows users to open the source PDF directly from the DocuMind interface.

🛡️ Grounding Strategy

DocuMind uses a dedicated grounding prompt to reduce unsupported document claims.

The model is instructed to:

Treat retrieved document context as the primary source of truth

Avoid fabricating document facts

Distinguish document-supported information from generated explanations

Avoid inventing sources or page numbers

State when information is unavailable in the retrieved context

Provide source attribution when metadata is available

This makes the system more transparent and suitable for technical demonstrations.

📊 Performance Metrics

The application tracks:

Retrieval time

Generation time

Total response time

Retrieved chunk count

Source count

Example:

Retrieval Time : 1.416s
Generation Time: 12.771s
Total Time     : 14.189s

Actual response times depend on network conditions, retrieval size, and Gemini API availability.

🚀 Production Deployment

The application uses Gunicorn as the production WSGI server.

Build command

pip install -r requirements.txt

Start command

gunicorn --bind 0.0.0.0:$PORT app:app

Production flow

GitHub
   ↓
Install Python dependencies
   ↓
Gunicorn
   ↓
Flask
   ↓
Load existing FAISS database
   ↓
Hybrid Retrieval
   ↓
LangChain RAG
   ↓
Google Gemini

The production application should load the existing vectorstore/ instead of rebuilding embeddings during every deployment.

🔐 Security Notes

Store Gemini API keys in environment variables

Never commit .env to GitHub

Never expose API keys in frontend JavaScript

Keep deployment secrets outside the source repository

Restrict document access through the Flask document-serving route

Validate API request payloads before executing the RAG pipeline

💼 Resume / Interview Description

Resume-ready project description

Built a production-oriented multi-document RAG system using Python, Flask, LangChain, FAISS, BM25, Reciprocal Rank Fusion, Gemini Embedding 2, and Google Gemini. Implemented PDF ingestion, recursive chunking, semantic and lexical hybrid retrieval, context augmentation, grounded answer generation, source attribution, performance monitoring, and cloud deployment with Gunicorn.

Technical interview explanation

"I implemented hybrid retrieval instead of relying solely on vector similarity. FAISS handles semantic retrieval, BM25 handles exact lexical matching, and Reciprocal Rank Fusion combines both rankings before sending the highest-quality context to the LangChain RAG generation pipeline."

🔮 Future Enhancements

Potential next improvements include:

Cross-encoder reranking

Query expansion

Metadata-aware filtering

Conversational memory

Streaming Gemini responses

Authentication and authorization

Per-user document collections

OCR support for scanned PDFs

Evaluation with retrieval precision / recall metrics

Automated RAG evaluation and hallucination detection

👨‍💻 Author

Yogeshwar Arala

AI / Machine Learning / Python Developer

⭐ Project Highlights

Multi-Document RAG          ✅
LangChain LCEL              ✅
FAISS Semantic Search       ✅
BM25 Keyword Search         ✅
RRF Hybrid Retrieval        ✅
Gemini Embeddings           ✅
Google Gemini LLM           ✅
Grounded Prompting          ✅
Source Attribution          ✅
PDF Viewer                  ✅
Flask REST API              ✅
Performance Metrics         ✅
Gunicorn Deployment         ✅

🔄 End-to-End System Flow

The complete DocuMind request lifecycle can be visualized as follows:

┌──────────────────────────────┐
│         USER / CLIENT        │
│   Ask question in browser    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        FLASK REST API        │
│        POST /api/ask         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      QUERY VALIDATION        │
│  Clean question + Top-K      │
└──────────────┬───────────────┘
               │
               ▼
      ┌────────────────────┐
      │   HYBRID RETRIEVER │
      └─────────┬──────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌───────────────┐  ┌───────────────┐
│ FAISS Search  │  │ BM25 Search   │
│ Semantic      │  │ Keyword       │
│ Similarity    │  │ Matching      │
└───────┬───────┘  └───────┬───────┘
        │                  │
        └────────┬─────────┘
                 ▼
      ┌──────────────────────┐
      │   RRF RANK FUSION    │
      │ Combine both rankings│
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ RELEVANT TOP CHUNKS  │
      │ Document + page data │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ CONTEXT AUGMENTATION │
      │ Build grounded       │
      │ context from chunks  │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │    LANGCHAIN LCEL    │
      │ Grounding Prompt     │
      │ + Retrieved Context  │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │   GOOGLE GEMINI LLM  │
      │ Answer Generation    │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │   RESPONSE BUILDER   │
      │ Answer + Sources +   │
      │ Performance Metrics  │
      └──────────┬───────────┘
                 │
                 ▼
┌──────────────────────────────┐
│          WEB UI              │
│  Answer + Sources + Metrics  │
│  + Clickable PDF Documents   │
└──────────────────────────────┘

🏭 Knowledge Base Build Flow

The offline indexing process is separated from the production query path:

┌──────────────────────┐
│     PDF DOCUMENTS    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    PyPDFLoader       │
│  Text Extraction     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Recursive Text Split │
│ 800 / 150 chunks     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Gemini Embedding 2   │
│  Semantic Vectors    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      FAISS Index     │
│ index.faiss + .pkl   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Production Runtime   │
│ Load Existing Index  │
└──────────────────────┘

Production principle: indexing is an offline/manual operation; the deployed Flask/Gunicorn service loads the existing FAISS knowledge base instead of rebuilding embeddings during startup.