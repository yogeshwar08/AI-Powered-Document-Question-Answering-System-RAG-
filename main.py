# ============================================================
# DOCUMIND RAG
# AI-POWERED DOCUMENT QUESTION ANSWERING SYSTEM
#
# PROFESSIONAL MULTI-DOCUMENT RAG PIPELINE
#
# 1. COLLECTION OF KNOWLEDGE BASE
# 2. PARSING AND PREPROCESSING
# 3. DOCUMENT CHUNKING
# 4. SEMANTIC EMBEDDING
# 5. FAISS VECTOR DATABASE
# 6. USER QUERY
# 7. SEMANTIC RETRIEVAL
# 8. CONTEXT AUGMENTATION
# 9. ANSWER GENERATION
#
# PRODUCTION DESIGN
# ------------------------------------------------------------
# - LangChain
# - Gemini Embedding 2
# - FAISS
# - Google GenAI SDK
# - No local PyTorch / Sentence Transformers
# - Existing FAISS database loaded at runtime
# - Knowledge-base rebuilding is manual only
# - Retry handling for Gemini embedding requests
# - Lightweight Flask/Gunicorn startup
# ============================================================


# ============================================================
# STANDARD LIBRARY
# ============================================================

import os
import time

from functools import lru_cache
from pathlib import Path


# ============================================================
# ENVIRONMENT
# ============================================================

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# API CONFIGURATION
# ============================================================

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or
    os.getenv("GEMINI_API_KEY")
)


if not GOOGLE_API_KEY:

    raise RuntimeError(

        "GOOGLE_API_KEY is not configured.\n\n"

        "For local development, create a .env file:\n"

        "GOOGLE_API_KEY=YOUR_API_KEY\n\n"

        "For deployment, configure GOOGLE_API_KEY "
        "as an environment variable."
    )


# ============================================================
# LANGCHAIN DOCUMENT LOADER
# ============================================================

from langchain_community.document_loaders import (
    PyPDFLoader
)


# ============================================================
# LANGCHAIN TEXT SPLITTER
# ============================================================

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# ============================================================
# LANGCHAIN EMBEDDING INTERFACE
# ============================================================

from langchain_core.embeddings import (
    Embeddings
)


# ============================================================
# LANGCHAIN FAISS
# ============================================================

from langchain_community.vectorstores import (
    FAISS
)


# ============================================================
# GOOGLE GENAI
# ============================================================

from google import genai

from google.genai import types


# ============================================================
# PROJECT BASE DIRECTORY
# ============================================================

BASE_DIR = (

    Path(__file__)
    .resolve()
    .parent

)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

DOCUMENTS_PATH = (

    BASE_DIR
    /
    "documents"

)


VECTORSTORE_PATH = (

    BASE_DIR
    /
    "vectorstore"

)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBEDDING_MODEL = (

    os.getenv(

        "EMBEDDING_MODEL",

        "gemini-embedding-2"

    )

)


# Gemini Embedding 2 output dimension.
EMBEDDING_DIMENSION = int(

    os.getenv(

        "EMBEDDING_DIMENSION",

        "768"

    )

)


# ============================================================
# EMBEDDING RETRY CONFIGURATION
# ============================================================

EMBEDDING_MAX_RETRIES = int(

    os.getenv(

        "EMBEDDING_MAX_RETRIES",

        "5"

    )

)


EMBEDDING_RETRY_DELAY = float(

    os.getenv(

        "EMBEDDING_RETRY_DELAY",

        "2"

    )

)


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = int(

    os.getenv(

        "CHUNK_SIZE",

        "800"

    )

)


CHUNK_OVERLAP = int(

    os.getenv(

        "CHUNK_OVERLAP",

        "150"

    )

)


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

DEFAULT_TOP_K = int(

    os.getenv(

        "TOP_K",

        "4"

    )

)


MAX_TOP_K = 10


# ============================================================
# STEP 4
# GEMINI EMBEDDING CLASS
# ============================================================

class GeminiEmbeddings(Embeddings):
    """
    LangChain-compatible embedding implementation
    using the Google Gemini Embedding API.

    The application uses Gemini for semantic embeddings
    instead of loading a local Sentence Transformer model.

    This keeps the production container lightweight.
    """

    def __init__(self):

        self.client = (

            genai.Client(

                api_key=GOOGLE_API_KEY

            )

        )


    # ========================================================
    # SINGLE EMBEDDING REQUEST
    # ========================================================

    def _embed_single(
        self,
        text,
        position=None
    ):
        """
        Generate one embedding with retry handling.

        Temporary network failures such as:

        - RemoteProtocolError
        - Server disconnected
        - Connection reset
        - Timeout

        are retried using exponential backoff.
        """

        last_error = None


        for attempt in range(

            1,

            EMBEDDING_MAX_RETRIES + 1

        ):

            try:

                response = (

                    self.client.models.embed_content(

                        model=EMBEDDING_MODEL,

                        contents=text,

                        config=(

                            types.EmbedContentConfig(

                                output_dimensionality=(
                                    EMBEDDING_DIMENSION
                                )

                            )

                        )

                    )

                )


                if not response.embeddings:

                    raise RuntimeError(

                        "Gemini returned no embedding."
                    )


                vector = (

                    response
                    .embeddings[0]
                    .values

                )


                if not vector:

                    raise RuntimeError(

                        "Gemini returned an empty embedding."
                    )


                return list(vector)


            except Exception as error:

                last_error = error


                error_text = str(
                    error
                ).lower()


                retryable = any(

                    term in error_text

                    for term in [

                        "remoteprotocolerror",

                        "server disconnected",

                        "peer closed connection",

                        "incomplete chunked read",

                        "connection reset",

                        "connection aborted",

                        "connecterror",

                        "timeout",

                        "timed out",

                        "temporarily unavailable",

                        "service unavailable",

                        "internal server error",

                        "bad gateway",

                        "gateway timeout",

                        "429",

                        "500",

                        "502",

                        "503",

                        "504"

                    ]

                )


                print()

                print(

                    "  ⚠ Gemini embedding request failed."

                )


                if position is not None:

                    print(

                        f"    Chunk: {position}"

                    )


                print(

                    f"    Attempt: "
                    f"{attempt}/"
                    f"{EMBEDDING_MAX_RETRIES}"

                )


                print(

                    f"    Error: {error}"

                )


                if (

                    not retryable

                    or

                    attempt >= EMBEDDING_MAX_RETRIES

                ):

                    raise


                delay = (

                    EMBEDDING_RETRY_DELAY
                    *
                    (2 ** (attempt - 1))

                )


                print(

                    f"    Retrying in "
                    f"{delay:.1f} seconds..."

                )


                time.sleep(
                    delay
                )


        raise RuntimeError(

            "Gemini embedding failed after "
            f"{EMBEDDING_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"

        )


    # ========================================================
    # MULTIPLE DOCUMENT EMBEDDINGS
    # ========================================================

    def embed_documents(
        self,
        texts
    ):
        """
        Generate exactly one embedding for every
        document chunk.

        This guarantees:

            documents == embeddings

        which is required by FAISS.
        """

        if not texts:

            return []


        embeddings = []

        total = len(
            texts
        )


        for index, text in enumerate(

            texts,

            start=1

        ):

            if not text or not text.strip():

                raise ValueError(

                    f"Empty text encountered "
                    f"at chunk {index}."

                )


            vector = (

                self._embed_single(

                    text,

                    position=(

                        f"{index}/{total}"

                    )

                )

            )


            embeddings.append(
                vector
            )


            print(

                f"  Embedded chunk "
                f"{index}/{total}",

                end="\r"

            )


        print()


        if len(embeddings) != len(texts):

            raise RuntimeError(

                "Embedding count mismatch: "

                f"documents={len(texts)}, "

                f"embeddings={len(embeddings)}"

            )


        return embeddings


    # ========================================================
    # SINGLE QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        text
    ):
        """
        Generate an embedding for a user query.
        """

        if not text or not text.strip():

            raise ValueError(

                "Query cannot be empty."

            )


        return (

            self._embed_single(

                text,

                position="query"

            )

        )


# ============================================================
# CACHED EMBEDDING CLIENT
# ============================================================

@lru_cache(
    maxsize=1
)
def create_embeddings():

    print()

    print(
        "=" * 65
    )

    print(
        "STEP 4: SEMANTIC EMBEDDING"
    )

    print(
        "=" * 65
    )

    print(

        f"Embedding model : "
        f"{EMBEDDING_MODEL}"

    )

    print(

        f"Embedding size  : "
        f"{EMBEDDING_DIMENSION}"

    )


    embeddings = (

        GeminiEmbeddings()

    )


    print(

        "✓ Gemini embedding client initialized."

    )


    return embeddings


# ============================================================
# STEP 1
# COLLECTION OF KNOWLEDGE BASE
# ============================================================

def collect_knowledge_base():

    print()

    print(
        "=" * 65
    )

    print(
        "STEP 1: COLLECTION OF KNOWLEDGE BASE"
    )

    print(
        "=" * 65
    )


    DOCUMENTS_PATH.mkdir(

        parents=True,

        exist_ok=True

    )


    pdf_files = sorted(

        DOCUMENTS_PATH.glob(
            "*.pdf"
        )

    )


    if not pdf_files:

        raise FileNotFoundError(

            "\nNo PDF documents found.\n\n"

            f"Expected directory:\n"
            f"{DOCUMENTS_PATH}\n\n"

            "Place your PDF files inside "
            "the documents folder."

        )


    print(

        f"PDF documents found: "
        f"{len(pdf_files)}"

    )


    for pdf_file in pdf_files:

        print(

            f"  ✓ {pdf_file.name}"

        )


    return pdf_files


# ============================================================
# STEP 2
# PARSING AND PREPROCESSING
# ============================================================

def parse_documents(
    pdf_files
):

    print()

    print(
        "=" * 65
    )

    print(
        "STEP 2: PARSING AND PREPROCESSING"
    )

    print(
        "=" * 65
    )


    documents = []


    for pdf_file in pdf_files:

        print()

        print(

            f"Parsing: "
            f"{pdf_file.name}"

        )


        try:

            loader = (

                PyPDFLoader(

                    str(pdf_file)

                )

            )


            pdf_documents = (

                loader.load()

            )


            for document in pdf_documents:

                document.metadata[

                    "source"

                ] = pdf_file.name


                documents.append(
                    document
                )


            print(

                f"  ✓ Pages loaded: "
                f"{len(pdf_documents)}"

            )


        except Exception as error:

            print()

            print(

                f"  ✗ Failed to parse: "
                f"{pdf_file.name}"

            )

            print(

                f"    Error: {error}"

            )


    if not documents:

        raise ValueError(

            "No readable text could be "
            "extracted from the PDF files."

        )


    print()

    print(

        f"Total pages parsed: "
        f"{len(documents)}"

    )


    return documents


# ============================================================
# STEP 3
# DOCUMENT CHUNKING
# ============================================================

def chunk_documents(
    documents
):

    print()

    print(
        "=" * 65
    )

    print(
        "STEP 3: DOCUMENT CHUNKING"
    )

    print(
        "=" * 65
    )


    text_splitter = (

        RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            length_function=len,

            separators=[

                "\n\n",

                "\n",

                ". ",

                " ",

                ""

            ]

        )

    )


    chunks = (

        text_splitter.split_documents(

            documents

        )

    )


    if not chunks:

        raise ValueError(

            "Document chunking produced "
            "zero chunks."

        )


    print(

        f"Chunk size    : "
        f"{CHUNK_SIZE}"

    )


    print(

        f"Chunk overlap : "
        f"{CHUNK_OVERLAP}"

    )


    print(

        f"Total chunks  : "
        f"{len(chunks)}"

    )


    return chunks


# ============================================================
# STEP 5
# CREATE FAISS VECTOR DATABASE
# ============================================================

def create_vector_database(
    chunks
):

    print()

    print(
        "=" * 65
    )

    print(
        "STEP 5: VECTOR DATABASE"
    )

    print(
        "=" * 65
    )


    if not chunks:

        raise ValueError(

            "Cannot create vector database "
            "without document chunks."

        )


    embeddings = (
        create_embeddings()
    )


    print()

    print(
        "Creating FAISS vector database..."
    )


    vectorstore = (

        FAISS.from_documents(

            chunks,

            embeddings

        )

    )


    VECTORSTORE_PATH.mkdir(

        parents=True,

        exist_ok=True

    )


    vectorstore.save_local(

        str(

            VECTORSTORE_PATH

        )

    )


    print()

    print(

        "✓ FAISS vector database "
        "created successfully."

    )


    print(

        f"Location: "
        f"{VECTORSTORE_PATH}"

    )


    return vectorstore


# ============================================================
# CHECK EXISTING VECTOR DATABASE
# ============================================================

def vectorstore_exists():

    faiss_file = (

        VECTORSTORE_PATH
        /
        "index.faiss"

    )


    metadata_file = (

        VECTORSTORE_PATH
        /
        "index.pkl"

    )


    return (

        faiss_file.exists()

        and

        metadata_file.exists()

    )


# ============================================================
# LOAD EXISTING FAISS VECTOR DATABASE
# ============================================================
#
# IMPORTANT:
#
# Production Flask/Gunicorn MUST NOT automatically
# rebuild the vector database.
#
# If vectorstore files are missing, fail clearly.
#
# ============================================================

@lru_cache(
    maxsize=1
)
def load_vectorstore():

    print()

    print(
        "=" * 65
    )

    print(
        "LOADING RAG VECTOR DATABASE"
    )

    print(
        "=" * 65
    )


    if not vectorstore_exists():

        raise FileNotFoundError(

            "\nFAISS vector database was not found.\n\n"

            "Expected files:\n"

            f"  {VECTORSTORE_PATH / 'index.faiss'}\n"

            f"  {VECTORSTORE_PATH / 'index.pkl'}\n\n"

            "Build the vector database locally with:\n"

            "  python main.py\n\n"

            "Then commit the vectorstore directory "
            "to GitHub before deployment."

        )


    print(

        "Existing FAISS database found."

    )


    print(

        "Loading vector database..."

    )


    embeddings = (

        create_embeddings()

    )


    vectorstore = (

        FAISS.load_local(

            str(

                VECTORSTORE_PATH

            ),

            embeddings,

            allow_dangerous_deserialization=True

        )

    )


    print(

        "✓ FAISS vector database "
        "loaded successfully."

    )


    return vectorstore


# ============================================================
# STEP 6
# USER QUERY VALIDATION
# ============================================================

def validate_query(
    question
):

    if question is None:

        raise ValueError(

            "Question cannot be empty."

        )


    if not isinstance(

        question,

        str

    ):

        raise ValueError(

            "Question must be text."

        )


    question = (

        question.strip()

    )


    if not question:

        raise ValueError(

            "Question cannot be empty."

        )


    if len(question) < 2:

        raise ValueError(

            "Please enter a meaningful question."

        )


    if len(question) > 5000:

        raise ValueError(

            "Question is too long. "
            "Please keep it below 5000 characters."

        )


    return question


# ============================================================
# STEP 7
# SEMANTIC RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    k=DEFAULT_TOP_K
):

    print()

    print(
        "=" * 65
    )

    print(
        "STEP 7: SEMANTIC RETRIEVAL"
    )

    print(
        "=" * 65
    )


    question = (

        validate_query(

            question

        )

    )


    try:

        k = int(k)

    except (

        TypeError,

        ValueError

    ):

        k = DEFAULT_TOP_K


    k = max(

        1,

        min(

            k,

            MAX_TOP_K

        )

    )


    print(

        f"Query  : {question}"

    )


    print(

        f"Top-K  : {k}"

    )


    vectorstore = (

        load_vectorstore()

    )


    retriever = (

        vectorstore.as_retriever(

            search_type="similarity",

            search_kwargs={

                "k": k

            }

        )

    )


    documents = (

        retriever.invoke(

            question

        )

    )


    print()

    print(

        f"Relevant chunks retrieved: "
        f"{len(documents)}"

    )


    for index, document in enumerate(

        documents,

        start=1

    ):

        source = (

            document.metadata.get(

                "source",

                "Unknown"

            )

        )


        page = (

            document.metadata.get(

                "page",

                "Unknown"

            )

        )


        # PyPDFLoader uses zero-based page indexes.
        if isinstance(page, int):

            page += 1


        print(

            f"  {index}. "
            f"{source} | "
            f"Page {page}"

        )


    return documents


# ============================================================
# STEP 8
# CONTEXT AUGMENTATION
# ============================================================

def build_context(
    documents
):

    print()

    print(
        "=" * 65
    )

    print(
        "STEP 8: CONTEXT AUGMENTATION"
    )

    print(
        "=" * 65
    )


    if not documents:

        return ""


    context_parts = []


    for index, document in enumerate(

        documents,

        start=1

    ):

        source = (

            document.metadata.get(

                "source",

                "Unknown"

            )

        )


        page = (

            document.metadata.get(

                "page",

                "Unknown"

            )

        )


        if isinstance(page, int):

            page += 1


        content = (

            document.page_content

            or

            ""

        )


        context_block = (

            f"SOURCE {index}\n"

            f"DOCUMENT: {source}\n"

            f"PAGE: {page}\n\n"

            f"CONTENT:\n"

            f"{content}"

        )


        context_parts.append(

            context_block

        )


    final_context = (

        "\n\n".join(

            context_parts

        )

    )


    print(

        f"Context created from "
        f"{len(documents)} chunks."

    )


    print(

        f"Context characters: "
        f"{len(final_context)}"

    )


    return final_context


# ============================================================
# SOURCE INFORMATION
# ============================================================

def get_sources(
    documents
):

    sources = []

    seen = set()


    for document in documents:

        source = (

            document.metadata.get(

                "source",

                "Unknown"

            )

        )


        page = (

            document.metadata.get(

                "page",

                "Unknown"

            )

        )


        if isinstance(page, int):

            page += 1


        source_key = (

            source,

            page

        )


        if source_key not in seen:

            seen.add(

                source_key

            )


            sources.append(

                (

                    source,

                    page

                )

            )


    return sources


# ============================================================
# KNOWLEDGE BASE STATISTICS
# ============================================================
#
# IMPORTANT:
# This function does NOT load FAISS.
# This function does NOT call Gemini.
#
# It is safe for Flask startup.
# ============================================================

def get_document_stats():

    DOCUMENTS_PATH.mkdir(

        parents=True,

        exist_ok=True

    )


    pdf_files = sorted(

        DOCUMENTS_PATH.glob(

            "*.pdf"

        )

    )


    pdf_names = [

        pdf_file.name

        for pdf_file in pdf_files

    ]


    return (

        len(pdf_files),

        pdf_names

    )


# ============================================================
# BUILD / REBUILD KNOWLEDGE BASE
# ============================================================
#
# IMPORTANT:
#
# This function is ONLY for manually rebuilding FAISS.
#
# Run:
#
#     python main.py
#
# Do NOT use this as the Gunicorn startup command.
# ============================================================

def build_vector_database():

    print()

    print(
        "=" * 65
    )

    print(
        "       BUILDING RAG KNOWLEDGE BASE"
    )

    print(
        "=" * 65
    )


    # ========================================================
    # STEP 1
    # ========================================================

    pdf_files = (

        collect_knowledge_base()

    )


    # ========================================================
    # STEP 2
    # ========================================================

    documents = (

        parse_documents(

            pdf_files

        )

    )


    # ========================================================
    # STEP 3
    # ========================================================

    chunks = (

        chunk_documents(

            documents

        )

    )


    # ========================================================
    # STEP 4 + STEP 5
    # ========================================================

    create_vector_database(

        chunks

    )


    # ========================================================
    # CLEAR CACHES
    # ========================================================

    load_vectorstore.cache_clear()

    create_embeddings.cache_clear()


    print()

    print(
        "=" * 65
    )

    print(
        "          KNOWLEDGE BASE READY"
    )

    print(
        "=" * 65
    )


    print()

    print(

        f"PDF Documents : "
        f"{len(pdf_files)}"

    )


    print(

        f"Pages         : "
        f"{len(documents)}"

    )


    print(

        f"Chunks        : "
        f"{len(chunks)}"

    )


    print(
        "Vector DB     : FAISS"
    )


    print(

        f"Embeddings    : "
        f"{EMBEDDING_MODEL}"

    )


    print(

        f"Embedding Dim : "
        f"{EMBEDDING_DIMENSION}"

    )


    print()

    print(
        "Vector database files:"
    )

    print(

        f"  {VECTORSTORE_PATH / 'index.faiss'}"

    )

    print(

        f"  {VECTORSTORE_PATH / 'index.pkl'}"

    )

    print()


# ============================================================
# COMMAND LINE BUILD MODE
# ============================================================

if __name__ == "__main__":

    build_vector_database()