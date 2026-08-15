# ============================================================
# DOCUMIND RAG
# AI-POWERED DOCUMENT QUESTION ANSWERING SYSTEM
#
# PROFESSIONAL MULTI-DOCUMENT RAG PIPELINE
#
# 1. COLLECTION OF KNOWLEDGE BASE
# 2. PARSING AND PREPROCESSING
# 3. CHUNKING
# 4. EMBEDDING
# 5. VECTOR DATABASE
# 6. USER QUERY
# 7. SEMANTIC RETRIEVAL
# 8. CONTEXT AUGMENTATION
# 9. ANSWER GENERATION
# ============================================================


# ============================================================
# STANDARD LIBRARY
# ============================================================

from functools import lru_cache
from pathlib import Path


# ============================================================
# LANGCHAIN - DOCUMENT LOADING
# ============================================================

from langchain_community.document_loaders import (
    PyPDFLoader
)


# ============================================================
# LANGCHAIN - TEXT SPLITTING
# ============================================================

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# ============================================================
# LANGCHAIN - EMBEDDINGS
# ============================================================

from langchain_huggingface import (
    HuggingFaceEmbeddings
)


# ============================================================
# LANGCHAIN - VECTOR DATABASE
# ============================================================

from langchain_community.vectorstores import (
    FAISS
)


# ============================================================
# BASE DIRECTORY
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
    BASE_DIR / "documents"
)


VECTORSTORE_PATH = (
    BASE_DIR / "vectorstore"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 4

MAX_TOP_K = 10


# ============================================================
# STEP 1
# COLLECTION OF KNOWLEDGE BASE
# ============================================================

def collect_knowledge_base():

    print()
    print("=" * 65)
    print("STEP 1: COLLECTION OF KNOWLEDGE BASE")
    print("=" * 65)

    # --------------------------------------------------------
    # Ensure documents directory exists
    # --------------------------------------------------------

    DOCUMENTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find all PDF documents
    # --------------------------------------------------------

    pdf_files = sorted(
        DOCUMENTS_PATH.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            "\nNo PDF documents found.\n\n"
            f"Expected location:\n"
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
    print("=" * 65)
    print("STEP 2: PARSING AND PREPROCESSING")
    print("=" * 65)

    documents = []

    for pdf_file in pdf_files:

        print(
            f"Parsing: "
            f"{pdf_file.name}"
        )

        try:

            loader = PyPDFLoader(
                str(pdf_file)
            )

            pdf_documents = (
                loader.load()
            )

            # ------------------------------------------------
            # Preserve document-level metadata
            # ------------------------------------------------

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
    print("=" * 65)
    print("STEP 3: DOCUMENT CHUNKING")
    print("=" * 65)

    # --------------------------------------------------------
    # LangChain Recursive Character Splitter
    # --------------------------------------------------------

    text_splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=
                CHUNK_SIZE,

            chunk_overlap=
                CHUNK_OVERLAP,

            length_function=
                len,

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
# STEP 4
# SEMANTIC EMBEDDINGS
#
# IMPORTANT:
# lru_cache ensures the embedding model is loaded
# only once per Python process.
# ============================================================

@lru_cache(
    maxsize=1
)
def create_embeddings():

    print()
    print("=" * 65)
    print("STEP 4: SEMANTIC EMBEDDING")
    print("=" * 65)

    print(
        "Embedding model:"
    )

    print(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # CPU explicitly selected
    #
    # This is important for Render because GPU is not
    # available on the normal deployment environment.
    # --------------------------------------------------------

    embeddings = (
        HuggingFaceEmbeddings(

            model_name=
                EMBEDDING_MODEL,

            model_kwargs={
                "device": "cpu"
            },

            encode_kwargs={
                "normalize_embeddings": True
            }

        )
    )

    print(
        "✓ Embedding model loaded."
    )

    return embeddings


# ============================================================
# STEP 5
# CREATE FAISS VECTOR DATABASE
# ============================================================

def create_vector_database(
    chunks
):

    print()
    print("=" * 65)
    print("STEP 5: VECTOR DATABASE")
    print("=" * 65)

    if not chunks:

        raise ValueError(
            "Cannot create vector database "
            "without document chunks."
        )

    # --------------------------------------------------------
    # Get cached embedding model
    # --------------------------------------------------------

    embeddings = (
        create_embeddings()
    )

    print()
    print(
        "Creating FAISS vector database..."
    )

    # --------------------------------------------------------
    # Convert document chunks into vectors
    # --------------------------------------------------------

    vectorstore = (
        FAISS.from_documents(
            chunks,
            embeddings
        )
    )

    # --------------------------------------------------------
    # Ensure vectorstore directory exists
    # --------------------------------------------------------

    VECTORSTORE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS database
    # --------------------------------------------------------

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
# CHECK VECTOR DATABASE
# ============================================================

def vectorstore_exists():

    faiss_file = (
        VECTORSTORE_PATH /
        "index.faiss"
    )

    metadata_file = (
        VECTORSTORE_PATH /
        "index.pkl"
    )

    return (
        faiss_file.exists()
        and
        metadata_file.exists()
    )


# ============================================================
# LOAD FAISS VECTOR DATABASE
#
# IMPORTANT:
# This function is cached.
#
# The embedding model and FAISS database are therefore
# initialized only once per Gunicorn worker.
#
# It is NOT executed when app.py merely imports main.py.
# ============================================================

@lru_cache(
    maxsize=1
)
def load_vectorstore():

    print()
    print("=" * 65)
    print("LOADING RAG VECTOR DATABASE")
    print("=" * 65)

    # ========================================================
    # EXISTING VECTOR DATABASE
    # ========================================================

    if vectorstore_exists():

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

                allow_dangerous_deserialization=
                    True

            )
        )

        print(
            "✓ FAISS vector database "
            "loaded successfully."
        )

        return vectorstore


    # ========================================================
    # VECTOR DATABASE DOES NOT EXIST
    #
    # Build it automatically.
    # ========================================================

    print(
        "FAISS database not found."
    )

    print(
        "Building knowledge base..."
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    pdf_files = (
        collect_knowledge_base()
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    documents = (
        parse_documents(
            pdf_files
        )
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    chunks = (
        chunk_documents(
            documents
        )
    )

    # --------------------------------------------------------
    # STEP 4 + STEP 5
    # --------------------------------------------------------

    vectorstore = (
        create_vector_database(
            chunks
        )
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
            "Please enter a meaningful "
            "question."
        )

    # --------------------------------------------------------
    # Prevent unnecessarily huge queries
    # --------------------------------------------------------

    if len(question) > 5000:

        raise ValueError(
            "Question is too long. "
            "Please keep it below 5000 characters."
        )

    return question


# ============================================================
# STEP 7
# SEMANTIC RETRIEVAL
#
# LANGCHAIN RETRIEVER
# ============================================================

def retrieve_documents(
    question,
    k=DEFAULT_TOP_K
):

    print()
    print("=" * 65)
    print("STEP 7: SEMANTIC RETRIEVAL")
    print("=" * 65)

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    question = (
        validate_query(
            question
        )
    )

    # --------------------------------------------------------
    # Validate Top-K
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load cached FAISS
    # --------------------------------------------------------

    vectorstore = (
        load_vectorstore()
    )

    # --------------------------------------------------------
    # LangChain Retriever
    # --------------------------------------------------------

    retriever = (
        vectorstore.as_retriever(

            search_type=
                "similarity",

            search_kwargs={
                "k": k
            }

        )
    )

    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Display source information
    # --------------------------------------------------------

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

        # PyPDFLoader uses zero-based pages
        if isinstance(
            page,
            int
        ):

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
    print("=" * 65)
    print("STEP 8: CONTEXT AUGMENTATION")
    print("=" * 65)

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

        if isinstance(
            page,
            int
        ):

            page += 1

        content = (
            document.page_content
            or ""
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
        "\n\n"
        .join(
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

        if isinstance(
            page,
            int
        ):

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
#
# USED BY app.py
#
# IMPORTANT:
# This does NOT load FAISS.
# This does NOT load HuggingFace.
# This keeps Flask startup lightweight.
# ============================================================

def get_document_stats():

    DOCUMENTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        DOCUMENTS_PATH.glob("*.pdf")
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
#
# This is used when you explicitly run:
#
#     python main.py
#
# ============================================================

def build_vector_database():

    print()
    print("=" * 65)
    print("       BUILDING RAG KNOWLEDGE BASE")
    print("=" * 65)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    pdf_files = (
        collect_knowledge_base()
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    documents = (
        parse_documents(
            pdf_files
        )
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    chunks = (
        chunk_documents(
            documents
        )
    )

    # --------------------------------------------------------
    # STEP 4 + STEP 5
    # --------------------------------------------------------

    create_vector_database(
        chunks
    )

    # --------------------------------------------------------
    # Clear cached objects
    #
    # This matters if build_vector_database() is called
    # more than once during the same Python process.
    # --------------------------------------------------------

    load_vectorstore.cache_clear()

    create_embeddings.cache_clear()

    print()
    print("=" * 65)
    print("          KNOWLEDGE BASE READY")
    print("=" * 65)

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

    print()



if __name__ == "__main__":

    build_vector_database()