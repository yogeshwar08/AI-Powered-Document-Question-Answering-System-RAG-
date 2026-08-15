# ============================================================
# DOCUMIND RAG
# AI-POWERED DOCUMENT QUESTION ANSWERING SYSTEM
#
# RAG PIPELINE
#
# 1. COLLECTION OF KNOWLEDGE BASE
# 2. PARSING AND PREPROCESSING
# 3. CHUNKING
# 4. EMBEDDING
# 5. VECTOR DATABASE
# 6. USER QUERY
# 7. SEARCH QUERY AGAINST VECTOR DATABASE
# 8. AUGMENTATION
# 9. GENERATE ANSWER
# ============================================================


from pathlib import Path


# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


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


# ============================================================
# STEP 1
# COLLECTION OF KNOWLEDGE BASE
# ============================================================

def collect_knowledge_base():

    print()
    print("=" * 65)
    print("STEP 1: COLLECTION OF KNOWLEDGE BASE")
    print("=" * 65)

    # Create documents directory
    # if it does not exist

    DOCUMENTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find PDF documents

    pdf_files = sorted(
        DOCUMENTS_PATH.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF documents found in:\n"
            f"{DOCUMENTS_PATH}\n\n"
            f"Please place your PDF files inside "
            f"the 'documents' folder."
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

            # Add source metadata

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
                f"  ✗ Failed to parse "
                f"{pdf_file.name}"
            )

            print(
                f"    Error: {error}"
            )

    if not documents:

        raise ValueError(
            "No text could be extracted "
            "from the provided PDF documents."
        )

    print()
    print(
        f"Total pages parsed: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# STEP 3
# CHUNKING
# ============================================================

def chunk_documents(
    documents
):

    print()
    print("=" * 65)
    print("STEP 3: DOCUMENT CHUNKING")
    print("=" * 65)

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
        f"Chunk size: "
        f"{CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: "
        f"{CHUNK_OVERLAP}"
    )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    return chunks


# ============================================================
# STEP 4
# EMBEDDING
# ============================================================

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

    embeddings = (
        HuggingFaceEmbeddings(
            model_name=
                EMBEDDING_MODEL
        )
    )

    print(
        "✓ Embedding model loaded."
    )

    return embeddings


# ============================================================
# STEP 5
# CREATE VECTOR DATABASE
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
    # Create embeddings
    # --------------------------------------------------------

    embeddings = (
        create_embeddings()
    )

    print()
    print(
        "Creating FAISS vector database..."
    )

    # --------------------------------------------------------
    # Convert documents into vectors
    # --------------------------------------------------------

    vectorstore = (
        FAISS.from_documents(
            chunks,
            embeddings
        )
    )

    # --------------------------------------------------------
    # Create vectorstore directory
    # --------------------------------------------------------

    VECTORSTORE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    vectorstore.save_local(
        str(VECTORSTORE_PATH)
    )

    print()
    print(
        "✓ FAISS vector database "
        "created successfully."
    )

    print(
        f"Location:"
    )

    print(
        VECTORSTORE_PATH
    )

    return vectorstore


# ============================================================
# CHECK VECTOR DATABASE
# ============================================================

def vectorstore_exists():

    index_file = (
        VECTORSTORE_PATH /
        "index.faiss"
    )

    metadata_file = (
        VECTORSTORE_PATH /
        "index.pkl"
    )

    return (
        index_file.exists()
        and
        metadata_file.exists()
    )


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def load_vectorstore():

    # --------------------------------------------------------
    # Check whether FAISS already exists
    # --------------------------------------------------------

    if not vectorstore_exists():

        print()
        print(
            "FAISS vector database "
            "was not found."
        )

        print(
            "Building knowledge base..."
        )

        # Step 1

        pdf_files = (
            collect_knowledge_base()
        )

        # Step 2

        documents = (
            parse_documents(
                pdf_files
            )
        )

        # Step 3

        chunks = (
            chunk_documents(
                documents
            )
        )

        # Step 4 + Step 5

        return create_vector_database(
            chunks
        )

    # --------------------------------------------------------
    # Load existing FAISS database
    # --------------------------------------------------------

    print()
    print(
        "Loading existing FAISS "
        "vector database..."
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

    return question


# ============================================================
# STEP 7
# SEARCH QUERY AGAINST VECTOR DATABASE
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
    # Validate user query
    # --------------------------------------------------------

    question = validate_query(
        question
    )

    # --------------------------------------------------------
    # Validate Top-K
    # --------------------------------------------------------

    k = max(
        1,
        min(
            int(k),
            10
        )
    )

    print(
        f"Query: {question}"
    )

    print(
        f"Top-K: {k}"
    )

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    vectorstore = (
        load_vectorstore()
    )

    # --------------------------------------------------------
    # Create LangChain Retriever
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
    # Perform semantic search
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
    # Display retrieved sources
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

        # PyPDFLoader pages are
        # zero-indexed

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

        context = (
            f"SOURCE {index}\n"
            f"DOCUMENT: {source}\n"
            f"PAGE: {page}\n\n"
            f"CONTENT:\n"
            f"{document.page_content}"
        )

        context_parts.append(
            context
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
# ============================================================

def get_document_stats():

    # Make sure documents directory exists

    DOCUMENTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find all PDF documents

    pdf_files = sorted(
        DOCUMENTS_PATH.glob("*.pdf")
    )

    # Extract file names

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

def build_vector_database():

    print()
    print("=" * 65)
    print("          BUILDING RAG KNOWLEDGE BASE")
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
        f"Vector DB     : "
        f"FAISS"
    )

    print(
        f"Embeddings    : "
        f"{EMBEDDING_MODEL}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_vector_database()