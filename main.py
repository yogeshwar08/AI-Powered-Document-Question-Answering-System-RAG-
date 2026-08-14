from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# BASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# SETTINGS
# ============================================================

DOCUMENTS_PATH = BASE_DIR / "documents"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    # Make sure documents folder exists
    DOCUMENTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = list(
        DOCUMENTS_PATH.glob("*.pdf")
    )

    if not pdf_files:

        print(
            f"No PDF files found in: "
            f"{DOCUMENTS_PATH}"
        )

        return documents

    for pdf_file in pdf_files:

        print(
            f"Loading PDF: {pdf_file.name}"
        )

        try:

            loader = PyPDFLoader(
                str(pdf_file)
            )

            pdf_documents = loader.load()

            documents.extend(
                pdf_documents
            )

        except Exception as e:

            print(
                f"Error loading "
                f"{pdf_file.name}: {e}"
            )

    print(
        f"Total PDF pages loaded: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )
    )

    chunks = (
        text_splitter.split_documents(
            documents
        )
    )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    print(
        "Loading embedding model:"
    )

    print(
        EMBEDDING_MODEL
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded successfully."
    )

    return embeddings


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

def create_vectorstore(chunks):

    if not chunks:

        raise ValueError(
            "No document chunks available "
            "to create vector store."
        )

    print(
        "Creating embeddings..."
    )

    embeddings = create_embeddings()

    print(
        "Creating FAISS vector store..."
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Create directory
    VECTORSTORE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS index
    vectorstore.save_local(
        str(VECTORSTORE_PATH)
    )

    print(
        "FAISS vector store created successfully."
    )

    print(
        f"Saved to: {VECTORSTORE_PATH}"
    )

    return vectorstore


# ============================================================
# CHECK VECTORSTORE
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
        and metadata_file.exists()
    )


# ============================================================
# LOAD EXISTING VECTOR STORE
# ============================================================

def load_vectorstore():

    # --------------------------------------------------------
    # If vectorstore doesn't exist, build it automatically
    # --------------------------------------------------------

    if not vectorstore_exists():

        print(
            "FAISS vector store not found."
        )

        print(
            "Attempting to build vector store..."
        )

        documents = load_documents()

        if not documents:

            raise FileNotFoundError(
                "No PDF documents found. "
                f"Please add PDF files to: "
                f"{DOCUMENTS_PATH}"
            )

        chunks = split_documents(
            documents
        )

        return create_vectorstore(
            chunks
        )

    # --------------------------------------------------------
    # Load existing vectorstore
    # --------------------------------------------------------

    print(
        "Loading existing FAISS vector store..."
    )

    print(
        f"Vectorstore path: "
        f"{VECTORSTORE_PATH}"
    )

    embeddings = create_embeddings()

    try:

        vectorstore = FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception as e:

        raise RuntimeError(
            "Failed to load FAISS vector store. "
            f"Error: {e}"
        )

    print(
        "FAISS vector store loaded successfully."
    )

    return vectorstore


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    k=4
):

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    vectorstore = load_vectorstore()

    documents = (
        vectorstore.similarity_search(
            question,
            k=k
        )
    )

    return documents


# ============================================================
# BUILD VECTOR DATABASE MANUALLY
# ============================================================

def build_vector_database():

    print()
    print(
        "========================================"
    )
    print(
        "       Building RAG Vector Database"
    )
    print(
        "========================================"
    )
    print()

    # --------------------------------------------------------
    # Load PDFs
    # --------------------------------------------------------

    documents = load_documents()

    print(
        f"Documents loaded: "
        f"{len(documents)}"
    )

    if not documents:

        print()
        print(
            "Please put at least one PDF inside:"
        )

        print(
            DOCUMENTS_PATH
        )

        return

    # --------------------------------------------------------
    # Split PDFs
    # --------------------------------------------------------

    chunks = split_documents(
        documents
    )

    # --------------------------------------------------------
    # Create vectorstore
    # --------------------------------------------------------

    create_vectorstore(
        chunks
    )

    print()
    print(
        "========================================"
    )
    print(
        "       Vector Database Ready"
    )
    print(
        "========================================"
    )
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_vector_database()