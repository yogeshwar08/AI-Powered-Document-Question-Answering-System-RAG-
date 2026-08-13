from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# SETTINGS
# ============================================================

DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    pdf_files = list(
        Path(DOCUMENTS_PATH).glob("*.pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return documents

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(
            str(pdf_file)
        )

        pdf_documents = loader.load()

        documents.extend(
            pdf_documents
        )

    return documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

def create_vectorstore(chunks):

    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    print(
        "FAISS vector store created successfully."
    )

    return vectorstore


# ============================================================
# LOAD EXISTING VECTOR STORE
# ============================================================

def load_vectorstore():

    vectorstore_path = Path(
        VECTORSTORE_PATH
    )

    if not vectorstore_path.exists():

        raise FileNotFoundError(
            "The 'vectorstore' folder was not found."
        )

    embeddings = create_embeddings()

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    k=4,
):

    vectorstore = load_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=k,
    )

    return documents


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "     Building RAG Vector Database"
    )

    print(
        "========================================\n"
    )

    documents = load_documents()

    print(
        f"Documents loaded: {len(documents)}"
    )

    if not documents:

        print(
            "\nPlease put at least one PDF inside:"
        )

        print(
            "documents/"
        )

        raise SystemExit

    chunks = split_documents(
        documents
    )

    create_vectorstore(
        chunks
    )

    print(
        "\n========================================"
    )

    print(
        "   Vector Database Ready"
    )

    print(
        "========================================\n"
    )