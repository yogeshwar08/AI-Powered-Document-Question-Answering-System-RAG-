import os
import logging
import warnings

# Disable Hugging Face progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Disable Windows symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Hide Hugging Face Hub HTTP warnings
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Hide matching Python warnings
warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests to the HF Hub.*"
)

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"


# --------------------------------------------------
# 1. Load PDF Documents
# --------------------------------------------------

def load_documents():

    documents = []

    pdf_files = list(Path(DOCUMENTS_PATH).glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the documents folder.")
        return documents

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        pdf_documents = loader.load()

        documents.extend(pdf_documents)

    return documents


# --------------------------------------------------
# 2. Split Documents into Chunks
# --------------------------------------------------

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# --------------------------------------------------
# 3. Create Embedding Model
# --------------------------------------------------

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# --------------------------------------------------
# 4. Create FAISS Vector Store
# --------------------------------------------------

def create_vectorstore(chunks):

    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(VECTORSTORE_PATH)

    print("FAISS vector store created successfully.")

    return vectorstore


# --------------------------------------------------
# 5. Load Existing FAISS Vector Store
# --------------------------------------------------

def load_vectorstore():

    embeddings = create_embeddings()

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# --------------------------------------------------
# 6. Retrieve Relevant Documents
# --------------------------------------------------

def retrieve_documents(question, k=4):

    vectorstore = load_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=k
    )

    return documents


# --------------------------------------------------
# Main Program
# --------------------------------------------------

if __name__ == "__main__":

    print("\nStarting RAG document processing...\n")

    # Load PDFs
    documents = load_documents()

    print(f"\nDocuments loaded: {len(documents)}")

    if not documents:
        print("\nPlease place at least one PDF inside the 'documents' folder.")
        exit()

    # Split documents
    chunks = split_documents(documents)

    # Create FAISS database
    create_vectorstore(chunks)

    print("\nRAG document processing completed successfully!")
    print("FAISS vector database is ready.")