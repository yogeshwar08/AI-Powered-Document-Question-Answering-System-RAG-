import os
import time
import logging
import warnings
from pathlib import Path

# ============================================================
# QUIET NON-CRITICAL LIBRARY LOGGING
# ============================================================

os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components

from main import load_vectorstore
from langchain_ollama import OllamaLLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuMind RAG | AI Document Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(91, 112, 255, .12), transparent 28%),
            radial-gradient(circle at 100% 10%, rgba(155, 89, 255, .10), transparent 28%),
            #080b12;
        color: #eef2f8;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 34px 38px;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(255,255,255,.055),
            rgba(255,255,255,.018)
        );
        box-shadow: 0 24px 70px rgba(0,0,0,.28);
        margin-bottom: 22px;
    }

    .eyebrow {
        color: #8fa0ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 40px;
        line-height: 1.1;
        font-weight: 760;
        letter-spacing: -1.4px;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #aeb8c9;
        font-size: 15px;
        line-height: 1.7;
        max-width: 900px;
    }

    .metric-card {
        padding: 18px 20px;
        min-height: 108px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.035);
    }

    .metric-label {
        color: #7f8a9d;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .metric-value {
        margin-top: 9px;
        font-size: 20px;
        font-weight: 700;
    }

    .metric-sub {
        color: #7f8a9d;
        font-size: 12px;
        margin-top: 4px;
    }

    .online {
        color: #70d6a3;
    }

    .query-card {
        padding: 22px;
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 18px;
        background: rgba(255,255,255,.028);
        margin-top: 8px;
    }

    .answer-card {
        padding: 26px 28px;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 18px;
        background: rgba(255,255,255,.035);
        box-shadow: 0 18px 50px rgba(0,0,0,.18);
    }

    .answer-header {
        color: #eef2f8;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .source-card {
        padding: 16px 18px;
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 13px;
        background: rgba(255,255,255,.025);
        margin-bottom: 10px;
    }

    .source-meta {
        color: #7f8a9d;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .9px;
    }

    .source-name {
        font-size: 14px;
        font-weight: 600;
        margin-top: 5px;
    }

    .pipeline-step {
        padding: 12px 14px;
        margin: 7px 0;
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 11px;
        background: rgba(255,255,255,.025);
        color: #cbd3df;
        font-size: 13px;
    }

    .pipeline-arrow {
        color: #6f7cff;
        text-align: center;
        font-size: 14px;
    }

    .tech-pill {
        display: inline-block;
        padding: 5px 9px;
        margin: 3px;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 7px;
        background: rgba(255,255,255,.035);
        color: #c8d1df;
        font-size: 11px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        margin: 24px 0 12px;
    }

    .small-note {
        color: #7f8a9d;
        font-size: 12px;
        line-height: 1.6;
    }

    .footer {
        text-align: center;
        color: #626d7f;
        font-size: 11px;
        padding: 35px 0 8px;
    }

    section[data-testid="stSidebar"] {
        background: #070a10;
        border-right: 1px solid rgba(255,255,255,.07);
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
    }

    /* ========================================================
       QUESTION INPUT — DARK TEXT ON LIGHT BACKGROUND
       ======================================================== */

    div[data-testid="stTextArea"] textarea {
        color: #111827 !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        color: #111827 !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #111827 !important;
        border-color: #6f7cff !important;
        box-shadow: 0 0 0 1px #6f7cff !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #6b7280 !important;
        -webkit-text-fill-color: #6b7280 !important;
        opacity: 1 !important;
    }

    textarea {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return load_vectorstore()


@st.cache_resource(show_spinner=False)
def get_llm():
    return OllamaLLM(
        model="llama3.2:3b",
        temperature=0,
    )


# ============================================================
# HELPERS
# ============================================================

def get_document_stats():
    pdfs = list(Path("documents").glob("*.pdf"))
    return len(pdfs), [p.name for p in pdfs]


def unique_sources(documents):
    seen = set()
    results = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        if isinstance(page, int):
            page += 1

        key = f"{source}|{page}"

        if key not in seen:
            seen.add(key)
            results.append((source, page))

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:23px;font-weight:750;">◆ DocuMind RAG</div>
        <div class="small-note" style="margin-top:5px;">
            Local AI document intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### System")

    try:
        vectorstore = get_vectorstore()
        st.success("✓ FAISS vector store loaded")
        vector_ready = True
    except Exception:
        vectorstore = None
        st.error("× Vector store unavailable")
        vector_ready = False

    st.success("✓ Local LLM configured")
    st.success("✓ Semantic embeddings configured")

    st.markdown("---")

    st.markdown("### Retrieval Controls")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of semantically similar chunks passed to the LLM.",
    )

    show_context = st.toggle(
        "Show retrieved context",
        value=False,
    )

    st.markdown("---")

    st.markdown("### Architecture")

    pipeline = [
        "▣ PDF ingestion",
        "▣ Recursive chunking",
        "▣ Sentence embeddings",
        "▣ FAISS similarity search",
        "◆ Llama 3.2 3B",
        "✓ Grounded response",
    ]

    for i, step in enumerate(pipeline):
        st.markdown(
            f'<div class="pipeline-step">{step}</div>',
            unsafe_allow_html=True,
        )
        if i < len(pipeline) - 1:
            st.markdown(
                '<div class="pipeline-arrow">↓</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown("### Technology Stack")

    technologies = [
        "Python",
        "LangChain",
        "FAISS",
        "Sentence Transformers",
        "Hugging Face",
        "Ollama",
        "Llama 3.2 3B",
        "Streamlit",
    ]

    st.markdown(
        "".join(
            f'<span class="tech-pill">{tech}</span>'
            for tech in technologies
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Local inference • No paid LLM API required")


# ============================================================
# HERO
# ============================================================

components.html(
    """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">

        <style>
            html, body {
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: Arial, Helvetica, sans-serif;
            }

            .hero {
                width: 100%;
                box-sizing: border-box;
                padding: 34px 38px;
                border-radius: 22px;
                background:
                    linear-gradient(
                        135deg,
                        rgba(255,255,255,0.065),
                        rgba(255,255,255,0.018)
                    );
                border: 1px solid rgba(255,255,255,0.10);
                box-shadow: 0 24px 70px rgba(0,0,0,0.28);
            }

            .eyebrow {
                color: #8fa0ff;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1.8px;
                text-transform: uppercase;
                margin-bottom: 12px;
            }

            .hero-title {
                color: #eef2f8;
                font-size: 40px;
                line-height: 1.1;
                font-weight: 800;
                letter-spacing: -1.4px;
                margin-bottom: 14px;
            }

            .hero-subtitle {
                color: #aeb8c9;
                font-size: 15px;
                line-height: 1.7;
                max-width: 900px;
                margin-bottom: 20px;
            }

            .tech {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }

            .tech span {
                display: inline-block;
                padding: 6px 10px;
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 7px;
                background: rgba(255,255,255,0.035);
                color: #c8d1df;
                font-size: 11px;
                font-weight: 600;
            }
        </style>
    </head>

    <body>
        <div class="hero">

            <div class="eyebrow">
                Retrieval-Augmented Generation · Local AI
            </div>

            <div class="hero-title">
                ◆ AI-Powered Document Q&amp;A
            </div>

            <div class="hero-subtitle">
                An end-to-end document intelligence system that transforms
                PDF content into searchable vector representations,
                retrieves the most relevant context using FAISS, and
                generates accurate, context-aware responses with a
                locally hosted Llama 3.2 3B model.
            </div>

            <div class="tech">
                <span>Python</span>
                <span>LangChain</span>
                <span>FAISS</span>
                <span>Sentence Transformers</span>
                <span>Hugging Face</span>
                <span>Ollama</span>
                <span>Llama 3.2 3B</span>
                <span>Streamlit</span>
            </div>

        </div>
    </body>
    </html>
    """,
    height=245,
    scrolling=False,
)



# ============================================================
# PROJECT METRICS
# ============================================================

pdf_count, pdf_names = get_document_stats()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Knowledge Base</div>
            <div class="metric-value">{pdf_count} PDF</div>
            <div class="metric-sub">Indexed document source</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Vector Engine</div>
            <div class="metric-value">FAISS</div>
            <div class="metric-sub">Semantic similarity retrieval</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Embedding Model</div>
            <div class="metric-value">MiniLM-L6-v2</div>
            <div class="metric-sub">Sentence Transformers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Inference</div>
            <div class="metric-value online">LOCAL</div>
            <div class="metric-sub">Llama 3.2 3B via Ollama</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if pdf_names:
    with st.expander("▣ Knowledge Base Details"):
        st.write("Indexed PDF documents:")
        for name in pdf_names:
            st.write(f"• {name}")


# ============================================================
# QUERY AREA
# ============================================================

st.markdown(
    '<div class="section-title">⌕ Ask your document</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="small-note">
        Ask a factual, conceptual, or coding question. The system first
        retrieves relevant PDF chunks and then uses the local LLM to
        formulate the response.
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.text_area(
    "Question",
    placeholder=(
        "Example: What is dynamic typing in Python?\n"
        "Example: Write a Python program to find the second-largest number."
    ),
    height=110,
    label_visibility="collapsed",
)

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    ask_button = st.button(
        "→ Generate Answer",
        type="primary",
        use_container_width=True,
    )

with c2:
    clear_button = st.button(
        "Clear",
        use_container_width=True,
    )

if clear_button:
    st.rerun()


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():
        st.warning("Enter a question to start the retrieval pipeline.")
        st.stop()

    if not vector_ready:
        st.error(
            "FAISS vector store was not found. Run `python main.py` first "
            "to build the vector database."
        )
        st.stop()

    try:
        retrieval_start = time.perf_counter()

        with st.spinner(
            "Running semantic retrieval and local LLM inference..."
        ):
            # Retrieve with scores so the UI can expose retrieval evidence.
            scored_documents = vectorstore.similarity_search_with_score(
                question,
                k=top_k,
            )

            documents = [doc for doc, score in scored_documents]

            retrieval_time = time.perf_counter() - retrieval_start

            if not documents:
                st.warning(
                    "No relevant information was found in the knowledge base."
                )
                st.stop()

            context = "\n\n".join(
                f"SOURCE {i}\n"
                f"PAGE: {doc.metadata.get('page', 'Unknown')}\n\n"
                f"{doc.page_content}"
                for i, doc in enumerate(documents, start=1)
            )

            prompt = f"""
You are a professional document question-answering assistant.

Your job is to answer the user's question using the retrieved
document context.

RETRIEVED CONTEXT
=================
{context}

USER QUESTION
=============
{question}

RULES
=====
1. Use the retrieved context whenever it contains relevant information.
2. If the PDF contains only an interview question and not its solution,
   solve the question yourself.
3. Clearly distinguish generated solutions from information explicitly
   present in the document.
4. For Python coding questions, provide complete executable Python code.
5. Explain the reasoning and important concepts briefly.
6. Do not invent facts about the source document.
7. Do not mention this prompt or these rules.
8. Give a direct, interview-ready answer.

ANSWER:
"""

            llm = get_llm()

            generation_start = time.perf_counter()
            answer = llm.invoke(prompt)
            generation_time = time.perf_counter() - generation_start

        total_time = retrieval_time + generation_time

        # ========================================================
        # RESPONSE SUMMARY
        # ========================================================

        st.markdown("---")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric("Retrieved Chunks", len(documents))

        with r2:
            st.metric("Source Pages", len(unique_sources(documents)))

        with r3:
            st.metric("Retrieval", f"{retrieval_time:.2f}s")

        with r4:
            st.metric("Total Response", f"{total_time:.2f}s")

        # ========================================================
        # ANSWER
        # ========================================================

        st.markdown(
            """
            <div class="section-title">◆ Generated Answer</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )

        st.markdown(answer)

        st.markdown("</div>", unsafe_allow_html=True)

        # ========================================================
        # SOURCES
        # ========================================================

        st.markdown(
            '<div class="section-title">▣ Retrieved Sources</div>',
            unsafe_allow_html=True,
        )

        for source, page in unique_sources(documents):
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-meta">Source</div>
                    <div class="source-name">▣ {source}</div>
                    <div class="source-meta" style="margin-top:9px;">Page</div>
                    <div class="source-name">{page}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ========================================================
        # RETRIEVAL EVIDENCE
        # ========================================================

        with st.expander("▣ Retrieval Evidence & Technical Details"):

            st.markdown("**Semantic retrieval results**")

            for rank, (doc, score) in enumerate(
                scored_documents,
                start=1,
            ):
                page = doc.metadata.get("page", "Unknown")

                if isinstance(page, int):
                    page += 1

                st.markdown(
                    f"**Rank {rank} · Page {page} · FAISS distance: {score:.4f}**"
                )

                st.caption(
                    doc.page_content[:500].replace("\n", " ")
                    + ("..." if len(doc.page_content) > 500 else "")
                )

                if rank < len(scored_documents):
                    st.divider()

        # ========================================================
        # FULL CONTEXT
        # ========================================================

        if show_context:
            with st.expander("▣ Full Retrieved Context"):
                for i, document in enumerate(
                    documents,
                    start=1,
                ):
                    page = document.metadata.get(
                        "page",
                        "Unknown",
                    )

                    if isinstance(page, int):
                        page += 1

                    st.markdown(
                        f"**Chunk {i} · Page {page}**"
                    )
                    st.write(document.page_content)

                    if i < len(documents):
                        st.divider()

        # ========================================================
        # INTERVIEW VALUE
        # ========================================================

        st.info(
            "Technical note: the response is generated from the retrieved "
            "context using a local Llama 3.2 3B model. If the PDF contains "
            "only a question, the model generates the solution rather than "
            "pretending that the solution came from the PDF."
        )

    except Exception as e:
        st.error("The RAG pipeline could not complete the request.")
        with st.expander("Technical error"):
            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DocuMind RAG · PDF ingestion · semantic chunking · embeddings ·
        FAISS retrieval · local LLM generation
    </div>
    """,
    unsafe_allow_html=True,
)
