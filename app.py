import os
import time
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
from google import genai

from main import load_vectorstore


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind RAG | AI Document Q&A",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.5-flash"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is not configured."
    )

    st.info(
        "Please add GOOGLE_API_KEY to your "
        "environment variables."
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(91,112,255,.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 100% 10%,
            rgba(155,89,255,.10),
            transparent 28%
        ),
        #080b12;
    color: #eef2f8;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    padding: 20px;
    min-height: 105px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(255,255,255,.035);
    box-sizing: border-box;
}

.metric-label {
    color: #7f8a9d;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 10px;
    font-size: 20px;
    font-weight: 700;
    color: #eef2f8;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

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


/* =========================================================
   PIPELINE
   ========================================================= */

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


/* =========================================================
   TECHNOLOGY PILLS
   ========================================================= */

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


/* =========================================================
   ANSWER
   ========================================================= */

.answer-card {
    padding: 26px 28px;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 18px;
    background: rgba(255,255,255,.035);
    box-shadow: 0 18px 50px rgba(0,0,0,.18);
}


/* =========================================================
   SOURCE
   ========================================================= */

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


/* =========================================================
   QUESTION INPUT
   ========================================================= */

div[data-testid="stTextArea"] textarea {
    color: #111827 !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #070a10;
    border-right: 1px solid rgba(255,255,255,.07);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD VECTOR STORE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return load_vectorstore()


try:

    vectorstore = get_vectorstore()

    vector_ready = True

    vector_error = None

except Exception as e:

    vectorstore = None

    vector_ready = False

    vector_error = str(e)


# ============================================================
# DOCUMENT STATISTICS
# ============================================================

def get_document_stats():

    pdf_files = list(
        Path("documents").glob("*.pdf")
    )

    return (
        len(pdf_files),
        [
            pdf.name
            for pdf in pdf_files
        ],
    )


# ============================================================
# UNIQUE SOURCES
# ============================================================

def unique_sources(documents):

    seen = set()
    results = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page, int):
            page += 1

        key = f"{source}|{page}"

        if key not in seen:

            seen.add(key)

            results.append(
                (
                    source,
                    page
                )
            )

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="
            font-size:23px;
            font-weight:750;
            color:#eef2f8;
        ">
            ◆ DocuMind RAG
        </div>

        <div style="
            color:#7f8a9d;
            font-size:12px;
            margin-top:5px;
        ">
            AI Document Intelligence
        </div>
        """
    )

    st.markdown("### System")

    if vector_ready:

        st.success(
            "✓ FAISS vector store loaded"
        )

    else:

        st.error(
            "× Vector store unavailable"
        )

        with st.expander(
            "Technical error"
        ):

            st.write(vector_error)


    st.success(
        "✓ Gemini API configured"
    )

    st.success(
        "✓ MiniLM embeddings configured"
    )


    st.markdown("---")

    st.markdown(
        "### Retrieval Controls"
    )


    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=4,
    )


    show_context = st.toggle(
        "Show retrieved context",
        value=False,
    )


    st.markdown("---")

    st.markdown(
        "### Architecture"
    )


    pipeline = [
        "▣ PDF ingestion",
        "▣ Text chunking",
        "▣ MiniLM embeddings",
        "▣ FAISS similarity search",
        "◆ Gemini AI",
        "✓ Grounded response",
    ]


    for i, step in enumerate(pipeline):

        st.html(
            f"""
            <div class="pipeline-step">
                {step}
            </div>
            """
        )

        if i < len(pipeline) - 1:

            st.html(
                """
                <div class="pipeline-arrow">
                    ↓
                </div>
                """
            )


    st.markdown("---")

    st.markdown(
        "### Technology Stack"
    )


    technologies = [
        "Python",
        "LangChain",
        "FAISS",
        "Sentence Transformers",
        "Hugging Face",
        "Gemini API",
        "Streamlit",
    ]


    tech_html = "".join(
        f"""
        <span class="tech-pill">
            {tech}
        </span>
        """
        for tech in technologies
    )


    st.html(
        tech_html
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div style="
        padding:34px 38px;
        border-radius:22px;
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,.065),
                rgba(255,255,255,.018)
            );
        border:1px solid rgba(255,255,255,.10);
        box-shadow:0 24px 70px rgba(0,0,0,.28);
        margin-bottom:22px;
    ">

        <div style="
            color:#8fa0ff;
            font-size:12px;
            font-weight:700;
            letter-spacing:1.8px;
            text-transform:uppercase;
            margin-bottom:10px;
        ">
            Retrieval-Augmented Generation
        </div>

        <div style="
            color:#eef2f8;
            font-size:40px;
            line-height:1.1;
            font-weight:800;
            letter-spacing:-1.4px;
            margin-bottom:14px;
        ">
            ◆ AI-Powered Document Q&amp;A
        </div>

        <div style="
            color:#aeb8c9;
            font-size:15px;
            line-height:1.7;
            max-width:900px;
        ">
            Ask questions about your PDF documents
            using semantic search, FAISS retrieval,
            and Gemini AI.
        </div>

    </div>
    """
)


# ============================================================
# METRICS
# ============================================================

pdf_count, pdf_names = get_document_stats()

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Knowledge Base
            </div>

            <div class="metric-value">
                {pdf_count} PDF
            </div>

        </div>
        """
    )


with m2:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Vector Engine
            </div>

            <div class="metric-value">
                FAISS
            </div>

        </div>
        """
    )


with m3:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                Embeddings
            </div>

            <div class="metric-value">
                MiniLM-L6-v2
            </div>

        </div>
        """
    )


with m4:

    st.html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                LLM
            </div>

            <div class="metric-value">
                {LLM_MODEL}
            </div>

        </div>
        """
    )


# ============================================================
# KNOWLEDGE BASE DETAILS
# ============================================================

if pdf_names:

    with st.expander(
        "▣ Knowledge Base Details"
    ):

        st.write(
            "Indexed PDF documents:"
        )

        for name in pdf_names:

            st.write(
                f"• {name}"
            )


# ============================================================
# QUESTION AREA
# ============================================================

st.html(
    """
    <div class="section-title">
        ⌕ Ask your document
    </div>
    """
)


st.html(
    """
    <div class="small-note">
        Ask a factual, conceptual, or coding
        question. The system retrieves relevant
        PDF chunks and sends the context to Gemini.
    </div>
    """
)


question = st.text_area(
    "Question",
    placeholder=(
        "Example: What is indentation "
        "and why is it important in Python?"
    ),
    height=110,
    label_visibility="collapsed",
)


c1, c2, c3 = st.columns(
    [1, 1, 2]
)


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

        st.warning(
            "Please enter a question."
        )

        st.stop()


    if not vector_ready:

        st.error(
            "FAISS vector store is unavailable."
        )

        st.stop()


    # ========================================================
    # RETRIEVAL
    # ========================================================

    try:

        retrieval_start = (
            time.perf_counter()
        )


        with st.spinner(
            "Searching your documents..."
        ):

            scored_documents = (
                vectorstore
                .similarity_search_with_score(
                    question,
                    k=top_k,
                )
            )


        documents = [
            document
            for document, score
            in scored_documents
        ]


        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )


        if not documents:

            st.warning(
                "No relevant information "
                "was found in the PDF."
            )

            st.stop()


        # ====================================================
        # CONTEXT
        # ====================================================

        context = "\n\n".join(
            f"""
SOURCE {i}
PAGE: {document.metadata.get(
    'page',
    'Unknown'
)}

{document.page_content}
"""
            for i, document
            in enumerate(
                documents,
                start=1,
            )
        )


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
You are a professional document
question-answering assistant.

Answer the user's question using
the retrieved PDF context.

RETRIEVED CONTEXT
=================

{context}

USER QUESTION
=============

{question}

RULES
=====

1. Use the retrieved context whenever
   it contains relevant information.

2. If the PDF contains only a question
   and not its solution, solve it yourself.

3. Clearly distinguish generated solutions
   from information explicitly found
   in the document.

4. For Python questions, provide complete
   executable Python code.

5. Explain important concepts briefly.

6. Do not invent facts about the PDF.

7. Give a direct, interview-ready answer.
"""


        # ====================================================
        # GEMINI GENERATION
        # ====================================================

        answer = None

        generation_start = (
            time.perf_counter()
        )


        with st.spinner(
            "Generating answer with Gemini..."
        ):

            for attempt in range(3):

                try:

                    response = (
                        client
                        .models
                        .generate_content(
                            model=LLM_MODEL,
                            contents=prompt,
                        )
                    )


                    answer = response.text

                    break


                except Exception as e:

                    error_message = str(e)


                    temporary_error = (
                        "502" in error_message
                        or "503" in error_message
                        or "504" in error_message
                        or "UNAVAILABLE"
                        in error_message
                        or "INTERNAL"
                        in error_message
                    )


                    if (
                        temporary_error
                        and attempt < 2
                    ):

                        wait_time = (
                            2 ** attempt
                        )

                        time.sleep(
                            wait_time
                        )

                        continue


                    st.error(
                        "Gemini could not "
                        "generate the answer."
                    )


                    with st.expander(
                        "Technical error"
                    ):

                        st.code(
                            error_message
                        )


                    st.stop()


        generation_time = (
            time.perf_counter()
            - generation_start
        )


        if not answer:

            st.error(
                "Gemini returned an empty answer."
            )

            st.stop()


        total_time = (
            retrieval_time
            + generation_time
        )


        # ====================================================
        # RESPONSE METRICS
        # ====================================================

        st.markdown("---")


        r1, r2, r3, r4 = st.columns(4)


        with r1:

            st.metric(
                "Retrieved Chunks",
                len(documents),
            )


        with r2:

            st.metric(
                "Source Pages",
                len(
                    unique_sources(
                        documents
                    )
                ),
            )


        with r3:

            st.metric(
                "Retrieval",
                f"{retrieval_time:.2f}s",
            )


        with r4:

            st.metric(
                "Total Response",
                f"{total_time:.2f}s",
            )


        # ====================================================
        # ANSWER
        # ====================================================

        st.html(
            """
            <div class="section-title">
                ◆ Generated Answer
            </div>
            """
        )


        st.html(
            """
            <div class="answer-card">
            """
        )

        st.markdown(
            answer
        )

        st.html(
            """
            </div>
            """
        )


        # ====================================================
        # SOURCES
        # ====================================================

        st.html(
            """
            <div class="section-title">
                ▣ Retrieved Sources
            </div>
            """
        )


        for source, page in unique_sources(
            documents
        ):

            st.html(
                f"""
                <div class="source-card">

                    <div class="source-meta">
                        Source
                    </div>

                    <div class="source-name">
                        ▣ {source}
                    </div>

                    <div class="source-meta"
                         style="margin-top:9px;">
                        Page
                    </div>

                    <div class="source-name">
                        {page}
                    </div>

                </div>
                """
            )


        # ====================================================
        # RETRIEVAL EVIDENCE
        # ====================================================

        with st.expander(
            "▣ Retrieval Evidence"
        ):

            for rank, (
                document,
                score
            ) in enumerate(
                scored_documents,
                start=1,
            ):

                page = document.metadata.get(
                    "page",
                    "Unknown",
                )


                if isinstance(page, int):

                    page += 1


                st.markdown(
                    f"""
                    **Rank {rank} ·
                    Page {page} ·
                    FAISS distance:
                    {score:.4f}**
                    """
                )


                preview = (
                    document.page_content[:500]
                    .replace(
                        "\n",
                        " "
                    )
                )


                if len(
                    document.page_content
                ) > 500:

                    preview += "..."


                st.caption(
                    preview
                )


        # ====================================================
        # FULL CONTEXT
        # ====================================================

        if show_context:

            with st.expander(
                "▣ Full Retrieved Context"
            ):

                for i, document in enumerate(
                    documents,
                    start=1,
                ):

                    page = document.metadata.get(
                        "page",
                        "Unknown",
                    )


                    if isinstance(
                        page,
                        int,
                    ):

                        page += 1


                    st.markdown(
                        f"""
                        **Chunk {i} · Page {page}**
                        """
                    )


                    st.write(
                        document.page_content
                    )


    except Exception as e:

        st.error(
            "The RAG pipeline could not "
            "complete the request."
        )


        with st.expander(
            "Technical error"
        ):

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        text-align:center;
        color:#626d7f;
        font-size:11px;
        padding:35px 0 8px;
    ">
        DocuMind RAG · PDF ingestion ·
        embeddings · FAISS retrieval ·
        Gemini generation
    </div>
    """
)