# ============================================================
# AI-POWERED DOCUMENT QUESTION ANSWERING
# PROFESSIONAL LANGCHAIN RAG QUERY ENGINE
#
# 6. USER QUERY
# 7. HYBRID RETRIEVAL
# 8. CONTEXT AUGMENTATION
# 9. GEMINI ANSWER GENERATION
#
# Production Features:
# - LangChain LCEL
# - Gemini LLM
# - FAISS retrieval
# - Retry handling
# - Connection error recovery
# - Exponential backoff
# - Source attribution
# - Performance metrics
# ============================================================


import os
import time


# ============================================================
# ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# LANGCHAIN GEMINI
# ============================================================

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)


# ============================================================
# LANGCHAIN PROMPT
# ============================================================

from langchain_core.prompts import (
    ChatPromptTemplate
)


# ============================================================
# LANGCHAIN OUTPUT PARSER
# ============================================================

from langchain_core.output_parsers import (
    StrOutputParser
)


# ============================================================
# PROJECT RAG FUNCTIONS
# ============================================================

from main import (
    retrieve_documents,
    build_context,
    get_sources
)


# ============================================================
# CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)


LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-2.5-flash"
)


# Maximum number of Gemini generation attempts
MAX_LLM_RETRIES = int(
    os.getenv(
        "LLM_MAX_RETRIES",
        "5"
    )
)


# Initial retry delay
RETRY_DELAY = float(
    os.getenv(
        "LLM_RETRY_DELAY",
        "2"
    )
)


# Maximum question length
MAX_QUESTION_LENGTH = 10000


# ============================================================
# API KEY VALIDATION
# ============================================================

if not GOOGLE_API_KEY:

    raise RuntimeError(
        "GOOGLE_API_KEY is not configured. "
        "Add GOOGLE_API_KEY to your .env file."
    )


# ============================================================
# GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4,
    max_retries=2
)


# ============================================================
# STEP 8
# PROFESSIONAL RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are DocuMind, a professional AI-powered
document question-answering assistant.

Your primary responsibility is to provide accurate,
clear, useful, and grounded answers based on the
retrieved document context.

============================================================
RETRIEVED DOCUMENT CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
GROUNDING RULES
============================================================

1. Treat the retrieved document context as the
   primary source of truth.

2. Answer using information that is supported by
   the retrieved context whenever possible.

3. Do NOT invent, fabricate, or assume information
   that is not supported by the retrieved documents.

4. If the retrieved context contains the answer,
   explain it clearly and directly.

5. If the retrieved context contains only a question
   and does not contain its answer, explicitly state:

   "The provided documents contain the question,
   but they do not provide its answer."

   You may then provide a solution using your general
   knowledge, but clearly label it as:

   "General Knowledge / Generated Explanation"

6. If the requested information is not available
   in the retrieved documents, say:

   "The requested information is not available
   in the provided documents."

   Do not pretend that information from general
   knowledge came from the documents.

7. Never claim that a fact, explanation, example,
   or solution came from a document unless it is
   actually supported by the retrieved context.

8. When multiple retrieved documents contain relevant
   information, combine them carefully and identify
   the relevant document sources when appropriate.

9. If the retrieved context is insufficient or
   ambiguous, acknowledge the limitation instead of
   guessing.

============================================================
ANSWER QUALITY
============================================================

10. Give a direct answer to the user's question first.

11. Then provide explanation, examples, or additional
    technical details when useful.

12. Keep the response professional, technically accurate,
    and interview-ready.

13. Use clear headings, bullet points, numbered steps,
    and code blocks when they improve readability.

14. For programming questions, provide complete,
    executable code when appropriate.

15. Explain important technical concepts in simple but
    professional language.

16. For comparison questions, use a table when it
    improves clarity.

17. For mathematical or technical questions, show the
    reasoning or steps clearly when appropriate.

============================================================
PROGRAMMING QUESTIONS
============================================================

When the user asks for programming help:

- Explain the concept briefly.
- Provide correct executable code.
- Explain the important parts of the code.
- Do not invent APIs, libraries, functions, or outputs.
- If the retrieved documents contain a specific
  implementation, prioritize that implementation.

============================================================
SOURCE ATTRIBUTION
============================================================

When the answer is based on retrieved documents,
mention the relevant document name and page number
when that information is available.

Do not create fake citations or page numbers.

============================================================
IMPORTANT BEHAVIOR
============================================================

The goal is NOT to answer every question at any cost.

The goal is to provide the most accurate answer possible
while clearly distinguishing:

A. Information supported by the retrieved documents
B. General knowledge or generated explanations
C. Information that is unavailable from the documents

Never hide uncertainty.
Never fabricate document content.
Never fabricate sources.
Never claim unsupported information is document-derived.

============================================================
FINAL RESPONSE
============================================================

Answer the user's question now.
"""
)

# ============================================================
# STEP 9
# LANGCHAIN LCEL RAG CHAIN
# ============================================================

rag_chain = (

    prompt

    |

    llm

    |

    StrOutputParser()

)


# ============================================================
# GEMINI GENERATION WITH RETRY
# ============================================================

def generate_answer(
    context,
    question
):
    """
    Generate an answer through the LangChain
    Gemini chain with retry and exponential
    backoff for transient network failures.
    """

    last_error = None


    for attempt in range(
        1,
        MAX_LLM_RETRIES + 1
    ):

        try:

            print(
                f"  Gemini generation attempt "
                f"{attempt}/{MAX_LLM_RETRIES}"
            )


            answer = rag_chain.invoke(

                {
                    "context": context,

                    "question": question
                }

            )


            # ------------------------------------------------
            # Validate response
            # ------------------------------------------------

            if answer is None:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )


            answer = str(
                answer
            ).strip()


            if not answer:

                raise RuntimeError(
                    "Gemini returned an empty answer."
                )


            print(
                "  ✓ Gemini answer generated successfully."
            )


            return answer


        except Exception as error:

            last_error = error


            print()
            print(
                f"  ⚠ Gemini generation failed "
                f"on attempt {attempt}/{MAX_LLM_RETRIES}."
            )

            print(
                f"    Error: {error}"
            )


            # ------------------------------------------------
            # Retry if attempts remain
            # ------------------------------------------------

            if attempt < MAX_LLM_RETRIES:

                delay = (
                    RETRY_DELAY
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


            else:

                print(
                    "  ✗ Gemini generation failed "
                    "after all retry attempts."
                )


    # ========================================================
    # FINAL FAILURE
    # ========================================================

    raise RuntimeError(
        "Gemini answer generation failed after "
        f"{MAX_LLM_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def answer_query(
    question,
    top_k=4
):
    """
    Complete Retrieval-Augmented Generation pipeline.

    Flow:

    User Question
          ↓
    Semantic Retrieval
          ↓
    Relevant Documents
          ↓
    Context Augmentation
          ↓
    LangChain LCEL
          ↓
    Gemini
          ↓
    Answer + Sources + Metrics
    """


    # ========================================================
    # STEP 6
    # USER QUERY VALIDATION
    # ========================================================

    if question is None:

        raise ValueError(
            "Question cannot be empty."
        )


    if not isinstance(
        question,
        str
    ):

        raise ValueError(
            "Question must be a string."
        )


    question = question.strip()


    if not question:

        raise ValueError(
            "Question cannot be empty."
        )


    if len(question) > MAX_QUESTION_LENGTH:

        raise ValueError(
            f"Question is too long. "
            f"Maximum length is "
            f"{MAX_QUESTION_LENGTH} characters."
        )


    # ========================================================
    # TOP-K VALIDATION
    # ========================================================

    try:

        top_k = int(
            top_k
        )

    except (
        TypeError,
        ValueError
    ):

        top_k = 4


    top_k = max(
        1,
        min(
            top_k,
            10
        )
    )


    # ========================================================
    # TOTAL TIMER
    # ========================================================

    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # STEP 7
    # HYBRID RETRIEVAL
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "STEP 7: SEMANTIC RETRIEVAL"
    )

    print(
        "=" * 60
    )


    retrieval_start = (
        time.perf_counter()
    )


    documents = retrieve_documents(

        question,

        k=top_k

    )


    retrieval_time = (

        time.perf_counter()

        -

        retrieval_start

    )


    print(
        f"Relevant chunks retrieved: "
        f"{len(documents)}"
    )


    # ========================================================
    # PRINT RETRIEVED SOURCES
    # ========================================================

    for index, document in enumerate(

        documents,

        start=1

    ):

        metadata = (
            getattr(
                document,
                "metadata",
                {}
            )
            or {}
        )


        source = metadata.get(
            "source",
            metadata.get(
                "file_name",
                "Unknown"
            )
        )


        page = metadata.get(
            "page",
            metadata.get(
                "page_number",
                "Unknown"
            )
        )


        print(
            f"  {index}. "
            f"{source} | Page {page}"
        )


    # ========================================================
    # NO DOCUMENTS FOUND
    # ========================================================

    if not documents:

        total_time = (

            time.perf_counter()

            -

            total_start

        )


        return {

            "answer":
                "No relevant information "
                "was found in the knowledge base.",

            "documents": [],

            "sources": [],

            "retrieval_time":
                retrieval_time,

            "generation_time":
                0.0,

            "total_time":
                total_time

        }


    # ========================================================
    # STEP 8
    # CONTEXT AUGMENTATION
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "STEP 8: CONTEXT AUGMENTATION"
    )

    print(
        "=" * 60
    )


    context = build_context(
        documents
    )


    print(
        f"Context created from "
        f"{len(documents)} chunks."
    )


    print(
        f"Context characters: "
        f"{len(context)}"
    )


    # ========================================================
    # STEP 9
    # ANSWER GENERATION
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "STEP 9: ANSWER GENERATION"
    )

    print(
        "=" * 60
    )


    generation_start = (
        time.perf_counter()
    )


    try:

        answer = generate_answer(

            context,

            question

        )


    except Exception as error:

        generation_time = (

            time.perf_counter()

            -

            generation_start

        )


        total_time = (

            time.perf_counter()

            -

            total_start

        )


        # ----------------------------------------------------
        # Return structured error instead of crashing
        # ----------------------------------------------------

        return {

            "answer":
                "I could not generate the answer "
                "because the Gemini service was "
                "temporarily unavailable. "
                "Please try again.",

            "documents":
                documents,

            "sources":
                get_sources(
                    documents
                ),

            "retrieval_time":
                retrieval_time,

            "generation_time":
                generation_time,

            "total_time":
                total_time,

            "error":
                str(error)

        }


    generation_time = (

        time.perf_counter()

        -

        generation_start

    )


    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_time = (

        time.perf_counter()

        -

        total_start

    )


    # ========================================================
    # SOURCES
    # ========================================================

    sources = get_sources(
        documents
    )


    # ========================================================
    # COMPLETE RESULT
    # ========================================================

    return {

        "answer":
            answer,

        "documents":
            documents,

        "sources":
            sources,

        "retrieval_time":
            retrieval_time,

        "generation_time":
            generation_time,

        "total_time":
            total_time,

        "top_k":
            top_k,

        "llm_model":
            LLM_MODEL,

        "rag_framework":
            "LangChain",

        "vector_database":
            "FAISS + BM25",

        "embedding_model":
            "gemini-embedding-2"

    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":


    print()
    print(
        "=" * 60
    )

    print(
        "       DOCUMIND"
    )

    print(
        "       AI-POWERED DOCUMENT Q&A"
    )

    print(
        "       LANGCHAIN RAG SYSTEM"
    )

    print(
        "=" * 60
    )


    print()
    print(
        f"LLM Model       : {LLM_MODEL}"
    )

    print(
        "Embedding Model : gemini-embedding-2"
    )

    print(
        "Vector Database : FAISS"
    )

    print(
        "Framework       : LangChain"
    )

    print(
        "=" * 60
    )


    question = input(
        "\nAsk a question about your PDF: "
    ).strip()


    if not question:

        print(
            "\nPlease enter a question."
        )

        raise SystemExit(
            0
        )


    try:

        result = answer_query(

            question,

            top_k=4

        )


        # ====================================================
        # ANSWER
        # ====================================================

        print()
        print(
            "=" * 60
        )

        print(
            "                    ANSWER"
        )

        print(
            "=" * 60
        )


        print(
            result.get(
                "answer",
                "No answer generated."
            )
        )


        # ====================================================
        # SOURCES
        # ====================================================

        print()
        print(
            "=" * 60
        )

        print(
            "                    SOURCES"
        )

        print(
            "=" * 60
        )


        sources = result.get(
            "sources",
            []
        )


        if sources:

            for source, page in sources:

                print(
                    f"{source} - "
                    f"Page {page}"
                )

        else:

            print(
                "No sources available."
            )


        # ====================================================
        # METRICS
        # ====================================================

        print()
        print(
            "=" * 60
        )

        print(
            "                    METRICS"
        )

        print(
            "=" * 60
        )


        print(
            f"Retrieval Time : "
            f"{result.get('retrieval_time', 0):.3f}s"
        )


        print(
            f"Generation Time: "
            f"{result.get('generation_time', 0):.3f}s"
        )


        print(
            f"Total Time     : "
            f"{result.get('total_time', 0):.3f}s"
        )


        print(
            f"LLM            : "
            f"{result.get('llm_model', LLM_MODEL)}"
        )


        print(
            "RAG Framework  : LangChain"
        )


        print(
            "Vector Retrieval: FAISS + BM25"
        )


        # ====================================================
        # ERROR INFORMATION
        # ====================================================

        if result.get(
            "error"
        ):

            print()
            print(
                "Generation service error:"
            )

            print(
                result["error"]
            )


    except KeyboardInterrupt:

        print(
            "\n\nOperation cancelled by user."
        )


    except Exception as error:

        print()
        print(
            "=" * 60
        )

        print(
            "RAG PIPELINE ERROR"
        )

        print(
            "=" * 60
        )

        print(
            str(error)
        )
