# ============================================================
# AI-POWERED DOCUMENT QUESTION ANSWERING
# RAG QUERY ENGINE
#
# 6. USER QUERY
# 7. RETRIEVAL
# 8. AUGMENTATION
# 9. GENERATE ANSWER
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
# OUTPUT PARSER
# ============================================================

from langchain_core.output_parsers import (
    StrOutputParser
)


# ============================================================
# PROJECT FUNCTIONS
# ============================================================

from main import (
    retrieve_documents,
    build_context,
    get_sources
)


# ============================================================
# API SETTINGS
# ============================================================

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)


LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not GOOGLE_API_KEY:

    raise RuntimeError(
        "GOOGLE_API_KEY is not configured."
    )


# ============================================================
# GEMINI LLM
# ============================================================

llm = (
    ChatGoogleGenerativeAI(

        model=LLM_MODEL,

        google_api_key=(
            GOOGLE_API_KEY
        ),

        temperature=0
    )
)


# ============================================================
# STEP 8
# AUGMENTATION PROMPT
# ============================================================

prompt = (
    ChatPromptTemplate.from_template(
        """
You are DocuMind, a professional
AI-powered document question-answering
assistant.

Your job is to answer the user's
question using the retrieved document
context.

==============================
RETRIEVED DOCUMENT CONTEXT
==============================

{context}

==============================
USER QUESTION
==============================

{question}

==============================
INSTRUCTIONS
==============================

1. Use the retrieved document context
   whenever relevant.

2. Do not invent facts about the
   provided documents.

3. If the answer is not available
   in the retrieved context, clearly
   say that the information is not
   available in the provided documents.

4. If the document contains a question
   but not its solution, you may solve
   the question using your knowledge,
   but clearly distinguish the generated
   solution from information found
   in the document.

5. For programming questions, provide
   complete executable Python code
   when appropriate.

6. Explain important technical concepts
   clearly and professionally.

7. Give a direct, technically accurate
   and interview-ready response.

8. Do not claim that generated information
   came from the document.

==============================
FINAL ANSWER
==============================
"""
    )
)


# ============================================================
# STEP 9
# LANGCHAIN RAG CHAIN
# ============================================================

rag_chain = (

    prompt

    |

    llm

    |

    StrOutputParser()

)


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def answer_query(
    question,
    top_k=4
):

    # ========================================================
    # STEP 6
    # USER QUERY
    # ========================================================

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    question = (
        question.strip()
    )

    # Start total timer

    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # STEP 7
    # SEMANTIC RETRIEVAL
    # ========================================================

    retrieval_start = (
        time.perf_counter()
    )

    documents = (
        retrieve_documents(
            question,
            k=top_k
        )
    )

    retrieval_time = (
        time.perf_counter()
        -
        retrieval_start
    )


    # ========================================================
    # CHECK RETRIEVAL
    # ========================================================

    if not documents:

        return {

            "answer":
                "No relevant information "
                "was found in the knowledge base.",

            "documents": [],

            "sources": [],

            "retrieval_time":
                retrieval_time,

            "generation_time": 0,

            "total_time":
                time.perf_counter()
                -
                total_start
        }


    # ========================================================
    # STEP 8
    # AUGMENTATION
    # ========================================================

    context = (
        build_context(
            documents
        )
    )


    # ========================================================
    # STEP 9
    # GENERATE ANSWER
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 9: ANSWER GENERATION")
    print("=" * 60)

    generation_start = (
        time.perf_counter()
    )

    answer = (
        rag_chain.invoke(

            {
                "context":
                    context,

                "question":
                    question
            }

        )
    )

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
    # RETURN COMPLETE RESULT
    # ========================================================

    return {

        "answer":
            answer,

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
            total_time
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       AI-POWERED DOCUMENT Q&A")
    print("       LANGCHAIN RAG SYSTEM")
    print("=" * 60)

    question = input(
        "\nAsk a question about your PDF: "
    ).strip()

    if not question:

        print(
            "\nPlease enter a question."
        )

        raise SystemExit


    try:

        result = (
            answer_query(
                question,
                top_k=4
            )
        )


        print()
        print("=" * 60)
        print("                    ANSWER")
        print("=" * 60)

        print(
            result["answer"]
        )


        print()
        print("=" * 60)
        print("                    SOURCES")
        print("=" * 60)


        for source, page in (
            result["sources"]
        ):

            print(
                f"{source} - "
                f"Page {page}"
            )


        print()
        print("=" * 60)
        print("                    METRICS")
        print("=" * 60)

        print(
            f"Retrieval Time: "
            f"{result['retrieval_time']:.3f}s"
        )

        print(
            f"Generation Time: "
            f"{result['generation_time']:.3f}s"
        )

        print(
            f"Total Response Time: "
            f"{result['total_time']:.3f}s"
        )


    except Exception as e:

        print(
            f"\nRAG pipeline failed: {e}"
        )