# ============================================================
# DOCUMIND
# AI-POWERED DOCUMENT QUESTION ANSWERING SYSTEM
#
# FLASK APPLICATION
#
# RAG PIPELINE
# 1. KNOWLEDGE BASE
# 2. PARSING
# 3. CHUNKING
# 4. EMBEDDING
# 5. FAISS VECTOR DATABASE
# 6. USER QUERY
# 7. RETRIEVAL
# 8. AUGMENTATION
# 9. ANSWER GENERATION
# ============================================================


# ============================================================
# STANDARD LIBRARY
# ============================================================

import os
import time
import traceback
from pathlib import Path


# ============================================================
# ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# FLASK
# ============================================================

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)


# ============================================================
# PROJECT MODULES
# ============================================================

from main import (
    get_document_stats,
)

from query import (
    answer_query,
)


# ============================================================
# APPLICATION
# ============================================================

app = Flask(
    __name__
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
# APPLICATION CONFIGURATION
# ============================================================

app.config[
    "MAX_CONTENT_LENGTH"
] = 16 * 1024 * 1024


# ============================================================
# MODEL INFORMATION
# ============================================================

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# STEP 1
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
#
# IMPORTANT:
# DO NOT LOAD FAISS OR HUGGINGFACE HERE.
#
# Render can call this endpoint to verify that the
# Flask application is alive.
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "application":
            "DocuMind",

        "description":
            "AI-Powered Document Question Answering System",

        "framework":
            "LangChain",

        "web_framework":
            "Flask",

        "retrieval":
            "FAISS",

        "embedding_model":
            "sentence-transformers/all-MiniLM-L6-v2",

        "llm":
            LLM_MODEL,

        "documents":
            "Multi-document",

        "rag":
            True

    }), 200


# ============================================================
# KNOWLEDGE BASE STATISTICS
#
# This only reads the PDF filenames.
#
# It DOES NOT load:
# - Embedding model
# - FAISS
# - PyTorch
# - Gemini
# ============================================================

@app.route(
    "/api/stats",
    methods=["GET"]
)
def document_stats():

    try:

        document_count, documents = (
            get_document_stats()
        )

        return jsonify({

            "success":
                True,

            "document_count":
                document_count,

            "documents":
                documents

        }), 200

    except Exception as error:

        print(
            "Statistics error:",
            error
        )

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                "Unable to read knowledge base statistics."

        }), 500


# ============================================================
# RAG QUESTION ANSWERING
#
# POST /api/ask
#
# JSON:
#
# {
#     "question": "What is type casting?",
#     "top_k": 4
# }
#
# ============================================================

@app.route(
    "/api/ask",
    methods=["POST"]
)
def ask_question():

    total_start = (
        time.perf_counter()
    )

    try:

        # ====================================================
        # READ REQUEST
        # ====================================================

        data = (
            request.get_json(
                silent=True
            )
        )

        if not isinstance(
            data,
            dict
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "Request body must contain valid JSON."

            }), 400


        # ====================================================
        # GET QUESTION
        # ====================================================

        question = data.get(
            "question"
        )


        if question is None:

            return jsonify({

                "success":
                    False,

                "error":
                    "Question is required."

            }), 400


        if not isinstance(
            question,
            str
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "Question must be a string."

            }), 400


        question = (
            question.strip()
        )


        if not question:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please enter a question."

            }), 400


        # ====================================================
        # QUESTION LENGTH PROTECTION
        # ====================================================

        if len(question) > 5000:

            return jsonify({

                "success":
                    False,

                "error":
                    "Question exceeds the maximum allowed length."

            }), 400


        # ====================================================
        # TOP-K
        # ====================================================

        top_k = data.get(
            "top_k",
            4
        )


        try:

            top_k = int(
                top_k
            )

        except (
            TypeError,
            ValueError
        ):

            top_k = 4


        # ====================================================
        # KEEP RETRIEVAL SIZE REASONABLE
        # ====================================================

        top_k = max(
            1,
            min(
                top_k,
                10
            )
        )


        # ====================================================
        # LOG QUERY
        # ====================================================

        print()
        print(
            "=" * 65
        )

        print(
            "NEW RAG REQUEST"
        )

        print(
            "=" * 65
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Top-K: {top_k}"
        )


        # ====================================================
        # EXECUTE RAG PIPELINE
        # ====================================================

        result = (
            answer_query(
                question,
                top_k=top_k
            )
        )


        # ====================================================
        # RESPONSE TIME
        # ====================================================

        total_time = (
            time.perf_counter()
            -
            total_start
        )


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if not isinstance(
            result,
            dict
        ):

            result = {

                "answer":
                    str(result),

                "sources":
                    [],

                "documents":
                    []

            }


        # ====================================================
        # SERIALIZATION
        #
        # IMPORTANT:
        #
        # LangChain Document objects cannot be directly
        # serialized by Flask's jsonify().
        #
        # Convert them into clean JSON dictionaries.
        # ====================================================

        documents = (
            result.get(
                "documents",
                []
            )
        )


        serialized_documents = []


        for document in documents:

            try:

                metadata = (
                    getattr(
                        document,
                        "metadata",
                        {}
                    )
                )

                content = (
                    getattr(
                        document,
                        "page_content",
                        ""
                    )
                )

                serialized_documents.append({

                    "content":
                        content,

                    "metadata":
                        metadata

                })

            except Exception:

                continue


        # ====================================================
        # SOURCES
        # ====================================================

        sources = (
            result.get(
                "sources",
                []
            )
        )


        serialized_sources = []


        for source in sources:

            try:

                if (
                    isinstance(
                        source,
                        (list, tuple)
                    )
                    and
                    len(source) >= 2
                ):

                    serialized_sources.append({

                        "document":
                            str(source[0]),

                        "page":
                            source[1]

                    })

            except Exception:

                continue


        # ====================================================
        # FINAL API RESPONSE
        # ====================================================

        response = {

            "success":
                True,

            "answer":
                result.get(
                    "answer",
                    ""
                ),

            "sources":
                serialized_sources,

            "documents":
                serialized_documents,

            "retrieval_time":
                round(
                    float(
                        result.get(
                            "retrieval_time",
                            0
                        )
                    ),
                    3
                ),

            "generation_time":
                round(
                    float(
                        result.get(
                            "generation_time",
                            0
                        )
                    ),
                    3
                ),

            "total_time":
                round(
                    total_time,
                    3
                ),

            "top_k":
                top_k,

            "llm":
                LLM_MODEL

        }


        # ====================================================
        # LOG RESULT
        # ====================================================

        print()
        print(
            f"Retrieved chunks: "
            f"{len(serialized_documents)}"
        )

        print(
            f"Sources: "
            f"{len(serialized_sources)}"
        )

        print(
            f"Total response time: "
            f"{total_time:.3f}s"
        )


        return jsonify(
            response
        ), 200


    # ========================================================
    # VALIDATION ERROR
    # ========================================================

    except ValueError as error:

        print(
            "Validation error:",
            error
        )

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 400


    # ========================================================
    # GOOGLE / LANGCHAIN / RAG ERRORS
    # ========================================================

    except Exception as error:

        print()
        print(
            "=" * 65
        )

        print(
            "RAG APPLICATION ERROR"
        )

        print(
            "=" * 65
        )

        print(
            f"Error: {error}"
        )

        traceback.print_exc()


        return jsonify({

            "success":
                False,

            "error":
                "Unable to generate an answer. "
                "Please try again."

        }), 500


# ============================================================
# OPTIONAL API INFORMATION
# ============================================================

@app.route(
    "/api/info",
    methods=["GET"]
)
def api_info():

    return jsonify({

        "application":
            "DocuMind",

        "version":
            "1.0.0",

        "framework":
            "Flask",

        "rag_framework":
            "LangChain",

        "vector_database":
            "FAISS",

        "embedding_model":
            "sentence-transformers/all-MiniLM-L6-v2",

        "llm":
            LLM_MODEL,

        "endpoints": {

            "home":
                "GET /",

            "health":
                "GET /health",

            "statistics":
                "GET /api/stats",

            "question_answering":
                "POST /api/ask",

            "information":
                "GET /api/info"

        }

    }), 200


# ============================================================
# 404 HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "API endpoint not found."

        }), 404


    return (
        render_template(
            "index.html"
        ),
        404
    )


# ============================================================
# 405 HANDLER
# ============================================================

@app.errorhandler(405)
def method_not_allowed(error):

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "HTTP method not allowed."

        }), 405


    return jsonify({

        "success":
            False,

        "error":
            "Method not allowed."

    }), 405


# ============================================================
# 413 HANDLER
# ============================================================

@app.errorhandler(413)
def request_too_large(error):

    return jsonify({

        "success":
            False,

        "error":
            "Request payload is too large."

    }), 413


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.errorhandler(Exception)
def handle_exception(error):

    print()
    print(
        "Unhandled application error:"
    )

    print(
        error
    )

    traceback.print_exc()


    return jsonify({

        "success":
            False,

        "error":
            "Internal server error."

    }), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    print()
    print(
        "=" * 65
    )

    print(
        "DOCUMIND RAG SERVER"
    )

    print(
        "=" * 65
    )

    print(
        f"Running on port: {port}"
    )

    print(
        "Debug mode: OFF"
    )

    print(
        "=" * 65
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )