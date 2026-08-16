# ============================================================
# DOCUMIND
# AI-POWERED DOCUMENT QUESTION ANSWERING SYSTEM
#
# FLASK APPLICATION
# PRODUCTION-READY RAG WEB API
#
# Features:
#   - Flask web application
#   - PDF document serving
#   - Multi-document knowledge base
#   - FAISS vector retrieval
#   - LangChain RAG pipeline
#   - Google Gemini
#   - Health monitoring
#   - API statistics
#   - Structured error handling
#   - Gunicorn / RunXBuild compatible
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
    send_from_directory,
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
# BASE DIRECTORIES
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


DOCUMENTS_DIR = (
    BASE_DIR /
    "documents"
)


VECTORSTORE_DIR = (
    BASE_DIR /
    "vectorstore"
)


TEMPLATES_DIR = (
    BASE_DIR /
    "templates"
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


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-2"
)


# ============================================================
# APPLICATION INFORMATION
# ============================================================

APP_NAME = (
    "DocuMind"
)


APP_VERSION = (
    "1.0.0"
)


# ============================================================
# STARTUP INFORMATION
# ============================================================

print()
print("=" * 70)
print("DOCUMIND")
print("AI-POWERED DOCUMENT QUESTION ANSWERING SYSTEM")
print("=" * 70)

print(
    f"Base Directory : {BASE_DIR}"
)

print(
    f"Documents      : {DOCUMENTS_DIR}"
)

print(
    f"Vectorstore    : {VECTORSTORE_DIR}"
)

print(
    f"LLM            : {LLM_MODEL}"
)

print(
    f"Embeddings     : {EMBEDDING_MODEL}"
)

print(
    "=" * 70
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
# STEP 2
# PDF DOCUMENT VIEWER
#
# This route allows the frontend to open:
#
# /documents/Python_Interview_Questions.pdf
#
# The browser will display the PDF directly.
# ============================================================

@app.route(
    "/documents/<path:filename>",
    methods=["GET"]
)
def serve_document(
    filename
):

    try:

        # ----------------------------------------------------
        # SECURITY
        # ----------------------------------------------------
        #
        # send_from_directory prevents users from accessing
        # arbitrary files outside the documents directory.
        # ----------------------------------------------------

        requested_file = (
            DOCUMENTS_DIR /
            filename
        )

        if not requested_file.is_file():

            return jsonify({

                "success":
                    False,

                "error":
                    "Document not found."

            }), 404


        # ----------------------------------------------------
        # SERVE PDF
        # ----------------------------------------------------

        return send_from_directory(

            DOCUMENTS_DIR,

            filename,

            mimetype="application/pdf",

            as_attachment=False,

            max_age=3600

        )


    except Exception as error:

        print()
        print(
            "Document serving error:"
        )

        print(
            error
        )

        traceback.print_exc()


        return jsonify({

            "success":
                False,

            "error":
                "Unable to open document."

        }), 500


# ============================================================
# HEALTH CHECK
#
# IMPORTANT:
# This endpoint does NOT load FAISS,
# embeddings, PyTorch or Gemini.
#
# This makes it lightweight for deployment
# health checks.
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
            APP_NAME,

        "version":
            APP_VERSION,

        "description":
            "AI-Powered Document Question Answering System",

        "framework":
            "LangChain",

        "web_framework":
            "Flask",

        "retrieval":
            "FAISS",

        "embedding_model":
            EMBEDDING_MODEL,

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
# GET /api/stats
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
                documents,

            "vector_database":
                "FAISS",

            "framework":
                "LangChain",

            "embedding_model":
                EMBEDDING_MODEL,

            "llm":
                LLM_MODEL

        }), 200


    except Exception as error:

        print()
        print(
            "Statistics error:"
        )

        print(
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
        # READ JSON REQUEST
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
        # QUESTION
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


        # Keep retrieval reasonable

        top_k = max(
            1,
            min(
                top_k,
                10
            )
        )


        # ====================================================
        # LOG REQUEST
        # ====================================================

        print()
        print("=" * 70)

        print(
            "NEW DOCUMIND RAG REQUEST"
        )

        print("=" * 70)

        print(
            f"Question : {question}"
        )

        print(
            f"Top-K    : {top_k}"
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
        # TOTAL RESPONSE TIME
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
                    [],

                "retrieval_time":
                    0,

                "generation_time":
                    0

            }


        # ====================================================
        # SERIALIZE DOCUMENTS
        #
        # LangChain Document objects cannot directly
        # be passed to jsonify().
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
        # SERIALIZE SOURCES
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
                            str(
                                source[0]
                            ),

                        "page":
                            source[1]

                    })

            except Exception:

                continue


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        response = {

            "success":
                True,

            "application":
                APP_NAME,

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
                LLM_MODEL,

            "embedding_model":
                EMBEDDING_MODEL,

            "vector_database":
                "FAISS"

        }


        # ====================================================
        # LOG RESULT
        # ====================================================

        print()

        print(
            f"Retrieved chunks : "
            f"{len(serialized_documents)}"
        )

        print(
            f"Sources          : "
            f"{len(serialized_sources)}"
        )

        print(
            f"Total time       : "
            f"{total_time:.3f}s"
        )

        print("=" * 70)


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
    # RAG / GEMINI / LANGCHAIN ERROR
    # ========================================================

    except Exception as error:

        print()
        print("=" * 70)

        print(
            "DOCUMIND RAG APPLICATION ERROR"
        )

        print("=" * 70)

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
# API INFORMATION
# ============================================================

@app.route(
    "/api/info",
    methods=["GET"]
)
def api_info():

    return jsonify({

        "application":
            APP_NAME,

        "description":
            "AI-Powered Multi-Document Question Answering System",

        "version":
            APP_VERSION,

        "framework":
            "Flask",

        "rag_framework":
            "LangChain",

        "vector_database":
            "FAISS",

        "embedding_model":
            EMBEDDING_MODEL,

        "llm":
            LLM_MODEL,

        "documents":
            "Multi-document",

        "features": [

            "PDF Knowledge Base",

            "Semantic Retrieval",

            "FAISS Vector Search",

            "Retrieval-Augmented Generation",

            "Source Attribution",

            "Google Gemini",

            "Document Viewer"

        ],

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
                "GET /api/info",

            "document_viewer":
                "GET /documents/<filename>"

        }

    }), 200


# ============================================================
# 404 HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    # API request

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "API endpoint not found."

        }), 404


    # Document request

    if request.path.startswith(
        "/documents/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Document not found."

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
        "=" * 70
    )

    print(
        "UNHANDLED APPLICATION ERROR"
    )

    print(
        "=" * 70
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
    print("=" * 70)

    print(
        "DOCUMIND RAG SERVER"
    )

    print("=" * 70)

    print(
        f"Port             : {port}"
    )

    print(
        f"LLM              : {LLM_MODEL}"
    )

    print(
        f"Embedding        : {EMBEDDING_MODEL}"
    )

    print(
        f"Vector Database  : FAISS"
    )

    print(
        f"Documents        : {DOCUMENTS_DIR}"
    )

    print(
        "Debug            : OFF"
    )

    print("=" * 70)


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )