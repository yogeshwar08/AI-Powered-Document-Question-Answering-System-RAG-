import os
import time
from pathlib import Path

from flask import Flask, request, render_template_string, jsonify
from dotenv import load_dotenv
from google import genai

from main import load_vectorstore

load_dotenv()

app = Flask(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not configured.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# ------------------------------------------------------------
# HTML / CSS
# ------------------------------------------------------------

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocuMind RAG | AI Document Q&A</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Inter, Arial, sans-serif;
            color: #eef2f8;
            background:
                radial-gradient(circle at 0% 0%, rgba(91,112,255,.12), transparent 28%),
                radial-gradient(circle at 100% 10%, rgba(155,89,255,.10), transparent 28%),
                #080b12;
        }
        .layout { display: flex; min-height: 100vh; }
        aside {
            width: 290px; padding: 25px 20px; flex-shrink: 0;
            background: #070a10;
            border-right: 1px solid rgba(255,255,255,.07);
        }
        main { flex: 1; padding: 34px; max-width: 1500px; margin: auto; width: 100%; }
        .brand { font-size: 23px; font-weight: 800; }
        .muted { color: #7f8a9d; font-size: 12px; line-height: 1.6; }
        .status {
            padding: 10px 12px; border-radius: 10px; margin: 8px 0;
            background: rgba(255,255,255,.035);
            border: 1px solid rgba(255,255,255,.07);
            font-size: 13px;
        }
        .ok { color: #8ff0b0; }
        .bad { color: #ff9b9b; }
        .hero {
            padding: 34px 38px; border-radius: 22px; margin-bottom: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,.065), rgba(255,255,255,.018));
            border: 1px solid rgba(255,255,255,.10);
            box-shadow: 0 24px 70px rgba(0,0,0,.28);
        }
        .eyebrow {
            color: #8fa0ff; font-size: 12px; font-weight: 700;
            letter-spacing: 1.8px; text-transform: uppercase; margin-bottom: 10px;
        }
        h1 { margin: 0 0 14px; font-size: 40px; line-height: 1.1; }
        h2 { font-size: 19px; margin: 26px 0 10px; }
        .hero p { color: #aeb8c9; font-size: 15px; line-height: 1.7; max-width: 900px; }
        .metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; }
        .card {
            padding: 20px; border-radius: 16px;
            border: 1px solid rgba(255,255,255,.08);
            background: rgba(255,255,255,.035);
        }
        .metric-label {
            color: #7f8a9d; font-size: 11px; font-weight: 700;
            letter-spacing: 1px; text-transform: uppercase;
        }
        .metric-value { margin-top: 10px; font-size: 20px; font-weight: 700; }
        textarea {
            width: 100%; min-height: 120px; resize: vertical;
            padding: 16px; border-radius: 12px;
            border: 1px solid #d1d5db; background: white; color: #111827;
            font: inherit; font-size: 15px;
        }
        .controls { display: flex; gap: 12px; align-items: center; margin-top: 12px; }
        button {
            border: 0; border-radius: 10px; padding: 12px 20px;
            cursor: pointer; font-weight: 700; font-size: 14px;
        }
        .primary { background: #6f7cff; color: white; }
        .secondary { background: #242a35; color: #eef2f8; }
        select, input[type=number] {
            background: #111722; color: #eef2f8; border: 1px solid #343b49;
            padding: 10px; border-radius: 9px;
        }
        .answer {
            padding: 26px 28px; border: 1px solid rgba(255,255,255,.10);
            border-radius: 18px; background: rgba(255,255,255,.035);
            box-shadow: 0 18px 50px rgba(0,0,0,.18);
            white-space: pre-wrap; line-height: 1.7;
        }
        .source {
            padding: 16px 18px; border: 1px solid rgba(255,255,255,.08);
            border-radius: 13px; background: rgba(255,255,255,.025); margin-bottom: 10px;
        }
        .source-meta { color: #7f8a9d; font-size: 11px; text-transform: uppercase; letter-spacing: .9px; }
        .source-name { font-size: 14px; font-weight: 600; margin-top: 5px; }
        .pill {
            display: inline-block; padding: 5px 9px; margin: 3px;
            border: 1px solid rgba(255,255,255,.10); border-radius: 7px;
            background: rgba(255,255,255,.035); color: #c8d1df; font-size: 11px;
        }
        details {
            margin-top: 14px; padding: 12px 15px; border-radius: 10px;
            background: rgba(255,255,255,.025);
            border: 1px solid rgba(255,255,255,.07);
        }
        pre { white-space: pre-wrap; overflow-x: auto; color: #cbd3df; }
        .error {
            color: #ffb4b4; background: rgba(255,80,80,.08);
            padding: 13px; border-radius: 10px; margin-top: 12px;
        }
        footer { text-align: center; color: #626d7f; font-size: 11px; padding: 35px 0 8px; }
        @media(max-width: 900px) {
            .layout { display: block; }
            aside { width: 100%; }
            main { padding: 18px; }
            .metrics { grid-template-columns: repeat(2,1fr); }
            h1 { font-size: 30px; }
        }
    </style>
</head>
<body>
<div class="layout">
<aside>
    <div class="brand">◆ DocuMind RAG</div>
    <div class="muted" style="margin-top:5px">AI Document Intelligence</div>

    <h3>System</h3>
    <div class="status ok">
        ✓ FAISS vector store available
    </div>
    <div class="status ok">✓ Gemini API configured</div>
    <div class="status ok">✓ MiniLM embeddings configured</div>

    {% if vector_error %}
    <details>
        <summary>Technical error</summary>
        <pre>{{ vector_error }}</pre>
    </details>
    {% endif %}

    <hr style="border-color:rgba(255,255,255,.07);margin:22px 0">

    <h3>Retrieval Controls</h3>
    <form method="POST">
        <label class="muted">Retrieved chunks</label><br>
        <input type="number" name="top_k" min="2" max="8" value="{{ top_k }}" style="width:80px;margin-top:7px">
        <div style="margin-top:12px">
            <label><input type="checkbox" name="show_context" {% if show_context %}checked{% endif %}> Show retrieved context</label>
        </div>
        <input type="hidden" name="question" value="{{ question }}">
        <button class="secondary" style="margin-top:12px" type="submit">Apply</button>
    </form>

    <hr style="border-color:rgba(255,255,255,.07);margin:22px 0">

    <h3>Architecture</h3>
    {% for step in ['▣ PDF ingestion','▣ Text chunking','▣ MiniLM embeddings','▣ FAISS similarity search','◆ Gemini AI','✓ Grounded response'] %}
        <div class="status">{{ step }}</div>
    {% endfor %}

    <h3>Technology Stack</h3>
    {% for tech in ['Python','Flask','LangChain','FAISS','Sentence Transformers','Hugging Face','Gemini API'] %}
        <span class="pill">{{ tech }}</span>
    {% endfor %}
</aside>

<main>
    <div class="hero">
        <div class="eyebrow">Retrieval-Augmented Generation</div>
        <h1>◆ AI-Powered Document Q&amp;A</h1>
        <p>Ask questions about your PDF documents using semantic search, FAISS retrieval, and Gemini AI.</p>
    </div>

    <div class="metrics">
        <div class="card"><div class="metric-label">Knowledge Base</div><div class="metric-value">{{ pdf_count }} PDF</div></div>
        <div class="card"><div class="metric-label">Vector Engine</div><div class="metric-value">FAISS</div></div>
        <div class="card"><div class="metric-label">Embeddings</div><div class="metric-value">MiniLM-L6-v2</div></div>
        <div class="card"><div class="metric-label">LLM</div><div class="metric-value">{{ llm_model }}</div></div>
    </div>

    {% if pdf_names %}
    <details>
        <summary>▣ Knowledge Base Details</summary>
        <p>Indexed PDF documents:</p>
        {% for name in pdf_names %}<div>• {{ name }}</div>{% endfor %}
    </details>
    {% endif %}

    <h2>⌕ Ask your document</h2>
    <div class="muted">Ask a factual, conceptual, or coding question. The system retrieves relevant PDF chunks and sends the context to Gemini.</div>

    <form method="POST" action="/">
        <textarea name="question" placeholder="Example: What is indentation and why is it important in Python?">{{ question }}</textarea>
        <div class="controls">
            <input type="hidden" name="top_k" value="{{ top_k }}">
            <input type="hidden" name="show_context" value="{{ 'on' if show_context else '' }}">
            <button class="primary" type="submit">→ Generate Answer</button>
            <a href="/" style="text-decoration:none"><button class="secondary" type="button">Clear</button></a>
        </div>
    </form>

    {% if warning %}<div class="error">{{ warning }}</div>{% endif %}
    {% if error %}<div class="error"><strong>{{ error }}</strong><details><pre>{{ error_detail }}</pre></details></div>{% endif %}

    {% if answer %}
    <h2>◆ Generated Answer</h2>
    <div class="answer">{{ answer }}</div>

    <h2>▣ Response Metrics</h2>
    <div class="metrics">
        <div class="card"><div class="metric-label">Retrieved Chunks</div><div class="metric-value">{{ documents|length }}</div></div>
        <div class="card"><div class="metric-label">Source Pages</div><div class="metric-value">{{ sources|length }}</div></div>
        <div class="card"><div class="metric-label">Retrieval</div><div class="metric-value">{{ retrieval_time }}s</div></div>
        <div class="card"><div class="metric-label">Total Response</div><div class="metric-value">{{ total_time }}s</div></div>
    </div>

    <h2>▣ Retrieved Sources</h2>
    {% for source, page in sources %}
    <div class="source">
        <div class="source-meta">Source</div>
        <div class="source-name">▣ {{ source }}</div>
        <div class="source-meta" style="margin-top:9px">Page</div>
        <div class="source-name">{{ page }}</div>
    </div>
    {% endfor %}

    <details>
        <summary>▣ Retrieval Evidence</summary>
        {% for document, score in scored_documents %}
            {% set page = document.metadata.get('page','Unknown') %}
            {% if page is number %}{% set page = page + 1 %}{% endif %}
            <p><strong>Rank {{ loop.index }} · Page {{ page }} · FAISS distance: {{ '%.4f'|format(score) }}</strong></p>
            <div class="muted">{{ document.page_content[:500] }}{% if document.page_content|length > 500 %}...{% endif %}</div>
        {% endfor %}
    </details>

    {% if show_context %}
    <details open>
        <summary>▣ Full Retrieved Context</summary>
        {% for document in documents %}
            {% set page = document.metadata.get('page','Unknown') %}
            {% if page is number %}{% set page = page + 1 %}{% endif %}
            <h4>Chunk {{ loop.index }} · Page {{ page }}</h4>
            <pre>{{ document.page_content }}</pre>
        {% endfor %}
    </details>
    {% endif %}
    {% endif %}

    <footer>DocuMind RAG · PDF ingestion · embeddings · FAISS retrieval · Gemini generation · Flask</footer>
</main>
</div>
</body>
</html>
"""

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

_vectorstore = None
_vector_error = None

def get_vectorstore():
    global _vectorstore, _vector_error
    if _vectorstore is None:
        try:
            _vectorstore = load_vectorstore()
            _vector_error = None
        except Exception as exc:
            _vector_error = str(exc)
            raise
    return _vectorstore

def get_document_stats():
    pdf_files = list(Path("documents").glob("*.pdf"))
    return len(pdf_files), [p.name for p in pdf_files]

def unique_sources(documents):
    seen = set()
    results = []
    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")
        if isinstance(page, int):
            page += 1
        key = f"{source}|{page}"
        if key not in seen:
            seen.add(key)
            results.append((source, page))
    return results

def build_context(documents):
    parts = []
    for i, document in enumerate(documents, start=1):
        page = document.metadata.get("page", "Unknown")
        if isinstance(page, int):
            page += 1
        parts.append(
            f"SOURCE {i}\nPAGE: {page}\n\n{document.page_content}"
        )
    return "\n\n".join(parts)

def generate_answer(question, context):
    prompt = f"""
You are a professional document question-answering assistant.

Answer the user's question using the retrieved PDF context.

RETRIEVED CONTEXT
=================
{context}

USER QUESTION
=============
{question}

RULES
=====
1. Use the retrieved context whenever it contains relevant information.
2. If the PDF contains only a question and not its solution, solve it yourself.
3. Clearly distinguish generated solutions from information explicitly found in the document.
4. For Python questions, provide complete executable Python code.
5. Explain important concepts briefly.
6. Do not invent facts about the PDF.
7. Give a direct, interview-ready answer.
"""
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as exc:
            last_error = str(exc)
            temporary = any(x in last_error.upper() for x in ["502", "503", "504", "UNAVAILABLE", "INTERNAL"])
            if temporary and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_error)
    raise RuntimeError(last_error or "Gemini returned no answer.")

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    pdf_count, pdf_names = get_document_stats()

    question = ""
    answer = None
    warning = None
    error = None
    error_detail = ""
    documents = []
    scored_documents = []
    sources = []
    retrieval_time = 0
    total_time = 0
    top_k = 4
    show_context = False

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        try:
            top_k = max(2, min(8, int(request.form.get("top_k", 4))))
        except ValueError:
            top_k = 4

        show_context = request.form.get("show_context") == "on"

        if not question:
            warning = "Please enter a question."
        else:
            try:
                vectorstore = get_vectorstore()

                start = time.perf_counter()
                scored_documents = vectorstore.similarity_search_with_score(
                    question, k=top_k
                )
                retrieval_time = time.perf_counter() - start

                documents = [doc for doc, _ in scored_documents]

                if not documents:
                    warning = "No relevant information was found in the PDF."
                else:
                    context = build_context(documents)

                    generation_start = time.perf_counter()
                    answer = generate_answer(question, context)
                    generation_time = time.perf_counter() - generation_start
                    total_time = retrieval_time + generation_time
                    sources = unique_sources(documents)

            except Exception as exc:
                error = "The RAG pipeline could not complete the request."
                error_detail = str(exc)

    return render_template_string(
        HTML,
        vector_ready=_vectorstore is not None,
        vector_error=_vector_error,
        pdf_count=pdf_count,
        pdf_names=pdf_names,
        llm_model=LLM_MODEL,
        question=question,
        answer=answer,
        warning=warning,
        error=error,
        error_detail=error_detail,
        documents=documents,
        scored_documents=scored_documents,
        sources=sources,
        retrieval_time=f"{retrieval_time:.2f}",
        total_time=f"{total_time:.2f}",
        top_k=top_k,
        show_context=show_context,
    )

@app.route("/health", methods=["GET"])
def health():
    try:
        get_vectorstore()
        return jsonify({"status": "ok", "vectorstore": "loaded"})
    except Exception as exc:
        return jsonify({"status": "error", "vectorstore": "unavailable", "error": str(exc)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
