# ============================================================
# DOCUMIND RAG
# AI-POWERED DOCUMENT QUESTION ANSWERING SYSTEM
#
# PROFESSIONAL RAG + PYTHON COMPILER UI
#
# RAG:
#   PDF
#    ↓
#   Parsing
#    ↓
#   Chunking
#    ↓
#   Embeddings
#    ↓
#   FAISS
#    ↓
#   LangChain Retriever
#    ↓
#   Augmentation
#    ↓
#   Gemini
#
# CODE EXECUTION:
#   Python Code
#       ↓
#   Flask API
#       ↓
#   Validation
#       ↓
#   Python Interpreter
#       ↓
#   Terminal Output
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import ast
import html
import subprocess
import sys
import tempfile
from pathlib import Path


# ============================================================
# FLASK
# ============================================================

from flask import (
    Flask,
    request,
    render_template_string,
    jsonify
)


# ============================================================
# ENVIRONMENT
# ============================================================

from dotenv import load_dotenv


# ============================================================
# MARKDOWN
# ============================================================

import markdown


# ============================================================
# BLEACH
# ============================================================

import bleach


# ============================================================
# RAG
# ============================================================

from query import answer_query

from main import (
    load_vectorstore,
    get_document_stats
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# SETTINGS
# ============================================================

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-2.5-flash"
)

MAX_CODE_LENGTH = 20000

CODE_TIMEOUT = 5


# ============================================================
# MARKDOWN SETTINGS
# ============================================================

MARKDOWN_EXTENSIONS = [
    "fenced_code",
    "tables",
    "nl2br",
    "sane_lists"
]


# ============================================================
# SAFE HTML TAGS
# ============================================================

ALLOWED_TAGS = [

    "p",
    "br",

    "strong",
    "em",

    "h1",
    "h2",
    "h3",
    "h4",

    "ul",
    "ol",
    "li",

    "blockquote",
    "hr",

    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",

    "div",
    "span",

    "pre",
    "code",

    "textarea",

    "button"

]


# ============================================================
# SAFE HTML ATTRIBUTES
# ============================================================

ALLOWED_ATTRIBUTES = {

    "div": [
        "class",
        "id"
    ],

    "span": [
        "class"
    ],

    "textarea": [
        "class",
        "id",
        "spellcheck"
    ],

    "button": [
        "class",
        "type"
    ],

    "code": [
        "class"
    ],

    "pre": [
        "class"
    ],

    "table": [
        "class"
    ]

}


# ============================================================
# SAFE PYTHON MODULES
#
# These are standard-library modules useful for
# educational examples.
# ============================================================

SAFE_MODULES = {

    "math",
    "statistics",
    "random",
    "datetime",
    "decimal",
    "fractions",
    "itertools",
    "collections",
    "string"

}


# ============================================================
# BLOCKED PYTHON OPERATIONS
#
# This is NOT a true security sandbox.
# It is an additional safety layer for local demos.
# ============================================================

BLOCKED_NAMES = {

    "eval",
    "exec",
    "compile",
    "open",
    "input",

    "breakpoint",

    "__import__",

    "globals",
    "locals",
    "vars",

    "getattr",
    "setattr",
    "delattr",

    "help",

    "exit",
    "quit",

    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",

    "importlib",

    "ctypes",

    "pickle",
    "marshal",

    "builtins"

}


# ============================================================
# RENDER MARKDOWN
# ============================================================

def render_markdown(
    text
):

    if not text:

        return ""

    # --------------------------------------------------------
    # Detect fenced Python code blocks.
    #
    # Example:
    #
    # ```python
    # print("Hello")
    # ```
    # --------------------------------------------------------

    code_blocks = []


    pattern = re.compile(

        r"```(?:python|py)?\s*\n?"
        r"(.*?)"
        r"```",

        re.DOTALL |
        re.IGNORECASE

    )


    def replace_code(
        match
    ):

        code = (
            match.group(1)
            .strip("\n")
        )


        block_id = (
            len(code_blocks)
        )


        code_blocks.append(
            code
        )


        placeholder = (
            f"\n\n"
            f"CODEBLOCKPLACEHOLDER"
            f"{block_id}"
            f"\n\n"
        )


        return placeholder


    processed_text = pattern.sub(

        replace_code,

        text

    )


    # --------------------------------------------------------
    # Markdown → HTML
    # --------------------------------------------------------

    converted = markdown.markdown(

        processed_text,

        extensions=
            MARKDOWN_EXTENSIONS

    )


    # --------------------------------------------------------
    # Create compiler blocks
    # --------------------------------------------------------

    for index, code in enumerate(
        code_blocks
    ):


        safe_code = html.escape(
            code
        )


        compiler_block = f"""

        <div class="compiler">

            <div class="compiler-header">

                <div class="compiler-title">

                    <span class="terminal-dot"></span>
                    <span class="terminal-dot"></span>
                    <span class="terminal-dot"></span>

                    <span class="language-label">
                        PYTHON
                    </span>

                </div>

                <div class="compiler-actions">

                    <button
                        class="copy-code"
                        type="button"
                    >
                        Copy
                    </button>

                    <button
                        class="run-code"
                        type="button"
                    >
                        ▶ Run Code
                    </button>

                </div>

            </div>


            <div class="editor">

                <div class="line-numbers"></div>

                <textarea
                    class="code-editor"
                    id="code-editor-{index}"
                    spellcheck="false"
                >{safe_code}</textarea>

            </div>


            <div class="output-panel">

                <div class="output-header">

                    <span>
                        TERMINAL
                    </span>

                    <span class="execution-status">
                        Ready
                    </span>

                </div>


                <pre class="output-content">Click "Run Code" to execute.</pre>

            </div>

        </div>

        """


        placeholder = (
            f"CODEBLOCKPLACEHOLDER"
            f"{index}"
        )


        converted = converted.replace(

            placeholder,

            compiler_block

        )


    # --------------------------------------------------------
    # Sanitize generated HTML
    # --------------------------------------------------------

    safe_html = bleach.clean(

        converted,

        tags=
            ALLOWED_TAGS,

        attributes=
            ALLOWED_ATTRIBUTES,

        strip=True

    )


    return safe_html


# ============================================================
# VALIDATE PYTHON CODE
# ============================================================

def validate_python_code(
    code
):

    # --------------------------------------------------------
    # Length check
    # --------------------------------------------------------

    if not code:

        raise ValueError(
            "No Python code supplied."
        )


    if len(code) > MAX_CODE_LENGTH:

        raise ValueError(
            "Code is too large."
        )


    # --------------------------------------------------------
    # Parse syntax
    # --------------------------------------------------------

    try:

        tree = ast.parse(
            code
        )

    except SyntaxError as error:

        raise ValueError(
            f"Syntax Error: {error}"
        )


    # --------------------------------------------------------
    # Inspect AST
    # --------------------------------------------------------

    for node in ast.walk(tree):


        # --------------------------------------------
        # Names
        # --------------------------------------------

        if isinstance(
            node,
            ast.Name
        ):

            if node.id in BLOCKED_NAMES:

                raise ValueError(

                    f"Operation '{node.id}' "
                    f"is not allowed."

                )


        # --------------------------------------------
        # Imports
        # --------------------------------------------

        if isinstance(
            node,
            ast.Import
        ):

            for alias in node.names:

                module = (
                    alias.name.split(".")[0]
                )

                if module not in SAFE_MODULES:

                    raise ValueError(

                        f"Import '{module}' "
                        f"is not allowed."

                    )


        # --------------------------------------------
        # From imports
        # --------------------------------------------

        if isinstance(
            node,
            ast.ImportFrom
        ):

            if node.module:

                module = (
                    node.module.split(".")[0]
                )

                if module not in SAFE_MODULES:

                    raise ValueError(

                        f"Import '{module}' "
                        f"is not allowed."

                    )


        # --------------------------------------------
        # Dangerous dunder attributes
        # --------------------------------------------

        if isinstance(
            node,
            ast.Attribute
        ):

            if node.attr.startswith("__"):

                raise ValueError(
                    "Dunder attribute access "
                    "is not allowed."
                )


    return True


# ============================================================
# EXECUTE PYTHON CODE
#
# LOCAL DEMO / EDUCATIONAL EXECUTOR
# ============================================================

def execute_python_code(
    code
):

    validate_python_code(
        code
    )


    # --------------------------------------------------------
    # Create isolated temporary directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:


        script_path = (
            Path(temp_dir)
            / "program.py"
        )


        script_path.write_text(

            code,

            encoding="utf-8"

        )


        # ----------------------------------------------------
        # Restricted environment
        # ----------------------------------------------------

        environment = {

            "PATH":
                os.environ.get(
                    "PATH",
                    ""
                ),

            "PYTHONIOENCODING":
                "utf-8",

            "PYTHONNOUSERSITE":
                "1"

        }


        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        try:

            result = subprocess.run(

                [

                    sys.executable,

                    "-I",

                    str(
                        script_path
                    )

                ],

                cwd=
                    temp_dir,

                capture_output=
                    True,

                text=
                    True,

                timeout=
                    CODE_TIMEOUT,

                env=
                    environment

            )


        except subprocess.TimeoutExpired:

            return {

                "success":
                    False,

                "output":
                    (
                        "Execution stopped: "
                        "time limit exceeded."
                    )

            }


        except Exception as error:

            return {

                "success":
                    False,

                "output":
                    str(error)

            }


        # ----------------------------------------------------
        # Combine output
        # ----------------------------------------------------

        stdout = (
            result.stdout
            or ""
        )


        stderr = (
            result.stderr
            or ""
        )


        if result.returncode == 0:

            output = stdout

            if not output:

                output = (
                    "Program finished "
                    "successfully with "
                    "no output."
                )


            return {

                "success":
                    True,

                "output":
                    output

            }


        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        return {

            "success":
                False,

            "output":
                stderr
                or stdout
                or
                "Program exited with an error."

        }


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
    DocuMind RAG | AI Document Intelligence
</title>


<style>

/* ============================================================
   GLOBAL
============================================================ */

* {
    box-sizing: border-box;
}


html {
    scroll-behavior: smooth;
}


body {

    margin: 0;

    min-height: 100vh;

    font-family:
        Inter,
        "Segoe UI",
        Arial,
        sans-serif;

    background:
        #080b12;

    color:
        #e7edf7;

}


/* ============================================================
   LAYOUT
============================================================ */

.layout {

    display: flex;

    min-height: 100vh;

}


/* ============================================================
   SIDEBAR
============================================================ */

aside {

    width: 280px;

    flex-shrink: 0;

    padding: 25px 20px;

    background: #070a10;

    border-right:
        1px solid
        rgba(255,255,255,.07);

}


.brand {

    font-size: 22px;

    font-weight: 800;

}


.brand-subtitle {

    margin-top: 5px;

    color: #778397;

    font-size: 12px;

}


.sidebar-title {

    margin-top: 25px;

    margin-bottom: 10px;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 1.2px;

    color: #7d899d;

}


.status {

    margin: 7px 0;

    padding: 9px 11px;

    border-radius: 8px;

    background:
        rgba(255,255,255,.035);

    border:
        1px solid
        rgba(255,255,255,.06);

    color: #aeb9ca;

    font-size: 12px;

}


.status-ok {

    color: #8ce9aa;

}


/* ============================================================
   MAIN
============================================================ */

main {

    flex: 1;

    max-width: 1500px;

    width: 100%;

    margin: auto;

    padding: 34px;

}


/* ============================================================
   HERO
============================================================ */

.hero {

    padding: 34px 38px;

    border-radius: 20px;

    margin-bottom: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.065),
            rgba(255,255,255,.015)
        );

    border:
        1px solid
        rgba(255,255,255,.08);

}


.eyebrow {

    color: #8e9cff;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 1.8px;

    text-transform: uppercase;

}


h1 {

    margin:
        8px 0 12px;

    font-size: 39px;

}


.hero-description {

    max-width: 900px;

    margin: 0;

    color: #a9b4c5;

    line-height: 1.7;

    font-size: 14px;

}


/* ============================================================
   METRICS
============================================================ */

.metrics {

    display: grid;

    grid-template-columns:
        repeat(4,1fr);

    gap: 13px;

}


.card {

    padding: 19px;

    border-radius: 15px;

    background:
        rgba(255,255,255,.035);

    border:
        1px solid
        rgba(255,255,255,.075);

}


.metric-label {

    color: #768296;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1px;

}


.metric-value {

    margin-top: 8px;

    font-size: 17px;

    font-weight: 700;

}


/* ============================================================
   QUERY
============================================================ */

.query-card {

    margin-top: 22px;

}


.section-title {

    margin:
        0 0 8px;

    font-size: 18px;

}


.section-description {

    color: #758195;

    font-size: 12px;

    line-height: 1.6;

    margin-bottom: 14px;

}


textarea {

    width: 100%;

    min-height: 125px;

    padding: 16px;

    resize: vertical;

    border:
        1px solid
        #d5d9df;

    border-radius: 10px;

    background: white;

    color: #111827;

    font-family:
        "Segoe UI",
        Arial,
        sans-serif;

    font-size: 15px;

}


.controls {

    display: flex;

    align-items: center;

    gap: 10px;

    flex-wrap: wrap;

    margin-top: 12px;

}


.controls-label {

    color: #7c8798;

    font-size: 12px;

}


input[type="number"] {

    width: 65px;

    padding: 9px;

    border:
        1px solid
        #343b49;

    border-radius: 8px;

    background: #111722;

    color: white;

}


button {

    border: 0;

    border-radius: 8px;

    padding:
        10px 17px;

    cursor: pointer;

    font-weight: 700;

}


.primary {

    background: #6f7cff;

    color: white;

}


.secondary {

    background: #252b36;

    color: #e7edf7;

}


/* ============================================================
   ANSWER
============================================================ */

.answer-wrapper {

    margin-top: 25px;

}


.answer-card {

    overflow: hidden;

    border:
        1px solid
        rgba(255,255,255,.09);

    border-radius: 16px;

    background: #0d1119;

}


.answer-header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding:
        12px 17px;

    background: #111722;

    border-bottom:
        1px solid
        rgba(255,255,255,.07);

}


.answer-title {

    color: #aeb9ca;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: .7px;

}


.copy-answer {

    padding: 6px 10px;

    background: #1c2330;

    color: #aeb9ca;

    font-size: 10px;

}


/* ============================================================
   ANSWER CONTENT
============================================================ */

.answer-content {

    padding: 28px;

    color: #dce4ef;

    line-height: 1.75;

    font-size: 15px;

}


.answer-content h1 {

    font-size: 26px;

}


.answer-content h2 {

    margin-top: 25px;

    font-size: 20px;

}


.answer-content h3 {

    margin-top: 21px;

    font-size: 17px;

}


.answer-content strong {

    color: white;

}


.answer-content li {

    margin: 6px 0;

}


.answer-content blockquote {

    padding:
        11px 16px;

    border-left:
        3px solid
        #6f7cff;

    background:
        rgba(111,124,255,.07);

}


/* ============================================================
   INLINE CODE
============================================================ */

.answer-content code {

    padding:
        2px 6px;

    border-radius: 5px;

    background: #171d28;

    color: #cbd5ff;

    font-family:
        Consolas,
        "Courier New",
        monospace;

    font-size: 13px;

}


/* ============================================================
   COMPILER
============================================================ */

.compiler {

    margin:
        20px 0 25px;

    overflow: hidden;

    border:
        1px solid
        #293241;

    border-radius: 12px;

    background: #090d13;

    box-shadow:
        0 12px 35px
        rgba(0,0,0,.25);

}


/* ============================================================
   COMPILER HEADER
============================================================ */

.compiler-header {

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding:
        9px 12px;

    background: #111722;

    border-bottom:
        1px solid
        #293241;

}


.compiler-title {

    display: flex;

    align-items: center;

    gap: 7px;

}


.terminal-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #5d687b;

}


.language-label {

    margin-left: 8px;

    color: #8d9aaf;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1px;

}


.compiler-actions {

    display: flex;

    gap: 7px;

}


.compiler-actions button {

    padding:
        6px 10px;

    border:
        1px solid
        #303949;

    border-radius: 6px;

    background: #171e29;

    color: #aeb9ca;

    font-size: 10px;

}


.compiler-actions .run-code {

    background: #1d5d3a;

    border-color: #28784a;

    color: #a7f3c5;

}


/* ============================================================
   CODE EDITOR
============================================================ */

.editor {

    display: flex;

    min-height: 180px;

}


.line-numbers {

    width: 48px;

    padding:
        14px 10px;

    text-align: right;

    user-select: none;

    background: #0c1118;

    color: #4e5a6c;

    font-family:
        Consolas,
        monospace;

    font-size: 13px;

    line-height: 1.7;

    white-space: pre;

}


.code-editor {

    flex: 1;

    min-height: 180px;

    width: 100%;

    resize: vertical;

    padding:
        14px;

    border: 0;

    outline: none;

    border-radius: 0;

    background: #090d13;

    color: #d8e2ef;

    font-family:
        Consolas,
        "Courier New",
        monospace;

    font-size: 13px;

    line-height: 1.7;

    tab-size: 4;

}


/* ============================================================
   OUTPUT
============================================================ */

.output-panel {

    border-top:
        1px solid
        #293241;

    background: #080c12;

}


.output-header {

    display: flex;

    justify-content: space-between;

    padding:
        8px 14px;

    color: #68758a;

    font-size: 9px;

    font-weight: 800;

    letter-spacing: 1px;

}


.execution-status {

    color: #6f7d91;

}


.output-content {

    min-height: 45px;

    margin: 0;

    padding:
        0 14px 15px;

    color: #cbd5e1;

    font-family:
        Consolas,
        "Courier New",
        monospace;

    font-size: 12px;

    line-height: 1.6;

    white-space: pre-wrap;

}


.output-success {

    color: #9ee6b7;

}


.output-error {

    color: #ffaaaa;

}


/* ============================================================
   SOURCES
============================================================ */

.source {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 9px;

    padding: 14px 16px;

    border:
        1px solid
        rgba(255,255,255,.07);

    border-radius: 10px;

    background:
        rgba(255,255,255,.025);

}


.source-name {

    color: #dce4ef;

    font-size: 13px;

    font-weight: 600;

}


.source-page {

    padding:
        5px 9px;

    border-radius: 6px;

    background: #171e2a;

    color: #9da9bb;

    font-size: 10px;

}


/* ============================================================
   ERROR
============================================================ */

.error {

    margin-top: 15px;

    padding: 14px;

    border:
        1px solid
        rgba(255,80,80,.18);

    border-radius: 10px;

    background:
        rgba(255,70,70,.07);

    color: #ffb9b9;

    font-size: 13px;

}


/* ============================================================
   FOOTER
============================================================ */

footer {

    padding:
        35px 0 10px;

    text-align: center;

    color: #566174;

    font-size: 10px;

}


/* ============================================================
   RESPONSIVE
============================================================ */

@media(max-width: 900px) {

    aside {

        width: 225px;

    }


    main {

        padding: 20px;

    }


    .metrics {

        grid-template-columns:
            repeat(2,1fr);

    }

}


@media(max-width: 700px) {

    .layout {

        display: block;

    }


    aside {

        width: 100%;

    }


    main {

        padding: 15px;

    }


    .metrics {

        grid-template-columns:
            1fr 1fr;

    }


    h1 {

        font-size: 30px;

    }

}

</style>

</head>


<body>


<div class="layout">


<!-- ========================================================
     SIDEBAR
========================================================= -->

<aside>


    <div class="brand">

        ◆ DocuMind RAG

    </div>


    <div class="brand-subtitle">

        AI Document Intelligence

    </div>


    <div class="sidebar-title">

        SYSTEM

    </div>


    <div class="status status-ok">

        ✓ LangChain RAG

    </div>


    <div class="status status-ok">

        ✓ FAISS Vector Database

    </div>


    <div class="status status-ok">

        ✓ MiniLM Embeddings

    </div>


    <div class="status status-ok">

        ✓ Gemini LLM

    </div>


    <div class="status status-ok">

        ✓ Python Compiler

    </div>


    <hr
        style="
            border-color:
            rgba(255,255,255,.07);
            margin:22px 0
        "
    >


    <div class="sidebar-title">

        RAG ARCHITECTURE

    </div>


    <div class="status">
        01 · Knowledge Base
    </div>

    <div class="status">
        02 · Parsing
    </div>

    <div class="status">
        03 · Chunking
    </div>

    <div class="status">
        04 · Embedding
    </div>

    <div class="status">
        05 · Vector Database
    </div>

    <div class="status">
        06 · User Query
    </div>

    <div class="status">
        07 · Retrieval
    </div>

    <div class="status">
        08 · Augmentation
    </div>

    <div class="status">
        09 · Generation
    </div>


    <hr
        style="
            border-color:
            rgba(255,255,255,.07);
            margin:22px 0
        "
    >


    <div class="sidebar-title">

        TECHNOLOGY

    </div>


    <div class="status">
        Python
    </div>

    <div class="status">
        Flask
    </div>

    <div class="status">
        LangChain
    </div>

    <div class="status">
        FAISS
    </div>

    <div class="status">
        HuggingFace
    </div>

    <div class="status">
        Google Gemini
    </div>


</aside>


<!-- ========================================================
     MAIN
========================================================= -->

<main>


    <section class="hero">


        <div class="eyebrow">

            Retrieval-Augmented Generation

        </div>


        <h1>

            ◆ AI-Powered Document Q&amp;A

        </h1>


        <p class="hero-description">

            Ask questions about your documents using
            LangChain retrieval, FAISS semantic search,
            context augmentation, and Google Gemini.
            Generated Python examples can be edited
            and executed directly in the built-in
            compiler.

        </p>


    </section>


    <!-- ====================================================
         METRICS
    ===================================================== -->

    <section class="metrics">


        <div class="card">

            <div class="metric-label">
                Knowledge Base
            </div>

            <div class="metric-value">
                {{ pdf_count }} PDF
            </div>

        </div>


        <div class="card">

            <div class="metric-label">
                Vector Database
            </div>

            <div class="metric-value">
                FAISS
            </div>

        </div>


        <div class="card">

            <div class="metric-label">
                Embeddings
            </div>

            <div class="metric-value">
                MiniLM-L6-v2
            </div>

        </div>


        <div class="card">

            <div class="metric-label">
                LLM
            </div>

            <div class="metric-value">
                {{ llm_model }}
            </div>

        </div>


    </section>


    <!-- ====================================================
         KNOWLEDGE BASE
    ===================================================== -->

    {% if pdf_names %}

    <section
        class="card"
        style="margin-top:22px"
    >


        <h2 class="section-title">

            ▣ Knowledge Base

        </h2>


        <div class="section-description">

            Documents indexed by the RAG pipeline.

        </div>


        {% for name in pdf_names %}

        <div
            style="
                padding:7px 0;
                color:#cbd5e2;
                font-size:13px
            "
        >

            ▸ {{ name }}

        </div>

        {% endfor %}


    </section>

    {% endif %}


    <!-- ====================================================
         QUERY
    ===================================================== -->

    <section class="card query-card">


        <h2 class="section-title">

            ⌕ Ask Your Document

        </h2>


        <div class="section-description">

            Ask a question about your indexed documents.
            Relevant context is retrieved before Gemini
            generates the answer.

        </div>


        <form
            method="POST"
            action="/"
        >


            <textarea
                name="question"
                placeholder="Example: What is type casting? Give examples."
                required
            >{{ question }}</textarea>


            <div class="controls">


                <span class="controls-label">

                    Retrieved chunks:

                </span>


                <input
                    type="number"
                    name="top_k"
                    min="1"
                    max="10"
                    value="{{ top_k }}"
                >


                <button
                    class="primary"
                    type="submit"
                >

                    → Generate Answer

                </button>


                <a
                    href="/"
                    style="text-decoration:none"
                >

                    <button
                        class="secondary"
                        type="button"
                    >

                        Clear

                    </button>

                </a>


            </div>


        </form>


    </section>


    <!-- ====================================================
         ERROR
    ===================================================== -->

    {% if error %}

    <div class="error">

        <strong>
            RAG Pipeline Error
        </strong>

        <br><br>

        {{ error }}

    </div>

    {% endif %}


    <!-- ====================================================
         ANSWER
    ===================================================== -->

    {% if answer %}


    <section class="answer-wrapper">


        <div class="answer-card">


            <div class="answer-header">


                <div class="answer-title">

                    ● AI GENERATED ANSWER

                </div>


                <button
                    class="copy-answer"
                    type="button"
                >

                    Copy Answer

                </button>


            </div>


            <div
                id="answer-content"
                class="answer-content"
            >

                {{ answer_html | safe }}

            </div>


        </div>


    </section>


    <!-- ====================================================
         RESPONSE METRICS
    ===================================================== -->

    <section style="margin-top:25px">


        <h2 class="section-title">

            ▣ Response Metrics

        </h2>


        <div class="metrics">


            <div class="card">

                <div class="metric-label">
                    Retrieved Chunks
                </div>

                <div class="metric-value">
                    {{ documents|length }}
                </div>

            </div>


            <div class="card">

                <div class="metric-label">
                    Retrieval
                </div>

                <div class="metric-value">
                    {{ retrieval_time }}s
                </div>

            </div>


            <div class="card">

                <div class="metric-label">
                    Generation
                </div>

                <div class="metric-value">
                    {{ generation_time }}s
                </div>

            </div>


            <div class="card">

                <div class="metric-label">
                    Total
                </div>

                <div class="metric-value">
                    {{ total_time }}s
                </div>

            </div>


        </div>


    </section>


    <!-- ====================================================
         SOURCES
    ===================================================== -->

    <section style="margin-top:25px">


        <h2 class="section-title">

            ▣ Retrieved Sources

        </h2>


        <div class="section-description">

            Document chunks used to ground the generated
            response.

        </div>


        {% for source, page in sources %}


        <div class="source">


            <div class="source-name">

                ▸ {{ source }}

            </div>


            <div class="source-page">

                PAGE {{ page }}

            </div>


        </div>


        {% endfor %}


    </section>


    {% endif %}


    <footer>

        DocuMind RAG · LangChain · FAISS · Gemini ·
        Python Compiler

    </footer>


</main>


</div>


<script>

/* ============================================================
   UPDATE LINE NUMBERS
============================================================ */

function updateLineNumbers(editor) {

    const container =
        editor
        .closest(".editor")
        .querySelector(
            ".line-numbers"
        );


    const lines =
        editor.value.split("\n").length;


    let numbers = "";


    for (
        let i = 1;
        i <= lines;
        i++
    ) {

        numbers += i + "\n";

    }


    container.textContent =
        numbers;

}


/* ============================================================
   INITIALIZE EDITORS
============================================================ */

document
    .querySelectorAll(
        ".code-editor"
    )
    .forEach(

        function(editor) {

            updateLineNumbers(
                editor
            );


            editor.addEventListener(
                "input",
                function() {

                    updateLineNumbers(
                        editor
                    );

                }
            );

        }

    );


/* ============================================================
   COPY COMPLETE ANSWER
============================================================ */

const copyAnswerButton =
    document.querySelector(
        ".copy-answer"
    );


if (copyAnswerButton) {

    copyAnswerButton.addEventListener(

        "click",

        async function() {

            const answer =
                document.getElementById(
                    "answer-content"
                );


            await navigator
                .clipboard
                .writeText(
                    answer.innerText
                );


            const original =
                this.innerText;


            this.innerText =
                "Copied ✓";


            setTimeout(

                () => {

                    this.innerText =
                        original;

                },

                1500

            );

        }

    );

}


/* ============================================================
   COPY PYTHON CODE
============================================================ */

document
    .querySelectorAll(
        ".copy-code"
    )
    .forEach(

        function(button) {


            button.addEventListener(

                "click",

                async function() {


                    const compiler =
                        this.closest(
                            ".compiler"
                        );


                    const editor =
                        compiler.querySelector(
                            ".code-editor"
                        );


                    await navigator
                        .clipboard
                        .writeText(
                            editor.value
                        );


                    const original =
                        this.innerText;


                    this.innerText =
                        "Copied ✓";


                    setTimeout(

                        () => {

                            this.innerText =
                                original;

                        },

                        1500

                    );

                }

            );

        }

    );


/* ============================================================
   RUN PYTHON CODE
============================================================ */

document
    .querySelectorAll(
        ".run-code"
    )
    .forEach(

        function(button) {


            button.addEventListener(

                "click",

                async function() {


                    const compiler =
                        this.closest(
                            ".compiler"
                        );


                    const editor =
                        compiler.querySelector(
                            ".code-editor"
                        );


                    const output =
                        compiler.querySelector(
                            ".output-content"
                        );


                    const status =
                        compiler.querySelector(
                            ".execution-status"
                        );


                    const code =
                        editor.value;


                    if (!code.trim()) {

                        output.textContent =
                            "No Python code to execute.";

                        output.className =
                            "output-content output-error";

                        return;

                    }


                    /* ----------------------------------------
                       RUNNING STATE
                    ---------------------------------------- */

                    button.disabled =
                        true;


                    button.innerText =
                        "⏳ Running...";


                    status.innerText =
                        "Running";


                    status.style.color =
                        "#f3c969";


                    output.className =
                        "output-content";


                    output.textContent =
                        "Executing Python program...";


                    try {


                        const response =
                            await fetch(
                                "/api/execute",
                                {

                                    method:
                                        "POST",

                                    headers: {

                                        "Content-Type":
                                            "application/json"

                                    },

                                    body:
                                        JSON.stringify({
                                            code: code
                                        })

                                }
                            );


                        const result =
                            await response.json();


                        output.textContent =
                            result.output
                            || "No output.";


                        if (result.success) {


                            output.className =
                                "output-content output-success";


                            status.innerText =
                                "Completed";


                            status.style.color =
                                "#8ce9aa";


                        } else {


                            output.className =
                                "output-content output-error";


                            status.innerText =
                                "Error";


                            status.style.color =
                                "#ffaaaa";

                        }


                    } catch (error) {


                        output.textContent =
                            "Connection error: "
                            + error.message;


                        output.className =
                            "output-content output-error";


                        status.innerText =
                            "Failed";


                        status.style.color =
                            "#ffaaaa";

                    }


                    button.disabled =
                        false;


                    button.innerText =
                        "▶ Run Code";

                }

            );

        }

    );

</script>


</body>

</html>
"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def home():


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    pdf_count, pdf_names = (
        get_document_stats()
    )


    # ========================================================
    # DEFAULTS
    # ========================================================

    question = ""

    answer = None

    answer_html = ""

    documents = []

    sources = []

    error = None


    retrieval_time = "0.00"

    generation_time = "0.00"

    total_time = "0.00"

    top_k = 4


    # ========================================================
    # POST
    # ========================================================

    if request.method == "POST":


        question = (
            request.form
            .get(
                "question",
                ""
            )
            .strip()
        )


        try:

            top_k = int(

                request.form
                .get(
                    "top_k",
                    4
                )

            )

        except ValueError:

            top_k = 4


        top_k = max(
            1,
            min(
                top_k,
                10
            )
        )


        if not question:

            error = (
                "Please enter a question."
            )


        else:


            try:


                # ============================================
                # LANGCHAIN RAG
                # ============================================

                result = (
                    answer_query(

                        question,

                        top_k=
                            top_k

                    )
                )


                # ============================================
                # ANSWER
                # ============================================

                answer = (
                    result[
                        "answer"
                    ]
                )


                # ============================================
                # RENDER MARKDOWN
                # ============================================

                answer_html = (
                    render_markdown(
                        answer
                    )
                )


                # ============================================
                # DOCUMENTS
                # ============================================

                documents = (
                    result[
                        "documents"
                    ]
                )


                # ============================================
                # SOURCES
                # ============================================

                sources = (
                    result[
                        "sources"
                    ]
                )


                # ============================================
                # PERFORMANCE
                # ============================================

                retrieval_time = (
                    f"{result['retrieval_time']:.2f}"
                )


                generation_time = (
                    f"{result['generation_time']:.2f}"
                )


                total_time = (
                    f"{result['total_time']:.2f}"
                )


            except Exception as exc:


                print(
                    "\nRAG ERROR:"
                )

                print(
                    str(exc)
                )


                error = (
                    "The RAG pipeline "
                    "could not complete "
                    "the request."
                )


    # ========================================================
    # RENDER
    # ========================================================

    return render_template_string(

        HTML,

        pdf_count=
            pdf_count,

        pdf_names=
            pdf_names,

        question=
            question,

        answer=
            answer,

        answer_html=
            answer_html,

        documents=
            documents,

        sources=
            sources,

        error=
            error,

        retrieval_time=
            retrieval_time,

        generation_time=
            generation_time,

        total_time=
            total_time,

        top_k=
            top_k,

        llm_model=
            LLM_MODEL

    )


# ============================================================
# PYTHON COMPILER API
# ============================================================

@app.route(
    "/api/execute",
    methods=[
        "POST"
    ]
)
def execute_code_api():


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    data = (
        request.get_json(
            silent=True
        )
    )


    if not data:

        return jsonify({

            "success":
                False,

            "output":
                "Invalid request."

        }), 400


    # --------------------------------------------------------
    # CODE
    # --------------------------------------------------------

    code = data.get(
        "code",
        ""
    )


    if not isinstance(
        code,
        str
    ):

        return jsonify({

            "success":
                False,

            "output":
                "Code must be a string."

        }), 400


    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    try:


        result = (
            execute_python_code(
                code
            )
        )


        return jsonify(
            result
        )


    except ValueError as error:


        return jsonify({

            "success":
                False,

            "output":
                str(error)

        }), 400


    except Exception as error:


        print(
            "EXECUTION ERROR:",
            error
        )


        return jsonify({

            "success":
                False,

            "output":
                "Internal execution error."

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=[
        "GET"
    ]
)
def health():


    try:


        load_vectorstore()


        return jsonify({

            "status":
                "ok",

            "application":
                "DocuMind RAG",

            "framework":
                "LangChain",

            "web_framework":
                "Flask",

            "vector_database":
                "FAISS",

            "embedding_model":
                "all-MiniLM-L6-v2",

            "llm":
                LLM_MODEL,

            "compiler":
                "Python"

        })


    except Exception as exc:


        return jsonify({

            "status":
                "error",

            "application":
                "DocuMind RAG",

            "error":
                str(exc)

        }), 500


# ============================================================
# APPLICATION ENTRY POINT
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
        "                  DOCUMIND RAG"
    )

    print(
        "=" * 65
    )

    print(
        "RAG Framework   : LangChain"
    )

    print(
        "Web Framework   : Flask"
    )

    print(
        "Vector Database : FAISS"
    )

    print(
        "Embeddings      : MiniLM-L6-v2"
    )

    print(
        f"LLM             : {LLM_MODEL}"
    )

    print(
        "Compiler        : Python"
    )

    print(
        f"Port            : {port}"
    )

    print(
        "=" * 65
    )

    print()


    app.run(

        host=
            "0.0.0.0",

        port=
            port,

        debug=
            False

    )