import os
import time

from dotenv import load_dotenv
from google import genai

from main import retrieve_documents


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3-flash-preview"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Check your .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================================
# HEADER
# ============================================================

print("\n========================================")
print("       AI-Powered Document Q&A")
print("========================================")


# ============================================================
# QUESTION
# ============================================================

question = input(
    "\nAsk a question about your PDF: "
).strip()


if not question:

    print(
        "\nPlease enter a question."
    )

    raise SystemExit


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

print(
    "\nSearching PDF..."
)


try:

    documents = retrieve_documents(
        question,
        k=4,
    )

except Exception as e:

    print(
        "\nError while searching PDF:"
    )

    print(e)

    raise SystemExit


if not documents:

    print(
        "\nNo relevant information found."
    )

    raise SystemExit


# ============================================================
# BUILD CONTEXT
# ============================================================

context_parts = []


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


    context_parts.append(
        f"""
SOURCE {i}
PAGE: {page}

{document.page_content}
"""
    )


context = "\n".join(
    context_parts
)


# ============================================================
# PROMPT
# ============================================================

prompt = f"""
You are a professional Python programming
interview assistant.

The user asks a question about a PDF.

Use the retrieved PDF context below.

RETRIEVED PDF CONTENT
=====================

{context}

USER QUESTION
=============

{question}

INSTRUCTIONS
============

1. Understand exactly what the user is asking.

2. Use the retrieved PDF context whenever
   it contains relevant information.

3. If the PDF contains an answer,
   explain it clearly.

4. If the PDF contains only the question,
   generate the correct answer yourself.

5. For programming questions, provide
   complete executable Python code.

6. Explain the code briefly.

7. Do not invent facts about the PDF.

8. Clearly distinguish generated solutions
   from information found in the PDF.

9. Keep the answer suitable for a
   Python interview.

Return:

ANSWER:
<clear answer>

PYTHON CODE:
<code if applicable>

EXPLANATION:
<short explanation>
"""


# ============================================================
# GEMINI GENERATION WITH RETRIES
# ============================================================

print(
    "\nGenerating answer..."
)


answer = None


for attempt in range(3):

    try:

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
        )

        answer = response.text

        break


    except Exception as e:

        error_message = str(e)

        print(
            f"\nGemini request failed "
            f"(attempt {attempt + 1}/3)"
        )

        print(
            error_message
        )

        # Retry temporary server errors
        if (
            "502" in error_message
            or "503" in error_message
            or "504" in error_message
            or "UNAVAILABLE" in error_message
            or "INTERNAL" in error_message
        ):

            if attempt < 2:

                wait_time = 2 ** attempt

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

                continue

        # Do not repeatedly retry
        # authentication/model/quota errors
        print(
            "\nGemini request could not be completed."
        )

        raise SystemExit


# ============================================================
# CHECK ANSWER
# ============================================================

if not answer:

    print(
        "\nGemini returned no answer."
    )

    raise SystemExit


# ============================================================
# DISPLAY ANSWER
# ============================================================

print(
    "\n========================================"
)

print(
    "                 ANSWER"
)

print(
    "========================================\n"
)

print(
    answer
)


# ============================================================
# DISPLAY SOURCES
# ============================================================

print(
    "\n========================================"
)

print(
    "                 SOURCES"
)

print(
    "========================================"
)


shown_sources = set()


for document in documents:

    source = document.metadata.get(
        "source",
        "Unknown",
    )

    page = document.metadata.get(
        "page",
        "Unknown",
    )

    if isinstance(page, int):

        page += 1


    source_info = (
        f"{source} - Page {page}"
    )


    if source_info not in shown_sources:

        print(
            source_info
        )

        shown_sources.add(
            source_info
        )


print(
    "\nDone."
)