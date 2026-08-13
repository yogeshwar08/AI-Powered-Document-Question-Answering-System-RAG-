from main import retrieve_documents
from langchain_ollama import OllamaLLM


# ==================================================
# 1. LOAD LOCAL LLM
# ==================================================

llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0
)


# ==================================================
# 2. ASK QUESTION
# ==================================================

print("\n========================================")
print("   AI-Powered Document Q&A")
print("========================================")

question = input("\nAsk a question about your PDF: ")


# ==================================================
# 3. RETRIEVE RELEVANT PDF CONTENT
# ==================================================

print("\nSearching PDF...")

documents = retrieve_documents(
    question,
    k=4
)


if not documents:

    print("\nNo relevant information found in the PDF.")

    exit()


# ==================================================
# 4. CREATE CONTEXT
# ==================================================

context_parts = []

for i, document in enumerate(documents, start=1):

    page = document.metadata.get("page", "Unknown")

    if isinstance(page, int):
        page = page + 1

    context_parts.append(
        f"""
SOURCE {i}
PAGE: {page}

{document.page_content}
"""
    )


context = "\n".join(context_parts)


# ==================================================
# 5. RAG PROMPT
# ==================================================

prompt = f"""
You are a Python programming interview assistant.

The user has provided a question that was retrieved
from a PDF containing Python interview questions.

RETRIEVED PDF CONTENT:
----------------------
{context}
----------------------

USER QUESTION:
{question}

Your task:

1. Understand exactly what the user is asking.
2. Use the retrieved PDF content to identify the relevant
   interview question or information.
3. If the PDF contains an answer, explain that answer.
4. If the PDF contains only the question and does NOT
   contain an answer, generate the correct answer yourself.
5. For programming questions, provide complete Python code.
6. Explain the code step by step.
7. Keep the answer suitable for a Python interview.
8. Do not invent or claim that generated code came from the PDF.
9. Do not repeat the retrieved PDF text unnecessarily.

Return the answer in this format:

ANSWER:
<clear explanation>

PYTHON CODE:
<complete Python code if applicable>

EXPLANATION:
<short explanation of how the code works>
"""


# ==================================================
# 6. GENERATE ANSWER
# ==================================================

print("\nGenerating answer...\n")

answer = llm.invoke(prompt)


# ==================================================
# 7. DISPLAY ANSWER
# ==================================================

print("\n========================================")
print("                ANSWER")
print("========================================\n")

print(answer)


# ==================================================
# 8. DISPLAY SOURCES
# ==================================================

print("\n========================================")
print("                SOURCES")
print("========================================")

shown_sources = set()

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
        page = page + 1

    source_info = f"{source} - Page {page}"

    if source_info not in shown_sources:

        print(source_info)

        shown_sources.add(source_info)