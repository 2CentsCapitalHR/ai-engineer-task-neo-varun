[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)

# ADGM Corporate Agent (Document Intelligence)

A minimal Streamlit app that accepts .docx uploads, identifies document types, checks required ADGM checklists, flags red flags, inserts inline comments, and returns a marked-up .docx and JSON report. Uses RAG with a lightweight local vector index.

## Features
- Upload .docx files
- Auto-detect process (Company Incorporation, Licensing, Employment HR)
- Checklist verification for required documents
- Red flag detection (basic heuristics + LLM optional)
- Inline comments added in .docx
- JSON summary export

## Quickstart

1. Create a virtual environment and install dependencies:

```cmd
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the app:

```cmd
streamlit run app/main.py
```

3. Open the local URL and upload your .docx files.

## Config
- Set environment variables for LLMs if using cloud models:
  - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- By default, the app runs with a local-only mode for classification and regex heuristics. RAG uses a local FAISS index built from `rag_sources/`.

## Repo structure
```
app/
  main.py              # Streamlit UI + orchestration
  agent/
    __init__.py
    checklist.py       # Required docs per process
    classify.py        # Identify process and doc types
    parse_docx.py      # Parsing helpers
    redflags.py        # Heuristic red flags
    annotate.py        # Insert comments into .docx
    rag.py             # Build/query local vector index
    report.py          # JSON summary builder
rag_sources/
  README.md            # Place ADGM references here
outputs/
  # JSON and reviewed docx exports
```

## Notes
- The provided .docx in `docs/` are placeholders.
- Replace `rag_sources/` with your ADGM PDFs/links text dumps.
- This is a minimal demonstration scaffold for the evaluation task.
