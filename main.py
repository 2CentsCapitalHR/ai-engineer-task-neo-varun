"""
ADGM Corporate Agent - Document Intelligence System

A Streamlit application for analyzing corporate documents for ADGM compliance.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st

from agent.checklist import REQUIRED_DOCS_BY_PROCESS
from agent.classify import detect_process, detect_doc_type, detect_process_and_type_llm
from agent.parse_docx import read_docx_text
from agent.redflags import analyze_red_flags
from agent.annotate import annotate_docx
from agent.rag import RAGClient
from agent.report import build_report

# Configuration
OUTPUT_DIR = Path("outputs")
RAG_DIR = Path("rag_sources")

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Streamlit configuration
st.set_page_config(page_title="ADGM Corporate Agent", layout="wide")
st.title("ADGM Corporate Agent (Document Intelligence)")

with st.sidebar:
    st.header("Settings")
    st.caption("OpenAI LLM integration auto-enables when OPENAI_API_KEY is set in the environment.")

st.write("Upload one or more .docx files for review.")
uploads = st.file_uploader("Upload .docx documents", type=["docx"], accept_multiple_files=True)

if uploads:
    # Initialize RAG client once, optionally enabling OpenAI analysis
    openai_key = os.getenv("OPENAI_API_KEY")
    rag = RAGClient(RAG_DIR, openai_api_key=openai_key)

    docs_texts = {}
    for up in uploads:
        text = read_docx_text(up)
        docs_texts[up.name] = text

    # Process + doc-type detection (LLM first if available)
    llm_res = detect_process_and_type_llm(list(docs_texts.values()))
    if llm_res.get("process") and llm_res.get("doc_type"):
        process = llm_res["process"]
        # assign same detected type to all unless heuristic differs per file
        file_to_type = {fname: llm_res["doc_type"] for fname in docs_texts.keys()}
    else:
        process = detect_process(list(docs_texts.values()))
        file_to_type = {fname: detect_doc_type(text) for fname, text in docs_texts.items()}

    # Debug output
    st.write(f"DEBUG - Detected process: {process}")
    st.write(f"DEBUG - LLM result: {llm_res}")
    st.write(f"DEBUG - File to type mapping: {file_to_type}")

    # Checklist
    required = REQUIRED_DOCS_BY_PROCESS.get(process, [])
    st.write(f"DEBUG - Required docs from checklist: {required}")
    present_types = set(file_to_type.values())
    st.write(f"DEBUG - Present types: {present_types}")
    missing = [doc for doc in required if doc not in present_types]
    st.write(f"DEBUG - Missing docs: {missing}")

    # Red flags + annotation
    reviewed_files: List[Tuple[str, bytes]] = []
    issues = []
    for fname, text in docs_texts.items():
        doc_type = file_to_type[fname]
        redflag_items = analyze_red_flags(text, doc_type, rag)
        issues.extend([
            {
                "document": doc_type,
                "file": fname,
                **item,
            }
            for item in redflag_items
        ])
        # Annotate docx
        annotated_bytes = annotate_docx(filename=fname, uploaded_file_obj=next(u for u in uploads if u.name==fname), comments=redflag_items, rag=rag)
        reviewed_files.append((f"REVIEWED_{fname}", annotated_bytes))

    # Build report
    report = build_report(process=process, file_to_type=file_to_type, required=required, issues=issues)

    st.subheader("Detected Process")
    st.write(process)

    st.subheader("Checklist")
    st.write({"required": required, "present": list(present_types), "missing": missing})
    st.info(f"You uploaded {len(file_to_type)} document(s). Required for {process}: {len(required)}. Missing: {len(missing)}.")
    if missing:
        st.warning("Missing required documents: " + ", ".join(missing))

    st.subheader("Issues Found")
    st.json(issues)

    # Optional: Intelligent insights using RAG + OpenAI, if API key is set
    if openai_key:
        st.subheader("Intelligent Insights (AI)")
        # Use doc types as lightweight queries for insights
        unique_queries = sorted(set(file_to_type.values()))
        for q in unique_queries:
            insights = rag.intelligent_search(q, context=f"Process: {process}")
            with st.expander(f"Insights for: {q}"):
                for fname, snippet, analysis in insights:
                    st.markdown(f"- Source: {fname}")
                    st.caption(snippet[:300] + ("…" if len(snippet) > 300 else ""))
                    st.write(analysis)

    st.subheader("Download Reviewed Files")
    for out_name, out_bytes in reviewed_files:
        st.download_button(
            label=f"Download {out_name}",
            data=out_bytes,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    st.subheader("JSON Summary")
    st.json(report)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    for out_name, out_bytes in reviewed_files:
        (OUTPUT_DIR / out_name).write_bytes(out_bytes)
else:
    st.info("No files uploaded yet.")
