"""
OpenAI LLM integration for ADGM Corporate Agent.
"""

import os
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client if API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def llm_classify_process_and_type(client: OpenAI, text: str, candidates: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Classify document process and type using OpenAI.
    
    Args:
        client: OpenAI client instance
        text: Document text to classify
        candidates: Available process and document type candidates
        
    Returns:
        Dictionary with 'process' and 'doc_type' keys
    """
    system_prompt = (
        "You are a legal assistant for ADGM corporate compliance. "
        "Classify the document's process type and document type strictly from the provided candidates. "
        "Return strict JSON with keys: process, doc_type. No additional text."
    )
    
    user_prompt = (
        f"Document text:\n{text[:8000]}\n\n"
        f"Available processes: {', '.join(candidates.keys())}\n"
        f"Available document types: {', '.join(sum(candidates.values(), []))}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {}


def llm_redflags(client: OpenAI, text: str, doc_type: str, rag_snippets: List[str]) -> List[Dict[str, str]]:
    """
    Analyze document for red flags using OpenAI.
    
    Args:
        client: OpenAI client instance
        text: Document text to analyze
        doc_type: Type of document being analyzed
        rag_snippets: Relevant RAG context snippets
        
    Returns:
        List of red flag issues found
    """
    system_prompt = (
        "You are an ADGM-focused legal reviewer. Analyze the document for compliance red flags. "
        "Use the provided RAG snippets for context and citations. "
        "Return a JSON array of objects with keys: section, issue, severity (High/Medium/Low), suggestion, citation."
    )
    
    rag_context = "\n\n".join(rag_snippets)[:6000]
    user_prompt = (
        f"Document type: {doc_type}\n\n"
        f"Reference context:\n{rag_context}\n\n"
        f"Document excerpt:\n{text[:9000]}\n\n"
        "Analyze for compliance issues and return as JSON array."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        data = json.loads(response.choices[0].message.content)
        
        # Handle different response formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "issues" in data:
            return data["issues"]
        elif isinstance(data, dict) and "red_flags" in data:
            return data["red_flags"]
        
    except Exception:
        pass
    
    return []
