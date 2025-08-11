"""
Document parsing utilities for ADGM Corporate Agent.
This module provides enhanced document parsing with structured output.
"""

from typing import Dict, List, TypedDict
from agent.llm import get_openai_client


class DocParseResult(TypedDict):
    """Structured result from document parsing."""
    doc_type: str
    process_type: str
    sections: List[Dict[str, str]]
    red_flags: List[Dict[str, str]]


def parse_document(text: str, allowed_doc_types: List[str], allowed_processes: List[str]) -> DocParseResult:
    """
    Parse document using OpenAI, identifying type, process, sections, and red flags.
    
    Args:
        text: Document text to parse
        allowed_doc_types: List of allowed document types
        allowed_processes: List of allowed process types
        
    Returns:
        DocParseResult with parsed information
    """
    client = get_openai_client()
    if not client:
        return DocParseResult(
            doc_type="unknown",
            process_type="unknown",
            sections=[],
            red_flags=[]
        )

    system_prompt = """You are an ADGM legal document analyzer. Parse the given document text and:
1. Identify its type from the allowed list
2. Identify the process it belongs to from the allowed list  
3. Extract key sections and their content
4. Flag potential issues or red flags

Return a strict JSON object with: doc_type, process_type, sections (array of {name, content}), red_flags (array of {issue, severity, suggestion})."""

    user_prompt = f"""Document text excerpt:
{text[:8000]}

Allowed document types: {', '.join(allowed_doc_types)}
Allowed processes: {', '.join(allowed_processes)}

Parse and return JSON with the requested fields."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        return DocParseResult(
            doc_type=result.get("doc_type", "unknown"),
            process_type=result.get("process_type", "unknown"),
            sections=result.get("sections", []),
            red_flags=result.get("red_flags", [])
        )
        
    except Exception as e:
        print(f"Document parsing failed: {e}")
        return DocParseResult(
            doc_type="unknown",
            process_type="unknown",
            sections=[],
            red_flags=[]
        )
