"""
Red flag analysis for ADGM compliance checking.
"""

import re
from typing import List, Dict

try:
    from agent.llm import get_openai_client, llm_redflags
except Exception:
    get_openai_client = None  # type: ignore
    llm_redflags = None  # type: ignore

# ADGM compliance patterns
ADGM_JURIS_TERMS = ["adgm", "abu dhabi global market"]
BAD_JURIS_TERMS = ["uae federal court", "dubai courts"]
AMBIGUOUS_PHRASES = ["best efforts", "may consider", "if convenient"]

REQUIRED_SECTIONS = {
    "Articles of Association": ["jurisdiction", "governing law"],
    "Memorandum of Association": ["registered office", "objectives"],
    "Employment Contract": ["signatory", "term", "governing law"],
}


def analyze_red_flags(text: str, doc_type: str, rag) -> List[Dict]:
    """
    Analyze document for compliance red flags.
    
    Args:
        text: Document text to analyze
        doc_type: Type of document
        rag: RAG system for citations
        
    Returns:
        List of red flag issues found
    """
    # Try LLM analysis first if available
    if get_openai_client and llm_redflags:
        client = get_openai_client()
        if client:
            try:
                rag_results = rag.search(f"{doc_type} compliance ADGM", k=3)
                # Handle both old and new RAG return formats
                if rag_results and hasattr(rag_results[0], 'filename'):
                    rag_snippets = [f"{result.filename}: {result.content}" for result in rag_results]
                else:
                    rag_snippets = [f"{name}: {snippet}" for name, snippet in rag_results]
                
                llm_issues = llm_redflags(client, text, doc_type, rag_snippets)
                if llm_issues:
                    return llm_issues
            except Exception:
                pass

    # Fallback to heuristic analysis
    issues: List[Dict] = []
    text_lower = text.lower()

    # Check jurisdiction compliance
    if not any(term in text_lower for term in ADGM_JURIS_TERMS):
        issues.append({
            "section": "Jurisdiction",
            "issue": "Jurisdiction clause does not specify ADGM",
            "severity": "High",
            "suggestion": "Specify ADGM Courts as governing jurisdiction.",
            "citation": rag.cite("companies regulation jurisdiction"),
        })
    
    if any(term in text_lower for term in BAD_JURIS_TERMS):
        issues.append({
            "section": "Jurisdiction", 
            "issue": "References non-ADGM courts",
            "severity": "High",
            "suggestion": "Replace with ADGM Courts.",
            "citation": rag.cite("companies regulation jurisdiction"),
        })

    # Check for ambiguous language
    for phrase in AMBIGUOUS_PHRASES:
        if phrase in text_lower:
            issues.append({
                "section": "Ambiguity",
                "issue": f"Ambiguous phrase detected: '{phrase}'",
                "severity": "Medium", 
                "suggestion": "Use clear, enforceable language.",
                "citation": rag.cite("contract clarity"),
            })

    # Check for missing required sections
    for required_section in REQUIRED_SECTIONS.get(doc_type, []):
        if required_section not in text_lower:
            issues.append({
                "section": required_section.title(),
                "issue": f"Missing expected section: {required_section}",
                "severity": "Medium",
                "suggestion": f"Add a {required_section} section aligned with ADGM templates.",
                "citation": rag.cite("templates"),
            })

    # Check for signature requirements
    if "signature" not in text_lower and "signed" not in text_lower:
        issues.append({
            "section": "Signatures",
            "issue": "Missing signatory section",
            "severity": "Medium",
            "suggestion": "Add signatory blocks with names, titles, and dates.",
            "citation": rag.cite("execution"),
        })

    return issues
