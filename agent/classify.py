from typing import List, Dict, Tuple
from difflib import SequenceMatcher

try:
    from agent.llm import get_openai_client, llm_classify_process_and_type
except Exception:
    get_openai_client = None  # type: ignore
    llm_classify_process_and_type = None  # type: ignore

PROC_KEYWORDS: Dict[str, List[str]] = {
    "Company Incorporation": [
        "incorporation", "articles of association", "memorandum of association", "ubo", "register of members",
    ],
    "Employment HR": [
        "employment", "employment contract", "employee",
    ],
    "Licensing": [
        "license", "licence", "regulatory filing",
    ],
}

DOC_TYPE_PATTERNS: Dict[str, List[str]] = {
    "Articles of Association": ["articles of association", "aoa"],
    "Memorandum of Association": ["memorandum of association", "moa", "mou"],
    "Board Resolution": ["board resolution"],
    "Shareholder Resolution": ["shareholder resolution"],
    "Incorporation Application Form": ["incorporation application"],
    "UBO Declaration Form": ["ubo declaration"],
    "Register of Members and Directors": ["register of members", "register of directors"],
    "Change of Registered Address Notice": ["change of registered address"],
    "Employment Contract": ["employment contract"],
}

def _sim(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def detect_process_and_type(text: str, use_llm: bool = True) -> Tuple[str, str]:
    """Detect process and document type, using LLM if available."""
    if use_llm and get_openai_client and llm_classify_process_and_type:
        try:
            client = get_openai_client()
            if client:
                # Map each process to its relevant document types
                candidates: Dict[str, List[str]] = {
                    "Company Incorporation": [
                        "Articles of Association", "Memorandum of Association", "Board Resolution",
                        "Shareholder Resolution", "Incorporation Application Form", "UBO Declaration Form",
                        "Register of Members and Directors", "Change of Registered Address Notice"
                    ],
                    "Employment HR": ["Employment Contract"],
                    "Licensing": list(DOC_TYPE_PATTERNS.keys())  # For licensing, allow all types for flexibility
                }
                result = llm_classify_process_and_type(client, text, candidates)
                if result.get("process") and result.get("doc_type"):
                    return result["process"], result["doc_type"]
        except Exception:
            pass

    # Fallback to heuristic matching
    low = text.lower()
    
    # Detect process
    best_proc_score = -1.0
    process = "Company Incorporation"  # default
    for proc, kws in PROC_KEYWORDS.items():
        score = max(_sim(low, kw) for kw in kws)
        if any(kw in low for kw in kws):
            score += 0.3  # bonus for exact match
        if score > best_proc_score:
            best_proc_score = score
            process = proc
    
    # Detect doc type
    best_doc_score = -1.0
    doc_type = "Articles of Association"  # default
    for label, patterns in DOC_TYPE_PATTERNS.items():
        score = max(_sim(low, p) for p in patterns)
        if any(p in low for p in patterns):
            score += 0.3  # bonus for exact match
        if score > best_doc_score:
            best_doc_score = score
            doc_type = label
    
    return process, doc_type


def detect_process(texts: List[str]) -> str:
    """Detect process from multiple text inputs."""
    joined = "\n".join(texts)
    process, _ = detect_process_and_type(joined)
    return process


def detect_doc_type(text: str) -> str:
    """Detect document type from text."""
    _, doc_type = detect_process_and_type(text)
    return doc_type


def detect_process_and_type_llm(texts: List[str]) -> Dict[str, str]:
    """LLM-based detection for multiple texts."""
    if not get_openai_client or not llm_classify_process_and_type:
        return {}
    
    client = get_openai_client()
    if not client:
        return {}
    
    joined = "\n".join(texts)
    # Use the same process-specific document mapping as in detect_process_and_type
    candidates: Dict[str, List[str]] = {
        "Company Incorporation": [
            "Articles of Association", "Memorandum of Association", "Board Resolution",
            "Shareholder Resolution", "Incorporation Application Form", "UBO Declaration Form",
            "Register of Members and Directors", "Change of Registered Address Notice"
        ],
        "Employment HR": ["Employment Contract"],
        "Licensing": list(DOC_TYPE_PATTERNS.keys())  # For licensing, allow all types for flexibility
    }
    
    try:
        return llm_classify_process_and_type(client, joined, candidates) or {}
    except Exception:
        return {}
