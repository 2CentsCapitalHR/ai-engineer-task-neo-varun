from typing import Dict, List


def build_report(process: str, file_to_type: Dict[str, str], required: List[str], issues: List[dict]) -> dict:
    present_types = list(set(file_to_type.values()))
    missing = [doc for doc in required if doc not in present_types]
    return {
        "process": process,
        "documents_uploaded": len(file_to_type),
        "required_documents": len(required),
        "missing_documents": missing,
        "files": file_to_type,
        "issues_found": issues,
    }
