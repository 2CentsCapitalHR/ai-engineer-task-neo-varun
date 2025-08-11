from typing import Dict, List

REQUIRED_DOCS_BY_PROCESS: Dict[str, List[str]] = {
    "Company Incorporation": [
        "Articles of Association",
        "Memorandum of Association",
        "Board Resolution",
        "Shareholder Resolution",
        "Incorporation Application Form",
        "UBO Declaration Form",
        "Register of Members and Directors",
        "Change of Registered Address Notice",
    ],
    "Employment HR": [
        "Employment Contract"
        # Note: Offer Letter and Employee NDA have been removed from requirements
    ],
}
