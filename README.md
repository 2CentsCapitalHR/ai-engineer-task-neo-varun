# ADGM Corporate Agent - Document Intelligence System

A comprehensive Streamlit application designed for automated analysis and compliance checking of corporate documents within the Abu Dhabi Global Market (ADGM) regulatory framework. The system combines document parsing, intelligent classification, compliance verification, and automated annotation to streamline corporate document workflows.

## 🚀 Key Features

### Document Processing
- **Multi-format Support**: Upload and process .docx documents
- **Intelligent Text Extraction**: Advanced parsing of Word documents with formatting preservation
- **Batch Processing**: Handle multiple documents simultaneously

### AI-Powered Classification
- **Automatic Process Detection**: Identifies document categories (Company Incorporation, Employment HR, Licensing)
- **Document Type Recognition**: Classifies specific document types (Employment Contract, Articles of Association, etc.)
- **Hybrid AI Approach**: Combines LLM analysis with rule-based heuristics for robust classification

### Compliance & Verification
- **ADGM Checklist Verification**: Automated checking against required document sets
- **Red Flag Detection**: Identifies potential compliance issues and regulatory concerns
- **Missing Document Analysis**: Highlights gaps in required documentation

### Document Enhancement
- **Intelligent Annotation**: Automatically inserts contextual comments into documents
- **Compliance Suggestions**: Provides specific recommendations for ADGM alignment
- **Citation Integration**: Links suggestions to relevant ADGM reference materials

### Knowledge Base Integration
- **RAG (Retrieval-Augmented Generation)**: Local vector database of ADGM reference materials
- **Smart Search**: Contextual retrieval of relevant regulatory information
- **AI Insights**: Optional OpenAI integration for enhanced analysis

## 📋 Supported Document Types

### Company Incorporation
- Articles of Association (AoA)
- Memorandum of Association (MoA)
- Board Resolution Templates
- Shareholder Resolution Templates
- Incorporation Application Forms
- UBO (Ultimate Beneficial Owner) Declaration Forms
- Register of Members and Directors
- Change of Registered Address Notices

### Employment & HR
- Employment Contracts (2019 & 2024 standards)
- Offer Letters
- Employee NDAs

### Future Support
- Licensing documents
- Regulatory filings
- Additional corporate governance documents

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the Repository**
   ```cmd
   git clone <repository-url>
   cd ai-engineer-task-neo-varun
   ```

2. **Create Virtual Environment**
   ```cmd
   py -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Set Environment Variables (Optional)**
   ```cmd
   set OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the Application**
   ```cmd
   streamlit run main.py
   ```

6. **Access the Interface**
   - Open your browser to `http://localhost:8501`
   - Upload .docx files and start analyzing

## ⚙️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: Enables advanced AI analysis and intelligent insights
- `ANTHROPIC_API_KEY`: Alternative AI provider (future support)

### Local Mode
When no API keys are provided, the system operates in local-only mode using:
- Rule-based document classification
- Heuristic compliance checking
- TF-IDF based similarity matching
- Local vector database for RAG

## 📁 Project Structure

```
ai-engineer-task-neo-varun/
├── main.py                 # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── Task.pdf               # Project specification
│
├── agent/                 # Core processing modules
│   ├── __init__.py
│   ├── annotate.py        # Document annotation engine
│   ├── checklist.py       # ADGM compliance checklists
│   ├── classify.py        # Document classification logic
│   ├── document_parser.py # Document parsing utilities
│   ├── llm.py            # AI/LLM integration layer
│   ├── parse_docx.py     # Word document processing
│   ├── rag.py            # Retrieval-Augmented Generation
│   ├── redflags.py       # Compliance issue detection
│   └── report.py         # Report generation
│
├── rag_sources/          # ADGM reference knowledge base
│   ├── 5.1-company-formation/
│   │   ├── Articles_of_Association_AoA.txt
│   │   ├── Board_Resolution_Template.txt
│   │   ├── Change_of_Registered_Address_Notice.txt
│   │   ├── Incorporation_Application_Form.txt
│   │   ├── Memorandum_of_Association_MoA.txt
│   │   ├── Register_of_Members_and_Directors.txt
│   │   ├── Shareholder_Resolution_Template.txt
│   │   └── UBO_Declaration_Form.txt
│   └── 5.2-employment-hr-contracts/
│       ├── Employment_Contract_2019.txt
│       └── Employment_Contract_2024.txt
│
└── outputs/              # Generated results
    ├── summary.json      # Analysis summary
    └── REVIEWED_*.docx   # Annotated documents
```

## 🔄 Workflow Overview

1. **Document Upload**: Users upload one or more .docx files
2. **Text Extraction**: System extracts and preprocesses document content
3. **Classification**: AI/heuristic hybrid determines process type and document category
4. **Compliance Check**: Validates against ADGM requirements and identifies missing documents
5. **Red Flag Analysis**: Scans for potential compliance issues and regulatory concerns
6. **Annotation**: Inserts intelligent comments and suggestions into documents
7. **Report Generation**: Creates comprehensive JSON summary and downloadable files

## 📊 Output Formats

### Annotated Documents
- Original documents with inline comments
- Compliance suggestions and recommendations
- Citations to relevant ADGM materials
- Downloadable as `REVIEWED_[filename].docx`

### JSON Summary Report
```json
{
  "process": "Employment HR",
  "total_documents": 1,
  "required_documents": 3,
  "missing_documents": ["Offer Letter", "Employee NDA"],
  "compliance_score": 0.67,
  "issues": [
    {
      "document": "Employment Contract",
      "file": "contract.docx",
      "section": "Governing Law",
      "issue": "Missing ADGM jurisdiction clause",
      "severity": "High",
      "suggestion": "Add ADGM governing law clause...",
      "citation": "ADGM Employment Regulations 2024"
    }
  ]
}
```

## 🧠 AI Integration

### Local Processing (Default)
- TF-IDF vectorization for document similarity
- Rule-based pattern matching
- Heuristic compliance checking
- No external API dependencies

### Enhanced AI Mode (with OpenAI)
- GPT-powered document classification
- Intelligent red flag detection
- Contextual compliance suggestions
- Advanced natural language understanding

## 🛡️ Compliance Framework

### ADGM Employment Regulations
- 2019 and 2024 standards support
- Automatic version detection
- Compliance gap analysis
- Regulatory update tracking

### Company Formation Requirements
- Complete incorporation document sets
- UBO declaration compliance
- Corporate governance standards
- Regulatory filing requirements

## 🔧 Customization

### Adding New Document Types
1. Update `DOC_TYPE_PATTERNS` in `agent/classify.py`
2. Add corresponding checklist in `agent/checklist.py`
3. Include reference materials in `rag_sources/`
4. Update red flag patterns in `agent/redflags.py`

### Extending Compliance Rules
1. Modify `REQUIRED_SECTIONS` in `agent/redflags.py`
2. Add new regulatory patterns
3. Update RAG knowledge base
4. Test with sample documents

## 🚦 Performance Considerations

- **Processing Speed**: ~2-5 seconds per document (local mode)
- **Memory Usage**: ~100-200MB for typical document sets
- **Storage**: RAG index requires ~10-50MB depending on knowledge base size
- **Scalability**: Suitable for 1-100 documents per session

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed correctly
2. **Memory Issues**: Process documents in smaller batches
3. **API Errors**: Verify OpenAI API key format and validity
4. **Performance**: Check system resources and document complexity

### Debug Mode
Set environment variable for verbose logging:
```cmd
set STREAMLIT_LOGGER_LEVEL=debug
```

## 📈 Future Enhancements

- [ ] Support for PDF documents
- [ ] Multi-language document processing
- [ ] Advanced workflow automation
- [ ] Integration with document management systems
- [ ] Real-time collaboration features
- [ ] Audit trail and version control
- [ ] Mobile-responsive interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request
5. Follow code review process

## 📄 License

This project is developed for educational and evaluation purposes within the ADGM regulatory framework.

## 📞 Support

For technical support or questions:
- Review the troubleshooting section
- Check the project documentation
- Submit issues through the repository

---

**Note**: This system is designed for document analysis and compliance checking assistance. It should not replace professional legal advice or official ADGM regulatory guidance.
