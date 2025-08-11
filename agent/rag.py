import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """Available search methods for document retrieval."""
    TFIDF = "tfidf"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class DocumentMetadata:
    """Metadata for a document in the RAG system."""
    filename: str
    filepath: Path
    category: str
    size: int
    last_modified: float
    encoding: str = "utf-8"


@dataclass
class SearchResult:
    """Result from document search operations."""
    filename: str
    content: str
    score: float
    metadata: DocumentMetadata
    snippet_start: int = 0
    snippet_end: int = 0


class DocumentProcessor:
    """Handles document loading, preprocessing, and chunking."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.text', '.log'}
    MAX_CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    
    @staticmethod
    def load_document(filepath: Path) -> Tuple[str, DocumentMetadata]:
        """Load a document and extract its metadata."""
        try:
            # Try UTF-8 first, fallback to other encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    content = filepath.read_text(encoding=encoding, errors='ignore')
                    break
                except UnicodeDecodeError:
                    continue
            else:
                content = filepath.read_text(errors='ignore')
                encoding = 'unknown'
            
            # Extract category from directory structure
            category = filepath.parent.name if filepath.parent.name != filepath.root else 'general'
            
            metadata = DocumentMetadata(
                filename=filepath.name,
                filepath=filepath,
                category=category,
                size=len(content),
                last_modified=filepath.stat().st_mtime,
                encoding=encoding
            )
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Error loading document {filepath}: {e}")
            raise
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and preprocess text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        # Normalize case for better matching
        return text.strip()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks for better retrieval."""
        chunk_size = chunk_size or DocumentProcessor.MAX_CHUNK_SIZE
        overlap = overlap or DocumentProcessor.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size * 0.7:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + chunk_size - overlap, end)
            
            if start >= len(text):
                break
        
        return chunks


class EnhancedRAGSystem:
    
    def __init__(
        self,
        source_dir: Union[str, Path],
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_tokens: int = 2000,
        temperature: float = 0.3
    ):

        self.source_dir = Path(source_dir)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Document storage
        self.documents: List[Tuple[str, DocumentMetadata]] = []
        self.document_chunks: List[str] = []
        self.chunk_to_doc_map: List[int] = []
        
        # Search components
        self.vectorizer = None
        self.tfidf_matrix = None
        self.openai_client = None
        
        # Initialize components
        self._initialize_search_components()
        self._initialize_openai(openai_api_key)
        self._load_documents()
    
    def _initialize_search_components(self):
        """Initialize search-related components."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. TF-IDF search will be disabled.")
            return
        
        # Enhanced TF-IDF with better parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
            max_features=10000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm='l2'
        )
    
    def _initialize_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client if available."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available. AI-powered features will be disabled.")
            return
        
        if api_key or os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.info("No OpenAI API key provided. AI features will be disabled.")
    
    def _load_documents(self):
        """Load all documents from the source directory."""
        if not self.source_dir.exists():
            logger.error(f"Source directory {self.source_dir} does not exist")
            return
        
        logger.info(f"Loading documents from {self.source_dir}")
        
        for filepath in self.source_dir.rglob("*"):
            if (filepath.is_file() and 
                filepath.suffix.lower() in DocumentProcessor.SUPPORTED_EXTENSIONS):
                
                try:
                    content, metadata = DocumentProcessor.load_document(filepath)
                    processed_content = DocumentProcessor.preprocess_text(content)
                    
                    # Chunk the document
                    chunks = DocumentProcessor.chunk_text(processed_content)
                    
                    # Store document and chunks
                    doc_idx = len(self.documents)
                    self.documents.append((processed_content, metadata))
                    
                    for chunk in chunks:
                        self.document_chunks.append(chunk)
                        self.chunk_to_doc_map.append(doc_idx)
                    
                    logger.info(f"Loaded {filepath.name} ({len(chunks)} chunks)")
                    
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
        
        # Build TF-IDF index
        if self.document_chunks and SKLEARN_AVAILABLE:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.document_chunks)
            logger.info(f"Built TF-IDF index for {len(self.document_chunks)} chunks")
    
    def search(
        self,
        query: str,
        method: SearchMethod = SearchMethod.HYBRID,
        k: int = 5,
        min_score: float = 0.1
    ) -> List[SearchResult]:

        if not self.document_chunks:
            logger.warning("No documents loaded")
            return []
        
        if method == SearchMethod.TFIDF or method == SearchMethod.HYBRID:
            return self._tfidf_search(query, k, min_score)
        elif method == SearchMethod.SEMANTIC:
            return self._semantic_search(query, k, min_score)
        else:
            raise ValueError(f"Unknown search method: {method}")
    
    def _tfidf_search(self, query: str, k: int, min_score: float) -> List[SearchResult]:
        """Perform TF-IDF based search."""
        if not SKLEARN_AVAILABLE or self.tfidf_matrix is None:
            logger.warning("TF-IDF search not available")
            return []
        
        # Preprocess query
        processed_query = DocumentProcessor.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score < min_score:
                break
            
            doc_idx = self.chunk_to_doc_map[idx]
            chunk_content = self.document_chunks[idx]
            _, metadata = self.documents[doc_idx]
            
            result = SearchResult(
                filename=metadata.filename,
                content=chunk_content,
                score=score,
                metadata=metadata
            )
            results.append(result)
        
        return results
    
    def _semantic_search(self, query: str, k: int, min_score: float) -> List[SearchResult]:
        """Perform semantic search using OpenAI embeddings."""
        if not self.openai_client:
            logger.warning("Semantic search requires OpenAI client")
            return self._tfidf_search(query, k, min_score)
        
        # This would require implementing OpenAI embeddings
        # For now, fallback to TF-IDF
        logger.info("Semantic search not fully implemented, using TF-IDF")
        return self._tfidf_search(query, k, min_score)
    
    def cite(self, topic: str) -> str:
        """
        Generate a citation for a given topic (legacy compatibility method).
        
        Args:
            topic: Topic to search for citation
            
        Returns:
            Citation string
        """
        results = self.search(topic, k=1)
        if not results:
            return "[Ref: ADGM reference not found in local RAG]"
        
        result = results[0]
        snippet = result.content[:100] + "..." if len(result.content) > 100 else result.content
        return f"[Ref: {result.filename} — {snippet}]"
    
    def generate_citation(self, query: str, max_citations: int = 3) -> str:
        """Generate citations for a given query."""
        results = self.search(query, k=max_citations)
        
        if not results:
            return "[No relevant references found]"
        
        citations = []
        for i, result in enumerate(results[:max_citations], 1):
            snippet = result.content[:150] + "..." if len(result.content) > 150 else result.content
            citation = f"[{i}] {result.filename}: {snippet}"
            citations.append(citation)
        
        return " | ".join(citations)
    
    def analyze_document_with_ai(
        self,
        text: str,
        task: str,
        context: str = ""
    ) -> str:

        if not self.openai_client:
            return "AI analysis not available (OpenAI client not initialized)"
        
        try:
            system_prompt = (
                "You are an expert document analyzer specializing in legal, business, "
                "and regulatory documents. Provide clear, accurate, and well-structured analysis."
            )
            
            user_prompt = f"""
Task: {task}

{f"Context: {context}" if context else ""}

Document Content:
{text[:4000]}  # Limit to avoid token limits

Please provide a comprehensive analysis addressing the specified task.
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return f"AI analysis failed: {str(e)}"
    
    def intelligent_query(
        self,
        query: str,
        context: str = "",
        include_analysis: bool = True,
        k: int = 3
    ) -> Dict[str, Any]:

        # Get search results
        results = self.search(query, k=k)
        
        response = {
            "query": query,
            "results_count": len(results),
            "results": [],
            "summary": "",
            "citations": self.generate_citation(query, k)
        }
        
        for result in results:
            result_data = {
                "filename": result.filename,
                "content": result.content,
                "score": result.score,
                "category": result.metadata.category,
                "analysis": ""
            }
            
            # Add AI analysis if requested and available
            if include_analysis and self.openai_client:
                analysis_task = f"Analyze how this content relates to: {query}"
                if context:
                    analysis_task += f"\nAdditional context: {context}"
                
                result_data["analysis"] = self.analyze_document_with_ai(
                    result.content,
                    analysis_task
                )
            
            response["results"].append(result_data)
        
        # Generate overall summary
        if self.openai_client and results:
            combined_content = "\n\n".join([r.content for r in results[:3]])
            response["summary"] = self.analyze_document_with_ai(
                combined_content,
                f"Provide a comprehensive summary addressing the query: {query}",
                context
            )
        
        return response
    
    def intelligent_search(self, query: str, context: str = "") -> List[Tuple[str, str, str]]:
        """
        Legacy compatibility method: Combine TF-IDF search with OpenAI analysis for more intelligent results.
        
        Args:
            query: Search query
            context: Additional context for the analysis
            
        Returns:
            List of tuples: (filename, snippet, AI analysis)
        """
        # Get search results using the new method
        results = self.search(query, k=3)
        
        if not self.openai_client:
            # Return without AI analysis if OpenAI client not available
            return [(result.filename, result.content, "OpenAI analysis not available") for result in results]
        
        enhanced_results = []
        for result in results:
            analysis_task = f"Analyze how this content relates to the query: {query}"
            if context:
                analysis_task += f"\nAdditional context: {context}"
                
            analysis = self.analyze_document_with_ai(
                result.content,
                analysis_task
            )
            enhanced_results.append((result.filename, result.content, analysis))
        
        return enhanced_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics and information."""
        categories = {}
        total_size = 0
        
        for content, metadata in self.documents:
            categories[metadata.category] = categories.get(metadata.category, 0) + 1
            total_size += metadata.size
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.document_chunks),
            "total_size_bytes": total_size,
            "categories": categories,
            "search_available": SKLEARN_AVAILABLE,
            "ai_available": self.openai_client is not None,
            "source_directory": str(self.source_dir)
        }


# Legacy compatibility alias
RAGClient = EnhancedRAGSystem
