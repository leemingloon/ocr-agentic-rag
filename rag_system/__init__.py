"""
RAG System Module

Hybrid retrieval + agentic orchestration + multimodal support

Components:
- DocumentChunker: Structure-preserving chunking
- HybridRetriever: BM25 + BGE-M3 dense retrieval
- BGEReranker: Cross-encoder reranking
- AgenticRAG: LangGraph orchestration
- MultimodalRAG: Vision-language enhanced RAG (NEW)
"""

from .chunking import DocumentChunker
from .retrieval import HybridRetriever
from .reranking import BGEReranker
from .agentic import AgenticRAG, ToolRegistry, ConversationMemory
from .multimodal_rag import MultimodalRAG  # NEW

__version__ = "1.0.0"

__all__ = [
    "DocumentChunker",
    "HybridRetriever",
    "BGEReranker",
    "AgenticRAG",
    "ToolRegistry",
    "ConversationMemory",
    "MultimodalRAG",  # NEW
]