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

__version__ = "1.0.0"

__all__ = [
    "DocumentChunker",
    "HybridRetriever",
    "BGEReranker",
    "AgenticRAG",
    "ToolRegistry",
    "ConversationMemory",
    "MultimodalRAG",
]


def __getattr__(name):
    """Lazy-load agentic and multimodal_rag so scripts that only need chunking/retrieval don't require anthropic/langgraph."""
    if name in ("AgenticRAG", "ToolRegistry", "ConversationMemory"):
        from .agentic import AgenticRAG, ToolRegistry, ConversationMemory
        return AgenticRAG if name == "AgenticRAG" else (ToolRegistry if name == "ToolRegistry" else ConversationMemory)
    if name == "MultimodalRAG":
        from .multimodal_rag import MultimodalRAG
        return MultimodalRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")