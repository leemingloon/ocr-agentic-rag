# rag_system/reranking.py

"""
BGE Reranker using sentence-transformers CrossEncoder

Compatible with transformers 5.x (replaced FlagEmbedding)

Usage:
    reranker = BGEReranker()
    reranked_docs = reranker.rerank(
        query="What is the revenue?",
        documents=["doc1", "doc2", "doc3"],
        top_k=5
    )

LOGGING RULE: Do not print or log "Loading ... model", "loaded successfully", or any
model-weight loading messages to stdout/stderr. They hide evaluation and debug logs.
Load silently; use RAG_DEBUG or explicit flags for diagnostics.
"""

import os
import sys
from typing import List
from sentence_transformers import CrossEncoder


def _load_cross_encoder_quiet(model_name: str, **kwargs) -> "CrossEncoder":
    """Load CrossEncoder with progress/output suppressed so eval debug logs stay visible."""
    prev_tqdm = os.environ.get("TQDM_DISABLE")
    prev_hf = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    _stdout, _stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        return CrossEncoder(model_name, **kwargs)
    finally:
        if sys.stdout is not _stdout and getattr(sys.stdout, "close", None):
            sys.stdout.close()
        if sys.stderr is not _stderr and getattr(sys.stderr, "close", None):
            sys.stderr.close()
        sys.stdout, sys.stderr = _stdout, _stderr
        if prev_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = prev_tqdm
        if prev_hf is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf


class BGEReranker:
    """
    Reranker using sentence-transformers CrossEncoder
    
    Replaced FlagEmbedding (incompatible with transformers 5.x)
    Uses BAAI/bge-reranker-v2-m3 model (same quality as FlagEmbedding)
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize BGE reranker
        
        Args:
            model_name: HuggingFace model name for reranking
        """
        # Only skip when explicitly requested (e.g. to save memory). Table-style finance QA benefits from reranker.
        # Set RAG_SKIP_RERANKER=1 to skip loading when needed (e.g. 16GB RAM with embedding model already loaded).
        skip = os.environ.get("RAG_SKIP_RERANKER", "").strip().lower() in ("1", "true", "yes")
        if skip:
            self.model = None
            return
        try:
            # No loading logs here; see module LOGGING RULE. Suppress tqdm/transformers output via _load_cross_encoder_quiet.
            self.model = _load_cross_encoder_quiet(model_name, max_length=512)
        except Exception as e:
            print(f"Warning: Could not load reranker: {e}")
            print("Warning: Using fallback (no reranking)")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[str]:
        """
        Rerank documents by relevance to query
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents (top K)
        """
        if self.model is None:
            # Fallback: no reranking
            return documents[:top_k]
        
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"Warning: Reranking failed: {e}, returning original order")
            return documents[:top_k]
        
        # Combine documents with scores
        scored_docs = list(zip(documents, scores))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K documents
        return [doc for doc, score in scored_docs[:top_k]]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Rerank documents by relevance; return (document, score) tuples for thresholding.
        """
        if self.model is None or not documents:
            return list(zip(documents[:top_k], [0.0] * min(len(documents), top_k)))
        pairs = [[query, doc] for doc in documents]
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"Warning: Reranking failed: {e}, returning original order")
            return list(zip(documents[:top_k], [0.0] * min(len(documents), top_k)))
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif hasattr(scores, "__iter__") and not isinstance(scores, (list, tuple)):
            scores = list(scores)
        if len(scores) != len(documents):
            scores = [0.0] * len(documents)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# Example usage
if __name__ == "__main__":
    reranker = BGEReranker()
    
    query = "What is the company's revenue?"
    
    documents = [
        "The company's revenue was $10M in Q3.",
        "The weather is sunny today.",
        "Revenue increased by 15% year-over-year.",
        "The CEO announced a new product.",
    ]
    
    reranked = reranker.rerank(query, documents, top_k=2)
    
    print("Query:", query)
    print("\nTop 2 documents:")
    for i, doc in enumerate(reranked, 1):
        print(f"{i}. {doc}")