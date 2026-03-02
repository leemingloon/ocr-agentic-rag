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
"""

import os
from typing import List
from sentence_transformers import CrossEncoder


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
        # Skip loading in debug to avoid OOM when embedding model already loaded (e.g. 16GB RAM).
        # Set RAG_USE_RERANKER=1 to force loading even when RAG_DEBUG=1 (recommended for numeric tables).
        force_reranker = os.environ.get("RAG_USE_RERANKER", "").strip().lower() in ("1", "true", "yes")
        skip = (os.environ.get("RAG_DEBUG") == "1" and not force_reranker) or os.environ.get("RAG_SKIP_RERANKER", "").lower() in ("1", "true", "yes")
        if skip:
            print("Warning: Skipping reranker load (RAG_DEBUG or RAG_SKIP_RERANKER set; using retrieval order as-is)")
            self.model = None
            return
        try:
            print(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name, max_length=512)
            print("Reranker loaded successfully")
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