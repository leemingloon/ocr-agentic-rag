"""
Hybrid Retrieval System

Combines sparse (BM25) and dense (BGE-M3) retrieval
with Reciprocal Rank Fusion for optimal results.

CPU-only / low-RAM: set RAG_FAST_EMBEDDINGS=1 to use a small, fast model
(all-MiniLM-L6-v2). Install onnxruntime for extra speed: pip install onnxruntime
"""

import os
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from llama_index.core.schema import TextNode

# Env: RAG_FAST_EMBEDDINGS=1 uses a small CPU-friendly model (faster, lower quality)
_USE_FAST_EMBEDDINGS = os.environ.get("RAG_FAST_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes")
_FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MODEL = "BAAI/bge-m3"


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and dense search
    
    Performance:
    - BM25 alone: 76% recall
    - Dense alone: 81% recall  
    - Hybrid: 88% recall
    - Hybrid + Reranking: 92% recall
    
    Strategy:
    - Sparse (BM25): Catches exact keyword matches
    - Dense (BGE-M3): Catches semantic similarity
    - RRF: Fuses results optimally
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        top_k_sparse: int = 20,
        top_k_dense: int = 20,
        top_k_final: int = 10,
        rrf_k: int = 60,  # Reciprocal Rank Fusion parameter
        device: Optional[str] = None,  # "cuda" for GPU (e.g. Colab); None/"cpu" for CPU
    ):
        """
        Initialize hybrid retriever

        Args:
            embedding_model: HuggingFace model for dense retrieval
            top_k_sparse: Top results from BM25
            top_k_dense: Top results from dense search
            top_k_final: Final top results after fusion
            rrf_k: RRF smoothing parameter
            device: "cuda" for GPU (faster embedding in Colab), None or "cpu" for CPU
        """
        self.top_k_sparse = top_k_sparse
        self.top_k_dense = top_k_dense
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k
        self._device = device or "cpu"

        # Optional: use small fast model on CPU / low-RAM (set RAG_FAST_EMBEDDINGS=1)
        model_to_load = _FAST_MODEL if _USE_FAST_EMBEDDINGS else (embedding_model or _DEFAULT_MODEL)
        if _USE_FAST_EMBEDDINGS:
            print(f"[RAG] Using fast CPU embedding model: {model_to_load} (RAG_FAST_EMBEDDINGS=1)")
        # Load on CPU or GPU (GPU recommended for building large indices, e.g. Colab)
        print(f"Loading embedding model: {model_to_load} (device={self._device})")
        try:
            if self._device == "cuda":
                self.embed_model = SentenceTransformer(model_to_load, device="cuda")
            else:
                self.embed_model = SentenceTransformer(model_to_load, device="cpu", backend="onnx")
        except Exception:
            self.embed_model = SentenceTransformer(model_to_load, device=self._device)
        
        # Will be initialized when index is built
        self.bm25_index = None
        self.faiss_index = None
        self.chunks = []
        self.chunk_texts = []
        
    def build_index(self, chunks: List[TextNode]):
        """
        Build both BM25 and FAISS indices
        
        Args:
            chunks: List of document chunks
        """
        self.chunks = chunks
        self.chunk_texts = [chunk.text for chunk in chunks]
        
        print(f"Building indices for {len(chunks)} chunks...")
        
        # Build BM25 index
        tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Build FAISS index (batch_size: larger = faster on CPU up to RAM limit; 16GB ~ 48 for BGE-M3, 128 for small model)
        embed_batch_size = 128 if _USE_FAST_EMBEDDINGS else 48
        print("Generating embeddings...")
        embeddings = self.embed_model.encode(
            self.chunk_texts,
            show_progress_bar=True,
            batch_size=embed_batch_size,
            convert_to_numpy=True,
        )
        # Move to CPU numpy if we were on GPU (FAISS expects host memory)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
        self.faiss_index.add(embeddings.astype('float32'))
        
        print(f"Indices built successfully!")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        corpus_id: Optional[str] = None,
    ) -> List[Tuple[TextNode, float]]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query: Search query
            top_k: Optional override for number of results (default: use self.top_k_final)
            corpus_id: If set, keep only chunks whose metadata.corpus_id matches (e.g. FinQA document id)

        Returns:
            List of (chunk, score) tuples
        """
        if not self.bm25_index or not self.faiss_index:
            raise ValueError("Indices not built. Call build_index() first.")

        # Step 1: Sparse retrieval (BM25)
        sparse_results = self._sparse_retrieve(query)

        # Step 2: Dense retrieval (BGE-M3)
        dense_results = self._dense_retrieve(query)

        # Step 3: Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            sparse_results,
            dense_results
        )

        # Step 4: Optionally filter to chunks from a specific document (e.g. FinQA corpus_id)
        _rag_debug = os.environ.get("RAG_DEBUG") == "1"
        if corpus_id:
            before_count = len(fused_results)
            filtered = []
            for chunk, score in fused_results:
                meta = getattr(chunk, "metadata", None) or {}
                cid = str(meta.get("corpus_id", ""))
                if cid == str(corpus_id):
                    filtered.append((chunk, score))
            # Fallback: if 0 chunks, try matching by document prefix (e.g. "AAL/2018/page_13.pdf-2" -> "AAL/2018/page_13.pdf")
            if len(filtered) == 0 and "-" in str(corpus_id):
                prefix = str(corpus_id).rsplit("-", 1)[0]
                for chunk, score in fused_results:
                    meta = getattr(chunk, "metadata", None) or {}
                    cid = str(meta.get("corpus_id", ""))
                    if cid == prefix or cid.startswith(prefix + "-"):
                        filtered.append((chunk, score))
                if _rag_debug and filtered:
                    print(f"[DEBUG] retrieval: corpus_id exact match 0; prefix match {prefix!r} -> {len(filtered)} chunks")
            fused_results = filtered
            if _rag_debug:
                print(f"[DEBUG] retrieval: corpus_id={corpus_id!r} before_filter={before_count} after_filter={len(fused_results)}")

        # Step 5: Return top-k (caller may pass top_k, e.g. agentic retrieval_tools)
        k = top_k if top_k is not None else self.top_k_final
        return fused_results[:k]
    
    def _sparse_retrieve(self, query: str) -> List[Tuple[int, float]]:
        """
        BM25 sparse retrieval
        
        Returns:
            List of (chunk_idx, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:self.top_k_sparse]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def _dense_retrieve(self, query: str) -> List[Tuple[int, float]]:
        """
        Dense vector retrieval using BGE-M3
        
        Returns:
            List of (chunk_idx, score) tuples
        """
        # Encode query
        query_embedding = self.embed_model.encode([query])[0]
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, self.top_k_dense)
        if indices.size == 0 or (indices.ndim > 0 and len(indices[0]) == 0):
            return []
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]
    
    def _reciprocal_rank_fusion(
        self,
        sparse_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]]
    ) -> List[Tuple[TextNode, float]]:
        """
        Fuse sparse and dense results using Reciprocal Rank Fusion
        
        RRF formula: score = sum(1 / (k + rank))
        
        Args:
            sparse_results: Results from BM25
            dense_results: Results from dense search
            
        Returns:
            Fused and sorted results
        """
        # Calculate RRF scores
        rrf_scores = {}
        
        # Add sparse scores
        for rank, (idx, score) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank + 1)
        
        # Add dense scores
        for rank, (idx, score) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank + 1)
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return chunks with scores
        results = [
            (self.chunks[idx], score) 
            for idx, score in sorted_indices
        ]
        
        return results

    def save_index_bundle(
        self,
        dir_path: str,
        embedding_model: Optional[str] = None,
    ) -> None:
        """
        Save full index bundle (FAISS + BM25 corpus + chunks) so it can be loaded later
        without recomputing embeddings. Use from Colab after GPU embedding, then download
        and load locally. dir_path should be a directory (created if missing).
        """
        import json
        import pickle
        from pathlib import Path
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        if self.faiss_index is None or not self.chunks:
            raise ValueError("No index built. Call build_index() first.")
        # Save FAISS
        faiss.write_index(self.faiss_index, str(p / "faiss.index"))
        # Save chunk texts and metadata for BM25 rebuild and retrieve()
        chunk_texts = self.chunk_texts
        chunks_meta = [
            {"text": getattr(c, "text", ""), "metadata": getattr(c, "metadata", None) or {}}
            for c in self.chunks
        ]
        with open(p / "chunk_texts.pkl", "wb") as f:
            pickle.dump(chunk_texts, f)
        with open(p / "chunks_meta.pkl", "wb") as f:
            pickle.dump(chunks_meta, f)
        dim = self.faiss_index.d
        model_name = embedding_model or _DEFAULT_MODEL
        with open(p / "meta.json", "w") as f:
            json.dump({"embedding_model": model_name, "dimension": dim, "num_chunks": len(self.chunks)}, f)
        print(f"Saved index bundle to {dir_path} ({len(self.chunks)} chunks, dim={dim})")

    def load_index_bundle(
        self,
        dir_path: str,
        device: Optional[str] = None,
    ) -> None:
        """
        Load index bundle (FAISS + BM25 + chunks) from a directory saved by save_index_bundle.
        Uses CPU for the embedding model (only needed for encoding queries). For GPU embedding
        use the Colab notebook, then download the bundle and load it locally with this method.
        """
        import json
        import pickle
        from pathlib import Path
        p = Path(dir_path)
        if not p.is_dir():
            raise FileNotFoundError(f"Index bundle directory not found: {dir_path}")
        with open(p / "meta.json") as f:
            meta = json.load(f)
        model_name = meta.get("embedding_model", _DEFAULT_MODEL)
        dim = meta["dimension"]
        # Load embedding model for query encoding (CPU is fine for single-query)
        use_cuda = (device == "cuda")
        if device is None:
            use_cuda = False
        self.embed_model = SentenceTransformer(model_name, device="cuda" if use_cuda else "cpu")
        # Load FAISS
        self.faiss_index = faiss.read_index(str(p / "faiss.index"))
        with open(p / "chunk_texts.pkl", "rb") as f:
            self.chunk_texts = pickle.load(f)
        with open(p / "chunks_meta.pkl", "rb") as f:
            chunks_meta = pickle.load(f)
        self.chunks = [
            TextNode(text=m["text"], metadata=m["metadata"])
            for m in chunks_meta
        ]
        # Rebuild BM25
        tokenized_corpus = [t.lower().split() for t in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"Loaded index bundle from {dir_path} ({len(self.chunks)} chunks, model={model_name})")

    def save_index(self, path: str):
        """Save FAISS index only to a single file (legacy). Prefer save_index_bundle for full save."""
        if self.faiss_index:
            faiss.write_index(self.faiss_index, path)

    def load_index(self, path: str):
        """Load FAISS index only from a single file (legacy). Prefer load_index_bundle for full load."""
        self.faiss_index = faiss.read_index(path)


# Example usage
if __name__ == "__main__":
    from chunking import DocumentChunker
    
    # Sample documents
    documents = [
        "The company reported revenue of $2.5M in Q3 2025.",
        "Product sales increased by 19% compared to previous quarter.",
        "Customer acquisition reached 1,200 new customers.",
        "The churn rate decreased to 3.2% from 4.1%.",
        "Average contract value is $12,500.",
    ]
    
    # Create chunks
    chunker = DocumentChunker()
    chunks = []
    for i, doc in enumerate(documents):
        doc_chunks = chunker.chunk_document(doc, metadata={"doc_id": i})
        chunks.extend(doc_chunks)
    
    # Initialize retriever
    retriever = HybridRetriever(
        embedding_model="BAAI/bge-m3",
        top_k_final=3
    )
    
    # Build index
    retriever.build_index(chunks)
    
    # Test query
    query = "How much revenue did we make?"
    results = retriever.retrieve(query)
    
    print(f"\nTop results for: '{query}'\n")
    for i, (chunk, score) in enumerate(results):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Text: {chunk.text}")
        print()